#!/usr/bin/env python3
"""물타기 격자 하한 재확장(25·35bp) + **두 잣대 동시 평가** (2026-09-06, 사용자 요청).

경위:
  · 168셀(4bp) -> 세 창 양수 7셀인데 **전부 add_bp=100(격자 하한)** 에 몰림.
  · 315셀로 확장(add 50·75 추가) -> 세 창 양수 126/243, 일CI 하한>0 52/243,
    다중성 귀무 100백분위(무작위 격자 평균 41.9셀). **산술평균 기준으로는 통과.**
    그런데 통과 셀이 **또 새 하한(50bp)에 최다(53건)** 몰렸다.
  · ⭐기하/파산 검사에서 **다섯 대표 셀 전부 1년 파산확률 82~100%**. 산술평균이 틀린 잣대였다.

이 스크립트: 사용자 요청대로 하한을 **25·35bp까지** 확장하고, 셀마다 **두 잣대를 같이** 낸다.
  (A) 산술: 세 창 계좌 %, 일CI, 같은 측면 무작위 초과  -- 이전 격자와 비교 가능하게
  (B) ⭐기하: E[log(1+r)] (전손 제외분), 전손 건수, 순차 복리 파산 시점, **1년 파산확률**
      전손(투입 증거금 전액)은 트랜치×(max_adds+1)=100%일 때 계좌 −100%이고 log는 −∞다.
      **실제 판정 기준은 (B)** -- (A)가 아무리 좋아도 파산하면 자본이 사라진다.

⚠️`add_bp=25`는 가격 0.25%로 ETH 5분봉 ATR 한 개 수준이다 -- 물타기가 매우 자주 걸린다.
평균 트랜치 수를 함께 보고하며, 이 값이 크면 수수료가 그만큼 곱해진다.
⚠️다중성 귀무는 이번 회차에서 생략한다(315셀 기준 무작위 평균 41.9/243 = 17%가 산술 기준으로
이미 확인됨). 파산 관문을 통과하는 셀이 나오면 그 셀에 대해서만 별도로 돌린다.

규약 상속: 진입 open[i+1], 익절=평균 진입가 ±tp, 물타기=역행 add_bp에서 1트랜치, 손절=역행 sl_bp
전량 청산(물타기 판정 뒤), 청산=역행 100%/leverage, 수수료는 트랜치마다 왕복 4bp(명목, 사용자 실측
메이커 = 증거금 1.20%), 봉내 순서 비관. HOLDOUT(≥2026-04-01) 미접촉.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TRAIN_END, VAL_END, OOS_END = (pd.Timestamp(x) for x in ("2025-09-01", "2026-01-01", "2026-04-01"))
COST_BP, LEVERAGE, HOLD = 4.0, 30.0, 72
LIQ_BP = 1e4 / LEVERAGE
B_BOOT, N_RUIN_SIM, SEED = 300, 150, 20260906
OUT = ROOT / "data/research/eth_chip_avgdown_intrabar_fix_20260906"

TP_GRID = [16.7, 33.3, 50.0]
ADD_GRID = [25.0, 35.0, 50.0, 75.0, 100.0, 150.0, 200.0]   # ⭐25·35 신규
SL_GRID = [100.0, 150.0, 200.0, 250.0, None]
ARMS = [(0.20, 0), (0.20, 2), (0.20, 4), (0.10, 4), (0.10, 9)]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def log(m: str) -> None:
    print(f"[fix] {m}", flush=True)


_META = _load("meta_live7", "scripts/live_evidence_signal_metalabel_20260829.py")
CHIP = {n: int(c["horizon_bars"]) for n, c in _META.METALABEL_SIGNALS.items()}


def first_fire_positions(fired: np.ndarray, horizon: int) -> np.ndarray:
    keep, last = [], -10**9
    for i in np.flatnonzero(fired):
        if i - last > horizon:
            keep.append(i)
        last = i
    return np.asarray(keep, dtype=int)


def simulate(o, h, l, c, pos, sgn, tp_bp, add_bp, max_adds, sl_bp, tranche, n):
    if pos + 1 >= n:
        return np.nan, "", 0
    entry = o[pos + 1]
    avg, k = entry, 1
    end = min(pos + 1 + HOLD, n - 1)
    scale = tranche * LEVERAGE * 100.0
    for j in range(pos + 1, end + 1):
        hi, lo = h[j], l[j]
        k_at_bar_start = k
        adv = (avg - lo) / avg if sgn > 0 else (hi - avg) / avg
        fav = (hi - avg) / avg if sgn > 0 else (avg - lo) / avg
        if adv * 1e4 >= LIQ_BP:
            return -tranche * k * 100.0, "liq", k
        while k <= max_adds and adv * 1e4 >= add_bp:
            add_px = avg * (1 - sgn * add_bp / 1e4)
            avg = (avg * k + add_px) / (k + 1)
            k += 1
            adv = (avg - lo) / avg if sgn > 0 else (hi - avg) / avg
            fav = (hi - avg) / avg if sgn > 0 else (avg - lo) / avg
            if adv * 1e4 >= LIQ_BP:
                return -tranche * k * 100.0, "liq", k
        if sl_bp is not None and adv * 1e4 >= sl_bp:
            return (-sl_bp * k - COST_BP * k) / 1e4 * scale, "sl", k
        # ⭐2026-09-06 봉내 순서 수정: 이 봉에서 물타기가 있었으면 **같은 봉의 고가로 익절하지 않는다**.
        # 이전 판은 저가로 평균가를 낮춘 뒤 같은 봉 고가로 익절을 판정해 "모든 봉이 먼저 내려가
        # 물타기를 채워주고 그다음 올라가 익절시켜준다"고 가정했다 -- 봉내 순서를 최대 낙관으로
        # 고른 것이고, add_bp가 좁을수록(25bp) 거의 매 봉이 공짜 이익이 됐다(전손 0%, 자본 x1e65).
        if k == k_at_bar_start and fav * 1e4 >= tp_bp:
            return (tp_bp * k - COST_BP * k) / 1e4 * scale, "tp", k
    move = sgn * (c[end] / avg - 1) * 1e4
    return (move * k - COST_BP * k) / 1e4 * scale, "timeout", k


def day_ci(vals, days, rng, b=B_BOOT):
    u = np.unique(days)
    if len(u) < 2:
        return np.nan, np.nan
    by = {d: vals[days == d] for d in u}
    m = np.empty(b)
    for i in range(b):
        m[i] = np.concatenate([by[d] for d in rng.choice(u, size=len(u), replace=True)]).mean()
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ruin_stats(rr, rng, horizon_trades):
    """순차 복리 + 1년 파산확률. rr = 계좌 수익률(비율)."""
    eq, ruin_at = 1.0, None
    for i, x in enumerate(rr):
        eq *= (1.0 + x)
        if eq <= 1e-9:
            ruin_at = i + 1
            break
    ruined, finals = 0, []
    cap = min(horizon_trades, 15000)
    for _ in range(N_RUIN_SIM):
        e2 = 1.0
        for x in rr[rng.choice(len(rr), size=cap, replace=True)]:
            e2 *= (1.0 + x)
            if e2 <= 1e-9:
                ruined += 1; e2 = 0.0; break
        finals.append(e2)
    return ruin_at, (None if ruin_at else float(eq)), ruined / N_RUIN_SIM, finals


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    log("신호 프레임 재구성...")
    _s1 = _load("s1_ax", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
    _s1.VAL_END = OOS_END
    sig, _f, _e = _s1.build_sig()
    ts = pd.to_datetime(sig["timestamp"]).dt.tz_localize(None)
    m0 = (ts < OOS_END).to_numpy()
    sig, ts = sig.loc[m0].reset_index(drop=True), ts.loc[m0].reset_index(drop=True)
    o, h, l, c = (sig[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    n = len(sig)
    day = ts.dt.floor("D").to_numpy()
    split = np.where(ts < TRAIN_END, "TRAIN", np.where(ts < VAL_END, "VAL", "OOS"))

    P, S = [], []
    for name, hz in CHIP.items():
        for side in ("bottom", "top"):
            col = f"{side}_{name}"
            if col not in sig.columns:
                continue
            for pos in first_fire_positions(sig[col].fillna(False).to_numpy(bool), hz):
                P.append(int(pos)); S.append(1.0 if side == "bottom" else -1.0)
    P, S = np.asarray(P), np.asarray(S)
    ok = P + 1 + HOLD < n
    P, S = P[ok], S[ok]
    order = np.argsort(P); P, S = P[order], S[order]
    spP, dayP = split[P], day[P]
    days_total = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400
    per_day = len(P) / days_total
    horizon = int(per_day * 365)
    log(f"  칩 발동 {len(P):,}건 · {per_day:.1f}건/일 · 1년 지평 {horizon:,}건 · 비용 {COST_BP:g}bp")

    pool = np.flatnonzero(np.arange(n) + 1 + HOLD < n)
    Pn, Sn = [], []
    for sp in ("TRAIN", "VAL", "OOS"):
        cnt = int((spP == sp).sum())
        cand = pool[split[pool] == sp]
        Pn.append(rng.choice(cand, size=cnt, replace=True))
        Sn.append(rng.choice(S[spP == sp], size=cnt, replace=False))
    Pn, Sn = np.concatenate(Pn), np.concatenate(Sn)
    spN = split[Pn]

    combos = [(tp, add, ma, sl, tr) for tp in TP_GRID for tr, ma in ARMS
              for add in (ADD_GRID if ma else [ADD_GRID[0]]) for sl in SL_GRID]
    log(f"  격자 {len(combos)}셀 (add {ADD_GRID})")

    cells = []
    for i, (tp, add, ma, sl, tr) in enumerate(combos, 1):
        r = [simulate(o, h, l, c, p, s, tp, add, ma, sl, tr, n) for p, s in zip(P, S)]
        v = np.array([x[0] for x in r], dtype=float)
        oc = np.array([x[1] for x in r], dtype=object)
        kk = np.array([x[2] for x in r], dtype=int)
        nv = np.array([simulate(o, h, l, c, p, s, tp, add, ma, sl, tr, n)[0]
                       for p, s in zip(Pn, Sn)], dtype=float)
        rr = v / 100.0
        n_ruin = int((rr <= -0.999).sum())
        fin = rr > -0.999
        geo = float(np.log1p(rr[fin]).mean()) if fin.any() else float("nan")
        ruin_at, final_eq, p_ruin, _f2 = ruin_stats(rr, rng, horizon)
        row = {"tp_bp": tp, "add_bp": add, "max_adds": ma, "sl_bp": sl, "tranche": tr,
               "degenerate": bool(ma and sl is not None and sl <= add),
               "mean_tranches": round(float(kk.mean()), 3),
               "geo_ex_ruin_pct": round(geo * 100, 4), "n_total_ruin": n_ruin,
               "total_ruin_rate": round(n_ruin / len(rr), 6),
               "liq_rate": round(float((oc == "liq").mean()), 5),
               "seq_ruin_at": ruin_at, "seq_final_equity": (None if ruin_at else round(final_eq, 6)),
               "p_ruin_1y": round(p_ruin, 4), "splits": {}}
        for sp in ("TRAIN", "VAL", "OOS"):
            m2 = spP == sp
            if m2.sum() < 30:
                continue
            lo, hi = day_ci(v[m2], dayP[m2], rng)
            nm = float(nv[spN == sp].mean())
            row["splits"][sp] = {"n": int(m2.sum()), "account_pct": round(float(v[m2].mean()), 4),
                                 "ci": [round(lo, 4), round(hi, 4)],
                                 "excess_vs_null": round(float(v[m2].mean() - nm), 4)}
        cells.append(row)
        if i % 40 == 0:
            log(f"    {i}/{len(combos)} ({time.time()-t0:.0f}s)")

    valid = [x for x in cells if not x["degenerate"] and len(x["splits"]) == 3]
    arith_pass = [x for x in valid if all(x["splits"][s]["account_pct"] > 0 for s in x["splits"])]
    ci_pass = [x for x in valid if all(x["splits"][s]["ci"][0] > 0 for s in x["splits"])]
    ruin_ok = [x for x in valid if x["p_ruin_1y"] < 0.05]
    both = [x for x in arith_pass if x["p_ruin_1y"] < 0.05 and x["geo_ex_ruin_pct"] > 0]
    log(f"\n=== 결과 ({len(cells)}셀, 유효 {len(valid)}) ===")
    log(f"  (A 산술) 세 창 계좌 양수      : {len(arith_pass)}/{len(valid)}")
    log(f"  (A 산술) 세 창 일CI 하한>0    : {len(ci_pass)}/{len(valid)}")
    log(f"  ⭐(B 기하) 1년 파산확률 < 5%   : {len(ruin_ok)}/{len(valid)}")
    log(f"  ⭐⭐둘 다 통과 (산술 ∧ 파산<5% ∧ 기하>0) : {len(both)}/{len(valid)}")

    import collections
    if arith_pass:
        log("\n  (A) 산술 통과 셀의 add_bp 분포 -- 경계 확인:")
        log(f"    {dict(sorted(collections.Counter(str(x['add_bp']) for x in arith_pass).items(), key=lambda t: float(t[0])))}")
    log("\n  파산확률 최소 8셀 (파산 위험 순):")
    log(f"    {'TP':>5s} {'물타기':>6s} {'추가':>4s} {'손절':>5s} {'트랜치':>5s} {'평균T':>5s} | "
        f"{'파산1y':>7s} {'기하%':>8s} {'전손':>7s} | {'T':>7s} {'V':>7s} {'O':>7s}")
    for x in sorted(valid, key=lambda z: (z["p_ruin_1y"], -z["geo_ex_ruin_pct"]))[:8]:
        sl_t = "없음" if x["sl_bp"] is None else format(x["sl_bp"], ".0f")
        t_, v_, o_ = (x["splits"][s]["account_pct"] for s in ("TRAIN", "VAL", "OOS"))
        log(f"    {x['tp_bp']:5.1f} {x['add_bp']:6.0f} {x['max_adds']:4d} {sl_t:>5s} "
            f"{x['tranche']:5.0%} {x['mean_tranches']:5.2f} | {x['p_ruin_1y']:7.1%} "
            f"{x['geo_ex_ruin_pct']:+8.4f} {x['total_ruin_rate']:7.4%} | "
            f"{t_:+7.3f} {v_:+7.3f} {o_:+7.3f}")
    if both:
        log("\n  ⭐⭐두 잣대 동시 통과 셀:")
        for x in sorted(both, key=lambda z: -z["geo_ex_ruin_pct"])[:10]:
            sl_t = "없음" if x["sl_bp"] is None else format(x["sl_bp"], ".0f")
            log(f"    TP{x['tp_bp']:5.1f} 물타기{x['add_bp']:4.0f}×{x['max_adds']}회 손절{sl_t:>5s} "
                f"트랜치{x['tranche']:.0%} | 기하 {x['geo_ex_ruin_pct']:+.4f}%/건 · "
                f"파산1y {x['p_ruin_1y']:.1%} · 최종자본 x{x['seq_final_equity']}")
    else:
        log("\n  ⛔두 잣대를 동시에 통과한 셀 없음.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(
        {"cost_bp": COST_BP, "leverage": LEVERAGE, "add_grid": ADD_GRID, "holdout_touched": False,
         "n_cells": len(cells), "n_valid": len(valid), "arith_pass": len(arith_pass),
         "ci_pass": len(ci_pass), "ruin_ok": len(ruin_ok), "both_pass": len(both),
         "cells": cells}, ensure_ascii=False, indent=2))
    log(f"\n산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
