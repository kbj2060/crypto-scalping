#!/usr/bin/env python3
"""§5.30 미해결 (a)격자 경계 + (b)다중성 귀무 (2026-09-05, 사용자 요청 "ab부터 진행해줘").

선행: `research_eth_chip_avgdown_stoploss_tuning_20260905.py`(168셀, 비용 4bp에서 세 창 양수 7셀).
그 7셀이 **전부** `add_bp=100`(격자 하한) ∧ `max_adds=4`(자본 상한)에 몰려 있었다 -- 이 저장소가
여러 번 데인 **격자 경계 아티팩트** 패턴(§5.19 §5-A, §5.29 §7-3).

## (a) 경계 확장 -- 두 경계를 각각 다른 방식으로 뚫는다
  · `add_bp` 하한: 100 -> **{50, 75}** 추가 (순수 격자 경계)
  · `max_adds` 상한 4: 이건 격자가 아니라 **자본 제약**이다(20% × 5트랜치 = 100%).
    트랜치를 줄이면 더 갈 수 있으므로 **tranche 10% × max_adds ≤ 9** 변형군을 따로 둔다.
    ⚠️트랜치를 줄이면 건당 명목도 줄어 계좌 %가 기계적으로 작아진다 -- 비교는 **무작위 초과**로 한다.

## (b) 다중성 귀무 -- "N셀 통과"가 우연으로 나오는 개수
  칩 발동 대신 **같은 측면 비율·같은 창별 건수의 무작위 봉**으로 같은 격자를 B회 반복해
  "세 창 전부 계좌 양수" 셀 수의 귀무분포를 낸다. 실측 통과 수가 이 분포의 어디인지가 판정이다.
  (§5.12 계열 규율: 격자 통과 수는 무작위 부분표집 귀무 없이는 무의미.)
  ⚠️귀무 반복에서는 일CI 부트스트랩을 생략한다(평균만 필요 -- 속도).

규약은 선행 스크립트 상속: 진입 open[i+1], 익절=평균 진입가 ±tp, 물타기=역행 add_bp에서 1트랜치,
손절=역행 sl_bp 전량 청산(물타기 판정 뒤), 청산=역행 100%/leverage, 수수료는 트랜치마다 왕복,
봉내 순서 비관. 비용은 **사용자 실측 메이커 왕복 4bp(명목) = 증거금 1.20%**.
HOLDOUT(≥2026-04-01) 미접촉.
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
COST_BP, LEVERAGE, HOLD = 4.0, 30.0, 72          # ⭐사용자 실측 메이커 왕복
LIQ_BP = 1e4 / LEVERAGE
B_BOOT, B_NULL, SEED = 400, 20, 20260905
OUT = ROOT / "data/research/eth_chip_avgdown_boundary_multiplicity_20260905"

# (a) 확장 격자
TP_GRID = [16.7, 33.3, 50.0]
ADD_GRID = [50.0, 75.0, 100.0, 150.0, 200.0]      # ⭐50·75 신규(하한 확장)
SL_GRID = [100.0, 150.0, 200.0, 250.0, None]
ARMS = ([(0.20, ma) for ma in (0, 2, 4)] +        # 트랜치 20% (자본 상한 4회)
        [(0.10, ma) for ma in (4, 9)])            # ⭐트랜치 10% (자본 상한 9회) -- max_adds 경계 돌파


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def log(m: str) -> None:
    print(f"[bnd-mult] {m}", flush=True)


_META = _load("meta_live5", "scripts/live_evidence_signal_metalabel_20260829.py")
CHIP = {n: int(c["horizon_bars"]) for n, c in _META.METALABEL_SIGNALS.items()}


def first_fire_positions(fired: np.ndarray, horizon: int) -> np.ndarray:
    keep, last = [], -10**9
    for i in np.flatnonzero(fired):
        if i - last > horizon:
            keep.append(i)
        last = i
    return np.asarray(keep, dtype=int)


def simulate(o, h, l, c, pos, sgn, tp_bp, add_bp, max_adds, sl_bp, tranche, n):
    """경로 의존 시뮬레이션. 반환 (account_pct, outcome, n_tranches)."""
    if pos + 1 >= n:
        return np.nan, "", 0
    entry = o[pos + 1]
    avg, k = entry, 1
    end = min(pos + 1 + HOLD, n - 1)
    scale = tranche * LEVERAGE * 100.0
    for j in range(pos + 1, end + 1):
        hi, lo = h[j], l[j]
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
        if fav * 1e4 >= tp_bp:
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


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    log("신호 프레임 재구성...")
    _s1 = _load("s1_bm", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
    _s1.VAL_END = OOS_END
    sig, _f, _e = _s1.build_sig()
    ts = pd.to_datetime(sig["timestamp"]).dt.tz_localize(None)
    msk = (ts < OOS_END).to_numpy()
    sig, ts = sig.loc[msk].reset_index(drop=True), ts.loc[msk].reset_index(drop=True)
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
    spP, dayP = split[P], day[P]
    log(f"  칩 발동 {len(P):,}건 · 비용 {COST_BP:g}bp(명목) = 증거금 {COST_BP*LEVERAGE/100:.2f}%")

    combos = []
    for tp in TP_GRID:
        for tr, ma in ARMS:
            for add in (ADD_GRID if ma else [ADD_GRID[0]]):
                for sl in SL_GRID:
                    combos.append((tp, add, ma, sl, tr))
    log(f"  (a) 확장 격자 {len(combos)}셀")

    # ── 같은 측면 무작위 귀무 풀(셀별 초과 계산용, 창별 건수 매칭) ──
    pool = np.flatnonzero(np.arange(n) + 1 + HOLD < n)
    Pn, Sn = [], []
    for sp in ("TRAIN", "VAL", "OOS"):
        cnt = int((spP == sp).sum())
        cand = pool[split[pool] == sp]
        Pn.append(rng.choice(cand, size=cnt, replace=True))
        Sn.append(rng.choice(S[spP == sp], size=cnt, replace=False) if cnt else np.array([]))
    Pn, Sn = np.concatenate(Pn), np.concatenate(Sn)
    spN = split[Pn]

    cells = []
    for i, (tp, add, ma, sl, tr) in enumerate(combos, 1):
        r = [simulate(o, h, l, c, p, s, tp, add, ma, sl, tr, n) for p, s in zip(P, S)]
        v = np.array([x[0] for x in r], dtype=float)
        oc = np.array([x[1] for x in r], dtype=object)
        nv = np.array([simulate(o, h, l, c, p, s, tp, add, ma, sl, tr, n)[0]
                       for p, s in zip(Pn, Sn)], dtype=float)
        row = {"tp_bp": tp, "add_bp": add, "max_adds": ma, "sl_bp": sl, "tranche": tr,
               "degenerate": bool(ma and sl is not None and sl <= add), "splits": {}}
        for sp in ("TRAIN", "VAL", "OOS"):
            m2 = spP == sp
            if m2.sum() < 30:
                continue
            lo, hi = day_ci(v[m2], dayP[m2], rng)
            nm = float(nv[spN == sp].mean())
            row["splits"][sp] = {"n": int(m2.sum()), "account_pct": round(float(v[m2].mean()), 4),
                                 "ci": [round(lo, 4), round(hi, 4)],
                                 "win_rate": round(float((oc[m2] == "tp").mean()), 4),
                                 "ruin_rate": round(float((oc[m2] == "liq").mean()), 4),
                                 "null_account_pct": round(nm, 4),
                                 "excess_vs_null": round(float(v[m2].mean() - nm), 4)}
        cells.append(row)
        if i % 30 == 0:
            log(f"    {i}/{len(combos)} ({time.time()-t0:.0f}s)")

    valid = [x for x in cells if not x["degenerate"] and len(x["splits"]) == 3]
    pos_all = [x for x in valid if all(x["splits"][s]["account_pct"] > 0 for s in x["splits"])]
    ci_pos = [x for x in valid if all(x["splits"][s]["ci"][0] > 0 for s in x["splits"])]
    log(f"\n=== (a) 확장 격자 결과 ({len(cells)}셀, 유효 {len(valid)}) ===")
    log(f"  세 창 전부 계좌 양수   : {len(pos_all)}/{len(valid)}")
    log(f"  세 창 전부 일CI 하한>0 : {len(ci_pos)}/{len(valid)}")
    import collections
    if pos_all:
        log("  통과 셀 파라미터 분포(경계 확인):")
        for f in ("tp_bp", "add_bp", "max_adds", "sl_bp", "tranche"):
            log(f"    {f:10s} {dict(sorted(collections.Counter(str(x[f]) for x in pos_all).items()))}")
        log("\n  통과 셀 상위 8 (TRAIN 초과 기준):")
        for x in sorted(pos_all, key=lambda z: -z["splits"]["TRAIN"]["excess_vs_null"])[:8]:
            t_, v_, o_ = (x["splits"][s] for s in ("TRAIN", "VAL", "OOS"))
            sl_txt = "없음" if x["sl_bp"] is None else format(x["sl_bp"], ".0f")
            log(f"    TP{x['tp_bp']:5.1f} 물타기{x['add_bp']:5.0f}×{x['max_adds']}회 손절{sl_txt:>5s}"
                f" 트랜치{x['tranche']:.0%} | T {t_['account_pct']:+.3f}({t_['excess_vs_null']:+.3f})"
                f" V {v_['account_pct']:+.3f}({v_['excess_vs_null']:+.3f})"
                f" O {o_['account_pct']:+.3f}({o_['excess_vs_null']:+.3f}) | 전손 {t_['ruin_rate']:.2%}")

    # ── (b) 다중성 귀무 ──
    log(f"\n=== (b) 다중성 귀무 {B_NULL}회 (칩 발동 -> 무작위 봉, 같은 창별 건수·측면 비율) ===")
    null_counts = []
    for b in range(B_NULL):
        Pb, Sb = [], []
        for sp in ("TRAIN", "VAL", "OOS"):
            cnt = int((spP == sp).sum())
            cand = pool[split[pool] == sp]
            Pb.append(rng.choice(cand, size=cnt, replace=True))
            Sb.append(rng.choice(S[spP == sp], size=cnt, replace=False))
        Pb, Sb = np.concatenate(Pb), np.concatenate(Sb)
        spB = split[Pb]
        cnt_pass = 0
        for (tp, add, ma, sl, tr) in combos:
            if ma and sl is not None and sl <= add:
                continue
            vb = np.array([simulate(o, h, l, c, p, s, tp, add, ma, sl, tr, n)[0]
                           for p, s in zip(Pb, Sb)], dtype=float)
            if all(vb[spB == sp].mean() > 0 for sp in ("TRAIN", "VAL", "OOS")):
                cnt_pass += 1
        null_counts.append(cnt_pass)
        log(f"    복제 {b+1}/{B_NULL}: 통과 {cnt_pass}셀 ({time.time()-t0:.0f}s)")
    nc = np.array(null_counts)
    pctl = float((nc < len(pos_all)).mean() * 100)
    log(f"\n  귀무 통과 셀 수: 평균 {nc.mean():.1f} · 중앙 {np.median(nc):.0f} · "
        f"95분위 {np.percentile(nc,95):.0f} · 최대 {nc.max()}")
    log(f"  ⭐실측 통과 {len(pos_all)}셀 -> 귀무분포의 {pctl:.1f} 백분위")
    verdict = ("우연으로 설명되지 않는다" if pctl >= 95 else
               "우연 범위 -- 통과 수가 신호가 아니다")
    log(f"  판정: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(
        {"cost_bp": COST_BP, "leverage": LEVERAGE, "hold_bars": HOLD, "holdout_touched": False,
         "n_cells": len(cells), "n_valid": len(valid),
         "pass_all_positive": len(pos_all), "pass_ci_lower_gt0": len(ci_pos),
         "null_pass_counts": null_counts, "null_percentile_of_observed": round(pctl, 1),
         "verdict": verdict, "cells": cells}, ensure_ascii=False, indent=2))
    log(f"\n산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
