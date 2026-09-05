#!/usr/bin/env python3
"""증거신호 칩 + 20% 트랜치 · 30배 · 물타기 + **손절라인** 격자 튜닝 (2026-09-05, 사용자 요청).

사용자: *"손절라인도 추가해서 시뮬레이션 테스트 튜닝해줘"*.

선행 3편:
  1. `..._displayed_convention_pnl_...`      칩 표시 익절가까지 보유 -> gross ≈ 0, 손실 = 수수료
  2. `..._tight_tp_high_leverage_...`        고정 TP/SL · 30배 전액 증거금 -> 90셀 전부 음수
  3. `..._partial_margin_averaging_down_...` 20% 트랜치 + 물타기(손절 없음) -> 전부 음수, 무작위 수준

이 편은 3에 **손절라인**을 축으로 추가하고 격자를 튜닝한다.

규약(3편 상속):
    트랜치 = equity의 20%, 배율 30 -> 트랜치당 명목 6.0×equity
    익절   = 평균 진입가 ± tp_bp 도달 -> 전량 청산
    물타기 = 평균 진입가 대비 역행 add_bp 도달 -> 트랜치 1개 추가(최대 max_adds회)
    ⭐손절 = 평균 진입가 대비 역행 sl_bp 도달 -> 전량 청산(그 시점 트랜치 전부)
             같은 봉에서 물타기 판정을 **먼저** 하므로, sl_bp > add_bp면 "물타기하다 못 버티면 손절",
             sl_bp <= add_bp면 물타기가 발동하지 못한다(축퇴 셀 -- 리포트에 degenerate로 표시)
    청산   = 역행 3.33%(=100%/30) -> 투입 증거금 전액 손실
    수수료 = 트랜치마다 왕복 10bp (물타기는 수수료도 트랜치 수만큼 낸다)
    봉내 순서는 **비관**: 청산 -> 물타기 -> 손절 -> 익절

⚠️**튜닝 규율**(이 저장소 §5-A·§5.29 §7-3):
  · 선택은 **TRAIN에서만**. VAL/OOS는 선택된 셀의 표본외 1회 조회로만 보고한다.
  · 격자 **경계 셀**을 의심한다 -- 최선이 격자 끝이면 경계 확장 없이는 채택하지 않는다.
  · 같은 측면 무작위 귀무를 셀마다 함께 낸다(하락장 드리프트 제거).
  · "몇 셀 통과"는 무의미하다 -- **무작위 부분표집 귀무**로 통과 셀 수의 기대치를 함께 낸다.
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
import os
COST_BP = float(os.environ.get("COST_BP", "10.0"))   # 명목 대비 왕복 bp.
# ⚠️단위 주의(2026-09-05 사용자 질문 "수수료는 1% 아니야?"): 여기 값은 **명목 대비**다.
#   증거금 대비 = COST_BP x leverage / 100 %   -> 10bp@30배 = 3.0%(테이커), 4bp = 1.2%(이론 메이커),
#   7.8bp = 2.34%(이 저장소 09-04 페그 메이커 실측). "수수료 1%"는 이론 메이커의 증거금 표기다.
LEVERAGE, TRANCHE = 30.0, 0.20
LIQ_BP = 1e4 / LEVERAGE
B_BOOT, SEED, HOLD = 600, 20260905, 72
OUT = ROOT / f"data/research/eth_chip_avgdown_stoploss_tuning_20260905/cost{COST_BP:g}bp"

TP_GRID = [16.7, 33.3, 50.0, 80.0]                  # 가격 bp
ADD_GRID = [100.0, 150.0, 200.0]
MAX_ADDS = [0, 2, 4]
SL_GRID = [50.0, 100.0, 150.0, 200.0, 250.0, None]  # None = 손절 없음(청산까지)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def log(m: str) -> None:
    print(f"[sl-tune] {m}", flush=True)


_META = _load("meta_live4", "scripts/live_evidence_signal_metalabel_20260829.py")
CHIP = {n: int(c["horizon_bars"]) for n, c in _META.METALABEL_SIGNALS.items()}


def first_fire_positions(fired: np.ndarray, horizon: int) -> np.ndarray:
    keep, last = [], -10**9
    for i in np.flatnonzero(fired):
        if i - last > horizon:
            keep.append(i)
        last = i
    return np.asarray(keep, dtype=int)


def simulate(o, h, l, c, pos, sgn, tp_bp, add_bp, max_adds, sl_bp, hold, n):
    """경로 의존 시뮬레이션. 반환 (account_pct, outcome, n_tranches)."""
    if pos + 1 >= n:
        return np.nan, "", 0
    entry = o[pos + 1]
    avg, k = entry, 1
    end = min(pos + 1 + hold, n - 1)
    for j in range(pos + 1, end + 1):
        hi, lo = h[j], l[j]
        adv = (avg - lo) / avg if sgn > 0 else (hi - avg) / avg
        fav = (hi - avg) / avg if sgn > 0 else (avg - lo) / avg
        if adv * 1e4 >= LIQ_BP:
            return -TRANCHE * k * 100.0, "liq", k
        while k <= max_adds and adv * 1e4 >= add_bp:
            add_px = avg * (1 - sgn * add_bp / 1e4)
            avg = (avg * k + add_px) / (k + 1)
            k += 1
            adv = (avg - lo) / avg if sgn > 0 else (hi - avg) / avg
            fav = (hi - avg) / avg if sgn > 0 else (avg - lo) / avg
            if adv * 1e4 >= LIQ_BP:
                return -TRANCHE * k * 100.0, "liq", k
        if sl_bp is not None and adv * 1e4 >= sl_bp:
            return ((-sl_bp * k - COST_BP * k) / 1e4 * TRANCHE * LEVERAGE * 100.0), "sl", k
        if fav * 1e4 >= tp_bp:
            return ((tp_bp * k - COST_BP * k) / 1e4 * TRANCHE * LEVERAGE * 100.0), "tp", k
    move = sgn * (c[end] / avg - 1) * 1e4
    return ((move * k - COST_BP * k) / 1e4 * TRANCHE * LEVERAGE * 100.0), "timeout", k


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
    _s1 = _load("s1_sl", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
    _s1.VAL_END = OOS_END
    sig, _f, _e = _s1.build_sig()
    ts = pd.to_datetime(sig["timestamp"]).dt.tz_localize(None)
    m = (ts < OOS_END).to_numpy()
    sig, ts = sig.loc[m].reset_index(drop=True), ts.loc[m].reset_index(drop=True)
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
    log(f"  ⭐비용 {COST_BP:g}bp(명목) = 증거금 대비 {COST_BP*LEVERAGE/100:.2f}%")
    log(f"  칩 발동 {len(P):,}건 · 트랜치 {TRANCHE:.0%}×{LEVERAGE:.0f}배 · 청산 역행 {LIQ_BP:.1f}bp")

    long_frac = float((S > 0).mean())
    pool = np.flatnonzero(np.arange(n) + 1 + HOLD < n)
    Pn = rng.choice(pool, size=15000, replace=True)
    Sn = np.where(rng.random(len(Pn)) < long_frac, 1.0, -1.0)
    spN = split[Pn]

    cells = []
    combos = []
    for tp in TP_GRID:
        for ma in MAX_ADDS:
            for add in (ADD_GRID if ma else [ADD_GRID[0]]):
                for sl in SL_GRID:
                    combos.append((tp, add, ma, sl))
    log(f"  격자 {len(combos)}셀 시뮬레이션...")
    for ci, (tp, add, ma, sl) in enumerate(combos, 1):
        r = [simulate(o, h, l, c, p, s, tp, add, ma, sl, HOLD, n) for p, s in zip(P, S)]
        v = np.array([x[0] for x in r], dtype=float)
        oc = np.array([x[1] for x in r], dtype=object)
        kk = np.array([x[2] for x in r], dtype=int)
        nv = np.array([simulate(o, h, l, c, p, s, tp, add, ma, sl, HOLD, n)[0]
                       for p, s in zip(Pn, Sn)], dtype=float)
        degenerate = bool(ma and sl is not None and sl <= add)
        row = {"tp_bp": tp, "add_bp": add, "max_adds": ma, "sl_bp": sl,
               "degenerate": degenerate, "splits": {}}
        for sp in ("TRAIN", "VAL", "OOS"):
            msk = spP == sp
            if msk.sum() < 30:
                continue
            lo, hi = day_ci(v[msk], dayP[msk], rng)
            nm = nv[spN == sp]
            row["splits"][sp] = {
                "n": int(msk.sum()), "win_rate": round(float((oc[msk] == "tp").mean()), 4),
                "account_pct": round(float(v[msk].mean()), 4),
                "ci": [round(lo, 4), round(hi, 4)],
                "ruin_rate": round(float((oc[msk] == "liq").mean()), 4),
                "sl_rate": round(float((oc[msk] == "sl").mean()), 4),
                "mean_tranches": round(float(kk[msk].mean()), 3),
                "null_account_pct": round(float(nm.mean()), 4) if len(nm) else None,
                "excess_vs_null": round(float(v[msk].mean() - (nm.mean() if len(nm) else 0)), 4)}
        cells.append(row)
        if ci % 20 == 0:
            log(f"    {ci}/{len(combos)} ({time.time()-t0:.0f}s)")

    valid = [x for x in cells if not x["degenerate"] and "TRAIN" in x["splits"]]
    # ── ⭐선택은 TRAIN에서만 (귀무 초과 기준). VAL/OOS는 표본외 1회 조회 ──
    best = max(valid, key=lambda x: x["splits"]["TRAIN"]["excess_vs_null"])
    log(f"\n=== 결과 요약 ({len(cells)}셀, 축퇴 {sum(1 for x in cells if x['degenerate'])}셀 제외) ===")
    pos_all = [x for x in valid if all(x["splits"][s]["account_pct"] > 0 for s in x["splits"])]
    pos_excess = [x for x in valid if all(x["splits"][s]["excess_vs_null"] > 0 for s in x["splits"])]
    ci_pos = [x for x in valid if all(x["splits"][s]["ci"][0] > 0 for s in x["splits"])]
    log(f"  세 창 전부 계좌 양수      : {len(pos_all)}/{len(valid)}")
    log(f"  세 창 전부 귀무 초과 양수 : {len(pos_excess)}/{len(valid)}")
    log(f"  세 창 전부 일CI 하한>0    : {len(ci_pos)}/{len(valid)}")

    log(f"\n⭐TRAIN에서 고른 최선 셀 (귀무 초과 기준) -- VAL/OOS는 표본외 1회 조회")
    log(f"  TP {best['tp_bp']}bp · 물타기 {best['add_bp']}bp × {best['max_adds']}회 · "
        f"손절 {best['sl_bp']}")
    for sp in ("TRAIN", "VAL", "OOS"):
        if sp not in best["splits"]:
            continue
        d = best["splits"][sp]
        log(f"   {sp:6s} n={d['n']:6,d} 승률 {d['win_rate']:6.1%} 계좌 {d['account_pct']:+7.3f}% "
            f"CI[{d['ci'][0]:+7.3f},{d['ci'][1]:+7.3f}] 손절 {d['sl_rate']:5.1%} "
            f"전손 {d['ruin_rate']:5.2%} 무작위 {d['null_account_pct']:+7.3f}% "
            f"초과 {d['excess_vs_null']:+7.3f}%")

    log("\n=== TRAIN 귀무초과 상위 10셀 (전부 표기 -- 경계 셀 확인용) ===")
    log(f"{'TP':>6s} {'물타기':>7s} {'횟수':>4s} {'손절':>6s} | "
        f"{'TRAIN 계좌%':>11s} {'초과':>7s} | {'VAL 계좌%':>10s} {'초과':>7s} | "
        f"{'OOS 계좌%':>10s} {'초과':>7s} | {'전손':>6s}")
    for x in sorted(valid, key=lambda z: -z["splits"]["TRAIN"]["excess_vs_null"])[:10]:
        t_, v_, o_ = (x["splits"].get(s) for s in ("TRAIN", "VAL", "OOS"))
        sl_txt = "없음" if x["sl_bp"] is None else f"{x['sl_bp']:.0f}"
        log(f"{x['tp_bp']:6.1f} {x['add_bp']:7.0f} {x['max_adds']:4d} {sl_txt:>6s} | "
            f"{t_['account_pct']:+11.3f} {t_['excess_vs_null']:+7.3f} | "
            f"{v_['account_pct']:+10.3f} {v_['excess_vs_null']:+7.3f} | "
            f"{o_['account_pct']:+10.3f} {o_['excess_vs_null']:+7.3f} | {t_['ruin_rate']:6.2%}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(
        {"tranche": TRANCHE, "leverage": LEVERAGE, "cost_bp": COST_BP, "hold_bars": HOLD,
         "liq_bp": round(LIQ_BP, 1), "holdout_touched": False,
         "selection_protocol": "TRAIN-only by excess_vs_null; VAL/OOS out-of-sample lookup",
         "n_cells": len(cells), "n_valid": len(valid),
         "pass_counts": {"all_positive": len(pos_all), "all_excess_positive": len(pos_excess),
                         "all_ci_lower_gt0": len(ci_pos)},
         "best_train": best, "cells": cells}, ensure_ascii=False, indent=2))
    log(f"\n산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
