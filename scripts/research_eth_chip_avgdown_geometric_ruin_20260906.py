#!/usr/bin/env python3
"""물타기 최선 셀의 **기하 성장률·파산 확률** 검사 (2026-09-06).

배경: `..._avgdown_boundary_and_multiplicity_20260905.py`가 확장 격자에서 세 창 전부 양수 126/243,
일CI 하한>0 52/243, 다중성 귀무 100백분위(무작위 격자 평균 41.9셀 대비 126셀)를 냈다. **산술평균
기준으로는 통과**다.

⚠️그런데 물타기 전략의 산술평균은 위험하다. 전손(투입 증거금 전액 손실)이 섞여 있으면
**산술평균이 양수여도 복리로는 자본이 사라진다**. 트랜치 20% × 5트랜치에서의 전손은 계좌 −100%이고,
log(1 + (−1)) = −∞ 이므로 그 사건이 한 번이라도 가능하면 장기 성장률은 −∞다.

그래서 재는 것:
  1. 산술평균 vs **기하 성장률** E[log(1+r)]
  2. 전손 구성: k(트랜치 수)별 전손 분포 -- k=5(트랜치20%)면 정확히 −100%
  3. **순차 복리 시뮬레이션**: 시간 순서대로 자본에 곱해가며 실제 자본곡선·최대낙폭·파산 시점
  4. 파산까지 기대 거래 수 / 일수

⚠️봉내 순서·수수료·규약은 선행 스크립트 상속(비용 4bp 명목 = 증거금 1.20%). HOLDOUT 미접촉.
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
SEED = 20260906
OUT = ROOT / "data/research/eth_chip_avgdown_geometric_ruin_20260906"

# 확장 격자 상위 셀들(TRAIN 초과 기준 상위 + 트랜치 10% 대표)
CELLS = [
    {"tp": 33.3, "add": 50.0, "ma": 4, "sl": 200.0, "tr": 0.20, "tag": "상위1"},
    {"tp": 33.3, "add": 50.0, "ma": 4, "sl": None, "tr": 0.20, "tag": "상위2(손절없음)"},
    {"tp": 33.3, "add": 50.0, "ma": 4, "sl": 100.0, "tr": 0.20, "tag": "손절 타이트"},
    {"tp": 33.3, "add": 50.0, "ma": 9, "sl": 200.0, "tr": 0.10, "tag": "트랜치10%×9회"},
    {"tp": 16.7, "add": 50.0, "ma": 4, "sl": 200.0, "tr": 0.20, "tag": "TP16.7(사용자값)"},
]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def log(m: str) -> None:
    print(f"[geo-ruin] {m}", flush=True)


_META = _load("meta_live6", "scripts/live_evidence_signal_metalabel_20260829.py")
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


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    log("신호 프레임 재구성...")
    _s1 = _load("s1_geo", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
    _s1.VAL_END = OOS_END
    sig, _f, _e = _s1.build_sig()
    ts = pd.to_datetime(sig["timestamp"]).dt.tz_localize(None)
    m0 = (ts < OOS_END).to_numpy()
    sig, ts = sig.loc[m0].reset_index(drop=True), ts.loc[m0].reset_index(drop=True)
    o, h, l, c = (sig[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    n = len(sig)
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
    order = np.argsort(P)                       # ⭐시간 순서(순차 복리용)
    P, S = P[order], S[order]
    days_total = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400
    log(f"  칩 발동 {len(P):,}건 · {days_total:.0f}일 · {len(P)/days_total:.1f}건/일 · 비용 {COST_BP:g}bp")

    report = {"cost_bp": COST_BP, "leverage": LEVERAGE, "holdout_touched": False, "cells": []}
    for cf in CELLS:
        r = [simulate(o, h, l, c, p, s, cf["tp"], cf["add"], cf["ma"], cf["sl"], cf["tr"], n)
             for p, s in zip(P, S)]
        v = np.array([x[0] for x in r], dtype=float)            # 계좌 %
        oc = np.array([x[1] for x in r], dtype=object)
        kk = np.array([x[2] for x in r], dtype=int)
        rr = v / 100.0                                          # 비율
        arith = float(rr.mean())
        n_total_ruin = int((rr <= -0.999).sum())                # 정확히 −100% (log = −inf)
        finite = rr > -0.999
        geo_ex = float(np.log1p(rr[finite]).mean()) if finite.any() else float("nan")

        # ── 순차 복리(자본 곡선) ──
        eq, curve, ruin_at = 1.0, [], None
        for i, x in enumerate(rr):
            eq *= (1.0 + x)
            curve.append(eq)
            if eq <= 1e-9 and ruin_at is None:
                ruin_at = i + 1
                break
        curve = np.asarray(curve)
        peak = np.maximum.accumulate(curve)
        mdd = float((curve / peak - 1).min())

        # 파산 확률(부트스트랩 순서 무작위, 1년치 = 발동률 x 365)
        per_day = len(P) / days_total
        horizon_trades = int(per_day * 365)
        n_sim, ruined = 300, 0
        med_final = []
        for _ in range(n_sim):
            pick = rng.choice(len(rr), size=min(horizon_trades, 20000), replace=True)
            eq2 = 1.0
            for x in rr[pick]:
                eq2 *= (1.0 + x)
                if eq2 <= 1e-9:
                    ruined += 1
                    eq2 = 0.0
                    break
            med_final.append(eq2)
        p_ruin = ruined / n_sim

        sl_txt = "없음" if cf["sl"] is None else f"{cf['sl']:.0f}"
        log(f"\n=== {cf['tag']}: TP{cf['tp']} 물타기{cf['add']:.0f}×{cf['ma']}회 손절{sl_txt} 트랜치{cf['tr']:.0%} ===")
        log(f"  산술평균   {arith*100:+.4f}%/건        (격자가 통과 판정에 쓴 값)")
        log(f"  ⭐기하평균  {geo_ex*100:+.4f}%/건 (전손 제외분)  · 전손(−100%) {n_total_ruin}건 "
            f"({n_total_ruin/len(rr):.3%}) -> 포함 시 장기 성장률 −∞")
        log(f"  청산 발생   {int((oc=='liq').sum()):,}건 ({(oc=='liq').mean():.2%}) · "
            f"그중 k={cf['ma']+1}(전액) {n_total_ruin:,}건 · 평균 트랜치 {kk.mean():.2f}")
        log(f"  순차 복리   {'파산(거래 '+str(ruin_at)+'번째)' if ruin_at else f'최종 자본 x{curve[-1]:.3g}'} "
            f"· 최대낙폭 {mdd:.1%}")
        log(f"  1년 파산확률(재표집 {n_sim}회, {horizon_trades:,}건) = **{p_ruin:.1%}** · "
            f"생존 시 자본 중앙 x{np.median([x for x in med_final if x>0]) if any(x>0 for x in med_final) else 0:.3g}")
        report["cells"].append({
            **{k: cf[k] for k in ("tp", "add", "ma", "sl", "tr", "tag")},
            "arith_pct": round(arith * 100, 4), "geo_ex_ruin_pct": round(geo_ex * 100, 4),
            "n_total_ruin": n_total_ruin, "total_ruin_rate": round(n_total_ruin / len(rr), 5),
            "liq_rate": round(float((oc == "liq").mean()), 4), "mean_tranches": round(float(kk.mean()), 3),
            "sequential_ruin_at_trade": ruin_at,
            "sequential_final_equity": (None if ruin_at else round(float(curve[-1]), 6)),
            "max_drawdown": round(mdd, 4), "p_ruin_1y": round(p_ruin, 4),
            "trades_per_day": round(per_day, 2)})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log(f"\n산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
