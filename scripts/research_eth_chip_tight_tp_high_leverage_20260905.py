#!/usr/bin/env python3
"""증거신호 칩 + **짧은 고정 익절 · 고배율** 규약의 성과 (2026-09-05, 사용자 실사용 파라미터).

사용자: *"30배 레버리지로 짧게 5%정도만 익절해서 그런가"*.

⭐앞선 측정(`research_eth_chip_displayed_convention_pnl_20260905.py`)과 **다른 전략**이다. 그건
칩이 화면에 표시하는 익절가(k×ATR = 가격 17~105bp)까지 들고 갔다. 사용자의 실제 규약은 훨씬
타이트한 고정 목표이고, 배율이 30배다.

레버리지 환산 (CLAUDE.md Futures Risk Sizing Contract -- 가격선에 레버리지를 다시 곱하지 않는다):
    PnL(계좌) = price_move × leverage        (전액 증거금 기준)
    계좌 +5% @ 30배  ->  가격 이동 5%/30 = 0.1667% = 16.7bp
⚠️"5%"를 계좌로 읽는다("짧게 …만"이라는 표현상 가격 5%=계좌 150%는 아니다). 격자에 33.3bp
(계좌 10%)·50bp(계좌 15%)를 같이 넣어 해석 민감도를 함께 본다.

⚠️**청산**: 30배에서 유지증거금을 무시한 대략적 파산점은 가격 역행 100%/30 = 3.33%(=333bp).
손절을 안 걸면 이게 사실상의 손절이고 계좌 −100%다. 이 전략의 위험은 평균이 아니라 이 꼬리에
있으므로 **역행 333bp 도달률**을 손절 설정과 무관하게 따로 센다.

⚠️**봉내 순서**: 같은 봉에서 TP·SL을 모두 건드리면 순서를 알 수 없다. 이 저장소 규약대로
**비관**(손절 먼저)으로 판정하고, 낙관과의 격차도 같이 낸다.

모집단·진입·비용은 앞 스크립트와 동일(칩 라이브 결정 모집단, open[i+1] 진입, 왕복 10bp).
같은 측면 무작위 귀무 동반. 일군집(하루) 부트스트랩 CI. HOLDOUT(≥2026-04-01) 미접촉.
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
COST_BP, LEVERAGE = 10.0, 30.0
LIQ_BP = 1e4 / LEVERAGE                       # 333.3bp 역행 = 계좌 −100%
B_BOOT, N_NULL_POOL, SEED = 600, 40000, 20260905
OUT = ROOT / "data/research/eth_chip_tight_tp_high_leverage_20260905"

TP_GRID = [16.7, 33.3, 50.0]                  # 계좌 +5% / +10% / +15% @30배
SL_GRID = [16.7, 33.3, 50.0, 100.0, LIQ_BP]   # 마지막 = 손절 없음(청산까지)
MAX_HOLD = [12, 72]                           # 1시간 / 6시간


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def log(m: str) -> None:
    print(f"[tight-tp] {m}", flush=True)


_META = _load("meta_live2", "scripts/live_evidence_signal_metalabel_20260829.py")
CHIP = {n: int(c["horizon_bars"]) for n, c in _META.METALABEL_SIGNALS.items()}


def first_fire_positions(fired: np.ndarray, horizon: int) -> np.ndarray:
    keep, last = [], -10**9
    for i in np.flatnonzero(fired):
        if i - last > horizon:
            keep.append(i)
        last = i
    return np.asarray(keep, dtype=int)


def bracket_vec(o, h, l, c, pos, sgn, tp_bp, sl_bp, hold):
    """벡터화 고정 TP/SL 브래킷. 반환 (price_bp_pess, price_bp_opt, is_tp, liq_touch).

    비관 = 같은 봉 동시 터치 시 손절 우선. 낙관 = 익절 우선.
    """
    e = o[pos + 1]
    H = np.stack([h[p + 1:p + 1 + hold] for p in pos])
    L = np.stack([l[p + 1:p + 1 + hold] for p in pos])
    C = np.stack([c[p + 1:p + 1 + hold] for p in pos])
    up = (sgn > 0)[:, None]
    tp_px = (e * (1 + sgn * tp_bp / 1e4))[:, None]
    sl_px = (e * (1 - sgn * sl_bp / 1e4))[:, None]
    lq_px = (e * (1 - sgn * LIQ_BP / 1e4))[:, None]
    hit_tp = np.where(up, H >= tp_px, L <= tp_px)
    hit_sl = np.where(up, L <= sl_px, H >= sl_px)
    hit_lq = np.where(up, L <= lq_px, H >= lq_px)
    big = hold + 10
    f_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), big)
    f_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), big)
    timeout = sgn * (C[:, -1] / e - 1) * 1e4
    # 비관: f_sl <= f_tp 이면 손절
    pess = np.where(f_sl <= f_tp, np.where(f_sl < big, -sl_bp, timeout),
                    np.where(f_tp < big, tp_bp, timeout))
    opt = np.where(f_tp <= f_sl, np.where(f_tp < big, tp_bp, timeout),
                   np.where(f_sl < big, -sl_bp, timeout))
    is_tp = (f_tp < f_sl) & (f_tp < big)
    return pess, opt, is_tp, hit_lq.any(1)


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
    _s1 = _load("s1_tight", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
    _s1.VAL_END = OOS_END
    sig, _f, _e = _s1.build_sig()
    ts = pd.to_datetime(sig["timestamp"]).dt.tz_localize(None)
    m = (ts < OOS_END).to_numpy()
    sig, ts = sig.loc[m].reset_index(drop=True), ts.loc[m].reset_index(drop=True)
    o, h, l, c = (sig[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    n = len(sig)
    day = ts.dt.floor("D").to_numpy()
    split = np.where(ts < TRAIN_END, "TRAIN", np.where(ts < VAL_END, "VAL", "OOS"))
    log(f"  {n:,}봉 · 청산선 역행 {LIQ_BP:.1f}bp (계좌 −100% @{LEVERAGE:.0f}배) · 비용 {COST_BP}bp")

    P, S = [], []
    for name, hz in CHIP.items():
        for side in ("bottom", "top"):
            col = f"{side}_{name}"
            if col not in sig.columns:
                continue
            for pos in first_fire_positions(sig[col].fillna(False).to_numpy(bool), hz):
                P.append(int(pos)); S.append(1.0 if side == "bottom" else -1.0)
    P, S = np.asarray(P), np.asarray(S)
    ok = P + 1 + max(MAX_HOLD) < n
    P, S = P[ok], S[ok]
    log(f"  칩 발동 {len(P):,}건 (롱 {int((S>0).sum()):,} · 숏 {int((S<0).sum()):,})")

    # 같은 측면 무작위 귀무 풀: 측면 비율을 유지한 무작위 봉(격자 셀마다 재사용)
    long_frac = float((S > 0).mean())
    npool = np.flatnonzero(np.arange(n) + 1 + max(MAX_HOLD) < n)
    Pn = rng.choice(npool, size=N_NULL_POOL, replace=True)
    Sn = np.where(rng.random(N_NULL_POOL) < long_frac, 1.0, -1.0)
    spn = split[Pn]

    report = {"leverage": LEVERAGE, "liq_bp": round(LIQ_BP, 1), "cost_bp": COST_BP,
              "account_pct_per_tp": {f"{t}bp": round(t / 1e4 * LEVERAGE * 100, 1) for t in TP_GRID},
              "intrabar": "pessimistic (SL first on tie); optimistic shown as delta",
              "holdout_touched": False, "n_fires": int(len(P)), "cells": []}

    log("\n=== 칩 발동 + 고정 TP/SL · 30배 (비관 봉내순서 · 비용 10bp 차감) ===")
    log(f"{'보유':>4s} {'TP(계좌)':>11s} {'SL':>9s} {'창':6s} {'n':>6s} {'익절률':>6s} "
        f"{'가격bp':>8s} {'계좌%':>7s} {'일CI(계좌%)':>18s} {'청산도달':>7s} {'무작위(계좌%)':>12s}")
    for hold in MAX_HOLD:
        for tp in TP_GRID:
            for sl in SL_GRID:
                pess, opt, is_tp, liq = bracket_vec(o, h, l, c, P, S, tp, sl, hold)
                pess = pess - COST_BP
                npess, _no, _ni, _nl = bracket_vec(o, h, l, c, Pn, Sn, tp, sl, hold)
                npess = npess - COST_BP
                sl_lbl = "없음(청산)" if sl >= LIQ_BP - 1 else f"{sl:.0f}bp"
                for sp in ("TRAIN", "VAL", "OOS"):
                    msk = split[P] == sp
                    if msk.sum() < 30:
                        continue
                    v, d = pess[msk], day[P][msk]
                    lo, hi = day_ci(v, d, rng)
                    acct = v.mean() / 1e4 * LEVERAGE * 100
                    nm = npess[spn == sp]
                    nacct = (nm.mean() / 1e4 * LEVERAGE * 100) if len(nm) else np.nan
                    log(f"{hold:4d} {tp:5.1f}(+{tp/1e4*LEVERAGE*100:4.1f}%) {sl_lbl:>9s} {sp:6s} "
                        f"{int(msk.sum()):6,d} {is_tp[msk].mean():6.1%} {v.mean():+8.2f} "
                        f"{acct:+7.2f} [{lo/1e4*LEVERAGE*100:+7.2f},{hi/1e4*LEVERAGE*100:+7.2f}] "
                        f"{liq[msk].mean():7.2%} {nacct:+12.2f}")
                    report["cells"].append({
                        "hold_bars": hold, "tp_bp": tp, "tp_account_pct": round(tp / 1e4 * LEVERAGE * 100, 2),
                        "sl_bp": (None if sl >= LIQ_BP - 1 else sl), "split": sp,
                        "n": int(msk.sum()), "tp_rate": round(float(is_tp[msk].mean()), 4),
                        "price_bp": round(float(v.mean()), 3), "account_pct": round(acct, 3),
                        "ci_account_pct": [round(lo / 1e4 * LEVERAGE * 100, 2),
                                           round(hi / 1e4 * LEVERAGE * 100, 2)],
                        "liq_touch_rate": round(float(liq[msk].mean()), 4),
                        "null_account_pct": (round(nacct, 3) if np.isfinite(nacct) else None),
                        "optimistic_price_bp": round(float((opt[msk] - COST_BP).mean()), 3)})
        log("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
