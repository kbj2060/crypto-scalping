#!/usr/bin/env python3
"""Sigma9: combined BTC+ETH 2-asset book vs ETH-Sigma6 alone, VAL 2025-07..12.

Honest finding from run_sigma9_btc_standalone_20260706.py: BTC has no Regime3 HMM (that sidecar
was trained on ETH-only features), so BTC trades every signal ungated. Its best standalone VAL
config (thr=0.60, lev=2, sl_atr=1.5) only reaches +16.6% with -9.6% MDD -- far weaker than
ETH-Sigma6 (+34.3% lev3 / +71.1% lev4, both regime-gated). This script tests whether blending a
WEAK-but-differently-timed BTC sleeve into the book still helps on a risk-adjusted basis (lower
combined MDD) versus just running ETH-Sigma6 alone, using a 50/50 capital-weighted "sleeve" model:
combined_equity(t) = 0.5 * eth_equity(t) + 0.5 * btc_equity(t), each sleeve run at its OWN
best-found VAL config and its own full margin (0.30) -- i.e. as if half the book is independently
managed by each strategy, not sharing margin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402
import run_sigma6_regime_trend_20260705 as s6  # noqa: E402

ETH_TAPE = ROOT / "tmp/causal_regen_20260516/sigma3_1h_hgb_20260705/tape_ensemble.parquet"
BTC_TAPE = ROOT / "tmp/causal_regen_20260516/sigma9_1h_btc_20260706/tape_btc_ensemble.parquet"
REG_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
CM_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma9_1h_btc_20260706"
PFX = s6.PFX


def backtest_with_curve(tape, *, leverage, margin, trail_atr, sl_atr, min_profit_atr, max_hold,
                         cooldown, reg_mode, reg_thr, stab_thr, fee_mult, start, end):
    """Copy of s6.backtest() that also returns a per-bar (timestamp, equity) curve."""
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    n = len(sub)
    close = sub["close"].to_numpy(np.float64)
    open_ = sub["open"].to_numpy(np.float64)
    side_arr = sub["primary_side"].to_numpy(np.int64)
    atr_arr = sub["atr_pct"].to_numpy(np.float64)
    bull = sub[f"{PFX}bull_prob"].to_numpy(np.float64)
    bear = sub[f"{PFX}bear_prob"].to_numpy(np.float64)
    chop = sub[f"{PFX}chop_prob"].to_numpy(np.float64)
    stab = sub["regime3_cmamba_h6_sidecar_stability_score"].fillna(1.0).to_numpy(np.float64)
    FEE, SLIP = 0.00020 * fee_mult, 0.00050 * fee_mult
    notional = margin * leverage
    cash = peak_eq = 1.0
    mdd = 0.0
    pos = 0
    entry_price = peak_unreal = entry_atr = 0.0
    hold_start = 0
    entry_equity = 1.0
    trades = []
    cooldown_until = -1
    curve_ts, curve_eq = [], []
    i = 0
    while i < n - 1:
        if pos == 0:
            curve_ts.append(sub["timestamp"].iloc[i]); curve_eq.append(cash)
            if i < cooldown_until or side_arr[i] == 0:
                i += 1
                continue
            side = int(side_arr[i])
            ok = True
            if reg_mode == "trend_agree":
                ok = (side > 0 and bull[i] >= reg_thr) or (side < 0 and bear[i] >= reg_thr)
            elif reg_mode == "not_chop":
                ok = chop[i] < reg_thr
            if ok and stab_thr > 0:
                ok = stab[i] >= stab_thr
            if not ok:
                i += 1
                continue
            entry_price = float(open_[min(i + 1, n - 1)]) * (1 + SLIP if side > 0 else 1 - SLIP)
            pos, hold_start, peak_unreal, entry_atr = side, i, 0.0, max(atr_arr[i], 1e-6)
            entry_equity = cash
            cash -= cash * FEE * notional
            i += 1
            continue
        px = close[i]
        raw = (px * (1 - SLIP) - entry_price) / entry_price if pos > 0 else (entry_price - px * (1 + SLIP)) / entry_price
        unreal = raw * notional
        eq = cash * (1 + unreal)
        curve_ts.append(sub["timestamp"].iloc[i]); curve_eq.append(eq)
        peak_eq = max(peak_eq, eq)
        mdd = min(mdd, eq / max(peak_eq, 1e-12) - 1)
        peak_unreal = max(peak_unreal, unreal)
        hold = i - hold_start
        reason = ""
        if unreal <= -sl_atr * entry_atr:
            reason = "stop"
        elif peak_unreal >= min_profit_atr * entry_atr and (peak_unreal - unreal) >= trail_atr * entry_atr:
            reason = "trail"
        elif hold >= max_hold:
            reason = "time"
        if reason:
            exit_price = close[i] * (1 - SLIP if pos > 0 else 1 + SLIP)
            rex = (exit_price - entry_price) / entry_price if pos > 0 else (entry_price - exit_price) / entry_price
            before = cash
            cash = cash * (1 + rex * notional)
            cash -= before * FEE * notional
            trades.append({"win": cash > entry_equity, "month": str(sub.iloc[hold_start]["timestamp"])[:7], "reason": reason, "ret": rex * notional})
            pos = 0
            cooldown_until = i + cooldown
        i += 1
    wins = sum(1 for t in trades if t["win"])
    curve = pd.Series(curve_eq, index=pd.to_datetime(curve_ts)).sort_index()
    curve = curve[~curve.index.duplicated(keep="last")]
    return {"pnl": (cash - 1) * 100, "mdd": mdd * 100, "trades": len(trades),
            "wr": wins / len(trades) if trades else 0.0, "curve": curve}


def mdd_of(curve: pd.Series) -> float:
    peak = curve.cummax()
    return float((curve / peak - 1).min() * 100)


def main() -> int:
    # ETH-Sigma6 leg: best VAL config (thr=0.70, lev=4, not_chop, reg_thr=0.42, stab_thr=0.55)
    eth_raw = s6.load_tape_with_regime()
    eth_tape = v2.apply_quality_threshold(eth_raw, 0.70)
    eth_base = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3,
                     reg_mode="not_chop", stab_thr=0.55)
    # documented frozen Sigma6 winners: lev3 uses reg_thr=0.50 (+34.3% VAL), lev4 uses 0.42 (+71.1% VAL)
    eth3 = backtest_with_curve(eth_tape, leverage=3.0, sl_atr=2.5, reg_thr=0.50, fee_mult=1.0, start=s6.VAL_START, end=s6.VAL_END, **eth_base)
    eth4 = backtest_with_curve(eth_tape, leverage=4.0, sl_atr=2.5, reg_thr=0.42, fee_mult=1.0, start=s6.VAL_START, end=s6.VAL_END, **eth_base)

    # BTC leg: best standalone VAL config (thr=0.60, lev=2, sl_atr=1.5), no regime filter
    btc_raw = pd.read_parquet(BTC_TAPE)
    btc_raw["timestamp"] = pd.to_datetime(btc_raw["timestamp"])
    btc_raw = btc_raw.sort_values("timestamp").reset_index(drop=True)
    btc_raw[f"{PFX}bull_prob"] = 0.0
    btc_raw[f"{PFX}bear_prob"] = 0.0
    btc_raw[f"{PFX}chop_prob"] = 0.0
    btc_raw["regime3_cmamba_h6_sidecar_stability_score"] = 1.0
    btc_tape = v2.apply_quality_threshold(btc_raw, 0.60)
    btc_base = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3,
                     reg_mode="none", reg_thr=0.0, stab_thr=0.0)
    btc2 = backtest_with_curve(btc_tape, leverage=2.0, sl_atr=1.5, fee_mult=1.0, start=s6.VAL_START, end=s6.VAL_END, **btc_base)

    print(f"ETH-Sigma6 lev3 alone: c1={eth3['pnl']:.1f}% mdd={eth3['mdd']:.1f}% tr={eth3['trades']} wr={eth3['wr']:.3f}", flush=True)
    print(f"ETH-Sigma6 lev4 alone: c1={eth4['pnl']:.1f}% mdd={eth4['mdd']:.1f}% tr={eth4['trades']} wr={eth4['wr']:.3f}", flush=True)
    print(f"BTC standalone (no regime): c1={btc2['pnl']:.1f}% mdd={btc2['mdd']:.1f}% tr={btc2['trades']} wr={btc2['wr']:.3f}", flush=True)

    for name, eth_leg in (("lev3", eth3), ("lev4", eth4)):
        idx = eth_leg["curve"].index.union(btc2["curve"].index)
        e = eth_leg["curve"].reindex(idx).ffill().fillna(1.0)
        b = btc2["curve"].reindex(idx).ffill().fillna(1.0)
        combined = 0.5 * e + 0.5 * b
        c_pnl = (combined.iloc[-1] - 1) * 100
        c_mdd = mdd_of(combined)
        print(f"\n=== Combined 50/50 book: ETH-{name} + BTC-standalone ===", flush=True)
        print(f"  combined c1={c_pnl:.1f}% mdd={c_mdd:.1f}%  (vs ETH-{name} alone c1={eth_leg['pnl']:.1f}% mdd={eth_leg['mdd']:.1f}%)", flush=True)
        combined.to_frame("equity").to_csv(OUT_DIR / f"combined_curve_eth_{name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
