#!/usr/bin/env python3
"""Deep-audit check (user flagged orthogonal_combo's 91-96% trailing-stop win rate as suspiciously
high): isolate how much of that win rate is the EXIT STRUCTURE itself (SL=4.0/ARM=0.5/Trail=0.1xATR
-- a very small arm distance means almost any bar that moves even slightly favorably locks in a
small win) versus genuine directional skill from the delta_z/funding_z CONFIRMATION leg.

Builds the "oscillator-only" fire population (p_fast/p_slow double-extreme, with NO delta_z/
funding_z confirmation -- i.e. drop the exact condition that makes this signal "orthogonal") using
the SAME cluster-anchor logic as the real signal (cluster_dedup_oscillator, unchanged, imported
verbatim), then runs it through the IDENTICAL trailing-stop exit structure and standard backtest
engine as backtest_eth_orthogonal_combo_trailing_gridsearch_20260831.py's winning config. If this
weaker, unconfirmed population shows a SIMILARLY high win rate/bp, the exit structure itself
explains most of the headline number, not the confirmation leg's skill. If it's meaningfully worse,
the confirmation leg is adding real economic value (matches the already-established classification-
AUC lift of 1.232x from random_bar_baseline_oscillator_only in the original screening script -- this
tests whether that same lift shows up in bp terms, not just hit-rate terms).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position
from research_eth_orthogonal_combo_metalabel_tabpfn_20260830 import build_raw_fires
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame, load_klines

MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
HORIZON_BARS, GAP = 24, 12
SL, ARM, TRAIL = 4.0, 0.5, 0.1  # orthogonal_combo's own winning config, unchanged

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")


def log(msg: str) -> None:
    print(f"[orthogonal_oscillator_only_baseline] {msg}", flush=True)


def build_oscillator_only_fires(indicator_frame: pd.DataFrame, klines: pd.DataFrame, gap: int, horizon: int) -> pd.DataFrame:
    """Same shape as build_raw_fires()'s output (pos/timestamp/side/atr_pct/...), but the fire
    condition is p_fast/p_slow double-extreme ALONE -- no delta_z/funding_z confirmation leg."""
    p_fast = indicator_frame["p_fast"]
    p_slow = indicator_frame["p_slow"]
    sig_fake = pd.DataFrame({
        "timestamp": klines["timestamp"], "high": klines["high"], "low": klines["low"], "close": klines["close"],
        "bottom_orthogonal_combo": (p_fast <= 0.10) & (p_slow <= 0.10),
        "top_orthogonal_combo": (p_fast >= 0.90) & (p_slow >= 0.90),
    })
    return build_raw_fires(indicator_frame, sig_fake, gap, horizon)


def run_backtest(fires: pd.DataFrame, klines: pd.DataFrame, mask: np.ndarray, tag: str) -> dict:
    decision_indices = fires["pos"].to_numpy(dtype=np.int64)[mask]
    is_long = (fires["side"] == "bottom").to_numpy()[mask]
    scores = np.where(is_long, 1.0, -1.0)
    atr = fires["atr_pct"].to_numpy()[mask]
    tp_placeholder = np.full(mask.sum(), 999.0)
    ts = klines["timestamp"]
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))

    result = simulate_single_position(
        timestamps=ts, open_px=open_px, high=high, low=low, close=close,
        decision_indices=decision_indices, scores=scores,
        tp_moves=tp_placeholder, sl_moves=SL * atr,
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        arm_moves=ARM * atr, trail_moves=TRAIL * atr,
    )
    ledger = result.ledger
    n_trades = int(len(ledger))
    avg_bp = float(ledger["trade_return"].mean() * 10000) if n_trades else float("nan")
    win_rate = float((ledger["price_move"] > 0).mean()) if n_trades else float("nan")
    log(f"  [{tag}] candidates={int(mask.sum())} trades={n_trades} skipped_while_open={result.skipped_while_open} "
        f"avg_trade={avg_bp:+.2f}bp win_rate={win_rate:.3f}")
    return {"tag": tag, "n_candidates": int(mask.sum()), "n_trades": n_trades, "avg_bp": avg_bp, "win_rate": win_rate}


def main() -> int:
    log("building indicator_frame + oscillator-only fires (no delta_z/funding_z confirmation)...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)

    osc_fires = build_oscillator_only_fires(indicator_frame, klines, GAP, HORIZON_BARS)
    osc_fires = osc_fires.dropna(subset=["atr_pct"]).reset_index(drop=True)
    osc_fires = osc_fires.loc[osc_fires["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    log(f"oscillator-only fires (excl. HOLDOUT): {len(osc_fires)} "
        f"(bottom={int((osc_fires['side']=='bottom').sum())}, top={int((osc_fires['side']=='top').sum())})")

    decision_indices = osc_fires["pos"].to_numpy(dtype=np.int64)
    ts = klines["timestamp"]
    eligible_val = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=HORIZON_BARS)
    eligible_oos = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=HORIZON_BARS)
    val_set = set(np.flatnonzero(eligible_val).tolist())
    oos_set = set(np.flatnonzero(eligible_oos).tolist())
    val_mask = np.array([d in val_set for d in decision_indices])
    oos_mask = np.array([d in oos_set for d in decision_indices])

    log(f"\n=== oscillator-only baseline, SAME exit structure (SL={SL}/ARM={ARM}/Trail={TRAIL}xATR) ===")
    val_r = run_backtest(osc_fires, klines, val_mask, "VAL (oscillator-only)")
    oos_r = run_backtest(osc_fires, klines, oos_mask, "OOS (oscillator-only)")

    log(f"\n=== comparison vs orthogonal_combo's real (confirmed) signal ===")
    log(f"  REAL   orthogonal_combo: VAL +9.36bp (win 91.5%, n=271)   OOS +15.13bp (win 96.0%, n=225)")
    log(f"  OSC-ONLY (no confirm):   VAL {val_r['avg_bp']:+.2f}bp (win {val_r['win_rate']*100:.1f}%, n={val_r['n_trades']})   "
        f"OOS {oos_r['avg_bp']:+.2f}bp (win {oos_r['win_rate']*100:.1f}%, n={oos_r['n_trades']})")

    out_dir = ROOT / "tmp/eth_orthogonal_combo_metalabel_tabpfn_20260830"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([val_r, oos_r]).to_csv(out_dir / "oscillator_only_baseline_economics.csv", index=False)
    log(f"saved -> {out_dir / 'oscillator_only_baseline_economics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
