#!/usr/bin/env python3
"""Apples-to-apples check: run this session's 8-diverse-seed ensemble tape
(tmp/causal_regen_20260516/sigma3_1h_seed_ensemble_20260801/ensemble_tape.parquet) through the
EXACT SAME trailing-stop backtest engine used to produce the "Sigma3-1h alone (thr0.6/lev3/sl1.5,
no regime filter)" baseline in this morning's joint-portfolio research
(scripts/run_sigma6_regime_trend_20260705.py's backtest(), reg_mode="none").

Important finding while tracing this: the frozen 07-05 checkpoint was ALREADY a 5-seed ensemble
(scripts/train_sigma3_1h_ensemble_20260705.py, SEEDS=[270705,270710,270715,270720,270725]) -- but
those 5 seeds are tightly clustered increments of one base value, not diverse RNG draws. This
session's 8-seed test used genuinely diverse seeds (270705, 314159, 27, 1000, 42, 8675309, 2026,
555) and found much larger variance than the original 5-seed test implied. This script checks
whether that wider, more honest seed sample changes the actual trailing-stop VAL/OOS numbers used
in the live joint-portfolio candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402
from run_sigma6_regime_trend_20260705 import backtest, PFX, REG_DIR, CM_DIR  # noqa: E402

ENSEMBLE_TAPE = ROOT / "tmp/causal_regen_20260516/sigma3_1h_seed_ensemble_20260801/ensemble_tape.parquet"
FROZEN_TAPE = ROOT / "tmp/causal_regen_20260516/sigma3_1h_hgb_20260705/tape_ensemble.parquet"

VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")

BASE = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3)
CFG = dict(leverage=3.0, sl_atr=1.5, reg_mode="none", reg_thr=0.34, stab_thr=0.0, fee_mult=1.0)
THR = 0.60


def load_with_regime(tape_path: Path) -> pd.DataFrame:
    t = pd.read_parquet(tape_path)
    t["timestamp"] = pd.to_datetime(t["timestamp"]).astype("datetime64[ns]")
    t = t.sort_values("timestamp").reset_index(drop=True)
    reg = pd.concat([
        pd.read_csv(REG_DIR / "training_features_2025_regime3_current_sensitive_hmm_wide24.csv", parse_dates=["timestamp"]),
        pd.read_csv(REG_DIR / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv", parse_dates=["timestamp"]),
    ], ignore_index=True).sort_values("timestamp")
    reg["timestamp"] = reg["timestamp"].astype("datetime64[ns]")
    keep = ["timestamp", f"{PFX}bull_prob", f"{PFX}bear_prob", f"{PFX}chop_prob"]
    t = pd.merge_asof(t, reg[keep], on="timestamp", direction="backward")
    cm = pd.concat([
        pd.read_csv(CM_DIR / "training_features_2025_regime3_cryptomamba_h6_sidecar_20260601.csv", parse_dates=["timestamp"]),
        pd.read_csv(CM_DIR / "training_features_2026_rebuilt_regime3_cryptomamba_h6_sidecar_20260601.csv", parse_dates=["timestamp"]),
    ], ignore_index=True).sort_values("timestamp")
    cm["timestamp"] = cm["timestamp"].astype("datetime64[ns]")
    t = pd.merge_asof(t, cm[["timestamp", "regime3_cmamba_h6_sidecar_stability_score"]], on="timestamp", direction="backward")
    return t.sort_values("i" if "i" in t.columns else "timestamp").reset_index(drop=True)


def run_both(tape_path: Path, label: str) -> None:
    raw = load_with_regime(tape_path)
    tape = v2.apply_quality_threshold(raw, THR)
    val = backtest(tape, start=VAL_START, end=VAL_END, **BASE, **CFG)
    oos = backtest(tape, start=OOS_START, end=OOS_END, **BASE, **CFG)
    print(f"\n=== {label} (thr{THR}/lev{CFG['leverage']}/sl{CFG['sl_atr']}, no regime filter) ===", flush=True)
    print(f"VAL 2025-10..12: pnl={val['pnl']:.2f}% mdd={val['mdd']:.2f}% trades={val['trades']} wr={val['wr']:.3f} months={len(val['by_month'])}", flush=True)
    print(f"OOS 2026-01..03: pnl={oos['pnl']:.2f}% mdd={oos['mdd']:.2f}% trades={oos['trades']} wr={oos['wr']:.3f} months={len(oos['by_month'])}", flush=True)


def main() -> int:
    run_both(FROZEN_TAPE, "FROZEN 5-seed (270705,270710,270715,270720,270725) -- original 07-05")
    run_both(ENSEMBLE_TAPE, "THIS SESSION 8 diverse-seed ensemble")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
