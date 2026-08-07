"""Same h48qual parent architecture/training as
train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708.py, pointed at a feature file
restricted to exactly ETH's live h48qual 102 base_cols (BTC's own 147 base_cols minus the 45
BTC-only cross-asset/microstructure columns: cvd_*, btc_ret_*, eth_btc_*, vwap_*, funding_oi_*,
wick/sweep/breakout flags, atr_pct_rank_288, bb_width_pct_rank_288, compression_*,
distance_to_day_high_low_pct -- see data/splits/year_oos_eth102_btc_20260802/ build step). Only the
input feature set differs: this is the reverse ablation of
train_eval_omega4_3head_parent72_loose_entry_quality_btc_metrics4_20260802.py (which ADDED 4 cols;
this REMOVES 45). Same labels, same architecture, same hyperparameters, same regime3 sidecar.
Mirrors the metrics4 script's monkeypatch convention exactly. Does NOT touch the live checkpoint
dir or the baseline feature files.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708 as parent_script  # noqa: E402

parent_script.omega.TRAIN_CSV = ROOT / "data/splits/year_oos_eth102_btc_20260802/btc_features_2025.csv"
parent_script.omega.EVAL_CSV = ROOT / "data/splits/year_oos_eth102_btc_20260802/btc_features_2026.csv"

if __name__ == "__main__":
    # Match the production h48qual run's exact args (its own report.json label_contract), same
    # convention as the metrics4/adaptive_squeeze/regime_docs42 BTC h48qual variants.
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "h48qual_eth102_20260802"]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "quality_label_action"]
    if "--quality-label-dir" not in sys.argv:
        sys.argv += ["--quality-label-dir",
                      "tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708"]
    raise SystemExit(parent_script.main())
