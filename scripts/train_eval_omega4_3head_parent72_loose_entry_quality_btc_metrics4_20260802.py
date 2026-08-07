"""Same h48qual parent architecture/training as
train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708.py, pointed at the metrics4
feature files (build_btc_features_metrics4_20260802.py + split_btc_features_metrics4_by_year_20260802.py)
instead of the original ones. Only the input feature set differs: adds taker_vol_ratio_z,
count_toptrader_ratio_z, toptrader_count_size_divergence, sig_whale, sig_oi_divergence -- same
labels, same architecture, same hyperparameters, same regime3 sidecar. Mirrors
train_eval_omega4_3head_parent72_loose_entry_quality_btc_adaptive_squeeze_20260720.py's
monkeypatch convention exactly. Does NOT touch the live checkpoint dir or the baseline feature
files.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708 as parent_script  # noqa: E402

parent_script.omega.TRAIN_CSV = ROOT / "data/splits/year_oos_metrics4_btc_20260802/btc_features_2025.csv"
parent_script.omega.EVAL_CSV = ROOT / "data/splits/year_oos_metrics4_btc_20260802/btc_features_2026.csv"

if __name__ == "__main__":
    # Match the production h48qual run's exact args (its own report.json label_contract), which
    # are NOT this script's own CLI defaults (default quality-mode is "hard_rule", production
    # used "quality_label_action" with an explicit quality-label-dir). Same convention as the
    # adaptive_squeeze/regime_docs42 BTC h48qual variants.
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "h48qual_metrics4_20260802"]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "quality_label_action"]
    if "--quality-label-dir" not in sys.argv:
        sys.argv += ["--quality-label-dir",
                      "tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708"]
    raise SystemExit(parent_script.main())
