#!/usr/bin/env python3
"""Extend the ilias 154-feature dataset (originally 2024-01-01..2026-06-30, built by
scripts/ilias_eth_154feature_dataset_build_20260821.py) forward through this session's Stage 1/2
end date (2026-08-19), by re-running the EXACT SAME build module with only DATE_END/OUT_DIR
overridden as module attributes before calling its main() -- same convention as
scripts/build_omega1_2_triple_barrier_labels_extended_20260809.py (tb.EVAL_CSV/tb.OUT_DIR override
+ tb.main()). Zero methodology drift: the construction code (VIF-clean 112 passthrough, 30 combo
multiplications, 12 financial-ML features, regime3 wide24 overlay merge) is byte-identical to the
original build -- only which raw rows flow through it changes.

Prerequisite done separately this session: data/splits/year_oos/training_features_2026_rebuilt.csv
extended to 2026-08-19, and its regime3_current_sensitive_hmm_wide24 sidecar re-applied (frozen
2024 joblib, no refit) to match -- see scripts/apply_regime3_wide24_sidecar_extended_20260820.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ilias_eth_154feature_dataset_build_20260821 as build_mod  # noqa: E402

build_mod.DATE_END = "2026-08-19 23:55:00"
build_mod.OUT_DIR = ROOT / "tmp/ilias_eth_154feature_dataset_extended_20260820"

if __name__ == "__main__":
    build_mod.main()
