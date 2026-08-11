"""ETH zig075 parent retrain, jmredesign regime3-current, restricted to a 15-feature curated set
(Step B univariate AUC screen + knockoff/mRMR cross-method consensus, see
docs/experiments/eth_knockoff_feature_comparison_h48qual_vs_zig075_20260811.md and this session's
tmp/eth_zig075_oracle_label_check_20260811/ artifacts for the full derivation trail).

Direct fork of scripts/train_eval_omega4_3head_parent72_loose_entry_quality_eth_zig075_regime_
jmlam4_20260809.py (the jmlam4 zig075 parent this supersedes), with three changes on top of the
usual regime3-current CSV swap:

  1. `omega._load_omega_frames` is replaced with a version that skips the cmamba/risk overlays
     entirely. Their 2025 (TRAIN-year) source CSVs are missing from the working tree (gitignore
     cleanup fallout -- see the omega-cmamba-risk-overlay-dead-code memory), and the live zig075
     bundle's base_cols contain zero cmamba/risk columns anyway, so nothing of value is lost.
  2. The same override adds a fourth overlay bridging in 5 of the 15 curated features
     (btc_lead_eth_follow_gap_3, btc_volume_impulse_z, btc_ret_3, vwap_dist_24, funding_roc_48)
     from data/splits/year_oos/eth_features_2024_2026_analysis.csv -- the research panel used for
     this session's whole feature-selection exercise -- because they don't exist in the production
     TRAIN_CSV/EVAL_CSV (alpha6_current lineage, 2026-05-29) at all, under any name.
  3. `omega._numeric_feature_cols` is replaced with a fixed 15-name allowlist instead of the
     automatic "every numeric column" selection, per this session's feature-curation decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

omega = parent_script.omega

TAG = "jmredesign_20260810"
SUP = ROOT / "data/ensemble/supervised"
omega.REGIME3_CURRENT_2025 = SUP / f"eth_regime3_current_hmm_{TAG}_2025_maskedname.csv"
omega.REGIME3_CURRENT_2026 = SUP / f"eth_regime3_current_hmm_{TAG}_2026_maskedname.csv"

BRIDGE_PANEL = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"
BRIDGE_COLS = [
    "btc_lead_eth_follow_gap_3", "btc_volume_impulse_z", "btc_ret_3",
    "vwap_dist_24", "funding_roc_48",
]

FINAL15 = [
    "regime3_current_sensitive_wide24_chop_prob", "rsi", "ofti", "btc_lead_eth_follow_gap_3",
    "btc_volume_impulse_z", "log_return", "btc_ret_3", "smart_money_flow", "cvp_poc_dist",
    "cvp_regime", "funding_roc_288", "ou_halflife", "vwap_dist_24", "funding_roc_48",
    "breakout_strength",
]


def _load_omega_frames_no_cmamba_risk():
    train = omega._read(omega.TRAIN_CSV)
    eval_df = omega._read(omega.EVAL_CSV)
    train, train_current = omega._overlay_required(
        train, omega.REGIME3_CURRENT_2025, omega.REGIME3_CURRENT_COLS, tag="train_regime3_current")
    eval_df, eval_current = omega._overlay_required(
        eval_df, omega.REGIME3_CURRENT_2026, omega.REGIME3_CURRENT_COLS, tag="eval_regime3_current")
    # TRAIN_CSV (2025, 201 cols) and EVAL_CSV (2026, 220 cols) are different feature-engineering
    # generations -- funding_roc_48 already exists natively in EVAL_CSV but not TRAIN_CSV. Drop any
    # bridge column that already exists natively before overlaying, so the research-panel version is
    # authoritative for ALL 5 bridge columns uniformly across train/eval (never mix two different
    # generations' computation of the "same"-named column across the split).
    train = train.drop(columns=[c for c in BRIDGE_COLS if c in train.columns])
    eval_df = eval_df.drop(columns=[c for c in BRIDGE_COLS if c in eval_df.columns])
    train, train_bridge = omega._overlay_required(
        train, BRIDGE_PANEL, BRIDGE_COLS, tag="train_feature_bridge")
    eval_df, eval_bridge = omega._overlay_required(
        eval_df, BRIDGE_PANEL, BRIDGE_COLS, tag="eval_feature_bridge")
    return train, eval_df, {
        "train_current": train_current, "eval_current": eval_current,
        "train_bridge": train_bridge, "eval_bridge": eval_bridge,
        "cmamba_risk_overlay": "skipped -- see omega-cmamba-risk-overlay-dead-code memory",
    }


def _numeric_feature_cols_final15(train: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    missing_train = [c for c in FINAL15 if c not in train.columns]
    missing_eval = [c for c in FINAL15 if c not in eval_df.columns]
    if missing_train or missing_eval:
        raise RuntimeError(f"final15 feature(s) missing -- train:{missing_train} eval:{missing_eval}")
    bad_dtype = [c for c in FINAL15 if not pd.api.types.is_numeric_dtype(train[c])
                 or not pd.api.types.is_numeric_dtype(eval_df[c])]
    if bad_dtype:
        raise RuntimeError(f"final15 feature(s) non-numeric: {bad_dtype}")
    return list(FINAL15)


omega._load_omega_frames = _load_omega_frames_no_cmamba_risk
omega._numeric_feature_cols = _numeric_feature_cols_final15

if __name__ == "__main__":
    defaults = [
        ("--out-suffix", "zig075_regime_jmredesign_20260810_final15"),
        ("--direction-label-dir",
         "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/"
         "label_contracts/zigzag_action_labels_20260531"),
        ("--quality-mode", "same_as_direction"),
        ("--exit-label-mode", "entry_label_terminal_giveback"),
    ]
    for flag, value in defaults:
        if flag not in sys.argv:
            sys.argv += [flag, value]
    raise SystemExit(parent_script.main())
