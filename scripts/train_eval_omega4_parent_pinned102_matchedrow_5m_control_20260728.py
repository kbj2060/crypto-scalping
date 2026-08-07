"""Matched-row 5m-HMM control for the 1h-HMM replacement experiment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_pinned102_20260727 as pinned  # noqa: E402


omega = pinned.omega
parent_script = pinned.parent_script
FIVE_MIN_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
ONE_HOUR_AVAILABILITY = ROOT / "tmp/causal_regen_20260516/regime3_1h_as_5m_contract_20260728/training_features_2024_2025_regime3_current_sensitive_hmm_1h_masked_wide24.csv"
omega.REGIME3_CURRENT_2025 = FIVE_MIN_DIR / "training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
omega.REGIME3_CURRENT_2026 = FIVE_MIN_DIR / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"


def _load_frames_matched() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = omega._read(omega.TRAIN_CSV)
    eval_df = omega._read(omega.EVAL_CSV)
    available_1h = set(pd.read_csv(ONE_HOUR_AVAILABILITY, usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"])
    unavailable_1h = ~train["timestamp"].isin(available_1h)
    cmamba_src = omega._read(omega.REGIME3_CMAMBA_2025)
    cmamba_bad = set(cmamba_src.loc[cmamba_src[omega.REGIME3_CMAMBA_COLS].isna().any(axis=1), "timestamp"])
    drop_mask = unavailable_1h | train["timestamp"].isin(cmamba_bad)
    train = train.loc[~drop_mask].reset_index(drop=True)
    print(
        f"[matchedrow_5m] exclusions matched to 1h candidate: unavailable1h={int(unavailable_1h.sum())}, "
        f"cmamba={int(train['timestamp'].isin(cmamba_bad).sum())}, union={int(drop_mask.sum())}",
        flush=True,
    )
    train, train_current = omega._overlay_required(train, omega.REGIME3_CURRENT_2025, omega.REGIME3_CURRENT_COLS, tag="train_regime3_current_5m")
    eval_df, eval_current = omega._overlay_required(eval_df, omega.REGIME3_CURRENT_2026, omega.REGIME3_CURRENT_COLS, tag="eval_regime3_current_5m")
    train, train_cmamba = omega._overlay_required(train, omega.REGIME3_CMAMBA_2025, omega.REGIME3_CMAMBA_COLS, tag="train_regime3_cmamba")
    eval_df, eval_cmamba = omega._overlay_required(eval_df, omega.REGIME3_CMAMBA_2026, omega.REGIME3_CMAMBA_COLS, tag="eval_regime3_cmamba")
    train, train_risk = omega._overlay_required(train, omega.REGIME3_RISK_2025, omega.REGIME3_RISK_COLS, tag="train_regime3_risk")
    eval_df, eval_risk = omega._overlay_required(eval_df, omega.REGIME3_RISK_2026, omega.REGIME3_RISK_COLS, tag="eval_regime3_risk")
    train = pinned._repair_train_columns(train)
    return train, eval_df, {
        "train_current": train_current, "eval_current": eval_current,
        "train_cmamba": train_cmamba, "eval_cmamba": eval_cmamba,
        "train_risk": train_risk, "eval_risk": eval_risk,
        "matched_row_contract": True,
        "dropped_union_rows": int(drop_mask.sum()),
    }


def _default(flag: str, *values: str) -> None:
    if flag not in sys.argv:
        sys.argv += [flag, *values]


def main() -> int:
    if "--pin-component" not in sys.argv:
        raise SystemExit("--pin-component {h48qual,zig075} is required")
    index = sys.argv.index("--pin-component")
    component = sys.argv[index + 1]
    if component not in pinned.LIVE_BUNDLES:
        raise SystemExit(f"--pin-component must be one of {sorted(pinned.LIVE_BUNDLES)}")
    del sys.argv[index : index + 2]
    _default("--epochs", "2")
    _default("--max-train-rows", "0")
    _default("--max-exit-samples", "30000")
    _default("--direction-label-dir", "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531")
    _default("--exit-label-mode", "entry_label_terminal_giveback")
    if component == "h48qual":
        _default("--quality-thresholds", "0.50")
        _default("--quality-mode", "quality_label_action")
        _default("--quality-label-dir", "tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps")
        _default("--out-suffix", "pinned102_matchedrow_5m_control_20260728_h48qual")
    else:
        _default("--quality-thresholds", "0.75")
        _default("--quality-mode", "same_as_direction")
        _default("--out-suffix", "pinned102_matchedrow_5m_control_20260728_zig075")
    pinned._install_pin(component)
    omega._load_omega_frames = _load_frames_matched
    return parent_script.main()


if __name__ == "__main__":
    raise SystemExit(main())
