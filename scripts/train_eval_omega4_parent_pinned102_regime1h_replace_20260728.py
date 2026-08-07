"""Research-only 2025 parent retrain replacing six 5m HMM inputs with 1h HMM values.

The 1h HMM is fit on 2024, so parent training is intentionally restricted to
2025. Reusing 2024 for the parent would expose it to in-sample HMM features.
"""

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
REGIME_DIR = ROOT / "tmp/causal_regen_20260516/regime3_1h_as_5m_contract_20260728"
omega.REGIME3_CURRENT_2025 = REGIME_DIR / "training_features_2024_2025_regime3_current_sensitive_hmm_1h_masked_wide24.csv"
omega.REGIME3_CURRENT_2026 = REGIME_DIR / "training_features_2026_regime3_current_sensitive_hmm_1h_masked_wide24.csv"


def _load_frames_regime1h() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = omega._read(omega.TRAIN_CSV)
    eval_df = omega._read(omega.EVAL_CSV)

    current_src = omega._read(omega.REGIME3_CURRENT_2025)
    current_ts = set(current_src["timestamp"])
    current_missing = ~train["timestamp"].isin(current_ts)

    cmamba_src = omega._read(omega.REGIME3_CMAMBA_2025)
    cmamba_bad = set(cmamba_src.loc[cmamba_src[omega.REGIME3_CMAMBA_COLS].isna().any(axis=1), "timestamp"])
    drop_mask = current_missing | train["timestamp"].isin(cmamba_bad)
    dropped_current = int(current_missing.sum())
    dropped_cmamba = int(train["timestamp"].isin(cmamba_bad).sum())
    if drop_mask.any():
        train = train.loc[~drop_mask].reset_index(drop=True)
    print(
        f"[regime1h_replace] explicit train-row exclusions: current1h_unavailable={dropped_current}, "
        f"cmamba_warmup={dropped_cmamba}, union={int(drop_mask.sum())}",
        flush=True,
    )

    train, train_current = omega._overlay_required(
        train, omega.REGIME3_CURRENT_2025, omega.REGIME3_CURRENT_COLS, tag="train_regime3_current_1h"
    )
    eval_df, eval_current = omega._overlay_required(
        eval_df, omega.REGIME3_CURRENT_2026, omega.REGIME3_CURRENT_COLS, tag="eval_regime3_current_1h"
    )
    train, train_cmamba = omega._overlay_required(
        train, omega.REGIME3_CMAMBA_2025, omega.REGIME3_CMAMBA_COLS, tag="train_regime3_cmamba"
    )
    eval_df, eval_cmamba = omega._overlay_required(
        eval_df, omega.REGIME3_CMAMBA_2026, omega.REGIME3_CMAMBA_COLS, tag="eval_regime3_cmamba"
    )
    train, train_risk = omega._overlay_required(
        train, omega.REGIME3_RISK_2025, omega.REGIME3_RISK_COLS, tag="train_regime3_risk"
    )
    eval_df, eval_risk = omega._overlay_required(
        eval_df, omega.REGIME3_RISK_2026, omega.REGIME3_RISK_COLS, tag="eval_regime3_risk"
    )
    train = pinned._repair_train_columns(train)
    return train, eval_df, {
        "train_current": train_current,
        "eval_current": eval_current,
        "train_cmamba": train_cmamba,
        "eval_cmamba": eval_cmamba,
        "train_risk": train_risk,
        "eval_risk": eval_risk,
        "dropped_current1h_unavailable_rows": dropped_current,
        "dropped_cmamba_warmup_rows": dropped_cmamba,
        "dropped_union_rows": int(drop_mask.sum()),
        "regime_contract": "six live 5m HMM columns replaced by causal completed-hour HMM values",
    }


def _append_default(flag: str, *values: str) -> None:
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

    _append_default("--epochs", "2")
    _append_default("--max-train-rows", "0")
    _append_default("--max-exit-samples", "30000")
    _append_default(
        "--direction-label-dir",
        "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
    )
    _append_default("--exit-label-mode", "entry_label_terminal_giveback")
    if component == "h48qual":
        _append_default("--quality-thresholds", "0.50")
        _append_default("--quality-mode", "quality_label_action")
        _append_default(
            "--quality-label-dir",
            "tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps",
        )
        _append_default("--out-suffix", "pinned102_regime1h_replace_2025only_20260728_h48qual")
    else:
        _append_default("--quality-thresholds", "0.75")
        _append_default("--quality-mode", "same_as_direction")
        _append_default("--out-suffix", "pinned102_regime1h_replace_2025only_20260728_zig075")

    pinned._install_pin(component)
    omega._load_omega_frames = _load_frames_regime1h
    return parent_script.main()


if __name__ == "__main__":
    raise SystemExit(main())
