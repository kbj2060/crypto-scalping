"""Research-only zig075 risk-sidecar retrain for the 2025-only 1h-HMM replacement parent."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import train_eval_omega4_parent_pinned102_regime1h_replace_20260728 as parent_replace  # noqa: E402


REGIME_DIR = ROOT / "tmp/causal_regen_20260516/regime3_1h_as_5m_contract_20260728"
sidecar.omega.REGIME3_CURRENT_2025 = REGIME_DIR / "training_features_2024_2025_regime3_current_sensitive_hmm_1h_masked_wide24.csv"
sidecar.omega.REGIME3_CURRENT_2026 = REGIME_DIR / "training_features_2026_regime3_current_sensitive_hmm_1h_masked_wide24.csv"
sidecar.omega._load_omega_frames = parent_replace._load_frames_regime1h
PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime1h_replace_2025only_20260728_zig075"


def _default(flag: str, *values: str) -> None:
    if flag not in sys.argv:
        sys.argv += [flag, *values]


def main() -> int:
    _default(
        "--train-csv",
        "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv",
    )
    _default(
        "--eval-csv",
        "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv",
    )
    _default(
        "--direction-label-dir",
        "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
    )
    _default("--baseline-bundle", str(PARENT_DIR / "true_3head_tabm_bundle.pt"))
    _default("--precomputed-prediction-dir", str(PARENT_DIR))
    _default("--precomputed-prediction-tag", "q075")
    _default("--risk-feature-mode", "parent_outputs")
    if "--side-split-model" not in sys.argv:
        sys.argv += ["--side-split-model"]
    if "--dynamic-leverage" not in sys.argv:
        sys.argv += ["--dynamic-leverage"]
    if "--require-dynamic-leverage-mapping" not in sys.argv:
        sys.argv += ["--require-dynamic-leverage-mapping"]
    if "--live-exposure-grid" not in sys.argv:
        sys.argv += ["--live-exposure-grid"]
    _default("--min-validation-avg-notional", "0.45")
    _default("--max-validation-avg-notional", "0.95")
    _default("--selection-objective", "log_risk")
    _default("--selection-scope", "validation_only")
    _default("--log-tail-penalty", "0.5")
    _default("--out-suffix", "regime1h_replace_2025only_zig075_q075_20260728")
    _default("--device", "cpu")
    return sidecar.main()


if __name__ == "__main__":
    raise SystemExit(main())
