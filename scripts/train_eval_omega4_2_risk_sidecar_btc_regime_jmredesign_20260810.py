"""BTC risk sidecar retrain against the redesigned-JM parent, at the live h48qual_q055 contract.

Direct fork of scripts/train_eval_omega4_2_risk_sidecar_btc_regime_docs42_20260721.py -- the
previous BTC regime-swap sidecar -- with only the regime overlay CSVs, the parent bundle/prediction
directory, and the out-suffix changed. Every selection flag (log_risk objective, validation-only
scope, side-split model, dynamic leverage with a required mapping, live exposure grid) is left at
the values that precedent used, so the sidecar stage is not a second uncontrolled variable on top
of the regime swap.

Thresholds match the live BTC contract read from CURRENT_LIVE_MANIFEST.json: quality 0.55,
exit 0.95, prediction tag q055.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_2_risk_sidecar_btc_20260708 as sidecar_script  # noqa: E402

TAG = "jmredesign_20260810"
SUP = ROOT / "data/ensemble/supervised"

sidecar_script.omega.REGIME3_CURRENT_2025 = SUP / f"btc_regime3_current_hmm_{TAG}_2025_maskedname.csv"
sidecar_script.omega.REGIME3_CURRENT_2026 = SUP / f"btc_regime3_current_hmm_{TAG}_2026_maskedname.csv"

if __name__ == "__main__":
    _parent_dir = (ROOT / "tmp/causal_regen_20260516"
                   / f"btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_regime_{TAG}")
    defaults = [
        ("--baseline-bundle", str(_parent_dir / "true_3head_tabm_bundle.pt")),
        ("--precomputed-prediction-dir", str(_parent_dir)),
        ("--precomputed-prediction-tag", "q055"),
        ("--quality-threshold", "0.55"),
        ("--exit-threshold", "0.95"),
        ("--risk-feature-mode", "parent_outputs"),
        ("--selection-objective", "log_risk"),
        ("--selection-scope", "validation_only"),
        ("--out-suffix", f"regime_{TAG}_q055"),
        ("--device", "cpu"),
    ]
    for flag, value in defaults:
        if flag not in sys.argv:
            sys.argv += [flag, value]
    for flag in ("--side-split-model", "--dynamic-leverage",
                 "--require-dynamic-leverage-mapping", "--live-exposure-grid"):
        if flag not in sys.argv:
            sys.argv += [flag]
    raise SystemExit(sidecar_script.main())
