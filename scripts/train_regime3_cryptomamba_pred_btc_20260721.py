"""BTC port of train_regime3_cryptomamba_pred_20260531.py (ETH's CryptoMamba h6 future-regime
prediction model) -- net-new for BTC. Points at BTC's own raw features and BTC's LIVE current-HMM
sidecar (btc_regime3_current_hmm_sensitive_wide24_20260708). Standalone research artifact only,
scoped to model-own accuracy/AUC, not wired into any trading decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_regime3_cryptomamba_pred_20260531 as base  # noqa: E402

base.MODEL_ID = "regime3_cryptomamba_pred_btc_h6_nocurrent_20260721"
base.DEFAULT_TRAIN_2024 = ROOT / "data/splits/year_oos/btc_features_2024.csv"
base.DEFAULT_TRANSFORMS = (
    ROOT / "data/splits/year_oos/btc_features_2024.csv",
    ROOT / "data/splits/year_oos/btc_features_2025.csv",
    ROOT / "data/splits/year_oos/btc_features_2026.csv",
)
base.DEFAULT_CURRENT_DIR = ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_20260708"
base.CURRENT_PREFIX = "regime3_current_sensitive_wide24_"
base.CURRENT_SIDECAR_STEM = "regime3_current_sensitive_hmm_wide24"
base.DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_btc_h6_nocurrent_20260721"
base.DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime3_cryptomamba_pred_btc_h6_nocurrent_20260721_report.json"

if __name__ == "__main__":
    raise SystemExit(base.main())
