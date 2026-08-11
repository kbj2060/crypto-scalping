"""Regime/feature 효과 분리를 위한 대조군 (HMM 레짐 + FINAL15 curated 피쳐).

scripts/train_eval_omega4_3head_parent72_eth_zig075_regime_jmredesign_final15_20260811.py의 정확한
fork -- REGIME3_CURRENT_2025/2026을 JM(jmredesign) 소스 대신 오버라이드하지 않아 parent_script의
기본값(HMM 기반 regime3_current_sensitive_wide24)을 쓴다. 그 외(cmamba/risk 오버레이 skip, 5개
피쳐 리서치 패널 브릿지, FINAL15 15개 피쳐 고정 allowlist)는 final15와 전부 동일.

2x2 설계의 (HMM, 15feat) 셀:
  (HMM, 172feat) = hmm_172feat_ablation  (JM,  172feat) = jmlam4_20260809
  (HMM, 15feat)  = 이 스크립트           (JM,  15feat)  = jmredesign_final15_20260811
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

omega = parent_script.omega
# REGIME3_CURRENT_2025/2026 오버라이드 없음(final15와의 유일한 실질 차이) -- 기본값(HMM
# sensitive_balancedish)을 그대로 사용.

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
        ("--out-suffix", "zig075_hmm_final15_ablation_20260812"),
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
