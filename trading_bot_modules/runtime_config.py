from __future__ import annotations

import os

from trading_bot_modules.omega4_6_2_source_parent_live import OMEGA462_SOURCE_PARENT_MODEL_ID
from trading_bot_modules.omega5_live import OMEGA5_MODEL_ID

try:
    from scripts.eval_sniper_day_ensemble_oos_2026 import DEFAULT_SNIPER_CKPT as FINAL_GOVERNOR_SNIPER_CKPT
except ImportError:
    FINAL_GOVERNOR_SNIPER_CKPT = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/regime_rl_meta_controller_v3/best.pth"

try:
    from scripts.train_event_masked_rl_meta_controller import (
        DEFAULT_MANIFEST as FINAL_GOVERNOR_MANIFEST,
        DEFAULT_POLICY as FINAL_GOVERNOR_POLICY,
    )
except ImportError:
    FINAL_GOVERNOR_MANIFEST = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/regime_dsac_event_masked_v9_regime_v3_full5/manifest.json"
    FINAL_GOVERNOR_POLICY = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/regime_dsac_event_masked_v9_regime_v3_full5/router_policy.json"


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


DASHBOARD_STATE_PATH = os.getenv("DASHBOARD_STATE_PATH", "data/live/dashboard_state.json")
COMPACT_DASHBOARD_STATE_PATH = os.getenv("COMPACT_DASHBOARD_STATE_PATH", "data/live/dashboard_state_governor.json")
DASHBOARD_EVENTS_PATH = os.getenv("DASHBOARD_EVENTS_PATH", "data/live/dashboard_events.jsonl")
TRADE_JOURNAL_PATH = os.getenv("TRADE_JOURNAL_PATH", "data/live/trade_journal.jsonl")
POSITION_ACCOUNTING_AUDIT_PATH = os.getenv("POSITION_ACCOUNTING_AUDIT_PATH", "data/live/position_accounting_audit.jsonl")
DAILY_TRADE_REPORT_STATE_PATH = os.getenv(
    "DAILY_TRADE_REPORT_STATE_PATH",
    "data/live/daily_trade_report_state.json",
)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
ENSEMBLE_BALANCED_METRICS_PATH = os.getenv(
    "ENSEMBLE_BALANCED_METRICS_PATH",
    "data/ensemble/metrics/param_ensemble_result.json",
)
ENSEMBLE_LOWFREQ_METRICS_PATH = os.getenv(
    "ENSEMBLE_LOWFREQ_METRICS_PATH",
    "data/ensemble/metrics/param_ensemble_lowfreq_grid.json",
)
ENSEMBLE_BALANCED_PARAMS_PATH = os.getenv(
    "ENSEMBLE_BALANCED_PARAMS_PATH",
    "data/ensemble/metrics/param_ensemble_result.json",
)
ENSEMBLE_LOWFREQ_PARAMS_PATH = os.getenv(
    "ENSEMBLE_LOWFREQ_PARAMS_PATH",
    "data/ensemble/metrics/param_ensemble_lowfreq_highpnl.json",
)
ENSEMBLE_TRACKER_STATE_PATH = os.getenv(
    "ENSEMBLE_TRACKER_STATE_PATH",
    "data/live/ensemble_tracker_state.json",
)
ENSEMBLE_TRACKER_RECORDS_PATH = os.getenv(
    "ENSEMBLE_TRACKER_RECORDS_PATH",
    "data/live/ensemble_trade_records.jsonl",
)
ENSEMBLE_TRACKER_FEE_RATE = float(os.getenv("ENSEMBLE_TRACKER_FEE_RATE", "0.0005"))
ENSEMBLE_TRACKER_SLIP_RATE = float(os.getenv("ENSEMBLE_TRACKER_SLIP_RATE", "0.0002"))
ENSEMBLE_TRACKER_ENABLED = _env_flag("ENSEMBLE_TRACKER_ENABLED", False)
ENSEMBLE_TRACKER_EXIT_ON_HOLD = _env_flag("ENSEMBLE_TRACKER_EXIT_ON_HOLD", False)
ENSEMBLE_OVERHEAT_Z_WIN = int(float(os.getenv("ENSEMBLE_OVERHEAT_Z_WIN", "120")))
ENSEMBLE_OVERHEAT_Z_MIN = int(float(os.getenv("ENSEMBLE_OVERHEAT_Z_MIN", "20")))
QUANT_MICRO_DB_PATH = os.getenv("QUANT_MICRO_DB_PATH", "data/live/microstructure.duckdb")
QUANT_TAIL_DB_PATH = os.getenv("QUANT_TAIL_DB_PATH", "data/live/tail_risk.duckdb")
QUANT_BAR_MINUTES = int(float(os.getenv("QUANT_BAR_MINUTES", "1")))
QUANT_LOOKBACK_MINUTES = int(float(os.getenv("QUANT_LOOKBACK_MINUTES", "15")))
QUANT_HORIZON_MINUTES = 30
QUANT_TOP_K_FEATURES = int(float(os.getenv("QUANT_TOP_K_FEATURES", "25")))
QUANT_MAX_HISTORY_ROWS = int(float(os.getenv("QUANT_MAX_HISTORY_ROWS", "3000")))
QUANT_LOGIC_PATH = os.getenv("QUANT_LOGIC_PATH", "quant/live_30m_direction_quant.py")
FINAL_GOVERNOR_DDH2_ENSEMBLE_ENABLE = False
FINAL_GOVERNOR_DDH2_REPORT_PATH = ""
FINAL_GOVERNOR_DDH2_AUDIT_PATH = ""
FINAL_GOVERNOR_MICRO_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_MICRO_MODEL_PATH",
    "",
)
FINAL_GOVERNOR_TREND_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_TREND_MODEL_PATH",
    "",
)
FINAL_GOVERNOR_EVENT_DETECTOR_PATH = os.getenv(
    "FINAL_GOVERNOR_EVENT_DETECTOR_PATH",
    "data/ensemble/event_driven/trade_candidate_detector_catboost_oof.pkl",
)
FINAL_GOVERNOR_SNIPER_MODEL_PATH = os.getenv("FINAL_GOVERNOR_SNIPER_MODEL_PATH", FINAL_GOVERNOR_SNIPER_CKPT)
FINAL_GOVERNOR_MANIFEST_PATH = os.getenv("FINAL_GOVERNOR_MANIFEST_PATH", FINAL_GOVERNOR_MANIFEST)
FINAL_GOVERNOR_POLICY_PATH = os.getenv("FINAL_GOVERNOR_POLICY_PATH", FINAL_GOVERNOR_POLICY)
FINAL_GOVERNOR_DISABLED_V13_1_ENABLE = False
FINAL_GOVERNOR_DISABLED_V13_1_REQUIRED = False
FINAL_GOVERNOR_DISABLED_V13_1_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_DISABLED_V13_1_MODEL_PATH",
    "",
)
FINAL_GOVERNOR_DISABLED_V13_1_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_DISABLED_V13_1_REPORT_PATH",
    "",
)
FINAL_GOVERNOR_V21_2_JACKPOT_ENABLE = _env_flag("FINAL_GOVERNOR_V21_2_JACKPOT_ENABLE", True)
FINAL_GOVERNOR_V21_2_JACKPOT_REQUIRED = _env_flag(
    "FINAL_GOVERNOR_V21_2_JACKPOT_REQUIRED",
    FINAL_GOVERNOR_V21_2_JACKPOT_ENABLE,
)
FINAL_GOVERNOR_V21_2_JACKPOT_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_V21_2_JACKPOT_MODEL_PATH",
    "data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl",
)
FINAL_GOVERNOR_V21_2_JACKPOT_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_V21_2_JACKPOT_REPORT_PATH",
    "data/ensemble/reports/hf_v13_jackpot_runner_v21_2_20260511_summary.json",
)
FINAL_GOVERNOR_V21_2_JACKPOT_AUDIT_PATH = os.getenv(
    "FINAL_GOVERNOR_V21_2_JACKPOT_AUDIT_PATH",
    "data/ensemble/reports/hf_v13_jackpot_runner_v21_2_20260511_audit.json",
)
FINAL_GOVERNOR_V21_2_ONLY = True
FINAL_GOVERNOR_ALPHA3_CANONICAL_DECISION_ENABLE = _env_flag(
    "FINAL_GOVERNOR_ALPHA3_CANONICAL_DECISION_ENABLE",
    True,
)
FINAL_GOVERNOR_ALPHA3_BACKTEST_COOLDOWN_PARITY_ENABLE = _env_flag(
    "FINAL_GOVERNOR_ALPHA3_BACKTEST_COOLDOWN_PARITY_ENABLE",
    False,
)
FINAL_GOVERNOR_ALPHA3_CSV_COOLDOWN_PARITY_ENABLE = _env_flag(
    "FINAL_GOVERNOR_ALPHA3_CSV_COOLDOWN_PARITY_ENABLE",
    True if FINAL_GOVERNOR_ALPHA3_CANONICAL_DECISION_ENABLE else FINAL_GOVERNOR_ALPHA3_BACKTEST_COOLDOWN_PARITY_ENABLE,
)
FINAL_GOVERNOR_ALPHA3_BACKTEST_MARK_PARITY_ENABLE = _env_flag(
    "FINAL_GOVERNOR_ALPHA3_BACKTEST_MARK_PARITY_ENABLE",
    False,
)
FINAL_GOVERNOR_ALPHA3_CSV_MARK_PARITY_ENABLE = _env_flag(
    "FINAL_GOVERNOR_ALPHA3_CSV_MARK_PARITY_ENABLE",
    True if FINAL_GOVERNOR_ALPHA3_CANONICAL_DECISION_ENABLE else FINAL_GOVERNOR_ALPHA3_BACKTEST_MARK_PARITY_ENABLE,
)
FINAL_GOVERNOR_ALPHA3_MODEL_ID = "alpha3_corrected_next_open_limit_touch0_fee20"
FINAL_GOVERNOR_ALPHA3_LIVE_CONTRACT_ID = "alpha3_csv_native_backtest_parity_20260516_live"
FINAL_GOVERNOR_ALPHA7_MODEL_ID = os.getenv(
    "FINAL_GOVERNOR_ALPHA7_MODEL_ID",
    "alpha7_submodel_01965_decontam_deep_stop_cd18_20260528",
)
FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID = FINAL_GOVERNOR_ALPHA7_MODEL_ID
FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION = os.getenv(
    "FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION",
    "Alpha7.1-01965-decontam-deep-stop-cd18",
)
FINAL_GOVERNOR_V31_ENABLE = _env_flag(
    "FINAL_GOVERNOR_V31_ENABLE",
    FINAL_GOVERNOR_ALPHA3_CANONICAL_DECISION_ENABLE,
)
FINAL_GOVERNOR_V31_REQUIRED = _env_flag("FINAL_GOVERNOR_V31_REQUIRED", FINAL_GOVERNOR_V31_ENABLE)
FINAL_GOVERNOR_V31_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_V31_REPORT_PATH",
    "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/v31_state24_v2_plus_pred_runtime_report.json",
)
FINAL_GOVERNOR_V31_AUDIT_PATH = os.getenv(
    "FINAL_GOVERNOR_V31_AUDIT_PATH",
    "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/v31_state24_v2_plus_pred_runtime_audit.json",
)
FINAL_GOVERNOR_V31_V27_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_V31_V27_MODEL_PATH",
    "data/ensemble/supervised/alpha3_regime4_state24_v2_plus_pred_full_retrain_20260526/deep_scout_state24_v2.pt",
)
FINAL_GOVERNOR_V31_DEEP_NOTIONAL = float(
    os.getenv(
        "FINAL_GOVERNOR_V31_DEEP_NOTIONAL",
        "2.0" if FINAL_GOVERNOR_ALPHA3_CANONICAL_DECISION_ENABLE else "0.0",
    )
)
FINAL_GOVERNOR_V31_TRAIL_ACTIVATION = float(
    os.getenv(
        "FINAL_GOVERNOR_V31_TRAIL_ACTIVATION",
        "0.0" if FINAL_GOVERNOR_ALPHA3_CANONICAL_DECISION_ENABLE else "0.008",
    )
)
FINAL_GOVERNOR_V31_TRAIL_MIN_SL_MULT = float(os.getenv("FINAL_GOVERNOR_V31_TRAIL_MIN_SL_MULT", "0.60"))
FINAL_GOVERNOR_ALPHA2_1_ENABLE = _env_flag("FINAL_GOVERNOR_ALPHA2_1_ENABLE", True)
FINAL_GOVERNOR_ALPHA2_1_MODEL_ID = "alpha2_1_teacher_l2_runtime_sweep_20260514"
FINAL_GOVERNOR_ALPHA2_1_TEACHER_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_ALPHA2_1_TEACHER_MODEL_PATH",
    "data/ensemble/supervised/alpha1_l2_teacher_deep_parent_20260514/teacher_deep_parent_l2_replay.pt",
)
FINAL_GOVERNOR_ALPHA2_1_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_ALPHA2_1_REPORT_PATH",
    "data/ensemble/reports/alpha2_teacher_l2_runtime_sweep_20260514_summary.json",
)
FINAL_GOVERNOR_ALPHA2_1_AUDIT_PATH = os.getenv(
    "FINAL_GOVERNOR_ALPHA2_1_AUDIT_PATH",
    "data/ensemble/reports/alpha2_teacher_l2_runtime_sweep_20260514_audit.json",
)
FINAL_GOVERNOR_ALPHA2_1_CONFIDENCE = 0.56
FINAL_GOVERNOR_ALPHA2_1_PARENT_NOTIONAL_SCALE = 1.10
FINAL_GOVERNOR_ALPHA2_1_MAX_NOTIONAL = 2.75
FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_ENABLE = False
FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_REQUIRED = False
FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_MODEL_PATH",
    "data/ensemble/supervised/clean_base_causal_sleeve_conformal_veto_v1_5/sleeve_conformal_veto.pkl",
)
FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_REPORT_PATH",
    "data/ensemble/reports/clean_base_causal_sleeve_conformal_veto_v1_5_2026.json",
)
FINAL_GOVERNOR_V1_5_BACKTEST_PARITY = _env_flag(
    "FINAL_GOVERNOR_V1_5_BACKTEST_PARITY",
    FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_ENABLE or FINAL_GOVERNOR_DISABLED_V13_1_ENABLE,
)
FINAL_GOVERNOR_NOTIONAL = float(os.getenv("FINAL_GOVERNOR_NOTIONAL", "5.0"))
FINAL_GOVERNOR_LEVERAGE = float(os.getenv("FINAL_GOVERNOR_LEVERAGE", "5.0"))
FINAL_GOVERNOR_DUST_ENTRY_EXPOSURE = float(os.getenv("FINAL_GOVERNOR_DUST_ENTRY_EXPOSURE", "0.22"))
FINAL_GOVERNOR_MIN_ENTRY_EXPOSURE = float(os.getenv("FINAL_GOVERNOR_MIN_ENTRY_EXPOSURE", "0.30"))
FINAL_GOVERNOR_V1_5_COST_FIREWALL_ENABLE = _env_flag(
    "FINAL_GOVERNOR_V1_5_COST_FIREWALL_ENABLE",
    FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_ENABLE,
)
FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_DISABLE = _env_flag(
    "FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_DISABLE",
    True,
)
FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_FEE = float(
    os.getenv("FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_FEE", "0.0015")
)
FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_SLIP = float(
    os.getenv("FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_SLIP", "0.0006")
)
FINAL_GOVERNOR_V1_5_COST_FIREWALL_STRESS_SLEEVE_MULT = float(
    os.getenv("FINAL_GOVERNOR_V1_5_COST_FIREWALL_STRESS_SLEEVE_MULT", "0.50")
)
FINAL_GOVERNOR_WINDOW_BARS = int(float(os.getenv("FINAL_GOVERNOR_WINDOW_BARS", "7000")))
FINAL_GOVERNOR_MACRO_ENABLE = False
FINAL_GOVERNOR_MACRO_LOOKBACK_BARS = int(float(os.getenv("FINAL_GOVERNOR_MACRO_LOOKBACK_BARS", "6048")))
FINAL_GOVERNOR_BUFFER_BARS = int(float(os.getenv(
    "FINAL_GOVERNOR_BUFFER_BARS",
    str(max(7000, FINAL_GOVERNOR_WINDOW_BARS, FINAL_GOVERNOR_MACRO_LOOKBACK_BARS + 512)),
)))
FINAL_GOVERNOR_MACRO_THRESHOLD = float(os.getenv("FINAL_GOVERNOR_MACRO_THRESHOLD", "0.05"))
FINAL_GOVERNOR_MACRO_PERSIST_UPDATES = int(float(os.getenv("FINAL_GOVERNOR_MACRO_PERSIST_UPDATES", "5")))
FINAL_GOVERNOR_MACRO_UPDATE_BARS = int(float(os.getenv("FINAL_GOVERNOR_MACRO_UPDATE_BARS", "288")))
FINAL_GOVERNOR_MACRO_NOTIONAL = float(os.getenv("FINAL_GOVERNOR_MACRO_NOTIONAL", "3.0"))
FINAL_GOVERNOR_MACRO_LEVERAGE = float(os.getenv("FINAL_GOVERNOR_MACRO_LEVERAGE", "5.0"))
FINAL_GOVERNOR_MACRO_BOOTSTRAP_CURRENT = _env_flag("FINAL_GOVERNOR_MACRO_BOOTSTRAP_CURRENT", True)
FINAL_GOVERNOR_MACRO_TAKE_PROFIT = float(os.getenv("FINAL_GOVERNOR_MACRO_TAKE_PROFIT", "1.25"))
FINAL_GOVERNOR_MACRO_STOP_LOSS = float(os.getenv("FINAL_GOVERNOR_MACRO_STOP_LOSS", "0.0"))
FINAL_GOVERNOR_MACRO_TRAILING_ARM = float(os.getenv("FINAL_GOVERNOR_MACRO_TRAILING_ARM", "0.0"))
FINAL_GOVERNOR_MACRO_TRAILING_GAP = float(os.getenv("FINAL_GOVERNOR_MACRO_TRAILING_GAP", "0.0"))
FINAL_GOVERNOR_MACRO_LOCKOUT_BARS = int(float(os.getenv("FINAL_GOVERNOR_MACRO_LOCKOUT_BARS", "24")))
FINAL_GOVERNOR_MACRO_LOCKOUT_UNTIL_SIGNAL_CHANGE = _env_flag("FINAL_GOVERNOR_MACRO_LOCKOUT_UNTIL_SIGNAL_CHANGE", False)
FINAL_GOVERNOR_MACRO_LOCKOUT_ON_ANY_CLOSE = _env_flag("FINAL_GOVERNOR_MACRO_LOCKOUT_ON_ANY_CLOSE", True)
FINAL_GOVERNOR_EXECUTION_POLICY_ENABLE = False
FINAL_GOVERNOR_EXECUTION_POLICY_PATH = os.getenv(
    "FINAL_GOVERNOR_EXECUTION_POLICY_PATH",
    "data/ensemble/supervised/learned_execution_policy_v3_tail.pkl",
)
FINAL_GOVERNOR_EXECUTION_POLICY_IGNORE_MAX_HOLD = _env_flag("FINAL_GOVERNOR_EXECUTION_POLICY_IGNORE_MAX_HOLD", True)
FINAL_GOVERNOR_EXECUTION_POLICY_QUALITY_OVERLAY = _env_flag("FINAL_GOVERNOR_EXECUTION_POLICY_QUALITY_OVERLAY", True)
FINAL_GOVERNOR_EXECUTION_POLICY_LOW_QUALITY = float(os.getenv("FINAL_GOVERNOR_EXECUTION_POLICY_LOW_QUALITY", "0.18"))
FINAL_GOVERNOR_EXECUTION_POLICY_TAIL_QUALITY = float(os.getenv("FINAL_GOVERNOR_EXECUTION_POLICY_TAIL_QUALITY", "0.28"))
OMEGA5_LIVE_PROMOTION_BLOCK_REASON = (
    "Omega5 live promotion blocked on 2026-07-02: side-thread audit found "
    "validation/test ledger dependence in the promoted model-selection path."
)
FINAL_GOVERNOR_OMEGA5_ENABLE = _env_flag("FINAL_GOVERNOR_OMEGA5_ENABLE", False)
if FINAL_GOVERNOR_OMEGA5_ENABLE:
    raise RuntimeError(OMEGA5_LIVE_PROMOTION_BLOCK_REASON)
FINAL_GOVERNOR_OMEGA5_MODEL_ID = os.getenv("FINAL_GOVERNOR_OMEGA5_MODEL_ID", OMEGA5_MODEL_ID)
if FINAL_GOVERNOR_OMEGA5_MODEL_ID != OMEGA5_MODEL_ID:
    raise RuntimeError(f"Omega5 model id contract mismatch: {FINAL_GOVERNOR_OMEGA5_MODEL_ID} != {OMEGA5_MODEL_ID}")
FINAL_GOVERNOR_OMEGA5_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA5_REPORT_PATH",
    "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/report.json",
)
FINAL_GOVERNOR_OMEGA5_FEATURE_VETO_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA5_FEATURE_VETO_REPORT_PATH",
    "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/report.json",
)
FINAL_GOVERNOR_OMEGA5_TWO_STAGE_VETO_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA5_TWO_STAGE_VETO_REPORT_PATH",
    "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/report.json",
)
FINAL_GOVERNOR_OMEGA5_PNL_TILT_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA5_PNL_TILT_REPORT_PATH",
    "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/report.json",
)
FINAL_GOVERNOR_OMEGA5_REDTEAM_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA5_REDTEAM_PATH",
    "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/redteam_audit_20260701.json",
)
FINAL_GOVERNOR_OMEGA5_FRONTIER_AUDIT_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA5_FRONTIER_AUDIT_PATH",
    "tmp/causal_regen_20260516/omega4_6_2_frontier_leakage_redteam_20260701/frontier_leakage_redteam_20260701.json",
)
FINAL_GOVERNOR_OMEGA5_CVP_AUDIT_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA5_CVP_AUDIT_PATH",
    "tmp/causal_regen_20260516/cvp_feature_causality_20260701/cvp_feature_causality_20260701.json",
)
FINAL_GOVERNOR_OMEGA5_ARTIFACT_INTEGRITY_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA5_ARTIFACT_INTEGRITY_PATH",
    "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/omega_artifact_integrity_audit_20260630.json",
)
FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE = _env_flag("FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE", False)
FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_MODEL_ID = os.getenv(
    "FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_MODEL_ID",
    OMEGA462_SOURCE_PARENT_MODEL_ID,
)
if FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_MODEL_ID != OMEGA462_SOURCE_PARENT_MODEL_ID:
    raise RuntimeError(
        "Omega5 source parent model id contract mismatch: "
        f"{FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_MODEL_ID} != {OMEGA462_SOURCE_PARENT_MODEL_ID}"
    )
FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_REPORT_PATH",
    "tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701/report.json",
)
FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_CAP220_CONTRACT_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_CAP220_CONTRACT_PATH",
    "tmp/causal_regen_20260516/omega4_6_2_cap220_short_boost125_time_stop120h_20260630/runtime_contract.json",
)
FINAL_GOVERNOR_OMEGA4_6_1_ENABLE = _env_flag("FINAL_GOVERNOR_OMEGA4_6_1_ENABLE", False)
# Both default to exactly today's behavior (gate on, no notional scaling). See
# docs/model_contracts/portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md -- backtest
# research found disabling this same duration gate and scaling ETH's own notional by 1.5x
# (holding margin_fraction fixed) materially improves ETH's OOS performance.
FINAL_GOVERNOR_OMEGA4_6_1_ETH_DURATION_GATE_OFF = _env_flag("FINAL_GOVERNOR_OMEGA4_6_1_ETH_DURATION_GATE_OFF", False)
FINAL_GOVERNOR_OMEGA4_6_1_ETH_NOTIONAL_MULTIPLIER = float(os.getenv("FINAL_GOVERNOR_OMEGA4_6_1_ETH_NOTIONAL_MULTIPLIER", "1.0"))
# 2026-07-20 session: real chop_prob-based notional sizing (supersedes and removes the
# 2026-07-19 shadow-only observer, which used a plain linear max(0, 1-chop_prob) shape).
# Threshold-gated shape instead (full size below the threshold, linear ramp to 0 only above it) --
# fresh-forward backtest (docs/model_contracts/eth_leverage_chop_softsize_fresh_forward_20260720.md)
# found the plain linear shape gives up too much PnL for its MDD benefit; threshold=0.3 recovers
# most of the PnL while keeping nearly all of the MDD reduction (VAL: dominates no-chop outright,
# PnL+MDD both better; OOS: PnL +68.78% vs no-chop's +77.11%, MDD -12.88% vs -15.48%). Default False.
FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_ENABLE = _env_flag("FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_ENABLE", False)
FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_THRESHOLD = float(os.getenv("FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_THRESHOLD", "0.3"))
# Default False: ETH's real decision path stays unaffected by the shared portfolio notional cap
# unless explicitly opted in. See _decide_omega4_6_1_entry and PortfolioRiskManager construction
# in main(). Folding ETH in changes live sizing the moment it's enabled, so it must not be implied
# by FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP alone (that cap already defaults to 3.0, not None).
FINAL_GOVERNOR_OMEGA4_6_1_ETH_PORTFOLIO_CAP_ENABLE = _env_flag("FINAL_GOVERNOR_OMEGA4_6_1_ETH_PORTFOLIO_CAP_ENABLE", False)
FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH",
    "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt",
)
FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_SIDECAR_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_SIDECAR_PATH",
    "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/risk_sidecar.pkl",
)
FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH",
    "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt",
)
FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_SIDECAR_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_SIDECAR_PATH",
    "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_zig075_q075_precomputed_20260630/risk_sidecar.pkl",
)
FINAL_GOVERNOR_OMEGA4_6_1_SHADOW_ASSETS_ENABLE = _env_flag("FINAL_GOVERNOR_OMEGA4_6_1_SHADOW_ASSETS_ENABLE", True)
FINAL_GOVERNOR_OMEGA4_6_1_SOL_ENABLE = _env_flag("FINAL_GOVERNOR_OMEGA4_6_1_SOL_ENABLE", True)
# v2 (2026-07-20): re-trained on adaptive_squeeze features (features/engineering.py's
# long_squeeze_risk/short_squeeze_risk/crowding_pressure no longer use ETH's fixed 0.0002
# funding-rate divisor for SOL -- self-normalizing rolling funding_z_score instead). Full-pipeline
# comparison at the matching stage (scale-map, gate off, no notional multiplier) vs the prior v1:
# VAL +16.75%/mdd-26.29% (v1 +33.73%/mdd-29.78%), OOS +57.94%/mdd-21.35% (v1 +33.98%/mdd-31.99%).
# v1 breached this project's own -25% OOS-MDD promotion-gate cap; v2 clears it. Artifact-integrity
# audit re-run and promotion_pass=true for v2
# (tmp/causal_regen_20260516/sol_adaptive_squeeze_artifact_integrity_20260720/).
FINAL_GOVERNOR_OMEGA4_6_1_SOL_BUNDLE_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA4_6_1_SOL_BUNDLE_PATH",
    "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720/true_3head_tabm_bundle.pt",
)
FINAL_GOVERNOR_OMEGA4_6_1_SOL_SIDECAR_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA4_6_1_SOL_SIDECAR_PATH",
    "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q070_20260720/risk_sidecar.pkl",
)
# 2026-08-07 BTC promotion: h48qual + swing_transition_prob (the 2026-08-06 candidate).
# Passed audit_omega_artifact_integrity_btc_20260712 (promotion_pass=true, same gate the previous
# live baseline was promoted under on 2026-07-13); worst-OOS-quarter -0.87% vs -7.11% for the
# previous bundle. The bundle requires the swing_transition_prob feature, computed live by
# trading_bot_modules/btc_swing_transition_live.py (auto-enabled only when the loaded bundle's
# base_cols require it). Rollback: set both env vars to the previous artifacts
#   tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_20260708/true_3head_tabm_bundle.pt
#   tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260708/risk_sidecar.pkl
# and restore duration_threshold 0.00541154875 below.
FINAL_GOVERNOR_OMEGA4_6_1_BTC_BUNDLE_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA4_6_1_BTC_BUNDLE_PATH",
    "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition/true_3head_tabm_bundle.pt",
)
FINAL_GOVERNOR_OMEGA4_6_1_BTC_SIDECAR_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA4_6_1_BTC_SIDECAR_PATH",
    "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260806_swingtransition/risk_sidecar.pkl",
)
# Each asset's own regime3-current HMM (must match the joblib used to build the
# regime3_current_sensitive_wide24_* sidecar columns the parent/risk-sidecar were TRAINED on --
# found 2026-07-20 that the shadow-asset loop was silently defaulting both of these to ETH's HMM
# via Omega461LiveAdapter's own default, a live-only train/inference mismatch never caught before
# since it doesn't affect any backtested VAL/OOS number (those are computed from the correct
# per-asset precomputed sidecar CSVs, not this live path).
FINAL_GOVERNOR_OMEGA4_6_1_SOL_REGIME3_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA4_6_1_SOL_REGIME3_PATH",
    "data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707/regime3_current_sensitive_hmm_wide24_2024.joblib",
)
FINAL_GOVERNOR_OMEGA4_6_1_BTC_REGIME3_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA4_6_1_BTC_REGIME3_PATH",
    "data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_20260708/regime3_current_sensitive_hmm_wide24_2024.joblib",
)
OMEGA4_6_1_SHADOW_ASSET_CONFIG = {
    "sol": {
        "label": "SOL",
        # kline/price data symbol -- USDT-quoted to match the backtested models'
        # price series; account_symbol (order/balance settlement) is also USDT-M.
        "symbol": "SOLUSDT",
        "account_symbol": "SOL/USDT:USDT",
        "component": "zig075",
        "quality_threshold": 0.70,
        "duration_threshold": 0.0055208323,
        "scale_map": {"zig075_L": 1.0, "zig075_S": 1.75},  # v2 20260720, re-derived for adaptive_squeeze bundle
        "bundle_path": FINAL_GOVERNOR_OMEGA4_6_1_SOL_BUNDLE_PATH,
        "sidecar_path": FINAL_GOVERNOR_OMEGA4_6_1_SOL_SIDECAR_PATH,
        "current_regime_path": FINAL_GOVERNOR_OMEGA4_6_1_SOL_REGIME3_PATH,
    },
    "btc": {
        "label": "BTC",
        "symbol": "BTCUSDT",
        "account_symbol": "BTC/USDT:USDT",
        "component": "h48qual",
        "quality_threshold": 0.55,
        # 2026-08-07: swingtransition candidate's own VAL-selected duration gate
        # (previous bundle's value was 0.00541154875)
        "duration_threshold": 0.0054143218,
        "scale_map": {"h48qual_L": 0.5, "h48qual_S": 2.5},
        "bundle_path": FINAL_GOVERNOR_OMEGA4_6_1_BTC_BUNDLE_PATH,
        "sidecar_path": FINAL_GOVERNOR_OMEGA4_6_1_BTC_SIDECAR_PATH,
        "current_regime_path": FINAL_GOVERNOR_OMEGA4_6_1_BTC_REGIME3_PATH,
    },
}
# SOL/BTC real execution: OFF by default, matching FINAL_GOVERNOR_OMEGA4_6_1_ENABLE /
# BINANCE_EXECUTION_ENABLED's existing opt-in convention. When False, the SOL/BTC shadow
# loop behaves exactly as before (decision-only, no orders). See
# docs/active_live/multi_asset_wiring_plan_20260712.md.
FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE = _env_flag(
    "FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE", False
)
# Default False = each asset's own existing duration_threshold value (unchanged). When True,
# both sol and btc use a disabled gate instead, matching the CURRENT_BASELINE backtest finding
# (see the comment above FINAL_GOVERNOR_OMEGA4_6_1_ETH_DURATION_GATE_OFF).
FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_DURATION_GATE_OFF = _env_flag(
    "FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_DURATION_GATE_OFF", False
)
# Default False: SOL/BTC don't get their own OrderBookRecorder unless explicitly enabled. Today
# only ETH's L2 book gets recorded into orderbook_decision_snapshots (data/live/microstructure.duckdb)
# -- SOL/BTC have zero order-book history. This flag starts accumulating it for both, into separate
# per-asset tables (orderbook_decision_snapshots_sol / _btc), reusing the same OrderBookRecorder
# class (already asset-parameterized via the fetcher argument -- no class changes needed).
FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_ORDERBOOK_RECORDER_ENABLE = _env_flag(
    "FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_ORDERBOOK_RECORDER_ENABLE", False
)
# Default False: starts two more MicrostructureScanner instances (BTC/SOL, public WS/REST only, no
# auth) writing to their own microstructure_1m_btc / microstructure_1m_sol tables. ETH's own
# microstructure_1m table/behavior is completely unaffected either way.
FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_MICROSTRUCTURE_SCANNER_ENABLE = _env_flag(
    "FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_MICROSTRUCTURE_SCANNER_ENABLE", False
)
# Default 1.0 reproduces decision.notional_exposure exactly (no behavior change). See
# docs/model_contracts/btc_sol_lowcost_tuning_sweep_20260713.md -- gate-off + 1.5x multiplier
# solo-backtested at oos_extended pnl +12.56%->+39.98% for SOL. No BTC equivalent: that same sweep
# found multiplier scaling doesn't help BTC (PnL stays flat while MDD grows), so only SOL gets one.
FINAL_GOVERNOR_OMEGA4_6_1_SOL_NOTIONAL_MULTIPLIER = float(
    os.getenv("FINAL_GOVERNOR_OMEGA4_6_1_SOL_NOTIONAL_MULTIPLIER", "1.0")
)
# 2026-07-22 session: BTC-only CryptoMamba h6 future-regime entry-time gate (skip an entry when the
# model's +6bar directional prediction disagrees with the entry side; never re-checked intra-trade,
# same design as ETH's own entry-time-only CryptoMamba filter -- the continuous/intra-trade variant
# was already found to whipsaw catastrophically). Fresh-forward VAL then OOS both improved or held
# steady (VAL pnl +7.45%->+7.98%, mdd unchanged; OOS pnl +12.59%->+26.79%, mdd -15.88%->-14.05%).
# SOL's own backtest of the same filter made both VAL and OOS worse -- BTC-only, default False.
FINAL_GOVERNOR_OMEGA4_6_1_BTC_CMAMBA_ENTRY_GATE_ENABLE = _env_flag(
    "FINAL_GOVERNOR_OMEGA4_6_1_BTC_CMAMBA_ENTRY_GATE_ENABLE", False
)
FINAL_GOVERNOR_OMEGA4_6_1_BTC_CMAMBA_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_OMEGA4_6_1_BTC_CMAMBA_MODEL_PATH",
    "data/ensemble/supervised/regime3_cryptomamba_pred_btc_h6_nocurrent_20260721/regime3_cryptomamba_pred_btc_h6_nocurrent_20260721_2024.pt",
)
# Portfolio-level notional budget shared across ETH/BTC/SOL real execution (see
# trading_bot_modules/portfolio_risk.py). Set to empty/"none"/"uncapped" (case-insensitive) to
# disable the cap entirely (PortfolioRiskManager already supports total_notional_cap=None --
# this was previously inexpressible via env var since float(os.getenv(...)) always parsed to a
# number). Default "3.0" is a live safety net; CURRENT_BASELINE's own backtest
# (docs/model_contracts/portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md) was run
# uncapped, so this default is a deliberate deviation for real-money safety, not a fidelity claim.
_FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP_RAW = os.getenv("FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP", "3.0").strip()
FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP = (
    None
    if _FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP_RAW.lower() in ("", "none", "uncapped", "null")
    else float(_FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP_RAW)
)
FINAL_GOVERNOR_PORTFOLIO_ETH_SHARE = float(os.getenv("FINAL_GOVERNOR_PORTFOLIO_ETH_SHARE", "0.5"))
FINAL_GOVERNOR_PORTFOLIO_BTC_SHARE = float(os.getenv("FINAL_GOVERNOR_PORTFOLIO_BTC_SHARE", "0.3"))
FINAL_GOVERNOR_PORTFOLIO_SOL_SHARE = float(os.getenv("FINAL_GOVERNOR_PORTFOLIO_SOL_SHARE", "0.2"))
# PortfolioRiskManager keys its budgets by whatever string is passed to scale_to_budget(), with no
# cross-caller usage tracking -- two independent strategies calling it with the SAME key would each
# get checked against the full per-key budget independently, silently allowing their combined real
# exposure to exceed that budget (found 2026-07-31 while evaluating an ETH second-slot candidate,
# see project memory project-eth-sigma3-1h-omega461-joint-portfolio-promising-20260731). Fix: ETH's
# share is split into per-SLOT sub-shares (not per-asset) so a future second ETH strategy gets its
# own non-competing carve-out of the eth share instead of colliding with Omega4.6.1's key. Defaults
# (1.0 / 0.0) reproduce today's live behavior exactly -- Omega4.6.1 keeps the entire eth share until
# a second slot's sub-share is explicitly set above 0.
FINAL_GOVERNOR_PORTFOLIO_ETH_OMEGA461_SUBSHARE = float(os.getenv("FINAL_GOVERNOR_PORTFOLIO_ETH_OMEGA461_SUBSHARE", "1.0"))
FINAL_GOVERNOR_PORTFOLIO_ETH_SIGMA3_1H_SUBSHARE = float(os.getenv("FINAL_GOVERNOR_PORTFOLIO_ETH_SIGMA3_1H_SUBSHARE", "0.0"))
FINAL_GOVERNOR_FULLY_LEARNED_ENABLE = _env_flag("FINAL_GOVERNOR_FULLY_LEARNED_ENABLE", True)
FINAL_GOVERNOR_FULLY_LEARNED_POLICY_PATH = os.getenv(
    "FINAL_GOVERNOR_FULLY_LEARNED_POLICY_PATH",
    "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/primary_parent.pkl",
)
FINAL_GOVERNOR_FULLY_LEARNED_SUMMARY_PATH = os.getenv(
    "FINAL_GOVERNOR_FULLY_LEARNED_SUMMARY_PATH",
    "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/primary_summary.json",
)
FINAL_GOVERNOR_FULLY_LEARNED_TP_SL_SCORE_PATH = os.getenv(
    "FINAL_GOVERNOR_FULLY_LEARNED_TP_SL_SCORE_PATH",
    "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/tp_sl_path_edge_predictor.pkl",
)
FINAL_GOVERNOR_FULLY_LEARNED_RUNTIME_CONFIG_PATH = os.getenv(
    "FINAL_GOVERNOR_FULLY_LEARNED_RUNTIME_CONFIG_PATH",
    "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/alpha7_decontam_deep_stop_cd18_runtime_config.json",
)
FINAL_GOVERNOR_FULLY_LEARNED_SCALE_ENABLE = _env_flag("FINAL_GOVERNOR_FULLY_LEARNED_SCALE_ENABLE", True)
FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_ENABLE = _env_flag("FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_ENABLE", True)
FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_POLICY_PATH = os.getenv(
    "FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_POLICY_PATH",
    "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/fallback_alpha43_no_legacy_parent.pkl",
)
FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_SUMMARY_PATH = os.getenv(
    "FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_SUMMARY_PATH",
    "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/fallback_alpha43_no_legacy_summary.json",
)
FINAL_GOVERNOR_FULLY_LEARNED_PRIMARY_LOW_CONF_THRESHOLD = 0.75
FINAL_GOVERNOR_FULLY_LEARNED_PRIMARY_LOW_CONF_TP_SCALE = 1.00
FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_TP_SCALE = float(
    os.getenv("FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_TP_SCALE", "1.00")
)
FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_EXIT_SUBMODEL_ENABLE = _env_flag(
    "FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_EXIT_SUBMODEL_ENABLE",
    False,
)
FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_EXIT_SUBMODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_EXIT_SUBMODEL_PATH",
    "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/alpha_submodel_fallback_exit_global_tp_084_20260526.json",
)
FINAL_GOVERNOR_CLEAN_REGIME4_STICKY_ENABLE = _env_flag("FINAL_GOVERNOR_CLEAN_REGIME4_STICKY_ENABLE", True)
FINAL_GOVERNOR_CLEAN_REGIME4_STICKY_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_CLEAN_REGIME4_STICKY_MODEL_PATH",
    "data/ensemble/supervised/clean_regime4_state24_sticky090_v2_20260517/clean_regime4_state24_sticky090_v2_2024.joblib",
)
FINAL_GOVERNOR_LIFECYCLE_V1_ENABLE = True
FINAL_GOVERNOR_LIFECYCLE_V1_POLICY_PATH = os.getenv(
    "FINAL_GOVERNOR_LIFECYCLE_V1_POLICY_PATH",
    (
        "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
        if FINAL_GOVERNOR_V21_2_JACKPOT_ENABLE
        else "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_v4_clean_train_to_2025_10.pkl"
    ),
)
FINAL_GOVERNOR_LIFECYCLE_V1_EXIT_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_LIFECYCLE_V1_EXIT_MODEL_PATH",
    "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_no_limit_exit_clean_train_to_2025_10.pkl",
)
FINAL_GOVERNOR_LIFECYCLE_V1_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_LIFECYCLE_V1_MODEL_PATH",
    "data/ensemble/supervised/clean_base_lifecycle_editor_v1/lifecycle_editor.pkl",
)
FINAL_GOVERNOR_LIFECYCLE_V1_EXIT_HARD_STOP = float(
    os.getenv(
        "FINAL_GOVERNOR_LIFECYCLE_V1_EXIT_HARD_STOP",
        "0.0" if (FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_ENABLE or FINAL_GOVERNOR_DISABLED_V13_1_ENABLE) else "0.025",
    )
)
FINAL_GOVERNOR_LIFECYCLE_V1_DISABLE_DAILY_LOSS_DD = _env_flag(
    "FINAL_GOVERNOR_LIFECYCLE_V1_DISABLE_DAILY_LOSS_DD",
    not _env_flag("BINANCE_EXECUTION_ENABLED", False),
)
FINAL_GOVERNOR_NEXT_OPEN_EXECUTION_ENABLE = _env_flag("FINAL_GOVERNOR_NEXT_OPEN_EXECUTION_ENABLE", True)
# Backtest/live parity mode: decide on the last completed signal bar and execute
# at the already-open next bar only if the live loop is still within the strict
# delay guard below. Do not schedule one more future bar by default.
FINAL_GOVERNOR_SCHEDULE_NEXT_BAR_OPEN_ENABLE = False
FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH = os.getenv(
    "FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH",
    "data/live/pending_next_open_intent.json",
)
FINAL_GOVERNOR_NEXT_OPEN_MAX_DELAY_SEC = float(os.getenv("FINAL_GOVERNOR_NEXT_OPEN_MAX_DELAY_SEC", "20"))
FINAL_GOVERNOR_NEXT_OPEN_WARN_DELAY_SEC = float(os.getenv("FINAL_GOVERNOR_NEXT_OPEN_WARN_DELAY_SEC", "15"))
FINAL_GOVERNOR_NEXT_OPEN_SHADOW_MAX_DELAY_SEC = float(
    os.getenv("FINAL_GOVERNOR_NEXT_OPEN_SHADOW_MAX_DELAY_SEC", "20")
)
FINAL_GOVERNOR_ALLOW_LATE_NEXT_OPEN_REAL_EXECUTION = _env_flag(
    "FINAL_GOVERNOR_ALLOW_LATE_NEXT_OPEN_REAL_EXECUTION",
    False,
)
FINAL_GOVERNOR_ALLOW_LATE_NEXT_OPEN_SHADOW_EXECUTION = _env_flag(
    "FINAL_GOVERNOR_ALLOW_LATE_NEXT_OPEN_SHADOW_EXECUTION",
    False,
)
FINAL_GOVERNOR_LIVE_COMPLETED_BAR_NEXT_OPEN_PROXY = _env_flag(
    "FINAL_GOVERNOR_LIVE_COMPLETED_BAR_NEXT_OPEN_PROXY",
    True,
)
FINAL_GOVERNOR_BAR_FETCH_DELAY_SEC = float(os.getenv("FINAL_GOVERNOR_BAR_FETCH_DELAY_SEC", "10.0"))
FINAL_GOVERNOR_TIMING_LOG_ENABLE = _env_flag("FINAL_GOVERNOR_TIMING_LOG_ENABLE", True)
FINAL_GOVERNOR_LIVE_PROCESS_BARS = int(float(os.getenv("FINAL_GOVERNOR_LIVE_PROCESS_BARS", "2500")))
FINAL_GOVERNOR_LIVE_MODEL_BARS = int(float(os.getenv("FINAL_GOVERNOR_LIVE_MODEL_BARS", "1200")))
FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_ENABLE = False
FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_REPORT_PATH",
    "data/ensemble/reports/clean_base_deep_constant_gross_v1_2026.json",
)
FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_TARGET_NOTIONAL = float(
    os.getenv("FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_TARGET_NOTIONAL", "3.6")
)
FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_FEE = float(
    os.getenv("FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_FEE", "0.0015")
)
FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_SLIP = float(
    os.getenv("FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_SLIP", "0.0006")
)
FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_NOTIONAL = float(
    os.getenv("FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_NOTIONAL", "0.0")
)
FINAL_GOVERNOR_DEEP_GATED_GROSS_ENABLE = False
FINAL_GOVERNOR_DEEP_GATED_GROSS_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_DEEP_GATED_GROSS_MODEL_PATH",
    "",
)
FINAL_GOVERNOR_DEEP_GATED_GROSS_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_DEEP_GATED_GROSS_REPORT_PATH",
    "",
)
FINAL_GOVERNOR_DEEP_GATED_GROSS_COST3_FEE = float(
    os.getenv("FINAL_GOVERNOR_DEEP_GATED_GROSS_COST3_FEE", "0.0015")
)
FINAL_GOVERNOR_DEEP_GATED_GROSS_COST3_SLIP = float(
    os.getenv("FINAL_GOVERNOR_DEEP_GATED_GROSS_COST3_SLIP", "0.0006")
)
FINAL_GOVERNOR_SAFE_LEARNED_CAP_ENABLE = False
FINAL_GOVERNOR_SAFE_LEARNED_CAP_AUDIT_PATH = os.getenv(
    "FINAL_GOVERNOR_SAFE_LEARNED_CAP_AUDIT_PATH",
    "",
)
FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_ENABLE = False
FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_MODEL_PATH",
    "",
)
FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_REPORT_PATH",
    "",
)
FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_AUDIT_PATH = os.getenv(
    "FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_AUDIT_PATH",
    "",
)
FINAL_GOVERNOR_V21_ENABLE = _env_flag(
    "FINAL_GOVERNOR_V21_ENABLE",
    False,
)
FINAL_GOVERNOR_V21_MODEL_PATH = os.getenv("FINAL_GOVERNOR_V21_MODEL_PATH", "")
FINAL_GOVERNOR_V21_REPORT_PATH = os.getenv("FINAL_GOVERNOR_V21_REPORT_PATH", "")
FINAL_GOVERNOR_V21_AUDIT_PATH = os.getenv("FINAL_GOVERNOR_V21_AUDIT_PATH", "")
FINAL_GOVERNOR_V21_PURE_MODE = _env_flag("FINAL_GOVERNOR_V21_PURE_MODE", True)
# Backtest parity: V21 selects core/scout/stop layers, while lifecycle
# candidate spacing and runtime gates stay active unless explicitly bypassed.
FINAL_GOVERNOR_V21_BYPASS_COOLDOWN = _env_flag("FINAL_GOVERNOR_V21_BYPASS_COOLDOWN", False)
FINAL_GOVERNOR_V21_BYPASS_RUNTIME_RISK_GATES = _env_flag(
    "FINAL_GOVERNOR_V21_BYPASS_RUNTIME_RISK_GATES",
    FINAL_GOVERNOR_V21_2_JACKPOT_ENABLE,
)
FINAL_GOVERNOR_V21_DISABLE_LEGACY_HARD_STOP = _env_flag(
    "FINAL_GOVERNOR_V21_DISABLE_LEGACY_HARD_STOP",
    True,
)
FINAL_GOVERNOR_V22_1_ENABLE = False
FINAL_GOVERNOR_V22_1_REQUIRED = False
FINAL_GOVERNOR_V22_1_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_V22_1_MODEL_PATH",
    "",
)
FINAL_GOVERNOR_V22_1_REPORT_PATH = os.getenv(
    "FINAL_GOVERNOR_V22_1_REPORT_PATH",
    "",
)
FINAL_GOVERNOR_V22_1_AUDIT_PATH = os.getenv(
    "FINAL_GOVERNOR_V22_1_AUDIT_PATH",
    "",
)
FINAL_GOVERNOR_DSAC_OVERLAY_ENABLE = False
FINAL_GOVERNOR_DSAC_OVERLAY_CKPT_PATH = os.getenv(
    "FINAL_GOVERNOR_DSAC_OVERLAY_CKPT_PATH",
    "data/ensemble/ckpt/dsac_full_retrain_v2/dsac_checkpoint.pth",
)
FINAL_GOVERNOR_DSAC_OVERLAY_MODE = os.getenv("FINAL_GOVERNOR_DSAC_OVERLAY_MODE", "half_if_opposite")
FINAL_GOVERNOR_DSAC_OVERLAY_THRESHOLD = float(os.getenv("FINAL_GOVERNOR_DSAC_OVERLAY_THRESHOLD", "0.50"))
FINAL_GOVERNOR_DSAC_OVERLAY_SCALE = float(os.getenv("FINAL_GOVERNOR_DSAC_OVERLAY_SCALE", "0.50"))
FINAL_GOVERNOR_DSAC_OVERLAY_COST_GATE_ENABLE = _env_flag("FINAL_GOVERNOR_DSAC_OVERLAY_COST_GATE_ENABLE", True)
FINAL_GOVERNOR_DSAC_OVERLAY_COST_BUFFER = float(os.getenv("FINAL_GOVERNOR_DSAC_OVERLAY_COST_BUFFER", "0.0035"))
FINAL_GOVERNOR_RECONCILE_DEFAULT_EXPOSURE = float(os.getenv("FINAL_GOVERNOR_RECONCILE_DEFAULT_EXPOSURE", "1.0"))
FINAL_GOVERNOR_RUNTIME_STATE_PATH = os.getenv(
    "FINAL_GOVERNOR_RUNTIME_STATE_PATH",
    "data/ensemble/final_governor_runtime_state.json",
)
FINAL_GOVERNOR_REGIME_PREDICTOR_ENABLE = _env_flag("FINAL_GOVERNOR_REGIME_PREDICTOR_ENABLE", False)
FINAL_GOVERNOR_ALLOW_LEGACY_CLEAN_REGIME_V4 = _env_flag("FINAL_GOVERNOR_ALLOW_LEGACY_CLEAN_REGIME_V4", False)
FINAL_GOVERNOR_REGIME_PREDICTOR_MODEL_PATH = os.getenv(
    "FINAL_GOVERNOR_REGIME_PREDICTOR_MODEL_PATH",
    "data/ensemble/supervised/certified_teacher_regime_moe_v1/model.pkl",
)
FINAL_GOVERNOR_REGIME_PREDICTOR_OVERRIDE = False
FINAL_GOVERNOR_REGIME_PREDICTOR_BLOCK = False
FINAL_GOVERNOR_REGIME_PREDICTOR_BLOCK_CONF = float(os.getenv("FINAL_GOVERNOR_REGIME_PREDICTOR_BLOCK_CONF", "0.85"))
FINAL_GOVERNOR_SNIPER_ENABLE = False
FINAL_GOVERNOR_TREND_ENABLE = False
FINAL_GOVERNOR_MICRO_ENABLE = False
CONSOLE_LOG_COMPACT = _env_flag("CONSOLE_LOG_COMPACT", True)
CONSOLE_LOG_MODEL_TRACE = _env_flag("CONSOLE_LOG_MODEL_TRACE", False)
CONSOLE_LOG_REFRESH = _env_flag("CONSOLE_LOG_REFRESH", False)
CONSOLE_LOG_COLOR = _env_flag("CONSOLE_LOG_COLOR", True) and not bool(os.getenv("NO_COLOR"))
CONSOLE_LOG_HEALTH_INTERVAL_SEC = float(os.getenv("CONSOLE_LOG_HEALTH_INTERVAL_SEC", "60"))
DATA_PIPELINE_HEALTH_ENABLE = _env_flag("DATA_PIPELINE_HEALTH_ENABLE", True)
DATA_PIPELINE_HEALTH_INTERVAL_SEC = float(os.getenv("DATA_PIPELINE_HEALTH_INTERVAL_SEC", "300"))
DATA_PIPELINE_HEALTH_PATH = os.getenv("DATA_PIPELINE_HEALTH_PATH", "data/live/data_pipeline_health.json")
DATA_PIPELINE_HEALTH_JSONL_PATH = os.getenv("DATA_PIPELINE_HEALTH_JSONL_PATH", "data/live/data_pipeline_health.jsonl")
DATA_PIPELINE_FEATURE_SNAPSHOT_ENABLE = _env_flag("DATA_PIPELINE_FEATURE_SNAPSHOT_ENABLE", True)
DATA_PIPELINE_FEATURE_SNAPSHOT_PATH = os.getenv("DATA_PIPELINE_FEATURE_SNAPSHOT_PATH", "data/live/decision_feature_snapshot.json")
DATA_PIPELINE_FEATURE_SNAPSHOT_JSONL_PATH = os.getenv("DATA_PIPELINE_FEATURE_SNAPSHOT_JSONL_PATH", "data/live/decision_feature_snapshot.jsonl")
DATA_PIPELINE_DECISION_HEARTBEAT_PATH = os.getenv(
    "DATA_PIPELINE_DECISION_HEARTBEAT_PATH",
    "data/live/trading_bot_decision_heartbeat.json",
)
DATA_PIPELINE_FEATURE_FRAME_SNAPSHOT_PATH = os.getenv(
    "DATA_PIPELINE_FEATURE_FRAME_SNAPSHOT_PATH",
    "data/live/decision_feature_frame_snapshot.pkl.gz",
)
DATA_PIPELINE_FEATURE_FRAME_DUCKDB_ENABLE = _env_flag("DATA_PIPELINE_FEATURE_FRAME_DUCKDB_ENABLE", True)
DATA_PIPELINE_FEATURE_FRAME_DUCKDB_PATH = os.getenv("DATA_PIPELINE_FEATURE_FRAME_DUCKDB_PATH", QUANT_MICRO_DB_PATH)
_DEFAULT_DECISION_FEATURE_FRAME_DUCKDB_TABLE = (
    f"decision_feature_frame_{OMEGA5_MODEL_ID}"
    if FINAL_GOVERNOR_OMEGA5_ENABLE
    else "decision_feature_frame_live_only_shadow_20260702"
)
DATA_PIPELINE_FEATURE_FRAME_DUCKDB_TABLE = os.getenv(
    "DATA_PIPELINE_FEATURE_FRAME_DUCKDB_TABLE",
    _DEFAULT_DECISION_FEATURE_FRAME_DUCKDB_TABLE,
)
PATCHTST_ERROR_ALERT_COOLDOWN_SEC = float(os.getenv("PATCHTST_ERROR_ALERT_COOLDOWN_SEC", "300"))
_DEFAULT_FINAL_GOVERNOR_AI_FEATURE_GROUPS = "tide,dlinear"
FINAL_GOVERNOR_AI_FEATURE_GROUPS = tuple(
    x.strip().lower()
    for x in os.getenv("FINAL_GOVERNOR_AI_FEATURE_GROUPS", _DEFAULT_FINAL_GOVERNOR_AI_FEATURE_GROUPS).split(",")
    if x.strip()
)
FINAL_GOVERNOR_AI_FEATURE_STALE_SEC = float(os.getenv("FINAL_GOVERNOR_AI_FEATURE_STALE_SEC", "1800"))
FINAL_GOVERNOR_AI_TIMING_LOG_ENABLE = _env_flag("FINAL_GOVERNOR_AI_TIMING_LOG_ENABLE", True)
