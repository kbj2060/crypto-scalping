import os
import sys
import asyncio
import time
import logging
import gc
import json
import math
import html
import hashlib
import importlib.util
import re
import tempfile
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from dotenv import load_dotenv
from torch.utils.data import DataLoader

load_dotenv()

noisy_loggers = [
    "pytorch_lightning",
    "pytorch_lightning.utilities.rank_zero",
    "lightning.pytorch",
    "lightning.pytorch.utilities.rank_zero",
    "lightning_fabric",
    "lightning_fabric.utilities.rank_zero",
    "neuralforecast",
    "nixtla"
]

for name in noisy_loggers:
    l = logging.getLogger(name)
    l.setLevel(logging.ERROR) # ERROR 이상만 출력되도록 격하
    l.propagate = False       # 핵심 ⭐: 루트 로거로 메시지가 전파되는 것을 물리적으로 절단

# Gemini SDK / HTTP 클라이언트 INFO 로그 정리
for name in ["httpx", "google", "google.genai", "google_genai"]:
    l = logging.getLogger(name)
    l.setLevel(logging.WARNING)
    l.propagate = False

# 2. Warning 메시지도 정규식 수준에서 차단
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch")
warnings.filterwarnings("ignore", ".*", module="lightning_fabric")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# 💡 [1. 경로 설정]
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PATHS = [
    _THIS_DIR,
    os.path.join(_THIS_DIR, "models"),
    os.path.join(_THIS_DIR, "uni2ts", "src"),
    os.path.join(_THIS_DIR, "strategies"),
    os.path.join(_THIS_DIR, "ensemble"),
]
for p in TARGET_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from trading_bot_modules.binance_live_fetcher import BinanceLiveFetcher
from trading_bot_modules.ensemble_predictor import EnsemblePredictor
from trading_bot_modules.async_jsonl_writer import AsyncJsonlWriter
from trading_bot_modules.process_lock import acquire_trading_bot_process_lock
from trading_bot_modules.state_transition_gate import StateTransitionGate
from trading_bot_modules.task_supervisor import AsyncTaskSupervisor
from trading_bot_modules.runtime_shutdown import shutdown_runtime_resources
from trading_bot_modules.duckdb_access import serialized_duckdb_access
from trading_bot_modules.position_sync import classify_account_position_snapshot, exchange_position_went_flat
from trading_bot_modules.binance_execution import BinanceFuturesExecutionAdapter
from trading_bot_modules.execution_health import (
    ExecutionAlertDeduper,
    build_execution_alert,
)
from trading_bot_modules.portfolio_risk import PortfolioRiskConfig, PortfolioRiskManager
from trading_bot_modules.orderbook_recorder import OrderBookRecorder
from trading_bot_modules.live_io import (
    _append_jsonl,
    _append_jsonl_many,
    _atomic_write_json,
    _file_age_sec,
    _read_json_safe,
)
from trading_bot_modules.position_accounting import (
    _accounting_equity_from_history,
    _build_position_accounting_audit_row,
    _price_return_frac,
    _safe_float,
)
from trading_bot_modules.omega5_live import (
    OMEGA5_MODEL_ID,
    OMEGA5_MODEL_VERSION,
    OMEGA5_OWNER,
    OMEGA5_SOURCE_MODEL_ID,
    Omega5LiveAdapter,
)
from trading_bot_modules.omega4_6_2_source_parent_live import (
    OMEGA462_SOURCE_PARENT_MODEL_ID,
    Omega462SourceParentConfig,
    Omega462SourceParentLiveAdapter,
)
from trading_bot_modules.omega4_6_1_live import (
    BTC_BASE_TEMPLATE,
    BTC_EXPERT_SCALES,
    LEVERAGE_CAP as OMEGA4_6_1_LEVERAGE_CAP,
    NOTIONAL_CAP as OMEGA4_6_1_NOTIONAL_CAP,
    OMEGA4_6_1_MODEL_ID,
    OMEGA4_6_1_MODEL_VERSION,
    OMEGA4_6_1_OWNER,
    SOL_BASE_TEMPLATE,
    SOL_EXPERT_SCALES,
    Omega461LiveAdapter,
)
from trading_bot_modules.omega4_6_1_runtime_contract import (
    EntryOverlayStatus,
    SizingDecision,
    direction_overlay_status,
    finalize_sizing,
)
from trading_bot_modules.omega4_6_1_shadow_state import (
    OMEGA4_6_1_SHADOW_ACTIVE_CONTRACT,
    OMEGA4_6_1_SHADOW_ACTIVE_STATE_KEY,
    validate_omega461_shadow_active_state,
)
from trading_bot_modules.omega4_6_1_btc_cmamba_entry_gate import BtcCmambaEntryGate
from trading_bot_modules.btc_swing_transition_live import BtcSwingTransitionLiveFeature
from trading_bot_modules.v15_conformal_sleeve_adapter import ConformalSleeveV15Adapter
from trading_bot_modules.v21_2_jackpot_adapter import JackpotRunnerV21_2Adapter

try:
    from pipeline.teacher_meta_side_features import append_side_teacher_features
except Exception:
    append_side_teacher_features = None

CLEAN_REGIME_2024_UNSUP_V4_PREFIX = "clean_regime_2024_unsup_v4_"
FORBIDDEN_ACTIVE_REGIME_PREFIXES = (
    CLEAN_REGIME_2024_UNSUP_V4_PREFIX,
    "clean_regime4_2024_unsup_v1_",
)
append_clean_regime = None

try:
    from scripts.retrain_clean_regime_hmm_engineered7_20260517 import (
        _with_state7 as _with_clean_regime4_state7,
    )
except Exception:
    _with_clean_regime4_state7 = None

try:
    from scripts.retrain_clean_regime4_hmm_raw_state12_20260517 import (
        PREFIX as CLEAN_REGIME4_STICKY_PREFIX,
        _append_factor_auxiliary as _append_clean_regime4_factor_auxiliary,
        _class_proba4 as _clean_regime4_class_proba,
        _output_frame as _output_clean_regime4_frame,
        _with_raw_state12 as _with_clean_regime4_raw_state12,
    )
except Exception as e:
    raise RuntimeError("clean_regime4_state24_sticky090_v2 live dependency is required") from e

CLEAN_REGIME4_STICKY_RUNTIME_PREFIX = "clean_regime4_state24_sticky090_v2_"

from features.engineering import FeatureEngineer
from features.elite import RegimeEngine
from enhanced_trading_engine import EnhancedTradingEngine
from ensemble.microstructure_wnc_sleeve import (
    MicrostructureSleeveConfig,
    microstructure_sleeve_decision,
    predict_microstructure_proba,
)
from ensemble.trend_bull_bear_sleeve import (
    TrendSleeveConfig,
    class_prob as _trend_class_prob,
    predict_trend_proba,
    trend_sleeve_decision,
)
from ensemble.macro_trend_sleeve import MacroTrendSleeveConfig, macro_trend_decision
from ensemble.learned_execution_policy import predict_learned_execution
from ensemble.fully_learned_governor_policy import (
    ACTION_CASH as FULLY_LEARNED_ACTION_CASH,
    ACTION_LONG as FULLY_LEARNED_ACTION_LONG,
    ACTION_SHORT as FULLY_LEARNED_ACTION_SHORT,
    prepare_features as prepare_fully_learned_governor_features,
    predict_policy_frame as predict_fully_learned_governor_frame,
)
from ensemble.train_trade_candidate_detector import FEATURE_COLS as EVENT_DETECTOR_FEATURE_COLS
from ensemble.train_rl_dsac_agent import DSACRouter, DSAC_STATE_DIM, GaussianActor

try:
    from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as _v31_live
except Exception:
    _v31_live = None

try:
    from scripts.train_eval_hf_v13_deep_entry_parent_lite_v38 import (
        DeepEntryParentLite as _Alpha2TeacherModel,
        _apply_norm as _alpha2_apply_norm,
        _seq_tensor as _alpha2_seq_tensor,
    )
except Exception:
    _Alpha2TeacherModel = None
    _alpha2_apply_norm = None
    _alpha2_seq_tensor = None

# ── HFT 마이크로스트럭처 및 꼬리 위험 요격기 ──
from microstructure_scanner import MicrostructureScanner
from tail_risk_interceptor import TailRiskInterceptor

try:
    from scripts.eval_sniper_day_ensemble_oos_2026 import (
        DEFAULT_SNIPER_CKPT as FINAL_GOVERNOR_SNIPER_CKPT,
        _action_side as _final_sniper_action_side,
        _load_discrete_actor as _load_final_sniper_actor,
        _sniper_action as _final_sniper_action,
    )
except ImportError:
    FINAL_GOVERNOR_SNIPER_CKPT = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/regime_rl_meta_controller_v3/best.pth"

    def _final_sniper_action_side(*_args: object, **_kwargs: object) -> int:
        return 0

    def _load_final_sniper_actor(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("optional sniper ensemble is unavailable; quarantined regime-v2 dependency is not loaded")

    def _final_sniper_action(*_args: object, **_kwargs: object) -> tuple[float, int]:
        return 0.0, 0

try:
    from scripts.train_event_masked_rl_meta_controller import (
        ACT_ALLOW_BEAR as FINAL_ACT_ALLOW_BEAR,
        ACT_ALLOW_BULL as FINAL_ACT_ALLOW_BULL,
        ACT_ALLOW_CURRENT as FINAL_ACT_ALLOW_CURRENT,
        ACT_CLOSE as FINAL_ACT_CLOSE,
        DEFAULT_MANIFEST as FINAL_GOVERNOR_MANIFEST,
        DEFAULT_POLICY as FINAL_GOVERNOR_POLICY,
        ExpertMetaTradingEnv,
        _ensure_event_aliases as _ensure_final_event_aliases,
        _load_manifest_policy as _load_final_manifest_policy,
    )
except ImportError:
    FINAL_ACT_ALLOW_CURRENT = 1
    FINAL_ACT_ALLOW_BULL = 2
    FINAL_ACT_ALLOW_BEAR = 3
    FINAL_ACT_CLOSE = 5
    FINAL_GOVERNOR_MANIFEST = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/regime_dsac_event_masked_v9_regime_v3_full5/manifest.json"
    FINAL_GOVERNOR_POLICY = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/regime_dsac_event_masked_v9_regime_v3_full5/router_policy.json"

    class ExpertMetaTradingEnv:  # type: ignore[no-redef]
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("optional event-masked meta controller is unavailable; quarantined regime-v2 dependency is not loaded")

    def _ensure_final_event_aliases(frame: pd.DataFrame) -> pd.DataFrame:
        return frame

    def _load_final_manifest_policy(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("optional event-masked meta controller is unavailable; quarantined regime-v2 dependency is not loaded")
try:
    from scripts.eval_hf_entry_overlay_grid import _quality_scaled_decisions as _lifecycle_quality_scaled_decisions
except Exception:
    def _lifecycle_quality_scaled_decisions(decisions, **_kwargs):
        return decisions
try:
    from scripts.train_eval_clean_base_exit_hazard_recalibrator_v1 import _bucket_from_vec as _lifecycle_bucket_from_vec
except Exception:
    def _lifecycle_bucket_from_vec(_vec, _thresholds):
        return "neutral"
try:
    from scripts import train_eval_clean_base_deep_constant_gross_v1 as _deep_cg
except Exception:
    class _UnavailableDeepConstantGross:
        @staticmethod
        def _row_features(*_args, **_kwargs):
            raise RuntimeError("deep_constant_gross_removed")

    _deep_cg = _UnavailableDeepConstantGross()
try:
    from scripts import train_eval_clean_base_deep_gated_gross_v2 as _deep_dgg
except Exception:
    class _UnavailableDeepGatedGross:
        @staticmethod
        def _row_signal(*_args, **_kwargs):
            raise RuntimeError("deep_gated_gross_removed")

    _deep_dgg = _UnavailableDeepGatedGross()
try:
    from scripts import train_eval_clean_base_deep_state_hybrid_v2 as _deep_v2
except Exception:
    class _UnavailableDeepV2:
        LOOKBACK = 0
        ENSEMBLE_EMBED_DIM = 0
        N_CLUSTERS = 0

        class EnhancedGRUStateEncoder:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("deep_state_hybrid_v2_removed")

        class GRUSeedEnsemble:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("deep_state_hybrid_v2_removed")

        @staticmethod
        def _deep_predict_v2(*_args, **_kwargs):
            raise RuntimeError("deep_state_hybrid_v2_removed")

    _deep_v2 = _UnavailableDeepV2()
try:
    from scripts import train_eval_clean_base_deep_state_hybrid_v1 as _deep_v1
except Exception:
    class _UnavailableDeepV1:
        LOOKBACK = 0
        EMBED_DIM = 0
        N_CLUSTERS = 0

        class base:
            @staticmethod
            def _stress(*_args, **_kwargs):
                return {}

        @staticmethod
        def _transform_sequence_matrix(*_args, **_kwargs):
            raise RuntimeError("deep_state_hybrid_v1_removed")

        @staticmethod
        def _sequence_tensor(*_args, **_kwargs):
            raise RuntimeError("deep_state_hybrid_v1_removed")

        @staticmethod
        def _state_features(*_args, **_kwargs):
            raise RuntimeError("deep_state_hybrid_v1_removed")

        @staticmethod
        def _predict_heads(*_args, **_kwargs):
            raise RuntimeError("deep_state_hybrid_v1_removed")

    _deep_v1 = _UnavailableDeepV1()
try:
    from scripts import train_eval_deep_state_safe_cap_reallocator_v15_context_router as _deep_state_v15
except Exception:
    class _UnavailableDeepStateV15:
        @staticmethod
        def _feature_matrix(*_args, **_kwargs):
            raise RuntimeError("deep_state_context_router_removed")

        @staticmethod
        def _predict_router(*_args, **_kwargs):
            raise RuntimeError("deep_state_context_router_removed")

    _deep_state_v15 = _UnavailableDeepStateV15()
try:
    from scripts import train_eval_deep_state_safe_cap_reallocator_v17_adaptive_calibrator as _deep_state_v17
    from scripts.train_eval_deep_state_safe_cap_reallocator_v17_adaptive_calibrator import (
        AdaptiveCalibrator,
        AdaptiveConfig,
    )
except Exception:
    class AdaptiveConfig:
        pass

    class AdaptiveCalibrator:
        pass

    class _UnavailableDeepStateV17:
        @staticmethod
        def _meta_frame(*_args, **_kwargs):
            raise RuntimeError("deep_state_adaptive_calibrator_removed")

        @staticmethod
        def _adaptive_q(*_args, **_kwargs):
            raise RuntimeError("deep_state_adaptive_calibrator_removed")

    _deep_state_v17 = _UnavailableDeepStateV17()
try:
    from scripts.train_eval_hf_no_limit_exit_governor import (
        _exit_probability_vec as _lifecycle_exit_probability_vec,
        _feature_vec_fast as _lifecycle_feature_vec_fast,
    )
except Exception:
    def _lifecycle_feature_vec_fast(*_args, **_kwargs):
        return np.zeros(1, dtype=float)

    def _lifecycle_exit_probability_vec(*_args, **_kwargs):
        return 0.0
from strategies.elite_builder import EliteSignals, row_to_market_row


class Colors:
    GREEN, RED, YELLOW, CYAN, BLUE, MAGENTA, DIM, RESET, BOLD = (
        '\033[92m', '\033[91m', '\033[93m', '\033[96m',
        '\033[94m', '\033[95m', '\033[2m', '\033[0m', '\033[1m',
    )

if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("LiveBot")


def _disabled_playbook_eval(action: int = 0, kelly: float = 0.0) -> dict:
    winner = {
        "matched": False,
        "name": "PLAYBOOK_DISABLED",
        "priority": 0,
        "action": int(action),
        "kelly": float(kelly),
        "reason": "display_only_disabled",
        "emergency_exit": False,
        "widen_trailing_stop": False,
        "meta": {},
    }
    return {
        "winner": dict(winner),
        "winner_hft": dict(winner),
        "winner_mft": dict(winner),
        "evaluations": [],
    }


from trading_bot_modules.runtime_config import (
    COMPACT_DASHBOARD_STATE_PATH,
    CONSOLE_LOG_COLOR,
    CONSOLE_LOG_COMPACT,
    CONSOLE_LOG_HEALTH_INTERVAL_SEC,
    CONSOLE_LOG_MODEL_TRACE,
    CONSOLE_LOG_REFRESH,
    DAILY_TRADE_REPORT_STATE_PATH,
    DASHBOARD_EVENTS_PATH,
    DASHBOARD_STATE_PATH,
    DATA_PIPELINE_DECISION_HEARTBEAT_PATH,
    DATA_PIPELINE_FEATURE_FRAME_DUCKDB_ENABLE,
    DATA_PIPELINE_FEATURE_FRAME_DUCKDB_PATH,
    DATA_PIPELINE_FEATURE_FRAME_DUCKDB_TABLE,
    DATA_PIPELINE_FEATURE_FRAME_SNAPSHOT_PATH,
    DATA_PIPELINE_FEATURE_SNAPSHOT_ENABLE,
    DATA_PIPELINE_FEATURE_SNAPSHOT_JSONL_PATH,
    DATA_PIPELINE_FEATURE_SNAPSHOT_PATH,
    DATA_PIPELINE_HEALTH_ENABLE,
    DATA_PIPELINE_HEALTH_INTERVAL_SEC,
    DATA_PIPELINE_HEALTH_JSONL_PATH,
    DATA_PIPELINE_HEALTH_PATH,
    ENSEMBLE_BALANCED_METRICS_PATH,
    ENSEMBLE_BALANCED_PARAMS_PATH,
    ENSEMBLE_LOWFREQ_METRICS_PATH,
    ENSEMBLE_LOWFREQ_PARAMS_PATH,
    ENSEMBLE_OVERHEAT_Z_MIN,
    ENSEMBLE_OVERHEAT_Z_WIN,
    ENSEMBLE_TRACKER_ENABLED,
    ENSEMBLE_TRACKER_EXIT_ON_HOLD,
    ENSEMBLE_TRACKER_FEE_RATE,
    ENSEMBLE_TRACKER_RECORDS_PATH,
    ENSEMBLE_TRACKER_SLIP_RATE,
    ENSEMBLE_TRACKER_STATE_PATH,
    FINAL_GOVERNOR_AI_FEATURE_GROUPS,
    FINAL_GOVERNOR_AI_FEATURE_STALE_SEC,
    FINAL_GOVERNOR_AI_TIMING_LOG_ENABLE,
    FINAL_GOVERNOR_ALLOW_LATE_NEXT_OPEN_REAL_EXECUTION,
    FINAL_GOVERNOR_ALLOW_LATE_NEXT_OPEN_SHADOW_EXECUTION,
    FINAL_GOVERNOR_ALPHA2_1_AUDIT_PATH,
    FINAL_GOVERNOR_ALPHA2_1_CONFIDENCE,
    FINAL_GOVERNOR_ALPHA2_1_ENABLE,
    FINAL_GOVERNOR_ALPHA2_1_MAX_NOTIONAL,
    FINAL_GOVERNOR_ALPHA2_1_MODEL_ID,
    FINAL_GOVERNOR_ALPHA2_1_PARENT_NOTIONAL_SCALE,
    FINAL_GOVERNOR_ALPHA2_1_REPORT_PATH,
    FINAL_GOVERNOR_ALPHA2_1_TEACHER_MODEL_PATH,
    FINAL_GOVERNOR_ALPHA3_CANONICAL_DECISION_ENABLE,
    FINAL_GOVERNOR_ALPHA3_CSV_COOLDOWN_PARITY_ENABLE,
    FINAL_GOVERNOR_ALPHA3_CSV_MARK_PARITY_ENABLE,
    FINAL_GOVERNOR_ALPHA3_LIVE_CONTRACT_ID,
    FINAL_GOVERNOR_ALPHA3_MODEL_ID,
    FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
    FINAL_GOVERNOR_ALPHA7_MODEL_ID,
    FINAL_GOVERNOR_BAR_FETCH_DELAY_SEC,
    FINAL_GOVERNOR_BUFFER_BARS,
    FINAL_GOVERNOR_CLEAN_REGIME4_STICKY_ENABLE,
    FINAL_GOVERNOR_CLEAN_REGIME4_STICKY_MODEL_PATH,
    FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_ENABLE,
    FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_MODEL_PATH,
    FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_REPORT_PATH,
    FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_REQUIRED,
    FINAL_GOVERNOR_DDH2_AUDIT_PATH,
    FINAL_GOVERNOR_DDH2_ENSEMBLE_ENABLE,
    FINAL_GOVERNOR_DDH2_REPORT_PATH,
    FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_FEE,
    FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_NOTIONAL,
    FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_SLIP,
    FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_ENABLE,
    FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_REPORT_PATH,
    FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_TARGET_NOTIONAL,
    FINAL_GOVERNOR_DEEP_GATED_GROSS_COST3_FEE,
    FINAL_GOVERNOR_DEEP_GATED_GROSS_COST3_SLIP,
    FINAL_GOVERNOR_DEEP_GATED_GROSS_ENABLE,
    FINAL_GOVERNOR_DEEP_GATED_GROSS_MODEL_PATH,
    FINAL_GOVERNOR_DEEP_GATED_GROSS_REPORT_PATH,
    FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_AUDIT_PATH,
    FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_ENABLE,
    FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_MODEL_PATH,
    FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_REPORT_PATH,
    FINAL_GOVERNOR_DISABLED_V13_1_ENABLE,
    FINAL_GOVERNOR_DISABLED_V13_1_MODEL_PATH,
    FINAL_GOVERNOR_DISABLED_V13_1_REPORT_PATH,
    FINAL_GOVERNOR_DISABLED_V13_1_REQUIRED,
    FINAL_GOVERNOR_DSAC_OVERLAY_CKPT_PATH,
    FINAL_GOVERNOR_DSAC_OVERLAY_COST_BUFFER,
    FINAL_GOVERNOR_DSAC_OVERLAY_COST_GATE_ENABLE,
    FINAL_GOVERNOR_DSAC_OVERLAY_ENABLE,
    FINAL_GOVERNOR_DSAC_OVERLAY_MODE,
    FINAL_GOVERNOR_DSAC_OVERLAY_SCALE,
    FINAL_GOVERNOR_DSAC_OVERLAY_THRESHOLD,
    FINAL_GOVERNOR_DUST_ENTRY_EXPOSURE,
    FINAL_GOVERNOR_EVENT_DETECTOR_PATH,
    FINAL_GOVERNOR_EXECUTION_POLICY_ENABLE,
    FINAL_GOVERNOR_EXECUTION_POLICY_IGNORE_MAX_HOLD,
    FINAL_GOVERNOR_EXECUTION_POLICY_LOW_QUALITY,
    FINAL_GOVERNOR_EXECUTION_POLICY_PATH,
    FINAL_GOVERNOR_EXECUTION_POLICY_QUALITY_OVERLAY,
    FINAL_GOVERNOR_EXECUTION_POLICY_TAIL_QUALITY,
    FINAL_GOVERNOR_FULLY_LEARNED_ENABLE,
    FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_ENABLE,
    FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_EXIT_SUBMODEL_ENABLE,
    FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_EXIT_SUBMODEL_PATH,
    FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_POLICY_PATH,
    FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_SUMMARY_PATH,
    FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_TP_SCALE,
    FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION,
    FINAL_GOVERNOR_FULLY_LEARNED_POLICY_PATH,
    FINAL_GOVERNOR_FULLY_LEARNED_PRIMARY_LOW_CONF_THRESHOLD,
    FINAL_GOVERNOR_FULLY_LEARNED_PRIMARY_LOW_CONF_TP_SCALE,
    FINAL_GOVERNOR_FULLY_LEARNED_RUNTIME_CONFIG_PATH,
    FINAL_GOVERNOR_FULLY_LEARNED_SCALE_ENABLE,
    FINAL_GOVERNOR_FULLY_LEARNED_SUMMARY_PATH,
    FINAL_GOVERNOR_FULLY_LEARNED_TP_SL_SCORE_PATH,
    FINAL_GOVERNOR_LEVERAGE,
    FINAL_GOVERNOR_LIFECYCLE_V1_DISABLE_DAILY_LOSS_DD,
    FINAL_GOVERNOR_LIFECYCLE_V1_ENABLE,
    FINAL_GOVERNOR_LIFECYCLE_V1_EXIT_HARD_STOP,
    FINAL_GOVERNOR_LIFECYCLE_V1_EXIT_MODEL_PATH,
    FINAL_GOVERNOR_LIFECYCLE_V1_MODEL_PATH,
    FINAL_GOVERNOR_LIFECYCLE_V1_POLICY_PATH,
    FINAL_GOVERNOR_LIVE_COMPLETED_BAR_NEXT_OPEN_PROXY,
    FINAL_GOVERNOR_LIVE_MODEL_BARS,
    FINAL_GOVERNOR_LIVE_PROCESS_BARS,
    FINAL_GOVERNOR_MACRO_BOOTSTRAP_CURRENT,
    FINAL_GOVERNOR_MACRO_ENABLE,
    FINAL_GOVERNOR_MACRO_LEVERAGE,
    FINAL_GOVERNOR_MACRO_LOCKOUT_BARS,
    FINAL_GOVERNOR_MACRO_LOCKOUT_ON_ANY_CLOSE,
    FINAL_GOVERNOR_MACRO_LOCKOUT_UNTIL_SIGNAL_CHANGE,
    FINAL_GOVERNOR_MACRO_LOOKBACK_BARS,
    FINAL_GOVERNOR_MACRO_NOTIONAL,
    FINAL_GOVERNOR_MACRO_PERSIST_UPDATES,
    FINAL_GOVERNOR_MACRO_STOP_LOSS,
    FINAL_GOVERNOR_MACRO_TAKE_PROFIT,
    FINAL_GOVERNOR_MACRO_THRESHOLD,
    FINAL_GOVERNOR_MACRO_TRAILING_ARM,
    FINAL_GOVERNOR_MACRO_TRAILING_GAP,
    FINAL_GOVERNOR_MACRO_UPDATE_BARS,
    FINAL_GOVERNOR_MANIFEST_PATH,
    FINAL_GOVERNOR_MICRO_ENABLE,
    FINAL_GOVERNOR_MICRO_MODEL_PATH,
    FINAL_GOVERNOR_MIN_ENTRY_EXPOSURE,
    FINAL_GOVERNOR_NEXT_OPEN_EXECUTION_ENABLE,
    FINAL_GOVERNOR_NEXT_OPEN_MAX_DELAY_SEC,
    FINAL_GOVERNOR_NEXT_OPEN_SHADOW_MAX_DELAY_SEC,
    FINAL_GOVERNOR_NEXT_OPEN_WARN_DELAY_SEC,
    FINAL_GOVERNOR_NOTIONAL,
    FINAL_GOVERNOR_OMEGA4_6_1_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_THRESHOLD,
    FINAL_GOVERNOR_OMEGA4_6_1_ETH_DURATION_GATE_OFF,
    FINAL_GOVERNOR_OMEGA4_6_1_ETH_NOTIONAL_MULTIPLIER,
    FINAL_GOVERNOR_OMEGA4_6_1_ETH_PORTFOLIO_CAP_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_BTC_CMAMBA_ENTRY_GATE_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_BTC_CMAMBA_MODEL_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_SIDECAR_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_SHADOW_ASSETS_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_SOL_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_DURATION_GATE_OFF,
    FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_MICROSTRUCTURE_SCANNER_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_ORDERBOOK_RECORDER_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_SOL_NOTIONAL_MULTIPLIER,
    FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_SIDECAR_PATH,
    FINAL_GOVERNOR_OMEGA5_ARTIFACT_INTEGRITY_PATH,
    FINAL_GOVERNOR_OMEGA5_CVP_AUDIT_PATH,
    FINAL_GOVERNOR_OMEGA5_ENABLE,
    FINAL_GOVERNOR_OMEGA5_FEATURE_VETO_REPORT_PATH,
    FINAL_GOVERNOR_OMEGA5_FRONTIER_AUDIT_PATH,
    FINAL_GOVERNOR_OMEGA5_PNL_TILT_REPORT_PATH,
    FINAL_GOVERNOR_OMEGA5_REDTEAM_PATH,
    FINAL_GOVERNOR_OMEGA5_REPORT_PATH,
    FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_CAP220_CONTRACT_PATH,
    FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE,
    FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_REPORT_PATH,
    FINAL_GOVERNOR_OMEGA5_TWO_STAGE_VETO_REPORT_PATH,
    FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH,
    FINAL_GOVERNOR_POLICY_PATH,
    FINAL_GOVERNOR_PORTFOLIO_BTC_SHARE,
    FINAL_GOVERNOR_PORTFOLIO_ETH_OMEGA461_SUBSHARE,
    FINAL_GOVERNOR_PORTFOLIO_ETH_SHARE,
    FINAL_GOVERNOR_PORTFOLIO_ETH_SIGMA3_1H_SUBSHARE,
    FINAL_GOVERNOR_PORTFOLIO_SOL_SHARE,
    FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP,
    FINAL_GOVERNOR_REGIME_PREDICTOR_BLOCK,
    FINAL_GOVERNOR_REGIME_PREDICTOR_BLOCK_CONF,
    FINAL_GOVERNOR_REGIME_PREDICTOR_ENABLE,
    FINAL_GOVERNOR_REGIME_PREDICTOR_MODEL_PATH,
    FINAL_GOVERNOR_RUNTIME_STATE_PATH,
    FINAL_GOVERNOR_SAFE_LEARNED_CAP_AUDIT_PATH,
    FINAL_GOVERNOR_SAFE_LEARNED_CAP_ENABLE,
    FINAL_GOVERNOR_SCHEDULE_NEXT_BAR_OPEN_ENABLE,
    FINAL_GOVERNOR_SNIPER_ENABLE,
    FINAL_GOVERNOR_SNIPER_MODEL_PATH,
    FINAL_GOVERNOR_TIMING_LOG_ENABLE,
    FINAL_GOVERNOR_TREND_ENABLE,
    FINAL_GOVERNOR_TREND_MODEL_PATH,
    FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_DISABLE,
    FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_FEE,
    FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_SLIP,
    FINAL_GOVERNOR_V1_5_COST_FIREWALL_ENABLE,
    FINAL_GOVERNOR_V1_5_COST_FIREWALL_STRESS_SLEEVE_MULT,
    FINAL_GOVERNOR_V21_2_JACKPOT_AUDIT_PATH,
    FINAL_GOVERNOR_V21_2_JACKPOT_ENABLE,
    FINAL_GOVERNOR_V21_2_JACKPOT_MODEL_PATH,
    FINAL_GOVERNOR_V21_2_JACKPOT_REPORT_PATH,
    FINAL_GOVERNOR_V21_2_JACKPOT_REQUIRED,
    FINAL_GOVERNOR_V21_AUDIT_PATH,
    FINAL_GOVERNOR_V21_BYPASS_COOLDOWN,
    FINAL_GOVERNOR_V21_BYPASS_RUNTIME_RISK_GATES,
    FINAL_GOVERNOR_V21_DISABLE_LEGACY_HARD_STOP,
    FINAL_GOVERNOR_V21_ENABLE,
    FINAL_GOVERNOR_V21_MODEL_PATH,
    FINAL_GOVERNOR_V21_PURE_MODE,
    FINAL_GOVERNOR_V21_REPORT_PATH,
    FINAL_GOVERNOR_V22_1_AUDIT_PATH,
    FINAL_GOVERNOR_V22_1_ENABLE,
    FINAL_GOVERNOR_V22_1_MODEL_PATH,
    FINAL_GOVERNOR_V22_1_REPORT_PATH,
    FINAL_GOVERNOR_V22_1_REQUIRED,
    FINAL_GOVERNOR_V31_AUDIT_PATH,
    FINAL_GOVERNOR_V31_DEEP_NOTIONAL,
    FINAL_GOVERNOR_V31_ENABLE,
    FINAL_GOVERNOR_V31_REPORT_PATH,
    FINAL_GOVERNOR_V31_REQUIRED,
    FINAL_GOVERNOR_V31_TRAIL_ACTIVATION,
    FINAL_GOVERNOR_V31_TRAIL_MIN_SL_MULT,
    FINAL_GOVERNOR_V31_V27_MODEL_PATH,
    FINAL_GOVERNOR_WINDOW_BARS,
    OMEGA4_6_1_SHADOW_ASSET_CONFIG,
    PATCHTST_ERROR_ALERT_COOLDOWN_SEC,
    POSITION_ACCOUNTING_AUDIT_PATH,
    QUANT_BAR_MINUTES,
    QUANT_HORIZON_MINUTES,
    QUANT_LOGIC_PATH,
    QUANT_LOOKBACK_MINUTES,
    QUANT_MAX_HISTORY_ROWS,
    QUANT_MICRO_DB_PATH,
    QUANT_TAIL_DB_PATH,
    QUANT_TOP_K_FEATURES,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TRADE_JOURNAL_PATH,
)

if CONSOLE_LOG_COMPACT:
    # Scanner modules still write to DuckDB; their 10s INFO panels are replaced by a
    # single consolidated health line from trading_bot.py.
    for _quiet_logger_name in ("microstructure_scanner", "tail_risk_interceptor"):
        logging.getLogger(_quiet_logger_name).setLevel(logging.WARNING)

_ENSEMBLE_OI_DELTA_WIN: deque[float] = deque(maxlen=max(30, ENSEMBLE_OVERHEAT_Z_WIN))
_ENSEMBLE_FUNDING_WIN: deque[float] = deque(maxlen=max(30, ENSEMBLE_OVERHEAT_Z_WIN))
_ENSEMBLE_LAST_OVERHEAT_OBS: tuple[float, float] | None = None
_QUANT_CARD_CACHE: dict[str, object] = {"minute_key": "", "payload": {}}
_QUANT_SNAPSHOT_MTIME: float | None = None


def _resolve_quant_logic_path() -> str:
    candidates = [
        os.path.join(_THIS_DIR, QUANT_LOGIC_PATH),
        os.path.join(_THIS_DIR, "quant", "live_30m_direction_quant.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _load_quant_snapshot_fn():
    global _QUANT_SNAPSHOT_MTIME
    script_path = _resolve_quant_logic_path()
    if not os.path.exists(script_path):
        _QUANT_SNAPSHOT_MTIME = None
        return None
    try:
        spec = importlib.util.spec_from_file_location("build_geometric_objective_dataset", script_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        # dataclass/typing introspection requires module to exist in sys.modules
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _QUANT_SNAPSHOT_MTIME = float(os.path.getmtime(script_path))
        return getattr(mod, "compute_live_quant_snapshot", None)
    except Exception as e:
        logger.warning("퀀트 카드 스크립트 로딩 실패: %s", e)
        return None


_QUANT_SNAPSHOT_FN = None


@serialized_duckdb_access(lambda *_args, **_kwargs: QUANT_MICRO_DB_PATH)
def _build_quant_formula_card(eth_df: pd.DataFrame, current_price: float, current_time_kst) -> dict:
    global _QUANT_SNAPSHOT_FN, _QUANT_SNAPSHOT_MTIME
    minute_key = pd.Timestamp(current_time_kst).strftime("%Y-%m-%d %H:%M")
    cached = dict(_QUANT_CARD_CACHE.get("payload", {}) or {})
    if _QUANT_CARD_CACHE.get("minute_key") == minute_key and cached:
        return cached
    script_path = _resolve_quant_logic_path()
    try:
        now_mtime = float(os.path.getmtime(script_path)) if os.path.exists(script_path) else None
    except Exception:
        now_mtime = None
    if now_mtime is not None and _QUANT_SNAPSHOT_MTIME is not None and now_mtime != _QUANT_SNAPSHOT_MTIME:
        _QUANT_SNAPSHOT_FN = _load_quant_snapshot_fn()
    if _QUANT_SNAPSHOT_FN is None:
        _QUANT_SNAPSHOT_FN = _load_quant_snapshot_fn()
    if _QUANT_SNAPSHOT_FN is None:
        payload = {
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "signal": "HOLD",
            "direction": "NEUTRAL",
            "prob_up": 0.5,
            "prob_down": 0.5,
            "pred_price_30m": float(current_price or 0.0),
            "current_price": float(current_price or 0.0),
            "expected_return_pct": 0.0,
            "confidence": 0.0,
            "win_rate_model": 0.0,
            "win_rate_baseline": 0.0,
            "rmse_model": 0.0,
            "rmse_naive": 0.0,
            "r2_model": 0.0,
            "r2_naive": 0.0,
            "alpha": 0.0,
            "l2": 0.0,
            "error": "quant_fn_not_loaded",
        }
        return payload
    try:
        close_df = pd.DataFrame({
            "ts": pd.to_datetime(eth_df["timestamp"], utc=True, errors="coerce"),
            "close": pd.to_numeric(eth_df["close"], errors="coerce"),
        }).dropna(subset=["ts", "close"])
        payload = _QUANT_SNAPSHOT_FN(
            micro_db_path=QUANT_MICRO_DB_PATH,
            tail_db_path=QUANT_TAIL_DB_PATH,
            close_df=close_df,
            current_price=float(current_price or 0.0),
            lookback_minutes=int(max(1, QUANT_LOOKBACK_MINUTES)),
            horizon_minutes=int(max(1, QUANT_HORIZON_MINUTES)),
            bar_minutes=int(max(1, QUANT_BAR_MINUTES)),
            top_k_features=int(max(5, QUANT_TOP_K_FEATURES)),
            max_history_rows=int(max(500, QUANT_MAX_HISTORY_ROWS)),
        )
        _QUANT_CARD_CACHE["minute_key"] = minute_key
        _QUANT_CARD_CACHE["payload"] = payload
        return dict(payload)
    except Exception as e:
        payload = {
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "signal": "HOLD",
            "direction": "NEUTRAL",
            "prob_up": 0.5,
            "prob_down": 0.5,
            "pred_price_30m": float(current_price or 0.0),
            "current_price": float(current_price or 0.0),
            "expected_return_pct": 0.0,
            "confidence": 0.0,
            "win_rate_model": 0.0,
            "win_rate_baseline": 0.0,
            "rmse_model": 0.0,
            "rmse_naive": 0.0,
            "r2_model": 0.0,
            "r2_naive": 0.0,
            "alpha": 0.0,
            "l2": 0.0,
            "error": str(e),
        }
        # 에러 시에는 분 캐시를 고정하지 않아 다음 사이클 즉시 재시도
        return payload

# ════════════════════════════════════════════════════════════════
# 0. 공통 헬퍼
# ════════════════════════════════════════════════════════════════








def _now_kst_iso() -> str:
    return pd.Timestamp.now(tz="Asia/Seoul").isoformat()




def _load_ensemble_cards() -> dict:
    now_iso = _now_kst_iso()
    bal_raw = _read_json_safe(ENSEMBLE_BALANCED_METRICS_PATH)
    low_raw = _read_json_safe(ENSEMBLE_LOWFREQ_METRICS_PATH)
    try:
        bal_updated = pd.Timestamp.fromtimestamp(os.path.getmtime(ENSEMBLE_BALANCED_METRICS_PATH), tz="UTC").isoformat()
    except Exception:
        bal_updated = now_iso
    try:
        low_updated = pd.Timestamp.fromtimestamp(os.path.getmtime(ENSEMBLE_LOWFREQ_METRICS_PATH), tz="UTC").isoformat()
    except Exception:
        low_updated = now_iso

    bal_res = dict(bal_raw.get("ensemble_result", {}) or {})
    bal_search = dict(bal_raw.get("search", {}) or {})
    bal = {
        "name": "균형 앙상블",
        "spec": f"k={int(bal_search.get('top_k', 10) or 10)} / votes={int(bal_search.get('min_votes', 6) or 6)}",
        "param_updated_at": bal_updated,
        "update_cycle": "주 1회 업데이트",
        "pnl_pct": float(bal_res.get("pnl_pct", 0.0) or 0.0),
        "mdd_pct": float(bal_res.get("mdd_pct", 0.0) or 0.0),
        "trades": int(bal_res.get("trades", 0) or 0),
        "win_rate": float(bal_res.get("win_rate", 0.0) or 0.0),
        "sharpe": float(bal_res.get("sharpe", 0.0) or 0.0),
    }

    low_best = dict(low_raw.get("best", {}) or {})
    low = {
        "name": "저빈도 고수익 앙상블",
        "spec": f"k={int(low_best.get('k', 10) or 10)} / votes={int(low_best.get('votes', 7) or 7)}",
        "param_updated_at": low_updated,
        "update_cycle": "주 1회 업데이트",
        "pnl_pct": float(low_best.get("pnl_pct", 0.0) or 0.0),
        "mdd_pct": float(low_best.get("mdd_pct", 0.0) or 0.0),
        "trades": int(low_best.get("trades", 0) or 0),
        "win_rate": float(low_best.get("win_rate", 0.0) or 0.0),
        "sharpe": float(low_best.get("sharpe", 0.0) or 0.0),
    }

    return {
        "updated_at": now_iso,
        "balanced": bal,
        "lowfreq": low,
    }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _pb_score(ev: dict | None) -> float:
    if not isinstance(ev, dict):
        return 0.0
    meta = ev.get("meta", {}) or {}
    raw = meta.get("unified_score", None)
    try:
        if raw is not None and np.isfinite(float(raw)):
            return _clamp01(float(raw))
    except Exception:
        pass
    return 1.0 if bool(ev.get("matched", False)) else 0.0


def _pb_action(ev: dict | None, default: int = 0) -> int:
    if not isinstance(ev, dict):
        return int(default)
    try:
        a = int(ev.get("action", default))
        return a if a in (0, 1, 2) else int(default)
    except Exception:
        return int(default)


def _action_text(a: int) -> str:
    return "LONG" if int(a) == 1 else ("SHORT" if int(a) == 2 else "HOLD")










def _build_trades_tail_from_router(router, limit: int = 120) -> list[dict]:
    rows = list(getattr(router, "trade_history", []) or [])
    trades_tail: list[dict] = []
    eq = 1.0
    for row in rows:
        pnl_frac = float(row.get("pnl_frac", row.get("pnl", 0.0)) or 0.0)
        eq *= (1.0 + pnl_frac)
        trades_tail.append({
            "ts": str(row.get("ts", "")),
            "pnl_pct": pnl_frac * 100.0,
            "equity": eq,
        })
    return trades_tail[-max(1, int(limit)) :]


def _router_closed_trade_equity(router) -> float:
    eq = 1.0
    for row in list(getattr(router, "trade_history", []) or []):
        pnl_frac = float(row.get("pnl_frac", row.get("pnl", 0.0)) or 0.0)
        eq *= (1.0 + pnl_frac)
    return float(eq)


def _router_open_mark_pnl_frac(router, current_price: float) -> float:
    if getattr(router, "pos", None) is None or float(current_price or 0.0) <= 0.0:
        return 0.0
    mark_fn = getattr(router, "_mark_pnl_frac", None)
    if callable(mark_fn):
        return float(mark_fn(float(current_price)))
    unreal_fn = getattr(router, "unrealized_pnl", None)
    if callable(unreal_fn):
        return float(unreal_fn(float(current_price))) / 100.0
    return 0.0


def _router_strategy_equity(router, current_price: float) -> float:
    closed_eq = _router_closed_trade_equity(router)
    open_mark = _router_open_mark_pnl_frac(router, current_price)
    return float(closed_eq * (1.0 + open_mark))


def _build_compact_dashboard_state(base_state: dict, compact_router, current_price: float, now_kst) -> dict:
    state = dict(base_state or {})
    compact_signal = dict(state.get("compact_signal", {}) or state.get("signal", {}) or {})
    base_position = dict(state.get("position", {}) or {})
    agents = dict(state.get("agents", {}) or {})
    compact_agent = dict(agents.get("compact", {}) or agents.get("governor", {}) or agents.get("primary", {}) or {})
    omega4_6_1_agent = dict(agents.get("omega4_6_1", {}) or {})
    risk_take_profit = float(base_position.get("take_profit", compact_signal.get("take_profit", 0.0)) or 0.0)
    risk_stop_loss = float(base_position.get("stop_loss", compact_signal.get("stop_loss", 0.0)) or 0.0)
    risk_source = str(base_position.get("risk_source", compact_signal.get("risk_source", compact_signal.get("source", ""))) or "")
    if compact_router.pos in {"LONG", "SHORT"} and risk_take_profit <= 0.0 and risk_stop_loss <= 0.0:
        governor_owner = str(
            compact_signal.get(
                "governor_owner",
                dict(agents.get("governor", {}) or {}).get("owner", ""),
            )
            or ""
        )
        omega_take_profit = float(omega4_6_1_agent.get("active_take_profit", 0.0) or 0.0)
        omega_stop_loss = float(omega4_6_1_agent.get("active_stop_loss", 0.0) or 0.0)
        if governor_owner == OMEGA4_6_1_OWNER and (omega_take_profit > 0.0 or omega_stop_loss > 0.0):
            risk_take_profit = omega_take_profit
            risk_stop_loss = omega_stop_loss
            risk_source = OMEGA4_6_1_OWNER
    perf = compact_router.performance_metrics(now_kst)
    strategy_equity = _router_strategy_equity(compact_router, current_price)
    open_mark_pnl = _router_open_mark_pnl_frac(compact_router, current_price)
    state["schema_version"] = "live.dashboard.compact.v1"
    state["signal"] = compact_signal
    state["agents"] = {
        "primary": compact_agent,
        "governor": dict(agents.get("governor", compact_agent) or {}),
        "omega4_6_1": omega4_6_1_agent,
        "agreement_count": int(agents.get("agreement_count", 0) or 0),
        "net_score": float(agents.get("net_score", 0.0) or 0.0),
        "conviction": float(agents.get("conviction", 0.0) or 0.0),
    }
    state["regime"] = str(compact_signal.get("regime", state.get("regime", "UNKNOWN")))
    state["position"] = {
        "current": compact_router.pos or "NONE",
        "trade_id": str(compact_router.open_trade_id or ""),
        "entry_price": float(compact_router.entry_price or 0.0),
        "entry_price_source": str(compact_router.entry_price_source or ""),
        "entry_decision_price": float(compact_router.entry_decision_price or 0.0),
        "exchange_entry_price": float(compact_router.exchange_entry_price or 0.0),
        "entry_execution_liquidity": str(compact_router.entry_execution_liquidity or ""),
        "entry_execution_route": str(compact_router.entry_execution_route or ""),
        "entry_execution_order_type": str(compact_router.entry_execution_order_type or ""),
        "decision_at": str(compact_router.decision_at or compact_signal.get("decision_at", "")),
        "opened_at": str(compact_router.opened_at or ""),
        "hold_bars": int(compact_router.hold_count or 0),
        "position_fraction": float(compact_router.position_fraction or 0.0),
        "margin_fraction": float(compact_router.position_fraction or 0.0),
        "execution_leverage": float(compact_router.execution_leverage or 1.0),
        "notional_exposure": float(compact_router.current_leverage or 0.0),
        "total_exposure": float(compact_router.current_leverage or 0.0),
        "unrealized_pnl_pct": float(compact_router.unrealized_pnl(current_price) if compact_router.pos and current_price > 0.0 else 0.0),
        "position_realized_pnl_frac": float(compact_router.position_realized_pnl_frac or 0.0),
        "position_realized_pnl_pct": float((compact_router.position_realized_pnl_frac or 0.0) * 100.0),
        "last_resize_realized_pnl_frac": float(compact_router.last_resize_realized_pnl_frac or 0.0),
        "strategy_equity": float(strategy_equity),
        "closed_trade_equity": float(_router_closed_trade_equity(compact_router)),
        "deployed_equity": float(strategy_equity * float(compact_router.position_fraction or 0.0)),
        "gross_exposure_equity": float(strategy_equity * float(compact_router.current_leverage or 0.0)),
        "unrealized_pnl_amount": float(strategy_equity * open_mark_pnl),
        "trade_pnl_pct": None,
        "take_profit": float(risk_take_profit),
        "stop_loss": float(risk_stop_loss),
        "max_hold_bars": int(base_position.get("max_hold_bars", compact_signal.get("max_hold_bars", 0)) or 0),
        "max_hold_remaining_bars": int(base_position.get("max_hold_remaining_bars", compact_signal.get("max_hold_remaining_bars", 0)) or 0),
        "take_profit_price": float(base_position.get("take_profit_price", compact_signal.get("take_profit_price", 0.0)) or 0.0),
        "tp_price": float(base_position.get("tp_price", compact_signal.get("tp_price", 0.0)) or 0.0),
        "stop_price": float(base_position.get("stop_price", compact_signal.get("stop_price", 0.0)) or 0.0),
        "sl_price": float(base_position.get("sl_price", compact_signal.get("sl_price", 0.0)) or 0.0),
        "effective_take_profit": float(base_position.get("effective_take_profit", compact_signal.get("effective_take_profit", risk_take_profit)) or risk_take_profit),
        "effective_stop_loss": float(base_position.get("effective_stop_loss", compact_signal.get("effective_stop_loss", risk_stop_loss)) or risk_stop_loss),
        "risk_source": str(risk_source),
    }
    if compact_router.pos not in {"LONG", "SHORT"}:
        state["position"].update(
            {
                "take_profit": 0.0,
                "stop_loss": 0.0,
                "max_hold_bars": 0,
                "max_hold_remaining_bars": 0,
                "take_profit_price": 0.0,
                "tp_price": 0.0,
                "stop_price": 0.0,
                "sl_price": 0.0,
                "effective_take_profit": 0.0,
                "effective_stop_loss": 0.0,
                "risk_source": "flat",
            }
        )
    state["performance"] = {
        "pnl_24h": float(perf.get("pnl_24h", 0.0)),
        "wr_24h": float(perf.get("wr_24h", 0.0)),
        "mdd_24h": float(perf.get("mdd_24h", 0.0)),
        "pnl_7d": float(perf.get("pnl_7d", 0.0)),
        "wr_7d": float(perf.get("wr_7d", 0.0)),
        "mdd_7d": float(perf.get("mdd_7d", 0.0)),
        "pnl_all": float(perf.get("pnl_all", 0.0)),
        "pnl_24h_sum": float(perf.get("pnl_24h_sum", perf.get("pnl_24h", 0.0))),
        "pnl_7d_sum": float(perf.get("pnl_7d_sum", perf.get("pnl_7d", 0.0))),
        "pnl_all_sum": float(perf.get("pnl_all_sum", perf.get("pnl_all", 0.0))),
        "wr_all": float(perf.get("wr_all", 0.0)),
        "mdd_all": float(perf.get("mdd_all", 0.0)),
    }
    state["trades_tail"] = _build_trades_tail_from_router(compact_router)
    state["compact_mode"] = True
    state["governor_mode"] = True
    return state


def _zscore_last(win: deque[float], x: float) -> float:
    if len(win) < max(2, ENSEMBLE_OVERHEAT_Z_MIN):
        return 0.0
    arr = np.array(win, dtype=np.float64)
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    return float((x - mu) / sd)


def _update_overheat_score(ms: dict) -> float:
    global _ENSEMBLE_LAST_OVERHEAT_OBS
    oi_delta = _safe_float(ms.get("oi_delta_pct", 0.0), 0.0)
    funding = _safe_float(ms.get("funding_rate", 0.0), 0.0)
    obs = (round(oi_delta, 12), round(funding, 12))
    if _ENSEMBLE_LAST_OVERHEAT_OBS != obs:
        _ENSEMBLE_OI_DELTA_WIN.append(float(oi_delta))
        _ENSEMBLE_FUNDING_WIN.append(float(funding))
        _ENSEMBLE_LAST_OVERHEAT_OBS = obs
    oi_z = _zscore_last(_ENSEMBLE_OI_DELTA_WIN, oi_delta)
    funding_z = _zscore_last(_ENSEMBLE_FUNDING_WIN, funding)
    if len(_ENSEMBLE_OI_DELTA_WIN) < ENSEMBLE_OVERHEAT_Z_MIN or len(_ENSEMBLE_FUNDING_WIN) < ENSEMBLE_OVERHEAT_Z_MIN:
        # 워밍업 구간 fallback: 기존 스케일 유지
        overheat = (oi_delta * 100.0) + (funding * 1500.0)
    else:
        overheat = oi_z + funding_z
    ms["overheat_score"] = float(overheat)
    ms["overheat_oi_z"] = float(oi_z)
    ms["overheat_funding_z"] = float(funding_z)
    ms["overheat_samples"] = int(min(len(_ENSEMBLE_OI_DELTA_WIN), len(_ENSEMBLE_FUNDING_WIN)))
    return float(overheat)


def _load_param_pool(path: str, top_key: str) -> list[dict]:
    raw = _read_json_safe(path)
    arr = list(raw.get(top_key, []) or [])
    out: list[dict] = []
    for item in arr:
        p = dict((item or {}).get("params", {}) or {})
        if p:
            out.append(p)
    return out


def _param_vote_from_snapshot(p: dict, ms: dict, tr: dict, current_pos: str = "NONE") -> tuple[int, float, float]:
    obi = _safe_float(ms.get("obi", 0.0), 0.0)
    taker_buy_ratio = _safe_float(ms.get("taker_buy_ratio", 0.5), 0.5)
    flow = 2.0 * max(0.0, min(1.0, taker_buy_ratio)) - 1.0
    nif = _safe_float(ms.get("nif_whale", 0.0), 0.0)
    absb = _safe_float(ms.get("shadow_absorption_score", 0.0), 0.0)
    tox = _safe_float(ms.get("shadow_toxicity_score", 0.0), 0.0)
    qcol = _safe_float(ms.get("shadow_queue_collapse", 0.0), 0.0)
    eai = _safe_float(ms.get("eai", 0.0), 0.0)
    oi_delta = _safe_float(ms.get("oi_delta_pct", 0.0), 0.0)
    funding = _safe_float(ms.get("funding_rate", 0.0), 0.0)
    pv30 = abs(_safe_float(ms.get("price_volatility_30m", 0.0), 0.0))
    warmup_price_samples = int(_safe_float(ms.get("warmup_price_samples", 0.0), 0.0))
    data_stale = bool(ms.get("data_stale", False))
    long_usd = _safe_float(tr.get("long_usd_1m", 0.0), 0.0)
    short_usd = _safe_float(tr.get("short_usd_1m", 0.0), 0.0)
    aft = _safe_float(tr.get("aftershock_prob", tr.get("shadow_aftershock_prob", 0.0)), 0.0)

    liq = (short_usd - long_usd) / (abs(short_usd) + abs(long_usd) + 1e-8)
    overheat = _safe_float(ms.get("overheat_score", (oi_delta * 100.0) + (funding * 1500.0)), 0.0)

    raw = (
        _safe_float(p.get("w_nif", 0.0)) * nif
        + _safe_float(p.get("w_flow", 0.0)) * flow
        + _safe_float(p.get("w_obi", 0.0)) * (-obi)
        + _safe_float(p.get("w_abs", 0.0)) * absb
        + _safe_float(p.get("w_liq", 0.0)) * liq
        + _safe_float(p.get("w_eai", 0.0)) * np.tanh(eai / 2.0)
        - _safe_float(p.get("w_tox", 0.0)) * tox
        - _safe_float(p.get("w_aft", 0.0)) * aft
    )

    temp = max(1e-4, _safe_float(p.get("temp", 0.25), 0.25))
    bias = _safe_float(p.get("bias", 0.0), 0.0)
    long_gate = 1.0 if overheat < _safe_float(p.get("overheat_long_max", 1.0), 1.0) else 0.0
    short_boost = _safe_float(p.get("short_boost", 1.0), 1.0) if overheat > _safe_float(p.get("overheat_short_min", 1.0), 1.0) else 1.0
    base_long = float(1.0 / (1.0 + np.exp(-np.clip((raw - bias) / temp, -40.0, 40.0))))
    base_short = float(1.0 / (1.0 + np.exp(-np.clip((-raw - bias) / temp, -40.0, 40.0))))
    tail_pen = float(np.clip(1.0 - (_safe_float(p.get("tail_tox", 0.0)) * tox + _safe_float(p.get("tail_qc", 0.0)) * qcol + _safe_float(p.get("tail_aft", 0.0)) * aft), 0.0, 1.0))
    ls = float(base_long * long_gate * tail_pen)
    ss = float(base_short * short_boost * tail_pen)

    atr_min = _safe_float(p.get("atr_min", 0.0008), 0.0008)
    # 30분 변동성(pv30)이 준비되지 않은 재시작 초기 구간에서는
    # WS가 정상(LIVE)이고 최소 가격 샘플(5분) 확보 시 soft-score 투표를 허용한다.
    tradable = bool(pv30 >= atr_min * 0.5) or bool((pv30 <= 0.0) and (not data_stale) and (warmup_price_samples >= 5))
    entry = _safe_float(p.get("entry", 0.7), 0.7)
    fire = float(np.clip(entry * 0.85, 0.0, 1.0))
    pos_u = str(current_pos or "NONE").upper()
    long_th = fire if pos_u == "LONG" else entry
    short_th = fire if pos_u == "SHORT" else entry
    if not tradable:
        return 0, ls, ss
    needed = long_th if ls >= ss else short_th
    if max(ls, ss) < needed:
        return 0, ls, ss
    if abs(ls - ss) < 0.05:
        return 0, ls, ss
    if ls > ss and ls >= long_th:
        return 1, ls, ss
    if ss > ls and ss >= short_th:
        return 2, ls, ss
    return 0, ls, ss


def _ensemble_vote_runtime(params: list[dict], min_votes: int, ms: dict, tr: dict, base_kelly: float, veto_on: bool, tag: str, current_pos: str = "NONE") -> dict:
    if not params:
        return {
            "decision": "HOLD",
            "action": 0,
            "confidence_score": 0,
            "kelly_weight": 0.0,
            "reason": f"{tag}_NO_PARAMS",
            "votes_long": 0,
            "votes_short": 0,
            "votes_hold": 0,
            "pool_k": 0,
            "required_votes": int(min_votes),
        }
    votes_l = 0
    votes_s = 0
    votes_h = 0
    ls_sum = 0.0
    ss_sum = 0.0
    blocked_by_tradable = 0
    blocked_by_fire = 0
    for p in params:
        a, ls, ss = _param_vote_from_snapshot(p, ms, tr, current_pos=current_pos)
        ls_sum += float(ls)
        ss_sum += float(ss)
        atr_min = _safe_float(p.get("atr_min", 0.0008), 0.0008)
        entry = _safe_float(p.get("entry", 0.7), 0.7)
        fire = float(np.clip(entry * 0.85, 0.0, 1.0))
        pos_u = str(current_pos or "NONE").upper()
        long_th = fire if pos_u == "LONG" else entry
        short_th = fire if pos_u == "SHORT" else entry
        pv30 = abs(_safe_float(ms.get("price_volatility_30m", 0.0), 0.0))
        warmup_price_samples = int(_safe_float(ms.get("warmup_price_samples", 0.0), 0.0))
        data_stale = bool(ms.get("data_stale", False))
        tradable = bool(pv30 >= atr_min * 0.5) or bool((pv30 <= 0.0) and (not data_stale) and (warmup_price_samples >= 5))
        if not tradable:
            blocked_by_tradable += 1
        else:
            needed = long_th if ls >= ss else short_th
            if max(ls, ss) < needed or abs(ls - ss) < 0.05:
                blocked_by_fire += 1
        if a == 1:
            votes_l += 1
        elif a == 2:
            votes_s += 1
        else:
            votes_h += 1

    k = len(params)
    if veto_on:
        action = 0
        reason = f"{tag}_VETO_SHIELD_ACTIVE"
    elif votes_l >= int(min_votes) and votes_l > votes_s:
        action = 1
        reason = f"{tag}_VOTE_LONG"
    elif votes_s >= int(min_votes) and votes_s > votes_l:
        action = 2
        reason = f"{tag}_VOTE_SHORT"
    else:
        action = 0
        reason = f"{tag}_VOTE_INSUFFICIENT"

    win_votes = votes_l if action == 1 else (votes_s if action == 2 else max(votes_l, votes_s))
    conf = int(max(0.0, min(100.0, 100.0 * (float(win_votes) / max(k, 1)))))
    kelly = 0.0 if action == 0 else float(min(1.0, max(0.0, float(base_kelly) * (0.40 + 0.60 * (float(win_votes) / max(k, 1))))))

    return {
        "decision": _action_text(action),
        "action": int(action),
        "confidence_score": int(conf),
        "kelly_weight": float(kelly),
        "reason": str(reason),
        "votes_long": int(votes_l),
        "votes_short": int(votes_s),
        "votes_hold": int(votes_h),
        "pool_k": int(k),
        "required_votes": int(min_votes),
        "long_score_avg": float(ls_sum / max(k, 1)),
        "short_score_avg": float(ss_sum / max(k, 1)),
        "blocked_by_tradable": int(blocked_by_tradable),
        "blocked_by_fire": int(blocked_by_fire),
    }


def _build_ensemble_runtime(pb_list: list[dict], base_action: int, base_kelly: float, ms: dict | None = None, tr: dict | None = None) -> dict:
    now_iso = _now_kst_iso()
    ms = dict(ms or {})
    _update_overheat_score(ms)
    tr = dict(tr or {})
    _trk_state = _load_ensemble_tracker_state()
    bal_pos = str(((_trk_state.get("balanced", {}) or {}).get("pos", "NONE"))).upper()
    low_pos = str(((_trk_state.get("lowfreq", {}) or {}).get("pos", "NONE"))).upper()
    by = {}
    for x in (pb_list or []):
        n = str((x or {}).get("name", ""))
        if n:
            by[n] = dict(x or {})

    veto = by.get("PB_VETO_SHIELD")
    # 소프트 점수만으로 강제 HOLD하지 않고, 실제 매칭 시에만 VETO를 건다.
    veto_on = bool((veto or {}).get("matched", False))
    static_cards = _load_ensemble_cards()
    bal_params = _load_param_pool(ENSEMBLE_BALANCED_PARAMS_PATH, "top_params")
    low_params = _load_param_pool(ENSEMBLE_LOWFREQ_PARAMS_PATH, "top10_singles")
    bal_min_votes = int((_read_json_safe(ENSEMBLE_BALANCED_PARAMS_PATH).get("search", {}) or {}).get("min_votes", 6) or 6)
    low_min_votes = int((_read_json_safe(ENSEMBLE_LOWFREQ_METRICS_PATH).get("best", {}) or {}).get("votes", 7) or 7)
    balanced_live = _ensemble_vote_runtime(
        params=bal_params[:10],
        min_votes=bal_min_votes,
        ms=ms,
        tr=tr,
        base_kelly=base_kelly,
        veto_on=veto_on,
        tag="BALANCED",
        current_pos=bal_pos,
    )
    lowfreq_live = _ensemble_vote_runtime(
        params=(low_params[:10] if low_params else bal_params[:10]),
        min_votes=low_min_votes,
        ms=ms,
        tr=tr,
        base_kelly=base_kelly,
        veto_on=veto_on,
        tag="LOWFREQ",
        current_pos=low_pos,
    )
    balanced_live["updated_at"] = str(now_iso)
    lowfreq_live["updated_at"] = str(now_iso)
    return {
        "updated_at": now_iso,
        "balanced": {
            **dict(static_cards.get("balanced", {}) or {}),
            "live": balanced_live,
        },
        "lowfreq": {
            **dict(static_cards.get("lowfreq", {}) or {}),
            "live": lowfreq_live,
        },
    }


def _default_tracker_state() -> dict:
    now = _now_kst_iso()
    return {
        "balanced": {
            "pos": "NONE",
            "entry_price": 0.0,
            "entry_kelly": 0.0,
            "opened_at": "",
            "unrealized_pnl_pct": 0.0,
            "equity": 1.0,
            "peak_equity": 1.0,
            "mdd_pct": 0.0,
            "trades": 0,
            "wins": 0,
            "last_pnl_pct": 0.0,
            "updated_at": now,
        },
        "lowfreq": {
            "pos": "NONE",
            "entry_price": 0.0,
            "entry_kelly": 0.0,
            "opened_at": "",
            "unrealized_pnl_pct": 0.0,
            "equity": 1.0,
            "peak_equity": 1.0,
            "mdd_pct": 0.0,
            "trades": 0,
            "wins": 0,
            "last_pnl_pct": 0.0,
            "updated_at": now,
        },
    }


def _load_ensemble_tracker_state() -> dict:
    # records를 진실 소스로 사용: records가 비면 누적도 즉시 초기화
    st = _default_tracker_state()
    if not ENSEMBLE_TRACKER_ENABLED:
        return st
    try:
        if not os.path.exists(ENSEMBLE_TRACKER_RECORDS_PATH):
            return st
        if os.path.getsize(ENSEMBLE_TRACKER_RECORDS_PATH) <= 0:
            return st
        with open(ENSEMBLE_TRACKER_RECORDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                k = str(row.get("ensemble", ""))
                if k not in ("balanced", "lowfreq"):
                    continue
                n = st[k]
                ev = str(row.get("event", "")).upper()
                ts = str(row.get("ts", n.get("updated_at", _now_kst_iso())))
                if ev == "OPEN":
                    n["pos"] = str(row.get("side", "NONE")).upper()
                    n["entry_price"] = _safe_float(row.get("entry_price", 0.0), 0.0)
                    n["entry_kelly"] = _safe_float(row.get("entry_kelly", n.get("entry_kelly", 0.0)), 0.0)
                    n["unrealized_pnl_pct"] = 0.0
                    n["opened_at"] = ts
                    n["updated_at"] = ts
                elif ev == "CLOSE":
                    pnl_pct = _safe_float(row.get("pnl_pct", 0.0), 0.0)
                    eq_row = row.get("equity", None)
                    if eq_row is not None:
                        n["equity"] = _safe_float(eq_row, n.get("equity", 1.0))
                    else:
                        n["equity"] = float(n.get("equity", 1.0)) * (1.0 + pnl_pct / 100.0)
                    peak_eq = float(n.get("peak_equity", 1.0) or 1.0)
                    cur_eq = float(n.get("equity", 1.0) or 1.0)
                    peak_eq = max(peak_eq, cur_eq, 1e-12)
                    dd_pct = float((1.0 - (cur_eq / peak_eq)) * 100.0)
                    n["peak_equity"] = float(peak_eq)
                    n["mdd_pct"] = float(max(_safe_float(n.get("mdd_pct", 0.0), 0.0), dd_pct))
                    n["trades"] = int(n.get("trades", 0) or 0) + 1
                    if pnl_pct > 0.0:
                        n["wins"] = int(n.get("wins", 0) or 0) + 1
                    n["last_pnl_pct"] = float(pnl_pct)
                    n["pos"] = "NONE"
                    n["entry_price"] = 0.0
                    n["entry_kelly"] = 0.0
                    n["opened_at"] = ""
                    n["unrealized_pnl_pct"] = 0.0
                    n["updated_at"] = ts
    except Exception:
        # records 파싱 실패 시 기존 state 파일 fallback
        raw = _read_json_safe(ENSEMBLE_TRACKER_STATE_PATH)
        if raw:
            for k in ("balanced", "lowfreq"):
                node = dict(raw.get(k, {}) or {})
                if node:
                    st[k].update(node)
    return st


def _save_ensemble_tracker_state(state: dict) -> None:
    _atomic_write_json(ENSEMBLE_TRACKER_STATE_PATH, state)


def _close_tracker_trade(node: dict, name: str, now_iso: str, price: float) -> None:
    pos = str(node.get("pos", "NONE"))
    if pos not in ("LONG", "SHORT"):
        return
    entry = float(node.get("entry_price", 0.0) or 0.0)
    if entry <= 0.0 or price <= 0.0:
        node["pos"] = "NONE"
        node["entry_price"] = 0.0
        return

    slip = float(ENSEMBLE_TRACKER_SLIP_RATE)
    fee = float(ENSEMBLE_TRACKER_FEE_RATE)
    exit_px = price * (1.0 - slip if pos == "LONG" else 1.0 + slip)
    rr = (exit_px - entry) / max(entry, 1e-12)
    if pos == "SHORT":
        rr = -rr
    pnl_frac = float(rr - (2.0 * fee))

    eq = float(node.get("equity", 1.0) or 1.0)
    eq *= (1.0 + pnl_frac)
    node["equity"] = float(eq)
    peak_eq = float(node.get("peak_equity", 1.0) or 1.0)
    peak_eq = max(peak_eq, eq, 1e-12)
    dd_pct = float((1.0 - (eq / peak_eq)) * 100.0)
    node["peak_equity"] = float(peak_eq)
    node["mdd_pct"] = float(max(_safe_float(node.get("mdd_pct", 0.0), 0.0), dd_pct))
    node["trades"] = int(node.get("trades", 0) or 0) + 1
    if pnl_frac > 0:
        node["wins"] = int(node.get("wins", 0) or 0) + 1
    node["last_pnl_pct"] = float(pnl_frac * 100.0)
    node["updated_at"] = now_iso

    _append_jsonl(ENSEMBLE_TRACKER_RECORDS_PATH, {
        "ts": now_iso,
        "ensemble": name,
        "event": "CLOSE",
        "side": pos,
        "entry_price": entry,
        "exit_price": float(exit_px),
        "pnl_pct": float(pnl_frac * 100.0),
        "equity": float(eq),
    })
    node["pos"] = "NONE"
    node["entry_price"] = 0.0
    node["entry_kelly"] = 0.0
    node["opened_at"] = ""
    node["unrealized_pnl_pct"] = 0.0


def _open_tracker_trade(node: dict, name: str, now_iso: str, price: float, action: int, kelly: float = 0.0) -> None:
    if price <= 0.0 or int(action) not in (1, 2):
        return
    side = "LONG" if int(action) == 1 else "SHORT"
    slip = float(ENSEMBLE_TRACKER_SLIP_RATE)
    entry = price * (1.0 + slip if side == "LONG" else 1.0 - slip)
    node["pos"] = side
    node["entry_price"] = float(entry)
    node["entry_kelly"] = float(max(0.0, kelly))
    node["unrealized_pnl_pct"] = 0.0
    node["opened_at"] = str(now_iso)
    node["updated_at"] = now_iso
    _append_jsonl(ENSEMBLE_TRACKER_RECORDS_PATH, {
        "ts": now_iso,
        "ensemble": name,
        "event": "OPEN",
        "side": side,
        "entry_price": float(entry),
        "entry_kelly": float(max(0.0, kelly)),
    })


def _update_ensemble_tracker(ensembles: dict, current_price: float, now_iso: str) -> dict:
    if not ENSEMBLE_TRACKER_ENABLED:
        st = _default_tracker_state()
        _save_ensemble_tracker_state(st)
        return st

    st = _load_ensemble_tracker_state()
    price = float(current_price or 0.0)

    for key in ("balanced", "lowfreq"):
        node = dict(st.get(key, {}) or {})
        live = dict((ensembles.get(key, {}) or {}).get("live", {}) or {})
        action = int(live.get("action", 0) or 0)
        live_kelly = _safe_float(live.get("kelly_weight", 0.0), 0.0)
        cur_pos = str(node.get("pos", "NONE"))

        # 과거 포맷(OPEN에 entry_kelly 없음)로 열린 포지션 복구:
        # 포지션이 살아있고 entry_kelly가 비어있으면 현재 live_kelly를 1회 백필
        if cur_pos in ("LONG", "SHORT") and _safe_float(node.get("entry_kelly", 0.0), 0.0) <= 0.0 and live_kelly > 0.0:
            node["entry_kelly"] = float(live_kelly)
            node["updated_at"] = now_iso

        if cur_pos in ("LONG", "SHORT"):
            entry = float(node.get("entry_price", 0.0) or 0.0)
            if price > 0.0 and entry > 0.0:
                mark_px = price * (1.0 - ENSEMBLE_TRACKER_SLIP_RATE if cur_pos == "LONG" else 1.0 + ENSEMBLE_TRACKER_SLIP_RATE)
                rr = (mark_px - entry) / max(entry, 1e-12)
                if cur_pos == "SHORT":
                    rr = -rr
                node["unrealized_pnl_pct"] = float((rr - (2.0 * ENSEMBLE_TRACKER_FEE_RATE)) * 100.0)
            else:
                node["unrealized_pnl_pct"] = 0.0

            pos_action = 1 if cur_pos == "LONG" else 2
            should_close = False
            if action in (1, 2) and action != pos_action:
                # 명시적 반대 시그널일 때만 청산/전환
                should_close = True
            elif action == 0 and ENSEMBLE_TRACKER_EXIT_ON_HOLD:
                # 옵션: HOLD 시 즉시 청산(기본 OFF)
                should_close = True
            if should_close:
                _close_tracker_trade(node=node, name=key, now_iso=now_iso, price=price)
                if action in (1, 2):
                    _open_tracker_trade(node=node, name=key, now_iso=now_iso, price=price, action=action, kelly=live_kelly)
        else:
            node["unrealized_pnl_pct"] = 0.0
            if action in (1, 2):
                _open_tracker_trade(node=node, name=key, now_iso=now_iso, price=price, action=action, kelly=live_kelly)

        st[key] = node

    _save_ensemble_tracker_state(st)
    return st


def _ensemble_tracker_summary(tracker_state: dict) -> dict:
    out = {}
    for key in ("balanced", "lowfreq"):
        n = dict((tracker_state or {}).get(key, {}) or {})
        eq = float(n.get("equity", 1.0) or 1.0)
        tr = int(n.get("trades", 0) or 0)
        wins = int(n.get("wins", 0) or 0)
        wr = (100.0 * wins / tr) if tr > 0 else 0.0
        out[key] = {
            "total_return_pct": float((eq - 1.0) * 100.0),
            "trades": tr,
            "win_rate": float(wr),
            "pos": str(n.get("pos", "NONE")),
            "entry_price": float(n.get("entry_price", 0.0) or 0.0),
            "entry_kelly": float(n.get("entry_kelly", 0.0) or 0.0),
            "opened_at": str(n.get("opened_at", "")),
            "unrealized_pnl_pct": float(n.get("unrealized_pnl_pct", 0.0) or 0.0),
            "last_pnl_pct": float(n.get("last_pnl_pct", 0.0) or 0.0),
            "mdd_pct": float(n.get("mdd_pct", 0.0) or 0.0),
            "updated_at": str(n.get("updated_at", _now_kst_iso())),
        }
    return out




def _confidence_from_std(std: float) -> float:
    s = max(float(std), 1e-6)
    return float(1.0 / (1.0 + s))


def _norm_tanh(x: float, scale: float) -> float:
    s = max(float(scale), 1e-8)
    return float(np.tanh(float(x) / s))


def _regime_signed(regime: dict[str, float] | None) -> float:
    if not isinstance(regime, dict):
        return 0.0
    if float(regime.get("regime_bull", 0.0)) >= 0.5:
        return 1.0
    if float(regime.get("regime_bear", 0.0)) >= 0.5:
        return -1.0
    return 0.0


def _trend_from_row(row: pd.Series | dict) -> tuple[float, float]:
    get = row.get if hasattr(row, "get") else lambda k, d=0.0: d
    mtf_1h = float(get("mtf_trend_1h", 0.0) or 0.0)
    mtf_4h = float(get("mtf_trend_4h", 0.0) or 0.0)
    trend_strength = float(np.clip(0.5 * (abs(mtf_1h) + abs(mtf_4h)), 0.0, 1.0))
    signed = float(np.sign(mtf_1h + 0.75 * mtf_4h))
    return signed, trend_strength


# ════════════════════════════════════════════════════════════════
# 1. Binance data/execution adapters live in dedicated modules
# ════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════
# 2-A. Final Governor AI feature suppliers
# ════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════
# 2-D. 텔레그램 알림
# ════════════════════════════════════════════════════════════════
class TelegramNotifier:
    _API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self):
        self.token   = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self._ok     = bool(self.token and self.chat_id)
        if not self._ok:
            logger.warning("⚠️ 텔레그램 미설정 — TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 환경변수 필요")

    def _do_send(self, text: str) -> None:
        import urllib.request as _ur
        import urllib.error as _ue
        import json as _json
        url  = self._API.format(token=self.token)
        body = _json.dumps({'chat_id': self.chat_id, 'text': text,
                            'parse_mode': 'HTML'}).encode()
        req  = _ur.Request(url, data=body,
                           headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with _ur.urlopen(req, timeout=8) as r:
                raw = r.read().decode('utf-8', errors='ignore')
            logger.info("📨 텔레그램 전송 완료")
        except Exception as e:
            logger.warning(f"⚠️ 텔레그램 전송 예외: {e}")

    async def notify(self, text: str) -> None:
        if not self._ok:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._do_send, text)


def _tg_trade_msg(ex_code: str, current_price: float,
                  timestamp_kst, regime_name: str, meta_result: dict) -> str:
    fa    = int(meta_result.get('final_action', 0))
    kelly = float(meta_result.get('unified_kelly', 0.0))
    ts_   = meta_result.get('trend_signal') or {}
    t_dir = {0: '▼ DOWN', 1: '─ FLAT', 2: '▲ UP'}.get(int(ts_.get('trend_dir', 1)), '?')
    icon  = {
        'ENTER_LONG':           '🟩',
        'ENTER_SHORT':          '🟥',
        'EXIT_LONG':            '⬜',
        'EXIT_SHORT':           '⬜',
        'FLIP_LONG_TO_SHORT':   '🔄',
        'FLIP_SHORT_TO_LONG':   '🔄',
    }.get(ex_code, '🟨')
    action_word = {
        'ENTER_LONG': 'LONG',
        'ENTER_SHORT': 'SHORT',
        'EXIT_LONG': 'FLAT',
        'EXIT_SHORT': 'FLAT',
        'FLIP_LONG_TO_SHORT': 'SHORT',
        'FLIP_SHORT_TO_LONG': 'LONG',
    }.get(ex_code, {1: 'LONG', 2: 'SHORT', 0: 'HOLD'}.get(fa, '?'))
    pnl_line = ""
    trade_pnl = meta_result.get("trade_pnl_pct", None) if isinstance(meta_result, dict) else None
    if trade_pnl is not None:
        try:
            p = float(trade_pnl)
            p_icon = "🟢" if p > 0 else ("🔴" if p < 0 else "🟨")
            pnl_line = f"\n{p_icon} Event PnL: {p:+.2f}%"
        except Exception:
            pass
    elif ex_code.startswith("ENTER_"):
        pnl_line = "\n🟨 Event PnL: +0.00% (entry)"
    return (
        f"{icon} <b>{ex_code}</b>  ({action_word})\n"
        f"💰 ETH ${current_price:,.2f}   🕐 {timestamp_kst.strftime('%m-%d %H:%M')} KST\n"
        f"🌍 {regime_name}   Kelly: {kelly:.3f}{pnl_line}\n"
        f"📈 Trend: {t_dir}   Source: {meta_result.get('source', 'FINAL_GOVERNOR')}"
    )

def _load_trade_journal_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return []
    return rows


def _trade_rows_for_kst_day(rows: list[dict], day_kst: pd.Timestamp) -> list[dict]:
    base = pd.Timestamp(day_kst)
    if base.tzinfo is None:
        base = base.tz_localize("Asia/Seoul")
    else:
        base = base.tz_convert("Asia/Seoul")
    start = base.normalize()
    end = start + pd.Timedelta(days=1)
    out: list[dict] = []
    for row in rows:
        ts_raw = row.get("ts") or row.get("closed_at") or row.get("opened_at")
        try:
            ts = pd.Timestamp(ts_raw)
            if ts.tzinfo is None:
                ts = ts.tz_localize("Asia/Seoul")
            else:
                ts = ts.tz_convert("Asia/Seoul")
        except Exception:
            continue
        if start <= ts < end:
            out.append(dict(row))
    return out


def _fmt_trade_journal_line(row: dict) -> str:
    event = str(row.get("event", "") or row.get("kind", "") or "TRADE")
    side = str(row.get("side", "") or "NONE")
    pnl_pct = row.get("pnl_pct", None)
    hold_bars = int(float(row.get("hold_bars", 0) or 0))
    entry_price = float(row.get("entry_price", 0.0) or 0.0)
    exit_price = float(row.get("exit_price", 0.0) or 0.0)
    exposure = float(row.get("total_exposure", 0.0) or 0.0)
    regime = str(row.get("regime", "") or "-")
    reason = str(row.get("reason", "") or row.get("source", "") or "-")
    reason = html.escape(reason[:120] + ("..." if len(reason) > 120 else ""))
    ts_raw = row.get("closed_at") or row.get("opened_at") or row.get("ts")
    try:
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Seoul")
        else:
            ts = ts.tz_convert("Asia/Seoul")
        ts_txt = ts.strftime("%H:%M")
    except Exception:
        ts_txt = "--:--"
    pnl_txt = "-"
    if pnl_pct is not None:
        try:
            pnl_txt = f"{float(pnl_pct):+.2f}%"
        except Exception:
            pass
    px_txt = ""
    if entry_price > 0.0:
        px_txt = f" @{entry_price:,.2f}"
        if exit_price > 0.0:
            px_txt += f" -> {exit_price:,.2f}"
    return (
        f"• <b>{html.escape(ts_txt)} {html.escape(event)}</b> [{html.escape(side)}] {pnl_txt}\n"
        f"  exp={exposure:.3f} hold={hold_bars}b regime={html.escape(regime)}{px_txt}\n"
        f"  reason={reason}"
    )


def _build_daily_trade_journal_message(report_day_kst: pd.Timestamp, rows: list[dict]) -> str:
    close_rows = [r for r in rows if str(r.get("kind", "")).upper() == "CLOSE"]
    open_rows = [r for r in rows if str(r.get("kind", "")).upper() == "OPEN"]
    pnl_list: list[float] = []
    for row in close_rows:
        try:
            pnl_list.append(float(row.get("pnl_pct", 0.0) or 0.0))
        except Exception:
            pnl_list.append(0.0)
    total = float(sum(pnl_list))
    wins = int(sum(1 for x in pnl_list if x > 0.0))
    losses = int(sum(1 for x in pnl_list if x < 0.0))
    win_rate = (100.0 * wins / len(close_rows)) if close_rows else 0.0
    avg = (total / len(close_rows)) if close_rows else 0.0
    best = max(pnl_list) if pnl_list else 0.0
    worst = min(pnl_list) if pnl_list else 0.0
    avg_hold = (
        sum(int(float(r.get("hold_bars", 0) or 0)) for r in close_rows) / len(close_rows)
        if close_rows else 0.0
    )
    day_txt = pd.Timestamp(report_day_kst).strftime("%Y-%m-%d")
    lines = [
        f"<b>Daily Trade Journal</b>  {html.escape(day_txt)} KST",
        f"• closes={len(close_rows)} opens={len(open_rows)}",
        f"• pnl={total:+.2f}% avg={avg:+.2f}% winrate={win_rate:.1f}%",
        f"• wins={wins} losses={losses} best={best:+.2f}% worst={worst:+.2f}%",
        f"• avg_hold={avg_hold:.1f} bars",
    ]
    if not close_rows:
        lines.append("• no closed trades")
        return "\n".join(lines)
    detail_rows = sorted(close_rows, key=lambda r: str(r.get("closed_at") or r.get("ts") or ""))[-12:]
    lines.extend(["", "<b>Closed Trades</b>"])
    lines.extend(_fmt_trade_journal_line(r) for r in detail_rows)
    return "\n".join(lines)


def _last_daily_report_date(path: str) -> str:
    return str(_read_json_safe(path).get("last_report_date", "") or "")


def _save_daily_report_date(path: str, report_day_kst: pd.Timestamp) -> None:
    _atomic_write_json(path, {
        "last_report_date": pd.Timestamp(report_day_kst).strftime("%Y-%m-%d"),
        "updated_at": _now_kst_iso(),
    })


def _compute_regime(df, window=24):
    regime_cols = ['regime_bull', 'regime_bear', 'regime_chop', 'regime_whipsaw', 'regime_normal']
    if all(col in df.columns for col in regime_cols):
        last = df.iloc[-1]
        vals = {col: float(last.get(col, 0.0)) for col in regime_cols}
        if any(np.isfinite(v) and abs(v) > 1e-8 for v in vals.values()):
            best_col = max(regime_cols, key=lambda c: vals[c])
            return {col: (1.0 if col == best_col else 0.0) for col in regime_cols}

    close = pd.to_numeric(df['close'], errors='coerce').ffill()
    ret = close.pct_change()
    diff_abs = close.diff().abs()
    net_24 = close - close.shift(24)
    net_48 = close - close.shift(48)
    er_24 = (net_24.abs() / (diff_abs.rolling(24, min_periods=4).sum() + 1e-12)).fillna(0)
    er_48 = (net_48.abs() / (diff_abs.rolling(48, min_periods=8).sum() + 1e-12)).fillna(0)
    ret_48 = (close / close.shift(48) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0)
    raw_vol = ret.rolling(24, min_periods=4).std().fillna(0)
    vol_z = (
        (raw_vol - raw_vol.rolling(288, min_periods=24).mean().ffill().fillna(0))
        / (raw_vol.rolling(288, min_periods=24).std().ffill().fillna(0) + 1e-8)
    )
    if "mtf_trend_1h" in df.columns:
        mtf = pd.to_numeric(df["mtf_trend_1h"], errors="coerce").fillna(0.0)
    else:
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        mtf = ((ema12 - ema26) / (ema26 + 1e-8)).fillna(0.0)
    ret_sign = np.sign(ret.where(ret.abs() >= 1e-8, np.nan)).ffill().fillna(0.0)
    sign_flip_24 = (
        (ret_sign != ret_sign.shift(1))
        .astype(float)
        .rolling(24, min_periods=4)
        .mean()
        .fillna(0.0)
    )

    er24_v = float(er_24.iloc[-1]) if er_24.notna().iloc[-1] else 0.0
    er48_v = float(er_48.iloc[-1]) if er_48.notna().iloc[-1] else 0.0
    volz_v = float(vol_z.iloc[-1]) if vol_z.notna().iloc[-1] else 0.0
    n48_v = float(net_48.iloc[-1]) if net_48.notna().iloc[-1] else 0.0
    r48_v = float(ret_48.iloc[-1]) if ret_48.notna().iloc[-1] else 0.0
    mtf_v = float(mtf.iloc[-1]) if mtf.notna().iloc[-1] else 0.0
    flip_v = float(sign_flip_24.iloc[-1]) if sign_flip_24.notna().iloc[-1] else 0.0

    trend = er24_v >= 0.20 or er48_v >= 0.16
    bull = trend and n48_v > 0 and mtf_v > 0
    bear = trend and n48_v < 0 and mtf_v < 0
    if (not bear) and n48_v > 0 and mtf_v > 0.00015 and er48_v >= 0.08 and r48_v > 0.0015:
        bull = True
    if (not bull) and n48_v < 0 and mtf_v < -0.00015 and er48_v >= 0.08 and r48_v < -0.0015:
        bear = True
    whip = (not bull) and (not bear) and volz_v > 0.5 and er24_v < 0.18 and flip_v > 0.52
    chop = (
        (not bull)
        and (not bear)
        and (not whip)
        and volz_v < -0.5
        and er24_v < 0.14
        and er48_v < 0.14
        and abs(mtf_v) < 0.0005
    )
    norm = not (bull or bear or chop or whip)
    return {
        'regime_bull': 1.0 if bull else 0.0, 'regime_bear': 1.0 if bear else 0.0,
        'regime_chop': 1.0 if chop else 0.0, 'regime_whipsaw': 1.0 if whip else 0.0,
        'regime_normal': 1.0 if norm else 0.0,
    }


def _pos_transition_label(prev_pos: str | None, cur_pos: str | None) -> str:
    if prev_pos == cur_pos:
        if cur_pos is None: return 'STAY FLAT'
        return f'HOLD {cur_pos}'
    if prev_pos is None and cur_pos is not None:
        return f'ENTER {cur_pos}'
    if prev_pos is not None and cur_pos is None:
        return f'EXIT {prev_pos}'
    return f'FLIP {prev_pos}->{cur_pos}'


def _session_flags_from_timestamp(ts) -> dict[str, float]:
    ts_kst = pd.Timestamp(ts)
    if ts_kst.tzinfo is None:
        ts_kst = ts_kst.tz_localize("Asia/Seoul")
    else:
        ts_kst = ts_kst.tz_convert("Asia/Seoul")
    ts_utc = ts_kst.tz_convert("UTC")
    try:
        import pandas_market_calendars as mcal
        day = ts_utc.date()
        flags = {}
        for name, cal_name in (("session_asia", "JPX"), ("session_europe", "LSE"), ("session_us", "NYSE")):
            cal = mcal.get_calendar(cal_name)
            sched = cal.schedule(start_date=day, end_date=day)
            active = False
            if not sched.empty:
                row = sched.iloc[0]
                ts_min = ts_utc.floor("min")
                market_open = pd.Timestamp(row.get("market_open"))
                market_close = pd.Timestamp(row.get("market_close"))
                break_start = row.get("break_start", pd.NaT)
                break_end = row.get("break_end", pd.NaT)
                in_main = bool(market_open <= ts_min <= market_close)
                in_break = False
                if pd.notna(break_start) and pd.notna(break_end):
                    break_start = pd.Timestamp(break_start)
                    break_end = pd.Timestamp(break_end)
                    in_break = bool(break_start <= ts_min < break_end)
                active = bool(in_main and not in_break)
            flags[name] = 1.0 if active else 0.0
        return flags
    except Exception:
        hour = ts_utc.hour + (ts_utc.minute / 60.0)
        return {
            "session_asia": 1.0 if 0.0 <= hour < 8.0 else 0.0,
            "session_europe": 1.0 if 8.0 <= hour < 16.0 else 0.0,
            "session_us": 1.0 if 14.5 <= hour < 21.0 else 0.0,
        }


def _action_word(a: int) -> str:
    return {0: "HOLD", 1: "LONG", 2: "SHORT"}.get(int(a), "UNKNOWN")


def _color(text: object, color: str) -> str:
    if not CONSOLE_LOG_COLOR:
        return str(text)
    return f"{color}{text}{Colors.RESET}"


def _log_tag(text: str, color: str) -> str:
    return _color(f"[{text}]", color)


def _compact_status(ok: bool, ok_label: str = "OK", bad_label: str = "BAD") -> str:
    return _color(ok_label, Colors.GREEN) if bool(ok) else _color(bad_label, Colors.RED)


def _colored_action(a: int) -> str:
    word = _action_word(int(a))
    color = {0: Colors.YELLOW, 1: Colors.GREEN, 2: Colors.RED}.get(int(a), Colors.RESET)
    return _color(word, color)


def _colored_pos(pos: str | None) -> str:
    pos_u = str(pos or "NONE").upper()
    color = Colors.GREEN if pos_u == "LONG" else (Colors.RED if pos_u == "SHORT" else Colors.YELLOW)
    return _color(pos_u, color)


def _colored_pnl(value: float) -> str:
    color = Colors.GREEN if value > 0 else (Colors.RED if value < 0 else Colors.YELLOW)
    return _color(f"{value:+.2f}%", color)


def _colored_regime(regime: str) -> str:
    reg = str(regime or "UNKNOWN").upper()
    color = {
        "BULL": Colors.GREEN,
        "BEAR": Colors.RED,
        "CHOP": Colors.BLUE,
        "WHIPSAW": Colors.MAGENTA,
        "NORMAL": Colors.CYAN,
    }.get(reg, Colors.CYAN)
    return _color(reg, color)


def _fmt_age_sec(v) -> str:
    try:
        if v is None:
            return "na"
        x = float(v)
        if not np.isfinite(x):
            return "na"
        if x < 90:
            return f"{x:.0f}s"
        return f"{x / 60.0:.1f}m"
    except Exception:
        return "na"




def _compact_source(src: str, max_len: int = 44) -> str:
    txt = str(src or "")
    return txt if len(txt) <= max_len else txt[: max_len - 1] + "…"


def _log_compact_ai_decision(
    *,
    timestamp_kst,
    current_price: float,
    regime_name: str,
    rl_action: int,
    rl_info: dict,
    meta_result: dict,
    prev_pos: str | None,
    cur_pos: str | None,
) -> None:
    fa = int(meta_result.get("final_action", 0))
    transition = _pos_transition_label(prev_pos, cur_pos)
    trend = dict(meta_result.get("trend_signal") or {})
    reg_pred = dict(
        meta_result.get("regime_predictor")
        or (dict(rl_info.get("sleeve_trace", {}) or {}).get("regime_predictor", {}) if isinstance(rl_info, dict) else {})
        or {}
    )
    reg_pred_part = ""
    if bool(reg_pred.get("enabled", False)):
        try:
            reg_pred_part = (
                f" | reg45={reg_pred.get('previous_label', '-')}>{reg_pred.get('pred_label', '-')}"
                f" conf={float(reg_pred.get('confidence', 0.0) or 0.0):.2f}"
            )
        except Exception:
            reg_pred_part = ""
    pos = str(cur_pos or "NONE")
    unrl = None
    try:
        if meta_result.get("position_unrealized_pnl_pct") is not None:
            unrl = float(meta_result.get("position_unrealized_pnl_pct", 0.0) or 0.0)
        elif cur_pos and float(meta_result.get("position_entry_price", 0.0) or 0.0) > 0.0:
            entry = float(meta_result.get("position_entry_price", 0.0))
            sign = 1.0 if cur_pos == "LONG" else -1.0
            unrl = sign * ((float(current_price) - entry) / max(abs(entry), 1e-8)) * 100.0
    except Exception:
        unrl = None
    logger.info(
        "%s %s px=%s regime=%s action=%s signal=%s pos=%s trans=%s exp=%.2f frac=%.2f lev=%.2fx | "
        "owner=%s sleeve=%s | trend U/D=%.0f/%.0f score=%+.3f conv=%.3f | "
        "reason=%s%s%s %s",
        _log_tag("AI", Colors.MAGENTA),
        _color(pd.Timestamp(timestamp_kst).strftime("%H:%M"), Colors.DIM),
        _color(f"{float(current_price):,.2f}", Colors.CYAN),
        _colored_regime(str(regime_name)),
        _colored_action(fa),
        _colored_action(int(rl_action)),
        _colored_pos(pos),
        _color(transition.replace(" ", "_"), Colors.BLUE),
        float(meta_result.get("unified_kelly", 0.0) or 0.0),
        float(meta_result.get("position_fraction", 0.0) or 0.0),
        float(meta_result.get("execution_leverage", 1.0) or 1.0),
        _color(_compact_source(str(rl_info.get("owner", "-") or "-"), 24), Colors.CYAN),
        _color(_compact_source(str(rl_info.get("source", meta_result.get("source", "-")) or "-"), 36), Colors.DIM),
        float(trend.get("prob_up", 0.0) or 0.0) * 100.0,
        float(trend.get("prob_dn", 0.0) or 0.0) * 100.0,
        float(meta_result.get("rl_score", 0.0) or 0.0),
        float(rl_info.get("conviction", 0.0) or 0.0),
        _color(_compact_source(str(meta_result.get("hold_reason") or meta_result.get("block_reason") or meta_result.get("position_reason") or "-"), 64), Colors.YELLOW),
        f" unrl={unrl:+.2f}%" if (cur_pos and unrl is not None) else "",
        _color(reg_pred_part, Colors.CYAN) if reg_pred_part else "",
        _color(_compact_source(str(meta_result.get("source", ""))), Colors.DIM),
    )
    if prev_pos != cur_pos or meta_result.get("trade_pnl_pct") is not None:
        pnl_pct = float(meta_result.get("trade_pnl_pct", 0.0) or 0.0)
        logger.info(
            "%s event=%s pnl=%s pos=%s px=%s %s",
            _log_tag("TRADE", Colors.BLUE),
            _color(transition.replace(" ", "_"), Colors.BLUE),
            _colored_pnl(pnl_pct),
            _colored_pos(pos),
            _color(f"{float(current_price):,.2f}", Colors.CYAN),
            _color(_compact_source(str(meta_result.get("source", ""))), Colors.DIM),
        )


def _log_compact_data_health(
    *,
    state: dict,
    ms: dict,
    tr: dict,
    dashboard_ok: bool,
    compact_dashboard_ok: bool,
) -> None:
    micro_age = _file_age_sec(QUANT_MICRO_DB_PATH)
    tail_age = _file_age_sec(QUANT_TAIL_DB_PATH)
    dash_age = _file_age_sec(DASHBOARD_STATE_PATH)
    compact_age = _file_age_sec(COMPACT_DASHBOARD_STATE_PATH)
    logger.info(
        "%s store dash=%s(%s) compact=%s(%s) db=%s/%s | "
        "%s %s stale=%s ws(D/T/P)=%s/%s/%s age=%s/%s/%s flow=%s nif=%s trades5m=%d ntl=%.2fM whales=%d | "
        "%s stream=%s ws=%s liq1m=%d L/S=%.2fM/%.2fM after=%.2f bucket=%s | "
        "px=%s pos=%s",
        _log_tag("DATA", Colors.CYAN),
        _compact_status(dashboard_ok),
        _fmt_age_sec(dash_age),
        _compact_status(compact_dashboard_ok),
        _fmt_age_sec(compact_age),
        _color(f"micro:{_fmt_age_sec(micro_age)}", Colors.CYAN),
        _color(f"tail:{_fmt_age_sec(tail_age)}", Colors.CYAN),
        _log_tag("MS", Colors.MAGENTA),
        _compact_status(not bool(ms.get("data_stale", True)), "LIVE", "STALE"),
        _color(str(bool(ms.get("data_stale", True))), Colors.YELLOW),
        _compact_status(ms.get("depth_connected", False)),
        _compact_status(ms.get("trade_connected", False)),
        _compact_status(ms.get("poll_connected", False)),
        _fmt_age_sec(ms.get("depth_age_sec")),
        _fmt_age_sec(ms.get("trade_age_sec")),
        _fmt_age_sec(ms.get("poll_age_sec")),
        _compact_status(ms.get("valid_taker_flow", False)),
        _compact_status(ms.get("valid_nif", False)),
        int(ms.get("recent_trade_count_5m", 0) or 0),
        float(ms.get("recent_trade_notional_5m", 0.0) or 0.0) / 1e6,
        int(ms.get("recent_whale_count_5m", 0) or 0),
        _log_tag("TR", Colors.BLUE),
        _compact_status(tr.get("valid_liq_stream", False)),
        _compact_status(tr.get("ws_connected", False)),
        int(tr.get("liq_event_count_1m", 0) or 0),
        float(tr.get("long_usd_1m", 0.0) or 0.0) / 1e6,
        float(tr.get("short_usd_1m", 0.0) or 0.0) / 1e6,
        float(tr.get("shadow_aftershock_prob", tr.get("aftershock_prob", 0.0)) or 0.0),
        _color(str(tr.get("shadow_risk_bucket", tr.get("risk_bucket", "normal")) or "normal"), Colors.YELLOW),
        _color(f"{float(state.get('price', 0.0) or 0.0):,.2f}", Colors.CYAN),
        _colored_pos(str((state.get("position") or {}).get("current", "NONE") or "NONE")),
    )


def _health_ts_utc(ts):
    try:
        out = pd.Timestamp(ts)
        if out.tzinfo is None:
            return out.tz_localize("UTC")
        return out.tz_convert("UTC")
    except Exception:
        return None


def _health_ts_kst_text(ts) -> str:
    out = _health_ts_utc(ts)
    if out is None:
        return ""
    return str(out.tz_convert("Asia/Seoul").tz_localize(None))


def _health_frame_stats(df: pd.DataFrame | None) -> dict:
    if df is None or len(df) == 0:
        return {"rows": 0, "last_ts": "", "age_sec": None, "last_gap_sec": None, "dup_ts": 0}
    out = {"rows": int(len(df)), "last_ts": "", "age_sec": None, "last_gap_sec": None, "dup_ts": 0}
    if "timestamp" not in df.columns:
        return out
    try:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        valid = ts.dropna()
        if len(valid):
            last_utc = _health_ts_utc(valid.iloc[-1])
            if last_utc is not None:
                out["last_ts"] = str(last_utc.tz_convert("Asia/Seoul").tz_localize(None))
                out["age_sec"] = float((pd.Timestamp.utcnow() - last_utc).total_seconds())
            if len(valid) >= 2:
                out["last_gap_sec"] = float((valid.iloc[-1] - valid.iloc[-2]).total_seconds())
        out["dup_ts"] = int(ts.duplicated().sum())
    except Exception:
        pass
    return out


def _health_numeric_quality(df: pd.DataFrame | None) -> dict:
    if df is None or len(df) == 0:
        return {"last_nan": 0, "last_inf": 0, "tail_nan_ratio": 1.0, "numeric_cols": 0}
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {"last_nan": 0, "last_inf": 0, "tail_nan_ratio": 1.0, "numeric_cols": 0}
    last = numeric.tail(1).replace([np.inf, -np.inf], np.nan)
    tail = numeric.tail(min(len(numeric), 120)).replace([np.inf, -np.inf], np.nan)
    raw_last = numeric.tail(1).to_numpy(dtype=np.float64, copy=False)
    return {
        "last_nan": int(last.isna().sum(axis=1).iloc[-1]),
        "last_inf": int(np.isinf(raw_last).sum()),
        "tail_nan_ratio": float(tail.isna().sum().sum() / max(1, tail.shape[0] * tail.shape[1])),
        "numeric_cols": int(numeric.shape[1]),
    }


def _build_data_pipeline_health(
    *,
    raw_processed_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    raw_eth_buffer: pd.DataFrame,
    eth_buffer: pd.DataFrame,
    signal_bar,
    execution_bar,
    next_open_execution: bool,
    decision_price: float,
    execution_price: float,
    active_info: dict,
    runtime_predictor,
    final_governor,
    ai_errors: list,
) -> dict:
    raw_eth_stats = _health_frame_stats(raw_eth_buffer)
    signal_eth_stats = _health_frame_stats(eth_buffer)
    raw_proc_stats = _health_frame_stats(raw_processed_df)
    prepared_frame = getattr(final_governor, "last_prepared_frame_for_health", None)
    feature_df = prepared_frame if isinstance(prepared_frame, pd.DataFrame) and len(prepared_frame) else processed_df
    proc_stats = _health_frame_stats(feature_df)
    proc_quality = _health_numeric_quality(feature_df)
    signal_ts = _health_ts_utc(signal_bar.get("timestamp") if hasattr(signal_bar, "get") else None)
    exec_ts = _health_ts_utc(execution_bar.get("timestamp") if hasattr(execution_bar, "get") else None)
    proc_ts = _health_ts_utc(feature_df.iloc[-1].get("timestamp") if len(feature_df) and "timestamp" in feature_df.columns else None)
    signal_align_ok = bool(signal_ts is not None and proc_ts is not None and abs((signal_ts - proc_ts).total_seconds()) < 1.0)
    latest_age = raw_eth_stats.get("age_sec")
    latest_gap = raw_eth_stats.get("last_gap_sec")

    missing_ohlcv = [c for c in ("open", "high", "low", "close", "volume") if c not in feature_df.columns]
    proc_close = _safe_float(feature_df.iloc[-1].get("close", decision_price), decision_price) if len(feature_df) else 0.0
    price_diff_bps = abs(proc_close - float(decision_price)) / max(abs(float(decision_price)), 1e-12) * 1e4

    active_ai_cols = sorted(
        {
            col
            for group in FINAL_GOVERNOR_AI_FEATURE_GROUPS
            for col in EnsemblePredictor.AI_FEATURE_COLUMNS.get(str(group).lower(), [])
        }
    )
    ai_missing = [c for c in active_ai_cols if c not in feature_df.columns]
    ai_nonfinite = []
    ai_zero = []
    if len(feature_df):
        last = feature_df.iloc[-1]
        for col in active_ai_cols:
            if col not in feature_df.columns:
                continue
            val = _safe_float(last.get(col, np.nan), np.nan)
            if not np.isfinite(val):
                ai_nonfinite.append(col)
            elif abs(val) <= 1e-12:
                ai_zero.append(col)

    v31_payload = dict(getattr(final_governor, "v31_v27_payload", {}) or {})
    v31_seq_cols = list(v31_payload.get("seq_cols") or [])
    v31_missing = [c for c in v31_seq_cols if c not in feature_df.columns]
    v31_nonfinite = []
    if len(feature_df):
        last = feature_df.iloc[-1]
        for col in v31_seq_cols:
            if col not in feature_df.columns:
                continue
            val = _safe_float(last.get(col, np.nan), np.nan)
            if not np.isfinite(val):
                v31_nonfinite.append(col)

    regime_cols = getattr(final_governor, "REGIME_COLS", ("regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal"))
    regime_vals = {}
    if len(feature_df):
        last = feature_df.iloc[-1]
        for col in regime_cols:
            regime_vals[col.replace("regime_", "")] = _safe_float(last.get(col, 0.0), 0.0)
    raw_regime = max(regime_vals, key=regime_vals.get) if regime_vals else "unknown"

    sleeve = dict((active_info or {}).get("sleeve_trace", {}) or {})
    v31 = dict(sleeve.get("v31", {}) or {})
    alpha2 = dict(sleeve.get("alpha2_1", {}) or {})
    regime_pred = dict(sleeve.get("regime_predictor", {}) or {})
    ai_timing = dict(sleeve.get("ai_timing", {}) or {})
    ai_trace = list(sleeve.get("ai_feature_trace", []) or [])
    execution_price_source = str((active_info or {}).get("execution_price_source", "") or "")
    execution_delay_sec = _safe_float((active_info or {}).get("execution_delay_sec", 0.0), 0.0)
    execution_delay_late = bool((active_info or {}).get("execution_delay_late", False))

    warnings_list = []
    if FINAL_GOVERNOR_LIVE_PROCESS_BARS < 2200:
        warnings_list.append("process_window_lt_2016_feature_requirement")
    if latest_age is None or float(latest_age) > 660.0:
        warnings_list.append("latest_eth_bar_stale")
    if latest_gap is not None and abs(float(latest_gap) - 300.0) > 30.0:
        warnings_list.append("eth_bar_gap_not_5m")
    if int(raw_eth_stats.get("dup_ts", 0) or 0) > 0:
        warnings_list.append("duplicate_eth_timestamps")
    if not signal_align_ok:
        warnings_list.append("processed_signal_bar_misaligned")
    if price_diff_bps > 1.0:
        warnings_list.append("processed_close_differs_from_signal_close")
    if missing_ohlcv:
        warnings_list.append("missing_ohlcv_cols")
    if ai_missing or ai_nonfinite:
        warnings_list.append("ai_feature_missing_or_nonfinite")
    # v31_missing is permanently non-empty: its frozen seq_cols require m7_/regime4_pred_
    # columns that were removed from the codebase on 2026-08-09 (see docs/subagents/
    # model_architect.md). V31 already skips itself safely when features are missing
    # (trading_bot.py _v31_predict_latest) and was already inert behind Omega4.6.1's
    # priority, so this is no longer an actionable pipeline-health warning -- it was
    # paging OpsWatchdogCheckFailing forever. Detail is still available via
    # missing_seq_cols/nonfinite_seq_cols in the v31 trace below for debugging.
    if proc_quality["last_nan"] > 0 or proc_quality["last_inf"] > 0:
        warnings_list.append("processed_last_row_nan_or_inf")
    if ai_errors:
        warnings_list.append("ai_runtime_errors")
    if (
        bool(next_open_execution)
        and execution_delay_late
        and not bool((active_info or {}).get("execution_delay_expected_proxy", False))
    ):
        warnings_list.append("next_open_execution_late")
    if bool(next_open_execution) and execution_price_source not in {"eth_buffer.open[-1]", "scheduled_next_bar_open"}:
        warnings_list.append("execution_price_not_next_open")

    return {
        "schema_version": "live.data_pipeline_health.v1",
        "updated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "status": "WARN" if warnings_list else "OK",
        "warnings": warnings_list,
        "bar_contract": "signal_close_next_open" if bool(next_open_execution) else "decision_bar",
        "raw_eth": raw_eth_stats,
        "signal_eth": signal_eth_stats,
        "raw_processed": raw_proc_stats,
        "processed": proc_stats,
        "quality": proc_quality,
        "signal_bar_kst": _health_ts_kst_text(signal_ts),
        "execution_bar_kst": _health_ts_kst_text(exec_ts),
        "processed_last_kst": _health_ts_kst_text(proc_ts),
        "signal_align_ok": bool(signal_align_ok),
        "decision_price": float(decision_price),
        "execution_price": float(execution_price),
        "execution_price_source": execution_price_source,
        "execution_delay_sec": float(execution_delay_sec),
        "execution_delay_late": bool(execution_delay_late),
        "processed_close": float(proc_close),
        "price_diff_bps": float(price_diff_bps),
        "missing_ohlcv_cols": missing_ohlcv,
        "ai": {
            "groups": list(FINAL_GOVERNOR_AI_FEATURE_GROUPS),
            "missing_cols": ai_missing[:30],
            "missing_count": int(len(ai_missing)),
            "nonfinite_cols": ai_nonfinite[:30],
            "nonfinite_count": int(len(ai_nonfinite)),
            "zero_cols": ai_zero[:30],
            "zero_count": int(len(ai_zero)),
            "errors": list(ai_errors or [])[:10],
            "timing": ai_timing,
            "trace_tail": ai_trace[-6:],
        },
        "v31": {
            "seq_col_count": int(len(v31_seq_cols)),
            "missing_seq_cols": v31_missing[:30],
            "missing_seq_count": int(len(v31_missing)),
            "nonfinite_seq_cols": v31_nonfinite[:30],
            "nonfinite_seq_count": int(len(v31_nonfinite)),
            "q_long": _safe_float(v31.get("q_long", 0.0), 0.0),
            "q_short": _safe_float(v31.get("q_short", 0.0), 0.0),
            "edge": _safe_float(v31.get("edge", 0.0), 0.0),
            "margin": _safe_float(v31.get("margin", 0.0), 0.0),
            "selected_side": str(v31.get("selected_side", "")),
            "pass_gate": bool(v31.get("pass_gate", False)),
        },
        "alpha2_1": {
            "parent_action": int(alpha2.get("parent_action_before", 0) or 0),
            "parent_side": int(alpha2.get("parent_side_before", 0) or 0),
            "teacher_pred_action": int(alpha2.get("teacher_pred_action", 0) or 0),
            "teacher_confidence": _safe_float(alpha2.get("teacher_confidence", 0.0), 0.0),
            "teacher_quality": _safe_float(alpha2.get("teacher_quality", 0.0), 0.0),
            "keep_parent": bool(alpha2.get("keep_parent", False)),
            "reason": str(alpha2.get("reason", "")),
        },
        "regime": {
            "raw": raw_regime,
            "raw_values": regime_vals,
            "clean_enabled": bool(regime_pred.get("enabled", False)),
            "clean_confidence": _safe_float(regime_pred.get("confidence", 0.0), 0.0),
            "clean_transition_risk": _safe_float(regime_pred.get("transition_risk", 0.0), 0.0),
            "clean_missing_input_col_count": int(regime_pred.get("missing_input_col_count", 0) or 0),
        },
        "decision": {
            "action": int(active_info.get("final_action", active_info.get("target_action", 0)) or 0),
            "source": str(active_info.get("source", "")),
            "position_signal": str(active_info.get("position_signal", "")),
            "position_reason": str(active_info.get("position_reason", "")),
            "score": _safe_float(active_info.get("score", 0.0), 0.0),
            "conviction": _safe_float(active_info.get("conviction", 0.0), 0.0),
        },
    }


def _log_data_pipeline_health(health: dict) -> None:
    status = str(health.get("status", "WARN"))
    log_fn = logger.warning if status != "OK" else logger.info
    raw_eth = dict(health.get("raw_eth", {}) or {})
    proc = dict(health.get("processed", {}) or {})
    quality = dict(health.get("quality", {}) or {})
    ai = dict(health.get("ai", {}) or {})
    v31 = dict(health.get("v31", {}) or {})
    alpha2 = dict(health.get("alpha2_1", {}) or {})
    regime = dict(health.get("regime", {}) or {})
    decision = dict(health.get("decision", {}) or {})
    log_fn(
        "%s %s raw=%d proc=%d tail=%d age=%s gap=%s align=%s pxΔ=%.2fbp | "
        "feat nan=%d inf=%d missAI=%d nfAI=%d zeroAI=%d v31miss=%d v31nf=%d | "
        "reg=%s clean(conf=%.2f,tr=%.2f,miss=%d) | "
        "a2 parent=%s teacher=%s conf=%.2f keep=%s | "
        "v31 side=%s pass=%s edge=%.4f margin=%.4f | action=%s src=%s reason=%s warn=%s",
        _log_tag("PIPE", Colors.BLUE),
        _compact_status(status == "OK", "OK", "WARN"),
        int(raw_eth.get("rows", 0) or 0),
        int(proc.get("rows", 0) or 0),
        int(FINAL_GOVERNOR_LIVE_PROCESS_BARS),
        _fmt_age_sec(raw_eth.get("age_sec")),
        _fmt_age_sec(raw_eth.get("last_gap_sec")),
        _compact_status(bool(health.get("signal_align_ok", False))),
        float(health.get("price_diff_bps", 0.0) or 0.0),
        int(quality.get("last_nan", 0) or 0),
        int(quality.get("last_inf", 0) or 0),
        int(ai.get("missing_count", 0) or 0),
        int(ai.get("nonfinite_count", 0) or 0),
        int(ai.get("zero_count", 0) or 0),
        int(v31.get("missing_seq_count", 0) or 0),
        int(v31.get("nonfinite_seq_count", 0) or 0),
        _colored_regime(str(regime.get("raw", "unknown"))),
        float(regime.get("clean_confidence", 0.0) or 0.0),
        float(regime.get("clean_transition_risk", 0.0) or 0.0),
        int(regime.get("clean_missing_input_col_count", 0) or 0),
        _colored_action(int(alpha2.get("parent_action", 0) or 0)),
        _colored_action(int(alpha2.get("teacher_pred_action", 0) or 0)),
        float(alpha2.get("teacher_confidence", 0.0) or 0.0),
        bool(alpha2.get("keep_parent", False)),
        str(v31.get("selected_side", "")),
        bool(v31.get("pass_gate", False)),
        float(v31.get("edge", 0.0) or 0.0),
        float(v31.get("margin", 0.0) or 0.0),
        _colored_action(int(decision.get("action", 0) or 0)),
        _compact_source(str(decision.get("source", "")), 30),
        _compact_source(str(decision.get("position_reason", "")), 42),
        ",".join(list(health.get("warnings", []) or [])[:4]) or "-",
    )


def _json_scalar(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    if isinstance(v, pd.Timestamp):
        return str(v)
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return str(v)


def _build_decision_feature_snapshot(feature_df: pd.DataFrame, active_info: dict, health: dict) -> dict:
    if not isinstance(feature_df, pd.DataFrame) or not len(feature_df):
        return {}
    last = feature_df.iloc[-1]
    values = {str(col): _json_scalar(last.get(col)) for col in feature_df.columns}
    payload_for_hash = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sleeve = dict((active_info or {}).get("sleeve_trace", {}) or {})
    return {
        "schema_version": "live.decision_feature_snapshot.v1",
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "timestamp": str(values.get("timestamp", "")),
        "row_count": int(len(feature_df)),
        "column_count": int(len(feature_df.columns)),
        "feature_hash_sha256": hashlib.sha256(payload_for_hash.encode("utf-8")).hexdigest(),
        "values": values,
        "decision": {
            "action": int((active_info or {}).get("final_action", (active_info or {}).get("target_action", 0)) or 0),
            "source": str((active_info or {}).get("source", "")),
            "position_signal": str((active_info or {}).get("position_signal", "")),
            "position_reason": str((active_info or {}).get("position_reason", "")),
            "score": _safe_float((active_info or {}).get("score", 0.0), 0.0),
            "conviction": _safe_float((active_info or {}).get("conviction", 0.0), 0.0),
        },
        "sleeve_trace": sleeve,
        "health_summary": {
            "status": str((health or {}).get("status", "")),
            "warnings": list((health or {}).get("warnings", []) or []),
            "bar_contract": str((health or {}).get("bar_contract", "")),
            "ai": dict((health or {}).get("ai", {}) or {}),
            "v31": dict((health or {}).get("v31", {}) or {}),
            "regime": dict((health or {}).get("regime", {}) or {}),
        },
    }


def _json_safe_value(value):
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    return _json_scalar(value)


def _json_safe_dict(row: dict | None) -> dict:
    return {str(k): _json_safe_value(v) for k, v in dict(row or {}).items()}


def _duckdb_table_identifier(table: str) -> str:
    name = str(table or "").strip()
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"invalid duckdb table name: {table!r}")
    return name


def _duckdb_feature_cell(value):
    safe = _json_safe_value(value)
    if isinstance(safe, (dict, list)):
        return json.dumps(safe, ensure_ascii=False, sort_keys=True)
    return safe


@serialized_duckdb_access(lambda db_path, *_args, **_kwargs: db_path)
def _write_decision_feature_frame_duckdb(
    db_path: str,
    table: str,
    feature_df: pd.DataFrame,
    active_info: dict,
    health: dict,
    router_snapshot: dict,
) -> None:
    if not db_path or not table or not isinstance(feature_df, pd.DataFrame) or not len(feature_df):
        return
    import duckdb

    table_name = _duckdb_table_identifier(table)
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    row = feature_df.tail(1).copy()
    if row.columns.duplicated().any():
        duplicated = [str(c) for c in row.columns[row.columns.duplicated()].tolist()]
        raise ValueError(f"decision_feature_frame has duplicated columns: {duplicated[:8]}")
    if not row.index.empty:
        row.insert(0, "live_feature_index", _duckdb_feature_cell(row.index[-1]))
    row.insert(0, "live_row_count", int(len(feature_df)))
    row.insert(0, "live_position_reason", str((active_info or {}).get("position_reason", "")))
    row.insert(0, "live_decision_source", str((active_info or {}).get("source", "")))
    row.insert(0, "live_pipeline_stage", str((health or {}).get("pipeline_stage", "")))
    row.insert(0, "live_recorded_at_kst", pd.Timestamp.now(tz="Asia/Seoul").isoformat())
    row["live_decision_action"] = int((active_info or {}).get("final_action", (active_info or {}).get("target_action", 0)) or 0)
    row["live_active_info_json"] = json.dumps(_json_safe_dict(active_info), ensure_ascii=False, sort_keys=True)
    row["live_health_json"] = json.dumps(_json_safe_dict(health), ensure_ascii=False, sort_keys=True)
    row["live_router_snapshot_json"] = json.dumps(_json_safe_dict(router_snapshot), ensure_ascii=False, sort_keys=True)

    for col in row.columns:
        if pd.api.types.is_object_dtype(row[col]) or isinstance(row[col].dtype, pd.CategoricalDtype):
            row[col] = row[col].map(_duckdb_feature_cell)
    row = row.replace([np.inf, -np.inf], np.nan)

    expected_cols = [str(c) for c in row.columns]
    con = duckdb.connect(str(db_path))
    try:
        con.register("_decision_feature_frame_live_row", row)
        exists = bool(
            con.execute(
                "SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()[0]
        )
        if not exists:
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _decision_feature_frame_live_row")
            return

        actual_cols = [str(r[1]) for r in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
        if set(expected_cols) != set(actual_cols):
            # The live feature frame's column set legitimately varies bar-to-bar (e.g. AI/PatchTST
            # columns are sometimes absent right after a restart, before enough history has
            # accumulated). Auto-migrate the table (ALTER TABLE ADD COLUMN) instead of hard-failing
            # every insert forever after the first schema drift -- found 2026-07-13, see project
            # memory: this previously broke ALL writes silently for 11+ days once any drift occurred.
            new_cols = [c for c in expected_cols if c not in actual_cols]
            if new_cols:
                for col in new_cols:
                    duck_type = "DOUBLE" if pd.api.types.is_numeric_dtype(row[col]) else "VARCHAR"
                    con.execute(f'ALTER TABLE {table_name} ADD COLUMN "{col}" {duck_type}')
                logger.info(
                    "decision_feature_frame schema migrated: added %d column(s) to %s: %s",
                    len(new_cols), table_name, new_cols[:8],
                )
        con.execute(f"INSERT INTO {table_name} BY NAME SELECT * FROM _decision_feature_frame_live_row")
    finally:
        con.close()


def _write_decision_feature_frame_snapshot(
    path: str,
    feature_df: pd.DataFrame,
    active_info: dict,
    health: dict,
    router_snapshot: dict,
) -> None:
    if not path or not isinstance(feature_df, pd.DataFrame) or not len(feature_df):
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    runtime_state = _read_json_safe(FINAL_GOVERNOR_RUNTIME_STATE_PATH)
    payload = {
        "schema_version": "live.decision_feature_frame_snapshot.v1",
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "timestamp": str(feature_df.iloc[-1].get("timestamp", "")),
        "row_count": int(len(feature_df)),
        "column_count": int(len(feature_df.columns)),
        "decision": dict(_build_decision_feature_snapshot(feature_df, active_info, health).get("decision", {}) or {}),
        "active_info": _json_safe_dict(active_info),
        "health_summary": {
            "status": str((health or {}).get("status", "")),
            "warnings": list((health or {}).get("warnings", []) or []),
            "bar_contract": str((health or {}).get("bar_contract", "")),
            "ai": dict((health or {}).get("ai", {}) or {}),
            "v31": dict((health or {}).get("v31", {}) or {}),
            "alpha2_1": dict((health or {}).get("alpha2_1", {}) or {}),
            "regime": dict((health or {}).get("regime", {}) or {}),
        },
        "router_snapshot": _json_safe_dict(router_snapshot),
        "governor_runtime_state": _json_safe_dict(runtime_state),
        "frame_attrs": _json_safe_dict(getattr(feature_df, "attrs", {}) or {}),
        "frame": feature_df.copy(),
    }
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{os.path.basename(path)}.",
        suffix=".tmp.gz",
        dir=parent or ".",
    )
    os.close(fd)
    try:
        pd.to_pickle(payload, tmp_path, compression="gzip")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _print_final_trade_summary(timestamp_kst, current_price: float,
                               regime_name: str, rl_action: int, rl_info: dict,
                               meta_result: dict,
                               prev_pos: str | None, cur_pos: str | None):
    if CONSOLE_LOG_COMPACT:
        _log_compact_ai_decision(
            timestamp_kst=timestamp_kst,
            current_price=current_price,
            regime_name=regime_name,
            rl_action=rl_action,
            rl_info=dict(rl_info or {}),
            meta_result=dict(meta_result or {}),
            prev_pos=prev_pos,
            cur_pos=cur_pos,
        )
        return
    C = Colors
    fa = int(meta_result.get('final_action', 0))

    def _action_word(a: int) -> str: return {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}.get(int(a), 'UNKNOWN')
    def _action_color(a: int) -> str: return {0: C.YELLOW, 1: C.GREEN, 2: C.RED}.get(int(a), C.RESET)
    def _bar(v: float, w: int = 8) -> str:
        x = float(np.clip(v, 0.0, 1.0))
        n = int(round(x * w))
        return '█' * n + '░' * (w - n)
    def _trend_word(tdir: int) -> str: return {0: 'DOWN', 1: 'FLAT', 2: 'UP'}.get(int(tdir), 'UNKNOWN')
    def _trend_color(tdir: int) -> str: return {0: C.RED, 1: C.YELLOW, 2: C.GREEN}.get(int(tdir), C.RESET)
    def _kelly_text(v: float) -> str:
        if v >= 0.70: return "강함"
        if v >= 0.40: return "보통"
        if v >= 0.15: return "약함"
        return "매우약함"
    def _conviction_text(v: float) -> str:
        if v >= 1.0: return "진입 강함"
        if v >= 0.60: return "진입 가능"
        if v >= 0.30: return "진입 약함"
        return "진입 부족"
    def _agreement_text(v: float) -> str:
        if v >= 1.5: return "방향 우위 뚜렷"
        if v >= 0.8: return "방향 우위 있음"
        if v >= 0.4: return "방향 우위 약함"
        return "방향 혼재"
    def _ambiguity_text(v: float) -> str:
        if v >= 2.0: return "양방향 충돌 큼"
        if v >= 1.0: return "양방향 경합"
        if v >= 0.0: return "약한 경합"
        return "방향 분리 양호"
    def _hibernation_text(v: float) -> str:
        if v >= 0.85: return "시장 과열/이상"
        if v >= 0.60: return "이상치 주의"
        if v >= 0.30: return "약한 이상 신호"
        return "정상 범위"
    def _amihud_text(v: float) -> str:
        if v >= 1.5: return "유동성 매우 나쁨"
        if v >= 0.8: return "유동성 나쁨"
        if v >= 0.2: return "유동성 보통"
        return "유동성 양호"
    def _gate(ok: bool, label: str, detail: str = "") -> str:
        icon = "✓" if ok else "✗"
        col = C.GREEN if ok else C.RED
        text = label + (f"/{detail}" if detail else "")
        return f"{col}{text}[{icon}]{C.RESET}"
    def _status_badge(ok: bool, ok_label: str = "PASS", fail_label: str = "FAIL") -> str:
        icon = "✓" if ok else "✗"
        col = C.GREEN if ok else C.RED
        label = ok_label if ok else fail_label
        return f"{col}[{label} {icon}]{C.RESET}"

    def _exec_code(pp: str | None, cp: str | None) -> tuple[str, str]:
        if pp == cp:
            if cp is None: return '·', 'STAY_FLAT'
            return '↔', 'HOLD_LONG' if cp == 'LONG' else 'HOLD_SHORT'
        if pp is None and cp == 'LONG': return '↗', 'ENTER_LONG'
        if pp is None and cp == 'SHORT': return '↘', 'ENTER_SHORT'
        if pp == 'LONG' and cp is None: return '✕', 'EXIT_LONG'
        if pp == 'SHORT' and cp is None: return '✕', 'EXIT_SHORT'
        if pp == 'LONG' and cp == 'SHORT': return '⇄', 'FLIP_LONG_TO_SHORT'
        if pp == 'SHORT' and cp == 'LONG': return '⇄', 'FLIP_SHORT_TO_LONG'
        return '·', _pos_transition_label(pp, cp)

    long_edge = float(rl_info.get('long_edge', 0.0))
    short_edge = float(rl_info.get('short_edge', 0.0))
    primary_action = int(rl_info.get("primary_action", 0))
    primary_raw = float(rl_info.get("primary_raw", 0.0))
    primary_kelly = float(rl_info.get("primary_kelly", 0.0))
    primary_disp_action = int(rl_info.get("primary_model_action", primary_action))
    primary_disp_raw = float(rl_info.get("primary_model_raw", primary_raw))
    primary_disp_kelly = float(rl_info.get("primary_model_kelly", primary_kelly))
    primary_disp_std = float(rl_info.get("primary_model_std", rl_info.get("primary_std", 0.0)))
    target_action = int(rl_info.get("target_action", 0))
    net_score = float(rl_info.get("net_score", 0.0))
    agreement_count = int(rl_info.get("agreement_count", 0))
    long_raw = float(rl_info.get('_long_raw', long_edge))
    short_raw = float(rl_info.get('_short_raw', short_edge))
    long_action = int(rl_info.get('_long_action', 1 if long_raw > 0.0 else 0))
    short_action = int(rl_info.get('_short_action', 2 if short_raw > 0.0 else 0))
    long_kelly = float(rl_info.get('_long_kelly', long_raw))
    short_kelly = float(rl_info.get('_short_kelly', short_raw))
    conviction = float(rl_info.get('conviction', abs(long_edge - short_edge)))
    agreement = float(rl_info.get('agreement', abs(long_edge - short_edge)))
    ambiguity = float(rl_info.get('ambiguity', min(long_edge, short_edge)))
    confidence = float(rl_info.get('confidence', 0.0))
    selected_side = str(rl_info.get('_selected_side', 'HOLD'))
    final_kelly = float(meta_result.get('unified_kelly', 0.0))
    source = str(meta_result.get('source', 'N/A'))
    ts = meta_result.get('trend_signal') or {}
    t_dir = 1
    t_strength = 0.0
    t_rev = 0.0
    p_dn = p_up = 0.0
    entry_price_reco = tp_price_reco = sl_price_reco = 0.0
    entry_offset_reco = tp_offset_reco = sl_offset_reco = 0.0
    cb_active = int(meta_result.get("cb_active", 0) or 0) if isinstance(meta_result, dict) else 0
    hibernation_score = float(meta_result.get("hibernation_score", 0.0)) if isinstance(meta_result, dict) else 0.0
    illiq_amihud = float(meta_result.get("illiq_amihud", 0.0)) if isinstance(meta_result, dict) else 0.0
    position_signal = str(meta_result.get("position_signal", "")) if isinstance(meta_result, dict) else ""
    position_reason = str(meta_result.get("position_reason", "")) if isinstance(meta_result, dict) else ""
    position_own_support = float(meta_result.get("position_own_support", 0.0)) if isinstance(meta_result, dict) else 0.0
    position_opp_pressure = float(meta_result.get("position_opp_pressure", 0.0)) if isinstance(meta_result, dict) else 0.0
    position_net_edge = float(meta_result.get("position_net_edge", 0.0)) if isinstance(meta_result, dict) else 0.0
    hold_reason = str(meta_result.get("hold_reason", "")) if isinstance(meta_result, dict) else ""
    block_reason = str(meta_result.get("block_reason", "")) if isinstance(meta_result, dict) else ""
    router_enter_threshold = float(meta_result.get("router_enter_threshold", 0.15)) if isinstance(meta_result, dict) else 0.15
    router_min_agreement_threshold = float(meta_result.get("router_min_agreement_threshold", 0.0)) if isinstance(meta_result, dict) else 0.0
    adaptive_enter_offset = float(meta_result.get("adaptive_enter_offset", 0.0)) if isinstance(meta_result, dict) else 0.0
    adaptive_agreement_offset = float(meta_result.get("adaptive_agreement_offset", 0.0)) if isinstance(meta_result, dict) else 0.0
    router_std_gate_ok = bool(meta_result.get("router_std_gate_ok", True)) if isinstance(meta_result, dict) else True
    router_dual_high_hold = bool(meta_result.get("router_dual_high_hold", False)) if isinstance(meta_result, dict) else False
    long_logit = float(rl_info.get("long_logit", 0.0))
    short_logit = float(rl_info.get("short_logit", 0.0))
    long_std = float(rl_info.get("long_std", 1.0))
    short_std = float(rl_info.get("short_std", 1.0))
    selected_std = float(rl_info.get("selected_std", long_std if long_raw >= short_raw else short_std))
    router_max_confidence_std = float(rl_info.get("max_confidence_std", 1.50))
    
    if isinstance(ts, dict) and ts:
        t_dir = int(ts.get('trend_dir', 1))
        t_strength = float(ts.get('strength', 0.0))
        t_rev = float(ts.get('rev_prob', 0.0))
        probs = ts.get('probs', [])
        if isinstance(probs, (list, tuple)) and len(probs) >= 2:
            p_dn, p_up = float(probs[0]), float(probs[1])
        p_dn = float(ts.get('prob_dn', ts.get('p_down', p_dn)))
        p_up = float(ts.get('prob_up', ts.get('p_up', p_up)))

    ex_icon, ex_code = _exec_code(prev_pos, cur_pos)

    edge_gap = abs(long_edge - short_edge)
    if long_edge > short_edge:
        edge_side_word, edge_side_color = 'LONG_BIAS', C.GREEN
    elif short_edge > long_edge:
        edge_side_word, edge_side_color = 'SHORT_BIAS', C.RED
    else:
        edge_side_word, edge_side_color = 'NEUTRAL_BIAS', C.YELLOW

    long_agent_arrow = {0: '─', 1: '▲', 2: '▼'}.get(int(long_action), '?')
    short_agent_arrow = {0: '─', 1: '▲', 2: '▼'}.get(int(short_action), '?')

    rl_word, rl_color = _action_word(rl_action), _action_color(rl_action)
    final_word, final_color = _action_word(fa), _action_color(fa)
    trend_word, trend_color = _trend_word(t_dir), _trend_color(t_dir)
    W = 62
    _SEP  = "─" * W
    _SEP2 = "═" * W

    def _action_arrow(a: int) -> str: return {0: '─', 1: '▲', 2: '▼'}.get(int(a), '?')
    def _trend_arrow(tdir: int) -> str: return {0: '▼', 1: '─', 2: '▲'}.get(int(tdir), '?')

    fa_arrow = _action_arrow(fa)
    rl_arrow = _action_arrow(rl_action)
    trend_arrow = _trend_arrow(t_dir)

    print(_SEP2)
    ts_str = timestamp_kst.strftime('%Y-%m-%d %H:%M')
    session_flags = _session_flags_from_timestamp(timestamp_kst)
    session_parts = []
    for label, key in (("ASIA", "session_asia"), ("EUROPE", "session_europe"), ("US", "session_us")):
        active = float(session_flags.get(key, 0.0)) >= 0.5
        scol, sword = (C.GREEN, "ON") if active else (C.YELLOW, "OFF")
        session_parts.append(f"{label}={scol}{sword}{C.RESET}")
    header_left = f"{final_color}{C.BOLD}{fa_arrow}{fa_arrow}  {final_word}  →  {ex_code}{C.RESET}"
    print(f" {header_left}  {C.CYAN}{ts_str}  ${current_price:,.2f}{C.RESET}")
    print(f"     {C.CYAN}{regime_name}{C.RESET}  {'  '.join(session_parts)}")
    print(_SEP)

    print(f"  {rl_color}{rl_arrow} 신호{C.RESET}  {rl_color}{rl_word:<6}{C.RESET}"
          f" {edge_side_color}{edge_side_word} {edge_gap:+.3f}{C.RESET}"
          f"  Kelly: {_bar(final_kelly, 8)} {final_kelly:.3f} ({_kelly_text(final_kelly)})")
    print(f"  {C.CYAN}• Governor Edge{C.RESET}  "
          f"L:{long_agent_arrow}{_action_word(long_action):<5} r={C.GREEN}{long_raw:.3f}{C.RESET} k={long_kelly:.3f}"
          f"  S:{short_agent_arrow}{_action_word(short_action):<5} r={C.RED}{short_raw:.3f}{C.RESET} k={short_kelly:.3f}")
    print(f"  {C.CYAN}• Active Sleeve{C.RESET} "
          f"{_action_arrow(primary_disp_action)}{_action_word(primary_disp_action):<5}"
          f" raw={primary_disp_raw:+.3f} k={primary_disp_kelly:.3f} std={primary_disp_std:.3f}"
          f"  → target={_action_word(target_action):<5}"
          f" net={net_score:+.3f} votes={agreement_count}")
    print(f"          → 결정 = {selected_side:<6}"
          f"  conv={conviction:.3f} ({_conviction_text(conviction)})"
          f"  agr={agreement:.3f} ({_agreement_text(agreement)})")
    print(f"  {C.CYAN}• 점수{C.RESET}  "
          f"L={C.GREEN}{long_logit:+.2f}{C.RESET}(±{long_std:.2f})"
          f"  S={C.RED}{short_logit:+.2f}{C.RESET}(±{short_std:.2f})"
          f"  amb={ambiguity:+.2f} ({_ambiguity_text(ambiguity)})"
          f"  conf={confidence:.3f}")

    dn_c, up_c = (C.RED if p_dn > 0.4 else C.RESET), (C.GREEN if p_up > 0.4 else C.RESET)
    trend_model = str(ts.get("trend_model", "N/A")) if isinstance(ts, dict) else "N/A"
    print(f"  {trend_color}{trend_arrow} 추세{C.RESET}    {trend_color}{trend_word:<6}{C.RESET}"
          f"  str={t_strength:.2f}  rev={t_rev:.2f}"
          f"  {dn_c}DN={p_dn:.0%}{C.RESET} {up_c}UP={p_up:.0%}{C.RESET}"
          f"  [{trend_model}]")
    if entry_price_reco > 0.0 or tp_price_reco > 0.0 or sl_price_reco > 0.0:
        print(f"  {C.CYAN}• 가격{C.RESET}    진입={entry_price_reco:,.2f}({entry_offset_reco:+.3%})"
              f"  TP={tp_price_reco:,.2f}({tp_offset_reco:+.3%})"
              f"  SL={sl_price_reco:,.2f}({sl_offset_reco:+.3%})")

    print(f"  {C.CYAN}• 보호{C.RESET}    hib={hibernation_score:.2f} ({_hibernation_text(hibernation_score)})"
          f"  cb={cb_active}  amihud={illiq_amihud:.2f} ({_amihud_text(illiq_amihud)})")
    if hold_reason or block_reason:
        print(f"  {C.CYAN}• HOLD{C.RESET}    {C.YELLOW}{hold_reason or '-'}{C.RESET}"
              f"  block={C.RED}{block_reason or '-'}{C.RESET}")

    _br = block_reason or ""
    _conv_ok = conviction >= router_enter_threshold
    _agr_ok  = agreement  >= router_min_agreement_threshold
    _std_ok  = router_std_gate_ok
    _dual_ok = not router_dual_high_hold
    hibernation_score_th = float(meta_result.get("hibernation_score_th", 0.85)) if isinstance(meta_result, dict) else 0.85
    _hib_ok  = hibernation_score < hibernation_score_th
    _cb_ok   = cb_active == 0
    _trend_ok = "trend" not in _br
    _intg_ok  = "integral" not in _br
    _cool_ok  = "cooldown" not in _br

    if cur_pos is None:
        g_conv = _gate(_conv_ok, f"CONV={conviction:.3f}", f"{router_enter_threshold:.3f}")
        g_agr  = _gate(_agr_ok,  f"AGR={agreement:.3f}",  f"{router_min_agreement_threshold:.3f}")
        g_std  = _gate(_std_ok,  f"STD={selected_std:.2f}", f"{router_max_confidence_std:.2f}")
        g_dual = _gate(_dual_ok, f"DUAL={ambiguity:.2f}")
        entry_result = _status_badge(final_word != "HOLD", "PASS", "FAIL")
        print(f"  {C.CYAN}• 진입장벽{C.RESET}  {entry_result}  {g_conv}  {g_agr}  {g_std}  {g_dual}")
        g_hib  = _gate(_hib_ok,  f"HIB={hibernation_score:.2f}", f"{hibernation_score_th:.2f}")
        g_cb   = _gate(_cb_ok,   "CB")
        g_trend = _gate(_trend_ok, "TREND")
        row2 = [g_hib, g_cb, g_trend]
        if not _intg_ok: row2.append(_gate(False, "INTG"))
        if not _cool_ok: row2.append(_gate(False, "COOL"))
        if adaptive_enter_offset != 0.0 or adaptive_agreement_offset != 0.0:
            row2.append(f"{C.CYAN}적응={adaptive_enter_offset:+.3f}/{adaptive_agreement_offset:+.3f}{C.RESET}")
        print(f"             {'  '.join(row2)}")
    else:
        _own_ok = position_own_support >= 1.10
        _opp_ok = position_opp_pressure < 0.90
        _net_ok = position_net_edge > -0.10
        g_own = _gate(_own_ok, f"OWN={position_own_support:.2f}", "1.10")
        g_opp = _gate(_opp_ok, f"OPP={position_opp_pressure:.2f}", "0.90")
        g_net = _gate(_net_ok, f"NET={position_net_edge:+.2f}", "−0.10")
        if position_signal == "EXIT":
            manage_result = _status_badge(False, "유지", "청산")
            g_action = _gate(True, f"EXIT:{position_reason or '-'}")
        elif position_signal == "REDUCE":
            manage_result = f"{C.YELLOW}[축소!]{C.RESET}"
            g_action = _gate(True, f"REDUCE:{position_reason or '-'}")
        else:
            manage_result = _status_badge(True, "유지", "청산")
            g_action = _gate(True, f"HOLD:{position_reason or 'ok'}")
        print(f"  {C.CYAN}• 청산장벽{C.RESET}  {manage_result}  {g_own}  {g_opp}  {g_net}  {g_action}")

    if prev_pos != cur_pos:
        trade_pnl = meta_result.get("trade_pnl_pct", None)
        if trade_pnl is None and prev_pos is None and cur_pos is not None:
            trade_pnl = 0.0
        if trade_pnl is not None:
            try:
                p = float(trade_pnl)
                p_col = C.GREEN if p > 0 else (C.RED if p < 0 else C.YELLOW)
                print(f"  {C.CYAN}• TRADE{C.RESET}   pnl={p_col}{p:+.2f}%{C.RESET}")
            except Exception:
                pass

    print(f"  {C.CYAN}• 소스{C.RESET}    {source}")
    print(_SEP)
    decision_chain = (f"SIGNAL={rl_color}{rl_word}{C.RESET} → "
                      f"추세={trend_color}{trend_word}{C.RESET} → "
                      f"FINAL={final_color}{final_word}{C.RESET} → "
                      f"EXEC={ex_icon} {ex_code}")
    print(f"  {decision_chain}")
    print(_SEP2)


# ════════════════════════════════════════════════════════════════
# 3-A. GovernorPositionRouter — live position ledger / journal state
# ════════════════════════════════════════════════════════════════
from trading_bot_modules.position_router import (
    GovernorPositionRouter,
    _bootstrap_virtual_router,
    _decode_exposure_bucket,
    _reset_virtual_router_state,
)


class FinalGovernorRuntime:
    """Live runtime for the fully learned governor; legacy sleeves load only when explicitly enabled."""

    TREND_REGIMES = {"bull", "bear"}
    MICRO_REGIMES = {"whipsaw", "normal", "chop"}
    REGIME_COLS = ("regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal")
    BEST_AI_FEATURE_COLS = (
        "ai_adverse_risk",
        "ai_reward_risk",
        "ai_vol_regime_pct",
        "tide_vol_raw",
        "tide_vol_zscore",
        "ai_anchor_revert_prob",
        "ai_anchor_overheat",
        "ai_anchor_trend_escape_prob",
        "timesnet_cycle_sin",
        "timesnet_cycle_cos",
        "timesnet_cycle_delta",
        "ai_flow_pressure",
        "ai_flow_exhaustion",
        "ai_flow_flip_prob",
        "ai_flow_slope",
        "dlinear_smf_ema",
        "dlinear_smf_slope",
        "ai_dir_edge",
        "ai_dir_p_up",
        "ai_dir_p_down",
        "ai_dir_p_flat",
        "ai_dir_entropy",
        "patchtst_median",
        "patchtst_regime_sim",
        "pred_patchtst",
        "conf_patchtst",
    )

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.notional = float(FINAL_GOVERNOR_NOTIONAL)
        self.leverage = float(FINAL_GOVERNOR_LEVERAGE)
        self.window_bars = int(max(100, FINAL_GOVERNOR_WINDOW_BARS))
        self.macro_cfg = MacroTrendSleeveConfig(
            lookback_bars=int(FINAL_GOVERNOR_MACRO_LOOKBACK_BARS),
            threshold=float(FINAL_GOVERNOR_MACRO_THRESHOLD),
            persist_updates=int(FINAL_GOVERNOR_MACRO_PERSIST_UPDATES),
            update_bars=int(FINAL_GOVERNOR_MACRO_UPDATE_BARS),
            notional_exposure=float(FINAL_GOVERNOR_MACRO_NOTIONAL),
            leverage=float(FINAL_GOVERNOR_MACRO_LEVERAGE),
            min_history_bars=int(FINAL_GOVERNOR_MACRO_LOOKBACK_BARS),
            bootstrap_current=bool(FINAL_GOVERNOR_MACRO_BOOTSTRAP_CURRENT),
            take_profit=float(FINAL_GOVERNOR_MACRO_TAKE_PROFIT),
            stop_loss=float(FINAL_GOVERNOR_MACRO_STOP_LOSS),
            trailing_arm=float(FINAL_GOVERNOR_MACRO_TRAILING_ARM),
            trailing_gap=float(FINAL_GOVERNOR_MACRO_TRAILING_GAP),
            lockout_bars=int(FINAL_GOVERNOR_MACRO_LOCKOUT_BARS),
            lockout_until_signal_change=bool(FINAL_GOVERNOR_MACRO_LOCKOUT_UNTIL_SIGNAL_CHANGE),
            lockout_on_any_close=bool(FINAL_GOVERNOR_MACRO_LOCKOUT_ON_ANY_CLOSE),
        )
        self.owner: str = ""
        self.owner_regime: str = ""
        self.peak_unrealized: float = 0.0
        self.last_prepared_frame_for_health: pd.DataFrame | None = None
        self.macro_lockout_signal: int = 0
        self.macro_lockout_bars_left: int = 0
        self.active_macro_take_profit: float = float(FINAL_GOVERNOR_MACRO_TAKE_PROFIT)
        self.active_macro_stop_loss: float = float(FINAL_GOVERNOR_MACRO_STOP_LOSS)
        self.active_macro_max_hold_bars: int = 0
        self.active_macro_quality_score: float = 0.0
        self.active_fully_learned_take_profit: float = 0.0
        self.active_fully_learned_stop_loss: float = 0.0
        self.active_fully_learned_max_hold_bars: int = 0
        self.active_fully_learned_cooldown_bars: int = 0
        self.active_fully_learned_quality_score: float = 0.0
        self.active_fully_learned_confidence: float = 0.0
        self.active_fully_learned_soft_stop_counter: int = 0
        self.last_fully_learned_entry_side: int = 0
        self.last_fully_learned_entry_bar: int = -10**9
        self.fully_learned_cooldown_left: int = 0
        self.active_omega5_take_profit: float = 0.0
        self.active_omega5_stop_loss: float = 0.0
        self.active_omega5_max_hold_bars: int = 0
        self.active_omega5_quality_score: float = 0.0
        self.active_omega5_confidence: float = 0.0
        self.active_omega5_notional: float = 0.0
        self.active_omega5_leverage: float = 1.0
        self.active_omega5_parent_exit_timestamp: str = ""
        self.active_omega5_roundtrip_cost: float = 0.0
        self.active_omega5_source_exit_reason: str = ""
        self.active_omega5_source_exit_price_move: float = 0.0
        self.active_omega5_sizing_trace: dict = {}
        self.last_omega5_entry_side: int = 0
        self.last_omega5_entry_bar: int = -10**9
        self.active_omega4_6_1_source_component: str = ""
        self.active_omega4_6_1_take_profit: float = 0.0
        self.active_omega4_6_1_stop_loss: float = 0.0
        self.active_omega4_6_1_notional: float = 0.0
        self.active_omega4_6_1_leverage: float = 1.0
        self.active_omega4_6_1_quality_score: float = 0.0
        self.active_omega4_6_1_confidence: float = 0.0
        self.active_omega4_6_1_mfe: float = 0.0
        self.active_omega4_6_1_mae: float = 0.0
        self.active_omega4_6_1_tp_order_id: str = ""
        self.active_omega4_6_1_sl_order_id: str = ""
        self.active_v13_1_take_profit: float = 0.0
        self.active_v13_1_stop_loss: float = 0.0
        self.active_v13_1_max_hold_bars: int = 0
        self.active_v13_1_cooldown_bars: int = 0
        self.active_v13_1_quality_score: float = 0.0
        self.active_v13_1_confidence: float = 0.0
        self.active_v13_1_notional: float = 0.0
        self.active_v13_1_leverage: float = 1.0
        self.active_v13_1_lane: str = ""
        self.active_v13_1_probability: float = 0.0
        self.active_v13_1_threshold: float = 0.0
        self.active_v13_1_regime: str = ""
        self.active_v13_1_regime_multiplier: float = 1.0
        self.v13_1_cooldown_left: int = 0
        self.active_lifecycle_v1_base_notional: float = 0.0
        self.active_lifecycle_v1_effective_notional: float = 0.0
        self.active_lifecycle_v1_leverage: float = 1.0
        self.active_lifecycle_v1_cooldown_bars: int = 0
        self.active_lifecycle_v1_quality_score: float = 0.0
        self.active_lifecycle_v1_confidence: float = 0.0
        self.active_lifecycle_v1_entry_bucket: str = ""
        self.active_lifecycle_v1_entry_hazard: float = 0.0
        self.active_lifecycle_v1_entry_support: int = 0
        self.active_lifecycle_v1_edit: str = ""
        self.active_lifecycle_v1_take_profit: float = 0.0
        self.active_lifecycle_v1_stop_loss: float = 0.0
        self.active_lifecycle_v1_max_hold_bars: int = 0
        self.active_lifecycle_v1_jackpot_added: bool = False
        self.active_lifecycle_v1_mae_unrealized: float = 0.0
        self.active_lifecycle_v1_v21_sleeve: str = ""
        self.active_lifecycle_v1_v21_stop_raw: float = 999.0
        self.active_lifecycle_v1_v21_peak_raw: float = -1e9
        self.active_lifecycle_v1_v21_stop_reasons: list[str] = []
        self.active_lifecycle_v1_scout_model_version: str = ""
        self.active_lifecycle_v1_scout_model_id: str = ""
        self.active_lifecycle_v1_scout_model_path: str = ""
        self.active_lifecycle_v1_scout_prob: float = 0.0
        self.active_lifecycle_v1_scout_frac: float = 0.0
        self.active_lifecycle_v1_scout_probability_threshold: float = 0.0
        self.active_lifecycle_v1_scout_cost_pass: bool = False
        self.active_v31_entry_edge: float = 0.0
        self.active_v31_entry_margin: float = 0.0
        self.active_v31_entry_vol_anchor: float = 0.0
        self.active_v31_entry_q_long: float = 0.0
        self.active_v31_entry_q_short: float = 0.0
        self.active_v31_entry_q_long_raw: float = 0.0
        self.active_v31_entry_q_short_raw: float = 0.0
        self.active_v31_entry_selected_side: str = ""
        self.active_v31_entry_guard_reason: str = ""
        self.active_lifecycle_v1_conformal_core_notional: float = 0.0
        self.active_lifecycle_v1_conformal_sleeve_notional: float = 0.0
        self.active_lifecycle_v1_conformal_sleeve_exit_bars: int = 0
        self.active_lifecycle_v1_conformal_sleeve_action: str = ""
        self.lifecycle_v1_cooldown_left: int = 0
        self.lifecycle_v1_policy_bundle: dict | None = None
        self.lifecycle_v1_exit_model = None
        self.lifecycle_v1_payload: dict | None = None
        self.lifecycle_v1_recalibrator: dict | None = None
        self.lifecycle_v1_cfg: dict = {}
        self.lifecycle_v1_entry_cfg: dict = {}
        self.lifecycle_v1_risk_cfg: dict = {}
        self.lifecycle_v1_exit_cfg: dict = {}
        self.lifecycle_v1_policy_path = self._repo_path(FINAL_GOVERNOR_LIFECYCLE_V1_POLICY_PATH)
        self.lifecycle_v1_exit_model_path = self._repo_path(FINAL_GOVERNOR_LIFECYCLE_V1_EXIT_MODEL_PATH)
        self.lifecycle_v1_model_path = self._repo_path(FINAL_GOVERNOR_LIFECYCLE_V1_MODEL_PATH)
        self.conformal_veto_v1_5_enabled = bool(FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_ENABLE)
        self.conformal_veto_v1_5_required = bool(FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_REQUIRED)
        self.conformal_veto_v1_5_model_path = self._repo_path(FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_MODEL_PATH)
        self.conformal_veto_v1_5_report_path = self._repo_path(FINAL_GOVERNOR_CONFORMAL_VETO_V1_5_REPORT_PATH)
        self.conformal_veto_v1_5_adapter: ConformalSleeveV15Adapter | None = None
        self.disabled_v13_1_enabled = bool(FINAL_GOVERNOR_DISABLED_V13_1_ENABLE)
        self.disabled_v13_1_required = bool(FINAL_GOVERNOR_DISABLED_V13_1_REQUIRED)
        self.disabled_v13_1_model_path = self._repo_path(FINAL_GOVERNOR_DISABLED_V13_1_MODEL_PATH)
        self.disabled_v13_1_report_path = self._repo_path(FINAL_GOVERNOR_DISABLED_V13_1_REPORT_PATH)
        self.disabled_v13_1_adapter: object | None = None
        self.deep_gated_gross_enabled = bool(FINAL_GOVERNOR_DEEP_GATED_GROSS_ENABLE)
        self.deep_gated_gross_model_path = self._repo_path(FINAL_GOVERNOR_DEEP_GATED_GROSS_MODEL_PATH)
        self.deep_gated_gross_report_path = self._repo_path(FINAL_GOVERNOR_DEEP_GATED_GROSS_REPORT_PATH)
        self.deep_gated_gross_report: dict = {}
        self.deep_gated_gross_payload: dict | None = None
        self.deep_gated_gross_deep_model = None
        self.deep_gated_gross_cfg: dict = {}
        self.safe_learned_cap_enabled = bool(FINAL_GOVERNOR_SAFE_LEARNED_CAP_ENABLE)
        self.safe_learned_cap_audit_path = self._repo_path(FINAL_GOVERNOR_SAFE_LEARNED_CAP_AUDIT_PATH)
        self.safe_learned_cap_candidate: dict = {}
        self.deep_state_adaptive_calibrator_enabled = bool(FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_ENABLE)
        self.deep_state_adaptive_calibrator_model_path = self._repo_path(
            FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_MODEL_PATH
        )
        self.deep_state_adaptive_calibrator_report_path = self._repo_path(
            FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_REPORT_PATH
        )
        self.deep_state_adaptive_calibrator_audit_path = self._repo_path(
            FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_AUDIT_PATH
        )
        self.deep_state_adaptive_calibrator_payload: dict | None = None
        self.deep_state_adaptive_calibrator_report: dict = {}
        self.deep_state_router_models = None
        self.deep_state_adaptive_calibrator: AdaptiveCalibrator | None = None
        self.deep_state_adaptive_config: AdaptiveConfig | dict | None = None
        self.deep_state_adaptive_future_rolling_q: float | None = None
        self.v21_enabled = bool(FINAL_GOVERNOR_V21_ENABLE)
        self.v21_pure_mode = bool(FINAL_GOVERNOR_V21_PURE_MODE)
        self.v21_bypass_cooldown = bool(FINAL_GOVERNOR_V21_BYPASS_COOLDOWN)
        self.v21_bypass_runtime_risk_gates = bool(FINAL_GOVERNOR_V21_BYPASS_RUNTIME_RISK_GATES)
        self.v21_disable_legacy_hard_stop = bool(FINAL_GOVERNOR_V21_DISABLE_LEGACY_HARD_STOP)
        self.v21_model_path = self._repo_path(FINAL_GOVERNOR_V21_MODEL_PATH)
        self.v21_report_path = self._repo_path(FINAL_GOVERNOR_V21_REPORT_PATH)
        self.v21_audit_path = self._repo_path(FINAL_GOVERNOR_V21_AUDIT_PATH)
        self.v21_payload: dict | None = None
        self.v21_report: dict = {}
        self.v21_audit: dict = {}
        self.v21_model_id: str = ""
        self.v21_adapter_version: str = "v21_rule_scout"
        self.v21_scout_config: dict = {}
        self.v21_stop_config: dict = {}
        self.v21_cap_candidate: dict = {}
        self.v21_path_model: dict | None = None
        self.v22_1_enabled = bool(FINAL_GOVERNOR_V22_1_ENABLE)
        self.v22_1_required = bool(FINAL_GOVERNOR_V22_1_REQUIRED)
        self.v22_1_model_path = self._repo_path(FINAL_GOVERNOR_V22_1_MODEL_PATH)
        self.v22_1_report_path = self._repo_path(FINAL_GOVERNOR_V22_1_REPORT_PATH)
        self.v22_1_audit_path = self._repo_path(FINAL_GOVERNOR_V22_1_AUDIT_PATH)
        self.v22_1_adapter: object | None = None
        self.v21_2_jackpot_enabled = bool(FINAL_GOVERNOR_V21_2_JACKPOT_ENABLE)
        self.v21_2_jackpot_required = bool(FINAL_GOVERNOR_V21_2_JACKPOT_REQUIRED)
        self.v21_2_jackpot_model_path = self._repo_path(FINAL_GOVERNOR_V21_2_JACKPOT_MODEL_PATH)
        self.v21_2_jackpot_report_path = self._repo_path(FINAL_GOVERNOR_V21_2_JACKPOT_REPORT_PATH)
        self.v21_2_jackpot_audit_path = self._repo_path(FINAL_GOVERNOR_V21_2_JACKPOT_AUDIT_PATH)
        self.v21_2_jackpot_adapter: JackpotRunnerV21_2Adapter | None = None
        self.v21_2_parent_bundle: dict | None = None
        self.v31_enabled = bool(FINAL_GOVERNOR_V31_ENABLE)
        self.v31_required = bool(FINAL_GOVERNOR_V31_REQUIRED)
        self.v31_report_path = self._repo_path(FINAL_GOVERNOR_V31_REPORT_PATH)
        self.v31_audit_path = self._repo_path(FINAL_GOVERNOR_V31_AUDIT_PATH)
        self.v31_v27_model_path = self._repo_path(FINAL_GOVERNOR_V31_V27_MODEL_PATH)
        self.v31_report: dict = {}
        self.v31_audit: dict = {}
        self.v31_cfg: dict = {}
        self.v31_v27_payload: dict | None = None
        self.v31_v27_model = None
        self.v31_deep_cooldown_left: int = 0
        self.alpha2_1_model_id = FINAL_GOVERNOR_ALPHA2_1_MODEL_ID
        self.alpha2_1_teacher_model_path = self._repo_path(FINAL_GOVERNOR_ALPHA2_1_TEACHER_MODEL_PATH)
        self.alpha2_1_report_path = self._repo_path(FINAL_GOVERNOR_ALPHA2_1_REPORT_PATH)
        self.alpha2_1_audit_path = self._repo_path(FINAL_GOVERNOR_ALPHA2_1_AUDIT_PATH)
        self.alpha2_1_confidence = float(FINAL_GOVERNOR_ALPHA2_1_CONFIDENCE)
        self.alpha2_1_parent_notional_scale = float(FINAL_GOVERNOR_ALPHA2_1_PARENT_NOTIONAL_SCALE)
        self.alpha2_1_max_notional = float(FINAL_GOVERNOR_ALPHA2_1_MAX_NOTIONAL)
        self.alpha2_1_report: dict = {}
        self.alpha2_1_audit: dict = {}
        self.alpha2_1_teacher_payload: dict | None = None
        self.alpha2_1_teacher_model = None
        self.alpha2_1_teacher_feature_cols: list[str] = []
        self.alpha2_1_teacher_norm: dict = {}
        self.alpha2_1_teacher_buckets: tuple[float, ...] = ()
        self.v21_scout_heads: dict | None = None
        self.v21_feature_cols: list[str] = []
        self.ddh2_ensemble_enabled = bool(FINAL_GOVERNOR_DDH2_ENSEMBLE_ENABLE)
        self.ddh2_report_path = self._repo_path(FINAL_GOVERNOR_DDH2_REPORT_PATH)
        self.ddh2_audit_path = self._repo_path(FINAL_GOVERNOR_DDH2_AUDIT_PATH)
        self.ddh2_report: dict = {}
        self.ddh2_audit: dict = {}
        self.ddh2_config: dict = {}
        self.ddh2_fallback_dd_block_active: bool = False
        if self.disabled_v13_1_enabled:
            raise RuntimeError("disabled_v13_1_adapter_was_removed")
        if self.conformal_veto_v1_5_enabled:
            try:
                missing = [
                    p for p in (self.conformal_veto_v1_5_model_path, self.conformal_veto_v1_5_report_path)
                    if not os.path.exists(p)
                ]
                if missing:
                    raise FileNotFoundError(",".join(missing))
                adapter = ConformalSleeveV15Adapter.load(
                    self.conformal_veto_v1_5_model_path,
                    self.conformal_veto_v1_5_report_path,
                )
                self.conformal_veto_v1_5_adapter = adapter
                logger.info(
                    "SYSTEM conformal_veto_v1_5=ON model=%s report=%s config=%s",
                    self.conformal_veto_v1_5_model_path,
                    self.conformal_veto_v1_5_report_path,
                    str(adapter.selected_config.get("name", "")),
                )
            except Exception as e:
                self.conformal_veto_v1_5_enabled = False
                self.conformal_veto_v1_5_adapter = None
                logger.error(
                    "SYSTEM conformal_veto_v1_5=BLOCKED reason=bad_required_artifact path=%s err=%s",
                    self.conformal_veto_v1_5_model_path,
                    e,
                )
                if self.conformal_veto_v1_5_required:
                    raise RuntimeError(
                        f"required_conformal_veto_v1_5_artifact_unavailable:{self.conformal_veto_v1_5_model_path}"
                    ) from e
        if self.ddh2_ensemble_enabled:
            try:
                missing = [
                    p for p in (self.ddh2_report_path, self.ddh2_audit_path)
                    if not os.path.exists(p)
                ]
                if missing:
                    raise FileNotFoundError(",".join(missing))
                with open(self.ddh2_audit_path, "r", encoding="utf-8") as f:
                    self.ddh2_audit = json.load(f)
                if str(self.ddh2_audit.get("status", "")).lower() != "pass":
                    raise ValueError(f"ddh2_audit_not_pass:{self.ddh2_audit.get('blocking', [])}")
                with open(self.ddh2_report_path, "r", encoding="utf-8") as f:
                    self.ddh2_report = json.load(f)
                self.ddh2_config = dict(self.ddh2_report.get("model_config", {}) or {})
                if not self.ddh2_config:
                    raise ValueError("ddh2_model_config_missing")
                self._validate_ddh2_full_1x_artifacts()
                logger.info(
                    "SYSTEM ddh2_ensemble=ON report=%s audit=%s warnings=%s",
                    self.ddh2_report_path,
                    self.ddh2_audit_path,
                    ",".join(str(x) for x in list(self.ddh2_audit.get("warnings", []) or [])),
                )
            except Exception as e:
                self.ddh2_ensemble_enabled = False
                self.ddh2_report = {}
                self.ddh2_audit = {}
                self.ddh2_config = {}
                logger.error("SYSTEM ddh2_ensemble=BLOCKED reason=bad_required_artifact path=%s err=%s", self.ddh2_report_path, e)
                raise RuntimeError(f"required_ddh2_artifact_unavailable:{self.ddh2_report_path}") from e
        if self.deep_gated_gross_enabled:
            try:
                missing = [
                    p for p in (self.deep_gated_gross_model_path, self.deep_gated_gross_report_path)
                    if not os.path.exists(p)
                ]
                if missing:
                    raise FileNotFoundError(",".join(missing))
                with open(self.deep_gated_gross_report_path, "r", encoding="utf-8") as f:
                    self.deep_gated_gross_report = json.load(f)
                payload = joblib.load(self.deep_gated_gross_model_path)
                _deep_v1.LOOKBACK = _deep_v2.LOOKBACK
                _deep_v1.EMBED_DIM = _deep_v2.ENSEMBLE_EMBED_DIM
                _deep_v1.N_CLUSTERS = _deep_v2.N_CLUSTERS
                seq_features = list(payload.get("sequence_features") or [])
                if not seq_features:
                    raise ValueError("missing_sequence_features")
                torch_model_path = str(payload.get("torch_model") or "")
                if torch_model_path and not os.path.isabs(torch_model_path):
                    torch_model_path = self._repo_path(torch_model_path)
                if not torch_model_path or not os.path.exists(torch_model_path):
                    raise FileNotFoundError(torch_model_path or "missing_torch_model")
                state = torch.load(torch_model_path, map_location="cpu")
                encoders = []
                for sd in list(state.get("models") or []):
                    model = _deep_v2.EnhancedGRUStateEncoder(input_dim=len(seq_features))
                    model.load_state_dict(sd)
                    model.eval()
                    encoders.append(model)
                if not encoders:
                    raise ValueError("missing_gru_ensemble_state")
                deep_model = _deep_v2.GRUSeedEnsemble(encoders)
                deep_model.eval()
                self.deep_gated_gross_payload = payload
                self.deep_gated_gross_deep_model = deep_model
                self.deep_gated_gross_cfg = dict(
                    payload.get("selected_parent_config", {})
                    or payload.get("selected_config", {})
                    or self.deep_gated_gross_report.get("selected_parent_config", {})
                    or self.deep_gated_gross_report.get("selected_config", {})
                    or {}
                )
                cap_candidate = dict(
                    payload.get("selected_cap_candidate", {})
                    or dict(self.deep_gated_gross_report.get("selected", {}) or {}).get("candidate", {})
                    or {}
                )
                if self.safe_learned_cap_enabled:
                    if not cap_candidate:
                        raise ValueError("safe_cap_candidate_missing")
                    if os.path.exists(self.safe_learned_cap_audit_path):
                        with open(self.safe_learned_cap_audit_path, "r", encoding="utf-8") as f:
                            cap_audit = json.load(f)
                        if str(cap_audit.get("status", "")).lower() != "pass":
                            raise ValueError(f"safe_cap_audit_not_pass:{cap_audit.get('blocking', [])}")
                    self.safe_learned_cap_candidate = cap_candidate
                logger.info(
                    "SYSTEM deep_gated_gross=ON model=%s report=%s config=%s safe_cap=%s",
                    self.deep_gated_gross_model_path,
                    self.deep_gated_gross_report_path,
                    self.deep_gated_gross_cfg.get("name", ""),
                    self.safe_learned_cap_candidate.get("name", "OFF") if self.safe_learned_cap_enabled else "OFF",
                )
            except Exception as e:
                logger.warning("SYSTEM deep_gated_gross=OFF reason=bad_artifact path=%s err=%s", self.deep_gated_gross_model_path, e)
                self.deep_gated_gross_enabled = False
                self.safe_learned_cap_enabled = False
                self.safe_learned_cap_candidate = {}
                self.deep_gated_gross_payload = None
                self.deep_gated_gross_deep_model = None
                self.deep_gated_gross_cfg = {}
        if self.deep_state_adaptive_calibrator_enabled:
            try:
                missing = [
                    p for p in (
                        self.deep_state_adaptive_calibrator_model_path,
                        self.deep_state_adaptive_calibrator_report_path,
                        self.deep_state_adaptive_calibrator_audit_path,
                    )
                    if not os.path.exists(p)
                ]
                if missing:
                    raise FileNotFoundError(",".join(missing))
                with open(self.deep_state_adaptive_calibrator_audit_path, "r", encoding="utf-8") as f:
                    adaptive_audit = json.load(f)
                if str(adaptive_audit.get("status", "")).lower() != "pass":
                    raise ValueError(f"adaptive_calibrator_audit_not_pass:{adaptive_audit.get('blocking', [])}")
                with open(self.deep_state_adaptive_calibrator_report_path, "r", encoding="utf-8") as f:
                    self.deep_state_adaptive_calibrator_report = json.load(f)
                main_mod = sys.modules.get("__main__")
                if main_mod is not None:
                    setattr(main_mod, "AdaptiveConfig", AdaptiveConfig)
                    setattr(main_mod, "AdaptiveCalibrator", AdaptiveCalibrator)
                adaptive_payload = joblib.load(self.deep_state_adaptive_calibrator_model_path)
                router_models = adaptive_payload.get("router_models") if isinstance(adaptive_payload, dict) else None
                calibrator = adaptive_payload.get("adaptive_calibrator") if isinstance(adaptive_payload, dict) else None
                cfg = adaptive_payload.get("selected_config") if isinstance(adaptive_payload, dict) else None
                if router_models is None or calibrator is None or cfg is None:
                    raise ValueError("adaptive_calibrator_payload_incomplete")
                payload_base_model = str(adaptive_payload.get("base_model", "") or "")
                if payload_base_model:
                    resolved_base = os.path.abspath(self._repo_path(payload_base_model))
                    active_base = os.path.abspath(self.deep_gated_gross_model_path)
                    if resolved_base != active_base:
                        logger.warning(
                            "SYSTEM adaptive_calibrator base_mismatch active_dgg=%s payload_base=%s",
                            active_base,
                            resolved_base,
                        )
                self.deep_state_adaptive_calibrator_payload = adaptive_payload
                self.deep_state_router_models = router_models
                self.deep_state_adaptive_calibrator = calibrator
                self.deep_state_adaptive_config = cfg
                rolling_q = adaptive_payload.get("future_rolling_q")
                self.deep_state_adaptive_future_rolling_q = (
                    float(rolling_q) if rolling_q is not None and np.isfinite(float(rolling_q)) else None
                )
                logger.info(
                    "SYSTEM deep_state_adaptive_calibrator=ON model=%s config=%s audit=%s",
                    self.deep_state_adaptive_calibrator_model_path,
                    str(self._adaptive_calibrator_cfg_get(cfg, "name", "")),
                    self.deep_state_adaptive_calibrator_audit_path,
                )
            except Exception as e:
                logger.warning(
                    "SYSTEM deep_state_adaptive_calibrator=OFF reason=bad_artifact path=%s err=%s",
                    self.deep_state_adaptive_calibrator_model_path,
                    e,
                )
                self.deep_state_adaptive_calibrator_enabled = False
                self.deep_state_adaptive_calibrator_payload = None
                self.deep_state_adaptive_calibrator_report = {}
                self.deep_state_router_models = None
                self.deep_state_adaptive_calibrator = None
                self.deep_state_adaptive_config = None
                self.deep_state_adaptive_future_rolling_q = None
        if self.v21_enabled:
            try:
                missing = [
                    p for p in (self.v21_model_path, self.v21_report_path, self.v21_audit_path)
                    if not os.path.exists(p)
                ]
                if missing:
                    raise FileNotFoundError(",".join(missing))
                with open(self.v21_audit_path, "r", encoding="utf-8") as f:
                    self.v21_audit = json.load(f)
                if str(self.v21_audit.get("status", "")).lower() != "pass":
                    raise ValueError(f"v21_audit_not_pass:{self.v21_audit.get('blocking', [])}")
                with open(self.v21_report_path, "r", encoding="utf-8") as f:
                    self.v21_report = json.load(f)
                payload = joblib.load(self.v21_model_path)
                if not isinstance(payload, dict):
                    raise ValueError("v21_payload_not_dict")
                model_id = str(payload.get("model_id", "") or "deep_state_safe_cap_reallocator_v21_nearmiss_scout_stop")
                learned_heads = payload.get("heads") if isinstance(payload, dict) else None
                if payload.get("selected_learned_config") and isinstance(learned_heads, dict):
                    adapter_version = "v22_1_learned_scout"
                    scout_cfg = dict(payload.get("selected_learned_config", {}) or {})
                else:
                    adapter_version = "v21_rule_scout"
                    scout_cfg = dict(payload.get("selected_scout_config", {}) or {})
                    learned_heads = None
                stop_cfg = dict(payload.get("selected_stop_config", {}) or {})
                path_model = payload.get("path_model") if isinstance(payload, dict) else None
                cap_candidate = dict(payload.get("selected_cap_candidate", {}) or {})
                if not scout_cfg or not stop_cfg or not isinstance(path_model, dict):
                    raise ValueError("v21_payload_incomplete")
                if adapter_version == "v22_1_learned_scout":
                    feature_cols = list(payload.get("feature_cols") or dict(learned_heads or {}).get("feature_cols") or [])
                    if not feature_cols:
                        raise ValueError("v22_1_feature_cols_missing")
                    if not dict(learned_heads or {}).get("gate_model") or not dict(learned_heads or {}).get("frac_model"):
                        raise ValueError("v22_1_heads_incomplete")
                    path_feature_cols = list(dict(path_model or {}).get("feature_cols") or [])
                    if not path_feature_cols:
                        raise ValueError("v22_1_path_feature_cols_missing")
                    scaler_n_features = getattr(dict(path_model or {}).get("scaler"), "n_features_in_", None)
                    if scaler_n_features is not None and int(scaler_n_features) != len(path_feature_cols):
                        raise ValueError(
                            f"v22_1_path_feature_count_mismatch:{int(scaler_n_features)}!={len(path_feature_cols)}"
                        )
                else:
                    feature_cols = []
                self.v21_payload = payload
                self.v21_model_id = model_id
                self.v21_adapter_version = adapter_version
                self.v21_scout_config = scout_cfg
                self.v21_stop_config = stop_cfg
                self.v21_cap_candidate = cap_candidate
                self.v21_path_model = path_model
                self.v21_scout_heads = learned_heads if adapter_version == "v22_1_learned_scout" else None
                self.v21_feature_cols = feature_cols
                if self.safe_learned_cap_enabled and cap_candidate:
                    self.safe_learned_cap_candidate = cap_candidate
                if dict(payload.get("selected_parent_config", {}) or {}):
                    self.deep_gated_gross_cfg = dict(payload.get("selected_parent_config", {}) or self.deep_gated_gross_cfg)
                logger.info(
                    "SYSTEM lifecycle_scout_layer=ON model_id=%s adapter=%s model=%s pure=%s bypass_cooldown=%s bypass_risk=%s scout=%s stop=%s audit=%s",
                    self.v21_model_id,
                    self.v21_adapter_version,
                    self.v21_model_path,
                    self.v21_pure_mode,
                    self.v21_bypass_cooldown,
                    self.v21_bypass_runtime_risk_gates,
                    str(self.v21_scout_config.get("name", "")),
                    str(self.v21_stop_config.get("name", "")),
                    self.v21_audit_path,
                )
            except Exception as e:
                logger.warning("SYSTEM v21_nearmiss_scout_stop=OFF reason=bad_artifact path=%s err=%s", self.v21_model_path, e)
                self.v21_enabled = False
                self.v21_payload = None
                self.v21_report = {}
                self.v21_audit = {}
                self.v21_model_id = ""
                self.v21_adapter_version = "v21_rule_scout"
                self.v21_scout_config = {}
                self.v21_stop_config = {}
                self.v21_cap_candidate = {}
                self.v21_path_model = None
                self.v21_scout_heads = None
                self.v21_feature_cols = []
        if self.v22_1_enabled:
            raise RuntimeError("v22_1_adapter_was_removed")
        if self.v21_2_jackpot_enabled:
            try:
                adapter = JackpotRunnerV21_2Adapter.load(
                    self.v21_2_jackpot_model_path,
                    self.v21_2_jackpot_report_path,
                    self.v21_2_jackpot_audit_path,
                )
                parent_override_path = str(os.getenv("FINAL_GOVERNOR_V21_2_PARENT_OVERRIDE_PATH", "") or "").strip()
                base_path = self._repo_path(parent_override_path or adapter.base_model_path)
                if not os.path.exists(base_path):
                    raise FileNotFoundError(base_path)
                self.v21_2_parent_bundle = joblib.load(base_path)
                self.v21_2_jackpot_adapter = adapter
                logger.info(
                    "SYSTEM v21_2_jackpot=ON model=%s parent=%s parent_override=%s report=%s audit=%s",
                    self.v21_2_jackpot_model_path,
                    base_path,
                    bool(parent_override_path),
                    self.v21_2_jackpot_report_path,
                    self.v21_2_jackpot_audit_path,
                )
            except Exception as e:
                logger.error("SYSTEM v21_2_jackpot=BLOCKED reason=bad_required_artifact path=%s err=%s", self.v21_2_jackpot_model_path, e)
                self.v21_2_jackpot_enabled = False
                self.v21_2_jackpot_adapter = None
                self.v21_2_parent_bundle = None
                if self.v21_2_jackpot_required:
                    raise RuntimeError(f"required_v21_2_jackpot_artifact_unavailable:{self.v21_2_jackpot_model_path}") from e
        if self.v31_enabled:
            try:
                if _v31_live is None:
                    raise RuntimeError("v31_backtest_module_unavailable")
                missing = [
                    p
                    for p in (self.v31_report_path, self.v31_audit_path, self.v31_v27_model_path)
                    if not os.path.exists(p)
                ]
                if missing:
                    raise FileNotFoundError(",".join(missing))
                with open(self.v31_audit_path, "r", encoding="utf-8") as f:
                    self.v31_audit = json.load(f)
                if str(self.v31_audit.get("status", "")).lower() != "pass":
                    raise ValueError(f"v31_audit_not_pass:{self.v31_audit.get('blocking', [])}")
                if bool(self.v31_audit.get("selection_uses_2026", False)):
                    raise ValueError("v31_audit_selection_uses_2026")
                if not bool(self.v31_audit.get("deep_sleeve_only_when_parent_cash", False)):
                    raise ValueError("v31_audit_deep_sleeve_contract_missing")
                with open(self.v31_report_path, "r", encoding="utf-8") as f:
                    self.v31_report = json.load(f)
                cfg = dict(self.v31_report.get("selected_config", {}) or self.v31_audit.get("selected_config", {}) or {})
                required_cfg = {
                    "edge_th",
                    "margin_th",
                    "notional",
                    "cooldown",
                    "base_tp",
                    "base_sl",
                    "base_hold",
                    "tp_util_mult",
                    "sl_vol_mult",
                    "trail_gap_mult",
                    "hold_decay_start",
                    "hold_decay_rate",
                    "tp_cap",
                    "sl_cap",
                }
                if not cfg or any(k not in cfg for k in required_cfg):
                    raise ValueError("v31_selected_config_incomplete")
                if float(FINAL_GOVERNOR_V31_DEEP_NOTIONAL) > 0.0:
                    cfg["notional"] = float(FINAL_GOVERNOR_V31_DEEP_NOTIONAL)
                    cfg["live_notional_override"] = True
                    cfg["live_notional_override_source"] = "FINAL_GOVERNOR_V31_DEEP_NOTIONAL"
                if bool(FINAL_GOVERNOR_ALPHA3_CANONICAL_DECISION_ENABLE):
                    cfg["trail_activation"] = float(FINAL_GOVERNOR_V31_TRAIL_ACTIVATION)
                    cfg["alpha3_canonical_decision_parity"] = True
                    cfg["alpha3_execution_contract"] = "next_open_limit_touch0_fee20"
                payload, model = _v31_live._load_v27(Path(self.v31_v27_model_path))
                self.v31_cfg = cfg
                self.v31_v27_payload = payload
                self.v31_v27_model = model
                logger.info(
                    "SYSTEM v31_frozen_v27_rule_exit=ON report=%s audit=%s v27=%s config=%s deep_notional=%.3f",
                    self.v31_report_path,
                    self.v31_audit_path,
                    self.v31_v27_model_path,
                    str(self.v31_cfg.get("name", "")),
                    float(self.v31_cfg.get("notional", 0.0) or 0.0),
                )
            except Exception as e:
                logger.error("SYSTEM v31_frozen_v27_rule_exit=BLOCKED reason=bad_required_artifact path=%s err=%s", self.v31_report_path, e)
                self.v31_enabled = False
                self.v31_report = {}
                self.v31_audit = {}
                self.v31_cfg = {}
                self.v31_v27_payload = None
                self.v31_v27_model = None
                if self.v31_required:
                    raise RuntimeError(f"required_v31_artifact_unavailable:{self.v31_report_path}") from e
        if bool(FINAL_GOVERNOR_ALPHA2_1_ENABLE):
            try:
                if _Alpha2TeacherModel is None or _alpha2_apply_norm is None or _alpha2_seq_tensor is None:
                    raise RuntimeError("alpha2_teacher_model_class_unavailable")
                missing = [
                    p
                    for p in (self.alpha2_1_teacher_model_path, self.alpha2_1_report_path, self.alpha2_1_audit_path)
                    if not os.path.exists(p)
                ]
                if missing:
                    raise FileNotFoundError(",".join(missing))
                with open(self.alpha2_1_audit_path, "r", encoding="utf-8") as f:
                    self.alpha2_1_audit = json.load(f)
                if str(self.alpha2_1_audit.get("status", "")).lower() != "pass":
                    raise ValueError(f"alpha2_1_audit_not_pass:{self.alpha2_1_audit.get('blocking', [])}")
                if bool(self.alpha2_1_audit.get("selection_uses_2026", False)):
                    raise ValueError("alpha2_1_audit_selection_uses_2026")
                with open(self.alpha2_1_report_path, "r", encoding="utf-8") as f:
                    self.alpha2_1_report = json.load(f)
                selected_runtime = dict(self.alpha2_1_audit.get("selected_runtime", {}) or {})
                if selected_runtime:
                    self.alpha2_1_confidence = float(selected_runtime.get("confidence", self.alpha2_1_confidence))
                    self.alpha2_1_parent_notional_scale = float(
                        selected_runtime.get("parent_notional_scale", self.alpha2_1_parent_notional_scale)
                    )
                    self.alpha2_1_max_notional = float(selected_runtime.get("max_notional", self.alpha2_1_max_notional))
                payload = torch.load(self.alpha2_1_teacher_model_path, map_location="cpu", weights_only=False)
                feature_cols = list(payload.get("feature_cols") or [])
                buckets = tuple(float(x) for x in payload.get("buckets") or ())
                norm = dict(dict(payload.get("train_meta", {}) or {}).get("norm", {}) or {})
                if not feature_cols or not buckets or not norm:
                    raise ValueError("alpha2_1_teacher_payload_incomplete")
                model = _Alpha2TeacherModel(len(feature_cols), notional_classes=len(buckets))
                model.load_state_dict(payload["state_dict"])
                model.cpu().eval()
                self.alpha2_1_teacher_payload = payload
                self.alpha2_1_teacher_model = model
                self.alpha2_1_teacher_feature_cols = feature_cols
                self.alpha2_1_teacher_norm = norm
                self.alpha2_1_teacher_buckets = buckets
                logger.info(
                    "SYSTEM alpha2_1=ON teacher=%s report=%s audit=%s confidence=%.3f parent_scale=%.3f max_notional=%.3f",
                    self.alpha2_1_teacher_model_path,
                    self.alpha2_1_report_path,
                    self.alpha2_1_audit_path,
                    self.alpha2_1_confidence,
                    self.alpha2_1_parent_notional_scale,
                    self.alpha2_1_max_notional,
                )
            except Exception as e:
                logger.error("SYSTEM alpha2_1=BLOCKED reason=bad_required_artifact path=%s err=%s", self.alpha2_1_teacher_model_path, e)
                raise RuntimeError(f"required_alpha2_1_artifact_unavailable:{self.alpha2_1_teacher_model_path}") from e
        else:
            logger.info("SYSTEM alpha2_1=OFF reason=disabled_by_env")
        self.dsac_overlay_enabled = bool(FINAL_GOVERNOR_DSAC_OVERLAY_ENABLE)
        self.dsac_overlay_ckpt_path = self._repo_path(FINAL_GOVERNOR_DSAC_OVERLAY_CKPT_PATH)
        self.dsac_overlay_mode = str(FINAL_GOVERNOR_DSAC_OVERLAY_MODE or "half_if_opposite").strip().lower()
        self.dsac_overlay_threshold = float(np.clip(FINAL_GOVERNOR_DSAC_OVERLAY_THRESHOLD, 0.0, 1.0))
        self.dsac_overlay_scale = float(np.clip(FINAL_GOVERNOR_DSAC_OVERLAY_SCALE, 0.0, 1.0))
        self.dsac_overlay_cost_gate_enabled = bool(FINAL_GOVERNOR_DSAC_OVERLAY_COST_GATE_ENABLE)
        self.dsac_overlay_cost_buffer = float(max(0.0, FINAL_GOVERNOR_DSAC_OVERLAY_COST_BUFFER))
        self.dsac_overlay_router: DSACRouter | None = None
        self.dsac_overlay_ckpt_meta: dict = {}
        if self.dsac_overlay_enabled:
            try:
                if not os.path.exists(self.dsac_overlay_ckpt_path):
                    raise FileNotFoundError(self.dsac_overlay_ckpt_path)
                dsac_ckpt = torch.load(self.dsac_overlay_ckpt_path, map_location=self.device)
                state_dim = int(dsac_ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)
                actor = GaussianActor(state_dim=state_dim).to(self.device)
                actor.load_state_dict(dsac_ckpt["actor"])
                actor.eval()
                self.dsac_overlay_router = DSACRouter(actor, device=self.device)
                self.dsac_overlay_ckpt_meta = {
                    "epoch": int(dsac_ckpt.get("epoch", 0) or 0),
                    "global_step": int(dsac_ckpt.get("global_step", 0) or 0),
                    "state_dim": int(state_dim),
                    "best_val_pnl": float(_safe_float(dsac_ckpt.get("best_val_pnl", 0.0), 0.0)),
                }
                logger.info(
                    "SYSTEM dsac_overlay=ON ckpt=%s mode=%s threshold=%.3f scale=%.3f cost_buffer=%.4f",
                    self.dsac_overlay_ckpt_path,
                    self.dsac_overlay_mode,
                    self.dsac_overlay_threshold,
                    self.dsac_overlay_scale,
                    self.dsac_overlay_cost_buffer,
                )
            except Exception as e:
                logger.warning("SYSTEM dsac_overlay=OFF reason=bad_artifact path=%s err=%s", self.dsac_overlay_ckpt_path, e)
                self.dsac_overlay_enabled = False
                self.dsac_overlay_router = None
                self.dsac_overlay_ckpt_meta = {}
        self.deep_constant_gross_enabled = bool(FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_ENABLE)
        self.deep_constant_gross_report_path = self._repo_path(FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_REPORT_PATH)
        self.deep_constant_gross_target_notional = float(FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_TARGET_NOTIONAL)
        self.deep_constant_gross_cost3_notional = float(FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_NOTIONAL)
        self.deep_constant_gross_report: dict = {}
        if self.deep_constant_gross_enabled:
            if os.path.exists(self.deep_constant_gross_report_path):
                try:
                    with open(self.deep_constant_gross_report_path, "r", encoding="utf-8") as f:
                        self.deep_constant_gross_report = json.load(f)
                    cfg = dict(self.deep_constant_gross_report.get("selected_config", {}) or {})
                    if "target_notional" in cfg:
                        self.deep_constant_gross_target_notional = float(cfg.get("target_notional") or self.deep_constant_gross_target_notional)
                    if "cost3_notional" in cfg:
                        self.deep_constant_gross_cost3_notional = float(cfg.get("cost3_notional") or self.deep_constant_gross_cost3_notional)
                except Exception as e:
                    logger.warning("SYSTEM deep_constant_gross=OFF reason=bad_report path=%s err=%s", self.deep_constant_gross_report_path, e)
                    self.deep_constant_gross_enabled = False
            else:
                logger.warning("SYSTEM deep_constant_gross=OFF reason=missing_report path=%s", self.deep_constant_gross_report_path)
                self.deep_constant_gross_enabled = False
        if bool(FINAL_GOVERNOR_LIFECYCLE_V1_ENABLE):
            missing = [
                p for p in (self.lifecycle_v1_policy_path, self.lifecycle_v1_exit_model_path, self.lifecycle_v1_model_path)
                if not os.path.exists(p)
            ]
            if missing:
                logger.warning("SYSTEM lifecycle_v1=OFF reason=missing_artifacts paths=%s", ",".join(missing))
            else:
                self.lifecycle_v1_policy_bundle = joblib.load(self.lifecycle_v1_policy_path)
                exit_payload = joblib.load(self.lifecycle_v1_exit_model_path)
                self.lifecycle_v1_exit_model = exit_payload.get("model") if isinstance(exit_payload, dict) else exit_payload
                self.lifecycle_v1_payload = joblib.load(self.lifecycle_v1_model_path)
                self.lifecycle_v1_recalibrator = dict(self.lifecycle_v1_payload.get("recalibrator", {}))
                self.lifecycle_v1_cfg = dict(self.lifecycle_v1_payload.get("selected_runtime_config", {}))
                self.lifecycle_v1_entry_cfg = dict(self.lifecycle_v1_payload.get("entry_config", {}))
                self.lifecycle_v1_risk_cfg = dict(self.lifecycle_v1_payload.get("risk_config", {}))
                self.lifecycle_v1_exit_cfg = dict(self.lifecycle_v1_payload.get("exit_config", {}))
        self.fully_learned_policy_bundle: dict | None = None
        self.fully_learned_policy_path = self._repo_path(FINAL_GOVERNOR_FULLY_LEARNED_POLICY_PATH)
        self.fully_learned_summary_path = self._repo_path(FINAL_GOVERNOR_FULLY_LEARNED_SUMMARY_PATH)
        self.fully_learned_tp_sl_score_path = self._repo_path(FINAL_GOVERNOR_FULLY_LEARNED_TP_SL_SCORE_PATH)
        self.fully_learned_runtime_config_path = self._repo_path(FINAL_GOVERNOR_FULLY_LEARNED_RUNTIME_CONFIG_PATH)
        self.fully_learned_runtime_config: dict[str, object] = self._load_fully_learned_runtime_config(
            self.fully_learned_runtime_config_path
        )
        if bool(FINAL_GOVERNOR_FULLY_LEARNED_ENABLE):
            self._apply_fully_learned_v31_runtime_config()
        self.fully_learned_scale_runtime: dict[str, float | str] | None = None
        self.fully_learned_fallback_policy_bundle: dict | None = None
        self.fully_learned_fallback_policy_path = self._repo_path(FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_POLICY_PATH)
        self.fully_learned_fallback_summary_path = self._repo_path(FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_SUMMARY_PATH)
        self.fully_learned_fallback_scale_runtime: dict[str, float | str] | None = None
        self.fully_learned_fallback_exit_submodel: dict[str, object] | None = None
        self.fully_learned_fallback_exit_submodel_path = self._repo_path(
            FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_EXIT_SUBMODEL_PATH
        )
        self.fully_learned_primary_low_conf_threshold = float(FINAL_GOVERNOR_FULLY_LEARNED_PRIMARY_LOW_CONF_THRESHOLD)
        self.fully_learned_primary_low_conf_tp_scale = float(FINAL_GOVERNOR_FULLY_LEARNED_PRIMARY_LOW_CONF_TP_SCALE)
        if not np.isfinite(self.fully_learned_primary_low_conf_threshold) or not (0.0 < self.fully_learned_primary_low_conf_threshold <= 1.0):
            raise RuntimeError(
                f"invalid FINAL_GOVERNOR_FULLY_LEARNED_PRIMARY_LOW_CONF_THRESHOLD={self.fully_learned_primary_low_conf_threshold}"
            )
        if not np.isfinite(self.fully_learned_primary_low_conf_tp_scale) or not (0.0 < self.fully_learned_primary_low_conf_tp_scale <= 1.0):
            raise RuntimeError(
                f"invalid FINAL_GOVERNOR_FULLY_LEARNED_PRIMARY_LOW_CONF_TP_SCALE={self.fully_learned_primary_low_conf_tp_scale}"
            )
        self.fully_learned_fallback_tp_scale = float(FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_TP_SCALE)
        if not np.isfinite(self.fully_learned_fallback_tp_scale) or self.fully_learned_fallback_tp_scale <= 0.0:
            raise RuntimeError(
                f"invalid FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_TP_SCALE={self.fully_learned_fallback_tp_scale}"
            )
        if bool(FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_EXIT_SUBMODEL_ENABLE):
            if not os.path.exists(self.fully_learned_fallback_exit_submodel_path):
                raise RuntimeError(
                    "missing fallback exit submodel: "
                    f"{self.fully_learned_fallback_exit_submodel_path}"
                )
            with open(self.fully_learned_fallback_exit_submodel_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.fully_learned_fallback_exit_submodel = self._normalize_fully_learned_fallback_exit_submodel(payload)
        self.fully_learned_tp_sl_score_bundle: dict | None = None
        self.last_fully_learned_contract_audit: dict[str, object] = {}
        self.last_fully_learned_selection_trace: dict[str, object] = {}
        self.fully_learned_contract_blocked: bool = False
        if bool(FINAL_GOVERNOR_FULLY_LEARNED_ENABLE):
            if not os.path.exists(self.fully_learned_policy_path):
                raise RuntimeError(f"missing fully learned governor model: {self.fully_learned_policy_path}")
            self.fully_learned_policy_bundle = joblib.load(self.fully_learned_policy_path)
            self.fully_learned_scale_runtime = self._load_fully_learned_scale_runtime(self.fully_learned_summary_path)
            if not os.path.exists(self.fully_learned_tp_sl_score_path):
                raise RuntimeError(f"missing fully learned tp/sl score model: {self.fully_learned_tp_sl_score_path}")
            self.fully_learned_tp_sl_score_bundle = joblib.load(self.fully_learned_tp_sl_score_path)
            if bool(FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_ENABLE):
                if not os.path.exists(self.fully_learned_fallback_policy_path):
                    raise RuntimeError(f"missing fully learned fallback model: {self.fully_learned_fallback_policy_path}")
                self.fully_learned_fallback_policy_bundle = joblib.load(self.fully_learned_fallback_policy_path)
                self.fully_learned_fallback_scale_runtime = self._load_fully_learned_scale_runtime(
                    self.fully_learned_fallback_summary_path
                )
        self.omega5_adapter: Omega5LiveAdapter | None = None
        self.omega4_6_2_source_parent_adapter: Omega462SourceParentLiveAdapter | None = None
        self.omega5_report_path = self._repo_path(FINAL_GOVERNOR_OMEGA5_REPORT_PATH)
        self.omega5_feature_veto_report_path = self._repo_path(FINAL_GOVERNOR_OMEGA5_FEATURE_VETO_REPORT_PATH)
        self.omega5_two_stage_veto_report_path = self._repo_path(FINAL_GOVERNOR_OMEGA5_TWO_STAGE_VETO_REPORT_PATH)
        self.omega5_pnl_tilt_report_path = self._repo_path(FINAL_GOVERNOR_OMEGA5_PNL_TILT_REPORT_PATH)
        self.omega5_redteam_path = self._repo_path(FINAL_GOVERNOR_OMEGA5_REDTEAM_PATH)
        self.omega5_frontier_audit_path = self._repo_path(FINAL_GOVERNOR_OMEGA5_FRONTIER_AUDIT_PATH)
        self.omega5_cvp_audit_path = self._repo_path(FINAL_GOVERNOR_OMEGA5_CVP_AUDIT_PATH)
        self.omega5_artifact_integrity_path = self._repo_path(FINAL_GOVERNOR_OMEGA5_ARTIFACT_INTEGRITY_PATH)
        self.omega5_source_parent_report_path = self._repo_path(FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_REPORT_PATH)
        self.omega5_source_parent_cap220_contract_path = self._repo_path(
            FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_CAP220_CONTRACT_PATH
        )
        if bool(FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE):
            missing = [
                p
                for p in (
                    self.omega5_source_parent_report_path,
                    self.omega5_source_parent_cap220_contract_path,
                )
                if not os.path.exists(p)
            ]
            if missing:
                raise RuntimeError(f"missing Omega4.6.2 source parent live artifacts: {missing}")
            self.omega4_6_2_source_parent_adapter = Omega462SourceParentLiveAdapter(
                Omega462SourceParentConfig(
                    source_parent_report_path=self.omega5_source_parent_report_path,
                    cap220_runtime_contract_path=self.omega5_source_parent_cap220_contract_path,
                    device=self.device,
                )
            )
            logger.info(
                "SYSTEM omega4_6_2_source_parent=ON model_id=%s report=%s cap220_contract=%s",
                OMEGA462_SOURCE_PARENT_MODEL_ID,
                self.omega5_source_parent_report_path,
                self.omega5_source_parent_cap220_contract_path,
            )
        if bool(FINAL_GOVERNOR_OMEGA5_ENABLE):
            if not bool(FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE):
                raise RuntimeError(
                    "Omega5 live promotion requires FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE=true; "
                    "Omega1.2.1/Omega3 parent substitution is forbidden."
                )
            if self.omega4_6_2_source_parent_adapter is None:
                raise RuntimeError(
                    "Omega5 live promotion requires Omega4.6.2 source parent adapter; "
                    "Omega1.2.1/Omega3 parent substitution is forbidden."
                )
            self.omega5_adapter = Omega5LiveAdapter(
                report_path=self.omega5_report_path,
                feature_veto_report_path=self.omega5_feature_veto_report_path,
                two_stage_veto_report_path=self.omega5_two_stage_veto_report_path,
                pnl_tilt_report_path=self.omega5_pnl_tilt_report_path,
                redteam_path=self.omega5_redteam_path,
                frontier_audit_path=self.omega5_frontier_audit_path,
                cvp_audit_path=self.omega5_cvp_audit_path,
                artifact_integrity_path=self.omega5_artifact_integrity_path,
            )
            logger.info(
                "SYSTEM omega5=ON model_id=%s source_report=%s redteam=%s artifact_integrity=%s",
                OMEGA5_MODEL_ID,
                self.omega5_report_path,
                self.omega5_redteam_path,
                self.omega5_artifact_integrity_path,
            )
        self.omega4_6_1_adapter: Omega461LiveAdapter | None = None
        # Set externally (main()) only if FINAL_GOVERNOR_OMEGA4_6_1_ETH_PORTFOLIO_CAP_ENABLE is
        # True; stays None (no behavior change) otherwise. See _decide_omega4_6_1_entry.
        self.omega4_6_1_portfolio_risk = None
        self.omega4_6_1_h48qual_bundle_path = self._repo_path(FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH)
        self.omega4_6_1_h48qual_sidecar_path = self._repo_path(FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_SIDECAR_PATH)
        self.omega4_6_1_zig075_bundle_path = self._repo_path(FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH)
        self.omega4_6_1_zig075_sidecar_path = self._repo_path(FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_SIDECAR_PATH)
        if bool(FINAL_GOVERNOR_OMEGA4_6_1_ENABLE):
            missing = [
                p
                for p in (
                    self.omega4_6_1_h48qual_bundle_path,
                    self.omega4_6_1_h48qual_sidecar_path,
                    self.omega4_6_1_zig075_bundle_path,
                    self.omega4_6_1_zig075_sidecar_path,
                )
                if not os.path.exists(p)
            ]
            if missing:
                raise RuntimeError(f"missing Omega4.6.1 live artifacts: {missing}")
            _omega4_6_1_adapter_kwargs = {}
            if bool(FINAL_GOVERNOR_OMEGA4_6_1_ETH_DURATION_GATE_OFF):
                _omega4_6_1_adapter_kwargs["duration_threshold"] = -999.0
            self.omega4_6_1_adapter = Omega461LiveAdapter(
                h48qual_bundle=self.omega4_6_1_h48qual_bundle_path,
                h48qual_sidecar=self.omega4_6_1_h48qual_sidecar_path,
                zig075_bundle=self.omega4_6_1_zig075_bundle_path,
                zig075_sidecar=self.omega4_6_1_zig075_sidecar_path,
                device=self.device,
                **_omega4_6_1_adapter_kwargs,
            )
            logger.info(
                "SYSTEM omega4_6_1=ON model_id=%s h48qual_bundle=%s zig075_bundle=%s",
                OMEGA4_6_1_MODEL_ID,
                self.omega4_6_1_h48qual_bundle_path,
                self.omega4_6_1_zig075_bundle_path,
            )
        self.clean_regime4_sticky_bundle: dict | None = None
        self.clean_regime4_sticky_path = self._repo_path(FINAL_GOVERNOR_CLEAN_REGIME4_STICKY_MODEL_PATH)
        if bool(FINAL_GOVERNOR_CLEAN_REGIME4_STICKY_ENABLE):
            if os.path.exists(self.clean_regime4_sticky_path):
                self.clean_regime4_sticky_bundle = joblib.load(self.clean_regime4_sticky_path)
            else:
                logger.warning("SYSTEM clean_regime4_sticky=OFF reason=missing_model path=%s", self.clean_regime4_sticky_path)
        self.execution_policy_bundle: dict | None = None
        self.execution_policy_path = self._repo_path(FINAL_GOVERNOR_EXECUTION_POLICY_PATH)
        if bool(FINAL_GOVERNOR_EXECUTION_POLICY_ENABLE and self.fully_learned_policy_bundle is None):
            if os.path.exists(self.execution_policy_path):
                self.execution_policy_bundle = joblib.load(self.execution_policy_path)
            else:
                logger.warning("SYSTEM execution_policy=OFF reason=missing_model path=%s", self.execution_policy_path)
        self.runtime_state_path = self._repo_path(FINAL_GOVERNOR_RUNTIME_STATE_PATH)
        self.last_exit_bar: int = -10**9
        self.bar_counter: int = 0
        self._load_runtime_state()

        self.micro_bundle: dict | None = None
        self.micro_cfg: MicrostructureSleeveConfig | None = None
        self.trend_bundle: dict | None = None
        self.trend_cfg: TrendSleeveConfig | None = None
        self.event_detector: dict | None = None
        self.event_feature_cols = list(EVENT_DETECTOR_FEATURE_COLS)
        self.regime_predictor_bundle: dict | None = None
        self.regime_predictor_path = self._repo_path(FINAL_GOVERNOR_REGIME_PREDICTOR_MODEL_PATH)
        if bool(FINAL_GOVERNOR_REGIME_PREDICTOR_ENABLE):
            raise RuntimeError(
                "FINAL_GOVERNOR_REGIME_PREDICTOR_ENABLE is removed from active runtime because it emits "
                "forbidden legacy regime features. Use clean_regime4_state24_sticky090_v2_* only."
            )

        self.sniper_actor = None
        self.sniper_ckpt = None
        self.manifest = None
        self.active_regimes: list[str] = []
        if self.fully_learned_policy_bundle is None and bool(
            FINAL_GOVERNOR_SNIPER_ENABLE or FINAL_GOVERNOR_TREND_ENABLE or FINAL_GOVERNOR_MICRO_ENABLE
        ):
            micro_path = self._repo_file_path(FINAL_GOVERNOR_MICRO_MODEL_PATH)
            trend_path = self._repo_file_path(FINAL_GOVERNOR_TREND_MODEL_PATH)
            event_path = self._repo_file_path(FINAL_GOVERNOR_EVENT_DETECTOR_PATH)
            if bool(FINAL_GOVERNOR_MICRO_ENABLE or FINAL_GOVERNOR_TREND_ENABLE) and (
                micro_path is None or trend_path is None or event_path is None
            ):
                logger.warning(
                    "SYSTEM sniper_trend_micro=OFF reason=missing_or_invalid_artifacts micro=%s trend=%s event=%s",
                    self._repo_path(FINAL_GOVERNOR_MICRO_MODEL_PATH),
                    self._repo_path(FINAL_GOVERNOR_TREND_MODEL_PATH),
                    self._repo_path(FINAL_GOVERNOR_EVENT_DETECTOR_PATH),
                )
            elif bool(FINAL_GOVERNOR_MICRO_ENABLE or FINAL_GOVERNOR_TREND_ENABLE):
                self.micro_bundle = joblib.load(micro_path)
                self.micro_cfg = MicrostructureSleeveConfig(**dict(self.micro_bundle.get("config", {})))
                self.trend_bundle = joblib.load(trend_path)
                self.trend_cfg = TrendSleeveConfig(**dict(self.trend_bundle.get("config", {})))
                if self.ddh2_ensemble_enabled:
                    self.micro_cfg = self._ddh2_override_micro_config(self.micro_cfg)
                    self.trend_cfg = self._ddh2_override_trend_config(self.trend_cfg)
                with open(event_path, "rb") as f:
                    self.event_detector = pickle.load(f)
                self.event_feature_cols = list(self.event_detector.get("feature_cols", EVENT_DETECTOR_FEATURE_COLS))
            if bool(FINAL_GOVERNOR_SNIPER_ENABLE):
                sniper_path = self._repo_file_path(FINAL_GOVERNOR_SNIPER_MODEL_PATH)
                manifest_path = self._repo_file_path(FINAL_GOVERNOR_MANIFEST_PATH)
                policy_path = self._repo_file_path(FINAL_GOVERNOR_POLICY_PATH)
                if sniper_path is None or manifest_path is None or policy_path is None:
                    logger.warning(
                        "SYSTEM sniper=OFF reason=missing_or_invalid_artifacts sniper=%s manifest=%s policy=%s",
                        self._repo_path(FINAL_GOVERNOR_SNIPER_MODEL_PATH),
                        self._repo_path(FINAL_GOVERNOR_MANIFEST_PATH),
                        self._repo_path(FINAL_GOVERNOR_POLICY_PATH),
                    )
                else:
                    self.sniper_actor, self.sniper_ckpt = _load_final_sniper_actor(sniper_path, self.device)
                    self.manifest, policy = _load_final_manifest_policy(
                        manifest_path,
                        policy_path,
                        ["bull", "bear", "chop", "whipsaw", "normal"],
                    )
                    self.active_regimes = [r for r in list(policy.active_regimes) if r in {"bull", "bear"}]
        logger.info(
            "SYSTEM governor=FINAL_RUNTIME device=%s omega5=%s alpha2_1=%s lifecycle_v1=%s v21_2=%s v31=%s fully_learned=%s fully_learned_fallback=%s primary_low_conf_tp=%.4f primary_low_conf_thr=%.4f fallback_tp_scale=%.4f fallback_exit_submodel=%s fully_learned_scale=%s clean4=%s tp_sl_score=%s sniper=%s trend=%s micro=%s regime_pred=%s exec_policy=%s alpha3_contract=%s mark_parity=%s cooldown_parity=%s",
            self.device,
            self.omega5_report_path if self.omega5_adapter is not None else "OFF",
            self.alpha2_1_teacher_model_path if self._alpha2_1_available() else "OFF",
            self.lifecycle_v1_model_path if self.lifecycle_v1_payload is not None else "OFF",
            self.v21_2_jackpot_model_path if self._v21_2_jackpot_available() else "OFF",
            self.v31_v27_model_path if self._v31_available() else "OFF",
            self.fully_learned_policy_path if self.fully_learned_policy_bundle is not None else "OFF",
            self.fully_learned_fallback_policy_path if self.fully_learned_fallback_policy_bundle is not None else "OFF",
            float(self.fully_learned_primary_low_conf_tp_scale),
            float(self.fully_learned_primary_low_conf_threshold),
            float(self.fully_learned_fallback_tp_scale),
            str((self.fully_learned_fallback_exit_submodel or {}).get("model_id", "OFF")),
            dict(self.fully_learned_scale_runtime or {}).get("name", "OFF"),
            self.clean_regime4_sticky_path if self.clean_regime4_sticky_bundle is not None else "OFF",
            self.fully_learned_tp_sl_score_path if self.fully_learned_tp_sl_score_bundle is not None else "OFF",
            self._repo_path(FINAL_GOVERNOR_SNIPER_MODEL_PATH) if bool(FINAL_GOVERNOR_SNIPER_ENABLE and self.fully_learned_policy_bundle is None) else "OFF",
            self._repo_path(FINAL_GOVERNOR_TREND_MODEL_PATH) if bool(FINAL_GOVERNOR_TREND_ENABLE and self.fully_learned_policy_bundle is None) else "OFF",
            self._repo_path(FINAL_GOVERNOR_MICRO_MODEL_PATH) if bool(FINAL_GOVERNOR_MICRO_ENABLE and self.fully_learned_policy_bundle is None) else "OFF",
            self.regime_predictor_path if self.regime_predictor_bundle is not None else "OFF",
            self.execution_policy_path if self.execution_policy_bundle is not None else "OFF",
            FINAL_GOVERNOR_ALPHA3_LIVE_CONTRACT_ID,
            bool(FINAL_GOVERNOR_ALPHA3_CSV_MARK_PARITY_ENABLE),
            bool(FINAL_GOVERNOR_ALPHA3_CSV_COOLDOWN_PARITY_ENABLE),
        )

    @staticmethod
    def _repo_path(path: str) -> str:
        p = str(path)
        return p if os.path.isabs(p) else os.path.join(_THIS_DIR, p)

    def _repo_file_path(self, path: str) -> str | None:
        p = self._repo_path(path)
        return p if path and os.path.isfile(p) else None

    def _load_fully_learned_runtime_config(self, path: str) -> dict[str, object]:
        if not str(path or ""):
            return {}
        if not os.path.exists(path):
            raise RuntimeError(f"missing fully learned runtime config: {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise RuntimeError(f"fully learned runtime config must be a JSON object: {path}")
        expected = str(payload.get("model_id", "") or "")
        if expected and expected != FINAL_GOVERNOR_ALPHA7_MODEL_ID:
            raise RuntimeError(
                "fully learned runtime config model_id mismatch: "
                f"{expected} != {FINAL_GOVERNOR_ALPHA7_MODEL_ID}"
            )
        required = [
            "entry_quality_min",
            "entry_conf_min",
            "parent_notional_mult",
            "parent_notional_cap",
            "parent_tp_mult",
            "parent_sl_mult",
            "parent_hold_mult",
            "parent_hold_cap",
            "alpha7_cash_fallthrough_to_alpha3",
        ]
        missing = [k for k in required if k not in payload]
        if missing:
            raise RuntimeError(f"fully learned runtime config missing keys: {missing}")
        return dict(payload)

    def _apply_fully_learned_v31_runtime_config(self) -> None:
        cfg = dict(getattr(self, "fully_learned_runtime_config", {}) or {})
        if not cfg or str(cfg.get("candidate", "")) != "01965_random_alpha7_combo_primary_fallback":
            return
        if not bool(getattr(self, "v31_cfg", {}) or {}):
            raise RuntimeError("alpha7.1-01965 requires v31 deep overlay config")
        deep_notional_mult = float(cfg.get("deep_notional_mult", 1.0) or 1.0)
        deep_tp_mult = float(cfg.get("deep_tp_mult", 1.0) or 1.0)
        deep_sl_mult = float(cfg.get("deep_sl_mult", 1.0) or 1.0)
        deep_hold_mult = float(cfg.get("deep_hold_mult", 1.0) or 1.0)
        deep_trail_activation = float(cfg.get("deep_trail_activation", 0.0) or 0.0)
        if min(deep_notional_mult, deep_tp_mult, deep_sl_mult, deep_hold_mult) <= 0.0:
            raise RuntimeError("alpha7.1-01965 v31 multipliers must be positive")
        self.v31_cfg = {
            **dict(self.v31_cfg),
            "name": "v31_tight_after_24_alpha7_01965_runtime",
            "edge_th": 0.010,
            "margin_th": 0.004,
            "notional": float(1.2 * deep_notional_mult),
            "cooldown": 12,
            "base_tp": float(0.040 * deep_tp_mult),
            "base_sl": float(0.018 * deep_sl_mult),
            "base_hold": int(round(48 * deep_hold_mult)),
            "tp_util_mult": 1.5,
            "sl_vol_mult": 2.4,
            "trail_gap_mult": 0.9,
            "trail_decay": 0.60,
            "hold_decay_start": 24,
            "hold_decay_rate": 0.030,
            "tp_cap": 0.080,
            "sl_cap": 0.040,
            "trail_activation": float(deep_trail_activation),
            "alpha7_01965_runtime_overlay": True,
        }
        deep_stop_cooldown_extra = int(round(float(cfg.get("deep_stop_cooldown_extra", 0) or 0)))
        if deep_stop_cooldown_extra < 0:
            raise RuntimeError("alpha7.1-01965 deep_stop_cooldown_extra must be non-negative")
        if deep_stop_cooldown_extra > 0:
            self.v31_cfg["deep_stop_cooldown_extra"] = int(deep_stop_cooldown_extra)
            self.v31_cfg["deep_stop_cooldown_scope"] = str(
                cfg.get("deep_stop_cooldown_scope", "deep_alpha_only_after_deep_alpha_stop_loss") or ""
            )
        if bool(cfg.get("deep_block_long_in_bear_regime", False)):
            self.v31_cfg["deep_block_long_in_bear_regime"] = True
        if bool(cfg.get("deep_side_specialist_gate", False)):
            for key in ("deep_long_edge_mult", "deep_long_margin_mult", "deep_short_edge_mult", "deep_short_margin_mult"):
                val = float(cfg.get(key, 1.0) or 1.0)
                if not np.isfinite(val) or val <= 0.0:
                    raise RuntimeError(f"alpha7.1-01965 {key} must be positive")
                self.v31_cfg[key] = float(val)
            self.v31_cfg["deep_side_specialist_gate"] = True
            self.v31_cfg["deep_long_block_regimes"] = list(cfg.get("deep_long_block_regimes", []) or [])
            self.v31_cfg["deep_short_block_regimes"] = list(cfg.get("deep_short_block_regimes", []) or [])

    def _load_fully_learned_scale_runtime(self, summary_path: str) -> dict[str, float | str] | None:
        if not bool(FINAL_GOVERNOR_FULLY_LEARNED_SCALE_ENABLE):
            return None
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            best_name = str(summary.get("best_by_selection", "") or "")
            for exp in list(summary.get("experiments", []) or []):
                if str(exp.get("name", "") or "") != best_name:
                    continue
                rt = dict(exp.get("selected_parent_scale_runtime", {}) or {})
                if not rt:
                    return None
                return {
                    "name": str(rt.get("name", "")),
                    "confidence": float(rt.get("confidence", 0.0) or 0.0),
                    "parent_notional_scale": float(rt.get("parent_notional_scale", 1.0) or 1.0),
                    "max_notional": float(rt.get("max_notional", 2.75) or 2.75),
                }
        except Exception as e:
            raise RuntimeError(f"invalid fully learned scale summary: {summary_path}") from e
        return None

    def _validate_ddh2_full_1x_artifacts(self) -> None:
        raise RuntimeError("ddh2_removed")

    def _ddh2_cfg_float(self, key: str, default: float = 0.0) -> float:
        return _safe_float(dict(self.ddh2_config or {}).get(key, default), default)

    def _ddh2_cfg_int(self, key: str, default: int = 0) -> int:
        return int(float(self._ddh2_cfg_float(key, float(default))))

    def _ddh2_override_micro_config(self, cfg: MicrostructureSleeveConfig) -> MicrostructureSleeveConfig:
        values = dict(cfg.__dict__)
        cfg_map = dict(self.ddh2_config or {})
        if self._ddh2_cfg_float("micro_max_notional", 0.0) > 0.0:
            values["max_notional_exposure"] = self._ddh2_cfg_float("micro_max_notional", cfg.max_notional_exposure)
        if self._ddh2_cfg_float("micro_max_leverage", 0.0) > 0.0:
            values["max_leverage"] = self._ddh2_cfg_float("micro_max_leverage", cfg.max_leverage)
        if "micro_entry_confidence" in cfg_map:
            values["entry_confidence"] = self._ddh2_cfg_float("micro_entry_confidence", cfg.entry_confidence)
        if "micro_entry_gap" in cfg_map:
            values["entry_gap"] = self._ddh2_cfg_float("micro_entry_gap", cfg.entry_gap)
        if self._ddh2_cfg_float("micro_whipsaw_mult_cap", 0.0) > 0.0:
            values["whipsaw_notional_mult"] = min(
                float(values.get("whipsaw_notional_mult", cfg.whipsaw_notional_mult)),
                self._ddh2_cfg_float("micro_whipsaw_mult_cap", cfg.whipsaw_notional_mult),
            )
        if self._ddh2_cfg_float("micro_chop_mult_cap", 0.0) > 0.0:
            values["chop_notional_mult"] = min(
                float(values.get("chop_notional_mult", cfg.chop_notional_mult)),
                self._ddh2_cfg_float("micro_chop_mult_cap", cfg.chop_notional_mult),
            )
        return MicrostructureSleeveConfig(**values)

    def _ddh2_override_trend_config(self, cfg: TrendSleeveConfig) -> TrendSleeveConfig:
        values = dict(cfg.__dict__)
        cfg_map = dict(self.ddh2_config or {})
        if self._ddh2_cfg_float("trend_max_notional", 0.0) > 0.0:
            values["max_notional_exposure"] = self._ddh2_cfg_float("trend_max_notional", cfg.max_notional_exposure)
        if self._ddh2_cfg_float("trend_max_leverage", 0.0) > 0.0:
            values["max_leverage"] = self._ddh2_cfg_float("trend_max_leverage", cfg.max_leverage)
        if "trend_entry_confidence" in cfg_map:
            values["entry_confidence"] = self._ddh2_cfg_float("trend_entry_confidence", cfg.entry_confidence)
        if "trend_entry_gap" in cfg_map:
            values["entry_gap"] = self._ddh2_cfg_float("trend_entry_gap", cfg.entry_gap)
        return TrendSleeveConfig(**values)

    def _ddh2_cost_stress_mult(self, meta_router) -> float:
        fee_mult = _safe_float(getattr(meta_router, "trade_fee", 0.0005), 0.0005) / max(0.0005, 1e-12)
        slip_mult = _safe_float(getattr(meta_router, "trade_slip", 0.0002), 0.0002) / max(0.0002, 1e-12)
        return float(max(1.0, fee_mult, slip_mult))

    def _ddh2_extra_gap(self, meta_router, scale: float) -> float:
        return float(max(0.0, self._ddh2_cost_stress_mult(meta_router) - 1.0) * max(0.0, float(scale)))

    @staticmethod
    def _ddh2_trend_edge(side: str, *, no_p: float, long_p: float, short_p: float) -> float:
        if str(side).upper() == "LONG":
            return float(long_p - max(short_p, 0.35 * no_p))
        if str(side).upper() == "SHORT":
            return float(short_p - max(long_p, 0.35 * no_p))
        return 0.0

    def _ddh2_regime_extra_gap(self, meta_router, *, source: str, regime: str) -> float:
        r = str(regime or "").lower()
        if source == "trend":
            if r == "bull":
                return self._ddh2_extra_gap(meta_router, self._ddh2_cfg_float("trend_bull_cost_mult_gap_scale", 0.0))
            if r == "bear":
                return self._ddh2_extra_gap(meta_router, self._ddh2_cfg_float("trend_bear_cost_mult_gap_scale", 0.0))
            return 0.0
        if r == "whipsaw":
            return self._ddh2_extra_gap(meta_router, self._ddh2_cfg_float("micro_whipsaw_cost_mult_gap_scale", 0.0))
        if r == "normal":
            return self._ddh2_extra_gap(meta_router, self._ddh2_cfg_float("micro_normal_cost_mult_gap_scale", 0.0))
        if r == "chop":
            return self._ddh2_extra_gap(meta_router, self._ddh2_cfg_float("micro_chop_cost_mult_gap_scale", 0.0))
        return 0.0

    def _ddh2_fallback_dd_blocks(self, meta_router) -> tuple[bool, dict]:
        block = self._ddh2_cfg_float("fallback_account_dd_block", 0.0)
        release = self._ddh2_cfg_float("fallback_account_dd_release", 0.0)
        stress_mult = self._ddh2_cost_stress_mult(meta_router)
        if stress_mult >= 3.0 and self._ddh2_cfg_float("fallback_cost3_account_dd_block", 0.0) > 0.0:
            block = self._ddh2_cfg_float("fallback_cost3_account_dd_block", block)
            release = self._ddh2_cfg_float("fallback_cost3_account_dd_release", release)
        elif stress_mult >= 2.0 and self._ddh2_cfg_float("fallback_cost2_account_dd_block", 0.0) > 0.0:
            block = self._ddh2_cfg_float("fallback_cost2_account_dd_block", block)
            release = self._ddh2_cfg_float("fallback_cost2_account_dd_release", release)
        equity = max(_safe_float(getattr(meta_router, "cur_equity", 1.0), 1.0), 1e-12)
        peak = max(_safe_float(getattr(meta_router, "peak_equity", 1.0), 1.0), 1e-12)
        account_dd = max(0.0, 1.0 - equity / peak)
        active = bool(self.ddh2_fallback_dd_block_active)
        if block <= 0.0:
            active = False
        elif active and release > 0.0 and release < block and account_dd <= release:
            active = False
        elif (not active) and account_dd >= block:
            active = True
        elif release <= 0.0 or release >= block:
            active = bool(account_dd >= block)
        changed = active != bool(self.ddh2_fallback_dd_block_active)
        self.ddh2_fallback_dd_block_active = bool(active)
        if changed:
            self._save_runtime_state()
        return bool(active), {
            "enabled": bool(self.ddh2_ensemble_enabled and block > 0.0),
            "active": bool(active),
            "account_dd": float(account_dd),
            "block": float(block),
            "release": float(release),
            "stress_mult": float(stress_mult),
        }

    def _ddh2_should_continue_after_lifecycle(self, decision: tuple[int, float, float, float, dict, str]) -> bool:
        if not self.ddh2_ensemble_enabled:
            return False
        try:
            action, exposure, fraction, _exec_lev, info, _regime = decision
        except Exception:
            return False
        if int(action) != 0 or float(exposure or 0.0) > 1e-12 or float(fraction or 0.0) > 1e-12:
            return False
        source = str(dict(info or {}).get("source", ""))
        signal = str(dict(info or {}).get("position_signal", ""))
        return bool(signal in {"HOLD", ""} and (source.startswith("lifecycle_v1|") or source.startswith("v22_1")))

    @staticmethod
    def _class_prob(proba: np.ndarray, classes: list[int], idx: int, cls: int) -> float:
        return float(proba[idx, classes.index(cls)]) if cls in classes else 0.0

    @staticmethod
    def _action_from_side(side: str) -> int:
        s = str(side or "").upper()
        if s == "LONG":
            return 1
        if s == "SHORT":
            return 2
        return 0

    def _raw_regime_from_row(self, row) -> str:
        vals: dict[str, float] = {}
        for col in self.REGIME_COLS:
            try:
                vals[col] = float(row.get(col, 0.0) or 0.0)
            except Exception:
                vals[col] = 0.0
        if not vals or max(abs(v) for v in vals.values()) <= 1e-12:
            return ""
        return max(vals, key=vals.get).replace("regime_", "")

    def _apply_regime_predictor(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        trace: dict[str, object] = {
            "enabled": False,
            "source": "removed",
            "legacy_regime_v2": "forbidden",
            "reason": "legacy_clean_regime_v4_removed_from_active_runtime",
        }
        if not bool(FINAL_GOVERNOR_REGIME_PREDICTOR_ENABLE):
            trace["reason"] = "disabled"
            out.attrs["regime_predictor_trace"] = trace
            return out
        raise RuntimeError(
            "legacy clean_regime_2024_unsup_v4 predictor is forbidden in active runtime. "
            "Disable FINAL_GOVERNOR_REGIME_PREDICTOR_ENABLE and use state24 sticky v2 features."
        )

    def _clean_regime4_transform_frame(self, frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
        if (
            _with_clean_regime4_raw_state12 is None
            or _clean_regime4_class_proba is None
            or _output_clean_regime4_frame is None
        ):
            raise RuntimeError("clean_regime4 transform helpers unavailable")
        canonical = frame.copy()
        for col in (
            "ai_dir_edge",
            "ai_flow_pressure",
            "ai_vol_regime_pct",
            "ai_flow_exhaustion",
            "ai_dir_entropy",
            "ai_flow_flip_prob",
            "tide_vol_zscore",
        ):
            if col in canonical.columns:
                canonical = canonical.drop(columns=[col])
        base = _with_clean_regime4_state7(canonical) if _with_clean_regime4_state7 is not None else canonical
        work = _with_clean_regime4_raw_state12(base)
        cols = [str(c) for c in list(bundle.get("feature_cols") or [])]
        medians = pd.Series(dict(bundle.get("feature_medians", {}) or {}), dtype="float64")
        x_raw = work.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x_raw = x_raw.fillna(medians).fillna(0.0)
        obs = bundle["scaler"].transform(x_raw)
        state = bundle["model"].filter_proba(obs)
        proba = _clean_regime4_class_proba(state, np.asarray(bundle["state_class_matrix"], dtype=np.float64))
        clean = _output_clean_regime4_frame(work["timestamp"], proba, work)
        clean.index = frame.index
        return clean

    def _apply_clean_regime4_sticky(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        trace: dict[str, object] = {
            "enabled": False,
            "source": "clean_regime4_state24_sticky090_v2",
            "lookahead_policy": "transform_only_current_and_past_frame_no_fit",
            "prefix": CLEAN_REGIME4_STICKY_RUNTIME_PREFIX,
            "artifact_prefix": CLEAN_REGIME4_STICKY_RUNTIME_PREFIX,
        }
        if not bool(FINAL_GOVERNOR_CLEAN_REGIME4_STICKY_ENABLE):
            trace["reason"] = "disabled"
            out.attrs["clean_regime4_sticky_trace"] = trace
            return out
        if self.clean_regime4_sticky_bundle is None:
            trace["reason"] = "missing_bundle_or_transform"
            trace["path"] = getattr(self, "clean_regime4_sticky_path", "")
            out.attrs["clean_regime4_sticky_trace"] = trace
            return out
        try:
            clean = self._clean_regime4_transform_frame(out, self.clean_regime4_sticky_bundle)
            clean_cols = [c for c in clean.columns if str(c).startswith(CLEAN_REGIME4_STICKY_PREFIX)]
            if not clean_cols:
                raise RuntimeError("clean_regime4 transform returned no source-prefix columns")
            for col in clean_cols:
                suffix = str(col)[len(CLEAN_REGIME4_STICKY_PREFIX) :]
                out[f"{CLEAN_REGIME4_STICKY_RUNTIME_PREFIX}{suffix}"] = clean[col].to_numpy()
            last = out.iloc[-1] if len(out) else pd.Series(dtype=float)
            trace.update(
                {
                    "enabled": True,
                    "path": self.clean_regime4_sticky_path,
                    "artifact_feature_col_count": int(sum(str(c).startswith(CLEAN_REGIME4_STICKY_RUNTIME_PREFIX) for c in out.columns)),
                    "confidence": float(last.get(f"{CLEAN_REGIME4_STICKY_RUNTIME_PREFIX}confidence", 0.0) or 0.0),
                    "entropy": float(last.get(f"{CLEAN_REGIME4_STICKY_RUNTIME_PREFIX}entropy", 0.0) or 0.0),
                    "margin": float(last.get(f"{CLEAN_REGIME4_STICKY_RUNTIME_PREFIX}margin", 0.0) or 0.0),
                    "state_count": int(self.clean_regime4_sticky_bundle.get("state_count", 0) or 0),
                    "sticky": float(self.clean_regime4_sticky_bundle.get("sticky", 0.0) or 0.0),
                }
            )
        except Exception as e:
            trace["reason"] = "transform_failed"
            trace["error"] = str(e)
            logger.warning("SYSTEM clean_regime4_sticky=OFF reason=transform_failed err=%s", e)
        out.attrs["clean_regime4_sticky_trace"] = trace
        return out

    def _apply_fully_learned_tp_sl_action_score(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        trace: dict[str, object] = {
            "enabled": False,
            "source": "alpha4_2_tp_sl_action_score",
            "lookahead_policy": "transform_only_current_and_past_frame_no_fit",
        }
        bundle = self.fully_learned_tp_sl_score_bundle
        if self.fully_learned_policy_bundle is None:
            trace["reason"] = "fully_learned_policy_disabled"
            out.attrs["tp_sl_action_score_trace"] = trace
            return out
        if not isinstance(bundle, dict):
            trace["reason"] = "missing_bundle"
            trace["path"] = getattr(self, "fully_learned_tp_sl_score_path", "")
            if self.fully_learned_policy_bundle is not None:
                raise RuntimeError(f"missing tp_sl_action_score bundle: {trace['path']}")
            out.attrs["tp_sl_action_score_trace"] = trace
            return out
        try:
            cols = [str(c) for c in list(bundle.get("feature_cols") or [])]
            derivable = {"side_hint", "mom_21d", "abs_mom_21d", "mom_3d", "abs_mom_3d", "mom_1d", "abs_mom_1d"}
            missing = [c for c in cols if c not in derivable and c not in out.columns]
            if missing:
                raise RuntimeError(f"tp_sl_action_score missing feature columns: {missing[:30]}")
            close = (
                pd.to_numeric(out["close"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .to_numpy(dtype=np.float64)
            )
            x = prepare_fully_learned_governor_features(out, side_hint=0, close=close, feature_cols=cols, strict=True)
            long_edge = np.asarray(bundle["long_model"].predict(x), dtype=np.float64)
            short_edge = np.asarray(bundle["short_model"].predict(x), dtype=np.float64)
            best = np.maximum(long_edge, short_edge)
            score = np.zeros(len(out), dtype=np.float64)
            long_mask = (best > 0.0) & (long_edge >= short_edge)
            short_mask = (best > 0.0) & (short_edge > long_edge)
            score[long_mask] = long_edge[long_mask]
            score[short_mask] = -short_edge[short_mask]
            out["tp_sl_action_score"] = np.clip(score, -1.0, 1.0).astype(np.float32)
            last = float(out["tp_sl_action_score"].iloc[-1]) if len(out) else 0.0
            trace.update(
                {
                    "enabled": True,
                    "path": self.fully_learned_tp_sl_score_path,
                    "feature_col_count": len(cols),
                    "score": last,
                    "horizon": int(bundle.get("horizon", 0) or 0),
                    "model_id": str(bundle.get("model_id", "")),
                }
            )
        except Exception as e:
            trace["reason"] = "transform_failed"
            trace["error"] = str(e)
            logger.warning("SYSTEM tp_sl_action_score=OFF reason=transform_failed err=%s", e)
        out.attrs["tp_sl_action_score_trace"] = trace
        return out

    @staticmethod
    def _regime_predictor_blocks_entry(trace: dict) -> bool:
        if not bool(FINAL_GOVERNOR_REGIME_PREDICTOR_BLOCK):
            return False
        if not bool(trace.get("enabled", False)):
            return False
        if not bool(trace.get("transition", False)):
            return False
        try:
            return float(trace.get("confidence", 0.0) or 0.0) >= float(FINAL_GOVERNOR_REGIME_PREDICTOR_BLOCK_CONF)
        except Exception:
            return False

    def _inject_live_model_outputs(
        self,
        frame: pd.DataFrame,
        *,
        m7_last: dict | None,
        trend_signal: dict | None,
    ) -> pd.DataFrame:
        out = frame.copy()
        last_idx = out.index[-1]
        for payload in (m7_last, trend_signal):
            if not isinstance(payload, dict):
                continue
            for key, val in payload.items():
                try:
                    if isinstance(val, (dict, list, tuple)):
                        continue
                    fval = float(val)
                    if np.isfinite(fval):
                        out.at[last_idx, str(key)] = fval
                except Exception:
                    continue
        return out

    def _ensure_regime_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        needs_raw = any(col not in out.columns for col in self.REGIME_COLS)
        if not needs_raw:
            vals = out[list(self.REGIME_COLS)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            needs_raw = bool((vals.sum(axis=1).abs() <= 1e-12).any())
        if needs_raw:
            try:
                out = RegimeEngine().compute(out)
            except Exception as e:
                logger.warning("Final Governor regime feature restore failed: %s", e)
                for col in self.REGIME_COLS:
                    if col not in out.columns:
                        out[col] = 0.0
        vals = out[list(self.REGIME_COLS)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        zero_mask = vals.sum(axis=1).abs() <= 1e-12
        if bool(zero_mask.any()):
            out.loc[zero_mask, "regime_normal"] = 1.0
        return out

    def _prepare_frame(
        self,
        processed_df: pd.DataFrame,
        *,
        m7_last: dict | None,
        trend_signal: dict | None,
    ) -> pd.DataFrame:
        frame = processed_df.tail(self.window_bars).copy().reset_index(drop=True)
        if len(frame) < 50:
            raise ValueError(f"final governor requires at least 50 bars, got {len(frame)}")
        omega_active = self.omega4_6_2_source_parent_adapter is not None
        frame = self._inject_live_model_outputs(frame, m7_last=m7_last, trend_signal=trend_signal)
        if omega_active:
            forbidden_cols = [
                c
                for c in frame.columns
                if str(c) == "tp_sl_action_score"
                or str(c).startswith("teacher_")
                or str(c).startswith("clean_regime4_")
                or str(c).startswith("clean_regime_2024_unsup_v4_")
            ]
            if forbidden_cols:
                frame = frame.drop(columns=forbidden_cols)
        if not omega_active and append_side_teacher_features is not None:
            try:
                frame = append_side_teacher_features(frame)
            except Exception as e:
                logger.warning("SYSTEM teacher_side_features=OFF reason=transform_failed err=%s", e)
        frame = self._ensure_regime_features(frame)
        if not omega_active:
            frame = self._apply_clean_regime4_sticky(frame)
            frame = self._apply_regime_predictor(frame)
            frame = self._apply_fully_learned_tp_sl_action_score(frame)
        active_ai_cols = {
            col
            for group in FINAL_GOVERNOR_AI_FEATURE_GROUPS
            for col in EnsemblePredictor.AI_FEATURE_COLUMNS.get(str(group).lower(), [])
        }
        missing_ai_cols: list[str] = []
        nonfinite_ai_cols: list[str] = []
        for col in self.BEST_AI_FEATURE_COLS:
            if col not in frame.columns:
                if col in active_ai_cols:
                    missing_ai_cols.append(col)
                continue
            vals = pd.to_numeric(frame[col], errors="coerce")
            if not np.isfinite(float(vals.iloc[-1]) if len(vals) else float("nan")):
                if col in active_ai_cols:
                    nonfinite_ai_cols.append(col)
        if missing_ai_cols or nonfinite_ai_cols:
            raise RuntimeError(
                "final governor AI feature contract failed "
                f"missing={sorted(set(missing_ai_cols))[:8]} nonfinite={sorted(set(nonfinite_ai_cols))[:8]}"
            )
        if "evt_candidate_label" not in frame.columns:
            frame["evt_candidate_label"] = 0
        if self.event_detector is not None:
            for col in self.event_feature_cols:
                if col not in frame.columns:
                    frame[col] = 0.0
            x = frame[self.event_feature_cols].replace([np.inf, -np.inf], np.nan)
            med = x.median(numeric_only=True)
            x = x.fillna(med).fillna(0.0)
            probs = self.event_detector["model"].predict_proba(x)
            frame["evt_det_available"] = 1
            frame["evt_det_none_prob"] = probs[:, 0]
            frame["evt_det_long_prob"] = probs[:, 1]
            frame["evt_det_short_prob"] = probs[:, 2]
            frame["evt_det_prob_max"] = probs.max(axis=1)
            frame["evt_det_edge"] = frame["evt_det_long_prob"] - frame["evt_det_short_prob"]
        else:
            frame["evt_det_available"] = 0
            frame["evt_det_none_prob"] = 1.0
            frame["evt_det_long_prob"] = 0.0
            frame["evt_det_short_prob"] = 0.0
            frame["evt_det_prob_max"] = 1.0
            frame["evt_det_edge"] = 0.0
        frame = _ensure_final_event_aliases(frame)
        self.last_prepared_frame_for_health = frame.tail(min(len(frame), 1200)).copy()
        return frame

    def _decision_frame_cache_key(
        self,
        frame: pd.DataFrame,
        *,
        bundle: dict | None,
        feature_cols: list[str] | None = None,
    ) -> tuple:
        ts = ""
        if "timestamp" in frame.columns and len(frame):
            ts = str(frame["timestamp"].iloc[-1])
        close = _safe_float(frame["close"].iloc[-1], 0.0) if "close" in frame.columns and len(frame) else 0.0
        fingerprint = 0
        if len(frame):
            try:
                fingerprint = int(pd.util.hash_pandas_object(frame.tail(1), index=True).iloc[0])
            except Exception:
                fingerprint = hash(tuple(str(v) for v in frame.tail(1).iloc[0].to_dict().values()))
        return (id(bundle), len(frame), ts, float(close), int(fingerprint), tuple(feature_cols or []))

    def _fully_learned_decision_frame(
        self,
        frame: pd.DataFrame,
        *,
        bundle: dict | None = None,
        feature_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        policy = bundle if bundle is not None else self.fully_learned_policy_bundle
        if policy is None:
            return None
        if policy is self.fully_learned_policy_bundle or bundle is not None:
            ok, audit = self._fully_learned_contract_ok(frame, policy)
            if policy is self.fully_learned_policy_bundle:
                self.last_fully_learned_contract_audit = audit
                self.fully_learned_contract_blocked = not bool(ok)
            if not ok:
                raise RuntimeError(
                    "fully learned feature contract missing: "
                    f"{json.dumps(audit, ensure_ascii=False, default=str)[:1200]}"
                )
        elif bundle is None:
            self.fully_learned_contract_blocked = False
        close = (
            pd.to_numeric(frame["close"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .to_numpy(dtype=np.float64)
        )
        cols = list(feature_cols or list(dict(policy or {}).get("feature_cols") or []))
        key = self._decision_frame_cache_key(frame, bundle=policy, feature_cols=cols)
        cache = getattr(self, "_fully_learned_decision_frame_cache", None)
        if isinstance(cache, dict) and cache.get("key") == key:
            return cache.get("value")
        features = prepare_fully_learned_governor_features(
            frame,
            side_hint=0,
            close=close,
            feature_cols=cols or None,
            strict=True,
        )
        if len(close) == len(features):
            features = features.copy()
            features["close"] = close
        decisions = predict_fully_learned_governor_frame(policy, features, close=close, strict=True)
        if bundle is None and policy is self.fully_learned_policy_bundle:
            decisions = self._scale_fully_learned_decisions(decisions)
        value = (decisions, features)
        self._fully_learned_decision_frame_cache = {"key": key, "value": value}
        return value

    def _scale_fully_learned_decisions(self, decisions: pd.DataFrame) -> pd.DataFrame:
        return self._scale_fully_learned_decisions_with_runtime(decisions, self.fully_learned_scale_runtime)

    def _scale_fully_learned_decisions_with_runtime(
        self,
        decisions: pd.DataFrame,
        rt: dict[str, float | str] | None,
    ) -> pd.DataFrame:
        if not isinstance(rt, dict):
            return decisions
        scale = float(rt.get("parent_notional_scale", 1.0) or 1.0)
        max_notional = float(rt.get("max_notional", 2.75) or 2.75)
        if abs(scale - 1.0) < 1e-12:
            return decisions
        out = decisions.copy()
        active = (out["action"].astype(int).to_numpy() != 0) & (out["side"].astype(int).to_numpy() != 0)
        notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        leverage = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        scaled = np.minimum(notional * scale, max_notional)
        out.loc[active, "notional_exposure"] = scaled[active]
        out.loc[active, "position_fraction"] = scaled[active] / np.maximum(leverage[active], 1e-12)
        return out

    def _fully_learned_contract_ok(self, frame: pd.DataFrame, policy: dict) -> tuple[bool, dict[str, object]]:
        feature_cols = [str(c) for c in list(dict(policy or {}).get("feature_cols") or [])]
        derivable = {"side_hint", "mom_21d", "abs_mom_21d", "mom_3d", "abs_mom_3d", "mom_1d", "abs_mom_1d"}
        missing_model_features = [c for c in feature_cols if c not in derivable and c not in frame.columns]
        critical = [
            c
            for c in feature_cols
            if c == "tp_sl_action_score" or c.startswith(CLEAN_REGIME4_STICKY_RUNTIME_PREFIX)
        ]
        missing = [c for c in critical if c not in frame.columns]
        zero_like = []
        advisory_zero_like = []
        if len(frame):
            for col in critical:
                if col not in frame.columns:
                    continue
                vals = pd.to_numeric(frame[col].tail(min(len(frame), 120)), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if float(vals.abs().sum()) <= 1e-12:
                    if col == "tp_sl_action_score":
                        advisory_zero_like.append(col)
                    else:
                        zero_like.append(col)
        audit = {
            "feature_count": len(feature_cols),
            "missing_model_features": missing_model_features[:40],
            "missing_model_feature_count": len(missing_model_features),
            "critical_feature_count": len(critical),
            "missing_critical": missing[:20],
            "missing_critical_count": len(missing),
            "zero_like_critical": zero_like[:20],
            "zero_like_critical_count": len(zero_like),
            "advisory_zero_like": advisory_zero_like[:20],
            "advisory_zero_like_count": len(advisory_zero_like),
            "clean_regime4_sticky": dict(frame.attrs.get("clean_regime4_sticky_trace", {}) or {}),
            "tp_sl_action_score": dict(frame.attrs.get("tp_sl_action_score_trace", {}) or {}),
        }
        return len(missing_model_features) == 0 and len(missing) == 0 and len(zero_like) == 0, audit

    @staticmethod
    def _fully_learned_decision_active(dec: pd.Series | None) -> bool:
        if dec is None:
            return False
        try:
            return int(dec.action) != int(FULLY_LEARNED_ACTION_CASH) and int(dec.side) != 0
        except Exception:
            return False

    @staticmethod
    def _normalize_fully_learned_fallback_exit_submodel(payload: dict[str, object]) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise RuntimeError("fallback exit submodel must be a JSON object")
        model_id = str(payload.get("model_id", "")).strip()
        if not model_id:
            raise RuntimeError("fallback exit submodel missing model_id")
        mode = str(payload.get("mode", "")).strip()
        allowed_modes = {
            "global_tp",
            "low_conf_tp",
            "low_qual_tp",
            "low_conf_or_qual_tp",
            "low_conf_and_qual_tp",
            "high_conf_keep_else_tp",
        }
        if mode not in allowed_modes:
            raise RuntimeError(f"fallback exit submodel invalid mode={mode}")
        tp_scale = float(payload.get("tp_scale", 0.0) or 0.0)
        if not np.isfinite(tp_scale) or tp_scale <= 0.0:
            raise RuntimeError(f"fallback exit submodel invalid tp_scale={tp_scale}")
        conf_thr = float(payload.get("conf_thr", 0.0) or 0.0)
        qual_thr = float(payload.get("qual_thr", 0.0) or 0.0)
        sl_scale = float(payload.get("sl_scale", 1.0) or 1.0)
        hold_cap = int(float(payload.get("hold_cap", 0) or 0))
        if not np.isfinite(conf_thr) or not np.isfinite(qual_thr):
            raise RuntimeError("fallback exit submodel invalid conf_thr/qual_thr")
        if not np.isfinite(sl_scale) or sl_scale <= 0.0:
            raise RuntimeError(f"fallback exit submodel invalid sl_scale={sl_scale}")
        if hold_cap < 0:
            raise RuntimeError(f"fallback exit submodel invalid hold_cap={hold_cap}")
        return {
            "model_id": model_id,
            "mode": mode,
            "tp_scale": tp_scale,
            "conf_thr": conf_thr,
            "qual_thr": qual_thr,
            "sl_scale": sl_scale,
            "hold_cap": hold_cap,
        }

    def _apply_fully_learned_runtime_config(self, dec: pd.Series) -> pd.Series:
        cfg = dict(self.fully_learned_runtime_config or {})
        if not cfg:
            return dec
        out = dec.copy()
        active = self._fully_learned_decision_active(out)
        if active:
            q_min = float(cfg.get("entry_quality_min", -999.0) or -999.0)
            conf_min = float(cfg.get("entry_conf_min", 0.0) or 0.0)
            quality = float(out.get("quality_score", 0.0) or 0.0)
            confidence = float(out.get("confidence", 0.0) or 0.0)
            if quality < q_min or confidence < conf_min:
                out.loc["action"] = int(FULLY_LEARNED_ACTION_CASH)
                out.loc["side"] = 0
                active = False
        if active:
            notional = float(out.get("notional_exposure", 0.0) or 0.0)
            leverage = float(out.get("leverage", 1.0) or 1.0)
            take_profit = float(out.get("take_profit", 0.0) or 0.0)
            stop_loss = float(out.get("stop_loss", 0.0) or 0.0)
            max_hold = int(float(out.get("max_hold_bars", 0) or 0))
            notional = float(
                np.clip(
                    notional * float(cfg.get("parent_notional_mult", 1.0) or 1.0),
                    0.0,
                    float(cfg.get("parent_notional_cap", notional) or notional),
                )
            )
            out.loc["notional_exposure"] = notional
            out.loc["take_profit"] = float(
                np.clip(take_profit * float(cfg.get("parent_tp_mult", 1.0) or 1.0), 0.001, 1.5)
            )
            out.loc["stop_loss"] = float(
                np.clip(stop_loss * float(cfg.get("parent_sl_mult", 1.0) or 1.0), 0.001, 0.30)
            )
            out.loc["max_hold_bars"] = int(
                np.clip(
                    round(max_hold * float(cfg.get("parent_hold_mult", 1.0) or 1.0)),
                    1,
                    float(cfg.get("parent_hold_cap", max_hold or 1) or (max_hold or 1)),
                )
            )
            out.loc["position_fraction"] = float(notional / max(leverage, 1e-12))
        return out

    def _fully_learned_cash_falls_through_to_alpha3(self) -> bool:
        return bool(dict(self.fully_learned_runtime_config or {}).get("alpha7_cash_fallthrough_to_alpha3", False))

    def _fully_learned_runtime_float(self, key: str, default: float) -> float:
        val = dict(self.fully_learned_runtime_config or {}).get(key, default)
        try:
            out = float(val)
        except Exception:
            return float(default)
        return out if np.isfinite(out) else float(default)

    def _fully_learned_runtime_int(self, key: str, default: int) -> int:
        return int(round(self._fully_learned_runtime_float(key, float(default))))

    @staticmethod
    def _fully_learned_time_sl_mult(hold_bars: int, early_bars: int, early_sl_mult: float) -> float:
        if early_bars <= 0 or hold_bars >= early_bars:
            return 1.0
        frac = 1.0 - float(max(0, hold_bars)) / float(max(early_bars, 1))
        return 1.0 + max(0.0, float(early_sl_mult) - 1.0) * frac

    @staticmethod
    def _fully_learned_row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
        try:
            val = float(row.get(col, default))
        except Exception:
            return float(default)
        return val if np.isfinite(val) else float(default)

    def _fully_learned_regime_bad(self, row: pd.Series) -> float:
        vals = []
        for col in (
            "regime_bear_id",
            "regime_whipsaw_id",
            "whipsaw_prob",
            "risk_off_prob",
            "instability_prob",
        ):
            if col in row.index:
                vals.append(self._fully_learned_row_float(row, col, 0.0))
        return float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 0.0

    def _fully_learned_flow_bad(self, row: pd.Series, side: int) -> float:
        side_sign = 1.0 if int(side) > 0 else -1.0
        adverse = [
            -(self._fully_learned_row_float(row, "net_taker_ratio", 0.0) * side_sign),
            -(self._fully_learned_row_float(row, "taker_acceleration", 0.0) * side_sign),
            -(self._fully_learned_row_float(row, "ofi_acceleration", 0.0) * side_sign),
            -(self._fully_learned_row_float(row, "ai_flow_pressure", 0.0) * side_sign),
        ]
        return float(np.mean(adverse))

    def _apply_fully_learned_primary_overlays(self, dec: pd.Series) -> pd.Series:
        out = dec.copy()
        conf = float(out.get("confidence", 0.0) or 0.0)
        tp_before = float(out.get("take_profit", 0.0) or 0.0)
        if (
            tp_before > 0.0
            and conf < self.fully_learned_primary_low_conf_threshold
            and abs(self.fully_learned_primary_low_conf_tp_scale - 1.0) > 1e-12
        ):
            out.loc["take_profit"] = float(tp_before * self.fully_learned_primary_low_conf_tp_scale)
        return out

    def _apply_fully_learned_fallback_overlays(self, dec: pd.Series) -> pd.Series:
        out = dec.copy()
        tp_before = float(out.get("take_profit", 0.0) or 0.0)
        sub = self.fully_learned_fallback_exit_submodel
        if sub is None:
            if tp_before > 0.0 and abs(self.fully_learned_fallback_tp_scale - 1.0) > 1e-12:
                out.loc["take_profit"] = float(tp_before * self.fully_learned_fallback_tp_scale)
            return out
        if not self._fully_learned_decision_active(out):
            return out
        conf = float(out.get("confidence", 0.0) or 0.0)
        qual = float(out.get("quality_score", 0.0) or 0.0)
        conf_thr = float(sub["conf_thr"])
        qual_thr = float(sub["qual_thr"])
        mode = str(sub["mode"])
        apply_tp = False
        if mode == "global_tp":
            apply_tp = True
        elif mode == "low_conf_tp":
            apply_tp = conf < conf_thr
        elif mode == "low_qual_tp":
            apply_tp = qual < qual_thr
        elif mode == "low_conf_or_qual_tp":
            apply_tp = (conf < conf_thr) or (qual < qual_thr)
        elif mode == "low_conf_and_qual_tp":
            apply_tp = (conf < conf_thr) and (qual < qual_thr)
        elif mode == "high_conf_keep_else_tp":
            apply_tp = not ((conf >= conf_thr) and (qual >= qual_thr))
        if apply_tp and tp_before > 0.0:
            out.loc["take_profit"] = float(tp_before * float(sub["tp_scale"]))
            sl_before = float(out.get("stop_loss", 0.0) or 0.0)
            sl_scale = float(sub["sl_scale"])
            if sl_before > 0.0 and abs(sl_scale - 1.0) > 1e-12:
                out.loc["stop_loss"] = float(sl_before * sl_scale)
            hold_cap = int(sub["hold_cap"])
            if hold_cap > 0:
                hold_before = int(float(out.get("max_hold_bars", 0) or 0))
                out.loc["max_hold_bars"] = int(max(1, min(hold_before, hold_cap)))
        return out

    def _fully_learned_latest_decision(self, frame: pd.DataFrame) -> pd.Series | None:
        self.last_fully_learned_selection_trace = {"mode": "primary_only"}
        result = self._fully_learned_decision_frame(frame)
        if result is None:
            return None
        decisions, _features = result
        if len(decisions) == 0:
            return None
        primary_raw = decisions.iloc[-1]
        primary_overlay = self._apply_fully_learned_primary_overlays(primary_raw)
        if self._fully_learned_decision_active(primary_overlay):
            primary = self._apply_fully_learned_runtime_config(primary_overlay)
            self.last_fully_learned_selection_trace = {
                "mode": "primary",
                "primary_model": os.path.basename(str(self.fully_learned_policy_path)),
                "fallback_enabled": bool(self.fully_learned_fallback_policy_bundle is not None),
                "primary_action": int(primary.action),
                "primary_side": int(primary.side),
                "primary_confidence": float(primary.confidence),
                "primary_take_profit_before": float(primary_raw.take_profit),
                "primary_take_profit_after_overlay": float(primary_overlay.take_profit),
                "primary_take_profit_after": float(primary.take_profit),
                "primary_low_conf_threshold": float(self.fully_learned_primary_low_conf_threshold),
                "primary_low_conf_tp_scale": float(self.fully_learned_primary_low_conf_tp_scale),
                "runtime_config": dict(self.fully_learned_runtime_config or {}),
            }
            if self._fully_learned_decision_active(primary):
                return primary
            if self._fully_learned_cash_falls_through_to_alpha3():
                self.last_fully_learned_selection_trace["mode"] = "primary_blocked_alpha3_fallthrough"
                return None
            return primary
        primary = primary_overlay
        fallback = self.fully_learned_fallback_policy_bundle
        if fallback is None:
            self.last_fully_learned_selection_trace = {
                "mode": "primary_cash",
                "primary_model": os.path.basename(str(self.fully_learned_policy_path)),
                "fallback_enabled": False,
                "primary_action": int(primary.action),
                "primary_side": int(primary.side),
                "primary_confidence": float(primary.confidence),
                "primary_take_profit_before": float(primary_raw.take_profit),
                "primary_take_profit_after": float(primary.take_profit),
                "primary_low_conf_threshold": float(self.fully_learned_primary_low_conf_threshold),
                "primary_low_conf_tp_scale": float(self.fully_learned_primary_low_conf_tp_scale),
            }
            if self._fully_learned_cash_falls_through_to_alpha3():
                self.last_fully_learned_selection_trace["mode"] = "primary_cash_alpha3_fallthrough"
                return None
            return primary
        fb_result = self._fully_learned_decision_frame(frame, bundle=fallback)
        if fb_result is None:
            self.last_fully_learned_selection_trace = {
                "mode": "primary_cash_fallback_unavailable",
                "primary_model": os.path.basename(str(self.fully_learned_policy_path)),
                "fallback_model": os.path.basename(str(self.fully_learned_fallback_policy_path)),
                "primary_action": int(primary.action),
                "primary_side": int(primary.side),
                "primary_confidence": float(primary.confidence),
                "primary_take_profit_before": float(primary_raw.take_profit),
                "primary_take_profit_after": float(primary.take_profit),
                "primary_low_conf_threshold": float(self.fully_learned_primary_low_conf_threshold),
                "primary_low_conf_tp_scale": float(self.fully_learned_primary_low_conf_tp_scale),
            }
            return primary
        fb_decisions, _fb_features = fb_result
        if len(fb_decisions) == 0:
            return primary
        fallback_decisions = self._scale_fully_learned_decisions_with_runtime(
            fb_decisions,
            self.fully_learned_fallback_scale_runtime,
        )
        secondary_raw = fallback_decisions.iloc[-1]
        secondary_overlay = self._apply_fully_learned_fallback_overlays(secondary_raw)
        if self._fully_learned_decision_active(secondary_overlay):
            secondary = self._apply_fully_learned_runtime_config(secondary_overlay)
            self.last_fully_learned_selection_trace = {
                "mode": "fallback",
                "primary_model": os.path.basename(str(self.fully_learned_policy_path)),
                "fallback_model": os.path.basename(str(self.fully_learned_fallback_policy_path)),
                "fallback_scale_runtime": dict(self.fully_learned_fallback_scale_runtime or {}),
                "fallback_tp_scale": float(self.fully_learned_fallback_tp_scale),
                "fallback_exit_submodel": str((self.fully_learned_fallback_exit_submodel or {}).get("model_id", "OFF")),
                "primary_action": int(primary.action),
                "primary_side": int(primary.side),
                "primary_confidence": float(primary.confidence),
                "primary_take_profit_before": float(primary_raw.take_profit),
                "primary_take_profit_after": float(primary.take_profit),
                "primary_low_conf_threshold": float(self.fully_learned_primary_low_conf_threshold),
                "primary_low_conf_tp_scale": float(self.fully_learned_primary_low_conf_tp_scale),
                "fallback_action": int(secondary.action),
                "fallback_side": int(secondary.side),
                "fallback_take_profit_before": float(secondary_raw.take_profit),
                "fallback_take_profit_after_overlay": float(secondary_overlay.take_profit),
                "fallback_take_profit_after": float(secondary.take_profit),
                "runtime_config": dict(self.fully_learned_runtime_config or {}),
            }
            if self._fully_learned_decision_active(secondary):
                return secondary
            if self._fully_learned_cash_falls_through_to_alpha3():
                self.last_fully_learned_selection_trace["mode"] = "fallback_blocked_alpha3_fallthrough"
                return None
            return secondary
        secondary = secondary_overlay
        self.last_fully_learned_selection_trace = {
            "mode": "cash",
            "primary_model": os.path.basename(str(self.fully_learned_policy_path)),
            "fallback_model": os.path.basename(str(self.fully_learned_fallback_policy_path)),
            "fallback_tp_scale": float(self.fully_learned_fallback_tp_scale),
            "fallback_exit_submodel": str((self.fully_learned_fallback_exit_submodel or {}).get("model_id", "OFF")),
            "primary_action": int(primary.action),
            "primary_side": int(primary.side),
            "primary_confidence": float(primary.confidence),
            "primary_take_profit_before": float(primary_raw.take_profit),
            "primary_take_profit_after": float(primary.take_profit),
            "primary_low_conf_threshold": float(self.fully_learned_primary_low_conf_threshold),
            "primary_low_conf_tp_scale": float(self.fully_learned_primary_low_conf_tp_scale),
            "fallback_action": int(secondary.action),
            "fallback_side": int(secondary.side),
            "fallback_take_profit_before": float(secondary_raw.take_profit),
            "fallback_take_profit_after": float(secondary.take_profit),
            "runtime_config": dict(self.fully_learned_runtime_config or {}),
        }
        if self._fully_learned_cash_falls_through_to_alpha3():
            self.last_fully_learned_selection_trace["mode"] = "cash_alpha3_fallthrough"
            return None
        return primary

    def _lifecycle_v1_available(self) -> bool:
        return bool(
            self.lifecycle_v1_policy_bundle is not None
            and self.lifecycle_v1_exit_model is not None
            and self.lifecycle_v1_recalibrator
            and self.lifecycle_v1_cfg
        )

    def _lifecycle_v1_latest(self, frame: pd.DataFrame) -> tuple[pd.Series, np.ndarray] | None:
        if not self._lifecycle_v1_available():
            return None
        bundle = self.v21_2_parent_bundle if self._v21_2_jackpot_available() else self.lifecycle_v1_policy_bundle
        result = self._fully_learned_decision_frame(frame, bundle=bundle)
        if result is None:
            return None
        decisions, features = result
        if len(decisions) == 0 or len(features) == 0:
            return None
        if self._v21_2_jackpot_available():
            return decisions.iloc[-1], features.tail(1).to_numpy(dtype=np.float32, copy=False)
        entry_cfg = {
            "notional_mult": 1.5,
            "max_notional": 3.6,
            "quality_floor": 0.0,
            "confidence_floor": 0.0,
        }
        entry_cfg.update(dict(self.lifecycle_v1_entry_cfg or {}))
        decisions = _lifecycle_quality_scaled_decisions(decisions.copy(), **entry_cfg)
        return decisions.iloc[-1], features.tail(1).to_numpy(dtype=np.float32, copy=False)

    def _lifecycle_v1_feature_vec(
        self,
        frame: pd.DataFrame,
        *,
        side: int,
        age: int,
        unrealized: float,
        peak_unrealized: float,
        notional: float,
        leverage: float,
        entry_quality: float,
        entry_confidence: float,
    ) -> tuple[np.ndarray, pd.Series] | None:
        latest = self._lifecycle_v1_latest(frame)
        if latest is None:
            return None
        dec, base_values = latest
        side_values = np.asarray([int(dec.side)], dtype=np.int64)
        quality_values = np.asarray([float(dec.quality_score)], dtype=np.float64)
        confidence_values = np.asarray([float(dec.confidence)], dtype=np.float64)
        vec = _lifecycle_feature_vec_fast(
            base_values,
            side_values,
            quality_values,
            confidence_values,
            i=0,
            side=int(side),
            age=int(age),
            unrealized=float(unrealized),
            peak_unrealized=float(peak_unrealized),
            notional=float(notional),
            leverage=float(leverage),
            entry_quality=float(entry_quality),
            entry_confidence=float(entry_confidence),
        )
        return vec, dec

    def _lifecycle_v1_hazard_info(self, vec: np.ndarray) -> tuple[str, float, int]:
        recal = dict(self.lifecycle_v1_recalibrator or {})
        bucket = _lifecycle_bucket_from_vec(vec, dict(recal.get("thresholds", {})))
        info = dict(dict(recal.get("buckets", {})).get(bucket, {}))
        hazard = float(info.get("hazard_rate", recal.get("global_hazard_rate", 0.0)) or 0.0)
        support = int(info.get("support", 0) or 0)
        return bucket, hazard, support

    def _lifecycle_v1_entry_edit(self, frame: pd.DataFrame, dec: pd.Series) -> tuple[float, str, dict]:
        base_notional = float(dec.notional_exposure)
        leverage = float(dec.leverage or 1.0)
        side = int(dec.side)
        vec_dec = self._lifecycle_v1_feature_vec(
            frame,
            side=side,
            age=0,
            unrealized=0.0,
            peak_unrealized=0.0,
            notional=base_notional,
            leverage=leverage,
            entry_quality=float(dec.quality_score),
            entry_confidence=float(dec.confidence),
        )
        if vec_dec is None:
            return base_notional, "noop", {"entry_bucket": "", "entry_hazard": 0.0, "entry_support": 0}
        vec, _ = vec_dec
        bucket, hazard, support = self._lifecycle_v1_hazard_info(vec)
        global_rate = float(dict(self.lifecycle_v1_recalibrator or {}).get("global_hazard_rate", 0.0) or 0.0)
        cfg = dict(self.lifecycle_v1_cfg or {})
        mult = 1.0
        kind = "noop"
        if hazard >= global_rate + float(cfg.get("shrink_margin", 999.0)):
            mult = float(cfg.get("shrink_mult", 1.0))
            kind = "shrink"
        elif hazard <= global_rate - float(cfg.get("boost_margin", 999.0)):
            mult = float(cfg.get("boost_mult", 1.0))
            kind = "boost"
        cap = float(min(float(cfg.get("max_notional", 3.6)), float(self.lifecycle_v1_risk_cfg.get("max_notional", 3.6))))
        effective = float(np.clip(base_notional * mult, 0.0, cap))
        return effective, kind, {"entry_bucket": bucket, "entry_hazard": hazard, "entry_support": support}

    def _lifecycle_v1_daily_context(self, meta_router, frame: pd.DataFrame) -> dict:
        if "timestamp" in frame.columns:
            ts = pd.Timestamp(frame["timestamp"].iloc[-1])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC").tz_convert("Asia/Seoul")
            else:
                ts = ts.tz_convert("Asia/Seoul")
        else:
            ts = pd.Timestamp.now(tz="Asia/Seoul")
        day = ts.date()
        pnls: list[float] = []
        realized_equity = 1.0
        realized_peak = 1.0
        for row in list(getattr(meta_router, "trade_history", []) or []):
            try:
                rts = pd.Timestamp(row.get("ts", ""))
                if rts.tzinfo is None:
                    rts = rts.tz_localize("Asia/Seoul")
                else:
                    rts = rts.tz_convert("Asia/Seoul")
                pnl = float(row.get("pnl_frac", row.get("pnl", 0.0)) or 0.0)
                realized_equity *= max(0.0, 1.0 + pnl)
                realized_peak = max(realized_peak, realized_equity)
                if rts.date() != day:
                    continue
                pnls.append(pnl)
            except Exception:
                continue
        daily_realized = float(np.prod([1.0 + p for p in pnls]) - 1.0) if pnls else 0.0
        router_dd = float(max(0.0, 1.0 - float(meta_router.cur_equity or 1.0) / max(float(meta_router.peak_equity or 1.0), 1e-12)))
        realized_dd = float(max(0.0, 1.0 - realized_equity / max(realized_peak, 1e-12)))
        return {
            "day": day.isoformat(),
            "daily_trades": int(len(pnls)),
            "daily_realized": daily_realized,
            "daily_dd_proxy": float(max(0.0, -daily_realized)),
            "account_dd": float(max(router_dd, realized_dd)),
            "account_dd_router": router_dd,
            "account_dd_realized": realized_dd,
            "account_equity_realized": float(realized_equity),
            "account_peak_equity_realized": float(realized_peak),
            "loss_streak": int(getattr(meta_router, "loss_streak", 0) or 0),
        }

    def _lifecycle_v1_apply_conformal_veto_v1_5(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        dec: pd.Series,
        current_notional: float,
    ) -> tuple[float, str, dict]:
        cap = float(min(
            float(self.lifecycle_v1_cfg.get("max_notional", 3.6) or 3.6),
            float(self.lifecycle_v1_risk_cfg.get("max_notional", 3.6) or 3.6),
            float(getattr(meta_router, "exposure_cap", 5.0) or 5.0),
            3.6,
        ))
        n0 = float(np.clip(float(current_notional), 0.0, cap))
        meta = {
            "enabled": bool(self.conformal_veto_v1_5_enabled),
            "applied": False,
            "model_id": "clean_base_causal_sleeve_conformal_veto_v1_5",
            "model_version": "V1.5",
            "model": str(self.conformal_veto_v1_5_model_path),
            "report": str(self.conformal_veto_v1_5_report_path),
            "input_notional": float(n0),
            "output_notional": float(n0),
            "cap": float(cap),
        }
        adapter = self.conformal_veto_v1_5_adapter
        if not bool(self.conformal_veto_v1_5_enabled) or adapter is None or n0 <= 1e-12:
            return float(n0), "noop", meta
        ctx = self._lifecycle_v1_daily_context(meta_router, frame)
        try:
            decision = adapter.decide(frame, dec, ctx, n0, max_total_notional=cap)
        except Exception as e:
            meta.update({"blocked": True, "reason": "conformal_veto_v1_5_signal_error", "error": str(e)})
            logger.warning("SYSTEM conformal_veto_v1_5 signal failed: %s", e)
            return float(n0), "conformal_veto_v1_5_signal_error", meta
        core_raw = float(decision.core_notional)
        sleeve_raw = float(decision.sleeve_notional)
        core_notional = float(np.clip(core_raw if np.isfinite(core_raw) else n0, 0.0, cap))
        sleeve_notional_in = float(max(0.0, sleeve_raw if np.isfinite(sleeve_raw) else 0.0))
        sleeve_notional = float(min(sleeve_notional_in, max(0.0, cap - core_notional)))
        fee = float(getattr(meta_router, "trade_fee", 0.0) or 0.0)
        slip = float(getattr(meta_router, "trade_slip", 0.0) or 0.0)
        cost3_mode = bool(
            fee >= float(FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_FEE)
            or slip >= float(FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_SLIP)
        )
        stress_state = bool(self._lifecycle_v1_stress_state(frame))
        firewall_reason = ""
        firewall_action = str(decision.action)
        firewall_applied = False
        if bool(FINAL_GOVERNOR_V1_5_COST_FIREWALL_ENABLE) and sleeve_notional > 1e-12:
            if cost3_mode and bool(FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_DISABLE):
                sleeve_notional = 0.0
                firewall_reason = "cost3_disable_sleeve"
                firewall_action = "COST_FIREWALL_DISABLE_SLEEVE"
                firewall_applied = True
            elif stress_state:
                stress_mult = float(np.clip(float(FINAL_GOVERNOR_V1_5_COST_FIREWALL_STRESS_SLEEVE_MULT), 0.0, 1.0))
                capped_sleeve = float(sleeve_notional * stress_mult)
                if capped_sleeve < sleeve_notional - 1e-12:
                    sleeve_notional = capped_sleeve
                    firewall_reason = "stress_cap_sleeve"
                    firewall_action = "COST_FIREWALL_STRESS_CAP_SLEEVE"
                    firewall_applied = True
        out = float(np.clip(core_notional + sleeve_notional, 0.0, cap))
        active_sleeve = bool(sleeve_notional > 1e-12)
        edit = "conformal_veto_v1_5_add_same_side" if active_sleeve else "conformal_veto_v1_5_core"
        if decision.action == "CONFORMAL_VETO":
            edit = "conformal_veto_v1_5_veto"
        if firewall_applied:
            edit = "conformal_veto_v1_5_cost_firewall"
        meta.update(
            {
                "applied": True,
                "action": str(firewall_action),
                "raw_action": str(decision.action),
                "reason": str(firewall_reason or decision.reason),
                "raw_reason": str(decision.reason),
                "core_notional": float(core_notional),
                "sleeve_notional": float(sleeve_notional),
                "raw_sleeve_notional": float(sleeve_notional_in),
                "sleeve_fraction": float(sleeve_notional / max(core_notional, 1e-12)),
                "raw_sleeve_fraction": float(decision.sleeve_fraction),
                "sleeve_exit_bars": int(decision.sleeve_exit_bars),
                "output_notional": float(out),
                "cost_firewall": {
                    "enabled": bool(FINAL_GOVERNOR_V1_5_COST_FIREWALL_ENABLE),
                    "applied": bool(firewall_applied),
                    "reason": str(firewall_reason),
                    "cost3_mode": bool(cost3_mode),
                    "stress_state": bool(stress_state),
                    "fee": float(fee),
                    "slip": float(slip),
                    "cost3_fee_threshold": float(FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_FEE),
                    "cost3_slip_threshold": float(FINAL_GOVERNOR_V1_5_COST_FIREWALL_COST3_SLIP),
                    "stress_sleeve_mult": float(FINAL_GOVERNOR_V1_5_COST_FIREWALL_STRESS_SLEEVE_MULT),
                    "input_sleeve_notional": float(sleeve_notional_in),
                    "output_sleeve_notional": float(sleeve_notional),
                },
                "features": dict(decision.features),
                "predictions": dict(decision.predictions),
                "selected_config": dict(adapter.selected_config),
            }
        )
        return float(out), edit, meta

    def _lifecycle_v1_apply_risk_gates(
        self,
        meta_router,
        frame: pd.DataFrame,
        notional: float,
        *,
        max_notional_override: float | None = None,
    ) -> tuple[float, bool, list[str], dict]:
        risk = dict(self.lifecycle_v1_risk_cfg or {})
        ctx = self._lifecycle_v1_daily_context(meta_router, frame)
        reasons: list[str] = []
        block = False
        max_daily_trades = int(risk.get("max_daily_trades", 999999) or 999999)
        daily_loss_limit = abs(float(risk.get("daily_loss_limit", 0.0) or 0.0))
        daily_dd_limit = abs(float(risk.get("daily_dd_limit", 0.0) or 0.0))
        configured_daily_loss_limit = float(daily_loss_limit)
        configured_daily_dd_limit = float(daily_dd_limit)
        if bool(FINAL_GOVERNOR_LIFECYCLE_V1_DISABLE_DAILY_LOSS_DD):
            daily_loss_limit = 0.0
            daily_dd_limit = 0.0
        if ctx["daily_trades"] >= max_daily_trades:
            block = True
            reasons.append("max_daily_trades")
        if daily_loss_limit > 0.0 and float(ctx["daily_realized"]) <= -daily_loss_limit:
            block = True
            reasons.append("daily_loss_limit")
        if daily_dd_limit > 0.0 and float(ctx["daily_dd_proxy"]) >= daily_dd_limit:
            block = True
            reasons.append("daily_dd_limit")

        n = float(notional)
        if not block and float(ctx["account_dd"]) >= float(risk.get("global_dd_cut", 999.0) or 999.0):
            n *= float(risk.get("global_dd_mult", 1.0) or 1.0)
            reasons.append("global_dd_shrink")
        loss_soft = int(risk.get("loss_streak_soft", 999999) or 999999)
        if not block and int(ctx["loss_streak"]) >= loss_soft:
            steps = int(ctx["loss_streak"]) - loss_soft + 1
            n *= float(risk.get("loss_streak_mult", 1.0) or 1.0) ** float(max(0, steps))
            reasons.append("loss_streak_shrink")
        if not block and float(ctx["daily_realized"]) >= float(risk.get("daily_profit_boost_start", 999.0) or 999.0):
            n *= float(risk.get("daily_profit_boost_mult", 1.0) or 1.0)
            reasons.append("daily_profit_boost")
        router_cap = float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0))
        if max_notional_override is not None and float(max_notional_override) > 0.0:
            cap = float(min(float(max_notional_override), router_cap))
            reasons.append("safe_learned_cap_risk_cap")
        else:
            cap = float(min(float(self.lifecycle_v1_cfg.get("max_notional", 3.6)), float(risk.get("max_notional", 3.6) or 3.6), router_cap))
        n = float(np.clip(n, 0.0, cap))
        if n <= 1e-12:
            block = True
            reasons.append("zero_effective_notional")
        ctx.update({
            "risk_adjusted_notional": float(n),
            "risk_cap": float(cap),
            "daily_loss_dd_gate_disabled": bool(FINAL_GOVERNOR_LIFECYCLE_V1_DISABLE_DAILY_LOSS_DD),
            "configured_daily_loss_limit": float(configured_daily_loss_limit),
            "configured_daily_dd_limit": float(configured_daily_dd_limit),
            "active_daily_loss_limit": float(daily_loss_limit),
            "active_daily_dd_limit": float(daily_dd_limit),
        })
        return n, block, reasons, ctx

    @staticmethod
    def _lifecycle_v1_row_float(row, col: str, default: float = 0.0) -> float:
        try:
            val = float(row.get(col, default) or default)
            return val if np.isfinite(val) else float(default)
        except Exception:
            return float(default)

    def _lifecycle_v1_stress_state(self, frame: pd.DataFrame) -> bool:
        if frame is None or len(frame) == 0:
            return False
        row = frame.iloc[-1]
        return bool(
            self._lifecycle_v1_row_float(row, "evt_tail_flag") > 0.0
            or abs(self._lifecycle_v1_row_float(row, "liquidity_vacuum")) > 1.0
            or abs(self._lifecycle_v1_row_float(row, "funding_pressure")) > 0.12
            or abs(self._lifecycle_v1_row_float(row, "ai_adverse_risk")) > 0.75
        )

    def _deep_gated_gross_live_signal(self, frame: pd.DataFrame, dec: pd.Series, current_notional: float, account_ctx: dict) -> dict | None:
        payload = self.deep_gated_gross_payload
        deep_model = self.deep_gated_gross_deep_model
        if payload is None or deep_model is None or frame is None or len(frame) == 0:
            return None
        seq_features = list(payload.get("sequence_features") or [])
        if not seq_features:
            return None
        work = frame.copy()
        for col in seq_features:
            if col not in work.columns:
                work[col] = 0.0
        try:
            scaled = _deep_v1._transform_sequence_matrix(work, seq_features, payload["sequence_scaler"])
            i = int(len(work) - 1)
            side = int(getattr(dec, "side", 0) or 0)
            close = float(pd.to_numeric(work["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().iloc[-1])
            ctx = {
                "trade_id": 0,
                "entry_idx": i,
                "core_exit_idx": i,
                "side": side,
                "entry_price": close,
                "core_notional": float(current_notional),
                "base_notional": float(getattr(dec, "notional_exposure", current_notional) or current_notional),
                "leverage": float(getattr(dec, "leverage", 1.0) or 1.0),
                "quality": float(getattr(dec, "quality_score", 0.0) or 0.0),
                "confidence": float(getattr(dec, "confidence", 0.0) or 0.0),
                "timestamp": str(work["timestamp"].iloc[i]) if "timestamp" in work.columns else str(i),
            }
            seq = _deep_v1._sequence_tensor(scaled, [ctx], lookback=_deep_v2.LOOKBACK)
            deep_meta = dict(payload.get("deep_meta", {}) or {})
            deep = _deep_v2._deep_predict_v2(
                deep_model,
                seq,
                list(deep_meta.get("target_mean") or [0.0, 0.0, 0.0]),
                list(deep_meta.get("target_std") or [1.0, 1.0, 1.0]),
            )
            _deep_v1.LOOKBACK = _deep_v2.LOOKBACK
            _deep_v1.EMBED_DIM = _deep_v2.ENSEMBLE_EMBED_DIM
            _deep_v1.N_CLUSTERS = _deep_v2.N_CLUSTERS
            state_df = _deep_v1._state_features(payload["state_model"], deep)
            row = _deep_cg._row_features(
                work,
                ctx,
                state_df,
                0,
                float(account_ctx.get("account_dd", 0.0) or 0.0),
                float(account_ctx.get("daily_dd_proxy", 0.0) or 0.0),
                int(account_ctx.get("loss_streak", 0) or 0),
            )
            same_pred, adverse_pred = _deep_v1._predict_heads(payload["head_model"], row)
            signal = _deep_dgg._row_signal(row, same_pred, adverse_pred)
            return {
                "same_pred": float(same_pred),
                "adverse_pred": float(adverse_pred),
                "conviction": float(signal.get("conviction", 0.0) or 0.0),
                "adverse": float(signal.get("adverse", 0.0) or 0.0),
                "deep_same": float(signal.get("deep_same", 0.0) or 0.0),
                "deep_full": float(signal.get("deep_full", 0.0) or 0.0),
                "deep_adverse": float(signal.get("deep_adverse", 0.0) or 0.0),
                "state_cluster_distance": float(row.get("state_cluster_distance", 0.0) or 0.0),
            }
        except Exception as e:
            logger.warning("SYSTEM deep_gated_gross signal failed: %s", e)
            return None

    def _lifecycle_v1_apply_deep_gated_gross(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        dec: pd.Series,
        current_notional: float,
    ) -> tuple[float, str, dict]:
        cfg = dict(self.deep_gated_gross_cfg or self.deep_gated_gross_report.get("selected_config", {}) or {})
        report_gate = dict(self.deep_gated_gross_report.get("promotion_gate", {}) or {})
        meta = {
            "enabled": bool(self.deep_gated_gross_enabled),
            "applied": False,
            "model_id": str(self.deep_gated_gross_report.get("model_id", "clean_base_deep_gated_gross_v2")),
            "model": str(self.deep_gated_gross_model_path),
            "report": str(self.deep_gated_gross_report_path),
            "selected_config": str(cfg.get("name", "")),
            "target_500_pnl": bool(report_gate.get("target_500_pnl", False)),
            "input_notional": float(current_notional),
        }
        if not bool(self.deep_gated_gross_enabled):
            return float(current_notional), "noop", meta

        cap = float(min(
            float(self.lifecycle_v1_cfg.get("max_notional", 3.6) or 3.6),
            float(self.lifecycle_v1_risk_cfg.get("max_notional", 3.6) or 3.6),
            float(getattr(meta_router, "exposure_cap", 5.0) or 5.0),
        ))
        account_ctx = self._lifecycle_v1_daily_context(meta_router, frame)
        signal = self._deep_gated_gross_live_signal(frame, dec, current_notional, account_ctx)
        if signal is None:
            meta.update({"error": "signal_unavailable", "cap": float(cap)})
            return float(current_notional), "noop", meta

        high_notional = float(cfg.get("high_notional", 3.6) or 3.6)
        mid_notional = float(cfg.get("mid_notional", 3.0) or 3.0)
        defensive_notional = float(cfg.get("defensive_notional", 3.0) or 3.0)
        high_threshold = float(cfg.get("high_threshold", -0.006) or -0.006)
        mid_threshold = float(cfg.get("mid_threshold", -0.012) or -0.012)
        adverse_cut = float(cfg.get("adverse_cut", 99.0) or 99.0)
        deep_full_floor = float(cfg.get("deep_full_floor", -0.010) or -0.010)
        cost3_notional = float(cfg.get("cost3_notional", 0.0) or 0.0)
        cost3_mode = bool(
            float(getattr(meta_router, "trade_fee", 0.0) or 0.0) >= float(FINAL_GOVERNOR_DEEP_GATED_GROSS_COST3_FEE)
            or float(getattr(meta_router, "trade_slip", 0.0) or 0.0) >= float(FINAL_GOVERNOR_DEEP_GATED_GROSS_COST3_SLIP)
        )
        stress_state = bool(
            self._lifecycle_v1_stress_state(frame)
            or _deep_v1.base._stress(frame, int(len(frame) - 1))
            or float(account_ctx.get("account_dd", 0.0) or 0.0) >= 0.30
        )
        reasons: list[str] = []
        if stress_state:
            reasons.append("stress_state")
        if float(signal["adverse"]) >= adverse_cut:
            reasons.append("deep_or_head_adverse_cut")
        if float(signal["deep_full"]) < deep_full_floor:
            reasons.append("deep_full_floor")

        if cost3_mode:
            out = float(cost3_notional)
            bucket = "COST3_CAPITAL_PRESERVE" if out <= 1e-12 else "COST3_LOW_NOTIONAL"
            reasons.append("cost3_capital_preserve")
        elif stress_state or float(signal["adverse"]) >= adverse_cut or float(signal["deep_full"]) < deep_full_floor:
            out = float(defensive_notional)
            bucket = "DEFENSIVE"
        elif float(signal["conviction"]) >= high_threshold:
            out = float(high_notional)
            bucket = "HIGH"
            reasons.append("deep_high_conviction")
        elif float(signal["conviction"]) >= mid_threshold:
            out = float(mid_notional)
            bucket = "MID"
            reasons.append("deep_mid_conviction")
        else:
            out = float(defensive_notional)
            bucket = "DEFENSIVE"
            reasons.append("deep_low_conviction")

        out = float(np.clip(out, 0.0, cap))
        edit = "deep_gated_gross_cost3_preserve" if cost3_mode and out <= 1e-12 else f"deep_gated_gross_{bucket.lower()}"
        meta.update(
            {
                "applied": True,
                "cap": float(cap),
                "output_notional": float(out),
                "edit": str(edit),
                "bucket": str(bucket),
                "reasons": list(reasons),
                "cost3_mode": bool(cost3_mode),
                "stress_state": bool(stress_state),
                "fee_rate": float(getattr(meta_router, "trade_fee", 0.0) or 0.0),
                "slippage_rate": float(getattr(meta_router, "trade_slip", 0.0) or 0.0),
                "high_notional": float(high_notional),
                "mid_notional": float(mid_notional),
                "defensive_notional": float(defensive_notional),
                "cost3_notional": float(cost3_notional),
                "high_threshold": float(high_threshold),
                "mid_threshold": float(mid_threshold),
                "adverse_cut": float(adverse_cut),
                "deep_full_floor": float(deep_full_floor),
                "account_context": dict(account_ctx),
                "signal": dict(signal),
            }
        )
        return float(out), str(edit), meta

    @staticmethod
    def _safe_cap_bin(value: float, cuts: list[float]) -> int:
        try:
            arr = np.asarray(list(cuts or []), dtype=np.float64)
            return int(np.searchsorted(arr, float(value), side="right"))
        except Exception:
            return 0

    @staticmethod
    def _safe_cap_vol_value(row) -> float:
        for col in ("garch_vol_z", "volatility_z", "regime_vol_z"):
            try:
                val = row.get(col, None)
                if val is not None:
                    x = float(val)
                    if np.isfinite(x):
                        return float(x)
            except Exception:
                continue
        return 0.0

    def _safe_cap_edge_proxy(self, frame: pd.DataFrame, deep_gated_gross_meta: dict | None) -> float:
        vals: list[float] = []
        signal = dict((deep_gated_gross_meta or {}).get("signal", {}) or {})
        for key in ("same_pred", "deep_same", "deep_full", "conviction"):
            vals.append(_safe_float(signal.get(key, 0.0), 0.0))
        finite = [float(x) for x in vals if np.isfinite(float(x))]
        return float(max([0.0] + finite))

    def _safe_learned_cap_bucket(
        self,
        frame: pd.DataFrame,
        *,
        scheme: str,
        action_bucket: str,
        side: int,
        edge: float,
        thresholds: dict,
    ) -> str:
        row = frame.iloc[-1] if frame is not None and len(frame) else {}
        side_key = "L" if int(side) > 0 else "S"
        action_key = str(action_bucket or "UNKNOWN").upper()
        edge3 = f"e{self._safe_cap_bin(edge, list(thresholds.get('edge3', [0.0, 0.0]) or [0.0, 0.0]))}"
        edge2 = f"e{self._safe_cap_bin(edge, list(thresholds.get('edge2', [0.0]) or [0.0]))}"
        vol2 = f"v{self._safe_cap_bin(self._safe_cap_vol_value(row), list(thresholds.get('vol2', [0.0]) or [0.0]))}"
        if scheme == "action_side_edge3":
            return "|".join((action_key, side_key, edge3))
        if scheme == "side_edge3_vol2":
            return "|".join((side_key, edge3, vol2))
        if scheme == "action_edge3":
            return "|".join((action_key, edge3))
        return "|".join((action_key, edge3))

    def _lifecycle_v1_apply_safe_learned_cap(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        dec: pd.Series,
        current_notional: float,
        deep_gated_gross_meta: dict | None,
    ) -> tuple[float, str, dict]:
        cand = dict(self.safe_learned_cap_candidate or {})
        cap_map = dict(cand.get("cap_map", {}) or {})
        meta = {
            "enabled": bool(self.safe_learned_cap_enabled),
            "applied": False,
            "blocked": False,
            "model_id": "clean_base_deep_gated_gross_v2_safe_cap_buckets",
            "candidate": str(cand.get("name", "")),
            "input_notional": float(current_notional),
        }
        if (
            not bool(self.safe_learned_cap_enabled)
            or not cand
            or not cap_map
            or current_notional <= 1e-12
            or not bool((deep_gated_gross_meta or {}).get("applied", False))
        ):
            return float(current_notional), "noop", meta

        router_cap = float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0))
        candidate_max = float(cand.get("max_notional", router_cap) or router_cap)
        fallback_cap = float(cand.get("fallback_cap", min(3.6, candidate_max)) or min(3.6, candidate_max))
        scheme = str(cand.get("scheme", "action_edge3") or "action_edge3")
        thresholds = dict(cand.get("thresholds", {}) or {})
        action_bucket = str((deep_gated_gross_meta or {}).get("bucket", "") or "UNKNOWN").upper()
        side = int(np.sign(int(getattr(dec, "side", 0) or 0)))
        edge = self._safe_cap_edge_proxy(frame, deep_gated_gross_meta)
        bucket = self._safe_learned_cap_bucket(
            frame,
            scheme=scheme,
            action_bucket=action_bucket,
            side=side,
            edge=edge,
            thresholds=thresholds,
        )
        learned_cap = float(cap_map.get(bucket, fallback_cap))
        learned_cap = float(np.clip(learned_cap, 0.0, min(candidate_max, router_cap)))
        planned = float(np.clip(float(current_notional) * learned_cap / 3.6, 0.0, learned_cap))
        gate_mode = str(cand.get("gate_notional_mode", "final") or "final").lower()
        gate_notional = float(current_notional) if gate_mode == "base" else float(planned)
        fee = float(getattr(meta_router, "trade_fee", 0.0) or 0.0)
        slip = float(getattr(meta_router, "trade_slip", 0.0) or 0.0)
        cost_buffer = float(cand.get("cost_buffer", 0.0035) or 0.0035)
        expected_edge = float(edge * max(gate_notional, 0.0))
        cost_hurdle = float(2.0 * (fee + slip) * max(gate_notional, 0.0) + cost_buffer)
        meta.update(
            {
                "applied": True,
                "scheme": scheme,
                "bucket": bucket,
                "action_bucket": action_bucket,
                "edge": float(edge),
                "learned_cap": float(learned_cap),
                "fallback_cap": float(fallback_cap),
                "candidate_max_notional": float(candidate_max),
                "output_notional": float(planned),
                "gate_mode": gate_mode,
                "gate_notional": float(gate_notional),
                "expected_equity_edge": float(expected_edge),
                "cost_hurdle": float(cost_hurdle),
                "cost_buffer": float(cost_buffer),
                "fee_rate": float(fee),
                "slippage_rate": float(slip),
            }
        )
        if expected_edge <= cost_hurdle:
            meta.update({"blocked": True, "reason": "safe_learned_cap_cost_gate_block", "output_notional": 0.0})
            return 0.0, "safe_learned_cap_cost_gate_block", meta
        edit = "safe_learned_cap_boost" if planned > current_notional + 1e-12 else "safe_learned_cap_base"
        meta["reason"] = edit
        return float(planned), edit, meta

    def _lifecycle_v21_available(self) -> bool:
        base_ready = bool(
            self.v21_enabled
            and self.v21_payload is not None
            and self.v21_scout_config
            and self.v21_stop_config
            and self.v21_path_model is not None
        )
        if not base_ready:
            return False
        if self.v21_adapter_version == "v22_1_learned_scout":
            return bool(
                isinstance(self.v21_scout_heads, dict)
                and self.v21_scout_heads.get("gate_model") is not None
                and self.v21_scout_heads.get("frac_model") is not None
                and self.v21_feature_cols
            )
        return True

    def _lifecycle_v21_pure_active(self) -> bool:
        return bool((self._v21_2_jackpot_available() or self._lifecycle_v22_1_available() or self._lifecycle_v21_available()) and self.v21_pure_mode)

    def _lifecycle_v22_1_available(self) -> bool:
        return bool(self.v22_1_enabled and self.v22_1_adapter is not None)

    def _v21_2_jackpot_available(self) -> bool:
        return bool(
            self.v21_2_jackpot_enabled
            and self.v21_2_jackpot_adapter is not None
            and self.v21_2_parent_bundle is not None
        )

    def _lifecycle_v22_1_cap_plan(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        dec: pd.Series,
        base_notional: float,
        deep_gated_gross_meta: dict | None,
    ) -> dict:
        adapter = self.v22_1_adapter
        cand = dict(adapter.cap_candidate if adapter is not None else {})
        cap_map = dict(cand.get("cap_map", {}) or {})
        router_cap = float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0))
        meta = {
            "candidate": str(cand.get("name", "")),
            "input_notional": float(base_notional),
            "planned_notional": 0.0,
            "cost_pass": False,
        }
        if (
            adapter is None
            or not cand
            or not cap_map
            or float(base_notional) <= 1e-12
            or not bool((deep_gated_gross_meta or {}).get("applied", False))
        ):
            meta["reason"] = "cap_plan_unavailable"
            return meta
        candidate_max = float(cand.get("max_notional", router_cap) or router_cap)
        fallback_cap = float(cand.get("fallback_cap", min(3.6, candidate_max)) or min(3.6, candidate_max))
        scheme = str(cand.get("scheme", "action_edge3") or "action_edge3")
        thresholds = dict(cand.get("thresholds", {}) or {})
        action_bucket = str((deep_gated_gross_meta or {}).get("bucket", "") or "UNKNOWN").upper()
        side = int(np.sign(int(getattr(dec, "side", 0) or 0)))
        edge = self._safe_cap_edge_proxy(frame, deep_gated_gross_meta)
        bucket = self._safe_learned_cap_bucket(
            frame,
            scheme=scheme,
            action_bucket=action_bucket,
            side=side,
            edge=edge,
            thresholds=thresholds,
        )
        learned_cap = float(cap_map.get(bucket, fallback_cap))
        learned_cap = float(np.clip(learned_cap, 0.0, min(candidate_max, router_cap)))
        planned = float(np.clip(float(base_notional) * learned_cap / 3.6, 0.0, learned_cap))
        gate_mode = str(cand.get("gate_notional_mode", "final") or "final").lower()
        gate_notional = float(base_notional) if gate_mode == "base" else float(planned)
        fee = float(getattr(meta_router, "trade_fee", 0.0) or 0.0)
        slip = float(getattr(meta_router, "trade_slip", 0.0) or 0.0)
        cost_buffer = float(cand.get("cost_buffer", 0.0035) or 0.0035)
        expected_edge = float(edge * max(gate_notional, 0.0))
        cost_hurdle = float(2.0 * (fee + slip) * max(gate_notional, 0.0) + cost_buffer)
        meta.update(
            {
                "reason": "cap_plan",
                "scheme": scheme,
                "bucket": bucket,
                "action_bucket": action_bucket,
                "edge": float(edge),
                "learned_cap": float(learned_cap),
                "fallback_cap": float(fallback_cap),
                "candidate_max_notional": float(candidate_max),
                "planned_notional": float(planned),
                "gate_mode": gate_mode,
                "gate_notional": float(gate_notional),
                "expected_equity_edge": float(expected_edge),
                "cost_hurdle": float(cost_hurdle),
                "cost_buffer": float(cost_buffer),
                "fee_rate": float(fee),
                "slippage_rate": float(slip),
                "cost_pass": bool(expected_edge > cost_hurdle),
            }
        )
        return meta

    def _lifecycle_v21_cap_plan(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        dec: pd.Series,
        base_notional: float,
        deep_gated_gross_meta: dict | None,
    ) -> dict:
        cand = dict(self.v21_cap_candidate or self.safe_learned_cap_candidate or {})
        cap_map = dict(cand.get("cap_map", {}) or {})
        router_cap = float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0))
        meta = {
            "candidate": str(cand.get("name", "")),
            "input_notional": float(base_notional),
            "planned_notional": 0.0,
            "cost_pass": False,
        }
        if (
            not cand
            or not cap_map
            or float(base_notional) <= 1e-12
            or not bool((deep_gated_gross_meta or {}).get("applied", False))
        ):
            meta["reason"] = "cap_plan_unavailable"
            return meta
        candidate_max = float(cand.get("max_notional", router_cap) or router_cap)
        fallback_cap = float(cand.get("fallback_cap", min(3.6, candidate_max)) or min(3.6, candidate_max))
        scheme = str(cand.get("scheme", "action_edge3") or "action_edge3")
        thresholds = dict(cand.get("thresholds", {}) or {})
        action_bucket = str((deep_gated_gross_meta or {}).get("bucket", "") or "UNKNOWN").upper()
        side = int(np.sign(int(getattr(dec, "side", 0) or 0)))
        edge = self._safe_cap_edge_proxy(frame, deep_gated_gross_meta)
        bucket = self._safe_learned_cap_bucket(
            frame,
            scheme=scheme,
            action_bucket=action_bucket,
            side=side,
            edge=edge,
            thresholds=thresholds,
        )
        learned_cap = float(cap_map.get(bucket, fallback_cap))
        learned_cap = float(np.clip(learned_cap, 0.0, min(candidate_max, router_cap)))
        planned = float(np.clip(float(base_notional) * learned_cap / 3.6, 0.0, learned_cap))
        gate_mode = str(cand.get("gate_notional_mode", "final") or "final").lower()
        gate_notional = float(base_notional) if gate_mode == "base" else float(planned)
        fee = float(getattr(meta_router, "trade_fee", 0.0) or 0.0)
        slip = float(getattr(meta_router, "trade_slip", 0.0) or 0.0)
        cost_buffer = float(cand.get("cost_buffer", 0.0035) or 0.0035)
        expected_edge = float(edge * max(gate_notional, 0.0))
        cost_hurdle = float(2.0 * (fee + slip) * max(gate_notional, 0.0) + cost_buffer)
        meta.update(
            {
                "reason": "cap_plan",
                "scheme": scheme,
                "bucket": bucket,
                "action_bucket": action_bucket,
                "edge": float(edge),
                "learned_cap": float(learned_cap),
                "fallback_cap": float(fallback_cap),
                "candidate_max_notional": float(candidate_max),
                "planned_notional": float(planned),
                "gate_mode": gate_mode,
                "gate_notional": float(gate_notional),
                "expected_equity_edge": float(expected_edge),
                "cost_hurdle": float(cost_hurdle),
                "cost_buffer": float(cost_buffer),
                "fee_rate": float(fee),
                "slippage_rate": float(slip),
                "cost_pass": bool(expected_edge > cost_hurdle),
            }
        )
        return meta

    def _lifecycle_v21_ledger_row(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        dec: pd.Series,
        pre_adaptive_notional: float,
        v17_notional: float,
        deep_gated_gross_meta: dict | None,
        adaptive_calibrator_meta: dict | None,
    ) -> pd.DataFrame:
        ledger = self._adaptive_calibrator_ledger_row(
            meta_router,
            frame,
            dec=dec,
            current_notional=float(pre_adaptive_notional),
            deep_gated_gross_meta=deep_gated_gross_meta,
        )
        scale = _safe_float((adaptive_calibrator_meta or {}).get("router_scale", 0.0), 0.0)
        if scale <= 0.0 and float(pre_adaptive_notional) > 1e-12:
            scale = float(np.clip(float(v17_notional) / max(float(pre_adaptive_notional), 1e-12), 0.0, 1.0))
        ledger["router_scale"] = float(scale)
        ledger["router_effective_core_notional_before"] = float(pre_adaptive_notional)
        # Current V22.1 artifact included these columns in its feature order, but
        # its replay/training frame did not populate them, so the learner saw 0.0.
        ledger["gross_notional"] = 0.0
        ledger["net_notional"] = 0.0
        ledger["effective_core_notional"] = float(v17_notional)
        ledger["adaptive_q"] = _safe_float((adaptive_calibrator_meta or {}).get("adaptive_q", 0.0), 0.0)
        ledger["adaptive_lower"] = _safe_float((adaptive_calibrator_meta or {}).get("adaptive_lower", 0.0), 0.0)
        ledger["adaptive_pred_pnl"] = _safe_float((adaptive_calibrator_meta or {}).get("adaptive_pred_pnl", 0.0), 0.0)
        ledger["adaptive_mode"] = str((adaptive_calibrator_meta or {}).get("adaptive_mode", ""))
        return ledger

    def _lifecycle_v21_attach_path_predictions(self, frame: pd.DataFrame, ledger: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        meta = {
            "enabled": bool(self._lifecycle_v21_available()),
            "applied": False,
            "model": str(self.v21_model_path),
        }
        path_model = self.v21_path_model
        if not isinstance(path_model, dict) or ledger is None or len(ledger) == 0:
            meta["reason"] = "path_model_unavailable"
            return ledger, meta
        try:
            out = ledger.copy()
            x = _deep_state_v15._feature_matrix(frame, out)
            xz = path_model["scaler"].transform(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
            out["pred_mae_raw"] = np.maximum(np.asarray(path_model["mae_model"].predict(xz), dtype=np.float64), 0.0)
            out["pred_tail_raw"] = np.maximum(np.asarray(path_model["tail_model"].predict(xz), dtype=np.float64), 0.0)
            out["pred_mfe_raw"] = np.maximum(np.asarray(path_model["mfe_model"].predict(xz), dtype=np.float64), 0.0)
            meta.update(
                {
                    "applied": True,
                    "pred_mae_raw": float(out["pred_mae_raw"].iloc[0]),
                    "pred_tail_raw": float(out["pred_tail_raw"].iloc[0]),
                    "pred_mfe_raw": float(out["pred_mfe_raw"].iloc[0]),
                    "train_meta": dict(path_model.get("train_meta", {}) or {}),
                }
            )
            return out, meta
        except Exception as e:
            meta.update({"error": str(e), "reason": "path_prediction_failed"})
            logger.warning("SYSTEM v21 path prediction failed: %s", e)
            return ledger, meta

    @staticmethod
    def _lifecycle_v22_regime_code(value: object) -> float:
        text = str(value or "").strip().upper()
        if text == "BULL":
            return 1.0
        if text == "BEAR":
            return -1.0
        if text == "CHOP":
            return 0.35
        if text == "WHIPSAW":
            return -0.35
        return 0.0

    def _lifecycle_v22_feature_matrix(self, ledger: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        cols = list(self.v21_feature_cols or [])
        missing: list[str] = []
        out = pd.DataFrame(index=ledger.index)
        for col in cols:
            if col == "regime_code":
                out[col] = (
                    ledger["regime"].map(self._lifecycle_v22_regime_code)
                    if "regime" in ledger.columns
                    else 0.0
                )
            elif col in ledger.columns:
                out[col] = pd.to_numeric(ledger[col], errors="coerce")
            else:
                out[col] = 0.0
                missing.append(str(col))
        return out.replace([np.inf, -np.inf], np.nan), missing

    @staticmethod
    def _lifecycle_v22_positive_probability(model, x: pd.DataFrame, positive_class: int = 1) -> np.ndarray:
        if isinstance(model, dict) and model.get("constant") is not None:
            return np.full(len(x), float(model["constant"]), dtype=float)
        proba = model.predict_proba(x)
        classes = list(getattr(model, "classes_", []))
        if positive_class not in classes:
            return np.zeros(len(x), dtype=float)
        return np.asarray(proba[:, classes.index(positive_class)], dtype=float)

    @staticmethod
    def _lifecycle_v22_frac_prediction(model, x: pd.DataFrame, gate_prob: np.ndarray) -> np.ndarray:
        if isinstance(model, dict) and model.get("constant") is not None:
            pred = np.full(len(x), float(model["constant"]), dtype=float)
        else:
            pred = np.asarray(model.predict(x), dtype=float)
            pred = np.where(pred > 1.0, pred / 100.0, pred)
        capped = np.where(
            gate_prob >= 0.85,
            np.minimum(pred, 0.75),
            np.where(gate_prob >= 0.70, np.minimum(pred, 0.50), np.minimum(pred, 0.25)),
        )
        return np.clip(capped, 0.25, 0.75)

    def _lifecycle_v22_scout_signal(self, ledger: pd.DataFrame, row: pd.Series) -> dict:
        cfg = dict(self.v21_scout_config or {})
        heads = dict(self.v21_scout_heads or {})
        x, missing = self._lifecycle_v22_feature_matrix(ledger)
        prob = self._lifecycle_v22_positive_probability(heads["gate_model"], x)
        frac = self._lifecycle_v22_frac_prediction(heads["frac_model"], x, prob)
        side = int(np.sign(int(row.get("core_side", 0))))
        parent_n = _safe_float(row.get("router_effective_core_notional_before", 0.0), 0.0)
        lower = _safe_float(row.get("adaptive_lower", -999.0), -999.0)
        pred = _safe_float(row.get("adaptive_pred_pnl", -999.0), -999.0)
        lower_floor = float(cfg.get("lower_safety_floor", -0.025) or -0.025)
        lower_ceiling = float(cfg.get("lower_nearmiss_ceiling", 0.0) or 0.0)
        pred_floor = float(cfg.get("pred_safety_floor", -0.025) or -0.025)
        threshold = float(cfg.get("probability_threshold", 0.80) or 0.80)
        broad_candidate = bool(
            side != 0
            and parent_n > 1e-12
            and lower > lower_floor
            and lower <= lower_ceiling
            and pred >= pred_floor
        )
        p = float(prob[0]) if len(prob) else 0.0
        f = float(frac[0]) if len(frac) else 0.25
        return {
            "adapter": "v22_1_learned_scout",
            "broad_candidate": bool(broad_candidate),
            "eligible": bool(broad_candidate and p >= threshold),
            "probability": float(p),
            "probability_threshold": float(threshold),
            "scout_frac": float(f),
            "feature_cols": list(self.v21_feature_cols or []),
            "missing_feature_cols": list(missing),
            "lower_safety_floor": float(lower_floor),
            "lower_nearmiss_ceiling": float(lower_ceiling),
            "pred_safety_floor": float(pred_floor),
            "adaptive_lower": float(lower),
            "adaptive_pred_pnl": float(pred),
        }

    def _lifecycle_v21_stop_for_row(self, row: pd.Series, bucket: str, *, sleeve: str) -> tuple[float, list[str]]:
        cfg = dict(self.v21_stop_config or {})
        base_stop = abs(float(cfg.get("base_stop", 0.0100) or 0.0100))
        stop = float(base_stop)
        reasons: list[str] = []
        pred_mae = _safe_float(row.get("pred_mae_raw", 0.0), 0.0)
        pred_tail = _safe_float(row.get("pred_tail_raw", 0.0), 0.0)
        lower = _safe_float(row.get("adaptive_lower", 0.0), 0.0)
        mae_high = float(cfg.get("mae_high", 0.0055) or 0.0055)
        mae_mid = float(cfg.get("mae_mid", 0.0035) or 0.0035)
        lower_tight = float(cfg.get("lower_tight", -0.0020) or -0.0020)
        armed = bool(lower <= lower_tight)
        if pred_mae >= mae_high or pred_tail >= mae_high:
            stop = min(stop, abs(float(cfg.get("tail_stop", stop) or stop)))
            reasons.append("tail_stop")
        elif pred_mae >= mae_mid or pred_tail >= mae_mid:
            stop = min(stop, abs(float(cfg.get("mid_stop", stop) or stop)))
            reasons.append("mid_stop")
        if "HIGH|L|e2" in str(bucket):
            stop *= float(cfg.get("long_high_mult", 1.0) or 1.0)
            reasons.append("long_high_tight")
        if lower <= lower_tight:
            stop *= float(cfg.get("lower_mult", 1.0) or 1.0)
            reasons.append("lower_tight")
        if not armed:
            reasons.append("stop_disarmed")
            if str(sleeve).lower() == "scout":
                reasons.append("scout_forced_stop")
                return float(np.clip(base_stop, 0.0025, 0.0300)), reasons
            return 999.0, reasons
        return float(np.clip(stop, 0.0025, 0.0300)), reasons

    def _lifecycle_v22_1_apply_entry_layer(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        dec: pd.Series,
        pre_adaptive_notional: float,
        v17_notional: float,
        current_notional: float,
        deep_gated_gross_meta: dict | None,
        adaptive_calibrator_meta: dict | None,
        safe_cap_meta: dict | None,
    ) -> tuple[float, str, dict]:
        adapter = self.v22_1_adapter
        meta = {
            "enabled": bool(self._lifecycle_v22_1_available()),
            "applied": False,
            "blocked": False,
            "model_id": "deep_state_safe_cap_reallocator_v22_1_scout_param_grid",
            "model_version": "V22.1",
            "adapter_version": "v22_1_learned_scout",
            "model": str(self.v22_1_model_path),
            "report": str(self.v22_1_report_path),
            "audit": str(self.v22_1_audit_path),
            "pure_mode": bool(self.v21_pure_mode),
            "input_notional": float(current_notional),
            "parent_notional": float(pre_adaptive_notional),
            "v17_notional": float(v17_notional),
        }
        if adapter is None:
            return float(current_notional), "", meta
        ledger = self._lifecycle_v21_ledger_row(
            meta_router,
            frame,
            dec=dec,
            pre_adaptive_notional=pre_adaptive_notional,
            v17_notional=v17_notional,
            deep_gated_gross_meta=deep_gated_gross_meta,
            adaptive_calibrator_meta=adaptive_calibrator_meta,
        )
        try:
            ledger, path_meta = adapter.attach_path_predictions(frame, ledger, _deep_state_v15._feature_matrix)
        except Exception as e:
            logger.warning("SYSTEM v22_1 path prediction failed closed: %s", e)
            meta.update(
                {
                    "blocked": True,
                    "reason": "v22_1_path_prediction_failed",
                    "error": str(e),
                    "output_notional": 0.0,
                    "path_model": {
                        "enabled": True,
                        "applied": False,
                        "reason": "path_prediction_failed",
                        "error": str(e),
                    },
                }
            )
            return 0.0, "", meta
        row = ledger.iloc[0]
        cap_plan_v17 = self._lifecycle_v22_1_cap_plan(
            meta_router,
            frame,
            dec=dec,
            base_notional=max(float(v17_notional), 0.0),
            deep_gated_gross_meta=deep_gated_gross_meta,
        )
        bucket = str((safe_cap_meta or {}).get("bucket", "") or cap_plan_v17.get("bucket", ""))
        if float(current_notional) > 1e-12:
            stop_raw, stop_reasons = adapter.stop_for_row(row, bucket, sleeve="core")
            scout_prob, scout_frac, _ = adapter.learned_scores(ledger)
            meta.update(
                {
                    "applied": True,
                    "sleeve": "core",
                    "output_notional": float(current_notional),
                    "bucket": bucket,
                    "path_model": dict(path_meta),
                    "cap_plan": dict(cap_plan_v17),
                    "stop_raw": float(stop_raw),
                    "stop_reasons": list(stop_reasons),
                    "scout_prob": float(scout_prob),
                    "scout_frac": float(scout_frac),
                    "scout": {
                        "adapter": "v22_1_learned_scout",
                        "source": "core_no_scout",
                        "scout_prob": float(scout_prob),
                        "scout_frac": float(scout_frac),
                        "probability_threshold": float(adapter.learned_config.get("probability_threshold", 0.0) or 0.0),
                        "cost_pass": bool(cap_plan_v17.get("cost_pass", False)),
                    },
                    "learned_config": dict(adapter.learned_config),
                    "stop_config": dict(adapter.stop_config),
                }
            )
            return float(current_notional), "core", meta

        parent_n = max(float(pre_adaptive_notional), 0.0)
        cap_plan_parent = self._lifecycle_v22_1_cap_plan(
            meta_router,
            frame,
            dec=dec,
            base_notional=parent_n,
            deep_gated_gross_meta=deep_gated_gross_meta,
        )
        router_cap = float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0))
        fee = float(getattr(meta_router, "trade_fee", 0.0) or 0.0)
        slip = float(getattr(meta_router, "trade_slip", 0.0) or 0.0)
        scout_n, scout_meta = adapter.scout_decision(
            ledger,
            cap_plan=cap_plan_parent,
            fee=fee,
            slip=slip,
            router_cap=router_cap,
        )
        bucket = str(cap_plan_parent.get("bucket", bucket))
        if scout_n <= 1e-12:
            meta.update(
                {
                    "blocked": True,
                    "reason": str(scout_meta.get("reason", "v22_1_scout_block")),
                    "scout": dict(scout_meta),
                    "scout_prob": float(scout_meta.get("scout_prob", 0.0) or 0.0),
                    "scout_frac": float(scout_meta.get("scout_frac", 0.0) or 0.0),
                    "path_model": dict(path_meta),
                }
            )
            return 0.0, "", meta

        stop_raw, stop_reasons = adapter.stop_for_row(row, bucket, sleeve="scout")
        meta.update(
            {
                "applied": True,
                "sleeve": "scout",
                "reason": "v22_1_learned_near_miss_scout",
                "output_notional": float(scout_n),
                "bucket": bucket,
                "path_model": dict(path_meta),
                "cap_plan": dict(cap_plan_parent),
                "scout": dict(scout_meta),
                "scout_prob": float(scout_meta.get("scout_prob", 0.0) or 0.0),
                "scout_frac": float(scout_meta.get("scout_frac", 0.0) or 0.0),
                "stop_raw": float(stop_raw),
                "stop_reasons": list(stop_reasons),
                "learned_config": dict(adapter.learned_config),
                "stop_config": dict(adapter.stop_config),
            }
        )
        return float(scout_n), "scout", meta

    def _lifecycle_v21_apply_entry_layer(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        dec: pd.Series,
        pre_adaptive_notional: float,
        v17_notional: float,
        current_notional: float,
        deep_gated_gross_meta: dict | None,
        adaptive_calibrator_meta: dict | None,
        safe_cap_meta: dict | None,
    ) -> tuple[float, str, dict]:
        meta = {
            "enabled": bool(self._lifecycle_v21_available()),
            "applied": False,
            "blocked": False,
            "model_id": str(self.v21_model_id or "deep_state_safe_cap_reallocator_v21_nearmiss_scout_stop"),
            "adapter_version": str(self.v21_adapter_version),
            "model_version": "V22.1" if self.v21_adapter_version == "v22_1_learned_scout" else "V21",
            "model": str(self.v21_model_path),
            "report": str(self.v21_report_path),
            "audit": str(self.v21_audit_path),
            "pure_mode": bool(self.v21_pure_mode),
            "input_notional": float(current_notional),
            "parent_notional": float(pre_adaptive_notional),
            "v17_notional": float(v17_notional),
        }
        if not self._lifecycle_v21_available():
            return float(current_notional), "", meta
        ledger = self._lifecycle_v21_ledger_row(
            meta_router,
            frame,
            dec=dec,
            pre_adaptive_notional=pre_adaptive_notional,
            v17_notional=v17_notional,
            deep_gated_gross_meta=deep_gated_gross_meta,
            adaptive_calibrator_meta=adaptive_calibrator_meta,
        )
        ledger, path_meta = self._lifecycle_v21_attach_path_predictions(frame, ledger)
        row = ledger.iloc[0]
        cap_plan_v17 = self._lifecycle_v21_cap_plan(
            meta_router,
            frame,
            dec=dec,
            base_notional=max(float(v17_notional), 0.0),
            deep_gated_gross_meta=deep_gated_gross_meta,
        )
        bucket = str((safe_cap_meta or {}).get("bucket", "") or cap_plan_v17.get("bucket", ""))
        if float(current_notional) > 1e-12:
            stop_raw, stop_reasons = self._lifecycle_v21_stop_for_row(row, bucket, sleeve="core")
            meta.update(
                {
                    "applied": True,
                    "sleeve": "core",
                    "output_notional": float(current_notional),
                    "bucket": bucket,
                    "path_model": dict(path_meta),
                    "cap_plan": dict(cap_plan_v17),
                    "stop_raw": float(stop_raw),
                    "stop_reasons": list(stop_reasons),
                    "scout_config": dict(self.v21_scout_config),
                    "stop_config": dict(self.v21_stop_config),
                    "scout": {
                        "adapter": str(self.v21_adapter_version),
                        "source": "core_no_scout",
                        "probability": 0.0,
                        "scout_frac": 0.0,
                    },
                }
            )
            return float(current_notional), "core", meta

        scout_cfg = dict(self.v21_scout_config or {})
        if self.v21_adapter_version == "v22_1_learned_scout":
            try:
                scout_meta = self._lifecycle_v22_scout_signal(ledger, row)
            except Exception as e:
                meta.update({"blocked": True, "reason": "v22_1_scout_signal_failed", "error": str(e), "path_model": dict(path_meta)})
                logger.warning("SYSTEM v22_1 learned scout signal failed: %s", e)
                return float(current_notional), "", meta
            if not scout_meta.get("broad_candidate", False):
                meta.update({"blocked": True, "reason": "v22_1_scout_not_candidate", "scout": scout_meta, "path_model": dict(path_meta)})
                return float(current_notional), "", meta
            if not scout_meta.get("eligible", False):
                meta.update({"blocked": True, "reason": "v22_1_scout_probability_block", "scout": scout_meta, "path_model": dict(path_meta)})
                return float(current_notional), "", meta

            parent_n = max(float(pre_adaptive_notional), 0.0)
            cap_plan_parent = self._lifecycle_v21_cap_plan(
                meta_router,
                frame,
                dec=dec,
                base_notional=parent_n,
                deep_gated_gross_meta=deep_gated_gross_meta,
            )
            router_cap = float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0))
            scout_max = float(scout_cfg.get("max_scout_notional", 2.0) or 2.0)
            scout_base = float(cap_plan_parent.get("planned_notional", 0.0) or 0.0)
            scout_frac = float(scout_meta.get("scout_frac", 0.25) or 0.25)
            scout_n = float(np.clip(scout_base * scout_frac, 0.0, min(scout_max, router_cap)))
            edge = float(cap_plan_parent.get("edge", 0.0) or 0.0)
            fee = float(getattr(meta_router, "trade_fee", 0.0) or 0.0)
            slip = float(getattr(meta_router, "trade_slip", 0.0) or 0.0)
            scout_buffer = float(scout_cfg.get("scout_cost_buffer", 0.0) or 0.0)
            expected_edge = float(edge * max(scout_n, 0.0))
            cost_hurdle = float(2.0 * (fee + slip) * max(scout_n, 0.0) + scout_buffer)
            scout_cost_pass = bool(expected_edge > cost_hurdle)
            bucket = str(cap_plan_parent.get("bucket", bucket))
            scout_meta.update(
                {
                    "max_scout_notional": float(scout_max),
                    "scout_base_notional": float(scout_base),
                    "output_notional": float(scout_n),
                    "edge": float(edge),
                    "expected_equity_edge": float(expected_edge),
                    "cost_hurdle": float(cost_hurdle),
                    "scout_cost_buffer": float(scout_buffer),
                    "cost_pass": bool(scout_cost_pass),
                    "cap_plan": dict(cap_plan_parent),
                }
            )
            if not scout_cost_pass or scout_n <= 1e-12:
                meta.update({"blocked": True, "reason": "v22_1_scout_cost_block", "scout": scout_meta, "path_model": dict(path_meta)})
                return float(current_notional), "", meta

            stop_raw, stop_reasons = self._lifecycle_v21_stop_for_row(row, bucket, sleeve="scout")
            meta.update(
                {
                    "applied": True,
                    "sleeve": "scout",
                    "reason": "v22_1_learned_scout",
                    "output_notional": float(scout_n),
                    "bucket": bucket,
                    "path_model": dict(path_meta),
                    "cap_plan": dict(cap_plan_parent),
                    "scout": scout_meta,
                    "stop_raw": float(stop_raw),
                    "stop_reasons": list(stop_reasons),
                    "scout_config": dict(self.v21_scout_config),
                    "stop_config": dict(self.v21_stop_config),
                }
            )
            return float(scout_n), "scout", meta

        lower = _safe_float(row.get("adaptive_lower", -999.0), -999.0)
        pred = _safe_float(row.get("adaptive_pred_pnl", -999.0), -999.0)
        parent_n = max(float(pre_adaptive_notional), 0.0)
        lower_min = float(scout_cfg.get("lower_min", -0.0100) or -0.0100)
        pred_min = float(scout_cfg.get("pred_min", -0.0060) or -0.0060)
        eligible = bool(parent_n > 1e-12 and lower > lower_min and lower <= -0.002 and pred >= pred_min)
        scout_meta = {
            "eligible": bool(eligible),
            "lower_min": float(lower_min),
            "pred_min": float(pred_min),
            "adaptive_lower": float(lower),
            "adaptive_pred_pnl": float(pred),
        }
        if not eligible:
            meta.update({"blocked": True, "reason": "v21_scout_not_eligible", "scout": scout_meta, "path_model": dict(path_meta)})
            return float(current_notional), "", meta

        cap_plan_parent = self._lifecycle_v21_cap_plan(
            meta_router,
            frame,
            dec=dec,
            base_notional=parent_n,
            deep_gated_gross_meta=deep_gated_gross_meta,
        )
        router_cap = float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0))
        scout_frac = float(scout_cfg.get("scout_frac", 0.25) or 0.25)
        scout_max = float(scout_cfg.get("max_scout_notional", 1.25) or 1.25)
        scout_base = float(cap_plan_parent.get("planned_notional", 0.0) or 0.0)
        scout_n = float(np.clip(scout_base * scout_frac, 0.0, min(scout_max, router_cap)))
        edge = float(cap_plan_parent.get("edge", 0.0) or 0.0)
        fee = float(getattr(meta_router, "trade_fee", 0.0) or 0.0)
        slip = float(getattr(meta_router, "trade_slip", 0.0) or 0.0)
        scout_buffer = float(scout_cfg.get("scout_cost_buffer", 0.0) or 0.0)
        expected_edge = float(edge * max(scout_n, 0.0))
        cost_hurdle = float(2.0 * (fee + slip) * max(scout_n, 0.0) + scout_buffer)
        scout_cost_pass = bool(expected_edge > cost_hurdle)
        bucket = str(cap_plan_parent.get("bucket", bucket))
        scout_meta.update(
            {
                "scout_frac": float(scout_frac),
                "max_scout_notional": float(scout_max),
                "scout_base_notional": float(scout_base),
                "output_notional": float(scout_n),
                "edge": float(edge),
                "expected_equity_edge": float(expected_edge),
                "cost_hurdle": float(cost_hurdle),
                "scout_cost_buffer": float(scout_buffer),
                "cost_pass": bool(scout_cost_pass),
                "cap_plan": dict(cap_plan_parent),
            }
        )
        if not scout_cost_pass or scout_n <= 1e-12:
            meta.update({"blocked": True, "reason": "v21_scout_cost_block", "scout": scout_meta, "path_model": dict(path_meta)})
            return float(current_notional), "", meta

        stop_raw, stop_reasons = self._lifecycle_v21_stop_for_row(row, bucket, sleeve="scout")
        meta.update(
            {
                "applied": True,
                "sleeve": "scout",
                "reason": "v21_near_miss_scout",
                "output_notional": float(scout_n),
                "bucket": bucket,
                "path_model": dict(path_meta),
                "cap_plan": dict(cap_plan_parent),
                "scout": scout_meta,
                "stop_raw": float(stop_raw),
                "stop_reasons": list(stop_reasons),
                "scout_config": dict(self.v21_scout_config),
                "stop_config": dict(self.v21_stop_config),
            }
        )
        return float(scout_n), "scout", meta

    @staticmethod
    def _adaptive_calibrator_cfg_get(cfg: AdaptiveConfig | dict | None, key: str, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    @staticmethod
    def _adaptive_calibrator_regime(row) -> str:
        if hasattr(row, "get"):
            for name in ("bull", "bear", "chop", "whipsaw", "normal"):
                if _safe_float(row.get(f"regime_{name}", 0.0), 0.0) >= 0.5:
                    return name.upper()
        return "UNKNOWN"

    def _adaptive_calibrator_ledger_row(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        dec: pd.Series,
        current_notional: float,
        deep_gated_gross_meta: dict | None,
    ) -> pd.DataFrame:
        i = int(max(0, len(frame) - 1))
        row = frame.iloc[i] if frame is not None and len(frame) else {}
        signal = dict((deep_gated_gross_meta or {}).get("signal", {}) or {})
        account_ctx = dict((deep_gated_gross_meta or {}).get("account_context", {}) or {})
        if not account_ctx:
            account_ctx = self._lifecycle_v1_daily_context(meta_router, frame)
        action = str((deep_gated_gross_meta or {}).get("bucket", "") or "UNKNOWN").upper()
        if action.startswith("COST3"):
            action = "DEFENSIVE"
        if action not in {"HIGH", "MID", "DEFENSIVE"}:
            action = "UNKNOWN"
        ts = str(row.get("timestamp", i)) if hasattr(row, "get") else str(i)
        side = int(np.sign(int(getattr(dec, "side", 0) or 0)))
        return pd.DataFrame(
            [
                {
                    "trade_id": 0,
                    "entry_idx": i,
                    "core_exit_idx": i,
                    "timestamp": ts,
                    "core_side": side,
                    "action": action,
                    "regime": self._adaptive_calibrator_regime(row),
                    "core_notional": float(getattr(dec, "notional_exposure", current_notional) or current_notional),
                    "effective_core_notional": float(current_notional),
                    "leverage": float(getattr(dec, "leverage", 1.0) or 1.0),
                    "account_dd_prior": _safe_float(account_ctx.get("account_dd", 0.0), 0.0),
                    "daily_dd_prior": _safe_float(account_ctx.get("daily_dd_proxy", 0.0), 0.0),
                    "loss_streak_prior": _safe_float(account_ctx.get("loss_streak", 0.0), 0.0),
                    "hybrid_same_pred": _safe_float(signal.get("same_pred", 0.0), 0.0),
                    "hybrid_adverse_pred": _safe_float(signal.get("adverse_pred", 0.0), 0.0),
                    "deep_pred_full": _safe_float(signal.get("deep_full", 0.0), 0.0),
                    "deep_pred_adverse": _safe_float(signal.get("deep_adverse", 0.0), 0.0),
                    "deep_pred_same": _safe_float(signal.get("deep_same", 0.0), 0.0),
                    "deep_conviction": _safe_float(signal.get("conviction", 0.0), 0.0),
                    "deep_adverse_gate": _safe_float(signal.get("adverse", 0.0), 0.0),
                }
            ]
        )

    def _lifecycle_v1_apply_deep_state_adaptive_calibrator(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        dec: pd.Series,
        current_notional: float,
        deep_gated_gross_meta: dict | None,
    ) -> tuple[float, str, dict]:
        cfg = self.deep_state_adaptive_config
        cal = self.deep_state_adaptive_calibrator
        models = self.deep_state_router_models
        meta = {
            "enabled": bool(self.deep_state_adaptive_calibrator_enabled),
            "applied": False,
            "blocked": False,
            "model_id": "deep_state_safe_cap_reallocator_v17_adaptive_calibrator",
            "model": str(self.deep_state_adaptive_calibrator_model_path),
            "report": str(self.deep_state_adaptive_calibrator_report_path),
            "audit": str(self.deep_state_adaptive_calibrator_audit_path),
            "selected_config": str(self._adaptive_calibrator_cfg_get(cfg, "name", "")),
            "input_notional": float(current_notional),
        }
        if (
            not bool(self.deep_state_adaptive_calibrator_enabled)
            or cfg is None
            or cal is None
            or models is None
            or frame is None
            or len(frame) == 0
            or current_notional <= 1e-12
            or not bool((deep_gated_gross_meta or {}).get("applied", False))
        ):
            return float(current_notional), "noop", meta
        try:
            ledger = self._adaptive_calibrator_ledger_row(
                meta_router,
                frame,
                dec=dec,
                current_notional=current_notional,
                deep_gated_gross_meta=deep_gated_gross_meta,
            )
            x = _deep_state_v15._feature_matrix(frame, ledger)
            pred = _deep_state_v15._predict_router(models, x)
            meta_frame = _deep_state_v17._meta_frame(frame, ledger, cal.risk_cuts)
            qv = _deep_state_v17._adaptive_q(meta_frame, cal, cfg, self.deep_state_adaptive_future_rolling_q)
            pred_pnl = float(pred["pred_pnl"][0]) if len(pred["pred_pnl"]) else 0.0
            adaptive_q = float(qv[0]) if len(qv) else 0.0
            lower = float(pred_pnl - adaptive_q)
            lower_block = float(self._adaptive_calibrator_cfg_get(cfg, "lower_block", -0.006) or -0.006)
            lower_keep = float(self._adaptive_calibrator_cfg_get(cfg, "lower_keep", -0.002) or -0.002)
            shrink_scale = float(self._adaptive_calibrator_cfg_get(cfg, "shrink_scale", 0.70) or 0.70)
            shrink_scale = float(np.clip(shrink_scale, 0.0, 1.0))
            if lower < lower_block:
                scale = 0.0
                reason = "adaptive_calibrator_block"
            elif lower < lower_keep:
                scale = shrink_scale
                reason = "adaptive_calibrator_shrink"
            else:
                scale = 1.0
                reason = "adaptive_calibrator_keep"
            out = float(np.clip(float(current_notional) * scale, 0.0, float(current_notional)))
            meta.update(
                {
                    "applied": True,
                    "blocked": bool(scale <= 1e-12),
                    "reason": reason,
                    "output_notional": float(out),
                    "router_scale": float(scale),
                    "adaptive_q": float(adaptive_q),
                    "adaptive_lower": float(lower),
                    "adaptive_pred_pnl": float(pred_pnl),
                    "adaptive_mode": str(self._adaptive_calibrator_cfg_get(cfg, "mode", "")),
                    "future_rolling_q": (
                        None
                        if self.deep_state_adaptive_future_rolling_q is None
                        else float(self.deep_state_adaptive_future_rolling_q)
                    ),
                    "lower_block": float(lower_block),
                    "lower_keep": float(lower_keep),
                    "shrink_scale": float(shrink_scale),
                    "pred_adverse": float(pred["pred_adverse"][0]) if len(pred["pred_adverse"]) else 0.0,
                    "cost2_survival": float(pred["cost2_survival"][0]) if len(pred["cost2_survival"]) else 0.0,
                    "cost3_survival": float(pred["cost3_survival"][0]) if len(pred["cost3_survival"]) else 0.0,
                    "cluster": int(pred["cluster"][0]) if len(pred["cluster"]) else 0,
                    "cluster_score": float(pred["cluster_score"][0]) if len(pred["cluster_score"]) else 0.0,
                    "anomaly_score": float(pred["anomaly_score"][0]) if len(pred["anomaly_score"]) else 0.0,
                    "risk_score": float(meta_frame["risk"].iloc[0]) if len(meta_frame) else 0.0,
                    "risk_bin": str(meta_frame["risk_bin"].iloc[0]) if len(meta_frame) else "",
                }
            )
            return float(out), reason, meta
        except Exception as e:
            meta.update({"error": str(e), "output_notional": float(current_notional)})
            logger.warning("SYSTEM deep_state_adaptive_calibrator signal failed: %s", e)
            return float(current_notional), "noop", meta

    def _lifecycle_v1_apply_deep_constant_gross(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        current_notional: float,
    ) -> tuple[float, str, dict]:
        report_cfg = dict(self.deep_constant_gross_report.get("selected_config", {}) or {})
        report_gate = dict(self.deep_constant_gross_report.get("promotion_gate", {}) or {})
        meta = {
            "enabled": bool(self.deep_constant_gross_enabled),
            "applied": False,
            "model_id": str(self.deep_constant_gross_report.get("model_id", "clean_base_deep_constant_gross_v1")),
            "report": str(self.deep_constant_gross_report_path),
            "selected_config": str(report_cfg.get("name", "")),
            "target_500_pnl": bool(report_gate.get("target_500_pnl", False)),
            "input_notional": float(current_notional),
            "target_notional": float(self.deep_constant_gross_target_notional),
            "cost3_notional": float(self.deep_constant_gross_cost3_notional),
        }
        if not bool(self.deep_constant_gross_enabled):
            return float(current_notional), "noop", meta

        cap = float(min(
            float(self.lifecycle_v1_cfg.get("max_notional", 3.6) or 3.6),
            float(self.lifecycle_v1_risk_cfg.get("max_notional", 3.6) or 3.6),
            float(getattr(meta_router, "exposure_cap", 5.0) or 5.0),
        ))
        target = float(np.clip(float(self.deep_constant_gross_target_notional), 0.0, cap))
        cost3_mode = bool(
            float(getattr(meta_router, "trade_fee", 0.0) or 0.0) >= float(FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_FEE)
            or float(getattr(meta_router, "trade_slip", 0.0) or 0.0) >= float(FINAL_GOVERNOR_DEEP_CONSTANT_GROSS_COST3_SLIP)
        )
        stress_state = bool(self._lifecycle_v1_stress_state(frame))
        if cost3_mode:
            out = float(np.clip(float(self.deep_constant_gross_cost3_notional), 0.0, cap))
            edit = "deep_constant_gross_cost3_preserve" if out <= 1e-12 else "deep_constant_gross_cost3_low"
        elif stress_state:
            out = float(np.clip(min(float(current_notional), target), 0.0, cap))
            edit = "deep_constant_gross_defensive"
        else:
            out = target
            edit = "deep_constant_gross_open"
        meta.update(
            {
                "applied": True,
                "cap": float(cap),
                "output_notional": float(out),
                "edit": str(edit),
                "cost3_mode": bool(cost3_mode),
                "stress_state": bool(stress_state),
                "fee_rate": float(getattr(meta_router, "trade_fee", 0.0) or 0.0),
                "slippage_rate": float(getattr(meta_router, "trade_slip", 0.0) or 0.0),
            }
        )
        return float(out), str(edit), meta

    def _dsac_overlay_features(self, frame: pd.DataFrame) -> dict:
        if frame is None or len(frame) == 0:
            return {}
        row = frame.iloc[-1]
        features: dict[str, float] = {}
        for col, val in row.items():
            if str(col) == "timestamp":
                continue
            try:
                x = float(val)
            except Exception:
                continue
            if np.isfinite(x):
                features[str(col)] = float(x)
        return features

    def _dsac_overlay_edge_proxy(self, frame: pd.DataFrame, deep_gated_gross_meta: dict | None) -> float:
        vals: list[float] = []
        signal = dict((deep_gated_gross_meta or {}).get("signal", {}) or {})
        for key in ("same_pred", "deep_same", "deep_full", "conviction"):
            vals.append(_safe_float(signal.get(key, 0.0), 0.0))
        finite = [float(x) for x in vals if np.isfinite(float(x))]
        return float(max([0.0] + finite))

    def _lifecycle_v1_apply_dsac_overlay(
        self,
        meta_router,
        frame: pd.DataFrame,
        *,
        side: int,
        current_notional: float,
        deep_gated_gross_meta: dict | None,
    ) -> tuple[float, str, dict]:
        cap = float(min(
            float(self.lifecycle_v1_cfg.get("max_notional", 3.6) or 3.6),
            float(self.lifecycle_v1_risk_cfg.get("max_notional", 3.6) or 3.6),
            float(getattr(meta_router, "exposure_cap", 5.0) or 5.0),
        ))
        n0 = float(np.clip(float(current_notional), 0.0, cap))
        meta = {
            "enabled": bool(self.dsac_overlay_enabled),
            "applied": False,
            "blocked": False,
            "checkpoint": str(self.dsac_overlay_ckpt_path),
            "checkpoint_meta": dict(self.dsac_overlay_ckpt_meta),
            "mode": str(self.dsac_overlay_mode),
            "threshold": float(self.dsac_overlay_threshold),
            "scale": float(self.dsac_overlay_scale),
            "cost_gate_enabled": bool(self.dsac_overlay_cost_gate_enabled),
            "cost_buffer": float(self.dsac_overlay_cost_buffer),
            "input_notional": float(n0),
            "output_notional": float(n0),
        }
        if not bool(self.dsac_overlay_enabled) or self.dsac_overlay_router is None or n0 <= 1e-12:
            return float(n0), "noop", meta

        dsac_side = 0
        dsac_score = 0.0
        dsac_info: dict = {}
        try:
            action_int, _dsac_leverage, info = self.dsac_overlay_router.decide(
                self._dsac_overlay_features(frame),
                {"type": None, "entry_price": 0.0, "unrealized": 0.0, "mdd": 0.0, "hold_count": 0.0},
            )
            dsac_info = dict(info or {})
            dsac_side = 1 if int(action_int) == 1 else (-1 if int(action_int) == 2 else 0)
            dsac_score = float(_safe_float(dsac_info.get("score", abs(_safe_float(dsac_info.get("raw_action", 0.0), 0.0))), 0.0))
        except Exception as e:
            meta.update({"error": str(e), "output_notional": float(n0)})
            logger.warning("SYSTEM dsac_overlay signal failed: %s", e)
            return float(n0), "dsac_signal_error", meta

        edge = self._dsac_overlay_edge_proxy(frame, deep_gated_gross_meta)
        fee = float(getattr(meta_router, "trade_fee", 0.0) or 0.0)
        slip = float(getattr(meta_router, "trade_slip", 0.0) or 0.0)
        expected_edge = float(edge * n0)
        round_trip_hurdle = float(2.0 * (fee + slip) * n0 + self.dsac_overlay_cost_buffer)
        meta.update(
            {
                "dsac_action": int(action_int),
                "dsac_side": int(dsac_side),
                "dsac_score": float(dsac_score),
                "dsac_info": dsac_info,
                "core_side": int(side),
                "edge_proxy": float(edge),
                "expected_edge": float(expected_edge),
                "round_trip_hurdle": float(round_trip_hurdle),
                "fee_rate": float(fee),
                "slippage_rate": float(slip),
            }
        )
        if bool(self.dsac_overlay_cost_gate_enabled) and expected_edge <= round_trip_hurdle:
            meta.update(
                {
                    "applied": True,
                    "blocked": True,
                    "reason": "dsac_cost_gate_block",
                    "output_notional": 0.0,
                }
            )
            return 0.0, "dsac_cost_gate_block", meta

        out = float(n0)
        edit = "dsac_confirm"
        if (
            self.dsac_overlay_mode == "half_if_opposite"
            and int(dsac_side) == -int(side)
            and float(dsac_score) >= float(self.dsac_overlay_threshold)
        ):
            out = float(np.clip(n0 * self.dsac_overlay_scale, 0.0, cap))
            edit = "dsac_half_opposite"
            meta.update({"applied": True, "reason": edit})
        else:
            meta.update({"applied": False, "reason": edit})
        meta["output_notional"] = float(out)
        return float(out), str(edit), meta

    def _reset_lifecycle_v1_position_state(self) -> None:
        self.active_lifecycle_v1_base_notional = 0.0
        self.active_lifecycle_v1_effective_notional = 0.0
        self.active_lifecycle_v1_leverage = 1.0
        self.active_lifecycle_v1_cooldown_bars = 0
        self.active_lifecycle_v1_quality_score = 0.0
        self.active_lifecycle_v1_confidence = 0.0
        self.active_lifecycle_v1_entry_bucket = ""
        self.active_lifecycle_v1_entry_hazard = 0.0
        self.active_lifecycle_v1_entry_support = 0
        self.active_lifecycle_v1_edit = ""
        self.active_lifecycle_v1_take_profit = 0.0
        self.active_lifecycle_v1_stop_loss = 0.0
        self.active_lifecycle_v1_max_hold_bars = 0
        self.active_lifecycle_v1_jackpot_added = False
        self.active_lifecycle_v1_mae_unrealized = 0.0
        self.active_lifecycle_v1_v21_sleeve = ""
        self.active_lifecycle_v1_v21_stop_raw = 999.0
        self.active_lifecycle_v1_v21_peak_raw = -1e9
        self.active_lifecycle_v1_v21_stop_reasons = []
        self.active_lifecycle_v1_scout_model_version = ""
        self.active_lifecycle_v1_scout_model_id = ""
        self.active_lifecycle_v1_scout_model_path = ""
        self.active_lifecycle_v1_scout_prob = 0.0
        self.active_lifecycle_v1_scout_frac = 0.0
        self.active_lifecycle_v1_scout_probability_threshold = 0.0
        self.active_lifecycle_v1_scout_cost_pass = False
        self.active_v31_entry_edge = 0.0
        self.active_v31_entry_margin = 0.0
        self.active_v31_entry_vol_anchor = 0.0
        self.active_v31_entry_q_long = 0.0
        self.active_v31_entry_q_short = 0.0
        self.active_v31_entry_q_long_raw = 0.0
        self.active_v31_entry_q_short_raw = 0.0
        self.active_v31_entry_selected_side = ""
        self.active_v31_entry_guard_reason = ""
        self.active_lifecycle_v1_conformal_core_notional = 0.0
        self.active_lifecycle_v1_conformal_sleeve_notional = 0.0
        self.active_lifecycle_v1_conformal_sleeve_exit_bars = 0
        self.active_lifecycle_v1_conformal_sleeve_action = ""

    def _v31_available(self) -> bool:
        return bool(
            self.v31_enabled
            and self.v31_v27_model is not None
            and isinstance(self.v31_v27_payload, dict)
            and bool(self.v31_cfg)
            and self._v21_2_jackpot_available()
        )

    def _alpha2_1_available(self) -> bool:
        return bool(
            self.alpha2_1_teacher_model is not None
            and self.alpha2_1_teacher_feature_cols
            and self.alpha2_1_teacher_norm
            and self._v31_available()
            and self._v21_2_jackpot_available()
        )

    def _alpha2_1_predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame | None:
        if not self._alpha2_1_available() or _alpha2_apply_norm is None or _alpha2_seq_tensor is None:
            return None
        close = (
            pd.to_numeric(frame["close"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .to_numpy(dtype=np.float64)
        )
        try:
            cols = list(self.alpha2_1_teacher_feature_cols)
            key = self._decision_frame_cache_key(frame, bundle=self.alpha2_1_teacher_payload, feature_cols=cols)
            cache = getattr(self, "_alpha2_1_teacher_decision_frame_cache", None)
            if isinstance(cache, dict) and cache.get("key") == key:
                return cache.get("value")
            features = prepare_fully_learned_governor_features(
                frame,
                side_hint=0,
                close=close,
                feature_cols=cols,
            )
            features = features.reindex(columns=cols, fill_value=0.0)
            if len(features) == 0:
                return None
            idx = np.arange(len(features), dtype=np.int64)
            seq = _alpha2_seq_tensor(features, idx, cols)
            x = _alpha2_apply_norm(seq, self.alpha2_1_teacher_norm)
            device = torch.device("cpu")
            try:
                device = next(self.alpha2_1_teacher_model.parameters()).device
            except Exception:
                pass
            action_rows: list[np.ndarray] = []
            notional_rows: list[np.ndarray] = []
            quality_rows: list[np.ndarray] = []
            try:
                self.alpha2_1_teacher_model.eval()
            except Exception:
                pass
            with torch.no_grad():
                for start in range(0, len(x), 512):
                    xb = torch.from_numpy(x[start : start + 512].astype(np.float32)).to(device)
                    logits, quality, nlogits = self.alpha2_1_teacher_model(xb)
                    action_rows.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                    notional_rows.append(torch.softmax(nlogits, dim=1).detach().cpu().numpy())
                    quality_rows.append(quality.detach().cpu().numpy().reshape(-1))
            action_proba = np.vstack(action_rows)
            notional_proba = np.vstack(notional_rows)
            quality = np.concatenate(quality_rows)
            pred_action = np.argmax(action_proba, axis=1).astype(np.int64)
            confidence = np.max(action_proba, axis=1).astype(np.float64)
            out = pd.DataFrame(
                {
                    "pred_action": pred_action,
                    "confidence": confidence,
                    "quality": quality.astype(np.float64),
                    "action_proba": [[float(v) for v in row] for row in action_proba],
                    "notional_proba": [[float(v) for v in row] for row in notional_proba],
                },
                index=features.index,
            )
            self._alpha2_1_teacher_decision_frame_cache = {"key": key, "value": out}
            return out
        except Exception as e:
            logger.warning("SYSTEM alpha2_1=SKIP reason=teacher_predict_failed err=%s", e)
            return None

    def _alpha2_1_predict_latest(self, frame: pd.DataFrame) -> dict | None:
        try:
            preds = self._alpha2_1_predict_frame(frame)
            if preds is None or len(preds) == 0:
                return None
            row = preds.iloc[-1]
            action_proba = list(row["action_proba"])
            notional_proba = list(row["notional_proba"])
            q = float(row["quality"])
            pred_action = int(np.argmax(action_proba))
            conf = float(np.max(action_proba))
            return {
                "action_proba": [float(x) for x in action_proba],
                "notional_proba": [float(x) for x in notional_proba],
                "pred_action": pred_action,
                "confidence": conf,
                "quality": q,
                "decision_frame_mode": "full_frame_latest",
                "decision_frame_rows": int(len(preds)),
            }
        except Exception as e:
            logger.warning("SYSTEM alpha2_1=SKIP reason=teacher_predict_failed err=%s", e)
            return None

    def _alpha2_1_apply_parent_gate(self, dec: pd.Series, frame: pd.DataFrame, trace: dict) -> tuple[pd.Series | None, dict]:
        meta = {
            "enabled": True,
            "model_id": self.alpha2_1_model_id,
            "teacher_model": str(self.alpha2_1_teacher_model_path),
            "report": str(self.alpha2_1_report_path),
            "audit": str(self.alpha2_1_audit_path),
            "runtime": {
                "name": "noflip_c0.56_parent_scale1.10",
                "confidence": float(self.alpha2_1_confidence),
                "skip_on_cash": True,
                "allow_flip": False,
                "use_learned_size": False,
                "parent_notional_scale": float(self.alpha2_1_parent_notional_scale),
                "max_notional": float(self.alpha2_1_max_notional),
            },
        }
        pred = self._alpha2_1_predict_latest(frame)
        if pred is None:
            meta.update({"blocked": True, "reason": "teacher_prediction_unavailable"})
            return None, meta
        teacher_active = int(dec.action) != int(FULLY_LEARNED_ACTION_CASH) and int(dec.side) != 0
        keep = bool(
            teacher_active
            and float(pred["confidence"]) >= float(self.alpha2_1_confidence)
            and int(pred["pred_action"]) != int(FULLY_LEARNED_ACTION_CASH)
        )
        meta.update(
            {
                "teacher_action_proba": list(pred["action_proba"]),
                "teacher_pred_action": int(pred["pred_action"]),
                "teacher_confidence": float(pred["confidence"]),
                "teacher_quality": float(pred["quality"]),
                "teacher_decision_frame_mode": str(pred.get("decision_frame_mode", "")),
                "teacher_decision_frame_rows": int(pred.get("decision_frame_rows", 0) or 0),
                "keep_parent": bool(keep),
                "parent_action_before": int(dec.action),
                "parent_side_before": int(dec.side),
                "parent_notional_before": float(dec.notional_exposure),
            }
        )
        if not keep:
            meta.update({"blocked": True, "reason": "alpha2_1_teacher_gate_pruned_parent"})
            return None, meta
        out = dec.copy()
        scaled = float(
            np.clip(
                float(out.notional_exposure) * float(self.alpha2_1_parent_notional_scale),
                0.0,
                float(self.alpha2_1_max_notional),
            )
        )
        leverage = float(max(float(out.leverage or 1.0), 1.0))
        out.loc["notional_exposure"] = scaled
        out.loc["position_fraction"] = float(scaled / max(leverage, 1e-12))
        out.loc["quality_score"] = float(pred["quality"])
        out.loc["confidence"] = float(pred["confidence"])
        meta.update(
            {
                "blocked": False,
                "reason": "alpha2_1_teacher_gate_keep_parent",
                "parent_notional_after": float(scaled),
                "parent_position_fraction_after": float(out.position_fraction),
            }
        )
        trace["alpha2_1"] = dict(meta)
        return out, meta

    def _alpha2_1_constrained_decision_row(self, dec: pd.Series, frame: pd.DataFrame) -> pd.Series | None:
        pred = self._alpha2_1_predict_latest(frame)
        if pred is None:
            return None
        out = dec.copy()
        pred_action = int(pred["pred_action"])
        conf = float(pred["confidence"])
        teacher_active = int(out.action) != int(FULLY_LEARNED_ACTION_CASH) and int(out.side) != 0
        active = bool(
            teacher_active
            and conf >= float(self.alpha2_1_confidence)
            and pred_action != int(FULLY_LEARNED_ACTION_CASH)
        )
        out.loc["quality_score"] = float(pred["quality"])
        out.loc["confidence"] = conf
        if not active:
            out.loc[["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
            out.loc["leverage"] = 1.0
            return out
        scaled = float(
            np.clip(
                float(out.notional_exposure) * float(self.alpha2_1_parent_notional_scale),
                0.0,
                float(self.alpha2_1_max_notional),
            )
        )
        side = int(out.side)
        leverage = float(max(float(out.leverage or 1.0), 1.0))
        out.loc["side"] = side
        out.loc["action"] = int(FULLY_LEARNED_ACTION_LONG if side > 0 else FULLY_LEARNED_ACTION_SHORT)
        out.loc["notional_exposure"] = scaled
        out.loc["position_fraction"] = float(scaled / max(leverage, 1e-12))
        return out

    def _v31_cfg_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(dict(self.v31_cfg or {}).get(key, default))
        except Exception:
            return float(default)

    def _v31_cfg_int(self, key: str, default: int = 0) -> int:
        return int(float(self._v31_cfg_float(key, float(default))))

    @staticmethod
    def _v31_state24_dominant_regime(row: pd.Series) -> str:
        cols = {
            "bull": f"{CLEAN_REGIME4_STICKY_RUNTIME_PREFIX}bull_prob",
            "bear": f"{CLEAN_REGIME4_STICKY_RUNTIME_PREFIX}bear_prob",
            "chop": f"{CLEAN_REGIME4_STICKY_RUNTIME_PREFIX}chop_prob",
            "whipsaw": f"{CLEAN_REGIME4_STICKY_RUNTIME_PREFIX}whipsaw_prob",
        }
        missing = [col for col in cols.values() if col not in row.index]
        if missing:
            raise RuntimeError(f"v31 side specialist missing state24 regime cols: {missing}")
        probs = {name: _safe_float(row.get(col, 0.0), 0.0) for name, col in cols.items()}
        if max(abs(v) for v in probs.values()) <= 1e-12:
            raise RuntimeError("v31 side specialist state24 regime probabilities are zero-like")
        return str(max(probs, key=probs.get))

    def _v31_predict_latest(self, frame: pd.DataFrame) -> tuple[float, float] | None:
        if not self._v31_available() or _v31_live is None:
            return None
        payload = dict(self.v31_v27_payload or {})
        seq_cols = list(payload.get("seq_cols") or [])
        norm = payload.get("norm")
        if not seq_cols or not isinstance(norm, dict):
            return None
        missing = [c for c in seq_cols if c not in frame.columns]
        if missing:
            logger.warning("SYSTEM v31_deep_alpha=SKIP reason=missing_sequence_features cols=%s", ",".join(missing[:8]))
            return None
        idx = len(frame) - 1
        if idx < 0:
            return None
        try:
            seq = _v31_live._seq_at(frame.reset_index(drop=True), idx, seq_cols)
            x = _v31_live._apply_norm(seq[None, :, :].astype(np.float32), norm)
            with torch.no_grad():
                out = self.v31_v27_model(torch.from_numpy(x)).detach().cpu().numpy()[0]
            return float(out[0]), float(out[1])
        except Exception as e:
            logger.warning("SYSTEM v31_deep_alpha=SKIP reason=predict_failed err=%s", e)
            return None

    def _v31_deep_alpha_entry_decision(
        self,
        frame: pd.DataFrame,
        *,
        meta_router,
        regime: str,
        raw_regime: str,
        parent_dec,
        parent_trace: dict,
    ) -> tuple[int, float, float, float, dict, str] | None:
        if not self._v31_available() or _v31_live is None:
            return None
        trace = dict(parent_trace or {})
        trace["v31"] = {
            "enabled": True,
            "report": str(self.v31_report_path),
            "audit": str(self.v31_audit_path),
            "v27_model": str(self.v31_v27_model_path),
            "selected_config": dict(self.v31_cfg or {}),
            "parent_action": int(getattr(parent_dec, "action", parent_dec.get("action", 0) if isinstance(parent_dec, pd.Series) else 0) or 0),
            "parent_side": int(getattr(parent_dec, "side", parent_dec.get("side", 0) if isinstance(parent_dec, pd.Series) else 0) or 0),
            "teacher_gate_result": "not_evaluated_parent_cash",
            "parent_contract": "deep_sleeve_only_when_parent_cash",
            "alpha2_1_contract": "parent_cash_preserved_by_teacher_gate",
            "decision_frame_mode": "full_frame_latest_sequence",
            "decision_frame_rows": int(len(frame)),
            "sequence_contract": "v27_seq_at_frame_latest",
        }
        if self.v31_deep_cooldown_left > 0:
            self.v31_deep_cooldown_left -= 1
            self._save_runtime_state()
            trace["v31"]["cooldown_left_after_decrement"] = int(self.v31_deep_cooldown_left)
            if self.v31_deep_cooldown_left > 0:
                info = {
                    "agent": "FINAL_GOVERNOR",
                    "source": "alpha3|v31_frozen_v27_rule_exit|cooldown",
                    "position_signal": "HOLD",
                    "position_reason": "v31_deep_alpha_cooldown",
                    "score": float(getattr(parent_dec, "quality_score", 0.0) if not isinstance(parent_dec, pd.Series) else parent_dec.get("quality_score", 0.0)),
                    "conviction": float(getattr(parent_dec, "confidence", 0.0) if not isinstance(parent_dec, pd.Series) else parent_dec.get("confidence", 0.0)),
                    "owner": "",
                    "regime": regime,
                    "decision_logic": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                    "model_version": "Alpha3",
                    "model_id": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                    "model_path": str(self.v31_v27_model_path),
                    "model_sleeve": "deep_alpha",
                    "scout_prob": 0.0,
                    "scout_frac": 0.0,
                    "scout_probability_threshold": float(self._v31_cfg_float("edge_th", 0.010)),
                    "scout_cost_pass": True,
                    "sleeve_trace": trace,
                }
                return 0, 0.0, 0.0, 1.0, info, regime.upper()

        pred = self._v31_predict_latest(frame)
        if pred is None:
            return None
        q_long, q_short = pred
        q_long_raw = float(q_long)
        q_short_raw = float(q_short)
        regime_u = str(regime or raw_regime or "").upper()
        regime_trace = dict(frame.attrs.get("regime_predictor_trace", {}) or {})
        transition_risk = float(regime_trace.get("transition_risk", 0.0) or 0.0)
        guard_reasons: list[str] = []
        side = 1 if q_long_raw > q_short_raw else -1
        edge = float(max(q_long_raw, q_short_raw))
        margin = float(abs(q_long_raw - q_short_raw))
        edge_th = float(self._v31_cfg_float("edge_th", 0.010))
        margin_th = float(self._v31_cfg_float("margin_th", 0.004))
        row = frame.iloc[-1]
        side_name = "long" if side > 0 else "short"
        effective_edge_th = float(edge_th)
        effective_margin_th = float(margin_th)
        state24_regime = ""
        if bool(dict(self.v31_cfg or {}).get("deep_side_specialist_gate", False)):
            effective_edge_th = float(edge_th * self._v31_cfg_float(f"deep_{side_name}_edge_mult", 1.0))
            effective_margin_th = float(margin_th * self._v31_cfg_float(f"deep_{side_name}_margin_mult", 1.0))
            state24_regime = self._v31_state24_dominant_regime(row)
            block_key = "deep_long_block_regimes" if side > 0 else "deep_short_block_regimes"
            blocked = {str(x).strip().lower() for x in list(dict(self.v31_cfg or {}).get(block_key, []) or [])}
            if state24_regime.lower() in blocked:
                guard_reasons.append(f"v31_deep_alpha_{state24_regime}_{side_name}_veto")
        pass_gate = bool(edge >= effective_edge_th and margin >= effective_margin_th and not guard_reasons)
        raw_margin = float(abs(q_long_raw - q_short_raw))
        if (
            bool(dict(self.v31_cfg or {}).get("deep_block_long_in_bear_regime", False))
            and side > 0
            and regime_u == "BEAR"
        ):
            guard_reasons.append("v31_deep_alpha_bear_long_veto")
            pass_gate = False
        vol_anchor = float(_v31_live._vol_anchor(row))
        trace["v31"].update(
            {
                "q_long": float(q_long_raw),
                "q_short": float(q_short_raw),
                "q_long_raw": float(q_long_raw),
                "q_short_raw": float(q_short_raw),
                "selected_side": "LONG" if side > 0 else "SHORT",
                "edge": float(edge),
                "margin": float(margin),
                "raw_margin": float(raw_margin),
                "edge_threshold": float(effective_edge_th),
                "margin_threshold": float(effective_margin_th),
                "base_edge_threshold": float(edge_th),
                "base_margin_threshold": float(margin_th),
                "side_specialist_gate": bool(dict(self.v31_cfg or {}).get("deep_side_specialist_gate", False)),
                "state24_dominant_regime": str(state24_regime),
                "regime_long_guard_reasons": list(guard_reasons),
                "transition_risk": float(transition_risk),
                "entry_vol_anchor_raw": float(vol_anchor),
                "entry_vol_anchor": float(vol_anchor),
                "pass_gate": bool(pass_gate),
                "decision_frame_mode": "full_frame_latest_sequence",
                "decision_frame_rows": int(len(frame)),
            }
        )
        if not pass_gate:
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "alpha3|v31_frozen_v27_rule_exit|deep_alpha_gate_fail",
                "position_signal": "HOLD",
                "position_reason": "|".join(guard_reasons) or "v31_deep_alpha_gate_fail",
                "score": float(edge),
                "conviction": float(margin),
                "owner": "",
                "regime": regime,
                "decision_logic": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                "model_version": "Alpha3",
                "model_id": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                "model_path": str(self.v31_v27_model_path),
                "model_sleeve": "deep_alpha",
                "scout_prob": float(edge),
                "scout_frac": 0.0,
                "scout_probability_threshold": float(edge_th),
                "scout_cost_pass": True,
                "sleeve_trace": trace,
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()

        router_cap = float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0))
        notional = float(np.clip(self._v31_cfg_float("notional", 1.0), 0.0, router_cap))
        if notional <= 1e-12:
            return None
        leverage = float(np.clip(max(notional, 1.0), 1.0, router_cap))
        fraction = float(np.clip(notional / max(leverage, 1e-8), 0.0, 1.0))
        action = int(FULLY_LEARNED_ACTION_LONG if side > 0 else FULLY_LEARNED_ACTION_SHORT)
        self.owner = "lifecycle_v1"
        self.owner_regime = regime
        self.peak_unrealized = 0.0
        self.active_lifecycle_v1_mae_unrealized = 0.0
        self.active_lifecycle_v1_base_notional = float(notional)
        self.active_lifecycle_v1_effective_notional = float(notional)
        self.active_lifecycle_v1_leverage = float(leverage)
        self.active_lifecycle_v1_cooldown_bars = int(self._v31_cfg_int("cooldown", 12))
        self.active_lifecycle_v1_quality_score = float(edge)
        self.active_lifecycle_v1_confidence = float(margin)
        self.active_lifecycle_v1_entry_bucket = "v31_deep_alpha"
        self.active_lifecycle_v1_entry_hazard = 0.0
        self.active_lifecycle_v1_entry_support = 0
        self.active_lifecycle_v1_edit = "v31_deep_alpha"
        self.active_lifecycle_v1_take_profit = float(self._v31_cfg_float("base_tp", 0.040))
        self.active_lifecycle_v1_stop_loss = float(self._v31_cfg_float("base_sl", 0.018))
        self.active_lifecycle_v1_max_hold_bars = int(self._v31_cfg_int("base_hold", 48))
        self.active_lifecycle_v1_jackpot_added = True
        self.active_lifecycle_v1_v21_sleeve = "deep_alpha"
        self.active_lifecycle_v1_v21_stop_raw = 999.0
        self.active_lifecycle_v1_v21_peak_raw = -1e9
        self.active_lifecycle_v1_v21_stop_reasons = ["v31_rule_exit_overlay"]
        self.active_lifecycle_v1_scout_model_version = "V31"
        self.active_lifecycle_v1_scout_model_id = "hf_v13_frozen_v27_rule_exit_overlay_v31_20260511"
        self.active_lifecycle_v1_scout_model_path = str(self.v31_v27_model_path)
        self.active_lifecycle_v1_scout_prob = float(edge)
        self.active_lifecycle_v1_scout_frac = float(notional)
        self.active_lifecycle_v1_scout_probability_threshold = float(edge_th)
        self.active_lifecycle_v1_scout_cost_pass = True
        self.active_v31_entry_edge = float(edge)
        self.active_v31_entry_margin = float(margin)
        self.active_v31_entry_vol_anchor = float(vol_anchor * notional)
        self.active_v31_entry_q_long = float(q_long_raw)
        self.active_v31_entry_q_short = float(q_short_raw)
        self.active_v31_entry_q_long_raw = float(q_long_raw)
        self.active_v31_entry_q_short_raw = float(q_short_raw)
        self.active_v31_entry_selected_side = "LONG" if side > 0 else "SHORT"
        self.active_v31_entry_guard_reason = "|".join(guard_reasons)
        self.active_lifecycle_v1_conformal_core_notional = 0.0
        self.active_lifecycle_v1_conformal_sleeve_notional = 0.0
        self.active_lifecycle_v1_conformal_sleeve_exit_bars = 0
        self.active_lifecycle_v1_conformal_sleeve_action = ""
        self._save_runtime_state()
        trace["v31"].update(
            {
                "notional": float(notional),
                "leverage": float(leverage),
                "entry_vol_anchor_raw": float(vol_anchor),
                "entry_vol_anchor": float(self.active_v31_entry_vol_anchor),
                "q_long": float(q_long_raw),
                "q_short": float(q_short_raw),
                "q_long_raw": float(q_long_raw),
                "q_short_raw": float(q_short_raw),
                "selected_side": "LONG" if side > 0 else "SHORT",
                "regime_long_guard_reasons": list(guard_reasons),
                "base_tp": float(self.active_lifecycle_v1_take_profit),
                "base_sl": float(self.active_lifecycle_v1_stop_loss),
                "base_hold": int(self.active_lifecycle_v1_max_hold_bars),
                "cooldown": int(self.active_lifecycle_v1_cooldown_bars),
            }
        )
        info = {
            "agent": "FINAL_GOVERNOR",
            "source": "alpha3|v31_frozen_v27_rule_exit|entry_deep_alpha",
            "position_signal": "LONG_ENTRY" if side > 0 else "SHORT_ENTRY",
            "position_reason": "v31_deep_alpha_entry_parent_cash",
            "score": float(edge),
            "conviction": float(margin),
            "owner": "lifecycle_v1",
            "regime": regime,
            "decision_logic": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
            "quality_score": float(edge),
            "confidence": float(margin),
            "model_version": "Alpha3",
            "model_id": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
            "model_path": str(self.v31_v27_model_path),
            "model_sleeve": "deep_alpha",
            "scout_prob": float(edge),
            "scout_frac": float(notional),
            "scout_probability_threshold": float(edge_th),
            "scout_cost_pass": True,
            "sleeve_trace": trace,
        }
        return action, float(notional), float(fraction), float(leverage), info, regime.upper()

    def _reset_v13_1_position_state(self) -> None:
        self.active_v13_1_take_profit = 0.0
        self.active_v13_1_stop_loss = 0.0
        self.active_v13_1_max_hold_bars = 0
        self.active_v13_1_cooldown_bars = 0
        self.active_v13_1_quality_score = 0.0
        self.active_v13_1_confidence = 0.0
        self.active_v13_1_notional = 0.0
        self.active_v13_1_leverage = 1.0
        self.active_v13_1_lane = ""
        self.active_v13_1_probability = 0.0
        self.active_v13_1_threshold = 0.0
        self.active_v13_1_regime = ""
        self.active_v13_1_regime_multiplier = 1.0

    def _v13_1_available(self) -> bool:
        return bool(self.disabled_v13_1_enabled and self.disabled_v13_1_adapter is not None)

    def _decide_v13_1_entry(
        self,
        frame: pd.DataFrame,
        *,
        meta_router,
        regime: str,
        raw_regime: str,
    ) -> tuple[int, float, float, float, dict, str] | None:
        adapter = self.disabled_v13_1_adapter
        if adapter is None:
            return None
        if self.v13_1_cooldown_left > 0:
            self.v13_1_cooldown_left -= 1
            self._save_runtime_state()
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "disabled_v13_1|cooldown",
                "position_signal": "HOLD",
                "position_reason": "disabled_v13_1_cooldown",
                "score": 0.0,
                "conviction": 0.0,
                "owner": "",
                "regime": regime,
                "decision_logic": "disabled_v13_1_model",
                "model_version": "Disabled V13.1",
                "model_id": str(adapter.model_id),
                "model_path": str(self.disabled_v13_1_model_path),
                "model_sleeve": "",
                "sleeve_trace": {
                    "decision_logic": "disabled_v13_1_model",
                    "cooldown_left": int(self.v13_1_cooldown_left),
                },
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()
        ctx = self._lifecycle_v1_daily_context(meta_router, frame)
        try:
            decision = adapter.decide(
                frame,
                account_drawdown=float(ctx.get("account_dd", 0.0) or 0.0),
                loss_cooldown_left=0,
                leverage_cap=float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0)),
            )
        except Exception as e:
            logger.warning("SYSTEM disabled_v13_1 signal failed closed: %s", e)
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "disabled_v13_1|signal_error",
                "position_signal": "HOLD",
                "position_reason": "disabled_v13_1_signal_error",
                "score": 0.0,
                "conviction": 0.0,
                "owner": "",
                "regime": regime,
                "decision_logic": "disabled_v13_1_model",
                "model_version": "Disabled V13.1",
                "model_id": str(adapter.model_id),
                "model_path": str(self.disabled_v13_1_model_path),
                "model_sleeve": "",
                "sleeve_trace": {"error": str(e), "raw_regime": raw_regime},
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()
        trace = dict(decision.trace)
        trace.update(
            {
                "raw_regime": raw_regime,
                "legacy_regime_removed": str(regime),
                "risk_context": dict(ctx),
            }
        )
        if int(decision.action) == 0 or int(decision.side) == 0 or float(decision.notional_exposure) <= 1e-12:
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": f"disabled_v13_1|{decision.reason}",
                "position_signal": "HOLD",
                "position_reason": str(decision.reason),
                "score": float(decision.probability),
                "conviction": float(decision.confidence),
                "owner": "",
                "regime": decision.regime,
                "decision_logic": "disabled_v13_1_model",
                "quality_score": float(decision.quality_score),
                "confidence": float(decision.confidence),
                "model_version": "Disabled V13.1",
                "model_id": str(adapter.model_id),
                "model_path": str(self.disabled_v13_1_model_path),
                "model_sleeve": str(decision.lane),
                "sleeve_trace": trace,
            }
            return 0, 0.0, 0.0, 1.0, info, decision.regime.upper()

        self.owner = "disabled_v13_1"
        self.owner_regime = str(decision.regime)
        self.peak_unrealized = 0.0
        self.active_v13_1_take_profit = float(decision.take_profit)
        self.active_v13_1_stop_loss = float(decision.stop_loss)
        self.active_v13_1_max_hold_bars = int(decision.max_hold_bars)
        self.active_v13_1_cooldown_bars = int(decision.cooldown_bars)
        self.active_v13_1_quality_score = float(decision.quality_score)
        self.active_v13_1_confidence = float(decision.confidence)
        self.active_v13_1_notional = float(decision.notional_exposure)
        self.active_v13_1_leverage = float(decision.leverage)
        self.active_v13_1_lane = str(decision.lane)
        self.active_v13_1_probability = float(decision.probability)
        self.active_v13_1_threshold = float(decision.threshold)
        self.active_v13_1_regime = str(decision.regime)
        self.active_v13_1_regime_multiplier = float(decision.regime_multiplier)
        self._save_runtime_state()
        info = {
            "agent": "FINAL_GOVERNOR",
            "source": f"disabled_v13_1|entry_{decision.lane}",
            "position_signal": "LONG_ENTRY" if int(decision.action) == 1 else "SHORT_ENTRY",
            "position_reason": f"disabled_v13_1_entry_{decision.lane}",
            "score": float(decision.probability),
            "conviction": float(decision.confidence),
            "owner": "disabled_v13_1",
            "regime": decision.regime,
            "decision_logic": "disabled_v13_1_model",
            "take_profit": float(decision.take_profit),
            "stop_loss": float(decision.stop_loss),
            "max_hold_bars": int(decision.max_hold_bars),
            "cooldown_bars": int(decision.cooldown_bars),
            "quality_score": float(decision.quality_score),
            "confidence": float(decision.confidence),
            "model_version": "Disabled V13.1",
            "model_id": str(adapter.model_id),
            "model_path": str(self.disabled_v13_1_model_path),
            "model_sleeve": str(decision.lane),
            "scout_prob": float(decision.probability) if decision.lane == "scout" else 0.0,
            "scout_frac": float(decision.notional_exposure),
            "scout_probability_threshold": float(decision.threshold) if decision.lane == "scout" else 0.0,
            "scout_cost_pass": True,
            "sleeve_trace": trace,
        }
        return (
            int(decision.action),
            float(decision.notional_exposure),
            float(decision.position_fraction),
            float(decision.leverage),
            info,
            decision.regime.upper(),
        )

    def _manage_v13_1_position(
        self,
        *,
        meta_router,
        current_price: float,
        regime: str,
        frame: pd.DataFrame,
    ) -> tuple[int, float, float, float, dict, str]:
        pos = str(meta_router.pos or "")
        action_hold = self._action_from_side(pos)
        exposure = float(meta_router.current_leverage or self.active_v13_1_notional or 0.0)
        exec_lev = float(meta_router.execution_leverage or self.active_v13_1_leverage or 1.0)
        fraction = float(meta_router.position_fraction or min(exposure / max(exec_lev, 1e-8), 1.0))
        mark_math = meta_router._trade_math(
            pos,
            float(meta_router.entry_price or 0.0),
            float(current_price or 0.0),
            exposure,
            entry_liquidity=str(getattr(meta_router, "entry_execution_liquidity", "") or ""),
        )
        gross_unrealized = float(mark_math.get("gross_return_frac", 0.0) or 0.0) * float(exposure)
        net_unrealized = float(meta_router._net_pnl_frac(current_price))
        self.peak_unrealized = max(float(self.peak_unrealized), gross_unrealized)
        hold_bars = int(meta_router.hold_count or 0)
        close = False
        reason = "disabled_v13_1_hold"
        if exposure <= 1e-12:
            close = True
            reason = "disabled_v13_1_reconcile_close"
        elif self.active_v13_1_take_profit > 0.0 and gross_unrealized >= float(self.active_v13_1_take_profit):
            close = True
            reason = "learned_take_profit"
        elif self.active_v13_1_stop_loss > 0.0 and gross_unrealized <= -abs(float(self.active_v13_1_stop_loss)):
            close = True
            reason = "learned_stop_loss"
        elif self.active_v13_1_max_hold_bars > 0 and hold_bars >= int(self.active_v13_1_max_hold_bars):
            close = True
            reason = "learned_max_hold"

        trace = {
            "decision_logic": "disabled_v13_1_model",
            "model_version": "Disabled V13.1",
            "model_id": "disabled_v13_1_model",
            "model": str(self.disabled_v13_1_model_path),
            "report": str(self.disabled_v13_1_report_path),
            "lane": str(self.active_v13_1_lane),
            "entry_regime": str(self.active_v13_1_regime),
            "current_regime": str(regime),
            "regime_multiplier": float(self.active_v13_1_regime_multiplier),
            "take_profit": float(self.active_v13_1_take_profit),
            "stop_loss": float(self.active_v13_1_stop_loss),
            "max_hold_bars": int(self.active_v13_1_max_hold_bars),
            "age_bars": int(hold_bars),
            "gross_mark_unrealized": float(gross_unrealized),
            "net_unrealized": float(net_unrealized),
            "unrealized_basis": "gross_mark_backtest_parity",
            "peak_unrealized": float(self.peak_unrealized),
            "probability": float(self.active_v13_1_probability),
            "threshold": float(self.active_v13_1_threshold),
            "quality_score": float(self.active_v13_1_quality_score),
            "confidence": float(self.active_v13_1_confidence),
            "regime_predictor": dict(frame.attrs.get("regime_predictor_trace", {}) or {}),
        }
        if close:
            self.last_exit_bar = self.bar_counter
            self.v13_1_cooldown_left = int(max(0, self.active_v13_1_cooldown_bars))
            trace["cooldown_armed_bars"] = int(self.v13_1_cooldown_left)
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": f"disabled_v13_1|{reason}",
                "position_signal": "EXIT",
                "position_reason": reason,
                "score": float(abs(gross_unrealized)),
                "conviction": float(self.active_v13_1_confidence),
                "owner": "disabled_v13_1",
                "regime": regime,
                "decision_logic": "disabled_v13_1_model",
                "quality_score": float(self.active_v13_1_quality_score),
                "confidence": float(self.active_v13_1_confidence),
                "model_version": "Disabled V13.1",
                "model_id": "disabled_v13_1_model",
                "model_path": str(self.disabled_v13_1_model_path),
                "model_sleeve": str(self.active_v13_1_lane),
                "scout_prob": float(self.active_v13_1_probability) if self.active_v13_1_lane == "scout" else 0.0,
                "scout_frac": float(self.active_v13_1_notional),
                "scout_probability_threshold": float(self.active_v13_1_threshold) if self.active_v13_1_lane == "scout" else 0.0,
                "scout_cost_pass": True,
                "sleeve_trace": trace,
            }
            self.owner = ""
            self.owner_regime = ""
            self.peak_unrealized = 0.0
            self._reset_v13_1_position_state()
            self._save_runtime_state()
            return 0, 0.0, 0.0, 1.0, info, regime.upper()

        info = {
            "agent": "FINAL_GOVERNOR",
            "source": "disabled_v13_1|hold",
            "position_signal": "HOLD",
            "position_reason": reason,
            "score": float(abs(gross_unrealized)),
            "conviction": float(self.active_v13_1_confidence),
            "owner": "disabled_v13_1",
            "regime": regime,
            "decision_logic": "disabled_v13_1_model",
            "quality_score": float(self.active_v13_1_quality_score),
            "confidence": float(self.active_v13_1_confidence),
            "model_version": "Disabled V13.1",
            "model_id": "disabled_v13_1_model",
            "model_path": str(self.disabled_v13_1_model_path),
            "model_sleeve": str(self.active_v13_1_lane),
            "scout_prob": float(self.active_v13_1_probability) if self.active_v13_1_lane == "scout" else 0.0,
            "scout_frac": float(self.active_v13_1_notional),
            "scout_probability_threshold": float(self.active_v13_1_threshold) if self.active_v13_1_lane == "scout" else 0.0,
            "scout_cost_pass": True,
            "sleeve_trace": trace,
        }
        return action_hold, exposure, fraction, exec_lev, info, regime.upper()

    def _manage_lifecycle_v1_position(self, *, meta_router, current_price: float, regime: str, frame: pd.DataFrame) -> tuple[int, float, float, float, dict, str]:
        pos = str(meta_router.pos or "")
        action_hold = self._action_from_side(pos)
        side = 1 if pos == "LONG" else -1
        exposure = float(meta_router.current_leverage or self.active_lifecycle_v1_effective_notional or 0.0)
        exec_lev = float(meta_router.execution_leverage or self.active_lifecycle_v1_leverage or 1.0)
        fraction = float(meta_router.position_fraction or min(exposure / max(exec_lev, 1e-8), 1.0))
        net_unrealized = float(meta_router._net_pnl_frac(current_price))
        mark_math = meta_router._trade_math(
            pos,
            float(meta_router.entry_price or 0.0),
            float(current_price or 0.0),
            exposure,
            entry_liquidity=str(getattr(meta_router, "entry_execution_liquidity", "") or ""),
        )
        gross_mark_unrealized = float(mark_math.get("gross_return_frac", 0.0) or 0.0) * float(exposure)
        if bool(FINAL_GOVERNOR_ALPHA3_CSV_MARK_PARITY_ENABLE) and str(self.active_lifecycle_v1_scout_model_version or "") in {"V21.2", "V31"}:
            entry_px = float(meta_router.entry_price or 0.0)
            mark_px = float(current_price or 0.0)
            slip = float(getattr(meta_router, "trade_slip", 0.0) or 0.0)
            if entry_px > 0.0 and mark_px > 0.0:
                if pos == "LONG":
                    gross_raw = (mark_px * (1.0 - slip) - entry_px) / max(entry_px, 1e-12)
                else:
                    gross_raw = (entry_px - mark_px * (1.0 + slip)) / max(entry_px, 1e-12)
                gross_mark_unrealized = float(gross_raw * float(exposure))
        unrealized = float(gross_mark_unrealized if bool(self.conformal_veto_v1_5_enabled) else net_unrealized)
        if bool(FINAL_GOVERNOR_ALPHA3_CSV_MARK_PARITY_ENABLE) and str(self.active_lifecycle_v1_scout_model_version or "") in {"V21.2", "V31"}:
            unrealized = float(gross_mark_unrealized)
        self.peak_unrealized = max(float(self.peak_unrealized), unrealized)
        self.active_lifecycle_v1_mae_unrealized = min(float(self.active_lifecycle_v1_mae_unrealized), float(unrealized))
        raw_unrealized = _price_return_frac(pos, float(meta_router.entry_price or 0.0), float(current_price or 0.0))
        if self.active_lifecycle_v1_v21_peak_raw <= -1e8:
            self.active_lifecycle_v1_v21_peak_raw = float(raw_unrealized)
        else:
            self.active_lifecycle_v1_v21_peak_raw = max(float(self.active_lifecycle_v1_v21_peak_raw), float(raw_unrealized))
        hold_bars = int(meta_router.hold_count or 0)
        entry_quality = float(self.active_lifecycle_v1_quality_score or 0.0)
        entry_confidence = float(self.active_lifecycle_v1_confidence or 0.0)
        cfg = dict(self.lifecycle_v1_cfg or {})
        exit_cfg = dict(self.lifecycle_v1_exit_cfg or {})
        base_threshold = float(exit_cfg.get("exit_threshold", 0.45))
        threshold = base_threshold
        base_min_age = max(1, int(exit_cfg.get("min_exit_age", 6)))
        min_age = base_min_age
        delta = 0.0
        global_rate = 0.0
        if exposure <= 1e-12:
            self.last_exit_bar = self.bar_counter
            reason = "lifecycle_v1_reconcile_close"
            close = True
            p_exit = 1.0
            bucket = ""
            hazard = 0.0
            support = 0
        elif str(self.active_lifecycle_v1_scout_model_version or "") == "V31":
            self.peak_unrealized = max(float(self.peak_unrealized), float(gross_mark_unrealized))
            p_exit = 0.0
            threshold = 1.0
            bucket = "v31_deep_alpha"
            hazard = 0.0
            support = 0
            close = False
            reason = "v31_deep_alpha_hold"
            edge_th = float(self._v31_cfg_float("edge_th", 0.010))
            base_tp = float(self._v31_cfg_float("base_tp", self.active_lifecycle_v1_take_profit or 0.040))
            base_sl = float(self._v31_cfg_float("base_sl", self.active_lifecycle_v1_stop_loss or 0.018))
            effective_tp = float(base_tp)
            effective_sl = float(base_sl)
            tp_util_mult = float(self._v31_cfg_float("tp_util_mult", 1.5))
            sl_vol_mult = float(self._v31_cfg_float("sl_vol_mult", 2.5))
            entry_edge = float(self.active_v31_entry_edge or 0.0)
            entry_vol_anchor = float(self.active_v31_entry_vol_anchor or 0.0)
            if entry_vol_anchor <= 0.0 and _v31_live is not None and len(frame) > 0:
                entry_vol_anchor = float(_v31_live._vol_anchor(frame.iloc[-1]) * max(exposure, 1e-8))
            if tp_util_mult > 0.0:
                util_gain = 1.0 + tp_util_mult * max(entry_edge - edge_th, 0.0) / max(0.02, edge_th)
                effective_tp = float(np.clip(base_tp * util_gain, base_tp * 0.8, self._v31_cfg_float("tp_cap", 0.075)))
            if sl_vol_mult > 0.0:
                effective_sl = float(
                    np.clip(
                        entry_vol_anchor * sl_vol_mult,
                        base_sl * 0.6,
                        self._v31_cfg_float("sl_cap", 0.036),
                    )
                )
            trail_gap_mult = float(self._v31_cfg_float("trail_gap_mult", 1.0))
            trail_activation = float(self._v31_cfg_float("trail_activation", FINAL_GOVERNOR_V31_TRAIL_ACTIVATION))
            min_trail_sl = 0.001
            if float(self.peak_unrealized) > 0.0 and float(self.peak_unrealized) >= trail_activation and trail_gap_mult > 0.0:
                trail_gap = float(entry_vol_anchor * trail_gap_mult)
                hold_decay_start = int(self._v31_cfg_int("hold_decay_start", 18))
                if hold_decay_start < 999 and hold_bars >= hold_decay_start:
                    decay_bars = int(hold_bars - hold_decay_start)
                    trail_gap = max(
                        float(entry_vol_anchor * 0.35),
                        float(trail_gap - self._v31_cfg_float("hold_decay_rate", 0.025) * decay_bars * entry_vol_anchor),
                    )
                trail_stop = max(-float(effective_sl), float(self.peak_unrealized) - float(trail_gap))
                effective_sl = min(float(effective_sl), max(0.001, float(trail_stop)))
            if effective_tp > 0.0 and gross_mark_unrealized >= effective_tp:
                close = True
                reason = "v31_deep_alpha_take_profit"
            elif effective_sl > 0.0 and gross_mark_unrealized <= -abs(effective_sl):
                close = True
                reason = "v31_deep_alpha_stop_loss"
            elif self._v31_cfg_int("base_hold", self.active_lifecycle_v1_max_hold_bars or 48) > 0 and hold_bars >= int(
                self._v31_cfg_int("base_hold", self.active_lifecycle_v1_max_hold_bars or 48)
            ):
                close = True
                reason = "v31_deep_alpha_max_hold"
        elif self._v21_2_jackpot_available() and str(self.active_lifecycle_v1_scout_model_version or "") in {"", "V21.2"}:
            if not str(self.active_lifecycle_v1_scout_model_version or ""):
                self.active_lifecycle_v1_scout_model_version = "V21.2"
                self.active_lifecycle_v1_scout_model_id = "hf_v13_jackpot_runner_v21_2_20260511"
                self.active_lifecycle_v1_scout_model_path = str(self.v21_2_jackpot_model_path)
            p_exit = 0.0
            threshold = 1.0
            bucket = "v21_2_parent"
            hazard = 0.0
            support = 0
            close = False
            reason = "v21_2_jackpot_hold"
            if self.active_lifecycle_v1_take_profit > 0.0 and gross_mark_unrealized >= float(self.active_lifecycle_v1_take_profit):
                close = True
                reason = "learned_take_profit"
            elif self.active_lifecycle_v1_stop_loss > 0.0 and gross_mark_unrealized <= -abs(float(self.active_lifecycle_v1_stop_loss)):
                close = True
                reason = "learned_stop_loss"
            elif self.active_lifecycle_v1_max_hold_bars > 0 and hold_bars >= int(self.active_lifecycle_v1_max_hold_bars):
                close = True
                reason = "learned_max_hold"
        else:
            vec_dec = self._lifecycle_v1_feature_vec(
                frame,
                side=side,
                age=hold_bars,
                unrealized=unrealized,
                peak_unrealized=float(self.peak_unrealized),
                notional=exposure,
                leverage=exec_lev,
                entry_quality=entry_quality,
                entry_confidence=entry_confidence,
            )
            if vec_dec is None:
                p_exit = 1.0
                threshold = 0.0
                bucket = ""
                hazard = 0.0
                support = 0
                close = True
                reason = "lifecycle_v1_model_unavailable_close"
            else:
                vec, _ = vec_dec
                p_exit = float(_lifecycle_exit_probability_vec(self.lifecycle_v1_exit_model, vec))
                bucket, hazard, support = self._lifecycle_v1_hazard_info(vec)
                global_rate = float(dict(self.lifecycle_v1_recalibrator or {}).get("global_hazard_rate", 0.0) or 0.0)
                delta = float(np.clip((global_rate - hazard) * float(cfg.get("delta_scale", 1.0)), -float(cfg.get("max_delta", 0.12)), float(cfg.get("max_delta", 0.12))))
                threshold = float(np.clip(base_threshold + float(cfg.get("threshold_shift", 0.0)) + delta, 0.05, 0.95))
                min_age = max(1, base_min_age + int(cfg.get("min_age_delta", 0)))
                hard_stop = abs(float(FINAL_GOVERNOR_LIFECYCLE_V1_EXIT_HARD_STOP))
                v21_stop_raw = abs(float(self.active_lifecycle_v1_v21_stop_raw or 999.0))
                v21_stop_reasons = list(self.active_lifecycle_v1_v21_stop_reasons or [])
                active_stop_cfg = (
                    dict(self.v22_1_adapter.stop_config)
                    if self._lifecycle_v22_1_available() and self.v22_1_adapter is not None
                    else dict(self.v21_stop_config or {})
                )
                v21_min_hold = int(max(1, float(active_stop_cfg.get("min_hold_bars", 1) or 1)))
                close = False
                reason = "lifecycle_v1_hold"
                if (
                    (self._lifecycle_v22_1_available() or self._lifecycle_v21_available())
                    and self.active_lifecycle_v1_v21_sleeve
                    and v21_stop_raw < 100.0
                    and hold_bars >= v21_min_hold
                    and raw_unrealized <= -v21_stop_raw
                ):
                    close = True
                    reason = f"lifecycle_v21_{self.active_lifecycle_v1_v21_sleeve}_adaptive_stop"
                    v21_stop_reasons.append("stop_triggered")
                    self.active_lifecycle_v1_v21_stop_reasons = v21_stop_reasons
                elif (
                    hard_stop > 0.0
                    and unrealized <= -hard_stop
                    and not (self._lifecycle_v21_pure_active() and self.v21_disable_legacy_hard_stop)
                ):
                    close = True
                    reason = "lifecycle_v1_hard_stop"
                elif hold_bars >= base_min_age and p_exit >= base_threshold:
                    close = True
                    reason = "lifecycle_v1_base_exit_governor"
                elif hold_bars >= min_age and p_exit >= threshold:
                    close = True
                    reason = "lifecycle_v1_exit_governor"

        lifecycle_decision_logic = FINAL_GOVERNOR_ALPHA3_MODEL_ID
        trace = {
            "decision_logic": lifecycle_decision_logic,
            "model": os.path.basename(str(self.lifecycle_v1_model_path)),
            "base_policy": os.path.basename(str(self.lifecycle_v1_policy_path)),
            "exit_model": os.path.basename(str(self.lifecycle_v1_exit_model_path)),
            "p_exit": float(p_exit),
            "threshold": float(threshold),
            "base_threshold": float(base_threshold),
            "adjusted_threshold": float(threshold),
            "base_min_age": int(base_min_age),
            "min_age": int(min_age),
            "hazard_delta": float(delta),
            "global_hazard_rate": float(global_rate),
            "bucket": str(bucket),
            "hazard": float(hazard),
            "support": int(support),
            "age_bars": int(hold_bars),
            "peak_unrealized": float(self.peak_unrealized),
            "mae_unrealized": float(self.active_lifecycle_v1_mae_unrealized),
            "net_unrealized": float(net_unrealized),
            "gross_mark_unrealized": float(gross_mark_unrealized),
            "unrealized_basis": (
                "gross_mark_backtest_parity"
                if bool(self.conformal_veto_v1_5_enabled)
                or (
                    bool(FINAL_GOVERNOR_ALPHA3_CSV_MARK_PARITY_ENABLE)
                    and str(self.active_lifecycle_v1_scout_model_version or "") in {"V21.2", "V31"}
                )
                else "net_live_cost_inclusive"
            ),
            "raw_unrealized": float(raw_unrealized),
            "take_profit": float(self.active_lifecycle_v1_take_profit),
            "stop_loss": float(self.active_lifecycle_v1_stop_loss),
            "max_hold_bars": int(self.active_lifecycle_v1_max_hold_bars),
            "regime": regime,
            "regime_predictor": dict(frame.attrs.get("regime_predictor_trace", {}) or {}),
            "alpha3_live_contract": FINAL_GOVERNOR_ALPHA3_LIVE_CONTRACT_ID,
            "execution_contract": "signal_close_next_open_limit_touch0_fee20_live_router",
            "mark_contract": "csv_exit_side_slippage_mark" if bool(FINAL_GOVERNOR_ALPHA3_CSV_MARK_PARITY_ENABLE) else "router_net_mark",
            "cooldown_contract": "csv_parent_and_deep_cooldown" if bool(FINAL_GOVERNOR_ALPHA3_CSV_COOLDOWN_PARITY_ENABLE) else "native_deep_cooldown_only_for_v31",
            "v21": {
                "enabled": bool(self._lifecycle_v22_1_available() or self._lifecycle_v21_available()),
                "pure_mode": bool(self._lifecycle_v21_pure_active()),
                "model_id": str(self.v22_1_adapter.model_id if self._lifecycle_v22_1_available() and self.v22_1_adapter is not None else self.v21_model_id),
                "model_version": "Alpha3",
                "adapter_version": "v22_1_learned_scout" if self._lifecycle_v22_1_available() else str(self.v21_adapter_version),
                "model": str(self.v22_1_model_path if self._lifecycle_v22_1_available() else self.v21_model_path),
                "sleeve": str(self.active_lifecycle_v1_v21_sleeve),
                "stop_raw": float(self.active_lifecycle_v1_v21_stop_raw),
                "peak_raw": float(self.active_lifecycle_v1_v21_peak_raw),
                "stop_reasons": list(self.active_lifecycle_v1_v21_stop_reasons or []),
                "legacy_hard_stop_disabled": bool(self._lifecycle_v21_pure_active() and self.v21_disable_legacy_hard_stop),
            },
            "conformal_veto_v1_5": {
                "enabled": bool(self.conformal_veto_v1_5_enabled),
                "model_id": "clean_base_causal_sleeve_conformal_veto_v1_5",
                "model": str(self.conformal_veto_v1_5_model_path),
                "report": str(self.conformal_veto_v1_5_report_path),
                "active_action": str(self.active_lifecycle_v1_conformal_sleeve_action),
                "core_notional": float(self.active_lifecycle_v1_conformal_core_notional),
                "sleeve_notional": float(self.active_lifecycle_v1_conformal_sleeve_notional),
                "sleeve_exit_bars": int(self.active_lifecycle_v1_conformal_sleeve_exit_bars),
            },
        }
        if str(self.active_lifecycle_v1_scout_model_version or "") == "V31":
            trace["v31"] = {
                "enabled": bool(self.v31_enabled),
                "report": str(self.v31_report_path),
                "audit": str(self.v31_audit_path),
                "v27_model": str(self.v31_v27_model_path),
                "selected_config": dict(self.v31_cfg or {}),
                "q_long": float(self.active_v31_entry_q_long),
                "q_short": float(self.active_v31_entry_q_short),
                "q_long_raw": float(self.active_v31_entry_q_long_raw),
                "q_short_raw": float(self.active_v31_entry_q_short_raw),
                "selected_side": str(self.active_v31_entry_selected_side),
                "entry_edge": float(self.active_v31_entry_edge),
                "entry_margin": float(self.active_v31_entry_margin),
                "entry_vol_anchor": float(self.active_v31_entry_vol_anchor),
                "effective_tp": float(locals().get("effective_tp", self.active_lifecycle_v1_take_profit)),
                "effective_sl": float(locals().get("effective_sl", self.active_lifecycle_v1_stop_loss)),
                "trail_activation": float(locals().get("trail_activation", FINAL_GOVERNOR_V31_TRAIL_ACTIVATION)),
                "min_trail_sl": float(locals().get("min_trail_sl", self.active_lifecycle_v1_stop_loss * FINAL_GOVERNOR_V31_TRAIL_MIN_SL_MULT)),
                "guard_reason": str(self.active_v31_entry_guard_reason),
                "exit_contract": "deep_alpha_rule_exit_overlay_only",
                "parent_and_jackpot_contract": "parent_positions_unchanged",
                "alpha2_1_contract": "v31_deep_alpha_under_alpha2_1",
                "decision_frame_mode": "active_position_entry_snapshot",
                "sequence_contract": "v27_seq_at_frame_latest_on_entry",
            }
        if lifecycle_decision_logic.startswith("ddh2_"):
            trace["sub_decision_logic"] = "deep_state_safe_cap_reallocator_v22_1_scout_param_grid"
            trace["ddh2_ensemble"] = {
                "enabled": True,
                "report": str(self.ddh2_report_path),
                "audit": str(self.ddh2_audit_path),
                "source_layer": "v22_1_sniper",
            }
        if (
            not close
            and self._v21_2_jackpot_available()
            and str(self.active_lifecycle_v1_scout_model_version or "") == "V21.2"
            and not bool(self.active_lifecycle_v1_jackpot_added)
        ):
            latest = self._lifecycle_v1_latest(frame)
            dec_for_runner = latest[0] if latest is not None else pd.Series(
                {
                    "side": side,
                    "confidence": self.active_lifecycle_v1_confidence,
                    "quality_score": self.active_lifecycle_v1_quality_score,
                }
            )
            alpha2_dec_for_runner = self._alpha2_1_constrained_decision_row(dec_for_runner, frame)
            if alpha2_dec_for_runner is not None:
                dec_for_runner = alpha2_dec_for_runner
            router_cap = float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0))
            parent_notional = float(self.active_lifecycle_v1_base_notional or exposure)
            jackpot_meta = self.v21_2_jackpot_adapter.add_on_decision(
                frame,
                dec_for_runner,
                side=side,
                parent_notional=parent_notional,
                current_notional=exposure,
                bars_since_entry=hold_bars,
                unrealized=gross_mark_unrealized,
                mfe=float(self.peak_unrealized),
                mae=float(self.active_lifecycle_v1_mae_unrealized),
                drawdown_abs=float(self._lifecycle_v1_daily_context(meta_router, frame).get("account_dd", 0.0) or 0.0),
                take_profit=float(self.active_lifecycle_v1_take_profit),
                stop_loss=float(self.active_lifecycle_v1_stop_loss),
                max_hold=int(self.active_lifecycle_v1_max_hold_bars),
                router_cap=router_cap,
                parent_bundle=self.v21_2_parent_bundle,
            )
            trace["v21_2_jackpot_add_on"] = dict(jackpot_meta)
            if str(jackpot_meta.get("reason", "")) == "v21_2_jackpot_reject":
                self.active_lifecycle_v1_jackpot_added = True
                self._save_runtime_state()
            if jackpot_meta.get("applied"):
                target_exposure = float(jackpot_meta.get("output_notional", exposure) or exposure)
                target_exposure = float(np.clip(target_exposure, 0.0, router_cap))
                target_leverage = float(np.clip(max(exec_lev, target_exposure, 1.0), 1.0, router_cap))
                target_fraction = float(np.clip(target_exposure / max(target_leverage, 1e-8), 0.0, 1.0))
                self.active_lifecycle_v1_jackpot_added = True
                self.active_lifecycle_v1_effective_notional = float(target_exposure)
                self.active_lifecycle_v1_leverage = float(target_leverage)
                self.active_lifecycle_v1_v21_sleeve = "jackpot"
                self.active_lifecycle_v1_scout_prob = float(jackpot_meta.get("p_jackpot", 0.0) or 0.0)
                self.active_lifecycle_v1_scout_frac = float(jackpot_meta.get("delta_notional", 0.0) or 0.0)
                self.active_lifecycle_v1_scout_probability_threshold = float(
                    self.v21_2_jackpot_adapter.selected_config.get("jackpot_p", 0.20) or 0.20
                )
                self.active_lifecycle_v1_scout_cost_pass = True
                self._save_runtime_state()
                info = {
                    "agent": "FINAL_GOVERNOR",
                    "source": "alpha3|v21_2_jackpot_runner|add_on_resize",
                    "position_signal": "HOLD",
                    "position_reason": "v21_2_jackpot_add_on_resize",
                    "score": float(jackpot_meta.get("p_jackpot", 0.0) or 0.0),
                    "conviction": float(jackpot_meta.get("q90", 0.0) or 0.0),
                    "owner": "lifecycle_v1",
                    "regime": regime,
                    "decision_logic": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                    "quality_score": entry_quality,
                    "confidence": entry_confidence,
                    "model_version": "Alpha3",
                    "model_id": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                    "model_path": str(self.v21_2_jackpot_model_path),
                    "model_sleeve": "jackpot",
                    "scout_prob": float(jackpot_meta.get("p_jackpot", 0.0) or 0.0),
                    "scout_frac": float(jackpot_meta.get("delta_notional", 0.0) or 0.0),
                    "scout_probability_threshold": float(self.active_lifecycle_v1_scout_probability_threshold),
                    "scout_cost_pass": True,
                    "sleeve_trace": trace,
                }
                return action_hold, target_exposure, target_fraction, target_leverage, info, regime.upper()
        conformal_core = float(self.active_lifecycle_v1_conformal_core_notional or 0.0)
        conformal_sleeve = float(self.active_lifecycle_v1_conformal_sleeve_notional or 0.0)
        conformal_exit_bars = int(self.active_lifecycle_v1_conformal_sleeve_exit_bars or 0)
        if conformal_sleeve > 1e-12 and conformal_core > 1e-12 and exposure <= conformal_core + 1e-9:
            self.active_lifecycle_v1_conformal_sleeve_notional = 0.0
            self.active_lifecycle_v1_conformal_sleeve_exit_bars = 0
            self.active_lifecycle_v1_conformal_sleeve_action = "SLEEVE_CLOSED"
            self._save_runtime_state()
        elif (
            not close
            and conformal_sleeve > 1e-12
            and conformal_core > 1e-12
            and conformal_exit_bars > 0
            and hold_bars >= conformal_exit_bars
            and exposure > conformal_core + 1e-9
        ):
            target_exposure = float(min(conformal_core, exposure))
            target_fraction = float(np.clip(target_exposure / max(exec_lev, 1e-8), 0.0, 1.0))
            trace["conformal_veto_v1_5"] = {
                **dict(trace.get("conformal_veto_v1_5", {}) or {}),
                "sleeve_exit_due": True,
                "target_exposure": float(target_exposure),
                "target_fraction": float(target_fraction),
            }
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "conformal_veto_v1_5|sleeve_exit_resize",
                "position_signal": "HOLD",
                "position_reason": "conformal_veto_v1_5_sleeve_exit_resize",
                "score": float(p_exit),
                "conviction": float(p_exit),
                "owner": "lifecycle_v1",
                "regime": regime,
                "decision_logic": lifecycle_decision_logic,
                "quality_score": entry_quality,
                "confidence": entry_confidence,
                "model_version": "V1.5",
                "model_id": "clean_base_causal_sleeve_conformal_veto_v1_5",
                "model_path": str(self.conformal_veto_v1_5_model_path),
                "model_sleeve": str(self.active_lifecycle_v1_conformal_sleeve_action),
                "sleeve_trace": trace,
            }
            return action_hold, target_exposure, target_fraction, exec_lev, info, regime.upper()
        if close:
            self.last_exit_bar = self.bar_counter
            close_cooldown = int(max(0, self.active_lifecycle_v1_cooldown_bars))
            if unrealized < 0.0:
                close_cooldown = max(close_cooldown, int(self.lifecycle_v1_risk_cfg.get("loss_cooldown_bars", 0) or 0))
            if str(self.active_lifecycle_v1_scout_model_version or "") == "V31":
                v31_close_cooldown = int(max(0, self._v31_cfg_int("cooldown", close_cooldown)))
                if "stop_loss" in str(reason):
                    v31_close_cooldown = int(
                        max(v31_close_cooldown, self._v31_cfg_int("deep_stop_cooldown_extra", 0))
                    )
                if not bool(FINAL_GOVERNOR_ALPHA3_CSV_COOLDOWN_PARITY_ENABLE):
                    close_cooldown = 0
                self.v31_deep_cooldown_left = int(v31_close_cooldown)
                trace["v31_deep_cooldown_armed_bars"] = int(self.v31_deep_cooldown_left)
                trace["v31_deep_stop_cooldown_extra_bars"] = int(
                    self._v31_cfg_int("deep_stop_cooldown_extra", 0) if "stop_loss" in str(reason) else 0
                )
                trace["alpha3_csv_cooldown_parent_cooldown_preserved"] = bool(FINAL_GOVERNOR_ALPHA3_CSV_COOLDOWN_PARITY_ENABLE)
            elif bool(FINAL_GOVERNOR_ALPHA3_CSV_COOLDOWN_PARITY_ENABLE) and self._v31_available():
                deep_close_cooldown = int(max(0, self._v31_cfg_int("cooldown", 0)))
                if deep_close_cooldown > 0:
                    self.v31_deep_cooldown_left = int(max(int(self.v31_deep_cooldown_left), deep_close_cooldown))
                    trace["alpha3_csv_cooldown_deep_cooldown_armed_bars"] = int(self.v31_deep_cooldown_left)
            if self._lifecycle_v21_pure_active() and self.v21_bypass_cooldown:
                trace["v21_close_cooldown_bypassed_bars"] = int(close_cooldown)
                close_cooldown = 0
            self.lifecycle_v1_cooldown_left = close_cooldown
            trace["cooldown_armed_bars"] = int(close_cooldown)
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": f"alpha3|lifecycle_v1|{reason}",
                "position_signal": "EXIT",
                "position_reason": reason,
                "score": float(p_exit),
                "conviction": float(p_exit),
                "owner": "lifecycle_v1",
                "regime": regime,
                "decision_logic": lifecycle_decision_logic,
                "quality_score": entry_quality,
                "confidence": entry_confidence,
                "model_version": "Alpha3",
                "model_id": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                "model_path": str(self.conformal_veto_v1_5_model_path) if bool(self.conformal_veto_v1_5_enabled) else str(self.active_lifecycle_v1_scout_model_path or self.v21_model_path),
                "model_sleeve": str(self.active_lifecycle_v1_conformal_sleeve_action if bool(self.conformal_veto_v1_5_enabled) else self.active_lifecycle_v1_v21_sleeve),
                "scout_prob": float(self.active_lifecycle_v1_scout_prob),
                "scout_frac": float(self.active_lifecycle_v1_scout_frac),
                "scout_probability_threshold": float(self.active_lifecycle_v1_scout_probability_threshold),
                "scout_cost_pass": bool(self.active_lifecycle_v1_scout_cost_pass),
                "sleeve_trace": trace,
            }
            self.owner = ""
            self.owner_regime = ""
            self.peak_unrealized = 0.0
            self._reset_lifecycle_v1_position_state()
            self._save_runtime_state()
            return 0, 0.0, 0.0, 1.0, info, regime.upper()

        info = {
            "agent": "FINAL_GOVERNOR",
            "source": "alpha3|lifecycle_v1|hold",
            "position_signal": "HOLD",
            "position_reason": reason,
            "score": float(p_exit),
            "conviction": float(p_exit),
            "owner": "lifecycle_v1",
            "regime": regime,
            "decision_logic": lifecycle_decision_logic,
            "quality_score": entry_quality,
            "confidence": entry_confidence,
            "model_version": "Alpha3",
            "model_id": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
            "model_path": str(self.conformal_veto_v1_5_model_path) if bool(self.conformal_veto_v1_5_enabled) else str(self.active_lifecycle_v1_scout_model_path or self.v21_model_path),
            "model_sleeve": str(self.active_lifecycle_v1_conformal_sleeve_action if bool(self.conformal_veto_v1_5_enabled) else self.active_lifecycle_v1_v21_sleeve),
            "scout_prob": float(self.active_lifecycle_v1_scout_prob),
            "scout_frac": float(self.active_lifecycle_v1_scout_frac),
            "scout_probability_threshold": float(self.active_lifecycle_v1_scout_probability_threshold),
            "scout_cost_pass": bool(self.active_lifecycle_v1_scout_cost_pass),
            "sleeve_trace": trace,
        }
        return action_hold, exposure, fraction, exec_lev, info, regime.upper()

    def _decide_lifecycle_v1_entry(self, frame: pd.DataFrame, *, meta_router, regime: str, raw_regime: str) -> tuple[int, float, float, float, dict, str] | None:
        latest = self._lifecycle_v1_latest(frame)
        if latest is None:
            return None
        dec, _base_values = latest
        side = int(dec.side)
        action = 1 if side > 0 else (2 if side < 0 else 0)
        entry_decision_logic = (
            "clean_base_causal_sleeve_conformal_veto_v1_5"
            if bool(self.conformal_veto_v1_5_enabled)
            else "clean_base_lifecycle_v1"
        )
        trace = {
            "decision_logic": entry_decision_logic,
            "model": os.path.basename(str(self.lifecycle_v1_model_path)),
            "base_policy": os.path.basename(str(self.lifecycle_v1_policy_path)),
            "exit_model": os.path.basename(str(self.lifecycle_v1_exit_model_path)),
            "regime": regime,
            "raw_regime": raw_regime,
            "regime_predictor": dict(frame.attrs.get("regime_predictor_trace", {}) or {}),
            "base_action": int(dec.action),
            "base_side": int(side),
            "base_notional": float(dec.notional_exposure),
            "base_leverage": float(dec.leverage),
            "base_position_fraction": float(dec.position_fraction),
            "base_cooldown_bars": int(dec.cooldown_bars),
            "quality_score": float(dec.quality_score),
            "confidence": float(dec.confidence),
            "parent_decision_frame_mode": "full_frame_latest",
            "parent_decision_frame_rows": int(len(frame)),
            "alpha3_live_contract": FINAL_GOVERNOR_ALPHA3_LIVE_CONTRACT_ID,
            "execution_contract": "signal_close_next_open_limit_touch0_fee20_live_router",
            "mark_contract": "csv_exit_side_slippage_mark" if bool(FINAL_GOVERNOR_ALPHA3_CSV_MARK_PARITY_ENABLE) else "router_net_mark",
            "cooldown_contract": "csv_parent_and_deep_cooldown" if bool(FINAL_GOVERNOR_ALPHA3_CSV_COOLDOWN_PARITY_ENABLE) else "native_deep_cooldown_only_for_v31",
        }
        if self.lifecycle_v1_cooldown_left > 0:
            if self._lifecycle_v21_pure_active() and self.v21_bypass_cooldown:
                trace["v21_cooldown_bypassed_bars"] = int(self.lifecycle_v1_cooldown_left)
                self.lifecycle_v1_cooldown_left = 0
                self._save_runtime_state()
            else:
                self.lifecycle_v1_cooldown_left -= 1
                self._save_runtime_state()
                info = {
                    "agent": "FINAL_GOVERNOR",
                    "source": "lifecycle_v1|cooldown",
                    "position_signal": "HOLD",
                    "position_reason": "lifecycle_v1_cooldown",
                    "score": float(dec.quality_score),
                    "conviction": float(dec.confidence),
                    "owner": "",
                    "regime": regime,
                    "decision_logic": entry_decision_logic,
                    "sleeve_trace": trace,
                }
                return 0, 0.0, 0.0, 1.0, info, regime.upper()
        if int(dec.action) == int(FULLY_LEARNED_ACTION_CASH) or action == 0:
            v31_decision = self._v31_deep_alpha_entry_decision(
                frame,
                meta_router=meta_router,
                regime=regime,
                raw_regime=raw_regime,
                parent_dec=dec,
                parent_trace=trace,
            )
            if v31_decision is not None:
                return v31_decision
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "lifecycle_v1|cash",
                "position_signal": "HOLD",
                "position_reason": "lifecycle_v1_clean_base_cash",
                "score": float(dec.quality_score),
                "conviction": float(dec.confidence),
                "owner": "",
                "regime": regime,
                "decision_logic": entry_decision_logic,
                "model_version": "V1.5" if bool(self.conformal_veto_v1_5_enabled) else ("V22.1" if self.v21_adapter_version == "v22_1_learned_scout" else "V21"),
                "model_id": "clean_base_causal_sleeve_conformal_veto_v1_5" if bool(self.conformal_veto_v1_5_enabled) else str(self.v21_model_id or "deep_state_safe_cap_reallocator_v21_nearmiss_scout_stop"),
                "model_path": str(self.conformal_veto_v1_5_model_path) if bool(self.conformal_veto_v1_5_enabled) else str(self.v21_model_path),
                "model_sleeve": "",
                "scout_prob": 0.0,
                "scout_frac": 0.0,
                "sleeve_trace": trace,
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()

        gated_dec, alpha2_meta = self._alpha2_1_apply_parent_gate(dec, frame, trace)
        trace["alpha2_1"] = dict(alpha2_meta)
        if gated_dec is None:
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "alpha3|teacher_gate_pruned_parent",
                "position_signal": "HOLD",
                "position_reason": str(alpha2_meta.get("reason", "alpha2_1_teacher_gate_pruned_parent")),
                "score": float(alpha2_meta.get("teacher_quality", getattr(dec, "quality_score", 0.0))),
                "conviction": float(alpha2_meta.get("teacher_confidence", getattr(dec, "confidence", 0.0))),
                "owner": "",
                "regime": regime,
                "decision_logic": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                "model_version": "Alpha3",
                "model_id": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                "model_path": str(self.alpha2_1_teacher_model_path),
                "model_sleeve": "teacher_parent_gate",
                "sleeve_trace": trace,
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()
        dec = gated_dec
        action = 1 if int(dec.side) > 0 else (2 if int(dec.side) < 0 else 0)

        v15_base_risk_applied = False
        v15_base_risk_reasons: list[str] = []
        v15_base_risk_ctx: dict = {}
        if bool(self.conformal_veto_v1_5_enabled):
            base_after_risk, v15_risk_blocked, v15_base_risk_reasons, v15_base_risk_ctx = self._lifecycle_v1_apply_risk_gates(
                meta_router,
                frame,
                float(dec.notional_exposure),
            )
            v15_base_risk_applied = True
            trace["base_notional_before_risk"] = float(dec.notional_exposure)
            trace["base_notional_after_risk"] = float(base_after_risk)
            trace["risk_reasons"] = list(v15_base_risk_reasons)
            trace["risk_context"] = dict(v15_base_risk_ctx)
            if v15_risk_blocked:
                info = {
                    "agent": "FINAL_GOVERNOR",
                    "source": "lifecycle_v1|risk_block",
                    "position_signal": "HOLD",
                    "position_reason": "|".join(v15_base_risk_reasons) or "lifecycle_v1_risk_block",
                    "score": float(dec.quality_score),
                    "conviction": float(dec.confidence),
                    "owner": "",
                    "regime": regime,
                    "decision_logic": entry_decision_logic,
                    "model_version": "V1.5",
                    "model_id": "clean_base_causal_sleeve_conformal_veto_v1_5",
                    "model_path": str(self.conformal_veto_v1_5_model_path),
                    "model_sleeve": "",
                    "sleeve_trace": trace,
                }
                return 0, 0.0, 0.0, 1.0, info, regime.upper()
            dec = dec.copy()
            dec.loc["notional_exposure"] = float(base_after_risk)

        if self._v21_2_jackpot_available():
            adapter = self.v21_2_jackpot_adapter
            cap = float(adapter.max_entry_notional() if adapter is not None else 2.75)
            effective_notional = float(np.clip(float(dec.notional_exposure), 0.0, cap))
            edit = "v21_2_jackpot_parent"
            edit_meta = {"entry_bucket": "v21_2_parent", "entry_hazard": 0.0, "entry_support": 0}
        else:
            effective_notional, edit, edit_meta = self._lifecycle_v1_entry_edit(frame, dec)
        conformal_veto_v1_5_meta: dict = {}
        if bool(self.conformal_veto_v1_5_enabled) and not self._v21_2_jackpot_available():
            override_notional, override_edit, conformal_veto_v1_5_meta = self._lifecycle_v1_apply_conformal_veto_v1_5(
                meta_router,
                frame,
                dec=dec,
                current_notional=effective_notional,
            )
            if conformal_veto_v1_5_meta.get("applied"):
                effective_notional = float(override_notional)
                if str(override_edit) != "conformal_veto_v1_5_core":
                    edit = str(override_edit)
        pre_deep_gated_gross_notional = float(effective_notional)
        deep_gated_gross_meta: dict = {}
        if bool(self.deep_gated_gross_enabled) and not self._v21_2_jackpot_available():
            override_notional, override_edit, deep_gated_gross_meta = self._lifecycle_v1_apply_deep_gated_gross(
                meta_router,
                frame,
                dec=dec,
                current_notional=effective_notional,
            )
            if deep_gated_gross_meta.get("applied"):
                effective_notional = float(override_notional)
                edit = str(override_edit)
        pre_adaptive_calibrator_notional = float(effective_notional)
        adaptive_calibrator_meta: dict = {}
        if bool(self.deep_state_adaptive_calibrator_enabled) and not self._v21_2_jackpot_available():
            override_notional, override_edit, adaptive_calibrator_meta = self._lifecycle_v1_apply_deep_state_adaptive_calibrator(
                meta_router,
                frame,
                dec=dec,
                current_notional=effective_notional,
                deep_gated_gross_meta=deep_gated_gross_meta,
            )
            if adaptive_calibrator_meta.get("applied") or adaptive_calibrator_meta.get("blocked"):
                effective_notional = float(override_notional)
                if adaptive_calibrator_meta.get("blocked") or float(adaptive_calibrator_meta.get("router_scale", 1.0) or 1.0) < 1.0:
                    edit = str(override_edit)
        pre_safe_cap_notional = float(effective_notional)
        safe_cap_meta: dict = {}
        safe_cap_risk_cap: float | None = None
        if bool(self.safe_learned_cap_enabled) and not self._v21_2_jackpot_available():
            override_notional, override_edit, safe_cap_meta = self._lifecycle_v1_apply_safe_learned_cap(
                meta_router,
                frame,
                dec=dec,
                current_notional=effective_notional,
                deep_gated_gross_meta=deep_gated_gross_meta,
            )
            if safe_cap_meta.get("applied") or safe_cap_meta.get("blocked"):
                effective_notional = float(override_notional)
                edit = str(override_edit)
                safe_cap_risk_cap = float(safe_cap_meta.get("candidate_max_notional", 0.0) or 0.0)
        v21_entry_meta: dict = {}
        v21_sleeve = ""
        if self._v21_2_jackpot_available():
            v21_entry_meta = {
                "enabled": True,
                "applied": True,
                "blocked": False,
                "sleeve": "core",
                "model_id": str(self.v21_2_jackpot_adapter.model_id),
                "model_version": "V21.2",
                "adapter_version": "v21_2_jackpot_runner",
                "model": str(self.v21_2_jackpot_model_path),
                "report": str(self.v21_2_jackpot_report_path),
                "audit": str(self.v21_2_jackpot_audit_path),
                "output_notional": float(effective_notional),
                "parent_notional": float(effective_notional),
                "stop_raw": 999.0,
                "stop_reasons": ["parent_tp_sl_max_hold"],
                "scout": {
                    "adapter": "v21_2_jackpot_runner",
                    "source": "core_parent_entry",
                    "scout_prob": 0.0,
                    "scout_frac": 0.0,
                    "probability_threshold": float(self.v21_2_jackpot_adapter.selected_config.get("jackpot_p", 0.20)),
                    "cost_pass": True,
                },
                "selected_config": dict(self.v21_2_jackpot_adapter.selected_config),
            }
            override_notional = float(effective_notional)
            override_sleeve = "core"
        elif self._lifecycle_v22_1_available():
            override_notional, override_sleeve, v21_entry_meta = self._lifecycle_v22_1_apply_entry_layer(
                meta_router,
                frame,
                dec=dec,
                pre_adaptive_notional=pre_adaptive_calibrator_notional,
                v17_notional=pre_safe_cap_notional,
                current_notional=effective_notional,
                deep_gated_gross_meta=deep_gated_gross_meta,
                adaptive_calibrator_meta=adaptive_calibrator_meta,
                safe_cap_meta=safe_cap_meta,
            )
        elif self._lifecycle_v21_available():
            override_notional, override_sleeve, v21_entry_meta = self._lifecycle_v21_apply_entry_layer(
                meta_router,
                frame,
                dec=dec,
                pre_adaptive_notional=pre_adaptive_calibrator_notional,
                v17_notional=pre_safe_cap_notional,
                current_notional=effective_notional,
                deep_gated_gross_meta=deep_gated_gross_meta,
                adaptive_calibrator_meta=adaptive_calibrator_meta,
                safe_cap_meta=safe_cap_meta,
            )
        if v21_entry_meta and (
            v21_entry_meta.get("applied") or (self._lifecycle_v21_pure_active() and v21_entry_meta.get("blocked"))
        ):
            effective_notional = float(override_notional)
            v21_sleeve = str(override_sleeve or v21_entry_meta.get("sleeve", ""))
            if v21_entry_meta.get("applied"):
                if v21_sleeve == "scout":
                    edit = "v22_1_learned_scout" if v21_entry_meta.get("model_version") == "V22.1" else "v21_near_miss_scout"
                    safe_cap_risk_cap = float(v21_entry_meta.get("output_notional", 0.0) or 0.0)
                elif v21_sleeve == "core":
                    safe_cap_risk_cap = float((safe_cap_meta or {}).get("candidate_max_notional", safe_cap_risk_cap or 0.0) or 0.0)
            else:
                edit = str(v21_entry_meta.get("reason", "v21_entry_block"))
                safe_cap_risk_cap = 0.0
        pre_constant_gross_notional = float(effective_notional)
        constant_gross_meta: dict = {}
        if (
            not self._lifecycle_v21_pure_active()
            and not deep_gated_gross_meta.get("applied")
            and not safe_cap_meta.get("applied")
            and bool(self.deep_constant_gross_enabled)
        ):
            override_notional, override_edit, constant_gross_meta = self._lifecycle_v1_apply_deep_constant_gross(
                meta_router,
                frame,
                current_notional=effective_notional,
            )
            effective_notional = float(override_notional)
            if constant_gross_meta.get("applied"):
                edit = str(override_edit)
        pre_dsac_overlay_notional = float(effective_notional)
        dsac_overlay_meta: dict = {}
        if not self._lifecycle_v21_pure_active() and bool(self.dsac_overlay_enabled):
            override_notional, override_edit, dsac_overlay_meta = self._lifecycle_v1_apply_dsac_overlay(
                meta_router,
                frame,
                side=side,
                current_notional=effective_notional,
                deep_gated_gross_meta=deep_gated_gross_meta,
            )
            effective_notional = float(override_notional)
            if dsac_overlay_meta.get("applied"):
                edit = str(override_edit)
        if bool(self.conformal_veto_v1_5_enabled) and v15_base_risk_applied:
            risk_blocked = False
            risk_reasons = list(v15_base_risk_reasons)
            risk_ctx = dict(v15_base_risk_ctx)
            risk_ctx.update(
                {
                    "risk_adjusted_notional": float(effective_notional),
                    "v1_5_risk_order": "base_risk_before_lifecycle_edit_and_conformal_sleeve",
                    "v1_5_total_notional_after_sleeve": float(effective_notional),
                }
            )
        elif self._lifecycle_v21_pure_active() and self.v21_bypass_runtime_risk_gates:
            risk_ctx = self._lifecycle_v1_daily_context(meta_router, frame)
            leverage_cap = float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0))
            effective_notional = float(np.clip(float(effective_notional), 0.0, leverage_cap))
            risk_blocked = bool(effective_notional <= 1e-12)
            risk_reasons = ["v21_pure_runtime_risk_gates_bypassed"]
            if risk_blocked:
                risk_reasons.append("zero_effective_notional")
            risk_ctx.update(
                {
                    "risk_adjusted_notional": float(effective_notional),
                    "risk_cap": float(leverage_cap),
                    "router_exposure_cap": float(leverage_cap),
                    "v21_pure_runtime_risk_gates_bypassed": True,
                    "max_notional_override_ignored": float(safe_cap_risk_cap or 0.0),
                }
            )
        else:
            effective_notional, risk_blocked, risk_reasons, risk_ctx = self._lifecycle_v1_apply_risk_gates(
                meta_router,
                frame,
                effective_notional,
                max_notional_override=safe_cap_risk_cap,
            )
        if deep_gated_gross_meta:
            deep_gated_gross_meta["pre_override_notional"] = float(pre_deep_gated_gross_notional)
            deep_gated_gross_meta["post_risk_notional"] = float(effective_notional)
        if adaptive_calibrator_meta:
            adaptive_calibrator_meta["pre_override_notional"] = float(pre_adaptive_calibrator_notional)
            adaptive_calibrator_meta["post_risk_notional"] = float(effective_notional)
        if safe_cap_meta:
            safe_cap_meta["pre_override_notional"] = float(pre_safe_cap_notional)
            safe_cap_meta["post_risk_notional"] = float(effective_notional)
        if constant_gross_meta:
            constant_gross_meta["pre_override_notional"] = float(pre_constant_gross_notional)
            constant_gross_meta["post_risk_notional"] = float(effective_notional)
        if dsac_overlay_meta:
            dsac_overlay_meta["pre_overlay_notional"] = float(pre_dsac_overlay_notional)
            dsac_overlay_meta["post_risk_notional"] = float(effective_notional)
        if v21_entry_meta:
            v21_entry_meta["post_risk_notional"] = float(effective_notional)
        if conformal_veto_v1_5_meta:
            conformal_veto_v1_5_meta["post_risk_notional"] = float(effective_notional)
        trace.update(
            {
                "risk_reasons": list(risk_reasons),
                "risk_context": dict(risk_ctx),
                "conformal_veto_v1_5": dict(conformal_veto_v1_5_meta),
                "deep_gated_gross": dict(deep_gated_gross_meta),
                "deep_state_adaptive_calibrator": dict(adaptive_calibrator_meta),
                "safe_learned_cap": dict(safe_cap_meta),
                "v21_nearmiss_scout_stop": dict(v21_entry_meta),
                "deep_constant_gross": dict(constant_gross_meta),
                "dsac_overlay": dict(dsac_overlay_meta),
            }
        )
        if risk_blocked:
            if self._lifecycle_v21_pure_active() and v21_entry_meta.get("blocked"):
                block_prefix = "v22_1_scout_param_grid" if v21_entry_meta.get("model_version") == "V22.1" else "v21_nearmiss_scout_stop"
                block_source = f"{block_prefix}|entry_block"
                block_reason = str(v21_entry_meta.get("reason", "v21_entry_block"))
            elif adaptive_calibrator_meta.get("blocked"):
                block_source = "lifecycle_v1|adaptive_calibrator_block"
                block_reason = str(adaptive_calibrator_meta.get("reason", "adaptive_calibrator_block"))
            elif safe_cap_meta.get("blocked"):
                block_source = "lifecycle_v1|safe_learned_cap_block"
                block_reason = str(safe_cap_meta.get("reason", "safe_learned_cap_cost_gate_block"))
            elif dsac_overlay_meta.get("blocked"):
                block_source = "lifecycle_v1|dsac_overlay_block"
                block_reason = str(dsac_overlay_meta.get("reason", ""))
            else:
                block_source = "lifecycle_v1|risk_block"
                block_reason = ""
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": block_source,
                "position_signal": "HOLD",
                "position_reason": block_reason or "|".join(risk_reasons) or "lifecycle_v1_risk_block",
                "score": float(dec.quality_score),
                "conviction": float(dec.confidence),
                "owner": "",
                "regime": regime,
                "decision_logic": entry_decision_logic,
                "sleeve_trace": trace,
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()
        leverage_cap = float(max(float(getattr(meta_router, "exposure_cap", 5.0) or 5.0), 1.0))
        if float(effective_notional) > leverage_cap:
            effective_notional = float(leverage_cap)
            risk_reasons.append("router_exposure_cap")
            trace["risk_reasons"] = list(risk_reasons)
            trace["risk_context"] = {**dict(risk_ctx), "router_exposure_cap": float(leverage_cap), "risk_adjusted_notional": float(effective_notional)}
            if "deep_gated_gross" in trace:
                trace["deep_gated_gross"] = {**dict(trace.get("deep_gated_gross", {}) or {}), "post_router_cap_notional": float(effective_notional)}
            if "deep_state_adaptive_calibrator" in trace:
                trace["deep_state_adaptive_calibrator"] = {**dict(trace.get("deep_state_adaptive_calibrator", {}) or {}), "post_router_cap_notional": float(effective_notional)}
            if "safe_learned_cap" in trace:
                trace["safe_learned_cap"] = {**dict(trace.get("safe_learned_cap", {}) or {}), "post_router_cap_notional": float(effective_notional)}
            if "conformal_veto_v1_5" in trace:
                trace["conformal_veto_v1_5"] = {**dict(trace.get("conformal_veto_v1_5", {}) or {}), "post_router_cap_notional": float(effective_notional)}
            if "v21_nearmiss_scout_stop" in trace:
                trace["v21_nearmiss_scout_stop"] = {**dict(trace.get("v21_nearmiss_scout_stop", {}) or {}), "post_router_cap_notional": float(effective_notional)}
            if "deep_constant_gross" in trace:
                trace["deep_constant_gross"] = {**dict(trace.get("deep_constant_gross", {}) or {}), "post_router_cap_notional": float(effective_notional)}
            if "dsac_overlay" in trace:
                trace["dsac_overlay"] = {**dict(trace.get("dsac_overlay", {}) or {}), "post_router_cap_notional": float(effective_notional)}
        entry_dust_floor = float(np.clip(float(FINAL_GOVERNOR_DUST_ENTRY_EXPOSURE), 0.0, leverage_cap))
        entry_min_floor = float(np.clip(float(FINAL_GOVERNOR_MIN_ENTRY_EXPOSURE), 0.0, leverage_cap))
        if entry_min_floor > 0.0 and entry_dust_floor > entry_min_floor:
            entry_dust_floor = float(entry_min_floor)
        if 0.0 < float(effective_notional) < entry_dust_floor:
            pre_filter_notional = float(effective_notional)
            risk_reasons.append("min_entry_dust_block")
            trace["risk_reasons"] = list(risk_reasons)
            trace["risk_context"] = {
                **dict(risk_ctx),
                "entry_dust_floor": float(entry_dust_floor),
                "entry_min_floor": float(entry_min_floor),
                "pre_min_entry_filter_notional": float(pre_filter_notional),
                "risk_adjusted_notional": 0.0,
            }
            trace["effective_notional_before_min_entry_filter"] = float(pre_filter_notional)
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "lifecycle_v1|min_entry_dust_block",
                "position_signal": "HOLD",
                "position_reason": "min_entry_dust_block",
                "score": float(dec.quality_score),
                "conviction": float(dec.confidence),
                "owner": "",
                "regime": regime,
                "decision_logic": entry_decision_logic,
                "sleeve_trace": trace,
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()
        if 0.0 < float(effective_notional) < entry_min_floor:
            pre_floor_notional = float(effective_notional)
            effective_notional = float(entry_min_floor)
            risk_reasons.append("min_entry_notional_floor")
            risk_ctx = {
                **dict(risk_ctx),
                "entry_dust_floor": float(entry_dust_floor),
                "entry_min_floor": float(entry_min_floor),
                "pre_min_entry_floor_notional": float(pre_floor_notional),
                "risk_adjusted_notional": float(effective_notional),
            }
            trace["risk_reasons"] = list(risk_reasons)
            trace["risk_context"] = dict(risk_ctx)
            trace["effective_notional_before_min_entry_floor"] = float(pre_floor_notional)
            for layer_key in (
                "deep_gated_gross",
                "deep_state_adaptive_calibrator",
                "safe_learned_cap",
                "conformal_veto_v1_5",
                "v21_nearmiss_scout_stop",
                "deep_constant_gross",
                "dsac_overlay",
            ):
                if layer_key in trace:
                    trace[layer_key] = {
                        **dict(trace.get(layer_key, {}) or {}),
                        "post_min_entry_floor_notional": float(effective_notional),
                    }
        leverage = float(np.clip(max(float(dec.leverage or 1.0), float(effective_notional), 1.0), 1.0, leverage_cap))
        fraction = float(np.clip(effective_notional / max(leverage, 1e-8), 0.0, 1.0))
        self.owner = "lifecycle_v1"
        self.owner_regime = regime
        self.peak_unrealized = 0.0
        self.active_lifecycle_v1_mae_unrealized = 0.0
        self.active_lifecycle_v1_base_notional = float(dec.notional_exposure)
        self.active_lifecycle_v1_effective_notional = float(effective_notional)
        self.active_lifecycle_v1_leverage = float(leverage)
        active_cooldown_bars = int(max(0, dec.cooldown_bars))
        if self._lifecycle_v21_pure_active() and self.v21_bypass_cooldown:
            trace["v21_entry_cooldown_bypassed_bars"] = int(active_cooldown_bars)
            active_cooldown_bars = 0
        self.active_lifecycle_v1_cooldown_bars = int(active_cooldown_bars)
        self.active_lifecycle_v1_quality_score = float(dec.quality_score)
        self.active_lifecycle_v1_confidence = float(dec.confidence)
        self.active_lifecycle_v1_entry_bucket = str(edit_meta.get("entry_bucket", ""))
        self.active_lifecycle_v1_entry_hazard = float(edit_meta.get("entry_hazard", 0.0) or 0.0)
        self.active_lifecycle_v1_entry_support = int(edit_meta.get("entry_support", 0) or 0)
        self.active_lifecycle_v1_edit = str(edit)
        self.active_lifecycle_v1_take_profit = float(dec.take_profit)
        self.active_lifecycle_v1_stop_loss = float(dec.stop_loss)
        self.active_lifecycle_v1_max_hold_bars = int(dec.max_hold_bars)
        self.active_lifecycle_v1_jackpot_added = False
        if v21_entry_meta.get("applied"):
            self.active_lifecycle_v1_v21_sleeve = str(v21_sleeve or v21_entry_meta.get("sleeve", ""))
            self.active_lifecycle_v1_v21_stop_raw = float(v21_entry_meta.get("stop_raw", 999.0) or 999.0)
            self.active_lifecycle_v1_v21_peak_raw = -1e9
            self.active_lifecycle_v1_v21_stop_reasons = list(v21_entry_meta.get("stop_reasons", []) or [])
            self.active_lifecycle_v1_scout_model_version = str(v21_entry_meta.get("model_version", "") or "")
            self.active_lifecycle_v1_scout_model_id = str(v21_entry_meta.get("model_id", "") or "")
            self.active_lifecycle_v1_scout_model_path = str(v21_entry_meta.get("model", "") or "")
            scout_row = dict(v21_entry_meta.get("scout", {}) or {})
            self.active_lifecycle_v1_scout_prob = float(
                scout_row.get("scout_prob", scout_row.get("probability", v21_entry_meta.get("scout_prob", 0.0))) or 0.0
            )
            self.active_lifecycle_v1_scout_frac = float(
                scout_row.get("scout_frac", scout_row.get("learned_scout_frac_pred", v21_entry_meta.get("scout_frac", 0.0))) or 0.0
            )
            self.active_lifecycle_v1_scout_probability_threshold = float(
                scout_row.get(
                    "probability_threshold",
                    scout_row.get("learned_scout_threshold", v21_entry_meta.get("probability_threshold", 0.0)),
                )
                or 0.0
            )
            self.active_lifecycle_v1_scout_cost_pass = bool(scout_row.get("cost_pass", False))
        else:
            self.active_lifecycle_v1_v21_sleeve = ""
            self.active_lifecycle_v1_v21_stop_raw = 999.0
            self.active_lifecycle_v1_v21_peak_raw = -1e9
            self.active_lifecycle_v1_v21_stop_reasons = []
            self.active_lifecycle_v1_scout_model_version = ""
            self.active_lifecycle_v1_scout_model_id = ""
            self.active_lifecycle_v1_scout_model_path = ""
            self.active_lifecycle_v1_scout_prob = 0.0
            self.active_lifecycle_v1_scout_frac = 0.0
            self.active_lifecycle_v1_scout_probability_threshold = 0.0
            self.active_lifecycle_v1_scout_cost_pass = False
        if conformal_veto_v1_5_meta.get("applied"):
            self.active_lifecycle_v1_conformal_core_notional = float(
                conformal_veto_v1_5_meta.get("core_notional", effective_notional) or 0.0
            )
            self.active_lifecycle_v1_conformal_sleeve_notional = float(
                conformal_veto_v1_5_meta.get("sleeve_notional", 0.0) or 0.0
            )
            self.active_lifecycle_v1_conformal_sleeve_exit_bars = int(
                conformal_veto_v1_5_meta.get("sleeve_exit_bars", 0) or 0
            )
            self.active_lifecycle_v1_conformal_sleeve_action = str(
                conformal_veto_v1_5_meta.get("action", "") or ""
            )
        else:
            self.active_lifecycle_v1_conformal_core_notional = 0.0
            self.active_lifecycle_v1_conformal_sleeve_notional = 0.0
            self.active_lifecycle_v1_conformal_sleeve_exit_bars = 0
            self.active_lifecycle_v1_conformal_sleeve_action = ""
        self._save_runtime_state()
        trace.update(
            {
                "lifecycle_edit": str(edit),
                "effective_notional": float(effective_notional),
                "effective_position_fraction": float(fraction),
                "v21_sleeve": str(self.active_lifecycle_v1_v21_sleeve),
                "v21_stop_raw": float(self.active_lifecycle_v1_v21_stop_raw),
                "v21_stop_reasons": list(self.active_lifecycle_v1_v21_stop_reasons or []),
                "conformal_veto_v1_5_action": str(self.active_lifecycle_v1_conformal_sleeve_action),
                "conformal_veto_v1_5_core_notional": float(self.active_lifecycle_v1_conformal_core_notional),
                "conformal_veto_v1_5_sleeve_notional": float(self.active_lifecycle_v1_conformal_sleeve_notional),
                "conformal_veto_v1_5_sleeve_exit_bars": int(self.active_lifecycle_v1_conformal_sleeve_exit_bars),
                **edit_meta,
            }
        )
        active_scout_model_id = str(
            (v21_entry_meta or {}).get("model_id", self.v21_model_id or "deep_state_safe_cap_reallocator_v21_nearmiss_scout_stop")
        )
        active_scout_model_version = str((v21_entry_meta or {}).get("model_version", "V21"))
        active_scout_prefix = (
            "v21_2_jackpot_runner"
            if active_scout_model_version == "V21.2"
            else ("v22_1_scout_param_grid" if active_scout_model_version == "V22.1" else "v21_nearmiss_scout_stop")
        )
        active_scout_model_path = str((v21_entry_meta or {}).get("model", self.v21_model_path))
        v15_applied = bool(conformal_veto_v1_5_meta.get("applied"))
        v15_model_id = str(conformal_veto_v1_5_meta.get("model_id", "clean_base_causal_sleeve_conformal_veto_v1_5"))
        v15_model_path = str(conformal_veto_v1_5_meta.get("model", self.conformal_veto_v1_5_model_path))
        sub_decision_logic = (
            active_scout_model_id
            if v21_entry_meta.get("applied")
            else (v15_model_id if v15_applied else "clean_base_lifecycle_v1")
        )
        decision_logic = FINAL_GOVERNOR_ALPHA3_MODEL_ID
        trace["decision_logic"] = decision_logic
        if decision_logic.startswith("ddh2_"):
            trace["sub_decision_logic"] = sub_decision_logic
            trace["ddh2_ensemble"] = {
                "enabled": True,
                "report": str(self.ddh2_report_path),
                "audit": str(self.ddh2_audit_path),
                "source_layer": "v22_1_sniper",
            }
        if v21_entry_meta.get("applied"):
            trace["model_version"] = active_scout_model_version
            trace["model_path"] = active_scout_model_path
            trace["scout_adapter_version"] = str((v21_entry_meta or {}).get("adapter_version", self.v21_adapter_version))
        elif v15_applied:
            trace["model_version"] = "V1.5"
            trace["model_path"] = v15_model_path
        if v21_entry_meta.get("applied"):
            source = f"alpha3|{active_scout_prefix}|entry_{self.active_lifecycle_v1_v21_sleeve or 'core'}"
            reason = f"alpha2_1_{active_scout_prefix}_entry_{self.active_lifecycle_v1_v21_sleeve or 'core'}"
        elif v15_applied:
            sleeve_action = str(conformal_veto_v1_5_meta.get("action", "NO_SLEEVE") or "NO_SLEEVE").lower()
            source = f"conformal_veto_v1_5|entry_{sleeve_action}"
            reason = f"conformal_veto_v1_5_entry_{sleeve_action}"
        else:
            source = f"lifecycle_v1|entry_{edit}"
            reason = f"lifecycle_v1_entry_{edit}"
        info = {
            "agent": "FINAL_GOVERNOR",
            "source": source,
            "position_signal": "LONG_ENTRY" if action == 1 else "SHORT_ENTRY",
            "position_reason": reason,
            "score": float(dec.quality_score),
            "conviction": float(dec.confidence),
            "owner": "lifecycle_v1",
            "regime": regime,
            "decision_logic": decision_logic,
            "quality_score": float(dec.quality_score),
            "confidence": float(dec.confidence),
            "model_version": "Alpha3",
            "model_id": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
            "model_path": active_scout_model_path if v21_entry_meta.get("applied") else (v15_model_path if v15_applied else str(self.lifecycle_v1_model_path)),
            "model_sleeve": str(
                self.active_lifecycle_v1_v21_sleeve
                or (self.active_lifecycle_v1_conformal_sleeve_action if v15_applied else "")
            ),
            "scout_prob": float(
                dict(v21_entry_meta.get("scout", {}) or {}).get(
                    "scout_prob",
                    dict(v21_entry_meta.get("scout", {}) or {}).get("probability", 0.0),
                )
                or 0.0
            ),
            "scout_frac": float(
                dict(v21_entry_meta.get("scout", {}) or {}).get(
                    "scout_frac",
                    dict(v21_entry_meta.get("scout", {}) or {}).get("learned_scout_frac_pred", 0.0),
                )
                or 0.0
            ),
            "scout_probability_threshold": float(self.active_lifecycle_v1_scout_probability_threshold),
            "scout_cost_pass": bool(self.active_lifecycle_v1_scout_cost_pass),
            "sleeve_trace": trace,
        }
        return action, float(effective_notional), fraction, leverage, info, regime.upper()

    def _reset_omega5_position_state(self) -> None:
        self.active_omega5_take_profit = 0.0
        self.active_omega5_stop_loss = 0.0
        self.active_omega5_max_hold_bars = 0
        self.active_omega5_quality_score = 0.0
        self.active_omega5_confidence = 0.0
        self.active_omega5_notional = 0.0
        self.active_omega5_leverage = 1.0
        self.active_omega5_parent_exit_timestamp = ""
        self.active_omega5_roundtrip_cost = 0.0
        self.active_omega5_source_exit_reason = ""
        self.active_omega5_source_exit_price_move = 0.0
        self.active_omega5_sizing_trace = {}

    def _manage_omega5_position(
        self,
        *,
        meta_router,
        current_price: float,
        frame: pd.DataFrame,
        regime: str,
    ) -> tuple[int, float, float, float, dict, str]:
        pos = str(meta_router.pos or "")
        action_hold = self._action_from_side(pos)
        unrealized = float(meta_router._net_pnl_frac(current_price))
        self.peak_unrealized = max(float(self.peak_unrealized), unrealized)
        hold_bars = int(meta_router.hold_count or 0)
        close = False
        reason = "omega5_hold"
        exit_price_override = 0.0
        state_unarmed = (
            self.active_omega5_take_profit <= 0.0
            and self.active_omega5_stop_loss <= 0.0
            and self.active_omega5_max_hold_bars <= 0
        )
        if state_unarmed:
            raise RuntimeError("Omega5 active position is missing TP/SL/max-hold state; reconcile close fallback is forbidden")
        if self.active_omega5_notional <= 0.0 or self.active_omega5_leverage <= 0.0:
            raise RuntimeError(
                "Omega5 active position is missing notional/leverage state; exposure fallback is forbidden"
            )
        latest = frame.iloc[-1] if len(frame) else {}
        entry = float(getattr(meta_router, "entry_price", 0.0) or 0.0)
        source_exit_armed = bool(self.active_omega5_parent_exit_timestamp)
        if source_exit_armed:
            latest_ts = pd.Timestamp(frame.iloc[-1].get("timestamp")) if len(frame) and "timestamp" in frame.columns else pd.NaT
            parent_exit_ts = pd.Timestamp(self.active_omega5_parent_exit_timestamp)
            if not pd.isna(latest_ts) and latest_ts >= parent_exit_ts:
                close = True
                source_reason = str(self.active_omega5_source_exit_reason or "")
                if source_reason == "roll8_bracket_tp":
                    reason = "omega5_take_profit"
                elif source_reason == "roll8_bracket_sl":
                    reason = "omega5_stop_loss"
                elif source_reason == "roll8_time_exit":
                    reason = "omega5_max_hold"
                else:
                    reason = "omega5_parent_final"
                raw_move = float(self.active_omega5_source_exit_price_move or 0.0)
                if entry > 0.0 and raw_move != 0.0 and pos in {"LONG", "SHORT"}:
                    exit_price_override = entry * (1.0 + raw_move) if pos == "LONG" else entry * (1.0 - raw_move)
        else:
            high = float(latest.get("high", current_price) if hasattr(latest, "get") else current_price)
            low = float(latest.get("low", current_price) if hasattr(latest, "get") else current_price)
            notional = float(self.active_omega5_notional)
            if entry > 0.0 and high > 0.0 and low > 0.0 and pos in {"LONG", "SHORT"}:
                tp_move = float(self.active_omega5_take_profit) / max(notional, 1.0e-12)
                sl_move = float(self.active_omega5_stop_loss) / max(notional, 1.0e-12)
                if pos == "LONG":
                    if tp_move > 0.0 and (high / entry - 1.0) >= tp_move:
                        close = True
                        reason = "omega5_take_profit"
                        exit_price_override = entry * (1.0 + tp_move)
                    elif sl_move > 0.0 and (low / entry - 1.0) <= -sl_move:
                        close = True
                        reason = "omega5_stop_loss"
                        exit_price_override = entry * (1.0 - sl_move)
                else:
                    if tp_move > 0.0 and (entry - low) / entry >= tp_move:
                        close = True
                        reason = "omega5_take_profit"
                        exit_price_override = entry * (1.0 - tp_move)
                    elif sl_move > 0.0 and (high - entry) / entry >= sl_move:
                        close = True
                        reason = "omega5_stop_loss"
                        exit_price_override = entry * (1.0 + sl_move)
            if (not close) and self.active_omega5_max_hold_bars > 0 and hold_bars >= max(0, self.active_omega5_max_hold_bars - 1):
                close = True
                reason = "omega5_max_hold"

        if close:
            if self.omega4_6_2_source_parent_adapter is None:
                raise RuntimeError("Omega5 close requires Omega4.6.2 source parent adapter for loss-governor state update")
            exit_px_for_governor = float(exit_price_override or current_price)
            if entry > 0.0 and exit_px_for_governor > 0.0 and pos in {"LONG", "SHORT"}:
                gross_for_governor = (
                    (exit_px_for_governor - entry) / max(entry, 1.0e-12)
                    if pos == "LONG"
                    else (entry - exit_px_for_governor) / max(entry, 1.0e-12)
                )
                net_for_governor = float(gross_for_governor - float(self.active_omega5_roundtrip_cost or 0.0))
                exit_ts_for_governor = (
                    pd.Timestamp(frame.iloc[-1].get("timestamp"))
                    if len(frame) and "timestamp" in frame.columns
                    else pd.Timestamp.utcnow()
                )
                self.omega4_6_2_source_parent_adapter.record_closed_trade(
                    exit_timestamp=exit_ts_for_governor,
                    net_per_notional=net_for_governor,
                )
            self.last_exit_bar = self.bar_counter
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": f"omega5|{reason}",
                "position_signal": "EXIT",
                "position_reason": reason,
                "score": abs(unrealized),
                "conviction": float(self.active_omega5_quality_score),
                "owner": OMEGA5_OWNER,
                "regime": regime,
                "decision_logic": OMEGA5_MODEL_ID,
                "model_version": OMEGA5_MODEL_VERSION,
                "model_id": OMEGA5_MODEL_ID,
                "model_sleeve": "omega5_event_risk_governor",
                "take_profit": float(self.active_omega5_take_profit),
                "stop_loss": float(self.active_omega5_stop_loss),
                "max_hold_bars": int(self.active_omega5_max_hold_bars),
                "quality_score": float(self.active_omega5_quality_score),
                "confidence": float(self.active_omega5_confidence),
                "omega5_source_roundtrip_cost": float(self.active_omega5_roundtrip_cost),
                "omega5_source_exit_reason": str(self.active_omega5_source_exit_reason),
                "omega5_source_exit_price_move": float(self.active_omega5_source_exit_price_move),
                "execution_price_override": float(exit_price_override or current_price),
                "sleeve_trace": dict(self.active_omega5_sizing_trace or {}),
            }
            self.owner = ""
            self.owner_regime = ""
            self.peak_unrealized = 0.0
            self._reset_omega5_position_state()
            self._save_runtime_state()
            return 0, 0.0, 0.0, 1.0, info, regime.upper()

        exposure = float(self.active_omega5_notional)
        exec_lev = float(self.active_omega5_leverage)
        fraction = float(exposure / max(exec_lev, 1e-8))
        info = {
            "agent": "FINAL_GOVERNOR",
            "source": "omega5|hold",
            "position_signal": "HOLD",
            "position_reason": reason,
            "score": abs(unrealized),
            "conviction": float(self.active_omega5_quality_score),
            "owner": OMEGA5_OWNER,
            "regime": regime,
            "decision_logic": OMEGA5_MODEL_ID,
            "model_version": OMEGA5_MODEL_VERSION,
            "model_id": OMEGA5_MODEL_ID,
            "model_sleeve": "omega5_event_risk_governor",
            "take_profit": float(self.active_omega5_take_profit),
            "stop_loss": float(self.active_omega5_stop_loss),
            "max_hold_bars": int(self.active_omega5_max_hold_bars),
            "quality_score": float(self.active_omega5_quality_score),
            "confidence": float(self.active_omega5_confidence),
            "omega5_source_roundtrip_cost": float(self.active_omega5_roundtrip_cost),
            "omega5_source_exit_reason": str(self.active_omega5_source_exit_reason),
            "omega5_source_exit_price_move": float(self.active_omega5_source_exit_price_move),
            "sleeve_trace": dict(self.active_omega5_sizing_trace or {}),
        }
        return action_hold, exposure, fraction, exec_lev, info, regime.upper()

    def _omega5_parent_decision(self, frame: pd.DataFrame):
        if self.omega4_6_2_source_parent_adapter is None:
            raise RuntimeError(
                "Omega5 parent decision requested without Omega4.6.2 source parent adapter; "
                "Omega1.2.1/Omega3 parent substitution is forbidden."
            )
        return self.omega4_6_2_source_parent_adapter.decide_latest(frame)

    def _decide_omega5_entry(
        self,
        frame: pd.DataFrame,
        *,
        regime: str,
        raw_regime: str,
    ) -> tuple[int, float, float, float, dict, str]:
        if self.omega5_adapter is None:
            raise RuntimeError("Omega5 entry requested while adapter is disabled")
        parent_dec = self._omega5_parent_decision(frame)
        dec = self.omega5_adapter.decide_latest(frame, parent_dec)
        trace = dict(dec.trace)
        trace.update(
            {
                "decision_logic": OMEGA5_MODEL_ID,
                "model_id": OMEGA5_MODEL_ID,
                "model_version": OMEGA5_MODEL_VERSION,
                "regime": regime,
                "raw_regime": raw_regime,
            }
        )
        if int(dec.action) == int(FULLY_LEARNED_ACTION_CASH) or int(dec.side) == 0:
            reason = str(trace.get("omega5_reason", "cash"))
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": f"omega5|{reason}",
                "position_signal": "HOLD",
                "position_reason": f"omega5_{reason}",
                "score": float(dec.quality_score),
                "conviction": float(dec.confidence),
                "owner": "",
                "regime": regime,
                "decision_logic": OMEGA5_MODEL_ID,
                "model_version": OMEGA5_MODEL_VERSION,
                "model_id": OMEGA5_MODEL_ID,
                "sleeve_trace": trace,
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()

        action = 1 if int(dec.action) == int(FULLY_LEARNED_ACTION_LONG) else 2
        self.owner = OMEGA5_OWNER
        self.owner_regime = regime
        self.peak_unrealized = 0.0
        self.active_omega5_take_profit = float(dec.take_profit)
        self.active_omega5_stop_loss = float(dec.stop_loss)
        self.active_omega5_max_hold_bars = int(dec.max_hold_bars)
        self.active_omega5_quality_score = float(dec.quality_score)
        self.active_omega5_confidence = float(dec.confidence)
        self.active_omega5_notional = float(dec.notional_exposure)
        self.active_omega5_leverage = float(dec.leverage)
        self.active_omega5_parent_exit_timestamp = str(
            (trace.get("parent_trace") or {}).get("source_policy_exit_timestamp", "") or ""
        )
        self.active_omega5_roundtrip_cost = float(trace.get("source_roundtrip_cost", 0.0) or 0.0)
        self.active_omega5_source_exit_reason = str(
            (trace.get("parent_trace") or {}).get("reference_policy_reason", "") or ""
        )
        self.active_omega5_source_exit_price_move = float(trace.get("source_raw_exit_price_move", 0.0) or 0.0)
        self.active_omega5_sizing_trace = dict(GovernorPositionRouter._journal_jsonable(trace))
        self.last_omega5_entry_side = int(dec.side)
        self.last_omega5_entry_bar = int(self.bar_counter)
        self._save_runtime_state()
        info = {
            "agent": "FINAL_GOVERNOR",
            "source": "omega5|entry",
            "position_signal": "LONG_ENTRY" if action == 1 else "SHORT_ENTRY",
            "position_reason": "omega5_entry",
            "score": float(dec.quality_score),
            "conviction": float(dec.confidence),
            "owner": OMEGA5_OWNER,
            "regime": regime,
            "decision_logic": OMEGA5_MODEL_ID,
            "model_version": OMEGA5_MODEL_VERSION,
            "model_id": OMEGA5_MODEL_ID,
            "model_sleeve": "omega5_event_risk_governor",
            "take_profit": float(dec.take_profit),
            "stop_loss": float(dec.stop_loss),
            "max_hold_bars": int(dec.max_hold_bars),
            "quality_score": float(dec.quality_score),
            "confidence": float(dec.confidence),
            "omega5_source_roundtrip_cost": float(self.active_omega5_roundtrip_cost),
            "omega5_source_exit_reason": str(self.active_omega5_source_exit_reason),
            "omega5_source_exit_price_move": float(self.active_omega5_source_exit_price_move),
            "sleeve_trace": trace,
        }
        return action, float(dec.notional_exposure), float(dec.position_fraction), float(dec.leverage), info, regime.upper()

    def _reset_omega4_6_1_position_state(self) -> None:
        self.active_omega4_6_1_source_component = ""
        self.active_omega4_6_1_take_profit = 0.0
        self.active_omega4_6_1_stop_loss = 0.0
        self.active_omega4_6_1_notional = 0.0
        self.active_omega4_6_1_leverage = 1.0
        self.active_omega4_6_1_quality_score = 0.0
        self.active_omega4_6_1_confidence = 0.0
        self.active_omega4_6_1_mfe = 0.0
        self.active_omega4_6_1_mae = 0.0
        self.active_omega4_6_1_tp_order_id = ""
        self.active_omega4_6_1_sl_order_id = ""

    def _decide_omega4_6_1_entry(
        self,
        frame: pd.DataFrame,
        *,
        regime: str,
        raw_regime: str,
    ) -> tuple[int, float, float, float, dict, str]:
        if self.omega4_6_1_adapter is None:
            raise RuntimeError("Omega4.6.1 entry requested while adapter is disabled")
        dec = self.omega4_6_1_adapter.decide_entry(frame)
        if dec is None:
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "omega4_6_1|cash",
                "position_signal": "HOLD",
                "position_reason": "omega4_6_1_cash",
                "score": 0.0,
                "conviction": 0.0,
                "owner": "",
                "regime": regime,
                "raw_regime": raw_regime,
                "decision_logic": OMEGA4_6_1_MODEL_ID,
                "model_version": OMEGA4_6_1_MODEL_VERSION,
                "model_id": OMEGA4_6_1_MODEL_ID,
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()

        action = 1 if dec.side > 0 else 2
        # Notional/leverage scaling (default multiplier 1.0 reproduces dec.notional_exposure /
        # dec.leverage exactly): holds margin_fraction fixed and rescales leverage, matching the
        # convention in scripts/replay_portfolio_concurrent_3asset_native_20260712.py. See
        # docs/model_contracts/portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md.
        _eth_notional = float(dec.notional_exposure) * float(FINAL_GOVERNOR_OMEGA4_6_1_ETH_NOTIONAL_MULTIPLIER)
        _eth_leverage = _eth_notional / max(float(dec.margin_fraction), 1e-12)
        # Shared portfolio notional cap (opt-in via FINAL_GOVERNOR_OMEGA4_6_1_ETH_PORTFOLIO_CAP_ENABLE).
        # self.omega4_6_1_portfolio_risk stays None unless that flag is set, so this is a no-op today.
        if self.omega4_6_1_portfolio_risk is not None:
            _approved_notional = self.omega4_6_1_portfolio_risk.scale_to_budget("eth_omega461", _eth_notional)
            if _approved_notional < self.omega4_6_1_portfolio_risk.config.min_notional:
                info = {
                    "agent": "FINAL_GOVERNOR",
                    "source": "omega4_6_1|cash",
                    "position_signal": "HOLD",
                    "position_reason": "omega4_6_1_portfolio_risk_blocked",
                    "score": 0.0,
                    "conviction": 0.0,
                    "owner": "",
                    "regime": regime,
                    "raw_regime": raw_regime,
                    "decision_logic": OMEGA4_6_1_MODEL_ID,
                    "model_version": OMEGA4_6_1_MODEL_VERSION,
                    "model_id": OMEGA4_6_1_MODEL_ID,
                }
                return 0, 0.0, 0.0, 1.0, info, regime.upper()
            if _approved_notional < _eth_notional - 1e-9:
                _eth_notional = _approved_notional
                _eth_leverage = _eth_notional / max(float(dec.margin_fraction), 1e-12)
        # Real (non-shadow) chop-based soft-sizing, opt-in via
        # FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_ENABLE. Threshold-gated shape (full size
        # below threshold, linear ramp to 0 only above it) -- see
        # docs/model_contracts/eth_leverage_chop_softsize_fresh_forward_20260720.md for the
        # fresh-forward backtest that selected threshold=0.3 over the shadow module's plain
        # max(0, 1-chop_prob) linear shape.
        _entry_blocked_reason = ""
        if bool(FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_ENABLE):
            try:
                _chop_prob = float(
                    self.omega4_6_1_adapter.regime3_current.append(frame)[
                        "regime3_current_sensitive_wide24_chop_prob"
                    ].iloc[-1]
                )
                _chop_thr = float(FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_THRESHOLD)
                if _chop_prob < _chop_thr:
                    _chop_soft_size_mult = 1.0
                else:
                    _chop_soft_size_mult = max(0.0, 1.0 - (_chop_prob - _chop_thr) / (1.0 - _chop_thr))
                _eth_notional = _eth_notional * _chop_soft_size_mult
                _eth_leverage = _eth_notional / max(float(dec.margin_fraction), 1e-12)
                if _eth_notional <= 0.0:
                    _entry_blocked_reason = "omega4_6_1_chop_soft_size_veto"
            except Exception as _chop_size_exc:
                logger.error("SYSTEM omega4_6_1_chop_soft_size=FAILED err=%s", _chop_size_exc, exc_info=True)
                _entry_blocked_reason = "omega4_6_1_chop_soft_size_unavailable"
        if _entry_blocked_reason:
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "omega4_6_1|cash",
                "position_signal": "HOLD",
                "position_reason": _entry_blocked_reason,
                "score": 0.0,
                "conviction": 0.0,
                "owner": "",
                "regime": regime,
                "raw_regime": raw_regime,
                "decision_logic": OMEGA4_6_1_MODEL_ID,
                "model_version": OMEGA4_6_1_MODEL_VERSION,
                "model_id": OMEGA4_6_1_MODEL_ID,
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()
        _final_sizing = finalize_sizing(
            margin_fraction=float(dec.margin_fraction),
            requested_notional=_eth_notional,
            max_leverage=OMEGA4_6_1_LEVERAGE_CAP,
            max_notional=OMEGA4_6_1_NOTIONAL_CAP,
        )
        _eth_notional = _final_sizing.notional
        _eth_leverage = _final_sizing.leverage
        self.owner = OMEGA4_6_1_OWNER
        self.owner_regime = regime
        self.peak_unrealized = 0.0
        self.active_omega4_6_1_source_component = str(dec.source_component)
        self.active_omega4_6_1_take_profit = float(dec.take_profit)
        self.active_omega4_6_1_stop_loss = float(dec.stop_loss)
        self.active_omega4_6_1_notional = _eth_notional
        self.active_omega4_6_1_leverage = _eth_leverage
        self.active_omega4_6_1_quality_score = float(dec.quality_score)
        self.active_omega4_6_1_confidence = float(dec.confidence)
        self.active_omega4_6_1_mfe = 0.0
        self.active_omega4_6_1_mae = 0.0
        self._save_runtime_state()
        info = {
            "agent": "FINAL_GOVERNOR",
            "source": f"omega4_6_1|{dec.source_component}",
            "position_signal": "LONG_ENTRY" if action == 1 else "SHORT_ENTRY",
            "position_reason": "omega4_6_1_entry",
            "score": float(dec.quality_score),
            "conviction": float(dec.confidence),
            "owner": OMEGA4_6_1_OWNER,
            "regime": regime,
            "raw_regime": raw_regime,
            "decision_logic": OMEGA4_6_1_MODEL_ID,
            "model_version": OMEGA4_6_1_MODEL_VERSION,
            "model_id": OMEGA4_6_1_MODEL_ID,
            "model_sleeve": "omega4_6_1_duration_ou_halflife_risk_gate",
            "source_component": str(dec.source_component),
            "take_profit": float(dec.take_profit),
            "stop_loss": float(dec.stop_loss),
            "quality_score": float(dec.quality_score),
            "confidence": float(dec.confidence),
            "sleeve_trace": dict(dec.trace or {}),
        }
        return action, _eth_notional, float(dec.margin_fraction), _eth_leverage, info, regime.upper()

    def _manage_omega4_6_1_position(
        self,
        *,
        meta_router,
        current_price: float,
        frame: pd.DataFrame,
        regime: str,
    ) -> tuple[int, float, float, float, dict, str]:
        pos = str(meta_router.pos or "")
        action_hold = self._action_from_side(pos)
        side = 1 if pos == "LONG" else -1
        hold_bars = int(meta_router.hold_count or 0)
        entry = float(getattr(meta_router, "entry_price", 0.0) or 0.0)
        unrealized = float(meta_router._net_pnl_frac(current_price))
        self.peak_unrealized = max(float(self.peak_unrealized), unrealized)

        try:
            SizingDecision(
                margin_fraction=float(meta_router.position_fraction),
                leverage=float(self.active_omega4_6_1_leverage),
                notional=float(self.active_omega4_6_1_notional),
            )
            state_valid = (
                bool(self.active_omega4_6_1_source_component)
                and np.isfinite(entry)
                and entry > 0.0
                and np.isfinite(self.active_omega4_6_1_take_profit)
                and self.active_omega4_6_1_take_profit > 0.0
                and np.isfinite(self.active_omega4_6_1_stop_loss)
                and self.active_omega4_6_1_stop_loss > 0.0
            )
        except ValueError:
            state_valid = False
        move = 0.0
        prob = 0.0
        if not state_valid:
            close = True
            reason = "omega4_6_1_reconcile_close"
        else:
            move = (current_price - entry) / entry if side > 0 else (entry - current_price) / entry
            self.active_omega4_6_1_mfe = max(float(self.active_omega4_6_1_mfe), move)
            self.active_omega4_6_1_mae = min(float(self.active_omega4_6_1_mae), move)
            # TP/SL barriers use the just-completed bar's high/low (not just its close): a resting
            # TP/SL order fills the instant price touches it intrabar. Both bars used here are
            # already fully closed, so this adds no lookahead -- only the fill itself still lands
            # on the next bar's open, per the existing execution-delay model.
            bar_high = float(frame["high"].iloc[-1])
            bar_low = float(frame["low"].iloc[-1])
            bar_high_move = (bar_high - entry) / entry if side > 0 else (entry - bar_low) / entry
            bar_low_move = (bar_low - entry) / entry if side > 0 else (entry - bar_high) / entry
            should_exit, reason_key, prob = self.omega4_6_1_adapter.evaluate_exit(
                frame,
                source_component=self.active_omega4_6_1_source_component,
                side=side,
                hold_bars=hold_bars,
                unrealized_move=move,
                mfe=float(self.active_omega4_6_1_mfe),
                mae=float(self.active_omega4_6_1_mae),
                notional=float(self.active_omega4_6_1_notional),
                leverage=float(self.active_omega4_6_1_leverage),
                take_profit=float(self.active_omega4_6_1_take_profit),
                stop_loss=float(self.active_omega4_6_1_stop_loss),
                bar_high_move=bar_high_move,
                bar_low_move=bar_low_move,
            )
            close = should_exit
            reason = f"omega4_6_1_{reason_key}" if should_exit else "omega4_6_1_hold"

        if close:
            self.last_exit_bar = self.bar_counter
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": f"omega4_6_1|{reason}",
                "position_signal": "EXIT",
                "position_reason": reason,
                "score": abs(unrealized),
                "conviction": float(self.active_omega4_6_1_quality_score),
                "owner": OMEGA4_6_1_OWNER,
                "regime": regime,
                "decision_logic": OMEGA4_6_1_MODEL_ID,
                "model_version": OMEGA4_6_1_MODEL_VERSION,
                "model_id": OMEGA4_6_1_MODEL_ID,
                "model_sleeve": "omega4_6_1_duration_ou_halflife_risk_gate",
                "source_component": str(self.active_omega4_6_1_source_component),
                "take_profit": float(self.active_omega4_6_1_take_profit),
                "stop_loss": float(self.active_omega4_6_1_stop_loss),
                "quality_score": float(self.active_omega4_6_1_quality_score),
                "confidence": float(self.active_omega4_6_1_confidence),
                "exit_probability": float(prob),
                "raw_unrealized_move": float(move),
            }
            self.owner = ""
            self.owner_regime = ""
            self.peak_unrealized = 0.0
            self._reset_omega4_6_1_position_state()
            self._save_runtime_state()
            return 0, 0.0, 0.0, 1.0, info, regime.upper()

        exposure = float(self.active_omega4_6_1_notional)
        exec_lev = float(self.active_omega4_6_1_leverage)
        fraction = float(exposure / max(exec_lev, 1e-8))
        self._save_runtime_state()
        info = {
            "agent": "FINAL_GOVERNOR",
            "source": "omega4_6_1|hold",
            "position_signal": "HOLD",
            "position_reason": reason,
            "score": abs(unrealized),
            "conviction": float(self.active_omega4_6_1_quality_score),
            "owner": OMEGA4_6_1_OWNER,
            "regime": regime,
            "decision_logic": OMEGA4_6_1_MODEL_ID,
            "model_version": OMEGA4_6_1_MODEL_VERSION,
            "model_id": OMEGA4_6_1_MODEL_ID,
            "model_sleeve": "omega4_6_1_duration_ou_halflife_risk_gate",
            "source_component": str(self.active_omega4_6_1_source_component),
            "take_profit": float(self.active_omega4_6_1_take_profit),
            "stop_loss": float(self.active_omega4_6_1_stop_loss),
            "quality_score": float(self.active_omega4_6_1_quality_score),
            "confidence": float(self.active_omega4_6_1_confidence),
            "exit_probability": float(prob),
            "raw_unrealized_move": float(move),
        }
        return action_hold, exposure, fraction, exec_lev, info, regime.upper()

    def _reset_fully_learned_position_state(self) -> None:
        self.active_fully_learned_take_profit = 0.0
        self.active_fully_learned_stop_loss = 0.0
        self.active_fully_learned_max_hold_bars = 0
        self.active_fully_learned_cooldown_bars = 0
        self.active_fully_learned_quality_score = 0.0
        self.active_fully_learned_confidence = 0.0
        self.active_fully_learned_soft_stop_counter = 0

    def _manage_fully_learned_position(self, *, meta_router, current_price: float, regime: str, frame: pd.DataFrame) -> tuple[int, float, float, float, dict, str]:
        pos = str(meta_router.pos or "")
        action_hold = self._action_from_side(pos)
        unrealized = float(meta_router._net_pnl_frac(current_price))
        self.peak_unrealized = max(float(self.peak_unrealized), unrealized)
        hold_bars = int(meta_router.hold_count or 0)
        close = False
        reason = "fully_learned_hold"
        state_unarmed = (
            self.active_fully_learned_take_profit <= 0.0
            and self.active_fully_learned_stop_loss <= 0.0
            and self.active_fully_learned_max_hold_bars <= 0
        )
        if state_unarmed:
            dec = self._fully_learned_latest_decision(frame)
            current_side = 1 if pos == "LONG" else -1
            if dec is not None and int(dec.side) == current_side:
                self.active_fully_learned_take_profit = float(dec.take_profit)
                self.active_fully_learned_stop_loss = float(dec.stop_loss)
                self.active_fully_learned_max_hold_bars = int(dec.max_hold_bars)
                self.active_fully_learned_cooldown_bars = int(dec.cooldown_bars)
                self.active_fully_learned_quality_score = float(dec.quality_score)
                self.active_fully_learned_confidence = float(dec.confidence)
                self.active_fully_learned_soft_stop_counter = 0
                self._save_runtime_state()
                reason = "fully_learned_repaired_state"
            else:
                close = True
                reason = "fully_learned_reconcile_close"
        latest_row = frame.iloc[-1] if len(frame) else pd.Series(dtype=float)
        side = 1 if pos == "LONG" else -1
        sl_time_mult = self._fully_learned_time_sl_mult(
            hold_bars,
            self._fully_learned_runtime_int("early_bars", 0),
            self._fully_learned_runtime_float("early_sl_mult", 1.0),
        )
        hard_sl = abs(float(self.active_fully_learned_stop_loss)) * self._fully_learned_runtime_float("hard_sl_mult", 1.0) * sl_time_mult
        soft_sl = abs(float(self.active_fully_learned_stop_loss)) * self._fully_learned_runtime_float("soft_sl_mult", 1.0) * sl_time_mult
        regime_bad = self._fully_learned_regime_bad(latest_row)
        flow_bad = self._fully_learned_flow_bad(latest_row, side)
        soft_hit = (
            soft_sl > 0.0
            and hold_bars >= self._fully_learned_runtime_int("soft_min_hold", 1)
            and unrealized <= -soft_sl
            and regime_bad >= self._fully_learned_runtime_float("regime_bad_th", 1.0)
            and flow_bad >= self._fully_learned_runtime_float("flow_bad_th", 0.0)
        )
        if soft_hit:
            self.active_fully_learned_soft_stop_counter += 1
        else:
            self.active_fully_learned_soft_stop_counter = 0
        giveback = (float(self.peak_unrealized) - unrealized) / max(abs(float(self.peak_unrealized)), 1e-12) if self.peak_unrealized > 0.0 else 0.0
        if self.active_fully_learned_take_profit > 0.0 and unrealized >= self.active_fully_learned_take_profit:
            close = True
            reason = "fully_learned_take_profit"
        elif hard_sl > 0.0 and unrealized <= -hard_sl:
            close = True
            reason = "fully_learned_hard_stop_loss"
        elif self.active_fully_learned_soft_stop_counter >= self._fully_learned_runtime_int("soft_persist_bars", 1):
            close = True
            reason = "fully_learned_soft_stop_loss"
        elif (
            hold_bars >= self._fully_learned_runtime_int("giveback_min_hold", 0)
            and self.peak_unrealized >= self._fully_learned_runtime_float("giveback_min_mfe", 1e9)
            and giveback >= self._fully_learned_runtime_float("giveback_trigger", 1.0)
        ):
            close = True
            reason = "fully_learned_giveback_exit"
        elif self.active_fully_learned_max_hold_bars > 0 and hold_bars >= self.active_fully_learned_max_hold_bars:
            close = True
            reason = "fully_learned_max_hold"

        if close:
            self.last_exit_bar = self.bar_counter
            self.fully_learned_cooldown_left = int(max(0, self.active_fully_learned_cooldown_bars))
            if "hard_stop_loss" in reason:
                self.fully_learned_cooldown_left = max(
                    self.fully_learned_cooldown_left,
                    self._fully_learned_runtime_int("cooldown_after_hard_stop", 0),
                )
            elif "soft_stop_loss" in reason:
                self.fully_learned_cooldown_left = max(
                    self.fully_learned_cooldown_left,
                    self._fully_learned_runtime_int("cooldown_after_soft_stop", 0),
                )
            elif "giveback_exit" in reason:
                self.fully_learned_cooldown_left = max(
                    self.fully_learned_cooldown_left,
                    self._fully_learned_runtime_int("cooldown_after_giveback", 0),
                )
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": f"fully_learned|{reason}",
                "position_signal": "EXIT",
                "position_reason": reason,
                "score": abs(unrealized),
                "conviction": float(self.active_fully_learned_quality_score),
                "owner": "fully_learned",
                "regime": regime,
                "decision_logic": "fully_learned_action_execution",
                "model_version": FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION,
                "model_id": FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
                "take_profit": float(self.active_fully_learned_take_profit),
                "stop_loss": float(self.active_fully_learned_stop_loss),
                "max_hold_bars": int(self.active_fully_learned_max_hold_bars),
                "cooldown_bars": int(self.active_fully_learned_cooldown_bars),
                "quality_score": float(self.active_fully_learned_quality_score),
                "confidence": float(self.active_fully_learned_confidence),
                "guard": {
                    "hard_sl": float(hard_sl),
                    "soft_sl": float(soft_sl),
                    "soft_stop_counter": int(self.active_fully_learned_soft_stop_counter),
                    "regime_bad": float(regime_bad),
                    "flow_bad": float(flow_bad),
                    "giveback": float(giveback),
                },
            }
            self.owner = ""
            self.owner_regime = ""
            self.peak_unrealized = 0.0
            self._reset_fully_learned_position_state()
            self._save_runtime_state()
            return 0, 0.0, 0.0, 1.0, info, regime.upper()

        exposure = float(meta_router.current_leverage or 0.0)
        exec_lev = float(meta_router.execution_leverage or 1.0)
        fraction = float(meta_router.position_fraction or min(exposure / max(exec_lev, 1e-8), 1.0))
        info = {
            "agent": "FINAL_GOVERNOR",
            "source": "fully_learned|hold",
            "position_signal": "HOLD",
            "position_reason": reason,
            "score": abs(unrealized),
            "conviction": float(self.active_fully_learned_quality_score),
            "owner": "fully_learned",
            "regime": regime,
            "decision_logic": "fully_learned_action_execution",
            "model_version": FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION,
            "model_id": FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
            "take_profit": float(self.active_fully_learned_take_profit),
            "stop_loss": float(self.active_fully_learned_stop_loss),
            "max_hold_bars": int(self.active_fully_learned_max_hold_bars),
            "cooldown_bars": int(self.active_fully_learned_cooldown_bars),
            "quality_score": float(self.active_fully_learned_quality_score),
            "confidence": float(self.active_fully_learned_confidence),
            "guard": {
                "hard_sl": float(hard_sl),
                "soft_sl": float(soft_sl),
                "soft_stop_counter": int(self.active_fully_learned_soft_stop_counter),
                "regime_bad": float(regime_bad),
                "flow_bad": float(flow_bad),
                "giveback": float(giveback),
            },
        }
        return action_hold, exposure, fraction, exec_lev, info, regime.upper()

    def _decide_fully_learned_entry(self, frame: pd.DataFrame, *, regime: str, raw_regime: str) -> tuple[int, float, float, float, dict, str] | None:
        dec = self._fully_learned_latest_decision(frame)
        if dec is None:
            if bool(getattr(self, "fully_learned_contract_blocked", False)):
                info = {
                    "agent": "FINAL_GOVERNOR",
                    "source": "fully_learned|feature_contract_block",
                    "position_signal": "HOLD",
                    "position_reason": "fully_learned_feature_contract_block",
                    "score": 0.0,
                    "conviction": 0.0,
                    "owner": "",
                    "regime": regime,
                    "decision_logic": "fully_learned_action_execution",
                    "model_version": FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION,
                    "model_id": FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
                    "sleeve_trace": {
                        "decision_logic": "fully_learned_action_execution",
                        "model_id": FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
                        "feature_contract": dict(getattr(self, "last_fully_learned_contract_audit", {}) or {}),
                    },
                }
                return 0, 0.0, 0.0, 1.0, info, regime.upper()
            return None
        trace = {
            "decision_logic": "fully_learned_action_execution",
            "model": os.path.basename(str(self.fully_learned_policy_path)),
            "model_id": FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
            "model_version": FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION,
            "regime": regime,
            "raw_regime": raw_regime,
            "regime_predictor": dict(frame.attrs.get("regime_predictor_trace", {}) or {}),
            "clean_regime4_sticky": dict(frame.attrs.get("clean_regime4_sticky_trace", {}) or {}),
            "tp_sl_action_score": dict(frame.attrs.get("tp_sl_action_score_trace", {}) or {}),
            "feature_contract": dict(getattr(self, "last_fully_learned_contract_audit", {}) or {}),
            "selection": dict(getattr(self, "last_fully_learned_selection_trace", {}) or {}),
            "scale_runtime": dict(self.fully_learned_scale_runtime or {}),
            "action": int(dec.action),
            "side": int(dec.side),
            "notional_exposure": float(dec.notional_exposure),
            "leverage": float(dec.leverage),
            "position_fraction": float(dec.position_fraction),
            "take_profit": float(dec.take_profit),
            "stop_loss": float(dec.stop_loss),
            "max_hold_bars": int(dec.max_hold_bars),
            "cooldown_bars": int(dec.cooldown_bars),
            "quality_score": float(dec.quality_score),
            "confidence": float(dec.confidence),
        }
        if self.fully_learned_cooldown_left > 0:
            self.fully_learned_cooldown_left -= 1
            self._save_runtime_state()
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "fully_learned|cooldown",
                "position_signal": "HOLD",
                "position_reason": "fully_learned_cooldown",
                "score": float(dec.quality_score),
                "conviction": float(dec.confidence),
                "owner": "",
                "regime": regime,
                "decision_logic": "fully_learned_action_execution",
                "model_version": FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION,
                "model_id": FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
                "sleeve_trace": trace,
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()

        if int(dec.action) == int(FULLY_LEARNED_ACTION_CASH) or int(dec.side) == 0:
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "fully_learned|cash",
                "position_signal": "HOLD",
                "position_reason": "fully_learned_cash",
                "score": float(dec.quality_score),
                "conviction": float(dec.confidence),
                "owner": "",
                "regime": regime,
                "decision_logic": "fully_learned_action_execution",
                "model_version": FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION,
                "model_id": FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
                "sleeve_trace": trace,
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()

        action = 1 if int(dec.action) == int(FULLY_LEARNED_ACTION_LONG) else 2
        same_side_gap = self._fully_learned_runtime_int("same_side_entry_gap", 0)
        if (
            same_side_gap > 0
            and int(dec.side) == int(self.last_fully_learned_entry_side)
            and (int(self.bar_counter) - int(self.last_fully_learned_entry_bar)) <= same_side_gap
        ):
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": "fully_learned|same_side_entry_gap",
                "position_signal": "HOLD",
                "position_reason": "fully_learned_same_side_entry_gap",
                "score": float(dec.quality_score),
                "conviction": float(dec.confidence),
                "owner": "",
                "regime": regime,
                "decision_logic": "fully_learned_action_execution",
                "model_version": FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION,
                "model_id": FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
                "sleeve_trace": trace,
            }
            return 0, 0.0, 0.0, 1.0, info, regime.upper()
        self.owner = "fully_learned"
        self.owner_regime = regime
        self.peak_unrealized = 0.0
        self.active_fully_learned_take_profit = float(dec.take_profit)
        self.active_fully_learned_stop_loss = float(dec.stop_loss)
        self.active_fully_learned_max_hold_bars = int(dec.max_hold_bars)
        self.active_fully_learned_cooldown_bars = int(dec.cooldown_bars)
        self.active_fully_learned_quality_score = float(dec.quality_score)
        self.active_fully_learned_confidence = float(dec.confidence)
        self.active_fully_learned_soft_stop_counter = 0
        self.last_fully_learned_entry_side = int(dec.side)
        self.last_fully_learned_entry_bar = int(self.bar_counter)
        self._save_runtime_state()
        info = {
            "agent": "FINAL_GOVERNOR",
            "source": "fully_learned|entry",
            "position_signal": "LONG_ENTRY" if action == 1 else "SHORT_ENTRY",
            "position_reason": "fully_learned_entry",
            "score": float(dec.quality_score),
            "conviction": float(dec.confidence),
            "owner": "fully_learned",
            "regime": regime,
            "decision_logic": "fully_learned_action_execution",
            "model_version": FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION,
            "model_id": FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
            "take_profit": float(dec.take_profit),
            "stop_loss": float(dec.stop_loss),
            "max_hold_bars": int(dec.max_hold_bars),
            "cooldown_bars": int(dec.cooldown_bars),
            "quality_score": float(dec.quality_score),
            "confidence": float(dec.confidence),
            "sleeve_trace": trace,
        }
        return (
            action,
            float(dec.notional_exposure),
            float(dec.position_fraction),
            float(dec.leverage),
            info,
            regime.upper(),
        )

    def _decide_fully_learned_alpha3_fallthrough_entry(
        self,
        frame: pd.DataFrame,
        *,
        meta_router,
        regime: str,
        raw_regime: str,
    ) -> tuple[int, float, float, float, dict, str]:
        parent_dec = pd.Series(
            {
                "action": int(FULLY_LEARNED_ACTION_CASH),
                "side": 0,
                "quality_score": 0.0,
                "confidence": 0.0,
            }
        )
        parent_trace = {
            "decision_logic": "alpha7_1_01965_cash_fallthrough",
            "model_version": FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION,
            "model_id": FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
            "selection": dict(getattr(self, "last_fully_learned_selection_trace", {}) or {}),
            "runtime_config": dict(self.fully_learned_runtime_config or {}),
            "contract": "alpha7_cash_direct_to_v31_deep_alpha",
        }
        v31_decision = self._v31_deep_alpha_entry_decision(
            frame,
            meta_router=meta_router,
            regime=regime,
            raw_regime=raw_regime,
            parent_dec=parent_dec,
            parent_trace=parent_trace,
        )
        if v31_decision is not None:
            return v31_decision
        info = {
            "agent": "FINAL_GOVERNOR",
            "source": "alpha7_1_01965|cash|v31_unavailable",
            "position_signal": "HOLD",
            "position_reason": "alpha7_1_01965_cash_v31_no_entry",
            "score": 0.0,
            "conviction": 0.0,
            "owner": "",
            "regime": regime,
            "decision_logic": "alpha7_1_01965_cash_fallthrough",
            "model_version": FINAL_GOVERNOR_FULLY_LEARNED_MODEL_VERSION,
            "model_id": FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID,
            "sleeve_trace": parent_trace,
        }
        return 0, 0.0, 0.0, 1.0, info, regime.upper()

    def _sniper_env(self, frame: pd.DataFrame, meta_router, current_price: float) -> ExpertMetaTradingEnv:
        env = ExpertMetaTradingEnv(
            frame,
            self.manifest,
            self._repo_path(FINAL_GOVERNOR_MANIFEST_PATH),
            self.active_regimes,
            self.device,
            phase="val",
            window_bars=min(self.window_bars, len(frame)),
            action_mode="all",
            manual_close_mode="always",
            gate_mode="soft",
            enable_fade_expert=True,
            fade_min_abs_action=0.35,
            target_min_trades_per_day=5.0,
            target_max_trades_per_day=20.0,
            enable_risk_engine=False,
        )
        env.reset(0)
        last = len(frame) - 1
        env.trade_env.current_step = last
        env.trade_env.end_step = last
        if meta_router.pos in {"LONG", "SHORT"}:
            env.trade_env.pos = meta_router.pos
            env.trade_env.entry_price = float(meta_router.entry_price or current_price or 0.0)
            env.trade_env.hold_count = int(meta_router.hold_count or 0)
            env.trade_env.entry_idx = max(0, last - int(meta_router.hold_count or 0))
            env.trade_env.current_notional_exposure = float(meta_router.current_leverage or 0.0)
            env.trade_env.current_margin_fraction = float(meta_router.position_fraction or 0.0)
            env.trade_env.current_leverage = float(meta_router.execution_leverage or 1.0)
            env.trade_env.unrealized_pnl = float(meta_router._net_pnl_frac(current_price))
            env.position_regime = self.owner_regime or self._raw_regime_from_row(frame.iloc[-1]) or "normal"
        return env

    def _load_runtime_state(self) -> None:
        path = str(getattr(self, "runtime_state_path", "") or "")
        if not path or not os.path.exists(path):
            return
        if os.path.getsize(path) <= 0:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.macro_lockout_signal = int(data.get("macro_lockout_signal", 0) or 0)
            self.macro_lockout_bars_left = int(data.get("macro_lockout_bars_left", 0) or 0)
            self.active_macro_take_profit = float(data.get("active_macro_take_profit", self.active_macro_take_profit) or 0.0)
            self.active_macro_stop_loss = float(data.get("active_macro_stop_loss", self.active_macro_stop_loss) or 0.0)
            self.active_macro_max_hold_bars = int(data.get("active_macro_max_hold_bars", self.active_macro_max_hold_bars) or 0)
            self.active_macro_quality_score = float(data.get("active_macro_quality_score", self.active_macro_quality_score) or 0.0)
            self.owner = str(data.get("owner", self.owner) or "")
            self.owner_regime = str(data.get("owner_regime", self.owner_regime) or "")
            self.active_fully_learned_take_profit = float(data.get("active_fully_learned_take_profit", self.active_fully_learned_take_profit) or 0.0)
            self.active_fully_learned_stop_loss = float(data.get("active_fully_learned_stop_loss", self.active_fully_learned_stop_loss) or 0.0)
            self.active_fully_learned_max_hold_bars = int(data.get("active_fully_learned_max_hold_bars", self.active_fully_learned_max_hold_bars) or 0)
            self.active_fully_learned_cooldown_bars = int(data.get("active_fully_learned_cooldown_bars", self.active_fully_learned_cooldown_bars) or 0)
            self.active_fully_learned_quality_score = float(data.get("active_fully_learned_quality_score", self.active_fully_learned_quality_score) or 0.0)
            self.active_fully_learned_confidence = float(data.get("active_fully_learned_confidence", self.active_fully_learned_confidence) or 0.0)
            self.active_fully_learned_soft_stop_counter = int(
                data.get("active_fully_learned_soft_stop_counter", self.active_fully_learned_soft_stop_counter) or 0
            )
            self.last_fully_learned_entry_side = int(
                data.get("last_fully_learned_entry_side", self.last_fully_learned_entry_side) or 0
            )
            self.last_fully_learned_entry_bar = int(
                data.get("last_fully_learned_entry_bar", self.last_fully_learned_entry_bar) or -10**9
            )
            self.fully_learned_cooldown_left = int(data.get("fully_learned_cooldown_left", self.fully_learned_cooldown_left) or 0)
            omega5_runtime_active = str(data.get("owner", self.owner) or "") == OMEGA5_OWNER
            if omega5_runtime_active:
                required_omega5_state_fields = [
                    "active_omega5_take_profit",
                    "active_omega5_stop_loss",
                    "active_omega5_max_hold_bars",
                    "active_omega5_quality_score",
                    "active_omega5_confidence",
                    "active_omega5_notional",
                    "active_omega5_leverage",
                    "active_omega5_roundtrip_cost",
                    "active_omega5_source_exit_reason",
                    "active_omega5_source_exit_price_move",
                    "active_omega5_sizing_trace",
                ]
                missing_omega5_state_fields = [
                    field for field in required_omega5_state_fields if field not in data
                ]
                if missing_omega5_state_fields:
                    raise RuntimeError(
                        "Omega5 runtime state contract mismatch: "
                        f"missing={missing_omega5_state_fields}"
                    )
                trace = data["active_omega5_sizing_trace"]
                if not isinstance(trace, dict) or not trace:
                    raise RuntimeError("Omega5 runtime state contract mismatch: missing active_omega5_sizing_trace")
                self.active_omega5_take_profit = float(data["active_omega5_take_profit"] or 0.0)
                self.active_omega5_stop_loss = float(data["active_omega5_stop_loss"] or 0.0)
                self.active_omega5_max_hold_bars = int(data["active_omega5_max_hold_bars"] or 0)
                self.active_omega5_quality_score = float(data["active_omega5_quality_score"] or 0.0)
                self.active_omega5_confidence = float(data["active_omega5_confidence"] or 0.0)
                self.active_omega5_notional = float(data["active_omega5_notional"] or 0.0)
                self.active_omega5_leverage = float(data["active_omega5_leverage"] or 1.0)
                self.active_omega5_parent_exit_timestamp = str(data.get("active_omega5_parent_exit_timestamp", "") or "")
                self.active_omega5_roundtrip_cost = float(data["active_omega5_roundtrip_cost"] or 0.0)
                self.active_omega5_source_exit_reason = str(data["active_omega5_source_exit_reason"] or "")
                self.active_omega5_source_exit_price_move = float(data["active_omega5_source_exit_price_move"] or 0.0)
                self.active_omega5_sizing_trace = dict(trace)
            else:
                self.active_omega5_take_profit = float(data.get("active_omega5_take_profit", self.active_omega5_take_profit) or 0.0)
                self.active_omega5_stop_loss = float(data.get("active_omega5_stop_loss", self.active_omega5_stop_loss) or 0.0)
                self.active_omega5_max_hold_bars = int(data.get("active_omega5_max_hold_bars", self.active_omega5_max_hold_bars) or 0)
                self.active_omega5_quality_score = float(data.get("active_omega5_quality_score", self.active_omega5_quality_score) or 0.0)
                self.active_omega5_confidence = float(data.get("active_omega5_confidence", self.active_omega5_confidence) or 0.0)
                self.active_omega5_notional = float(data.get("active_omega5_notional", self.active_omega5_notional) or 0.0)
                self.active_omega5_leverage = float(data.get("active_omega5_leverage", self.active_omega5_leverage) or 1.0)
                self.active_omega5_parent_exit_timestamp = str(
                    data.get("active_omega5_parent_exit_timestamp", self.active_omega5_parent_exit_timestamp) or ""
                )
                self.active_omega5_roundtrip_cost = float(
                    data.get("active_omega5_roundtrip_cost", self.active_omega5_roundtrip_cost) or 0.0
                )
                self.active_omega5_source_exit_reason = str(
                    data.get("active_omega5_source_exit_reason", self.active_omega5_source_exit_reason) or ""
                )
                self.active_omega5_source_exit_price_move = float(
                    data.get("active_omega5_source_exit_price_move", self.active_omega5_source_exit_price_move) or 0.0
                )
                trace = data.get("active_omega5_sizing_trace")
                self.active_omega5_sizing_trace = dict(trace) if isinstance(trace, dict) else {}
            self.last_omega5_entry_side = int(data.get("last_omega5_entry_side", self.last_omega5_entry_side) or 0)
            self.last_omega5_entry_bar = int(data.get("last_omega5_entry_bar", self.last_omega5_entry_bar) or -10**9)
            self.active_omega4_6_1_source_component = str(
                data.get("active_omega4_6_1_source_component", self.active_omega4_6_1_source_component) or ""
            )
            self.active_omega4_6_1_take_profit = float(
                data.get("active_omega4_6_1_take_profit", self.active_omega4_6_1_take_profit) or 0.0
            )
            self.active_omega4_6_1_stop_loss = float(
                data.get("active_omega4_6_1_stop_loss", self.active_omega4_6_1_stop_loss) or 0.0
            )
            self.active_omega4_6_1_notional = float(
                data.get("active_omega4_6_1_notional", self.active_omega4_6_1_notional) or 0.0
            )
            self.active_omega4_6_1_leverage = float(
                data.get("active_omega4_6_1_leverage", self.active_omega4_6_1_leverage) or 1.0
            )
            self.active_omega4_6_1_quality_score = float(
                data.get("active_omega4_6_1_quality_score", self.active_omega4_6_1_quality_score) or 0.0
            )
            self.active_omega4_6_1_confidence = float(
                data.get("active_omega4_6_1_confidence", self.active_omega4_6_1_confidence) or 0.0
            )
            self.active_omega4_6_1_mfe = float(data.get("active_omega4_6_1_mfe", self.active_omega4_6_1_mfe) or 0.0)
            self.active_omega4_6_1_mae = float(data.get("active_omega4_6_1_mae", self.active_omega4_6_1_mae) or 0.0)
            self.active_omega4_6_1_tp_order_id = str(
                data.get("active_omega4_6_1_tp_order_id", self.active_omega4_6_1_tp_order_id) or ""
            )
            self.active_omega4_6_1_sl_order_id = str(
                data.get("active_omega4_6_1_sl_order_id", self.active_omega4_6_1_sl_order_id) or ""
            )
            self.active_v13_1_take_profit = float(data.get("active_v13_1_take_profit", self.active_v13_1_take_profit) or 0.0)
            self.active_v13_1_stop_loss = float(data.get("active_v13_1_stop_loss", self.active_v13_1_stop_loss) or 0.0)
            self.active_v13_1_max_hold_bars = int(data.get("active_v13_1_max_hold_bars", self.active_v13_1_max_hold_bars) or 0)
            self.active_v13_1_cooldown_bars = int(data.get("active_v13_1_cooldown_bars", self.active_v13_1_cooldown_bars) or 0)
            self.active_v13_1_quality_score = float(data.get("active_v13_1_quality_score", self.active_v13_1_quality_score) or 0.0)
            self.active_v13_1_confidence = float(data.get("active_v13_1_confidence", self.active_v13_1_confidence) or 0.0)
            self.active_v13_1_notional = float(data.get("active_v13_1_notional", self.active_v13_1_notional) or 0.0)
            self.active_v13_1_leverage = float(data.get("active_v13_1_leverage", self.active_v13_1_leverage) or 1.0)
            self.active_v13_1_lane = str(data.get("active_v13_1_lane", self.active_v13_1_lane) or "")
            self.active_v13_1_probability = float(data.get("active_v13_1_probability", self.active_v13_1_probability) or 0.0)
            self.active_v13_1_threshold = float(data.get("active_v13_1_threshold", self.active_v13_1_threshold) or 0.0)
            self.active_v13_1_regime = str(data.get("active_v13_1_regime", self.active_v13_1_regime) or "")
            self.active_v13_1_regime_multiplier = float(data.get("active_v13_1_regime_multiplier", self.active_v13_1_regime_multiplier) or 1.0)
            self.v13_1_cooldown_left = int(data.get("v13_1_cooldown_left", self.v13_1_cooldown_left) or 0)
            self.active_lifecycle_v1_base_notional = float(data.get("active_lifecycle_v1_base_notional", self.active_lifecycle_v1_base_notional) or 0.0)
            self.active_lifecycle_v1_effective_notional = float(data.get("active_lifecycle_v1_effective_notional", self.active_lifecycle_v1_effective_notional) or 0.0)
            self.active_lifecycle_v1_leverage = float(data.get("active_lifecycle_v1_leverage", self.active_lifecycle_v1_leverage) or 1.0)
            self.active_lifecycle_v1_cooldown_bars = int(data.get("active_lifecycle_v1_cooldown_bars", self.active_lifecycle_v1_cooldown_bars) or 0)
            self.active_lifecycle_v1_quality_score = float(data.get("active_lifecycle_v1_quality_score", self.active_lifecycle_v1_quality_score) or 0.0)
            self.active_lifecycle_v1_confidence = float(data.get("active_lifecycle_v1_confidence", self.active_lifecycle_v1_confidence) or 0.0)
            self.active_lifecycle_v1_entry_bucket = str(data.get("active_lifecycle_v1_entry_bucket", self.active_lifecycle_v1_entry_bucket) or "")
            self.active_lifecycle_v1_entry_hazard = float(data.get("active_lifecycle_v1_entry_hazard", self.active_lifecycle_v1_entry_hazard) or 0.0)
            self.active_lifecycle_v1_entry_support = int(data.get("active_lifecycle_v1_entry_support", self.active_lifecycle_v1_entry_support) or 0)
            self.active_lifecycle_v1_edit = str(data.get("active_lifecycle_v1_edit", self.active_lifecycle_v1_edit) or "")
            self.active_lifecycle_v1_take_profit = float(data.get("active_lifecycle_v1_take_profit", self.active_lifecycle_v1_take_profit) or 0.0)
            self.active_lifecycle_v1_stop_loss = float(data.get("active_lifecycle_v1_stop_loss", self.active_lifecycle_v1_stop_loss) or 0.0)
            self.active_lifecycle_v1_max_hold_bars = int(data.get("active_lifecycle_v1_max_hold_bars", self.active_lifecycle_v1_max_hold_bars) or 0)
            self.active_lifecycle_v1_jackpot_added = bool(data.get("active_lifecycle_v1_jackpot_added", self.active_lifecycle_v1_jackpot_added))
            self.active_lifecycle_v1_mae_unrealized = float(
                data.get("active_lifecycle_v1_mae_unrealized", self.active_lifecycle_v1_mae_unrealized) or 0.0
            )
            self.active_lifecycle_v1_v21_sleeve = str(data.get("active_lifecycle_v1_v21_sleeve", self.active_lifecycle_v1_v21_sleeve) or "")
            self.active_lifecycle_v1_v21_stop_raw = float(data.get("active_lifecycle_v1_v21_stop_raw", self.active_lifecycle_v1_v21_stop_raw) or 999.0)
            self.active_lifecycle_v1_v21_peak_raw = float(data.get("active_lifecycle_v1_v21_peak_raw", self.active_lifecycle_v1_v21_peak_raw) or -1e9)
            raw_stop_reasons = data.get("active_lifecycle_v1_v21_stop_reasons", self.active_lifecycle_v1_v21_stop_reasons)
            if isinstance(raw_stop_reasons, list):
                self.active_lifecycle_v1_v21_stop_reasons = [str(x) for x in raw_stop_reasons]
            elif raw_stop_reasons:
                self.active_lifecycle_v1_v21_stop_reasons = [str(raw_stop_reasons)]
            self.active_lifecycle_v1_scout_model_version = str(
                data.get("active_lifecycle_v1_scout_model_version", self.active_lifecycle_v1_scout_model_version) or ""
            )
            self.active_lifecycle_v1_scout_model_id = str(
                data.get("active_lifecycle_v1_scout_model_id", self.active_lifecycle_v1_scout_model_id) or ""
            )
            self.active_lifecycle_v1_scout_model_path = str(
                data.get("active_lifecycle_v1_scout_model_path", self.active_lifecycle_v1_scout_model_path) or ""
            )
            self.active_lifecycle_v1_scout_prob = float(
                data.get("active_lifecycle_v1_scout_prob", self.active_lifecycle_v1_scout_prob) or 0.0
            )
            self.active_lifecycle_v1_scout_frac = float(
                data.get("active_lifecycle_v1_scout_frac", self.active_lifecycle_v1_scout_frac) or 0.0
            )
            self.active_lifecycle_v1_scout_probability_threshold = float(
                data.get(
                    "active_lifecycle_v1_scout_probability_threshold",
                    self.active_lifecycle_v1_scout_probability_threshold,
                )
                or 0.0
            )
            self.active_lifecycle_v1_scout_cost_pass = bool(
                data.get("active_lifecycle_v1_scout_cost_pass", self.active_lifecycle_v1_scout_cost_pass)
            )
            self.active_v31_entry_edge = float(data.get("active_v31_entry_edge", self.active_v31_entry_edge) or 0.0)
            self.active_v31_entry_margin = float(data.get("active_v31_entry_margin", self.active_v31_entry_margin) or 0.0)
            self.active_v31_entry_vol_anchor = float(data.get("active_v31_entry_vol_anchor", self.active_v31_entry_vol_anchor) or 0.0)
            self.active_v31_entry_q_long = float(data.get("active_v31_entry_q_long", self.active_v31_entry_q_long) or 0.0)
            self.active_v31_entry_q_short = float(data.get("active_v31_entry_q_short", self.active_v31_entry_q_short) or 0.0)
            self.active_v31_entry_q_long_raw = float(data.get("active_v31_entry_q_long_raw", self.active_v31_entry_q_long_raw) or 0.0)
            self.active_v31_entry_q_short_raw = float(data.get("active_v31_entry_q_short_raw", self.active_v31_entry_q_short_raw) or 0.0)
            self.active_v31_entry_selected_side = str(data.get("active_v31_entry_selected_side", self.active_v31_entry_selected_side) or "")
            self.active_v31_entry_guard_reason = str(data.get("active_v31_entry_guard_reason", self.active_v31_entry_guard_reason) or "")
            self.active_lifecycle_v1_conformal_core_notional = float(
                data.get(
                    "active_lifecycle_v1_conformal_core_notional",
                    self.active_lifecycle_v1_conformal_core_notional,
                )
                or 0.0
            )
            self.active_lifecycle_v1_conformal_sleeve_notional = float(
                data.get(
                    "active_lifecycle_v1_conformal_sleeve_notional",
                    self.active_lifecycle_v1_conformal_sleeve_notional,
                )
                or 0.0
            )
            self.active_lifecycle_v1_conformal_sleeve_exit_bars = int(
                data.get(
                    "active_lifecycle_v1_conformal_sleeve_exit_bars",
                    self.active_lifecycle_v1_conformal_sleeve_exit_bars,
                )
                or 0
            )
            self.active_lifecycle_v1_conformal_sleeve_action = str(
                data.get(
                    "active_lifecycle_v1_conformal_sleeve_action",
                    self.active_lifecycle_v1_conformal_sleeve_action,
                )
                or ""
            )
            self.lifecycle_v1_cooldown_left = int(data.get("lifecycle_v1_cooldown_left", self.lifecycle_v1_cooldown_left) or 0)
            self.v31_deep_cooldown_left = int(data.get("v31_deep_cooldown_left", self.v31_deep_cooldown_left) or 0)
            self.ddh2_fallback_dd_block_active = bool(
                data.get("ddh2_fallback_dd_block_active", self.ddh2_fallback_dd_block_active)
            )
        except Exception as e:
            raise RuntimeError(f"final_governor_runtime_state_load_failed:{path}") from e

    def _save_runtime_state(self) -> None:
        path = str(getattr(self, "runtime_state_path", "") or "")
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                "macro_lockout_signal": int(self.macro_lockout_signal),
                "macro_lockout_bars_left": int(self.macro_lockout_bars_left),
                "active_macro_take_profit": float(self.active_macro_take_profit),
                "active_macro_stop_loss": float(self.active_macro_stop_loss),
                "active_macro_max_hold_bars": int(self.active_macro_max_hold_bars),
                "active_macro_quality_score": float(self.active_macro_quality_score),
                "owner": str(self.owner),
                "owner_regime": str(self.owner_regime),
                "active_fully_learned_take_profit": float(self.active_fully_learned_take_profit),
                "active_fully_learned_stop_loss": float(self.active_fully_learned_stop_loss),
                "active_fully_learned_max_hold_bars": int(self.active_fully_learned_max_hold_bars),
                "active_fully_learned_cooldown_bars": int(self.active_fully_learned_cooldown_bars),
                "active_fully_learned_quality_score": float(self.active_fully_learned_quality_score),
                "active_fully_learned_confidence": float(self.active_fully_learned_confidence),
                "active_fully_learned_soft_stop_counter": int(self.active_fully_learned_soft_stop_counter),
                "last_fully_learned_entry_side": int(self.last_fully_learned_entry_side),
                "last_fully_learned_entry_bar": int(self.last_fully_learned_entry_bar),
                "fully_learned_cooldown_left": int(self.fully_learned_cooldown_left),
                "active_omega5_take_profit": float(self.active_omega5_take_profit),
                "active_omega5_stop_loss": float(self.active_omega5_stop_loss),
                "active_omega5_max_hold_bars": int(self.active_omega5_max_hold_bars),
                "active_omega5_quality_score": float(self.active_omega5_quality_score),
                "active_omega5_confidence": float(self.active_omega5_confidence),
                "active_omega5_notional": float(self.active_omega5_notional),
                "active_omega5_leverage": float(self.active_omega5_leverage),
                "active_omega5_parent_exit_timestamp": str(self.active_omega5_parent_exit_timestamp or ""),
                "active_omega5_roundtrip_cost": float(self.active_omega5_roundtrip_cost),
                "active_omega5_source_exit_reason": str(self.active_omega5_source_exit_reason or ""),
                "active_omega5_source_exit_price_move": float(self.active_omega5_source_exit_price_move),
                "active_omega5_sizing_trace": GovernorPositionRouter._journal_jsonable(
                    dict(self.active_omega5_sizing_trace or {})
                ),
                "last_omega5_entry_side": int(self.last_omega5_entry_side),
                "last_omega5_entry_bar": int(self.last_omega5_entry_bar),
                "active_omega4_6_1_source_component": str(self.active_omega4_6_1_source_component),
                "active_omega4_6_1_take_profit": float(self.active_omega4_6_1_take_profit),
                "active_omega4_6_1_stop_loss": float(self.active_omega4_6_1_stop_loss),
                "active_omega4_6_1_notional": float(self.active_omega4_6_1_notional),
                "active_omega4_6_1_leverage": float(self.active_omega4_6_1_leverage),
                "active_omega4_6_1_quality_score": float(self.active_omega4_6_1_quality_score),
                "active_omega4_6_1_confidence": float(self.active_omega4_6_1_confidence),
                "active_omega4_6_1_mfe": float(self.active_omega4_6_1_mfe),
                "active_omega4_6_1_mae": float(self.active_omega4_6_1_mae),
                "active_omega4_6_1_tp_order_id": str(self.active_omega4_6_1_tp_order_id or ""),
                "active_omega4_6_1_sl_order_id": str(self.active_omega4_6_1_sl_order_id or ""),
                "active_v13_1_take_profit": float(self.active_v13_1_take_profit),
                "active_v13_1_stop_loss": float(self.active_v13_1_stop_loss),
                "active_v13_1_max_hold_bars": int(self.active_v13_1_max_hold_bars),
                "active_v13_1_cooldown_bars": int(self.active_v13_1_cooldown_bars),
                "active_v13_1_quality_score": float(self.active_v13_1_quality_score),
                "active_v13_1_confidence": float(self.active_v13_1_confidence),
                "active_v13_1_notional": float(self.active_v13_1_notional),
                "active_v13_1_leverage": float(self.active_v13_1_leverage),
                "active_v13_1_lane": str(self.active_v13_1_lane),
                "active_v13_1_probability": float(self.active_v13_1_probability),
                "active_v13_1_threshold": float(self.active_v13_1_threshold),
                "active_v13_1_regime": str(self.active_v13_1_regime),
                "active_v13_1_regime_multiplier": float(self.active_v13_1_regime_multiplier),
                "v13_1_cooldown_left": int(self.v13_1_cooldown_left),
                "active_lifecycle_v1_base_notional": float(self.active_lifecycle_v1_base_notional),
                "active_lifecycle_v1_effective_notional": float(self.active_lifecycle_v1_effective_notional),
                "active_lifecycle_v1_leverage": float(self.active_lifecycle_v1_leverage),
                "active_lifecycle_v1_cooldown_bars": int(self.active_lifecycle_v1_cooldown_bars),
                "active_lifecycle_v1_quality_score": float(self.active_lifecycle_v1_quality_score),
                "active_lifecycle_v1_confidence": float(self.active_lifecycle_v1_confidence),
                "active_lifecycle_v1_entry_bucket": str(self.active_lifecycle_v1_entry_bucket),
                "active_lifecycle_v1_entry_hazard": float(self.active_lifecycle_v1_entry_hazard),
                "active_lifecycle_v1_entry_support": int(self.active_lifecycle_v1_entry_support),
                "active_lifecycle_v1_edit": str(self.active_lifecycle_v1_edit),
                "active_lifecycle_v1_take_profit": float(self.active_lifecycle_v1_take_profit),
                "active_lifecycle_v1_stop_loss": float(self.active_lifecycle_v1_stop_loss),
                "active_lifecycle_v1_max_hold_bars": int(self.active_lifecycle_v1_max_hold_bars),
                "active_lifecycle_v1_jackpot_added": bool(self.active_lifecycle_v1_jackpot_added),
                "active_lifecycle_v1_mae_unrealized": float(self.active_lifecycle_v1_mae_unrealized),
                "active_lifecycle_v1_v21_sleeve": str(self.active_lifecycle_v1_v21_sleeve),
                "active_lifecycle_v1_v21_stop_raw": float(self.active_lifecycle_v1_v21_stop_raw),
                "active_lifecycle_v1_v21_peak_raw": float(self.active_lifecycle_v1_v21_peak_raw),
                "active_lifecycle_v1_v21_stop_reasons": list(self.active_lifecycle_v1_v21_stop_reasons or []),
                "active_lifecycle_v1_scout_model_version": str(self.active_lifecycle_v1_scout_model_version),
                "active_lifecycle_v1_scout_model_id": str(self.active_lifecycle_v1_scout_model_id),
                "active_lifecycle_v1_scout_model_path": str(self.active_lifecycle_v1_scout_model_path),
                "active_lifecycle_v1_scout_prob": float(self.active_lifecycle_v1_scout_prob),
                "active_lifecycle_v1_scout_frac": float(self.active_lifecycle_v1_scout_frac),
                "active_lifecycle_v1_scout_probability_threshold": float(self.active_lifecycle_v1_scout_probability_threshold),
                "active_lifecycle_v1_scout_cost_pass": bool(self.active_lifecycle_v1_scout_cost_pass),
                "active_v31_entry_edge": float(self.active_v31_entry_edge),
                "active_v31_entry_margin": float(self.active_v31_entry_margin),
                "active_v31_entry_vol_anchor": float(self.active_v31_entry_vol_anchor),
                "active_v31_entry_q_long": float(self.active_v31_entry_q_long),
                "active_v31_entry_q_short": float(self.active_v31_entry_q_short),
                "active_v31_entry_q_long_raw": float(self.active_v31_entry_q_long_raw),
                "active_v31_entry_q_short_raw": float(self.active_v31_entry_q_short_raw),
                "active_v31_entry_selected_side": str(self.active_v31_entry_selected_side),
                "active_v31_entry_guard_reason": str(self.active_v31_entry_guard_reason),
                "active_lifecycle_v1_conformal_core_notional": float(self.active_lifecycle_v1_conformal_core_notional),
                "active_lifecycle_v1_conformal_sleeve_notional": float(self.active_lifecycle_v1_conformal_sleeve_notional),
                "active_lifecycle_v1_conformal_sleeve_exit_bars": int(self.active_lifecycle_v1_conformal_sleeve_exit_bars),
                "active_lifecycle_v1_conformal_sleeve_action": str(self.active_lifecycle_v1_conformal_sleeve_action),
                "lifecycle_v1_cooldown_left": int(self.lifecycle_v1_cooldown_left),
                "v31_deep_cooldown_left": int(self.v31_deep_cooldown_left),
                "ddh2_fallback_dd_block_active": bool(self.ddh2_fallback_dd_block_active),
                "saved_at": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
            }
            _atomic_write_json(path, payload)
        except Exception as e:
            raise RuntimeError(f"final_governor_runtime_state_save_failed:{path}") from e

    def _recover_omega5_state_from_open_journal(self, meta_router, regime: str) -> bool:
        if self.omega5_adapter is None or meta_router.pos not in {"LONG", "SHORT"}:
            return False
        trade_id = str(getattr(meta_router, "open_trade_id", "") or "")
        if not trade_id:
            return False
        try:
            rows = _load_trade_journal_rows(TRADE_JOURNAL_PATH)
        except Exception:
            rows = []
        open_row = None
        closed = False
        for row in rows:
            if str(row.get("trade_id", "") or "") != trade_id:
                continue
            kind = str(row.get("kind", "") or "").upper()
            if kind == "OPEN":
                open_row = row
                closed = False
            elif kind == "CLOSE":
                closed = True
        if open_row is None or closed:
            return False
        source = str(open_row.get("source", "") or "")
        model_id = str(open_row.get("model_id", "") or "")
        if not (source.startswith(f"{OMEGA5_OWNER}|") or model_id == OMEGA5_MODEL_ID):
            return False
        identity_changed = bool(
            str(getattr(meta_router, "open_model_id", "") or "") != model_id
            or str(getattr(meta_router, "open_source", "") or "") != source
        )
        meta_router._set_open_model_identity(open_row, source=source)
        meta_router._save_live_state()

        required_contract_fields = [
            "notional_exposure",
            "execution_leverage",
            "effective_take_profit",
            "effective_stop_loss",
            "max_hold_bars",
            "omega5_source_roundtrip_cost",
            "omega5_source_exit_reason",
            "omega5_source_exit_price_move",
            "omega5_sizing_trace",
        ]
        missing_contract_fields = [field for field in required_contract_fields if field not in open_row]
        if missing_contract_fields:
            raise RuntimeError(
                "Omega5 open journal is missing risk contract fields: "
                f"trade_id={trade_id} missing={missing_contract_fields}"
            )
        exposure = float(open_row["notional_exposure"])
        exec_lev = float(open_row["execution_leverage"])
        tp = float(open_row["effective_take_profit"])
        sl = float(open_row["effective_stop_loss"])
        max_hold_bars = int(open_row["max_hold_bars"])
        if exposure <= 0.0 or exec_lev <= 0.0 or tp <= 0.0 or sl <= 0.0 or max_hold_bars <= 0:
            raise RuntimeError(
                "Omega5 open journal has invalid risk contract fields: "
                f"trade_id={trade_id} exposure={exposure} leverage={exec_lev} "
                f"tp={tp} sl={sl} max_hold_bars={max_hold_bars}"
            )
        changed = bool(
            self.owner != OMEGA5_OWNER
            or abs(float(self.active_omega5_take_profit or 0.0) - float(tp)) > 1e-12
            or abs(float(self.active_omega5_stop_loss or 0.0) - float(sl)) > 1e-12
            or abs(float(self.active_omega5_notional or 0.0) - float(exposure)) > 1e-12
            or identity_changed
        )
        self.owner = OMEGA5_OWNER
        self.owner_regime = str(regime or self.owner_regime or "")
        self.active_omega5_take_profit = float(tp)
        self.active_omega5_stop_loss = float(sl)
        self.active_omega5_max_hold_bars = int(max_hold_bars)
        self.active_omega5_quality_score = float(
            open_row.get("quality_score", open_row.get("score", self.active_omega5_quality_score)) or 0.0
        )
        self.active_omega5_confidence = float(
            open_row.get("confidence", open_row.get("conviction", self.active_omega5_confidence)) or 0.0
        )
        self.active_omega5_notional = float(exposure)
        self.active_omega5_leverage = float(exec_lev)
        self.active_omega5_roundtrip_cost = float(
            open_row.get("omega5_source_roundtrip_cost", self.active_omega5_roundtrip_cost) or 0.0
        )
        self.active_omega5_source_exit_reason = str(
            open_row.get("omega5_source_exit_reason", self.active_omega5_source_exit_reason) or ""
        )
        self.active_omega5_source_exit_price_move = float(
            open_row.get("omega5_source_exit_price_move", self.active_omega5_source_exit_price_move) or 0.0
        )
        trace = open_row["omega5_sizing_trace"]
        if not isinstance(trace, dict) or not trace:
            raise RuntimeError(
                "Omega5 open journal is missing sizing trace contract: "
                f"trade_id={trade_id}"
            )
        parent_trace = trace.get("parent_trace")
        has_policy_row = (
            isinstance(parent_trace, dict)
            and int(parent_trace.get("source_parent_policy_row", -1) or -1) >= 0
        )
        has_live_native_artifact = (
            isinstance(parent_trace, dict)
            and bool(parent_trace.get("source_parent_live_native_adapter", False))
            and bool(parent_trace.get("source_parent_predictive_artifact"))
            and bool(parent_trace.get("source_parent_component_bundle"))
            and bool(parent_trace.get("source_parent_component_sidecar"))
        )
        if not isinstance(parent_trace, dict) or not (has_policy_row or has_live_native_artifact):
            raise RuntimeError(
                "Omega5 open journal is missing source-parent policy provenance: "
                f"trade_id={trade_id}"
            )
        self.active_omega5_sizing_trace = dict(trace)
        self.last_omega5_entry_side = 1 if str(meta_router.pos).upper() == "LONG" else -1
        self._reset_lifecycle_v1_position_state()
        if changed:
            self._save_runtime_state()
            logger.warning(
                "SYSTEM omega5_state_recovered_from_open_journal trade_id=%s tp=%.6f sl=%.6f exposure=%.4f leverage=%.2f",
                trade_id,
                float(self.active_omega5_take_profit),
                float(self.active_omega5_stop_loss),
                float(exposure),
                float(exec_lev),
            )
        return True

    def _recover_lifecycle_v1_state_from_open_journal(self, meta_router, regime: str) -> bool:
        if meta_router.pos not in {"LONG", "SHORT"}:
            return False
        if (
            self.active_lifecycle_v1_effective_notional > 0.0
            and (
                self.active_lifecycle_v1_take_profit > 0.0
                or self.active_lifecycle_v1_stop_loss > 0.0
                or self.active_lifecycle_v1_max_hold_bars > 0
            )
        ):
            return False
        trade_id = str(getattr(meta_router, "open_trade_id", "") or "")
        if not trade_id:
            return False
        try:
            rows = _load_trade_journal_rows(TRADE_JOURNAL_PATH)
        except Exception:
            rows = []
        open_row = None
        closed = False
        for row in rows:
            if str(row.get("trade_id", "") or "") != trade_id:
                continue
            kind = str(row.get("kind", "") or "").upper()
            if kind == "OPEN":
                open_row = row
                closed = False
            elif kind == "CLOSE":
                closed = True
        if open_row is None or closed:
            return False
        source = str(open_row.get("source", "") or "")
        model_id = str(open_row.get("model_id", "") or "")
        if source.startswith(f"{OMEGA5_OWNER}|") or model_id == OMEGA5_MODEL_ID:
            return False
        if source.startswith(f"{OMEGA4_6_1_OWNER}|") or model_id == OMEGA4_6_1_MODEL_ID:
            return False
        model_sleeve = str(open_row.get("model_sleeve", "") or "")
        is_lifecycle = (
            "lifecycle" in source
            or "v31" in source
            or model_sleeve in {"deep_alpha", "jackpot", "core"}
            or float(open_row.get("take_profit", 0.0) or 0.0) > 0.0
            or float(open_row.get("stop_loss", 0.0) or 0.0) > 0.0
            or int(open_row.get("max_hold_bars", 0) or 0) > 0
        )
        if not is_lifecycle:
            return False
        meta_router._set_open_model_identity(open_row, source=source)
        meta_router._save_live_state()

        exposure = float(
            open_row.get(
                "notional_exposure",
                open_row.get("total_exposure", getattr(meta_router, "current_leverage", 0.0)),
            )
            or 0.0
        )
        if exposure <= 0.0:
            exposure = float(getattr(meta_router, "current_leverage", 0.0) or 0.0)
        exec_lev = float(open_row.get("execution_leverage", getattr(meta_router, "execution_leverage", 1.0)) or 1.0)
        scout_model_version = str(open_row.get("scout_model_version", "") or "")
        is_v31 = (
            "v31" in source
            or model_sleeve == "deep_alpha"
            or str(open_row.get("v31_selected_side", "") or "")
            or float(open_row.get("v31_edge", 0.0) or 0.0) != 0.0
        )
        if is_v31:
            scout_model_version = "V31"
        elif not scout_model_version:
            scout_model_version = str(open_row.get("model_version", "") or "") or "V21.2"

        self.owner = "lifecycle_v1"
        self.owner_regime = str(regime or self.owner_regime or "")
        self.active_lifecycle_v1_base_notional = float(exposure)
        self.active_lifecycle_v1_effective_notional = float(exposure)
        self.active_lifecycle_v1_leverage = float(exec_lev)
        self.active_lifecycle_v1_quality_score = float(open_row.get("v31_edge", open_row.get("scout_prob", 0.0)) or 0.0)
        self.active_lifecycle_v1_confidence = float(open_row.get("v31_margin", 0.0) or 0.0)
        self.active_lifecycle_v1_entry_bucket = "v31_deep_alpha" if is_v31 else str(model_sleeve or "lifecycle_v1")
        self.active_lifecycle_v1_entry_hazard = 0.0
        self.active_lifecycle_v1_entry_support = 0
        self.active_lifecycle_v1_edit = str(model_sleeve or ("v31_deep_alpha" if is_v31 else "lifecycle_v1"))
        self.active_lifecycle_v1_take_profit = float(open_row.get("take_profit", 0.0) or 0.0)
        self.active_lifecycle_v1_stop_loss = float(open_row.get("stop_loss", 0.0) or 0.0)
        self.active_lifecycle_v1_max_hold_bars = int(open_row.get("max_hold_bars", 0) or 0)
        self.active_lifecycle_v1_jackpot_added = bool(is_v31 or open_row.get("active_lifecycle_v1_jackpot_added", False))
        self.active_lifecycle_v1_mae_unrealized = min(
            float(self.active_lifecycle_v1_mae_unrealized or 0.0),
            float(open_row.get("mae_unrealized", 0.0) or 0.0),
        )
        self.active_lifecycle_v1_v21_sleeve = str(model_sleeve or ("deep_alpha" if is_v31 else ""))
        self.active_lifecycle_v1_v21_stop_raw = float(open_row.get("v21_stop_raw", 999.0) or 999.0)
        self.active_lifecycle_v1_v21_peak_raw = float(open_row.get("v21_peak_raw", -1e9) or -1e9)
        self.active_lifecycle_v1_v21_stop_reasons = list(open_row.get("v21_stop_reasons", []) or [])
        if is_v31 and not self.active_lifecycle_v1_v21_stop_reasons:
            self.active_lifecycle_v1_v21_stop_reasons = ["v31_rule_exit_overlay"]
        self.active_lifecycle_v1_scout_model_version = str(scout_model_version)
        self.active_lifecycle_v1_scout_model_id = str(
            open_row.get(
                "scout_model_id",
                "hf_v13_frozen_v27_rule_exit_overlay_v31_20260511" if is_v31 else open_row.get("model_id", ""),
            )
            or ""
        )
        self.active_lifecycle_v1_scout_model_path = str(open_row.get("model_path", "") or "")
        self.active_lifecycle_v1_scout_prob = float(open_row.get("scout_prob", open_row.get("v31_edge", 0.0)) or 0.0)
        self.active_lifecycle_v1_scout_frac = float(open_row.get("scout_frac", exposure) or 0.0)
        self.active_lifecycle_v1_scout_probability_threshold = float(open_row.get("scout_probability_threshold", 0.0) or 0.0)
        self.active_lifecycle_v1_scout_cost_pass = bool(open_row.get("scout_cost_pass", False))
        self.active_v31_entry_edge = float(open_row.get("v31_edge", self.active_v31_entry_edge) or 0.0)
        self.active_v31_entry_margin = float(open_row.get("v31_margin", self.active_v31_entry_margin) or 0.0)
        self.active_v31_entry_vol_anchor = float(open_row.get("entry_vol_anchor", self.active_v31_entry_vol_anchor) or 0.0)
        self.active_v31_entry_q_long = float(open_row.get("v31_q_long", self.active_v31_entry_q_long) or 0.0)
        self.active_v31_entry_q_short = float(open_row.get("v31_q_short", self.active_v31_entry_q_short) or 0.0)
        self.active_v31_entry_q_long_raw = float(open_row.get("v31_q_long_raw", self.active_v31_entry_q_long_raw) or 0.0)
        self.active_v31_entry_q_short_raw = float(open_row.get("v31_q_short_raw", self.active_v31_entry_q_short_raw) or 0.0)
        self.active_v31_entry_selected_side = str(open_row.get("v31_selected_side", self.active_v31_entry_selected_side) or "")
        self.active_v31_entry_guard_reason = str(open_row.get("v31_guard_reason", self.active_v31_entry_guard_reason) or "")
        self._save_runtime_state()
        logger.warning(
            "SYSTEM lifecycle_v1_state_recovered_from_open_journal trade_id=%s scout=%s tp=%.6f sl=%.6f max_hold=%d exposure=%.4f",
            trade_id,
            str(self.active_lifecycle_v1_scout_model_version),
            float(self.active_lifecycle_v1_take_profit),
            float(self.active_lifecycle_v1_stop_loss),
            int(self.active_lifecycle_v1_max_hold_bars),
            float(self.active_lifecycle_v1_effective_notional),
        )
        return True

    def _sync_owner(self, meta_router, regime: str) -> None:
        if meta_router.pos not in {"LONG", "SHORT"}:
            changed = bool(
                self.owner
                or self.owner_regime
                or self.active_v13_1_notional > 0.0
                or self.active_v13_1_lane
                or self.active_lifecycle_v1_effective_notional > 0.0
                or self.active_lifecycle_v1_base_notional > 0.0
                or self.active_lifecycle_v1_edit
                or self.active_fully_learned_take_profit > 0.0
                or self.active_fully_learned_stop_loss > 0.0
                or self.active_fully_learned_max_hold_bars > 0
                or self.active_omega5_take_profit > 0.0
                or self.active_omega5_stop_loss > 0.0
                or self.active_omega5_max_hold_bars > 0
                or self.active_omega5_notional > 0.0
                or self.active_omega4_6_1_take_profit > 0.0
                or self.active_omega4_6_1_stop_loss > 0.0
                or self.active_omega4_6_1_notional > 0.0
            )
            self.owner = ""
            self.owner_regime = ""
            self.peak_unrealized = 0.0
            self._reset_v13_1_position_state()
            self._reset_lifecycle_v1_position_state()
            self._reset_fully_learned_position_state()
            self._reset_omega5_position_state()
            self._reset_omega4_6_1_position_state()
            if changed:
                self._save_runtime_state()
            return
        if self._recover_omega5_state_from_open_journal(meta_router, regime):
            return
        self._recover_lifecycle_v1_state_from_open_journal(meta_router, regime)
        if self.owner:
            return
        if self._v13_1_available() and self.active_v13_1_notional > 0.0:
            self.owner = "disabled_v13_1"
            self.owner_regime = str(self.active_v13_1_regime or regime)
            self.peak_unrealized = 0.0
            return
        if self._lifecycle_v1_available() and self.active_lifecycle_v1_effective_notional > 0.0:
            self.owner = "lifecycle_v1"
            self.owner_regime = regime
            self.peak_unrealized = 0.0
            return
        if (
            self.omega4_6_1_adapter is not None
            and (
                self.active_omega4_6_1_take_profit > 0.0
                or self.active_omega4_6_1_stop_loss > 0.0
                or self.active_omega4_6_1_notional > 0.0
            )
        ):
            self.owner = OMEGA4_6_1_OWNER
            self.owner_regime = regime
            self.peak_unrealized = 0.0
            return
        if (
            self.fully_learned_policy_bundle is not None
            and (
                self.active_fully_learned_take_profit > 0.0
                or self.active_fully_learned_stop_loss > 0.0
                or self.active_fully_learned_max_hold_bars > 0
            )
        ):
            self.owner = "fully_learned"
            self.owner_regime = regime
            self.peak_unrealized = 0.0
            return
        if (
            self.omega5_adapter is not None
            and (
                self.active_omega5_take_profit > 0.0
                or self.active_omega5_stop_loss > 0.0
                or self.active_omega5_max_hold_bars > 0
                or self.active_omega5_notional > 0.0
            )
        ):
            self.owner = OMEGA5_OWNER
            self.owner_regime = regime
            self.peak_unrealized = 0.0
            return
        if self.omega5_adapter is not None:
            self.owner = OMEGA5_OWNER
            self.owner_regime = regime
            self.peak_unrealized = 0.0
            return
        if self.fully_learned_policy_bundle is not None:
            self.owner = "fully_learned"
            self.owner_regime = regime
            self.peak_unrealized = 0.0
            return
        if FINAL_GOVERNOR_MACRO_ENABLE:
            self.owner = "macro"
            self.owner_regime = "macro"
            self.peak_unrealized = 0.0
            return
        self.owner = "trend" if regime in self.TREND_REGIMES else "micro"
        self.owner_regime = regime
        self.peak_unrealized = 0.0

    def _arm_macro_lockout(self, signal: int) -> None:
        sig = int(np.sign(signal))
        if sig == 0:
            return
        if bool(self.macro_cfg.lockout_until_signal_change):
            self.macro_lockout_signal = sig
        self.macro_lockout_bars_left = max(int(self.macro_lockout_bars_left), int(self.macro_cfg.lockout_bars))
        self._save_runtime_state()

    def _arm_macro_close_lockout(self, signal: int) -> None:
        if not bool(getattr(self.macro_cfg, "lockout_on_any_close", False)):
            return
        sig = int(np.sign(signal))
        if sig != 0 and bool(self.macro_cfg.lockout_until_signal_change):
            self.macro_lockout_signal = sig
        self.macro_lockout_bars_left = max(int(self.macro_lockout_bars_left), int(self.macro_cfg.lockout_bars))
        self._save_runtime_state()

    def _macro_lockout_active(self, signal: int) -> bool:
        sig = int(np.sign(signal))
        changed = False
        if self.macro_lockout_bars_left > 0:
            self.macro_lockout_bars_left -= 1
            changed = True
        if self.macro_lockout_signal and (sig == 0 or sig != self.macro_lockout_signal):
            self.macro_lockout_signal = 0
            self.macro_lockout_bars_left = 0
            changed = True
        if changed:
            self._save_runtime_state()
        return bool(
            (self.macro_lockout_signal and sig == self.macro_lockout_signal)
            or self.macro_lockout_bars_left > 0
        )

    @staticmethod
    def _macro_fixed_risk(macro) -> dict:
        return {
            "notional_exposure": float(macro.notional_exposure),
            "leverage": float(macro.leverage),
            "position_fraction": float(macro.position_fraction),
            "take_profit": float(FINAL_GOVERNOR_MACRO_TAKE_PROFIT),
            "stop_loss": float(FINAL_GOVERNOR_MACRO_STOP_LOSS),
            "max_hold_bars": 0,
            "quality_score": 0.0,
            "confidence": 0.0,
            "model": "fixed_macro_config",
        }

    def _macro_learned_risk(self, frame: pd.DataFrame, macro) -> dict:
        if self.execution_policy_bundle is None:
            return self._macro_fixed_risk(macro)
        side = 1 if str(macro.side).upper() == "LONG" else -1
        learned = predict_learned_execution(
            self.execution_policy_bundle,
            frame.iloc[-1],
            source="macro",
            side=side,
            macro_momentum=float(macro.momentum),
        )
        risk = learned.to_risk_decision()
        if bool(FINAL_GOVERNOR_EXECUTION_POLICY_IGNORE_MAX_HOLD):
            risk["max_hold_bars"] = 0
        if bool(FINAL_GOVERNOR_EXECUTION_POLICY_QUALITY_OVERLAY):
            q = float(risk.get("quality_score", 0.0) or 0.0)
            if q >= float(FINAL_GOVERNOR_EXECUTION_POLICY_TAIL_QUALITY):
                risk["notional_exposure"] = max(float(risk.get("notional_exposure", 0.0) or 0.0), 3.0)
                risk["leverage"] = max(float(risk.get("leverage", 1.0) or 1.0), 5.0)
                risk["take_profit"] = max(float(risk.get("take_profit", 0.0) or 0.0), 1.25)
            elif q < float(FINAL_GOVERNOR_EXECUTION_POLICY_LOW_QUALITY):
                risk["notional_exposure"] = min(float(risk.get("notional_exposure", 0.0) or 0.0), 1.00)
                risk["leverage"] = min(float(risk.get("leverage", 1.0) or 1.0), 3.0)
                risk["take_profit"] = min(float(risk.get("take_profit", 0.0) or 0.0), 0.10)
            else:
                risk["notional_exposure"] = min(float(risk.get("notional_exposure", 0.0) or 0.0), 2.00)
                risk["leverage"] = min(float(risk.get("leverage", 1.0) or 1.0), 4.0)
                risk["take_profit"] = min(max(float(risk.get("take_profit", 0.0) or 0.0), 0.10), 0.35)
            risk["position_fraction"] = float(
                np.clip(float(risk["notional_exposure"]) / max(float(risk["leverage"]), 1e-8), 0.0, 1.0)
            )
        risk["model"] = os.path.basename(str(self.execution_policy_path))
        return risk

    def _manage_open_position(
        self,
        *,
        meta_router,
        current_price: float,
        regime: str,
        trend_proba: np.ndarray,
        trend_classes: list[int],
        micro_proba: np.ndarray,
        micro_classes: list[int],
        sniper_env: ExpertMetaTradingEnv | None,
        frame: pd.DataFrame,
        ) -> tuple[int, float, float, float, dict]:
        pos = str(meta_router.pos or "")
        action_hold = self._action_from_side(pos)
        net_unrealized = float(meta_router._net_pnl_frac(current_price))
        raw_unrealized = _price_return_frac(pos, float(meta_router.entry_price or 0.0), float(current_price or 0.0))
        unrealized = raw_unrealized if self.owner in {"trend", "micro"} else net_unrealized
        self.peak_unrealized = max(float(self.peak_unrealized), unrealized)
        hold_bars = int(meta_router.hold_count or 0)
        idx = -1
        close = False
        reason = ""

        if self.owner == "macro":
            macro = macro_trend_decision(frame, self.macro_cfg)
            close = (not macro.allow_entry) or str(macro.side).upper() != pos
            reason = "macro_signal_close" if close else "macro_hold"
            take_profit = float(self.active_macro_take_profit or self.macro_cfg.take_profit)
            stop_loss = float(self.active_macro_stop_loss or self.macro_cfg.stop_loss)
            max_hold = int(self.active_macro_max_hold_bars or 0)
            if not close and take_profit > 0.0 and unrealized >= take_profit:
                close = True
                reason = "macro_take_profit"
                self._arm_macro_lockout(int(macro.signal))
            elif not close and stop_loss > 0.0 and unrealized <= -abs(stop_loss):
                close = True
                reason = "macro_stop_loss"
                self._arm_macro_lockout(int(macro.signal))
            elif not close and max_hold > 0 and hold_bars >= max_hold:
                close = True
                reason = "macro_max_hold"
            elif (
                not close
                and self.macro_cfg.trailing_arm > 0.0
                and self.macro_cfg.trailing_gap > 0.0
                and self.peak_unrealized >= float(self.macro_cfg.trailing_arm)
                and unrealized <= self.peak_unrealized - float(self.macro_cfg.trailing_gap)
            ):
                close = True
                reason = "macro_trailing_take_profit"
                self._arm_macro_lockout(int(macro.signal))
            if close:
                self._arm_macro_close_lockout(int(macro.signal))
        elif self.owner == "trend":
            if not bool(FINAL_GOVERNOR_TREND_ENABLE):
                close = True
                reason = "trend_disabled_close"
            else:
                no_p = _trend_class_prob(trend_proba, trend_classes, idx, 0)
                long_p = _trend_class_prob(trend_proba, trend_classes, idx, 1)
                short_p = _trend_class_prob(trend_proba, trend_classes, idx, 2)
                close = (
                    regime not in self.TREND_REGIMES
                    or unrealized <= -self.trend_cfg.stop_loss
                    or unrealized >= self.trend_cfg.take_profit
                    or (
                        self.peak_unrealized >= self.trend_cfg.trailing_stop * 1.15
                        and unrealized <= self.peak_unrealized - self.trend_cfg.trailing_stop
                    )
                    or hold_bars >= int(self.trend_cfg.max_hold_bars)
                    or (pos == "LONG" and short_p >= self.trend_cfg.entry_confidence + 0.10)
                    or (pos == "SHORT" and long_p >= self.trend_cfg.entry_confidence + 0.10)
                    or (pos == "LONG" and no_p >= long_p + 0.08)
                    or (pos == "SHORT" and no_p >= short_p + 0.08)
                )
                reason = "trend_close" if close else "trend_hold"
        elif self.owner == "micro":
            if not bool(FINAL_GOVERNOR_MICRO_ENABLE):
                close = True
                reason = "micro_disabled_close"
            else:
                long_p = self._class_prob(micro_proba, micro_classes, idx, 1)
                short_p = self._class_prob(micro_proba, micro_classes, idx, 2)
                close = (
                    unrealized <= -self.micro_cfg.stop_loss
                    or unrealized >= self.micro_cfg.take_profit
                    or (
                        self.peak_unrealized >= self.micro_cfg.trailing_stop * 1.15
                        and unrealized <= self.peak_unrealized - self.micro_cfg.trailing_stop
                    )
                    or hold_bars >= int(self.micro_cfg.max_hold_bars)
                    or (pos == "LONG" and short_p >= self.micro_cfg.entry_confidence + 0.12)
                    or (pos == "SHORT" and long_p >= self.micro_cfg.entry_confidence + 0.12)
                )
                reason = "micro_close" if close else "micro_hold"
        else:
            if not bool(FINAL_GOVERNOR_SNIPER_ENABLE):
                close = True
                reason = "sniper_disabled_close"
            else:
                if sniper_env is None:
                    close = True
                    reason = "sniper_env_missing_close"
                else:
                    _, sniper_action = _final_sniper_action(sniper_env, self.sniper_actor, self.sniper_ckpt, self.device)
                    close = int(sniper_action) == int(FINAL_ACT_CLOSE)
                    reason = "sniper_close" if close else "sniper_hold"

        if close:
            self.last_exit_bar = self.bar_counter
            info = {
                "agent": "FINAL_GOVERNOR",
                "source": f"{self.owner or 'sniper'}|{reason}",
                "position_signal": "EXIT",
                "position_reason": reason,
                "score": abs(unrealized),
                "conviction": abs(unrealized),
                "owner": self.owner or "sniper",
                "regime": regime,
                "decision_logic": "ddh2_v22_1_sniper_trend_micro_full_1x" if self.ddh2_ensemble_enabled else "oos_parity_sniper_trend_micro",
                "raw_unrealized": float(raw_unrealized),
                "net_unrealized": float(net_unrealized),
            }
            self.owner = ""
            self.owner_regime = ""
            self.peak_unrealized = 0.0
            self.active_macro_take_profit = float(FINAL_GOVERNOR_MACRO_TAKE_PROFIT)
            self.active_macro_stop_loss = float(FINAL_GOVERNOR_MACRO_STOP_LOSS)
            self.active_macro_max_hold_bars = 0
            self.active_macro_quality_score = 0.0
            self._save_runtime_state()
            return 0, 0.0, 0.0, 1.0, info

        exposure = float(meta_router.current_leverage or self.notional)
        fraction = float(meta_router.position_fraction or min(exposure / max(self.leverage, 1e-8), 1.0))
        exec_lev = float(meta_router.execution_leverage or self.leverage)
        info = {
            "agent": "FINAL_GOVERNOR",
            "source": f"{self.owner or 'sniper'}|{reason}",
            "position_signal": "HOLD",
            "position_reason": reason,
            "score": abs(unrealized),
            "conviction": abs(unrealized),
            "owner": self.owner or "sniper",
            "regime": regime,
            "decision_logic": "ddh2_v22_1_sniper_trend_micro_full_1x" if self.ddh2_ensemble_enabled else "oos_parity_sniper_trend_micro",
            "raw_unrealized": float(raw_unrealized),
            "net_unrealized": float(net_unrealized),
            "take_profit": float(self.active_macro_take_profit) if self.owner == "macro" else None,
            "stop_loss": float(self.active_macro_stop_loss) if self.owner == "macro" else None,
            "quality_score": float(self.active_macro_quality_score) if self.owner == "macro" else None,
        }
        return action_hold, exposure, fraction, exec_lev, info

    def decide(
        self,
        *,
        processed_df: pd.DataFrame,
        meta_router,
        current_price: float,
        m7_last: dict | None,
        trend_signal: dict | None,
    ) -> tuple[int, float, float, float, dict, str]:
        self.bar_counter += 1
        frame = self._prepare_frame(processed_df, m7_last=m7_last, trend_signal=trend_signal)
        raw_regime = self._raw_regime_from_row(frame.iloc[-1])
        regime = str(raw_regime or "normal").lower()
        self._sync_owner(meta_router, regime)
        if self.omega4_6_1_adapter is not None:
            if meta_router.pos in {"LONG", "SHORT"} and self.owner == OMEGA4_6_1_OWNER:
                return self._manage_omega4_6_1_position(
                    meta_router=meta_router,
                    current_price=current_price,
                    frame=frame,
                    regime=regime,
                )
            if meta_router.pos not in {"LONG", "SHORT"}:
                return self._decide_omega4_6_1_entry(
                    frame,
                    regime=regime,
                    raw_regime=raw_regime,
                )
        if self.omega5_adapter is not None:
            if meta_router.pos in {"LONG", "SHORT"} and self.owner == OMEGA5_OWNER:
                return self._manage_omega5_position(
                    meta_router=meta_router,
                    current_price=current_price,
                    frame=frame,
                    regime=regime,
                )
            if meta_router.pos not in {"LONG", "SHORT"}:
                return self._decide_omega5_entry(
                    frame,
                    regime=regime,
                    raw_regime=raw_regime,
                )
        if self.fully_learned_policy_bundle is not None:
            if meta_router.pos in {"LONG", "SHORT"} and self.owner == "fully_learned":
                return self._manage_fully_learned_position(
                    meta_router=meta_router,
                    current_price=current_price,
                    regime=regime,
                    frame=frame,
                )
            if meta_router.pos not in {"LONG", "SHORT"}:
                fully_learned_decision = self._decide_fully_learned_entry(
                    frame,
                    regime=regime,
                    raw_regime=raw_regime,
                )
                if fully_learned_decision is not None:
                    return fully_learned_decision
                if self._fully_learned_cash_falls_through_to_alpha3():
                    mode = str(dict(getattr(self, "last_fully_learned_selection_trace", {}) or {}).get("mode", ""))
                    if mode.endswith("alpha3_fallthrough"):
                        return self._decide_fully_learned_alpha3_fallthrough_entry(
                            frame,
                            meta_router=meta_router,
                            regime=regime,
                            raw_regime=raw_regime,
                        )
        if meta_router.pos in {"LONG", "SHORT"} and self.owner == "disabled_v13_1":
            return self._manage_v13_1_position(
                meta_router=meta_router,
                current_price=current_price,
                regime=regime,
                frame=frame,
            )
        if self._v13_1_available() and meta_router.pos not in {"LONG", "SHORT"}:
            v13_1_decision = self._decide_v13_1_entry(
                frame,
                meta_router=meta_router,
                regime=regime,
                raw_regime=raw_regime,
            )
            if v13_1_decision is not None:
                return v13_1_decision
        if self._lifecycle_v21_pure_active() and meta_router.pos in {"LONG", "SHORT"} and self.owner != "lifecycle_v1":
            self.owner = "lifecycle_v1"
            self.owner_regime = regime
            self.peak_unrealized = 0.0

        if self._lifecycle_v1_available():
            if meta_router.pos in {"LONG", "SHORT"} and self.owner == "lifecycle_v1":
                action, exposure, fraction, exec_lev, info, regime_name = self._manage_lifecycle_v1_position(
                    meta_router=meta_router,
                    current_price=current_price,
                    regime=regime,
                    frame=frame,
                )
                return action, exposure, fraction, exec_lev, info, regime_name
            if meta_router.pos not in {"LONG", "SHORT"}:
                lifecycle_decision = self._decide_lifecycle_v1_entry(
                    frame,
                    meta_router=meta_router,
                    regime=regime,
                    raw_regime=raw_regime,
                )
                if lifecycle_decision is not None:
                    return lifecycle_decision
                if self._lifecycle_v21_pure_active():
                    info = {
                        "agent": "FINAL_GOVERNOR",
                        "source": "alpha3|model_unavailable",
                        "position_signal": "HOLD",
                        "position_reason": "alpha3_lifecycle_latest_unavailable",
                        "score": 0.0,
                        "conviction": 0.0,
                        "owner": "",
                        "regime": regime,
                        "decision_logic": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                        "model_version": "Alpha3",
                        "model_id": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                        "sleeve_trace": {
                            "decision_logic": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
                            "lifecycle_v1_available": bool(self._lifecycle_v1_available()),
                            "v21_2_jackpot_available": bool(self._v21_2_jackpot_available()),
                            "v31_available": bool(self._v31_available()),
                            "alpha2_1_available": bool(self._alpha2_1_available()),
                        },
                    }
                    return 0, 0.0, 0.0, 1.0, info, regime.upper()

        info = {
            "agent": "FINAL_GOVERNOR",
            "source": "alpha3|model_unavailable",
            "position_signal": "HOLD",
            "position_reason": "alpha3_lifecycle_latest_unavailable",
            "score": 0.0,
            "conviction": 0.0,
            "owner": "",
            "regime": regime,
            "decision_logic": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
            "model_version": "Alpha3",
            "model_id": FINAL_GOVERNOR_ALPHA3_MODEL_ID,
            "sleeve_trace": {
                "alpha2_1_available": bool(self._alpha2_1_available()),
                "lifecycle_v1_available": bool(self._lifecycle_v1_available()),
                "v21_2_jackpot_available": bool(self._v21_2_jackpot_available()),
                "v31_available": bool(self._v31_available()),
            },
        }
        return 0, 0.0, 0.0, 1.0, info, regime.upper()



# ════════════════════════════════════════════════════════════════
# 3-B. Omega4.6.1 multi-asset shadow decisions
# ════════════════════════════════════════════════════════════════
def _omega461_shadow_price(side: str, entry: float, move: float, *, take_profit: bool) -> float:
    entry_n = float(entry or 0.0)
    move_n = float(move or 0.0)
    side_u = str(side or "").upper()
    if entry_n <= 0.0 or move_n <= 0.0 or side_u not in {"LONG", "SHORT"}:
        return 0.0
    if side_u == "LONG":
        return float(entry_n * (1.0 + move_n)) if take_profit else float(entry_n * max(0.0, 1.0 - move_n))
    return float(entry_n * max(0.0, 1.0 - move_n)) if take_profit else float(entry_n * (1.0 + move_n))


def _omega461_shadow_state(
    *,
    asset_key: str,
    cfg: dict,
    decision,
    current_price: float,
    updated_at,
    status: str = "ok",
    error: str = "",
    regime: str = "",
) -> dict:
    label = str(cfg.get("label", asset_key.upper()))
    symbol = str(cfg.get("symbol", ""))
    account_symbol = str(cfg.get("account_symbol", ""))
    ts = pd.Timestamp(updated_at).isoformat() if updated_at is not None else pd.Timestamp.utcnow().isoformat()
    if decision is None:
        final_action = 0
        side_text = "NONE"
        source_component = ""
        margin_fraction = 0.0
        leverage = 1.0
        notional = 0.0
        tp = 0.0
        sl = 0.0
        quality = 0.0
        confidence = 0.0
        trace = {}
        reason = "omega4_6_1_cash" if status == "ok" else "omega4_6_1_shadow_error"
    else:
        final_action = 1 if int(decision.side) > 0 else 2
        side_text = "LONG" if int(decision.side) > 0 else "SHORT"
        source_component = str(decision.source_component)
        margin_fraction = float(decision.margin_fraction)
        leverage = float(decision.leverage)
        notional = float(decision.notional_exposure)
        tp = float(decision.take_profit)
        sl = float(decision.stop_loss)
        quality = float(decision.quality_score)
        confidence = float(decision.confidence)
        trace = dict(decision.trace or {})
        reason = "omega4_6_1_shadow_entry"
    entry = float(current_price or 0.0) if final_action else 0.0
    tp_price = _omega461_shadow_price(side_text, entry, tp, take_profit=True)
    sl_price = _omega461_shadow_price(side_text, entry, sl, take_profit=False)
    return {
        "schema_version": "live.dashboard.asset_decision.v1",
        "asset": str(asset_key),
        "label": label,
        "symbol": symbol,
        "account": {"symbol": account_symbol},
        "updated_at": ts,
        "cycle_timestamp_kst": str(pd.Timestamp(updated_at) + pd.Timedelta(hours=9)) if updated_at is not None else "",
        "price": float(current_price or 0.0),
        "last_price": float(current_price or 0.0),
        "status": str(status),
        "error": str(error or ""),
        "regime": str(regime or ""),
        "position": {
            "current": side_text,
            "entry_price": float(entry),
            "decision_at": ts,
            "position_fraction": float(margin_fraction),
            "margin_fraction": float(margin_fraction),
            "execution_leverage": float(leverage),
            "notional_exposure": float(notional),
            "total_exposure": float(notional),
            "unrealized_pnl_pct": 0.0,
            "take_profit": float(tp),
            "stop_loss": float(sl),
            "effective_take_profit": float(tp),
            "effective_stop_loss": float(sl),
            "take_profit_price": float(tp_price),
            "tp_price": float(tp_price),
            "stop_price": float(sl_price),
            "sl_price": float(sl_price),
            "risk_source": "omega4_6_1_shadow",
        },
        "signal": {
            "rl_action": int(final_action),
            "final_action": int(final_action),
            "source": f"omega4_6_1_shadow|{source_component or 'cash'}",
            "position_signal": "LONG_ENTRY" if final_action == 1 else ("SHORT_ENTRY" if final_action == 2 else "HOLD"),
            "position_reason": reason,
            "unified_kelly": float(notional),
            "position_fraction": float(margin_fraction),
            "margin_fraction": float(margin_fraction),
            "execution_leverage": float(leverage),
            "notional_exposure": float(notional),
            "governor_owner": OMEGA4_6_1_OWNER,
            "governor_reason": reason,
            "take_profit": float(tp),
            "stop_loss": float(sl),
            "take_profit_price": float(tp_price),
            "tp_price": float(tp_price),
            "stop_price": float(sl_price),
            "sl_price": float(sl_price),
            "sleeve_trace": {"omega4_6_1_shadow": trace},
        },
        "agents": {
            "omega4_6_1": {
                "enabled": bool(status == "ok"),
                "shadow_only": True,
                "model_id": OMEGA4_6_1_MODEL_ID,
                "model_version": OMEGA4_6_1_MODEL_VERSION,
                "source_component": source_component,
                "active_take_profit": float(tp),
                "active_stop_loss": float(sl),
                "active_notional": float(notional),
                "active_leverage": float(leverage),
                "active_quality_score": float(quality),
                "active_confidence": float(confidence),
                "quality_threshold": float(cfg.get("quality_threshold", 0.0) or 0.0),
                "duration_threshold": float(cfg.get("duration_threshold", 0.0) or 0.0),
            }
        },
    }


def _omega461_shadow_action_for_pos(pos: str | None) -> int:
    pos_u = str(pos or "").upper()
    if pos_u == "LONG":
        return 1
    if pos_u == "SHORT":
        return 2
    return 0


def _omega461_shadow_decorate_trade_row(row: dict, *, asset_key: str, cfg: dict, real_execution_result: dict | None = None) -> dict:
    out = dict(row or {})
    out["asset"] = str(asset_key)
    out["symbol"] = str(cfg.get("symbol", ""))
    out["account_symbol"] = str(cfg.get("account_symbol", ""))
    out["market"] = str(cfg.get("label", asset_key.upper()))
    if real_execution_result is not None:
        out["shadow_only"] = False
        out["exchange_execution_enabled"] = bool(real_execution_result.get("enabled", False))
        out["exchange_execution_dry_run"] = bool(real_execution_result.get("dry_run", True))
        out["exchange_execution_status"] = str(real_execution_result.get("status", "unknown"))
        out["exchange_execution_orders"] = list(real_execution_result.get("orders", []) or [])
    else:
        out["shadow_only"] = True
        out["exchange_execution_enabled"] = False
        out["exchange_execution_dry_run"] = True
        out["exchange_execution_status"] = "shadow_only"
    return out


def _omega461_shadow_audit_context(
    *,
    asset_key: str,
    cfg: dict,
    price: float,
    updated_at,
    source: str,
    reason: str,
    active: dict,
    real_execution_result: dict | None = None,
) -> dict:
    ts = pd.Timestamp(updated_at)
    price_n = float(price or 0.0)
    side = str(active.get("side", "NONE") or "NONE")
    entry = float(active.get("entry_price", price_n) or price_n)
    tp = float(active.get("take_profit", 0.0) or 0.0)
    sl = float(active.get("stop_loss", 0.0) or 0.0)
    return {
        "ledger_ts_kind": "shadow_bar_close",
        "decision_made_at_kst": str(ts + pd.Timedelta(hours=9)),
        "decision_bar_ts": str(ts + pd.Timedelta(hours=9)),
        "decision_bar_utc": str(ts),
        "decision_bar_open": price_n,
        "decision_bar_high": price_n,
        "decision_bar_low": price_n,
        "decision_bar_close": price_n,
        "decision_bar_volume": 0.0,
        "decision_bar_is_complete": True,
        "decision_price": price_n,
        "decision_price_source": f"{str(cfg.get('symbol', asset_key)).lower()}.close[-1]",
        "execution_bar_ts": str(ts + pd.Timedelta(hours=9)),
        "execution_bar_utc": str(ts),
        "execution_bar_open": price_n,
        "execution_bar_high": price_n,
        "execution_bar_low": price_n,
        "execution_bar_close": price_n,
        "execution_bar_volume": 0.0,
        "execution_bar_is_current": True,
        "execution_price": price_n,
        "execution_price_source": f"{str(cfg.get('symbol', asset_key)).lower()}.shadow_close",
        "execution_delay_sec": 0.0,
        "execution_delay_late": False,
        "execution_delay_mode": "shadow_only_bar_close",
        "model_version": OMEGA4_6_1_MODEL_VERSION,
        "model_id": OMEGA4_6_1_MODEL_ID,
        "model_sleeve": "omega4_6_1_duration_ou_halflife_risk_gate",
        "take_profit": tp,
        "stop_loss": sl,
        "take_profit_price": _omega461_shadow_price(side, entry, tp, take_profit=True),
        "stop_price": _omega461_shadow_price(side, entry, sl, take_profit=False),
        "effective_take_profit": tp,
        "effective_stop_loss": sl,
        "source": source,
        "position_reason": reason,
        "asset": str(asset_key),
        "symbol": str(cfg.get("symbol", "")),
        "account_symbol": str(cfg.get("account_symbol", "")),
        "shadow_only": real_execution_result is None,
        "live_execution": (
            dict(real_execution_result)
            if real_execution_result is not None
            else {"enabled": False, "dry_run": True, "status": "shadow_only", "orders": []}
        ),
    }


def _omega461_shadow_state_from_router(
    *,
    asset_key: str,
    cfg: dict,
    router,
    active: dict,
    current_price: float,
    updated_at,
    status: str = "ok",
    error: str = "",
    regime: str = "",
) -> dict:
    base = _omega461_shadow_state(
        asset_key=asset_key,
        cfg=cfg,
        decision=None,
        current_price=current_price,
        updated_at=updated_at,
        status=status,
        error=error,
        regime=regime,
    )
    pos = str(getattr(router, "pos", None) or "NONE")
    action = _omega461_shadow_action_for_pos(pos)
    entry = float(getattr(router, "entry_price", 0.0) or 0.0)
    tp = float(active.get("take_profit", 0.0) or 0.0)
    sl = float(active.get("stop_loss", 0.0) or 0.0)
    exposure = float(getattr(router, "current_leverage", 0.0) or 0.0)
    leverage = float(getattr(router, "execution_leverage", 1.0) or 1.0)
    margin = float(getattr(router, "position_fraction", 0.0) or 0.0)
    unrealized_pnl_pct = float(router.unrealized_pnl(current_price) if action and current_price > 0.0 else 0.0)
    if action:
        base["position"].update({
            "current": pos,
            "entry_price": entry,
            "hold_bars": int(getattr(router, "hold_count", 0) or 0),
            "position_fraction": margin,
            "margin_fraction": margin,
            "execution_leverage": leverage,
            "notional_exposure": exposure,
            "total_exposure": exposure,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "take_profit": tp,
            "stop_loss": sl,
            "effective_take_profit": tp,
            "effective_stop_loss": sl,
            "take_profit_price": _omega461_shadow_price(pos, entry, tp, take_profit=True),
            "tp_price": _omega461_shadow_price(pos, entry, tp, take_profit=True),
            "stop_price": _omega461_shadow_price(pos, entry, sl, take_profit=False),
            "sl_price": _omega461_shadow_price(pos, entry, sl, take_profit=False),
            "trade_id": str(getattr(router, "open_trade_id", "") or ""),
            "opened_at": str(getattr(router, "opened_at", "") or ""),
        })
    base["signal"].update({
        "rl_action": action,
        "final_action": action,
        "source": str(active.get("source", "omega4_6_1_shadow|cash") or "omega4_6_1_shadow|cash"),
        "position_signal": "LONG_ENTRY" if action == 1 else ("SHORT_ENTRY" if action == 2 else "HOLD"),
        "position_reason": str(active.get("reason", "omega4_6_1_shadow_hold" if action else "omega4_6_1_cash")),
        "unified_kelly": exposure,
        "position_fraction": margin,
        "margin_fraction": margin,
        "execution_leverage": leverage,
        "notional_exposure": exposure,
        "take_profit": tp,
        "stop_loss": sl,
    })
    base["agents"]["omega4_6_1"].update({
        "source_component": str(active.get("source_component", "") or ""),
        "active_take_profit": tp,
        "active_stop_loss": sl,
        "active_notional": exposure,
        "active_leverage": leverage,
        "active_quality_score": float(active.get("quality_score", 0.0) or 0.0),
        "active_confidence": float(active.get("confidence", 0.0) or 0.0),
    })
    return base


def _omega461_shadow_error_state(
    *,
    asset_key: str,
    cfg: dict,
    router,
    active: dict,
    current_price: float,
    updated_at,
    error: str,
    regime: str = "",
) -> dict:
    pos = str(getattr(router, "pos", "") or "")
    if pos in {"LONG", "SHORT"}:
        fallback_price = float(current_price or getattr(router, "entry_price", 0.0) or 0.0)
        return _omega461_shadow_state_from_router(
            asset_key=asset_key,
            cfg=cfg,
            router=router,
            active=active,
            current_price=fallback_price,
            updated_at=updated_at,
            status="error",
            error=error,
            regime=regime,
        )
    return _omega461_shadow_state(
        asset_key=asset_key,
        cfg=cfg,
        decision=None,
        current_price=float(current_price or 0.0),
        updated_at=updated_at,
        status="error",
        error=error,
        regime=regime,
    )


def _omega461_persisted_open_assets(state_dir: Path) -> list[str]:
    open_assets: list[str] = []
    for asset_key in OMEGA4_6_1_SHADOW_ASSET_CONFIG:
        state = _read_json_safe(str(state_dir / f"omega4_6_1_shadow_{asset_key}_state.json"))
        if str(state.get("pos", "") or "") in {"LONG", "SHORT"}:
            open_assets.append(asset_key)
    return sorted(open_assets)


# ════════════════════════════════════════════════════════════════
# 4. 비동기 메인 루프
# ════════════════════════════════════════════════════════════════
async def main(use_local=False):
    fetcher = None
    journal_writer = None
    task_supervisor = None
    ms_scanner = None
    ms_scanner_sol = None
    ms_scanner_btc = None
    tr_interceptor = None
    omega461_shadow_assets: dict[str, dict] = {}
    try:
        # symbol stays ETHUSDT (kline/price data must match what the backtested models were trained
        # and validated on); account_symbol is USDT-M -- settlement currency matches the price series
        # the strategy operates on.
        fetcher      = BinanceLiveFetcher(limit=max(7000, int(FINAL_GOVERNOR_BUFFER_BARS)), account_symbol="ETH/USDT:USDT")
        journal_writer = AsyncJsonlWriter()
        journal_writer.start()
        transition_gate = StateTransitionGate()
        task_supervisor = AsyncTaskSupervisor(
            on_error=lambda name, error: logger.error("background_task_failed name=%s error=%s", name, error)
        )
        fe_engine    = FeatureEngineer()
        runtime_predictor = EnsemblePredictor()
        _acct_status = fetcher.account_status()
        logger.info(
            "SYSTEM binance_account=%s testnet=%s position_sync=%s symbol=%s",
            "ON" if _acct_status.get("ready") else "OFF",
            bool(_acct_status.get("testnet")),
            bool(_acct_status.get("position_sync_enabled")),
            str(_acct_status.get("symbol", "")),
        )
        live_executor = BinanceFuturesExecutionAdapter(fetcher)
        _exec_status = live_executor.status()
        logger.info(
            "SYSTEM binance_execution=%s dry_run=%s testnet=%s symbol=%s audit=%s",
            "ON" if _exec_status.get("enabled") else "OFF",
            bool(_exec_status.get("dry_run")),
            bool(_exec_status.get("testnet")),
            str(_exec_status.get("symbol", "")),
            str(_exec_status.get("audit_path", "")),
        )
        orderbook_recorder = OrderBookRecorder()
        _orderbook_status = orderbook_recorder.status()
        logger.info(
            "SYSTEM orderbook_recorder=%s storage=%s depth=%s db=%s table=%s path=%s",
            "ON" if _orderbook_status.get("enabled") else "OFF",
            str(_orderbook_status.get("storage", "")),
            int(_orderbook_status.get("depth", 0)),
            str(_orderbook_status.get("db_path", "")),
            str(_orderbook_status.get("table", "")),
            str(_orderbook_status.get("path", "")),
        )
        _pending_next_open_intent = {}
        if FINAL_GOVERNOR_SCHEDULE_NEXT_BAR_OPEN_ENABLE:
            _pending_next_open_intent = _read_json_safe(FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH)
            if _pending_next_open_intent:
                logger.info(
                    "SYSTEM pending_next_open_intent=LOADED execute_at=%s action=%s source=%s",
                    str(_pending_next_open_intent.get("execute_at_kst", "")),
                    str(_pending_next_open_intent.get("final_action", "")),
                    str(_pending_next_open_intent.get("source", "")),
                )
        else:
            _stale_pending_next_open = _read_json_safe(FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH)
            if _stale_pending_next_open:
                logger.info(
                    "SYSTEM pending_next_open_intent=DISCARDED reason=backtest_next_open_contract execute_at=%s action=%s",
                    str(_stale_pending_next_open.get("execute_at_kst", "")),
                    str(_stale_pending_next_open.get("final_action", "")),
                )
                try:
                    os.remove(FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH)
                except OSError:
                    pass
    
        # ── Final Governor: Lifecycle V1 first; fully learned and legacy stacks remain as fallback. ──
        meta_router = GovernorPositionRouter()
        compact_meta_router = meta_router
        elite_runtime = EliteSignals()
        final_governor = FinalGovernorRuntime()
        enhanced_engine = EnhancedTradingEngine()
        omega461_shadow_assets: dict[str, dict] = {}
        latest_asset_decisions: dict[str, dict] = {}
        omega461_portfolio_risk = PortfolioRiskManager(
            PortfolioRiskConfig(
                total_notional_cap=FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP,
                asset_shares={
                    # Keyed by SLOT, not by asset: two independent strategies on the same asset
                    # (e.g. a future ETH second slot) must never share a budget key -- see
                    # FINAL_GOVERNOR_PORTFOLIO_ETH_OMEGA461_SUBSHARE's docstring in runtime_config.py.
                    "eth_omega461": FINAL_GOVERNOR_PORTFOLIO_ETH_SHARE * FINAL_GOVERNOR_PORTFOLIO_ETH_OMEGA461_SUBSHARE,
                    "eth_sigma3_1h": FINAL_GOVERNOR_PORTFOLIO_ETH_SHARE * FINAL_GOVERNOR_PORTFOLIO_ETH_SIGMA3_1H_SUBSHARE,
                    "btc": FINAL_GOVERNOR_PORTFOLIO_BTC_SHARE,
                    "sol": FINAL_GOVERNOR_PORTFOLIO_SOL_SHARE,
                },
            )
        )
        if bool(FINAL_GOVERNOR_OMEGA4_6_1_ETH_PORTFOLIO_CAP_ENABLE):
            final_governor.omega4_6_1_portfolio_risk = omega461_portfolio_risk
        if bool(FINAL_GOVERNOR_OMEGA4_6_1_SHADOW_ASSETS_ENABLE):
            for _asset_key, _asset_cfg in OMEGA4_6_1_SHADOW_ASSET_CONFIG.items():
                if _asset_key == "sol" and not bool(FINAL_GOVERNOR_OMEGA4_6_1_SOL_ENABLE):
                    _disabled_router = GovernorPositionRouter()
                    _bootstrap_virtual_router(
                        _disabled_router,
                        str(Path(_THIS_DIR) / "data/ensemble/omega4_6_1_shadow_sol_state.json"),
                    )
                    _disabled_active = validate_omega461_shadow_active_state(
                        _disabled_router.strategy_state.get(OMEGA4_6_1_SHADOW_ACTIVE_STATE_KEY),
                        asset_key=_asset_key,
                        expected_component=str(_asset_cfg["component"]),
                        position=_disabled_router.pos,
                        entry_price=float(_disabled_router.entry_price or 0.0),
                        position_fraction=float(_disabled_router.position_fraction or 0.0),
                        execution_leverage=float(_disabled_router.execution_leverage or 1.0),
                        notional_exposure=float(_disabled_router.current_leverage or 0.0),
                    )
                    latest_asset_decisions[_asset_key] = _omega461_shadow_state_from_router(
                        asset_key=_asset_key,
                        cfg=dict(_asset_cfg),
                        router=_disabled_router,
                        active=_disabled_active,
                        current_price=float(_disabled_router.entry_price or 0.0),
                        updated_at=pd.Timestamp.utcnow(),
                        status="disabled",
                        error="model_disabled_by_config",
                    )
                    logger.warning(
                        "SYSTEM omega4_6_1_shadow asset=sol status=DISABLED persisted_position=%s",
                        str(_disabled_router.pos or "NONE"),
                    )
                    continue
                _bundle_path = Path(_THIS_DIR) / str(_asset_cfg["bundle_path"])
                _sidecar_path = Path(_THIS_DIR) / str(_asset_cfg["sidecar_path"])
                _missing = [str(p) for p in (_bundle_path, _sidecar_path) if not p.exists()]
                if _missing:
                    raise RuntimeError(f"missing Omega4.6.1 {_asset_key} shadow artifacts: {_missing}")
                _component = str(_asset_cfg["component"])
                _shadow_router = GovernorPositionRouter()
                _bootstrap_virtual_router(
                    _shadow_router,
                    str(Path(_THIS_DIR) / "data/ensemble" / f"omega4_6_1_shadow_{_asset_key}_state.json"),
                )
                _shadow_active = validate_omega461_shadow_active_state(
                    _shadow_router.strategy_state.get(OMEGA4_6_1_SHADOW_ACTIVE_STATE_KEY),
                    asset_key=_asset_key,
                    expected_component=_component,
                    position=_shadow_router.pos,
                    entry_price=float(_shadow_router.entry_price or 0.0),
                    position_fraction=float(_shadow_router.position_fraction or 0.0),
                    execution_leverage=float(_shadow_router.execution_leverage or 1.0),
                    notional_exposure=float(_shadow_router.current_leverage or 0.0),
                )
                _asset_fetcher = BinanceLiveFetcher(
                    symbol=str(_asset_cfg["symbol"]),
                    timeframe=fetcher.timeframe,
                    limit=max(7000, int(FINAL_GOVERNOR_BUFFER_BARS)),
                    # Explicit per-asset override -- without this, the process-wide
                    # BINANCE_ACCOUNT_SYMBOL env var (set to ETH's symbol for the existing real
                    # ETH deployment) would silently force every asset's account_symbol to ETH's.
                    account_symbol=str(_asset_cfg["account_symbol"]),
                )
                omega461_shadow_assets[_asset_key] = {
                    "cfg": dict(_asset_cfg),
                    "fetcher": _asset_fetcher,
                    # SOL's v2 bundle/sidecar (2026-07-20) was trained on adaptive_squeeze
                    # features (long_squeeze_risk/short_squeeze_risk/crowding_pressure use SOL's
                    # own rolling funding_z_score instead of ETH's fixed 0.0002 divisor) -- must
                    # match here or live inference silently diverges from what the model was
                    # trained on, the same failure class as the regime3 HMM bug fixed earlier
                    # this session.
                    "fe_engine": FeatureEngineer(adaptive_squeeze=(_asset_key == "sol")),
                    "eth_buffer": None,
                    "btc_buffer": None,
                    "router": _shadow_router,
                    "active": _shadow_active,
                    "adapter": Omega461LiveAdapter(
                        h48qual_bundle=FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH,
                        h48qual_sidecar=FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_SIDECAR_PATH,
                        zig075_bundle=FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH,
                        zig075_sidecar=FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_SIDECAR_PATH,
                        device=final_governor.device,
                        # Found 2026-07-20: without this, the adapter falls back to its own
                        # default (ETH's regime3 HMM) for every shadow asset, mismatching the
                        # asset-specific regime3_current_sensitive_wide24_* sidecar each parent/
                        # risk-sidecar was actually trained on.
                        current_regime_path=str(_asset_cfg["current_regime_path"]),
                        # Same fix, same date, for the sizing template/expert scales -- each
                        # asset has its own copy of these (currently identical values to ETH's,
                        # but the live path previously always used ETH's module regardless).
                        base_template=(SOL_BASE_TEMPLATE if _asset_key == "sol" else BTC_BASE_TEMPLATE),
                        expert_scales=(SOL_EXPERT_SCALES if _asset_key == "sol" else BTC_EXPERT_SCALES),
                        components_override={
                            _component: {
                                "bundle": _bundle_path,
                                "sidecar": _sidecar_path,
                                "quality_threshold": float(_asset_cfg["quality_threshold"]),
                                # SOL's live zig075 sidecar predates the validation_only-only
                                # Omega Artifact Integrity policy (2026-06-30) -- it was selected+
                                # promoted live under 'validation_oos_guard' on 2026-07-20 (see
                                # project-sol-adaptive-squeeze-v2-live-20260720 memory). BTC's
                                # sidecar is already validation_only and stays on the strict default.
                                **(
                                    {"allowed_selection_scopes": frozenset({"validation_only", "validation_oos_guard"})}
                                    if _asset_key == "sol" else {}
                                ),
                            }
                        },
                        priority=(_component,),
                        duration_threshold=(
                            -999.0
                            if bool(FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_DURATION_GATE_OFF)
                            else float(_asset_cfg["duration_threshold"])
                        ),
                        scale_map=dict(_asset_cfg["scale_map"]),
                    ),
                    # None unless FINAL_GOVERNOR_OMEGA4_6_1_BTC_CMAMBA_ENTRY_GATE_ENABLE is set (BTC
                    # only -- SOL's own backtest of this filter made both VAL and OOS worse).
                    "btc_cmamba_gate": (
                        BtcCmambaEntryGate(model_path=FINAL_GOVERNOR_OMEGA4_6_1_BTC_CMAMBA_MODEL_PATH, device="cuda")
                        if (_asset_key == "btc" and bool(FINAL_GOVERNOR_OMEGA4_6_1_BTC_CMAMBA_ENTRY_GATE_ENABLE))
                        else None
                    ),
                    # None unless FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE is set --
                    # when None, _refresh_omega461_shadow_asset behaves exactly as before (shadow only).
                    "executor": (
                        BinanceFuturesExecutionAdapter(_asset_fetcher)
                        if bool(FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE)
                        else None
                    ),
                    # None unless FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_ORDERBOOK_RECORDER_ENABLE is set --
                    # public market data only (fetcher.exchange), no account/auth needed.
                    "orderbook_recorder": (
                        OrderBookRecorder(table=f"orderbook_decision_snapshots_{_asset_key}")
                        if bool(FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_ORDERBOOK_RECORDER_ENABLE)
                        else None
                    ),
                }
                # 2026-08-07 BTC swingtransition promotion: the promoted parent bundle needs
                # swing_transition_prob, which the shared FeatureEngineer does not compute.
                # Auto-enabled ONLY when the loaded bundle's base_cols actually require it, so an
                # env-var rollback to the previous bundle also disables this provider (and its
                # Deribit DVOL dependency) without any further change.
                omega461_shadow_assets[_asset_key]["btc_swing_transition"] = (
                    BtcSwingTransitionLiveFeature()
                    if (
                        _asset_key == "btc"
                        and "swing_transition_prob"
                        in omega461_shadow_assets[_asset_key]["adapter"].components[_component].base_cols
                    )
                    else None
                )
            if omega461_shadow_assets:
                logger.info("SYSTEM omega4_6_1_shadow_assets=ON assets=%s", ",".join(sorted(omega461_shadow_assets)))
        else:
            _disabled_open_assets = _omega461_persisted_open_assets(Path(_THIS_DIR) / "data/ensemble")
            if _disabled_open_assets:
                raise RuntimeError(
                    "Omega4.6.1 shadow assets are disabled while persisted positions are open: "
                    + ",".join(_disabled_open_assets)
                )
            logger.info("SYSTEM omega4_6_1_shadow_assets=OFF open_positions=NONE")
        logger.info(
            "SYSTEM mode=FINAL_GOVERNOR stack=%s legacy_macro_sniper=%s trend=%s micro=%s next_open=%s scheduled_next_open=%s fetch_delay=%.2fs console=compact",
            FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID if final_governor.fully_learned_policy_bundle is not None else "lifecycle_v1_clean_base",
            not bool(final_governor.fully_learned_policy_bundle is not None),
            bool(FINAL_GOVERNOR_TREND_ENABLE),
            bool(FINAL_GOVERNOR_MICRO_ENABLE),
            bool(FINAL_GOVERNOR_NEXT_OPEN_EXECUTION_ENABLE),
            bool(FINAL_GOVERNOR_SCHEDULE_NEXT_BAR_OPEN_ENABLE),
            float(FINAL_GOVERNOR_BAR_FETCH_DELAY_SEC),
        )
        _prev_meta_pos: str | None = None
    
        # ── 선행 레이더 & 사후 요격기 시작 ──────────────────────────────────
        _symbol = fetcher.symbol.lower()  # e.g. "ethusdt"
        
        ms_scanner = MicrostructureScanner(symbol=_symbol)
        ms_scanner.start()
        ms_scanner_sol: MicrostructureScanner | None = None
        ms_scanner_btc: MicrostructureScanner | None = None
        if bool(FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_MICROSTRUCTURE_SCANNER_ENABLE):
            ms_scanner_sol = MicrostructureScanner(symbol="solusdt")
            ms_scanner_sol.start()
            ms_scanner_btc = MicrostructureScanner(symbol="btcusdt")
            ms_scanner_btc.start()

        # Standalone SOL orderbook recording, decoupled from SOL trading logic (2026-08-02): the
        # per-asset orderbook_recorder normally lives inside _refresh_omega461_shadow_asset, which
        # is entirely skipped when FINAL_GOVERNOR_OMEGA4_6_1_SOL_ENABLE=False -- so disabling SOL
        # trading silently killed SOL orderbook data collection too, even though the recorder only
        # needs public market data (fetcher.exchange) and has nothing to do with trading decisions.
        # This task exists only while SOL trading itself stays disabled; once SOL_ENABLE=True the
        # normal per-asset path already covers it and this would just double-record.
        if bool(FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_ORDERBOOK_RECORDER_ENABLE) and not bool(FINAL_GOVERNOR_OMEGA4_6_1_SOL_ENABLE):
            _sol_orderbook_fetcher = BinanceLiveFetcher(symbol="solusdt", account_symbol="SOL/USDT:USDT")
            _sol_orderbook_recorder = OrderBookRecorder(table="orderbook_decision_snapshots_sol")

            async def _sol_orderbook_recorder_loop() -> None:
                while True:
                    try:
                        await _sol_orderbook_recorder.record_decision_snapshot(
                            _sol_orderbook_fetcher,
                            timestamp_kst=pd.Timestamp.now(tz="Asia/Seoul"),
                            context={"record_reason": "sol_data_only_standalone_while_trading_disabled"},
                            force=True,
                        )
                    except Exception as _sol_ob_e:
                        logger.warning("SYSTEM sol_orderbook_recorder_standalone failed: %s", _sol_ob_e)
                    await asyncio.sleep(60.0)

            asyncio.create_task(_sol_orderbook_recorder_loop())
            logger.info("SYSTEM sol_orderbook_recorder_standalone=ON table=orderbook_decision_snapshots_sol interval=60s")

        tr_interceptor = TailRiskInterceptor(symbol=_symbol)
        tr_interceptor.start()
        _dashboard_shadow_task: asyncio.Task | None = None
        _daily_trade_report_task: asyncio.Task | None = None
        _shadow_prev_price: float | None = None
        _shadow_quant_minute_key: str = ""
        _last_exchange_reconcile_ts: float = 0.0
        _last_shadow_health_log_ts: float = 0.0
        _last_data_pipeline_health_log_ts: float = 0.0
    
        async def _fetch_quant_close_1m(limit: int = 1000) -> pd.DataFrame:
            klines = await fetcher._call_with_retry(
                f"fetch_quant_close_1m[{fetcher.symbol}]",
                lambda: fetcher.exchange.fapiPublicGetKlines(
                    {"symbol": fetcher.symbol, "interval": "1m", "limit": int(max(100, min(limit, 1500)))}
                ),
            )
            if not klines:
                return pd.DataFrame(columns=["timestamp", "close"])
            qdf = pd.DataFrame(klines).iloc[:, [0, 4]]
            qdf.columns = ["timestamp", "close"]
            qdf["timestamp"] = pd.to_datetime(qdf["timestamp"], unit="ms", utc=True, errors="coerce")
            qdf["close"] = pd.to_numeric(qdf["close"], errors="coerce")
            qdf = qdf.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
            return qdf
    
        async def _dashboard_shadow_loop():
            """10초마다 micro/tail/playbook 필드만 dashboard_state.json 갱신."""
            nonlocal _shadow_prev_price, _shadow_quant_minute_key, _last_exchange_reconcile_ts, _last_shadow_health_log_ts, _prev_meta_pos
            while True:
                try:
                    now = time.time()
                    await asyncio.sleep((10.0 - (now % 10.0)) + 0.05)
    
                    if not use_local:
                        reconcile_now = time.time()
                        if (reconcile_now - _last_exchange_reconcile_ts) >= 15.0:
                            async with transition_gate.transition("exchange_reconcile"):
                                exchange_position_state, restored = await _fetch_exchange_position()
                                _last_exchange_reconcile_ts = reconcile_now
                                # A resting exchange TP/SL order (see BinanceFuturesExecutionAdapter.
                                # place_tp_sl_orders) can fill between decision cycles, flattening the
                                # exchange position with the bot never told. Detect that transition here
                                # (previously this call was skipped entirely when `restored` was falsy,
                                # i.e. whenever the exchange was flat -- silently losing the close).
                                _went_flat = exchange_position_went_flat(exchange_position_state, meta_router.pos)
                                _tp_sl_fill_info = None
                                _tp_order_id = ""
                                _sl_order_id = ""
                                if _went_flat and final_governor.owner == OMEGA4_6_1_OWNER:
                                    _tp_order_id = str(getattr(final_governor, "active_omega4_6_1_tp_order_id", "") or "")
                                    _sl_order_id = str(getattr(final_governor, "active_omega4_6_1_sl_order_id", "") or "")
                                    if _tp_order_id or _sl_order_id:
                                        _tp_sl_fill_info = await live_executor.poll_tp_sl_orders(
                                            tp_order_id=_tp_order_id, sl_order_id=_sl_order_id
                                        )
                                if restored or _went_flat:
                                    meta_router.reconcile_external_position(
                                        restored.get("type") if restored else None,
                                        float(restored.get("entry_price", 0.0)) if restored else 0.0,
                                        float(restored.get("leverage", 0.0)) if restored else 0.0,
                                        notional=float(restored.get("notional", 0.0)) if restored else 0.0,
                                        account_equity=float(restored.get("account_equity_usdt", 0.0)) if restored else 0.0,
                                        notional_exposure=float(restored.get("notional_exposure", 0.0)) if restored else 0.0,
                                        tp_sl_fill_info=_tp_sl_fill_info,
                                        timestamp_kst=pd.Timestamp.now(tz="Asia/Seoul"),
                                        regime_name=str(getattr(final_governor, "owner_regime", "") or ""),
                                        governor_source=f"{OMEGA4_6_1_OWNER}|exchange_reconcile",
                                    )
                                    _reconcile_close_payload = getattr(meta_router, "_last_reconcile_close_payload", None)
                                    if _reconcile_close_payload:
                                        _prev_meta_pos = meta_router.pos
                                        await journal_writer.append_many(
                                            TRADE_JOURNAL_PATH, [dict(_reconcile_close_payload)]
                                        )
                                        if final_governor.owner == OMEGA4_6_1_OWNER:
                                            if _tp_order_id or _sl_order_id:
                                                await live_executor.cancel_tp_sl_orders(
                                                    tp_order_id=_tp_order_id, sl_order_id=_sl_order_id
                                                )
                                            final_governor._reset_omega4_6_1_position_state()
                                            final_governor.owner = ""
                                            final_governor.owner_regime = ""
                                            final_governor._save_runtime_state()
                                # Defensive: stale ids with no local position would otherwise permanently
                                # block future entries via execute_to_target's race guard. See the matching
                                # boot-time cleanup for how this can happen.
                                if meta_router.pos not in {"LONG", "SHORT"}:
                                    _stale_tp_id = str(getattr(final_governor, "active_omega4_6_1_tp_order_id", "") or "")
                                    _stale_sl_id = str(getattr(final_governor, "active_omega4_6_1_sl_order_id", "") or "")
                                    if _stale_tp_id or _stale_sl_id:
                                        logger.warning(
                                            "SYSTEM stale omega4_6_1 tp/sl order ids found with no local position "
                                            "tp_id=%s sl_id=%s -- cancelling and clearing", _stale_tp_id, _stale_sl_id
                                        )
                                        await live_executor.cancel_tp_sl_orders(
                                            tp_order_id=_stale_tp_id, sl_order_id=_stale_sl_id
                                        )
                                        final_governor._reset_omega4_6_1_position_state()
                                        final_governor._save_runtime_state()
    
                    ms = dict(ms_scanner.get_signal() or {})
                    tr_shadow = dict(getattr(tr_interceptor, "_shadow_state", {}) or {})
                    tr_bucket = str(tr_shadow.get("shadow_risk_bucket", "normal"))
                    tr_reco = "HOLD" if tr_bucket == "high" else ("REDUCE" if tr_bucket == "watch" else "FOLLOW")
    
                    state = {}
                    compact_state_seed = {}
                    try:
                        if os.path.exists(DASHBOARD_STATE_PATH):
                            with open(DASHBOARD_STATE_PATH, "r", encoding="utf-8") as f:
                                state = json.load(f)
                    except Exception:
                        state = {}
                    try:
                        if os.path.exists(COMPACT_DASHBOARD_STATE_PATH):
                            with open(COMPACT_DASHBOARD_STATE_PATH, "r", encoding="utf-8") as f:
                                compact_state_seed = json.load(f)
                    except Exception:
                        compact_state_seed = {}
    
                    _state_price = float(state.get("price", 0.0) or 0.0)
                    _mark_price = float(ms.get("mark_price", 0.0) or 0.0)
                    _cur_price = _mark_price if _mark_price > 0.0 else _state_price
                    _prev_price = float(_shadow_prev_price if _shadow_prev_price is not None else _cur_price)
                    _price_change_pct = (_cur_price - _prev_price) / max(abs(_prev_price), 1e-8) if _prev_price > 0 else 0.0
                    _shadow_prev_price = _cur_price if _cur_price > 0 else _shadow_prev_price
                    tr_pb = dict(
                        tr_interceptor.get_playbook_signal(
                            price_change_pct=_price_change_pct,
                            current_price=_cur_price,
                        ) or {}
                    )
                    _base_action = int((state.get("signal") or {}).get("final_action", 0) or 0)
                    _base_kelly = float((state.get("signal") or {}).get("unified_kelly", 0.0) or 0.0)
                    _base_pos = str((state.get("position") or {}).get("current", "NONE") or "NONE")
                    _pb_eval = _disabled_playbook_eval(action=_base_action, kelly=_base_kelly)
                    _pb = dict(_pb_eval.get("winner_mft", {}) or {})
                    _pb_hft = dict(_pb_eval.get("winner_hft", {}) or {})
                    _pb_mft = dict(_pb_eval.get("winner_mft", {}) or {})
                    _pb_list = list(_pb_eval.get("evaluations", []) or [])
                    _sess_flags = _session_flags_from_timestamp(
                        state.get("cycle_timestamp_kst", pd.Timestamp.now(tz="Asia/Seoul"))
                    )
    
                    # Keep `updated_at` as 5m main-cycle timestamp.
                    # Shadow loop (10s) uses separate marker so governor/agent cards
                    # are not perceived as refreshed every 10 seconds.
                    state["shadow_updated_at"] = pd.Timestamp.utcnow().isoformat()
                    state["governor_mode"] = True
                    if _cur_price > 0.0:
                        state["price"] = float(_cur_price)
                    _shadow_closed_trade_equity = _router_closed_trade_equity(meta_router)
                    _shadow_open_mark_pnl_frac = _router_open_mark_pnl_frac(meta_router, _cur_price)
                    _shadow_strategy_equity = _router_strategy_equity(meta_router, _cur_price)
                    _prev_position = dict(state.get("position", {}) or {})
                    _prev_signal = dict(state.get("signal", {}) or {})
                    _shadow_max_hold = int(_prev_position.get("max_hold_bars", _prev_signal.get("max_hold_bars", 0)) or 0)
                    _shadow_remaining = int(max(0, _shadow_max_hold - int(meta_router.hold_count or 0))) if _shadow_max_hold > 0 else int(_prev_position.get("max_hold_remaining_bars", _prev_signal.get("max_hold_remaining_bars", 0)) or 0)
                    try:
                        if meta_router.pos in {"LONG", "SHORT"}:
                            _shadow_regime = str(
                                _prev_signal.get(
                                    "regime",
                                    _prev_position.get("model_sleeve", _prev_position.get("regime", "unknown")),
                                )
                                or "unknown"
                            )
                            final_governor._sync_owner(meta_router, _shadow_regime)
                    except Exception as _sync_e:
                        logger.debug("dashboard shadow owner sync skip: %s", _sync_e)
                    _shadow_tp = float(_prev_position.get("take_profit", _prev_signal.get("take_profit", 0.0)) or 0.0)
                    _shadow_sl = float(_prev_position.get("stop_loss", _prev_signal.get("stop_loss", 0.0)) or 0.0)
                    _shadow_tp_price = float(_prev_position.get("take_profit_price", _prev_signal.get("take_profit_price", 0.0)) or 0.0)
                    _shadow_sl_price = float(_prev_position.get("stop_price", _prev_signal.get("stop_price", 0.0)) or 0.0)
                    _shadow_risk_source = str(_prev_position.get("risk_source", _prev_signal.get("risk_source", _prev_signal.get("source", ""))) or "")
                    # Keep the fast shadow refresh aligned with the live router state.
                    # Updating only `unrealized_pnl_pct` against a stale dashboard snapshot
                    # can mix an old side/entry with the new router position and flip the sign.
                    state["position"] = {
                        "current": meta_router.pos or "NONE",
                        "trade_id": str(meta_router.open_trade_id or ""),
                        "entry_price": float(meta_router.entry_price or 0.0),
                        "entry_price_source": str(meta_router.entry_price_source or ""),
                        "entry_decision_price": float(meta_router.entry_decision_price or 0.0),
                        "exchange_entry_price": float(meta_router.exchange_entry_price or 0.0),
                        "entry_execution_liquidity": str(meta_router.entry_execution_liquidity or ""),
                        "entry_execution_route": str(meta_router.entry_execution_route or ""),
                        "entry_execution_order_type": str(meta_router.entry_execution_order_type or ""),
                        "decision_at": str(meta_router.decision_at or ""),
                        "opened_at": str(meta_router.opened_at or ""),
                        "hold_bars": int(meta_router.hold_count or 0),
                        "position_fraction": float(meta_router.position_fraction or 0.0),
                        "margin_fraction": float(meta_router.position_fraction or 0.0),
                        "execution_leverage": float(meta_router.execution_leverage or 1.0),
                        "notional_exposure": float(meta_router.current_leverage or 0.0),
                        "total_exposure": float(meta_router.current_leverage or 0.0),
                        "unrealized_pnl_pct": float(meta_router.unrealized_pnl(_cur_price) if meta_router.pos and _cur_price > 0.0 else 0.0),
                        "position_realized_pnl_frac": float(meta_router.position_realized_pnl_frac or 0.0),
                        "position_realized_pnl_pct": float((meta_router.position_realized_pnl_frac or 0.0) * 100.0),
                        "last_resize_realized_pnl_frac": float(meta_router.last_resize_realized_pnl_frac or 0.0),
                        "strategy_equity": float(_shadow_strategy_equity),
                        "closed_trade_equity": float(_shadow_closed_trade_equity),
                        "deployed_equity": float(_shadow_strategy_equity * float(meta_router.position_fraction or 0.0)),
                        "gross_exposure_equity": float(_shadow_strategy_equity * float(meta_router.current_leverage or 0.0)),
                        "unrealized_pnl_amount": float(_shadow_strategy_equity * _shadow_open_mark_pnl_frac),
                        "trade_pnl_pct": (state.get("position", {}) or {}).get("trade_pnl_pct"),
                        "take_profit": float(_shadow_tp),
                        "stop_loss": float(_shadow_sl),
                        "max_hold_bars": int(_shadow_max_hold),
                        "max_hold_remaining_bars": int(_shadow_remaining),
                        "take_profit_price": float(_shadow_tp_price),
                        "tp_price": float(_shadow_tp_price),
                        "stop_price": float(_shadow_sl_price),
                        "sl_price": float(_shadow_sl_price),
                        "effective_take_profit": float(_shadow_tp),
                        "effective_stop_loss": float(_shadow_sl),
                        "risk_source": str(_shadow_risk_source),
                    }
                    state["session"] = {
                        "session_asia": float(_sess_flags.get("session_asia", 0.0)),
                        "session_europe": float(_sess_flags.get("session_europe", 0.0)),
                        "session_us": float(_sess_flags.get("session_us", 0.0)),
                    }
                    state["account"] = fetcher.account_status()
                    state["microstructure"] = {
                        "updated_at": pd.Timestamp.utcnow().isoformat(),
                        "mark_price": float(ms.get("mark_price", 0.0)),
                        "obi": float(ms.get("obi", 0.0)),
                        "taker_buy_ratio": float(ms.get("taker_buy_ratio", 0.5)),
                        "spoofing_score": float(ms.get("spoofing_score", 0.0)),
                        "spoofing_bias": int(ms.get("spoofing_bias", 0)),
                        "nif_whale": float(ms.get("nif_whale", 0.0)),
                        "nif_retail": float(ms.get("nif_retail", 0.0)),
                        "nif_bias": int(ms.get("nif_bias", 0)),
                        "eai": float(ms.get("eai", 0.0)),
                        "eai_bias": int(ms.get("eai_bias", 0)),
                        "oi_delta_pct": float(ms.get("oi_delta_pct", 0.0)),
                        "oi_delta_cum_5m": float(ms.get("oi_delta_cum_5m", 0.0)),
                        "oi_delta_cum_5m_bucket_start_ts": int(ms.get("oi_delta_cum_5m_bucket_start_ts", 0)),
                        "whale_flow_10s_ratio": float(ms.get("whale_flow_10s_ratio", 0.0)),
                        "whale_buy_10s_usd": float(ms.get("whale_buy_10s_usd", 0.0)),
                        "whale_sell_10s_usd": float(ms.get("whale_sell_10s_usd", 0.0)),
                        "whale_flow_cum_5m_ratio": float(ms.get("whale_flow_cum_5m_ratio", 0.0)),
                        "whale_buy_cum_5m_usd": float(ms.get("whale_buy_cum_5m_usd", 0.0)),
                        "whale_sell_cum_5m_usd": float(ms.get("whale_sell_cum_5m_usd", 0.0)),
                        "whale_flow_cum_5m_bucket_start_ts": int(ms.get("whale_flow_cum_5m_bucket_start_ts", 0)),
                        "funding_rate": float(ms.get("funding_rate", 0.0)),
                        "signal_bias": int(ms.get("signal_bias", 0)),
                        "kelly_mult": float(ms.get("kelly_mult", 1.0)),
                        "toxicity_score": float(ms.get("shadow_toxicity_score", 0.0)),
                        "toxicity_regime": str(ms.get("shadow_toxicity_regime", "normal")),
                        "queue_collapse": float(ms.get("shadow_queue_collapse", 0.0)),
                        "absorption_score": float(ms.get("shadow_absorption_score", 0.0)),
                        "queue_bias": int(ms.get("shadow_queue_bias", 0)),
                        "regime_tag": str(ms.get("shadow_regime_tag", "normal")),
                        "regime_conf": float(ms.get("shadow_regime_conf", 0.0)),
                        "price_change_30m": float(ms.get("price_change_30m", 0.0)),
                        "price_volatility_30m": float(ms.get("price_volatility_30m", 0.0)),
                        "vwap_gap_15m": float(ms.get("vwap_gap_15m", 0.0)),
                        "price_breakout_60m": bool(ms.get("price_breakout_60m", False)),
                        "price_breakdown_60m": bool(ms.get("price_breakdown_60m", False)),
                        "nif_whale_sum_30m": float(ms.get("nif_whale_sum_30m", 0.0)),
                        "nif_whale_avg_30m": float(ms.get("nif_whale_avg_30m", 0.0)),
                        "nif_whale_std_30m": float(ms.get("nif_whale_std_30m", 0.0)),
                        "whale_short_build_ratio_30m": float(ms.get("whale_short_build_ratio_30m", 0.0)),
                        "whale_long_close_ratio_30m": float(ms.get("whale_long_close_ratio_30m", 0.0)),
                        "whale_sell_presence_ratio_30m": float(ms.get("whale_sell_presence_ratio_30m", 0.0)),
                        "whale_sell_effective_ratio_30m": float(ms.get("whale_sell_effective_ratio_30m", 0.0)),
                        "whale_long_build_ratio_30m": float(ms.get("whale_long_build_ratio_30m", 0.0)),
                        "whale_short_cover_ratio_30m": float(ms.get("whale_short_cover_ratio_30m", 0.0)),
                        "whale_buy_presence_ratio_30m": float(ms.get("whale_buy_presence_ratio_30m", 0.0)),
                        "whale_buy_effective_ratio_30m": float(ms.get("whale_buy_effective_ratio_30m", 0.0)),
                        "whale_position_bias_30m": str(ms.get("whale_position_bias_30m", "중립")),
                        "whale_position_window_min": int(ms.get("whale_position_window_min", 5)),
                        "whale_position_estimate": str(ms.get("whale_position_estimate", "NEUTRAL")),
                        "whale_position_confidence": int(ms.get("whale_position_confidence", 0)),
                        "whale_position_score": float(ms.get("whale_position_score", 0.0)),
                        "absorption_avg_30m": float(ms.get("absorption_avg_30m", 0.0)),
                        "bias_avg_30m": float(ms.get("bias_avg_30m", 0.0)),
                        "toxicity_avg_30m": float(ms.get("toxicity_avg_30m", 0.0)),
                        "eai_delta_15m": float(ms.get("eai_delta_15m", 0.0)),
                        "data_stale": bool(ms.get("data_stale", False)),
                        "depth_connected": bool(ms.get("depth_connected", False)),
                        "trade_connected": bool(ms.get("trade_connected", False)),
                        "poll_connected": bool(ms.get("poll_connected", False)),
                        "depth_age_sec": (float(ms.get("depth_age_sec")) if ms.get("depth_age_sec") is not None else None),
                        "trade_age_sec": (float(ms.get("trade_age_sec")) if ms.get("trade_age_sec") is not None else None),
                        "poll_age_sec": (float(ms.get("poll_age_sec")) if ms.get("poll_age_sec") is not None else None),
                        "recent_trade_count_5m": int(ms.get("recent_trade_count_5m", 0)),
                        "recent_trade_notional_5m": float(ms.get("recent_trade_notional_5m", 0.0)),
                        "recent_whale_count_5m": int(ms.get("recent_whale_count_5m", 0)),
                        "valid_taker_flow": bool(ms.get("valid_taker_flow", False)),
                        "valid_nif": bool(ms.get("valid_nif", False)),
                        "status_line": str(ms_scanner.status_line()),
                    }
                    state["tail_risk"] = {
                        "updated_at": pd.Timestamp.utcnow().isoformat(),
                        "aftershock_prob": float(tr_shadow.get("shadow_aftershock_prob", 0.0)),
                        "half_life_min": float(tr_shadow.get("shadow_decay_half_life", 0.0)),
                        "risk_bucket": tr_bucket,
                        "z_long": float(tr_pb.get("z_long", 0.0)),
                        "z_short": float(tr_pb.get("z_short", 0.0)),
                        "lai": float(tr_pb.get("lai", 0.0)),
                        "long_usd_1m": float(tr_pb.get("long_usd_1m", 0.0)),
                        "short_usd_1m": float(tr_pb.get("short_usd_1m", 0.0)),
                        "liq_event_count_1m": int(tr_pb.get("liq_event_count_1m", 0)),
                        "ws_connected": bool(tr_pb.get("ws_connected", False)),
                        "ws_age_sec": (float(tr_pb.get("ws_age_sec")) if tr_pb.get("ws_age_sec") is not None else None),
                        "valid_liq_stream": bool(tr_pb.get("valid_liq_stream", False)),
                        "hawkes_active": bool(tr_pb.get("hawkes_active", False)),
                        "hawkes_decay_level": float(tr_pb.get("hawkes_decay_level", 0.0)),
                        "crisis_type": str(tr_pb.get("crisis_type", "")),
                        "liq_cluster_direction": int(tr_pb.get("liq_cluster_direction", 0)),
                        "liq_cluster_strength": float(tr_pb.get("liq_cluster_strength", 0.0)),
                        "distance_to_cluster_pct": float(tr_pb.get("distance_to_cluster_pct", 1.0)),
                        "liq_cluster_price": float(tr_pb.get("liq_cluster_price", 0.0)),
                        "z_bias": int(-1 if float(tr_pb.get("z_long", 0.0)) > float(tr_pb.get("z_short", 0.0)) else (1 if float(tr_pb.get("z_short", 0.0)) > float(tr_pb.get("z_long", 0.0)) else 0)),
                        "recommendation": tr_reco,
                        "status_line": str(tr_interceptor.status_line()),
                    }
                    state["playbook"] = {
                        "updated_at": pd.Timestamp.utcnow().isoformat(),
                        "matched": bool(_pb.get("matched", False)),
                        "name": str(_pb.get("name", "NONE")),
                        "priority": int(_pb.get("priority", 0)),
                        "action": int(_pb.get("action", _base_action)),
                        "kelly": float(_pb.get("kelly", _base_kelly)),
                        "reason": str(_pb.get("reason", "")),
                        "emergency_exit": bool(_pb.get("emergency_exit", False)),
                        "widen_trailing_stop": bool(_pb.get("widen_trailing_stop", False)),
                        "meta": dict(_pb.get("meta", {}) or {}),
                        "hft": _pb_hft,
                        "mft": _pb_mft,
                        "evaluations": _pb_list,
                    }
                    _ens = _build_ensemble_runtime(
                        pb_list=_pb_list,
                        base_action=_base_action,
                        base_kelly=_base_kelly,
                        ms=ms,
                        tr=tr_pb,
                    )
                    _loop = asyncio.get_running_loop()
                    _trk = await _loop.run_in_executor(
                        None,
                        lambda: _update_ensemble_tracker(
                            ensembles=_ens,
                            current_price=_cur_price,
                            now_iso=_now_kst_iso(),
                        ),
                    )
                    _ens["tracker"] = _ensemble_tracker_summary(_trk)
                    state["ensembles"] = _ens
    
                    now_kst = pd.Timestamp.now(tz="Asia/Seoul")
                    quant_minute_key = now_kst.strftime("%Y-%m-%d %H:%M")
                    if _shadow_quant_minute_key != quant_minute_key:
                        try:
                            qdf = await _fetch_quant_close_1m(limit=1000)
                            if len(qdf) > 0:
                                q_cur = float(qdf["close"].iloc[-1])
                                state["quant_formula"] = _build_quant_formula_card(
                                    eth_df=qdf,
                                    current_price=q_cur,
                                    current_time_kst=now_kst,
                                )
                                _shadow_quant_minute_key = quant_minute_key
                        except Exception as _qerr:
                            logger.debug("quant shadow update skip: %s", _qerr)
    
                    _shadow_agents = dict(state.get("agents", {}) or {})
                    _shadow_agents["omega4_6_1"] = {
                        "enabled": bool(final_governor.omega4_6_1_adapter is not None),
                        "model_id": OMEGA4_6_1_MODEL_ID,
                        "model_version": OMEGA4_6_1_MODEL_VERSION,
                        "source_component": str(final_governor.active_omega4_6_1_source_component),
                        "active_take_profit": float(final_governor.active_omega4_6_1_take_profit),
                        "active_stop_loss": float(final_governor.active_omega4_6_1_stop_loss),
                        "active_notional": float(final_governor.active_omega4_6_1_notional),
                        "active_leverage": float(final_governor.active_omega4_6_1_leverage),
                        "active_quality_score": float(final_governor.active_omega4_6_1_quality_score),
                        "active_confidence": float(final_governor.active_omega4_6_1_confidence),
                    }
                    state["agents"] = _shadow_agents
    
                    position_projection = _build_compact_dashboard_state(
                        state,
                        compact_meta_router,
                        float(_cur_price or state.get("price", 0.0) or 0.0),
                        now_kst,
                    ).get("position", {})
                    if isinstance(position_projection, dict) and position_projection:
                        state["position"] = position_projection
                    if latest_asset_decisions:
                        state["asset_decisions"] = dict(latest_asset_decisions)
                        state["asset_states"] = dict(latest_asset_decisions)
    
                    shadow_updates = {
                        key: state[key]
                        for key in (
                            "shadow_updated_at",
                            "governor_mode",
                            "price",
                            "position",
                            "session",
                            "microstructure",
                            "tail_risk",
                            "playbook",
                            "ensembles",
                            "quant_formula",
                            "account",
                            "agents",
                            "asset_decisions",
                            "asset_states",
                        )
                        if key in state
                    }
                    try:
                        latest_state = {}
                        if os.path.exists(DASHBOARD_STATE_PATH):
                            with open(DASHBOARD_STATE_PATH, "r", encoding="utf-8") as f:
                                latest_state = json.load(f)
                        if isinstance(latest_state, dict) and latest_state:
                            merged_state = dict(latest_state)
                            merged_state.update(shadow_updates)
                            state = merged_state
                        else:
                            state.update(shadow_updates)
                    except Exception:
                        state.update(shadow_updates)
                    state["governor_mode"] = True
    
                    dashboard_ok = False
                    compact_dashboard_ok = False
                    await _loop.run_in_executor(None, _atomic_write_json, DASHBOARD_STATE_PATH, state)
                    dashboard_ok = True
                    compact_state = _build_compact_dashboard_state(
                        state,
                        compact_meta_router,
                        float(_cur_price or state.get("price", 0.0) or 0.0),
                        now_kst,
                    )
                    await _loop.run_in_executor(None, _atomic_write_json, COMPACT_DASHBOARD_STATE_PATH, compact_state)
                    compact_dashboard_ok = True
                    if CONSOLE_LOG_COMPACT:
                        _now_log_ts = time.time()
                        if (_now_log_ts - _last_shadow_health_log_ts) >= max(10.0, CONSOLE_LOG_HEALTH_INTERVAL_SEC):
                            _last_shadow_health_log_ts = _now_log_ts
                            _log_compact_data_health(
                                state=state,
                                ms=ms,
                                tr=tr_pb,
                                dashboard_ok=dashboard_ok,
                                compact_dashboard_ok=compact_dashboard_ok,
                            )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if CONSOLE_LOG_COMPACT:
                        logger.warning("DATA store=BAD shadow_loop=%s", e)
                    else:
                        logger.debug("dashboard shadow loop skip: %s", e)
    
        async def _daily_trade_report_loop():
            while True:
                try:
                    now_kst = pd.Timestamp.now(tz="Asia/Seoul")
                    next_midnight = now_kst.normalize() + pd.Timedelta(days=1)
                    wait_sec = max(5.0, float((next_midnight - now_kst).total_seconds()) + 5.0)
                    await asyncio.sleep(wait_sec)
                    report_day = (pd.Timestamp.now(tz="Asia/Seoul") - pd.Timedelta(days=1)).normalize()
                    report_day_txt = report_day.strftime("%Y-%m-%d")
                    if _last_daily_report_date(DAILY_TRADE_REPORT_STATE_PATH) == report_day_txt:
                        continue
                    rows = _load_trade_journal_rows(TRADE_JOURNAL_PATH)
                    day_rows = _trade_rows_for_kst_day(rows, report_day)
                    message = _build_daily_trade_journal_message(report_day, day_rows)
                    await tg_notifier.notify(message)
                    _loop = asyncio.get_running_loop()
                    await _loop.run_in_executor(None, _save_daily_report_date, DAILY_TRADE_REPORT_STATE_PATH, report_day)
                    logger.info("?벂 daily trade journal sent for %s", report_day_txt)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning("?좑툘 daily trade journal skip: %s", e)
                    await asyncio.sleep(30.0)
    
        async def _fetch_exchange_position():
            snapshot = await fetcher.fetch_account_snapshot()
            return classify_account_position_snapshot(snapshot)
    
        def _bars_stale(eth_df: pd.DataFrame) -> bool:
            if eth_df is None or len(eth_df) == 0: return True
            last_ts = pd.Timestamp(eth_df['timestamp'].iloc[-1])
            if last_ts.tzinfo is not None: last_ts = last_ts.tz_localize(None)
            now_utc = pd.Timestamp.utcnow().tz_localize(None)
            if (now_utc - last_ts) > pd.Timedelta(minutes=15):
                logger.warning("⚠️ 최신 봉 지연 감지: last=%s age=%s", last_ts, (now_utc - last_ts))
                return True
            return False
    
        tg_notifier = TelegramNotifier()
        _execution_alert_deduper = ExecutionAlertDeduper()
        _execution_alert = build_execution_alert(
            _exec_status, observed_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat()
        )

        def _notify_execution_alert(alert: dict[str, object]) -> None:
            should_send = _execution_alert_deduper.should_notify(alert)
            try:
                dashboard_state = _read_json_safe(DASHBOARD_STATE_PATH)
                if isinstance(dashboard_state, dict):
                    dashboard_state["execution_alert"] = dict(alert)
                    _atomic_write_json(DASHBOARD_STATE_PATH, dashboard_state)
            except Exception as alert_state_error:
                logger.warning("execution alert dashboard write failed: %s", alert_state_error)
            if not should_send:
                return
            severity = str(alert.get("severity", "blocked")).upper()
            occurred_at = str(alert.get("occurred_at", "") or "-")
            reason = str(alert.get("reason", "") or "unknown_execution_state")
            message = (
                "<b>[트레이딩봇 실행 경고]</b>\n"
                f"level: <code>{html.escape(severity)}</code>\n"
                f"time: <code>{html.escape(occurred_at)}</code>\n"
                f"reason: <code>{html.escape(reason[:1000])}</code>"
            )
            task_supervisor.create(tg_notifier.notify(message), name="telegram-execution-alert")

        _notify_execution_alert(_execution_alert)

        def _notify_background_task_error(name: str, error: BaseException) -> None:
            logger.error("background_task_failed name=%s error=%s", name, error)
            alert = build_execution_alert(
                {
                    "requested_enabled": True,
                    "enabled": True,
                    "blocking": True,
                    "status": "background_task_error",
                    "error": f"{name}: {error}",
                },
                observed_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            )
            _notify_execution_alert(alert)

        task_supervisor.set_on_error(_notify_background_task_error)
        _patchtst_error_alert_last_sent: dict[str, float] = {}
    
        def _dedupe_ai_errors(errors: list[dict[str, object]]) -> list[dict[str, object]]:
            deduped: list[dict[str, object]] = []
            seen: set[tuple[str, str, str]] = set()
            for err in errors or []:
                if not isinstance(err, dict):
                    continue
                key = (
                    str(err.get("model", "")),
                    str(err.get("stage", "")),
                    str(err.get("error", ""))[:240],
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(dict(err))
            return deduped
    
        async def _notify_patchtst_ai_errors(errors: list[dict[str, object]], timestamp_kst) -> None:
            try:
                ts_text = timestamp_kst.isoformat()
            except Exception:
                ts_text = str(timestamp_kst)
            now = time.time()
            for err in _dedupe_ai_errors(errors):
                if str(err.get("model", "")).lower() != "patchtst":
                    continue
                error_text = str(err.get("error", "") or "unknown_error")
                stage = str(err.get("stage", "") or "unknown_stage")
                fallback = str(err.get("fallback", "") or "fallback")
                key = f"{stage}:{error_text[:240]}"
                if now - float(_patchtst_error_alert_last_sent.get(key, 0.0)) < float(PATCHTST_ERROR_ALERT_COOLDOWN_SEC):
                    continue
                _patchtst_error_alert_last_sent[key] = now
                msg = (
                    "<b>[PatchTST 예측 오류]</b>\n"
                    f"time: <code>{html.escape(ts_text)}</code>\n"
                    f"stage: <code>{html.escape(stage)}</code>\n"
                    f"fallback: <code>{html.escape(fallback)}</code>\n"
                    f"error: <code>{html.escape(error_text[:1000])}</code>"
                )
                task_supervisor.create(tg_notifier.notify(msg), name="telegram-ai-error")
    
        def _bar_get(row, key: str, default=None):
            try:
                if hasattr(row, "get"):
                    return row.get(key, default)
            except Exception:
                return default
            return default
    
        def _bar_get_float(row, key: str, default: float = 0.0) -> float:
            return _safe_float(_bar_get(row, key, default), default)
    
        def _bar_get_time_kst(row):
            try:
                return pd.Timestamp(_bar_get(row, "timestamp")) + pd.Timedelta(hours=9)
            except Exception:
                return pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    
        def _jsonable(value):
            if isinstance(value, dict):
                return {str(k): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, np.ndarray):
                return [_jsonable(v) for v in value.tolist()]
            if isinstance(value, (pd.Timestamp, datetime)):
                return str(value)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)
    
        async def _execute_pending_next_open(eth_buffer) -> None:
            nonlocal _prev_meta_pos, _pending_next_open_intent
    
            if not FINAL_GOVERNOR_SCHEDULE_NEXT_BAR_OPEN_ENABLE:
                if _pending_next_open_intent:
                    logger.info("SYSTEM pending_next_open dropped: disabled by backtest_next_open_contract")
                    _pending_next_open_intent = {}
                    try:
                        os.remove(FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH)
                    except OSError:
                        pass
                return
    
            intent = dict(_pending_next_open_intent or {})
            if not intent or eth_buffer is None or len(eth_buffer) == 0:
                return
    
            exec_bar = eth_buffer.iloc[-1]
            execution_time_kst = _bar_get_time_kst(exec_bar)
            try:
                execute_at = pd.Timestamp(intent.get("execute_at_kst", ""))
            except Exception:
                logger.warning("SYSTEM pending_next_open dropped: invalid execute_at=%s", str(intent.get("execute_at_kst", "")))
                _pending_next_open_intent = {}
                try:
                    os.remove(FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH)
                except OSError:
                    pass
                return
            if execute_at.tzinfo is not None:
                execute_at = execute_at.tz_convert("Asia/Seoul").tz_localize(None)
            exec_ts_cmp = pd.Timestamp(execution_time_kst)
            if exec_ts_cmp.tzinfo is not None:
                exec_ts_cmp = exec_ts_cmp.tz_convert("Asia/Seoul").tz_localize(None)
            if exec_ts_cmp < execute_at:
                return
    
            now_kst = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
            delay_sec = float((now_kst - exec_ts_cmp).total_seconds())
            exec_status = dict(live_executor.status())
            real_exchange_execution = bool(exec_status.get("enabled", False) and not bool(exec_status.get("dry_run", True)))
            max_delay = float(
                FINAL_GOVERNOR_NEXT_OPEN_MAX_DELAY_SEC
                if real_exchange_execution
                else FINAL_GOVERNOR_NEXT_OPEN_SHADOW_MAX_DELAY_SEC
            )
            if delay_sec < -2.0 or delay_sec > max(0.0, max_delay):
                logger.warning(
                    "SYSTEM pending_next_open skipped: execute_at=%s current_bar=%s delay=%.2fs max=%.2fs mode=%s",
                    str(intent.get("execute_at_kst", "")),
                    str(execution_time_kst),
                    delay_sec,
                    max_delay,
                    "real_exchange" if real_exchange_execution else "shadow",
                )
                _pending_next_open_intent = {}
                try:
                    os.remove(FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH)
                except OSError:
                    pass
                return
    
            current_price = float(_bar_get_float(exec_bar, "open", _safe_float(intent.get("decision_price", 0.0), 0.0)))
            if current_price <= 0.0:
                logger.warning("SYSTEM pending_next_open skipped: invalid execution open price")
                return
    
            final_action = int(intent.get("final_action", 0) or 0)
            target_exposure = float(intent.get("target_exposure", 0.0) or 0.0)
            target_fraction = float(intent.get("target_fraction", 0.0) or 0.0)
            target_exec_leverage = float(intent.get("target_exec_leverage", 1.0) or 1.0)
            active_info = dict(intent.get("active_info", {}) or {})
            active_info["scheduled_execution"] = True
            active_info["scheduled_decision_made_at_kst"] = str(intent.get("decision_made_at_kst", ""))
            active_info["scheduled_signal_bar_ts"] = str(intent.get("signal_bar_ts", ""))
            active_info["scheduled_execute_at_kst"] = str(intent.get("execute_at_kst", ""))
            active_info["live_bar_contract"] = "scheduled_signal_close_next_open"
            active_info["execution_bar_ts"] = str(execution_time_kst)
            active_info["execution_price"] = float(current_price)
            active_info["execution_price_source"] = "scheduled_next_bar_open"
            active_info["execution_delay_sec"] = float(delay_sec)
            active_info["execution_delay_late"] = bool(
                delay_sec > float(max(0.0, FINAL_GOVERNOR_NEXT_OPEN_WARN_DELAY_SEC))
            )
            active_info["execution_delay_mode"] = "real_exchange_strict" if real_exchange_execution else "shadow_extended"
            governor_source = str(intent.get("source", active_info.get("source", "FINAL_GOVERNOR")) or "FINAL_GOVERNOR")
            regime_name = str(intent.get("regime_name", "UNKNOWN") or "UNKNOWN")
            hold_reason = str(intent.get("hold_reason", active_info.get("position_reason", "")) or "")
            block_reason = ""
            prev_meta_pos = _prev_meta_pos
            prev_trade_snapshot = meta_router.position_snapshot()
    
            live_execution_result = dict(exec_status)
            live_execution_result.update({"ok": True, "blocking": False, "status": "disabled" if not live_executor.enabled else "pending"})
            if not use_local:
                live_execution_result = await live_executor.execute_to_target(
                    final_action=int(final_action),
                    target_exposure=float(target_exposure),
                    target_exec_leverage=float(target_exec_leverage),
                    current_price=float(current_price),
                    timestamp_kst=execution_time_kst,
                    decision_info=active_info,
                    existing_tp_sl_order_ids={
                        "tp_order_id": str(getattr(final_governor, "active_omega4_6_1_tp_order_id", "") or ""),
                        "sl_order_id": str(getattr(final_governor, "active_omega4_6_1_sl_order_id", "") or ""),
                    },
                )
                _tp_sl_result = dict(live_execution_result.get("tp_sl") or {})
                if _tp_sl_result.get("tp_order_id") or _tp_sl_result.get("sl_order_id"):
                    final_governor.active_omega4_6_1_tp_order_id = str(_tp_sl_result.get("tp_order_id", ""))
                    final_governor.active_omega4_6_1_sl_order_id = str(_tp_sl_result.get("sl_order_id", ""))
                    final_governor._save_runtime_state()
                _tp_sl_cancel_result = dict(live_execution_result.get("tp_sl_cancel") or {})
                if _tp_sl_cancel_result.get("tp_cancelled") or _tp_sl_cancel_result.get("sl_cancelled"):
                    final_governor.active_omega4_6_1_tp_order_id = ""
                    final_governor.active_omega4_6_1_sl_order_id = ""
                    final_governor._save_runtime_state()
            active_info["live_execution"] = dict(live_execution_result)
            if bool(live_execution_result.get("blocking", False)):
                logger.warning("SYSTEM pending_next_open binance_execution blocked: %s", live_execution_result)
                return
    
            _pending_entry_route = GovernorPositionRouter._execution_route_summary(
                list(live_execution_result.get("orders", []) or []),
                reduce_only=False,
            )
            meta_router._update_pos(
                final_action,
                current_price,
                intent.get("decision_made_at_kst") or intent.get("signal_bar_ts") or execution_time_kst,
                target_exposure,
                fraction=target_fraction,
                leverage_mult=target_exec_leverage,
                trend_signal=None,
                entry_price_source_override="scheduled_next_bar_open",
                entry_decision_price_override=float(intent.get("decision_price", current_price) or current_price),
                entry_execution_liquidity_override=str(_pending_entry_route.get("liquidity", "")),
                entry_execution_route_override=str(_pending_entry_route.get("route", "")),
                entry_execution_order_type_override=str(_pending_entry_route.get("order_type", "")),
            )
            meta_router.update_adaptive_gate(final_action=int(final_action), in_position=(meta_router.pos is not None))
    
            new_pos = meta_router.pos
            new_trade_snapshot = meta_router.position_snapshot()
            audit_ctx = dict(intent.get("audit_context", {}) or {})
            audit_ctx.update({
                "ledger_ts_kind": "scheduled_next_bar_open_execution",
                "decision_made_at_kst": str(intent.get("decision_made_at_kst", "")),
                "execution_bar_ts": str(execution_time_kst),
                "execution_bar_utc": str(_bar_get(exec_bar, "timestamp", "")),
                "execution_bar_open": _bar_get_float(exec_bar, "open", current_price),
                "execution_bar_high": _bar_get_float(exec_bar, "high", current_price),
                "execution_bar_low": _bar_get_float(exec_bar, "low", current_price),
                "execution_bar_close": _bar_get_float(exec_bar, "close", current_price),
                "execution_bar_volume": _bar_get_float(exec_bar, "volume", 0.0),
                "execution_bar_is_current": True,
                "execution_price": float(current_price),
                "execution_price_source": "scheduled_next_bar_open",
                "execution_delay_sec": float(delay_sec),
                "execution_delay_late": bool(active_info.get("execution_delay_late", False)),
                "execution_delay_mode": str(active_info.get("execution_delay_mode", "")),
                "live_execution": dict(live_execution_result),
            })
    
            transition_label = _pos_transition_label(prev_meta_pos, new_pos)
            resized = (
                prev_meta_pos is not None
                and new_pos == prev_meta_pos
                and (
                    abs(float(new_trade_snapshot.get("position_fraction", 0.0) or 0.0) - float(prev_trade_snapshot.get("position_fraction", 0.0) or 0.0)) > 1e-9
                    or abs(float(new_trade_snapshot.get("execution_leverage", 1.0) or 1.0) - float(prev_trade_snapshot.get("execution_leverage", 1.0) or 1.0)) > 1e-9
                    or abs(float(new_trade_snapshot.get("total_exposure", 0.0) or 0.0) - float(prev_trade_snapshot.get("total_exposure", 0.0) or 0.0)) > 1e-9
                )
            )
            trade_rows: list[dict] = []
            audit_rows: list[dict] = []
            close_payload: dict | None = None
            open_payload: dict | None = None
            equity_cursor = _accounting_equity_from_history(getattr(meta_router, "trade_history", []))
    
            if prev_meta_pos is not None and new_pos != prev_meta_pos:
                close_payload = meta_router.build_close_trade_payload(
                    snapshot=prev_trade_snapshot,
                    current_price=current_price,
                    timestamp_kst=execution_time_kst,
                    event=transition_label,
                    regime_name=regime_name,
                    source=governor_source,
                    reason=str(block_reason or hold_reason or active_info.get("position_reason", "") or governor_source),
                    next_side=new_pos,
                    audit_context=audit_ctx,
                )
                realized = float(close_payload.get("pnl_frac", 0.0))
                before = float(equity_cursor)
                after = float(before * max(0.0, 1.0 + realized))
                audit_rows.append(_build_position_accounting_audit_row(
                    trade_row=close_payload,
                    equity_before=before,
                    equity_after=after,
                    prev_snapshot=prev_trade_snapshot,
                    new_snapshot=new_trade_snapshot,
                    current_price=current_price,
                    from_pos=prev_meta_pos,
                    to_pos=new_pos,
                    decision_info=active_info,
                    fee_rate=float(meta_router.trade_fee),
                    slippage_rate=float(meta_router.trade_slip),
                ))
                equity_cursor = after
                enhanced_engine.on_trade_close(realized)
                meta_router.record_outcome(realized)
                meta_router.append_trade_history(execution_time_kst, realized, payload=close_payload)
                trade_rows.append(dict(close_payload))
    
            if new_pos is not None and new_pos != prev_meta_pos:
                enhanced_engine.on_position_open()
                open_payload = meta_router.build_open_trade_payload(
                    snapshot=new_trade_snapshot,
                    timestamp_kst=execution_time_kst,
                    event=transition_label,
                    regime_name=regime_name,
                    source=governor_source,
                    reason=str(active_info.get("position_reason", "") or hold_reason or governor_source),
                    audit_context=audit_ctx,
                )
                audit_rows.append(_build_position_accounting_audit_row(
                    trade_row=open_payload,
                    equity_before=float(equity_cursor),
                    equity_after=float(equity_cursor),
                    prev_snapshot=prev_trade_snapshot,
                    new_snapshot=new_trade_snapshot,
                    current_price=current_price,
                    from_pos=prev_meta_pos,
                    to_pos=new_pos,
                    decision_info=active_info,
                    fee_rate=float(meta_router.trade_fee),
                    slippage_rate=float(meta_router.trade_slip),
                ))
                trade_rows.append(dict(open_payload))
            elif resized:
                resize_payload = meta_router.build_resize_trade_payload(
                    prev_snapshot=prev_trade_snapshot,
                    new_snapshot=new_trade_snapshot,
                    current_price=current_price,
                    timestamp_kst=execution_time_kst,
                    regime_name=regime_name,
                    source=governor_source,
                    reason=str(active_info.get("position_reason", "") or hold_reason or governor_source),
                    audit_context=audit_ctx,
                )
                resize_realized = float(resize_payload.get("pnl_frac", 0.0) or 0.0)
                before = float(equity_cursor)
                after = float(before * max(0.0, 1.0 + resize_realized))
                audit_rows.append(_build_position_accounting_audit_row(
                    trade_row=resize_payload,
                    equity_before=before,
                    equity_after=after,
                    prev_snapshot=prev_trade_snapshot,
                    new_snapshot=new_trade_snapshot,
                    current_price=current_price,
                    from_pos=prev_meta_pos,
                    to_pos=new_pos,
                    decision_info=active_info,
                    fee_rate=float(meta_router.trade_fee),
                    slippage_rate=float(meta_router.trade_slip),
                ))
                if abs(resize_realized) > 1e-12:
                    meta_router.append_trade_history(execution_time_kst, resize_realized, payload=resize_payload)
                trade_rows.append(dict(resize_payload))
    
            if trade_rows:
                await journal_writer.append_many(TRADE_JOURNAL_PATH, trade_rows)
            if audit_rows:
                await journal_writer.append_many(POSITION_ACCOUNTING_AUDIT_PATH, audit_rows)
            if prev_meta_pos != new_pos:
                await journal_writer.append(DASHBOARD_EVENTS_PATH, {
                    "ts": str(execution_time_kst),
                    "event": transition_label,
                    "from": prev_meta_pos,
                    "to": new_pos,
                    "price": float(current_price),
                    "kelly": float(target_exposure),
                    "pnl_pct": float((close_payload or {}).get("pnl_pct", 0.0) or 0.0),
                    "regime": str(regime_name),
                    "scheduled_next_open": True,
                    "close_trade": close_payload,
                    "open_trade": open_payload,
                })
    
            _prev_meta_pos = new_pos
            _pending_next_open_intent = {}
            try:
                os.remove(FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH)
            except OSError:
                pass
            logger.info(
                "SYSTEM pending_next_open executed: decision_made=%s signal=%s execution=%s action=%s pos=%s price=%.4f delay=%.2fs source=%s",
                str(intent.get("decision_made_at_kst", "")),
                str(intent.get("signal_bar_ts", "")),
                str(execution_time_kst),
                int(final_action),
                str(new_pos or "NONE"),
                float(current_price),
                float(delay_sec),
                governor_source,
            )
    
        async def _refresh_omega461_shadow_asset(asset_key: str) -> dict:
            ctx = omega461_shadow_assets[asset_key]
            cfg = dict(ctx["cfg"])
            asset_fetcher: BinanceLiveFetcher = ctx["fetcher"]
            asset_fe: FeatureEngineer = ctx["fe_engine"]
            router: GovernorPositionRouter = ctx["router"]
            asset_regime = ""
            try:
                if ctx.get("eth_buffer") is None or ctx.get("btc_buffer") is None:
                    asset_eth, asset_btc = await asset_fetcher.fetch_initial_data()
                else:
                    new_eth, new_btc = await asset_fetcher.fetch_latest_patch()
                    keep_bars = max(7000, int(FINAL_GOVERNOR_BUFFER_BARS), int(FINAL_GOVERNOR_MACRO_LOOKBACK_BARS) + 512)
                    asset_eth = pd.concat([ctx["eth_buffer"], new_eth]).drop_duplicates("timestamp").tail(keep_bars)
                    asset_btc = pd.concat([ctx["btc_buffer"], new_btc]).drop_duplicates("timestamp").tail(keep_bars)
                ctx["eth_buffer"] = asset_eth
                ctx["btc_buffer"] = asset_btc
                process_eth = asset_eth
                process_btc = asset_btc
                if FINAL_GOVERNOR_LIVE_PROCESS_BARS > 0:
                    tail_n = int(max(600, FINAL_GOVERNOR_LIVE_PROCESS_BARS))
                    process_eth = asset_eth.tail(tail_n)
                    process_btc = asset_btc.tail(tail_n)
                processed_full = asset_fe.process(process_eth, process_btc)
                processed = processed_full
                if FINAL_GOVERNOR_LIVE_MODEL_BARS > 0:
                    processed = processed_full.tail(int(max(600, FINAL_GOVERNOR_LIVE_MODEL_BARS))).copy()
                if processed is None or len(processed) == 0:
                    raise RuntimeError("empty shadow processed frame")
                if ctx.get("btc_swing_transition") is not None:
                    # Per-bar causal computation of swing_transition_prob for the promoted BTC
                    # bundle (2026-08-07). asset_eth is this asset's OWN full kline buffer (the
                    # eth/btc naming is the fetcher's legacy return convention). Raises on any
                    # missing input or Deribit DVOL failure -- caught by this function's own
                    # error handling like every other per-asset refresh failure (no silent
                    # degraded-feature trading).
                    # layerA consumes the adapter's regime3_current_* columns, which the adapter
                    # normally appends INSIDE decide_entry -- pre-append them here so the swing
                    # provider sees them (found 2026-08-07 in the multislot shadow loop smoke
                    # test; the adapter recomputes/overwrites them identically afterwards, so
                    # this only fixes ordering, it does not change decision inputs).
                    processed = ctx["adapter"].regime3_current.append(processed)
                    processed = ctx["btc_swing_transition"].append(processed, raw_5m=asset_eth)
                try:
                    regime_frame = final_governor._ensure_regime_features(
                        processed.tail(final_governor.window_bars).copy().reset_index(drop=True)
                    )
                    asset_regime = str(final_governor._raw_regime_from_row(regime_frame.iloc[-1]) or "normal").lower()
                except Exception as regime_e:
                    logger.warning("SYSTEM omega4_6_1_shadow asset=%s regime_classify_failed reason=%s", asset_key, regime_e)
                    asset_regime = ""
                price = float(pd.to_numeric(asset_eth["close"], errors="coerce").iloc[-1])
                updated_at = pd.Timestamp(asset_eth["timestamp"].iloc[-1])
                timestamp_kst = updated_at + pd.Timedelta(hours=9)
                if ctx.get("orderbook_recorder") is not None:
                    try:
                        await ctx["orderbook_recorder"].record_decision_snapshot(
                            asset_fetcher,
                            timestamp_kst=timestamp_kst,
                            context={"record_reason": "omega4_6_1_shadow_decision", "asset": asset_key, "price": price},
                        )
                    except Exception as _ob_e:
                        logger.warning("SYSTEM %s orderbook_recorder failed: %s", asset_key, _ob_e)
                prev_pos = router.pos if router.pos in {"LONG", "SHORT"} else None
                prev_snapshot = router.position_snapshot()
                active = dict(ctx.get("active", {}) or {})
                action = 0
                source = "omega4_6_1_shadow|cash"
                reason = "omega4_6_1_cash"
                margin_fraction = 0.0
                leverage = 1.0
                notional = 0.0
    
                if prev_pos in {"LONG", "SHORT"}:
                    active = validate_omega461_shadow_active_state(
                        active,
                        asset_key=asset_key,
                        expected_component=str(cfg["component"]),
                        position=prev_pos,
                        entry_price=float(router.entry_price or 0.0),
                        position_fraction=float(router.position_fraction or 0.0),
                        execution_leverage=float(router.execution_leverage or 1.0),
                        notional_exposure=float(router.current_leverage or 0.0),
                    )
                    side = 1 if prev_pos == "LONG" else -1
                    entry = float(router.entry_price or 0.0)
                    move = (price - entry) / entry if side > 0 and entry > 0 else ((entry - price) / entry if entry > 0 else 0.0)
                    active["mfe"] = max(float(active.get("mfe", 0.0) or 0.0), float(move))
                    active["mae"] = min(float(active.get("mae", 0.0) or 0.0), float(move))
                    if not active.get("source_component") or entry <= 0.0:
                        raise RuntimeError(
                            f"omega4_6_1_shadow_active_contract_mismatch asset={asset_key} "
                            "field=runtime_state detail=unarmed_open_position"
                        )
                    else:
                        # See _manage_omega4_6_1_position: TP/SL barriers use the completed bar's
                        # high/low so a wick that touches the barrier isn't missed just because the
                        # bar's close fell back short of it.
                        bar_high = float(processed["high"].iloc[-1])
                        bar_low = float(processed["low"].iloc[-1])
                        bar_high_move = (bar_high - entry) / entry if side > 0 else (entry - bar_low) / entry
                        bar_low_move = (bar_low - entry) / entry if side > 0 else (entry - bar_high) / entry
                        should_exit, reason_key, exit_prob = ctx["adapter"].evaluate_exit(
                            processed,
                            source_component=str(active["source_component"]),
                            side=side,
                            hold_bars=int(router.hold_count or 0),
                            unrealized_move=float(move),
                            mfe=float(active.get("mfe", 0.0) or 0.0),
                            mae=float(active.get("mae", 0.0) or 0.0),
                            notional=float(active.get("notional_exposure", router.current_leverage) or 0.0),
                            leverage=float(active.get("leverage", router.execution_leverage) or 1.0),
                            take_profit=float(active.get("take_profit", 0.0) or 0.0),
                            stop_loss=float(active.get("stop_loss", 0.0) or 0.0),
                            bar_high_move=bar_high_move,
                            bar_low_move=bar_low_move,
                        )
                    if should_exit:
                        action = 0
                        reason = f"omega4_6_1_shadow_{reason_key}"
                        source = f"omega4_6_1_shadow|{reason_key}"
                    else:
                        action = _omega461_shadow_action_for_pos(prev_pos)
                        reason = "omega4_6_1_shadow_hold"
                        source = f"omega4_6_1_shadow|{active.get('source_component', 'hold')}"
                        margin_fraction = float(router.position_fraction or 0.0)
                        leverage = float(router.execution_leverage or 1.0)
                        notional = float(router.current_leverage or 0.0)
                        active["reason"] = reason
                        active["source"] = source
                        active["exit_probability"] = float(exit_prob)
                else:
                    decision = ctx["adapter"].decide_entry(processed)
                    if decision is not None and ctx.get("btc_cmamba_gate") is not None:
                        try:
                            _cmamba_sig = ctx["btc_cmamba_gate"].direction_signal(processed)
                            _cmamba_status = direction_overlay_status(
                                entry_side=int(decision.side),
                                predicted_direction=_cmamba_sig,
                            )
                        except Exception as _cmamba_gate_exc:
                            logger.error("SYSTEM btc_cmamba_entry_gate=FAILED err=%s", _cmamba_gate_exc, exc_info=True)
                            _cmamba_status = EntryOverlayStatus.UNAVAILABLE
                        if _cmamba_status is not EntryOverlayStatus.PASS:
                            decision = None
                    if decision is not None:
                        cand_margin_fraction = float(decision.margin_fraction)
                        cand_leverage = float(decision.leverage)
                        cand_notional = float(decision.notional_exposure)
                        if asset_key == "sol" and float(FINAL_GOVERNOR_OMEGA4_6_1_SOL_NOTIONAL_MULTIPLIER) != 1.0:
                            cand_notional *= float(FINAL_GOVERNOR_OMEGA4_6_1_SOL_NOTIONAL_MULTIPLIER)
                            cand_leverage = cand_notional / max(cand_margin_fraction, 1e-12)
                        risk_blocked = False
                        if ctx.get("executor") is not None:
                            approved_notional = omega461_portfolio_risk.scale_to_budget(asset_key, cand_notional)
                            if approved_notional < omega461_portfolio_risk.config.min_notional:
                                risk_blocked = True
                            elif approved_notional < cand_notional - 1e-9:
                                cand_leverage = approved_notional / max(cand_margin_fraction, 1e-12)
                                cand_notional = approved_notional
                        if not risk_blocked:
                            final_sizing = finalize_sizing(
                                margin_fraction=cand_margin_fraction,
                                requested_notional=cand_notional,
                                max_leverage=OMEGA4_6_1_LEVERAGE_CAP,
                                max_notional=OMEGA4_6_1_NOTIONAL_CAP,
                            )
                            cand_margin_fraction = final_sizing.margin_fraction
                            cand_leverage = final_sizing.leverage
                            cand_notional = final_sizing.notional
                            action = 1 if int(decision.side) > 0 else 2
                            margin_fraction = cand_margin_fraction
                            leverage = cand_leverage
                            notional = cand_notional
                            source = f"omega4_6_1_shadow|{decision.source_component}"
                            reason = "omega4_6_1_shadow_entry"
                            active = {
                                "contract_version": OMEGA4_6_1_SHADOW_ACTIVE_CONTRACT,
                                "side": "LONG" if action == 1 else "SHORT",
                                "source_component": str(decision.source_component),
                                "margin_fraction": margin_fraction,
                                "leverage": leverage,
                                "notional_exposure": notional,
                                "take_profit": float(decision.take_profit),
                                "stop_loss": float(decision.stop_loss),
                                "quality_score": float(decision.quality_score),
                                "confidence": float(decision.confidence),
                                "trace": dict(decision.trace or {}),
                                "mfe": 0.0,
                                "mae": 0.0,
                                "source": source,
                                "reason": reason,
                                "entry_price": price,
                            }
                        else:
                            reason = "omega4_6_1_shadow_portfolio_risk_blocked"
                            source = "omega4_6_1_shadow|cash"
    
                target_pos = "LONG" if int(action) == 1 else ("SHORT" if int(action) == 2 else None)
                persisted_active: dict = {}
                if target_pos is not None:
                    target_entry = float(router.entry_price or 0.0) if prev_pos is not None else float(price)
                    persisted_active = validate_omega461_shadow_active_state(
                        active,
                        asset_key=asset_key,
                        expected_component=str(cfg["component"]),
                        position=target_pos,
                        entry_price=target_entry,
                        position_fraction=float(margin_fraction),
                        execution_leverage=float(leverage),
                        notional_exposure=float(notional),
                    )

                real_execution_result: dict | None = None
                if ctx.get("executor") is not None:
                    real_execution_result = await ctx["executor"].execute_to_target(
                        final_action=int(action),
                        target_exposure=float(notional),
                        target_exec_leverage=float(leverage if action else 1.0),
                        current_price=float(price),
                        timestamp_kst=timestamp_kst,
                        decision_info=dict(active),
                    )

                # From here on, a real order may already have been placed/changed on the exchange
                # (whenever real_execution_result is not None). A failure below is therefore NOT
                # equivalent to a pre-execution decision failure: local position state and the trade
                # journal could fall out of sync with what the exchange actually did. Keep this in its
                # own try block so that case gets a distinct, loudly-logged status instead of being
                # reported the same way as a harmless feature/decision error (the next 15s reconcile
                # loop is what actually fixes the drift; this only makes the gap visible).
                try:
                    audit_context = _omega461_shadow_audit_context(
                        asset_key=asset_key,
                        cfg=cfg,
                        price=price,
                        updated_at=updated_at,
                        source=source,
                        reason=reason,
                        active=active,
                        real_execution_result=real_execution_result,
                    )
                    if target_pos is None:
                        router.strategy_state.pop(OMEGA4_6_1_SHADOW_ACTIVE_STATE_KEY, None)
                    else:
                        router.strategy_state[OMEGA4_6_1_SHADOW_ACTIVE_STATE_KEY] = persisted_active
                    router._update_pos(
                        action,
                        price,
                        timestamp_kst,
                        fraction=margin_fraction if action else None,
                        leverage_mult=leverage if action else None,
                        entry_price_source_override="shadow_bar_close",
                        entry_decision_price_override=price,
                    )
                    new_pos = router.pos if router.pos in {"LONG", "SHORT"} else None
                    new_snapshot = router.position_snapshot()
                    if new_pos is not None:
                        active["entry_price"] = float(new_snapshot.get("entry_price", price) or price)
                        active["reason"] = reason
                        active["source"] = source
                        ctx["active"] = active
                    else:
                        ctx["active"] = {}

                    rows: list[dict] = []
                    transition = _pos_transition_label(prev_pos, new_pos)
                    if prev_pos is not None and new_pos != prev_pos:
                        close_row = router.build_close_trade_payload(
                            snapshot=prev_snapshot,
                            current_price=price,
                            timestamp_kst=timestamp_kst,
                            event=transition,
                            regime_name="SHADOW",
                            source=source,
                            reason=reason,
                            next_side=new_pos,
                            audit_context=audit_context,
                        )
                        realized = float(close_row.get("pnl_frac", 0.0) or 0.0)
                        router.record_outcome(realized)
                        router.append_trade_history(timestamp_kst, realized, payload=close_row)
                        rows.append(_omega461_shadow_decorate_trade_row(close_row, asset_key=asset_key, cfg=cfg, real_execution_result=real_execution_result))
                    if new_pos is not None and new_pos != prev_pos:
                        open_row = router.build_open_trade_payload(
                            snapshot=new_snapshot,
                            timestamp_kst=timestamp_kst,
                            event=transition,
                            regime_name="SHADOW",
                            source=source,
                            reason=reason,
                            audit_context=audit_context,
                        )
                        rows.append(_omega461_shadow_decorate_trade_row(open_row, asset_key=asset_key, cfg=cfg, real_execution_result=real_execution_result))
                    if rows:
                        await journal_writer.append_many(TRADE_JOURNAL_PATH, rows)
                        await journal_writer.append(DASHBOARD_EVENTS_PATH, {
                            "ts": str(timestamp_kst),
                            "event": transition,
                            "from": prev_pos,
                            "to": new_pos,
                            "price": float(price),
                            "asset": str(asset_key),
                            "symbol": str(cfg.get("symbol", "")),
                            "shadow_only": real_execution_result is None,
                            "source": source,
                            "close_trade": rows[0] if prev_pos is not None and new_pos != prev_pos else None,
                            "open_trade": rows[-1] if new_pos is not None and new_pos != prev_pos else None,
                        })

                    return _omega461_shadow_state_from_router(
                        asset_key=asset_key,
                        cfg=cfg,
                        router=router,
                        active=dict(ctx.get("active", {}) or active),
                        current_price=price,
                        updated_at=updated_at,
                        regime=asset_regime,
                    )
                except Exception as post_exec_e:
                    if real_execution_result is not None:
                        logger.critical(
                            "SYSTEM omega4_6_1_shadow asset=%s DESYNC_RISK a real order call completed but "
                            "local position state/journal update failed afterward -- exchange and local "
                            "state may now disagree until the next reconcile pass: reason=%s",
                            asset_key, post_exec_e,
                        )
                    raise
            except Exception as e:
                logger.warning("SYSTEM omega4_6_1_shadow asset=%s status=BAD reason=%s", asset_key, e)
                fallback_price = 0.0
                fallback_ts = pd.Timestamp.utcnow()
                try:
                    buf = ctx.get("eth_buffer")
                    if isinstance(buf, pd.DataFrame) and len(buf):
                        fallback_price = float(pd.to_numeric(buf["close"], errors="coerce").iloc[-1])
                        fallback_ts = pd.Timestamp(buf["timestamp"].iloc[-1])
                except Exception:
                    pass
                return _omega461_shadow_error_state(
                    asset_key=asset_key,
                    cfg=cfg,
                    router=router,
                    active=dict(ctx.get("active", {}) or {}),
                    current_price=fallback_price,
                    updated_at=fallback_ts,
                    error=str(e),
                    regime=asset_regime,
                )
    
        async def _refresh_omega461_shadow_assets() -> dict[str, dict]:
            if not omega461_shadow_assets:
                return {}
            asset_keys = sorted(omega461_shadow_assets)
            values = await asyncio.gather(
                *[_refresh_omega461_shadow_asset(asset_key) for asset_key in asset_keys],
                return_exceptions=True,
            )
            out: dict[str, dict] = {}
            for asset_key, value in zip(asset_keys, values):
                if isinstance(value, Exception):
                    ctx = omega461_shadow_assets[asset_key]
                    cfg = dict(ctx["cfg"])
                    out[asset_key] = _omega461_shadow_error_state(
                        asset_key=asset_key,
                        cfg=cfg,
                        router=ctx["router"],
                        active=dict(ctx.get("active", {}) or {}),
                        current_price=0.0,
                        updated_at=pd.Timestamp.utcnow(),
                        error=str(value),
                    )
                else:
                    out[asset_key] = value
            return out
    
        async def _run_cycle(processed_df, eth_buffer):
            """한 사이클: final governor 판단 + 집행."""
            nonlocal _prev_meta_pos, _pending_next_open_intent, _last_data_pipeline_health_log_ts, latest_asset_decisions
    
            meta_router.decrement_cooldown()
    
            if processed_df is None or eth_buffer is None or len(processed_df) == 0 or len(eth_buffer) == 0:
                logger.warning("SYSTEM cycle skipped: empty processed_df or eth_buffer")
                return
    
            if omega461_shadow_assets:
                latest_asset_decisions.update(await _refresh_omega461_shadow_assets())
    
            _raw_processed_df = processed_df
            _raw_eth_buffer = eth_buffer
            _next_open_execution = bool(
                FINAL_GOVERNOR_NEXT_OPEN_EXECUTION_ENABLE
                and len(_raw_processed_df) >= 2
                and len(_raw_eth_buffer) >= 2
            )
            try:
                _pre_exec_status = dict(live_executor.status())
            except Exception:
                _pre_exec_status = {}
            _pre_real_exchange_execution = bool(
                _pre_exec_status.get("enabled", False)
                and not bool(_pre_exec_status.get("dry_run", True))
            )
            _live_completed_bar_next_open_proxy = bool(
                FINAL_GOVERNOR_LIVE_COMPLETED_BAR_NEXT_OPEN_PROXY
                and not _pre_real_exchange_execution
            )
            if _next_open_execution:
                # Keep the live contract aligned with the backtest contract:
                # signal on the previous completed candle, execute at the next
                # candle open. Do not proxy next-open execution with signal close.
                _signal_bar = _raw_eth_buffer.iloc[-2]
                _execution_bar = _raw_eth_buffer.iloc[-1]
                if "timestamp" in getattr(_raw_processed_df, "columns", []):
                    try:
                        _signal_ts = pd.Timestamp(_signal_bar.get("timestamp"))
                        _proc_ts = pd.to_datetime(_raw_processed_df["timestamp"])
                        _aligned_processed = _raw_processed_df.loc[_proc_ts <= _signal_ts].copy()
                        if len(_aligned_processed):
                            processed_df = _aligned_processed
                        elif _live_completed_bar_next_open_proxy:
                            processed_df = _raw_processed_df.copy()
                        else:
                            processed_df = _raw_processed_df.iloc[:-1].copy()
                    except Exception:
                        processed_df = _raw_processed_df.copy() if _live_completed_bar_next_open_proxy else _raw_processed_df.iloc[:-1].copy()
                else:
                    processed_df = _raw_processed_df.copy() if _live_completed_bar_next_open_proxy else _raw_processed_df.iloc[:-1].copy()
                eth_buffer = _raw_eth_buffer.copy() if _live_completed_bar_next_open_proxy else _raw_eth_buffer.iloc[:-1].copy()
            else:
                processed_df = _raw_processed_df.copy()
                eth_buffer = _raw_eth_buffer.copy()
                _signal_bar = _raw_eth_buffer.iloc[-1]
                _execution_bar = _raw_eth_buffer.iloc[-1]
    
            def _bar_value(row, key: str, default=None):
                try:
                    if hasattr(row, "get"):
                        return row.get(key, default)
                except Exception:
                    return default
                return default
    
            def _bar_float(row, key: str, default: float = 0.0) -> float:
                return _safe_float(_bar_value(row, key, default), default)
    
            def _bar_time_kst(row):
                try:
                    return pd.Timestamp(_bar_value(row, "timestamp")) + pd.Timedelta(hours=9)
                except Exception:
                    return pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    
            decision_time_kst = _bar_time_kst(_signal_bar)
            execution_time_kst = (
                decision_time_kst + pd.Timedelta(minutes=5)
                if _next_open_execution and _live_completed_bar_next_open_proxy
                else _bar_time_kst(_execution_bar) if _next_open_execution else decision_time_kst
            )
            decision_price = float(_bar_float(_signal_bar, "close", 0.0))
            execution_price = float(
                _bar_float(
                    _execution_bar,
                    "open" if _next_open_execution else "close",
                    decision_price,
                )
            )
            if decision_price <= 0.0:
                logger.warning("SYSTEM cycle skipped: invalid decision price %.8f", decision_price)
                return
            if execution_price <= 0.0:
                execution_price = float(decision_price)

            try:
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    _atomic_write_json,
                    DATA_PIPELINE_DECISION_HEARTBEAT_PATH,
                    {
                        "schema_version": "live.trading_bot_decision_heartbeat.v1",
                        "status": "cycle_input_ready",
                        "recorded_at_kst": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                        "decision_bar_ts": str(decision_time_kst),
                        "execution_bar_ts": str(execution_time_kst),
                        "decision_price": float(decision_price),
                        "execution_price": float(execution_price),
                        "next_open_execution": bool(_next_open_execution),
                    },
                )
            except Exception as _heartbeat_error:
                logger.warning("SYSTEM decision_heartbeat=FAILED reason=%s", _heartbeat_error)

            _ai_runtime_errors: list[dict[str, object]] = []

            async def _persist_data_pipeline_snapshot(active_info: dict, *, stage: str, error: Exception | None = None) -> None:
                nonlocal _last_data_pipeline_health_log_ts
                if not DATA_PIPELINE_HEALTH_ENABLE:
                    return
                _pipe_active = dict(active_info or {})
                _pipe_active.setdefault("agent", "FINAL_GOVERNOR")
                _pipe_active.setdefault("source", "FINAL_GOVERNOR")
                _pipe_active.setdefault("position_signal", "HOLD")
                _pipe_active.setdefault("position_reason", stage)
                _pipe_active.setdefault("final_action", 0)
                _pipe_active.setdefault("score", 0.0)
                _pipe_active.setdefault("conviction", 0.0)
                _pipe_active.setdefault(
                    "execution_price_source",
                    "eth_buffer.open[-1]" if _next_open_execution else "eth_buffer.close[-1]",
                )
                _pipe_active.setdefault("execution_delay_sec", float(_execution_delay_sec))
                _pipe_active.setdefault("execution_delay_late", bool(_next_open_delay_late))
                _pipe_active.setdefault(
                    "execution_delay_expected_proxy",
                    bool(_live_completed_bar_next_open_proxy and not _real_exchange_execution),
                )
                if error is not None:
                    _pipe_active["pipeline_error"] = str(error)
                    _pipe_active["pipeline_error_stage"] = str(stage)
                    _pipe_active["source"] = "FINAL_GOVERNOR_ERROR"
                    _pipe_active["position_reason"] = str(stage)
                try:
                    _pipe_health = _build_data_pipeline_health(
                        raw_processed_df=_raw_processed_df,
                        processed_df=processed_df,
                        raw_eth_buffer=_raw_eth_buffer,
                        eth_buffer=eth_buffer,
                        signal_bar=_signal_bar,
                        execution_bar=_execution_bar,
                        next_open_execution=_next_open_execution,
                        decision_price=float(decision_price),
                        execution_price=float(execution_price),
                        active_info=_pipe_active,
                        runtime_predictor=runtime_predictor,
                        final_governor=final_governor,
                        ai_errors=list(_ai_runtime_errors or []),
                    )
                    _pipe_health["pipeline_stage"] = str(stage)
                    if error is not None:
                        _warnings = list(_pipe_health.get("warnings", []) or [])
                        if "final_governor_pipeline_error" not in _warnings:
                            _warnings.append("final_governor_pipeline_error")
                        _pipe_health["warnings"] = _warnings
                        _pipe_health["status"] = "WARN"
                        _pipe_health["pipeline_error"] = {
                            "stage": str(stage),
                            "message": str(error),
                            "type": type(error).__name__,
                        }
                    if DATA_PIPELINE_FEATURE_SNAPSHOT_ENABLE:
                        _prev_snapshot_age = None
                        try:
                            if os.path.exists(DATA_PIPELINE_FEATURE_FRAME_SNAPSHOT_PATH):
                                _prev_snapshot_age = float(time.time() - os.path.getmtime(DATA_PIPELINE_FEATURE_FRAME_SNAPSHOT_PATH))
                        except OSError:
                            _prev_snapshot_age = None
                        _stale_before_write = bool(_prev_snapshot_age is not None and _prev_snapshot_age > 660.0)
                        _pipe_health["snapshot_watchdog"] = {
                            "path": DATA_PIPELINE_FEATURE_FRAME_SNAPSHOT_PATH,
                            "previous_age_sec": _prev_snapshot_age,
                            "stale_before_write": _stale_before_write,
                            "stale_threshold_sec": 660.0,
                        }
                        if _stale_before_write:
                            _warnings = list(_pipe_health.get("warnings", []) or [])
                            if "DATA_SNAPSHOT_STALE" not in _warnings:
                                _warnings.append("DATA_SNAPSHOT_STALE")
                            _pipe_health["warnings"] = _warnings
                            _pipe_health["status"] = "WARN"
                    _pipe_loop = asyncio.get_running_loop()
                    await _pipe_loop.run_in_executor(None, _atomic_write_json, DATA_PIPELINE_HEALTH_PATH, _pipe_health)
                    await journal_writer.append(DATA_PIPELINE_HEALTH_JSONL_PATH, _pipe_health)
                    if DATA_PIPELINE_FEATURE_SNAPSHOT_ENABLE:
                        _snapshot_frame = getattr(final_governor, "last_prepared_frame_for_health", None)
                        if not isinstance(_snapshot_frame, pd.DataFrame) or not len(_snapshot_frame):
                            _snapshot_frame = processed_df
                        _feature_snapshot = _build_decision_feature_snapshot(_snapshot_frame, _pipe_active, _pipe_health)
                        if _feature_snapshot:
                            await _pipe_loop.run_in_executor(None, _atomic_write_json, DATA_PIPELINE_FEATURE_SNAPSHOT_PATH, _feature_snapshot)
                            await journal_writer.append(DATA_PIPELINE_FEATURE_SNAPSHOT_JSONL_PATH, _feature_snapshot)
                        await _pipe_loop.run_in_executor(
                            None,
                            _write_decision_feature_frame_snapshot,
                            DATA_PIPELINE_FEATURE_FRAME_SNAPSHOT_PATH,
                            _snapshot_frame,
                            _pipe_active,
                            _pipe_health,
                            meta_router.position_snapshot(),
                        )
                        if DATA_PIPELINE_FEATURE_FRAME_DUCKDB_ENABLE:
                            await _pipe_loop.run_in_executor(
                                None,
                                _write_decision_feature_frame_duckdb,
                                DATA_PIPELINE_FEATURE_FRAME_DUCKDB_PATH,
                                DATA_PIPELINE_FEATURE_FRAME_DUCKDB_TABLE,
                                _snapshot_frame,
                                _pipe_active,
                                _pipe_health,
                                meta_router.position_snapshot(),
                            )
                    _now_pipe_log = time.time()
                    if (
                        _pipe_health.get("status") != "OK"
                        or (_now_pipe_log - _last_data_pipeline_health_log_ts) >= max(30.0, DATA_PIPELINE_HEALTH_INTERVAL_SEC)
                    ):
                        _last_data_pipeline_health_log_ts = _now_pipe_log
                        _log_data_pipeline_health(_pipe_health)
                except Exception as _pipe_e:
                    logger.warning("PIPE health=BAD reason=%s", _pipe_e)

            _execution_delay_sec = 0.0
            _next_open_delay_late = False
            if _next_open_execution:
                _now_kst = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
                _exec_ts_cmp = pd.Timestamp(execution_time_kst)
                if _exec_ts_cmp.tzinfo is not None:
                    _exec_ts_cmp = _exec_ts_cmp.tz_convert("Asia/Seoul").tz_localize(None)
                _execution_delay_sec = float((_now_kst - _exec_ts_cmp).total_seconds())
                _early_exec_status = dict(_pre_exec_status)
                _real_exchange_execution = bool(
                    _early_exec_status.get("enabled", False)
                    and not bool(_early_exec_status.get("dry_run", True))
                )
                _max_next_open_delay = float(
                    FINAL_GOVERNOR_NEXT_OPEN_MAX_DELAY_SEC
                    if _real_exchange_execution
                    else FINAL_GOVERNOR_NEXT_OPEN_SHADOW_MAX_DELAY_SEC
                )
                _next_open_delay_late = bool(
                    _execution_delay_sec > float(max(0.0, FINAL_GOVERNOR_NEXT_OPEN_WARN_DELAY_SEC))
                )
                _allow_late_next_open_fill = bool(
                    FINAL_GOVERNOR_ALLOW_LATE_NEXT_OPEN_REAL_EXECUTION
                    if _real_exchange_execution
                    else FINAL_GOVERNOR_ALLOW_LATE_NEXT_OPEN_SHADOW_EXECUTION
                )
                if (
                    (not FINAL_GOVERNOR_SCHEDULE_NEXT_BAR_OPEN_ENABLE)
                    and (
                        _execution_delay_sec < -2.0
                        or (
                            (not _allow_late_next_open_fill)
                            and _execution_delay_sec > float(max(0.0, _max_next_open_delay))
                        )
                    )
                ):
                    if not use_local:
                        try:
                            await orderbook_recorder.record_decision_snapshot(
                                fetcher,
                                timestamp_kst=execution_time_kst,
                                context={
                                    "record_reason": "next_open_execution_skipped",
                                    "decision_bar_ts": str(decision_time_kst),
                                    "execution_bar_ts": str(execution_time_kst),
                                    "execution_delay_sec": float(_execution_delay_sec),
                                },
                            )
                        except Exception as _skip_orderbook_error:
                            logger.warning(
                                "SYSTEM skipped_cycle_orderbook=FAILED reason=%s",
                                _skip_orderbook_error,
                            )
                    try:
                        if meta_router.pos in {"LONG", "SHORT"}:
                            final_governor._sync_owner(meta_router, "unknown")
                    except Exception as _sync_e:
                        logger.debug("next-open skip owner sync failed: %s", _sync_e)
                    logger.warning(
                        "SYSTEM next_open_execution skipped: signal=%s execution=%s delay=%.2fs max=%.2fs mode=%s",
                        decision_time_kst,
                        execution_time_kst,
                        _execution_delay_sec,
                        float(_max_next_open_delay),
                        "real_exchange" if _real_exchange_execution else "shadow",
                    )
                    await _persist_data_pipeline_snapshot(
                        {
                            "source": "FINAL_GOVERNOR",
                            "position_signal": "HOLD",
                            "position_reason": "next_open_execution_skipped",
                            "final_action": 0,
                        },
                        stage="next_open_execution_skipped",
                    )
                    return
                if (not FINAL_GOVERNOR_SCHEDULE_NEXT_BAR_OPEN_ENABLE) and _next_open_delay_late:
                    (logger.warning if _real_exchange_execution else logger.info)(
                        "SYSTEM next_open_execution late_fill_allowed: signal=%s execution=%s delay=%.2fs max=%.2fs mode=%s late_allowed=%s",
                        decision_time_kst,
                        execution_time_kst,
                        _execution_delay_sec,
                        float(_max_next_open_delay),
                        "real_exchange" if _real_exchange_execution else "shadow",
                        bool(_allow_late_next_open_fill),
                    )
    
            current_time_kst = execution_time_kst
            current_price = float(execution_price)
            regime_name = 'UNKNOWN'
            _last_idx = processed_df.index[-1]
            _last_row = processed_df.iloc[-1]
            if runtime_predictor is not None:
                try:
                    runtime_predictor.last_errors = []
                except Exception:
                    pass
            
            if runtime_predictor is not None and len(processed_df) > 0:
                try:
                    _ai_features = runtime_predictor.best_ai_features(processed_df)
                    for _ai_col, _ai_val in _ai_features.items():
                        processed_df.at[_last_idx, _ai_col] = float(_ai_val)
                except Exception as _ai_e:
                    logger.error("Final Governor AI feature injection failed: %s", _ai_e)
                    _ai_runtime_errors.append({
                        "model": "AI_FEATURES",
                        "stage": "best_ai_features",
                        "error": str(_ai_e),
                        "fallback": "skip_ai_feature_injection",
                        "ts": time.time(),
                    })
                _ai_runtime_errors.extend(list(getattr(runtime_predictor, "last_errors", []) or []))
                _ai_runtime_errors = _dedupe_ai_errors(_ai_runtime_errors)
                if _ai_runtime_errors:
                    await _notify_patchtst_ai_errors(_ai_runtime_errors, current_time_kst)
    
            # Some feature paths return a feature-only frame. The final governor and
            # regime fallback still need causal OHLCV columns from the aligned bars.
            try:
                for _ohlcv_col in ("open", "high", "low", "close", "volume"):
                    if _ohlcv_col not in processed_df.columns and _ohlcv_col in eth_buffer.columns:
                        _src = eth_buffer[_ohlcv_col].tail(len(processed_df)).reset_index(drop=True)
                        if len(_src) == len(processed_df):
                            processed_df[_ohlcv_col] = pd.to_numeric(_src, errors="coerce").to_numpy()
                        else:
                            processed_df[_ohlcv_col] = float(_bar_float(_signal_bar, _ohlcv_col, decision_price))
                if "close" not in processed_df.columns:
                    processed_df["close"] = float(decision_price)
                if "open" not in processed_df.columns:
                    processed_df["open"] = float(execution_price if _next_open_execution else decision_price)
            except Exception as _ohlcv_e:
                logger.debug("Final Governor OHLCV 보강 실패: %s", _ohlcv_e)
    
            try:
                _pre_last = processed_df.iloc[-1]
                _pre_prev = processed_df.iloc[-2] if len(processed_df) >= 2 else _pre_last
                _pre_smf_std = processed_df["smart_money_flow"].std() if "smart_money_flow" in processed_df.columns else 1.0
                _pre_cur = row_to_market_row(_pre_last)
                _pre_prev_mkt = row_to_market_row(_pre_prev)
                _pre_elite = elite_runtime.compute_all(
                    current=_pre_cur, prev=_pre_prev_mkt, smf_std=_pre_smf_std
                )
                for _sig_col, _sig_val in _pre_elite.items():
                    if isinstance(_sig_col, str) and _sig_col.startswith("sig_"):
                        processed_df.at[_last_idx, _sig_col] = float(_sig_val)
            except Exception as _pre_e:
                logger.debug("elite signals 사전 계산 실패: %s", _pre_e)

            m7_last = None
            trend_signal = None

            try:
                _fa, _kelly, _target_fraction, _target_exec_leverage, _active_info, regime_name = final_governor.decide(
                    processed_df=processed_df,
                    meta_router=meta_router,
                    current_price=decision_price,
                    m7_last=m7_last,
                    trend_signal=trend_signal,
                )
            except Exception as e:
                logger.exception("Final Governor 입력/추론 실패로 사이클 스킵: %s", e)
                await _persist_data_pipeline_snapshot(
                    {
                        "source": "FINAL_GOVERNOR_ERROR",
                        "position_signal": "HOLD",
                        "position_reason": "final_governor_error",
                        "final_action": 0,
                    },
                    stage="final_governor_error",
                    error=e,
                )
                return
    
            _active_info = dict(_active_info or {})
            _sleeve_info = dict(_active_info.get("sleeve_trace", {}) or {})
            _sleeve_info["ai_feature_trace"] = list(getattr(runtime_predictor, "last_trace", []) or [])
            _sleeve_info["ai_timing"] = dict(getattr(runtime_predictor, "last_timing", {}) or {})
            if _ai_runtime_errors:
                _sleeve_info["ai_feature_errors"] = list(_ai_runtime_errors)
                _active_info["ai_feature_error"] = True
                _active_info["ai_feature_error_count"] = int(len(_ai_runtime_errors))
            _active_info["sleeve_trace"] = _sleeve_info
            _active_info.setdefault("agent", "FINAL_GOVERNOR")
            _active_info.setdefault("source", "FINAL_GOVERNOR")
            _active_info["live_bar_contract"] = (
                "signal_close_next_open" if _next_open_execution else "decision_bar"
            )
            _active_info["decision_bar_ts"] = str(decision_time_kst)
            _active_info["decision_price"] = float(decision_price)
            _active_info["execution_bar_ts"] = str(execution_time_kst)
            _active_info["execution_price"] = float(execution_price)
            _active_info["execution_price_source"] = (
                "eth_buffer.open[-1]" if _next_open_execution else "eth_buffer.close[-1]"
            )
            _active_info["execution_delay_sec"] = float(_execution_delay_sec)
            _active_info["execution_delay_late"] = bool(_next_open_delay_late)
            _active_info["execution_delay_mode"] = (
                "real_exchange_late_forced"
                if bool(_next_open_delay_late) and bool(locals().get("_real_exchange_execution", False))
                else "backtest_next_open_late"
                if bool(_next_open_delay_late)
                else "live_completed_bar_next_open_proxy"
                if bool(_live_completed_bar_next_open_proxy)
                else "real_exchange_strict"
                if bool(locals().get("_real_exchange_execution", False))
                else "shadow_extended"
            )
            _active_info.setdefault("score", float(_active_info.get("conviction", 0.0) or 0.0))
            _active_info.setdefault("conviction", float(_active_info.get("score", 0.0) or 0.0))
            _active_info.setdefault("agreement", float(_active_info.get("conviction", 0.0) or 0.0))
            _active_info.setdefault("ambiguity", 0.0)
            _active_info.setdefault("final_action", int(_fa))
            _active_info.setdefault("final_kelly", float(_kelly))
            _active_info.setdefault("kelly", float(_kelly))
            _active_info.setdefault("primary_action", int(_fa))
            _active_info.setdefault("primary_raw", float(_active_info.get("score", 0.0) or 0.0))
            _active_info.setdefault("primary_kelly", float(_kelly))
            _edge_mag = float(max(
                abs(float(_active_info.get("probability_gap", 0.0) or 0.0)),
                abs(float(_active_info.get("score", 0.0) or 0.0)),
                abs(float(_active_info.get("conviction", 0.0) or 0.0)),
            ))
            if int(_fa) == 1:
                _active_info.setdefault("long_edge", _edge_mag)
                _active_info.setdefault("short_edge", 0.0)
                _active_info.setdefault("_long_raw", _edge_mag)
                _active_info.setdefault("_short_raw", 0.0)
                _active_info.setdefault("_long_action", 1)
                _active_info.setdefault("_short_action", 0)
                _active_info.setdefault("_long_kelly", float(_kelly))
                _active_info.setdefault("_short_kelly", 0.0)
                _active_info.setdefault("_selected_side", "LONG")
            elif int(_fa) == 2:
                _active_info.setdefault("long_edge", 0.0)
                _active_info.setdefault("short_edge", _edge_mag)
                _active_info.setdefault("_long_raw", 0.0)
                _active_info.setdefault("_short_raw", _edge_mag)
                _active_info.setdefault("_long_action", 0)
                _active_info.setdefault("_short_action", 2)
                _active_info.setdefault("_long_kelly", 0.0)
                _active_info.setdefault("_short_kelly", float(_kelly))
                _active_info.setdefault("_selected_side", "SHORT")
            else:
                _active_info.setdefault("_selected_side", "HOLD")
            info = dict(_active_info)
    
            await _persist_data_pipeline_snapshot(_active_info, stage="final_governor_success")
            
            _jump_z = float(processed_df.iloc[-1].get("jump_z", 0.0) or 0.0)
            _evt_z = float(processed_df.iloc[-1].get("evt_excess_z", 0.0) or 0.0)
            _hib_score = float(np.clip(max(
                min(abs(_jump_z) / 3.0, 1.5),
                min(abs(_evt_z) / 3.0, 1.5),
            ) / 1.5, 0.0, 1.0))
    
            prev_meta_pos = _prev_meta_pos
            _governor_source = str(_active_info.get("source", "FINAL_GOVERNOR"))
            _hold_reason = str(_active_info.get("position_reason", ""))
            _block_reason = ""
            _trend_exit_score = 0.0
            _exposure_cap = float(getattr(meta_router, "exposure_cap", 5.0))
    
            _fa = int(_fa)
            _kelly = float(np.clip(float(_kelly), 0.0, _exposure_cap))
            if _fa == 0 or _kelly <= 0.0:
                _target_fraction = 0.0
                _target_exec_leverage = 1.0
                _target_exposure = 0.0
            else:
                _target_exposure = float(_kelly)
                try:
                    _raw_target_fraction = float(_target_fraction)
                except (TypeError, ValueError):
                    _raw_target_fraction = 0.0
                try:
                    _raw_target_exec_leverage = float(_target_exec_leverage)
                except (TypeError, ValueError):
                    _raw_target_exec_leverage = 0.0
                _decoded_fraction, _decoded_exec_leverage = _decode_exposure_bucket(_target_exposure, cap=_exposure_cap)
                _target_fraction = float(np.clip(_raw_target_fraction, 0.0, 1.0))
                _target_exec_leverage = float(np.clip(_raw_target_exec_leverage, 1.0, _exposure_cap))
                _target_product = float(_target_fraction * _target_exec_leverage)
                _target_product_tol = max(1e-9, 1e-6 * max(abs(_target_exposure), 1.0))
                if (
                    _target_fraction <= 1e-12
                    or _raw_target_exec_leverage <= 1e-12
                    or abs(_target_product - _target_exposure) > _target_product_tol
                ):
                    _target_fraction = float(_decoded_fraction)
                    _target_exec_leverage = float(_decoded_exec_leverage)
                if _target_exposure <= 1e-12:
                    _target_fraction = 0.0
                    _target_exec_leverage = 1.0
                    _target_exposure = 0.0
                _kelly = _target_exposure
    
            _active_info["base_action"] = int(_fa)
            _active_info["base_kelly"] = float(_kelly)
            _active_info["final_action"] = int(_fa)
            _active_info["final_kelly"] = float(_kelly)
            _active_info["target_action"] = int(_fa)
            if not use_local:
                try:
                    _orderbook_snapshot = await orderbook_recorder.record_decision_snapshot(
                        fetcher,
                        timestamp_kst=current_time_kst,
                        context={
                            "record_reason": "final_governor_decision",
                            "live_bar_contract": str(_active_info.get("live_bar_contract", "")),
                            "decision_bar_ts": str(decision_time_kst),
                            "execution_bar_ts": str(execution_time_kst),
                            "decision_price": float(decision_price),
                            "execution_price": float(execution_price),
                            "final_action": int(_fa),
                            "target_exposure": float(_target_exposure),
                            "target_fraction": float(_target_fraction),
                            "target_exec_leverage": float(_target_exec_leverage),
                            "source": str(_active_info.get("source", "FINAL_GOVERNOR")),
                            "model_version": str(_active_info.get("model_version", "")),
                            "position_reason": str(_active_info.get("position_reason", "")),
                        },
                        force=bool(int(_fa) != 0 or meta_router.pos is not None),
                    )
                    _active_info["orderbook_snapshot"] = dict(_orderbook_snapshot)
                    _sleeve_info = dict(_active_info.get("sleeve_trace", {}) or {})
                    _sleeve_info["orderbook_snapshot"] = dict(_orderbook_snapshot)
                    _active_info["sleeve_trace"] = _sleeve_info
                except Exception as _ob_e:
                    logger.warning("SYSTEM orderbook_snapshot attach=FAILED reason=%s", _ob_e)
    
            _trend_exit = False
            _trend_exit_reason = ""
    
            # ── 📡 선행 레이더 (MicrostructureScanner) 개입 ────────────────
            ms_signal = ms_scanner.get_signal()
    
            # ── 🛡️ 사후 요격기 (TailRiskInterceptor) 개입 ────────────────
            # LAI(청산 흡수 지수) 계산을 위해 이전 1분봉 종가 획득
            prev_price = float(eth_buffer['close'].iloc[-2]) if len(eth_buffer) >= 2 else float(decision_price)
    
            _target_exposure = float(np.clip(_target_exposure, 0.0, _exposure_cap))
            if _fa == 0 or _target_exposure <= 0.0:
                _target_fraction = 0.0
                _target_exec_leverage = 1.0
                _target_exposure = 0.0
    
            # ── Playbook Router: 분석/대시보드 전용 (실제 매매결정에는 미개입) ──
            _ms_exec = dict(ms_signal or {})
            _price_change_pct_exec = (float(decision_price) - prev_price) / max(abs(prev_price), 1e-8) if prev_price > 0 else 0.0
            _tr_pb_exec = dict(
                tr_interceptor.get_playbook_signal(
                    price_change_pct=_price_change_pct_exec,
                    current_price=float(decision_price),
                ) or {}
            )
            _pb_exec_eval = _disabled_playbook_eval(action=int(_fa), kelly=float(_kelly))
            _pb_exec = dict(_pb_exec_eval.get("winner_mft", {}) or {})
            _pb_exec_hft = dict(_pb_exec_eval.get("winner_hft", {}) or {})
            _pb_exec_mft = dict(_pb_exec_eval.get("winner_mft", {}) or {})
            _pb_exec_list = list(_pb_exec_eval.get("evaluations", []) or [])
            _prev_trade_snapshot = meta_router.position_snapshot()
            _decision_made_at_kst = pd.Timestamp.now(tz="Asia/Seoul")
    
            def _target_side_from_action(action: int) -> str | None:
                if int(action) == 1:
                    return "LONG"
                if int(action) == 2:
                    return "SHORT"
                return None
    
            _target_side = _target_side_from_action(int(_fa))
            _current_side = str(meta_router.pos or "")
            _schedule_required = False
            if _target_side is None:
                _schedule_required = bool(meta_router.pos is not None)
            elif float(_target_exposure) > 1e-12:
                _schedule_required = bool(_current_side != _target_side)
                if not _schedule_required and meta_router.pos is not None:
                    _schedule_required = bool(
                        abs(float(_target_fraction) - float(meta_router.position_fraction or 0.0)) > 1e-9
                        or abs(float(_target_exec_leverage) - float(meta_router.execution_leverage or 1.0)) > 1e-9
                        or abs(float(_target_exposure) - float(meta_router.current_leverage or 0.0)) > 1e-9
                    )
    
            if FINAL_GOVERNOR_SCHEDULE_NEXT_BAR_OPEN_ENABLE and _next_open_execution and _schedule_required:
                # Backtest contract: a signal from the completed signal bar fills at
                # the immediately next bar open. Do not add another bar here.
                _pending_execute_at = pd.Timestamp(decision_time_kst)
                if _pending_execute_at.tzinfo is not None:
                    _pending_execute_at = _pending_execute_at.tz_convert("Asia/Seoul").tz_localize(None)
                _pending_execute_at = _pending_execute_at + pd.Timedelta(minutes=5)
                _decision_made_cmp = pd.Timestamp(_decision_made_at_kst)
                if _decision_made_cmp.tzinfo is not None:
                    _decision_made_cmp = _decision_made_cmp.tz_convert("Asia/Seoul").tz_localize(None)
                if _pending_execute_at <= _decision_made_cmp:
                    logger.warning(
                        "SYSTEM pending_next_open not_scheduled: reason=missed_exact_next_open signal=%s execute_at=%s decision_made=%s action=%s exposure=%.4f",
                        str(decision_time_kst),
                        str(_pending_execute_at),
                        str(_decision_made_at_kst),
                        int(_fa),
                        float(_target_exposure),
                    )
                    _pending_next_open_intent = {}
                    try:
                        os.remove(FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH)
                    except OSError:
                        pass
                    return
                _existing_execute_at = None
                if _pending_next_open_intent:
                    try:
                        _existing_execute_at = pd.Timestamp(_pending_next_open_intent.get("execute_at_kst", ""))
                        if _existing_execute_at.tzinfo is not None:
                            _existing_execute_at = _existing_execute_at.tz_convert("Asia/Seoul").tz_localize(None)
                    except Exception:
                        _existing_execute_at = None
                if _existing_execute_at is not None and _existing_execute_at >= _pending_execute_at:
                    logger.info(
                        "SYSTEM pending_next_open preserved: existing_execute_at=%s new_execute_at=%s action=%s exposure=%.4f",
                        str(_existing_execute_at),
                        str(_pending_execute_at),
                        int(_fa),
                        float(_target_exposure),
                    )
                    return
                _pending_trace = dict(_active_info.get("sleeve_trace", {}) or {})
                _pending_audit_context = {
                    "ledger_ts_kind": "scheduled_next_bar_open_pending",
                    "decision_made_at_kst": str(_decision_made_at_kst),
                    "decision_bar_ts": str(decision_time_kst),
                    "decision_bar_utc": str(_bar_value(_signal_bar, "timestamp", "")),
                    "decision_bar_open": _bar_float(_signal_bar, "open", decision_price),
                    "decision_bar_high": _bar_float(_signal_bar, "high", decision_price),
                    "decision_bar_low": _bar_float(_signal_bar, "low", decision_price),
                    "decision_bar_close": _bar_float(_signal_bar, "close", decision_price),
                    "decision_bar_volume": _bar_float(_signal_bar, "volume", 0.0),
                    "decision_bar_is_complete": True,
                    "decision_price": float(decision_price),
                    "decision_price_source": "eth_buffer.close[-2]",
                    "scheduled_execute_at_kst": str(_pending_execute_at),
                    "execution_price_source": "scheduled_next_bar_open",
                    "ai_timing": dict(_sleeve_info.get("ai_timing", {}) or {}),
                    "sleeve_trace": dict(_pending_trace),
                }
                _pending_scout_layer = dict(_pending_trace.get("v21_nearmiss_scout_stop", {}) or {})
                _pending_scout = dict(_pending_scout_layer.get("scout", {}) or {})
                _pending_audit_context.update(
                    {
                        "model_version": str(_active_info.get("model_version", _pending_trace.get("model_version", "")) or ""),
                        "model_id": str(_active_info.get("model_id", _pending_trace.get("decision_logic", "")) or ""),
                        "model_path": str(_active_info.get("model_path", _pending_trace.get("model_path", "")) or ""),
                        "model_sleeve": str(_active_info.get("model_sleeve", _pending_trace.get("v21_sleeve", "")) or ""),
                        "scout_prob": float(
                            _active_info.get(
                                "scout_prob",
                                _pending_scout.get("scout_prob", _pending_scout.get("probability", 0.0)),
                            )
                            or 0.0
                        ),
                        "scout_frac": float(_active_info.get("scout_frac", _pending_scout.get("scout_frac", 0.0)) or 0.0),
                        "scout_probability_threshold": float(
                            _active_info.get(
                                "scout_probability_threshold",
                                _pending_scout.get("probability_threshold", 0.0),
                            )
                            or 0.0
                        ),
                        "scout_cost_pass": bool(_active_info.get("scout_cost_pass", _pending_scout.get("cost_pass", False))),
                        "learned_config": dict(_pending_scout_layer.get("learned_config", _pending_scout.get("learned_config", {})) or {}),
                    }
                )
                _pending_v31_trace = dict(_pending_trace.get("v31", {}) or {})
                _pending_alpha2_trace = dict(_pending_trace.get("alpha2_1", {}) or {})
                _pending_audit_context.update(
                    {
                        "v31_q_long": float(_pending_v31_trace.get("q_long", 0.0) or 0.0),
                        "v31_q_short": float(_pending_v31_trace.get("q_short", 0.0) or 0.0),
                        "v31_edge": float(_pending_v31_trace.get("edge", 0.0) or 0.0),
                        "v31_margin": float(_pending_v31_trace.get("margin", 0.0) or 0.0),
                        "v31_selected_side": str(_pending_v31_trace.get("selected_side", "") or ""),
                        "v31_pass_gate": bool(_pending_v31_trace.get("pass_gate", False)),
                        "parent_action": int(
                            _pending_alpha2_trace.get("parent_action_before", _pending_trace.get("base_action", _pending_v31_trace.get("parent_action", 0))) or 0
                        ),
                        "parent_side": int(
                            _pending_alpha2_trace.get("parent_side_before", _pending_trace.get("base_side", _pending_v31_trace.get("parent_side", 0))) or 0
                        ),
                        "teacher_gate_result": str(
                            _pending_alpha2_trace.get("reason", _pending_v31_trace.get("teacher_gate_result", "")) or ""
                        ),
                        "teacher_pred_action": int(_pending_alpha2_trace.get("teacher_pred_action", 0) or 0),
                        "teacher_confidence": float(_pending_alpha2_trace.get("teacher_confidence", 0.0) or 0.0),
                        "teacher_quality": float(_pending_alpha2_trace.get("teacher_quality", 0.0) or 0.0),
                        "teacher_keep_parent": bool(_pending_alpha2_trace.get("keep_parent", False)),
                    }
                )
                _active_info["scheduled_execution"] = True
                _active_info["decision_made_at_kst"] = str(_decision_made_at_kst)
                _active_info["scheduled_execute_at_kst"] = str(_pending_execute_at)
                _active_info["execution_bar_ts"] = str(_pending_execute_at)
                _active_info["execution_price_source"] = "scheduled_next_bar_open"
                _pending_next_open_intent = _jsonable({
                    "schema_version": "pending_next_open_intent.v1",
                    "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                    "decision_made_at_kst": str(_decision_made_at_kst),
                    "signal_bar_ts": str(decision_time_kst),
                    "execute_at_kst": str(_pending_execute_at),
                    "final_action": int(_fa),
                    "target_exposure": float(_target_exposure),
                    "target_fraction": float(_target_fraction),
                    "target_exec_leverage": float(_target_exec_leverage),
                    "decision_price": float(decision_price),
                    "source": str(_governor_source),
                    "regime_name": str(regime_name),
                    "hold_reason": str(_hold_reason),
                    "active_info": dict(_active_info),
                    "audit_context": _pending_audit_context,
                })
                _loop = asyncio.get_running_loop()
                await _loop.run_in_executor(None, _atomic_write_json, FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH, _pending_next_open_intent)
                logger.info(
                    "SYSTEM pending_next_open scheduled: decision_made=%s signal=%s execute_at=%s action=%s exposure=%.4f fraction=%.4f lev=%.2f source=%s",
                    str(_decision_made_at_kst),
                    str(decision_time_kst),
                    str(_pending_execute_at),
                    int(_fa),
                    float(_target_exposure),
                    float(_target_fraction),
                    float(_target_exec_leverage),
                    str(_governor_source),
                )
                return
            _active_info.setdefault("decision_made_at_kst", str(_decision_made_at_kst))
            _lifecycle_v1_decision = (
                str(_governor_source).startswith("lifecycle_v1|")
                or str(_active_info.get("decision_logic", "")) == "clean_base_lifecycle_v1"
            )
            _entry_reco_signal = None if (_lifecycle_v1_decision or _next_open_execution) else trend_signal
            if _lifecycle_v1_decision or _next_open_execution:
                _active_info["entry_reco_disabled"] = "next_open_execution" if _next_open_execution else True
            _live_execution_result = dict(live_executor.status())
            _live_execution_result.update({"ok": True, "blocking": False, "status": "disabled" if not live_executor.enabled else "pending"})
            if not use_local:
                _live_execution_result = await live_executor.execute_to_target(
                    final_action=int(_fa),
                    target_exposure=float(_target_exposure),
                    target_exec_leverage=float(_target_exec_leverage),
                    current_price=float(current_price),
                    timestamp_kst=current_time_kst,
                    decision_info=_active_info,
                    existing_tp_sl_order_ids={
                        "tp_order_id": str(getattr(final_governor, "active_omega4_6_1_tp_order_id", "") or ""),
                        "sl_order_id": str(getattr(final_governor, "active_omega4_6_1_sl_order_id", "") or ""),
                    },
                )
                _active_info["live_execution"] = dict(_live_execution_result)
                _tp_sl_result = dict(_live_execution_result.get("tp_sl") or {})
                if _tp_sl_result.get("tp_order_id") or _tp_sl_result.get("sl_order_id"):
                    final_governor.active_omega4_6_1_tp_order_id = str(_tp_sl_result.get("tp_order_id", ""))
                    final_governor.active_omega4_6_1_sl_order_id = str(_tp_sl_result.get("sl_order_id", ""))
                    final_governor._save_runtime_state()
                _tp_sl_cancel_result = dict(_live_execution_result.get("tp_sl_cancel") or {})
                if _tp_sl_cancel_result.get("tp_cancelled") or _tp_sl_cancel_result.get("sl_cancelled"):
                    final_governor.active_omega4_6_1_tp_order_id = ""
                    final_governor.active_omega4_6_1_sl_order_id = ""
                    final_governor._save_runtime_state()
            _skip_local_position_update = bool(_live_execution_result.get("blocking", False))
            if _skip_local_position_update:
                _block_reason = f"binance_execution_{_live_execution_result.get('status', 'blocked')}"
                _active_info["position_reason"] = str(_block_reason)
                logger.warning("SYSTEM binance_execution blocked local state update: %s", _live_execution_result)
            _execution_alert = build_execution_alert(
                _live_execution_result,
                decision_reason=str(_active_info.get("position_reason", "") or _block_reason),
                observed_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            )
            _notify_execution_alert(_execution_alert)
            # ── 최종 포지션 업데이트 및 대시보드 저장 ──
            if not _skip_local_position_update:
                _entry_route_info = GovernorPositionRouter._execution_route_summary(
                    list(_live_execution_result.get("orders", []) or []),
                    reduce_only=False,
                )
                meta_router._update_pos(
                    _fa,
                    current_price,
                    decision_time_kst,
                    _target_exposure,
                    fraction=_target_fraction,
                    leverage_mult=_target_exec_leverage,
                    trend_signal=_entry_reco_signal,
                    entry_price_source_override=(
                        "next_bar_open" if _next_open_execution else None
                    ),
                    entry_decision_price_override=float(decision_price) if _next_open_execution else None,
                    entry_execution_liquidity_override=str(_entry_route_info.get("liquidity", "")),
                    entry_execution_route_override=str(_entry_route_info.get("route", "")),
                    entry_execution_order_type_override=str(_entry_route_info.get("order_type", "")),
                )
            meta_router.update_adaptive_gate(final_action=int(_fa), in_position=(meta_router.pos is not None))
    
            _sleeve_trace = dict(_active_info.get("sleeve_trace", {}) or {})
            _regime_predictor_trace = dict(_sleeve_trace.get("regime_predictor", {}) or {})
            _scout_layer_trace = dict(_sleeve_trace.get("v21_nearmiss_scout_stop", {}) or {})
            _scout_decision_trace = dict(_scout_layer_trace.get("scout", {}) or {})
            meta_result = {
                "final_action": _fa,
                "unified_kelly": _target_exposure,
                "position_fraction": _target_fraction,
                "execution_leverage": _target_exec_leverage,
                "live_bar_contract": (
                    "signal_close_next_open" if _next_open_execution else "decision_bar"
                ),
                "alpha3_live_contract": FINAL_GOVERNOR_ALPHA3_LIVE_CONTRACT_ID,
                "alpha3_mark_parity": bool(FINAL_GOVERNOR_ALPHA3_CSV_MARK_PARITY_ENABLE),
                "alpha3_cooldown_parity": bool(FINAL_GOVERNOR_ALPHA3_CSV_COOLDOWN_PARITY_ENABLE),
                "decision_time_kst": str(decision_time_kst),
                "decision_price": float(decision_price),
                "execution_time_kst": str(execution_time_kst),
                "execution_price": float(execution_price),
                "execution_delay_sec": float(_execution_delay_sec),
                "execution_delay_late": bool(_next_open_delay_late),
                "execution_delay_mode": str(_active_info.get("execution_delay_mode", "")),
                "source": _governor_source,
                "enhanced_source": _governor_source,
                "rl_score": float(_active_info.get("score", 0.0)),
                "rl_action": _fa,
                "trend_signal": trend_signal,
                "trend_exit_score": float(_trend_exit_score),
                "trend_mismatch_streak": int(meta_router.trend_mismatch_streak),
                "hibernation_score": float(_hib_score),
                "hibernation_score_th": float(meta_router.hibernation_score_th),
                "illiq_amihud": float(processed_df.iloc[-1].get("amihud_illiquidity_z", 0.0) or 0.0),
                "cb_active": 0,
                "position_signal": str(_active_info.get("position_signal", "")),
                "position_reason": str(_active_info.get("position_reason", "")),
                "position_own_support": float(_active_info.get("own_support", 0.0)),
                "position_opp_pressure": float(_active_info.get("opp_pressure", 0.0)),
                "position_net_edge": float(_active_info.get("net_edge", 0.0)),
                "hold_reason": str(_hold_reason),
                "block_reason": str(_block_reason),
                "live_execution": dict(_live_execution_result),
                "sleeve_trace": _sleeve_trace,
                "regime_predictor": _regime_predictor_trace,
                "model_version": str(_active_info.get("model_version", _sleeve_trace.get("model_version", "")) or ""),
                "model_id": str(_active_info.get("model_id", _sleeve_trace.get("decision_logic", "")) or ""),
                "model_path": str(_active_info.get("model_path", _sleeve_trace.get("model_path", "")) or ""),
                "model_sleeve": str(_active_info.get("model_sleeve", _sleeve_trace.get("v21_sleeve", "")) or ""),
                "scout_prob": float(
                    _active_info.get(
                        "scout_prob",
                        _scout_decision_trace.get("scout_prob", _scout_decision_trace.get("probability", 0.0)),
                    )
                    or 0.0
                ),
                "scout_frac": float(_active_info.get("scout_frac", _scout_decision_trace.get("scout_frac", 0.0)) or 0.0),
                "scout_probability_threshold": float(
                    _active_info.get(
                        "scout_probability_threshold",
                        _scout_decision_trace.get("probability_threshold", 0.0),
                    )
                    or 0.0
                ),
                "scout_cost_pass": bool(_active_info.get("scout_cost_pass", _scout_decision_trace.get("cost_pass", False))),
            }
            rl_action = int(_fa)
            trade_pnl_pct: float | None = None
    
            _new_pos = meta_router.pos
            _new_trade_snapshot = meta_router.position_snapshot()
            _last_bar = _signal_bar
            _exec_bar = _execution_bar
            _bar_ts = pd.Timestamp(decision_time_kst)
            _bar_ts_cmp = _bar_ts.tz_convert("Asia/Seoul").tz_localize(None) if _bar_ts.tzinfo is not None else _bar_ts
            _exec_ts = pd.Timestamp(execution_time_kst)
            _exec_ts_cmp = _exec_ts.tz_convert("Asia/Seoul").tz_localize(None) if _exec_ts.tzinfo is not None else _exec_ts
            _decision_bar_audit_context = {
                "ledger_ts_kind": "next_bar_open_execution" if _next_open_execution else "decision_bar",
                "decision_made_at_kst": str(_active_info.get("decision_made_at_kst", "")),
                "decision_bar_ts": str(decision_time_kst),
                "decision_bar_utc": str(_bar_value(_last_bar, "timestamp", "")),
                "decision_bar_open": _bar_float(_last_bar, "open", decision_price),
                "decision_bar_high": _bar_float(_last_bar, "high", decision_price),
                "decision_bar_low": _bar_float(_last_bar, "low", decision_price),
                "decision_bar_close": _bar_float(_last_bar, "close", decision_price),
                "decision_bar_volume": _bar_float(_last_bar, "volume", 0.0),
                "decision_bar_is_complete": bool(
                    _next_open_execution
                    or pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None) >= (_bar_ts_cmp + pd.Timedelta(minutes=5))
                ),
                "decision_price": float(decision_price),
                "decision_price_source": (
                    "eth_buffer.close[-1]"
                    if _next_open_execution and _live_completed_bar_next_open_proxy
                    else "eth_buffer.close[-2]" if _next_open_execution else "eth_buffer.close[-1]"
                ),
                "execution_bar_ts": str(execution_time_kst),
                "execution_bar_utc": str(_bar_value(_exec_bar, "timestamp", "")),
                "execution_bar_open": _bar_float(_exec_bar, "open", execution_price),
                "execution_bar_high": _bar_float(_exec_bar, "high", execution_price),
                "execution_bar_low": _bar_float(_exec_bar, "low", execution_price),
                "execution_bar_close": _bar_float(_exec_bar, "close", execution_price),
                "execution_bar_volume": _bar_float(_exec_bar, "volume", 0.0),
                "execution_bar_is_current": bool(
                    _next_open_execution
                    and pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None) < (_exec_ts_cmp + pd.Timedelta(minutes=5))
                ),
                "execution_price": float(execution_price),
                "execution_price_source": (
                    "eth_buffer.open[-1]" if _next_open_execution else "eth_buffer.close[-1]"
                ),
                "execution_delay_sec": float(_execution_delay_sec),
                "execution_delay_late": bool(_next_open_delay_late),
                "execution_delay_mode": str(_active_info.get("execution_delay_mode", "")),
                "live_execution": dict(_live_execution_result),
                "orderbook_snapshot": dict(_active_info.get("orderbook_snapshot", {}) or {}),
                "ai_timing": dict(_sleeve_trace.get("ai_timing", {}) or {}),
                "sleeve_trace": dict(_sleeve_trace),
                "model_version": str(_active_info.get("model_version", _sleeve_trace.get("model_version", "")) or ""),
                "model_id": str(_active_info.get("model_id", _sleeve_trace.get("decision_logic", "")) or ""),
                "model_path": str(_active_info.get("model_path", _sleeve_trace.get("model_path", "")) or ""),
                "model_sleeve": str(_active_info.get("model_sleeve", _sleeve_trace.get("v21_sleeve", "")) or ""),
                "scout_prob": float(
                    _active_info.get(
                        "scout_prob",
                        _scout_decision_trace.get("scout_prob", _scout_decision_trace.get("probability", 0.0)),
                    )
                    or 0.0
                ),
                "scout_frac": float(_active_info.get("scout_frac", _scout_decision_trace.get("scout_frac", 0.0)) or 0.0),
                "scout_probability_threshold": float(
                    _active_info.get(
                        "scout_probability_threshold",
                        _scout_decision_trace.get("probability_threshold", 0.0),
                    )
                    or 0.0
                ),
                "scout_cost_pass": bool(_active_info.get("scout_cost_pass", _scout_decision_trace.get("cost_pass", False))),
                "learned_config": dict(
                    dict(_sleeve_trace.get("v21_nearmiss_scout_stop", {}) or {}).get(
                        "learned_config",
                        _scout_decision_trace.get("learned_config", {}),
                    )
                    or {}
                ),
            }
            _v31_audit_trace = dict(_sleeve_trace.get("v31", {}) or {})
            _alpha2_audit_trace = dict(_sleeve_trace.get("alpha2_1", {}) or {})
            _decision_bar_audit_context.update(
                {
                    "v31_q_long": float(_v31_audit_trace.get("q_long", 0.0) or 0.0),
                    "v31_q_short": float(_v31_audit_trace.get("q_short", 0.0) or 0.0),
                    "v31_q_long_raw": float(_v31_audit_trace.get("q_long_raw", 0.0) or 0.0),
                    "v31_q_short_raw": float(_v31_audit_trace.get("q_short_raw", 0.0) or 0.0),
                    "v31_edge": float(_v31_audit_trace.get("edge", _v31_audit_trace.get("entry_edge", 0.0)) or 0.0),
                    "v31_margin": float(_v31_audit_trace.get("margin", _v31_audit_trace.get("entry_margin", 0.0)) or 0.0),
                    "v31_raw_margin": float(_v31_audit_trace.get("raw_margin", 0.0) or 0.0),
                    "v31_selected_side": str(_v31_audit_trace.get("selected_side", "") or ""),
                    "v31_pass_gate": bool(_v31_audit_trace.get("pass_gate", False)),
                    "v31_guard_reason": str(
                        _v31_audit_trace.get(
                            "guard_reason",
                            "|".join(list(_v31_audit_trace.get("regime_long_guard_reasons", []) or [])),
                        )
                        or ""
                    ),
                    "v31_transition_risk": float(_v31_audit_trace.get("transition_risk", 0.0) or 0.0),
                    "parent_action": int(
                        _alpha2_audit_trace.get("parent_action_before", _sleeve_trace.get("base_action", _v31_audit_trace.get("parent_action", 0))) or 0
                    ),
                    "parent_side": int(
                        _alpha2_audit_trace.get("parent_side_before", _sleeve_trace.get("base_side", _v31_audit_trace.get("parent_side", 0))) or 0
                    ),
                    "teacher_gate_result": str(_alpha2_audit_trace.get("reason", _v31_audit_trace.get("teacher_gate_result", "")) or ""),
                    "teacher_pred_action": int(_alpha2_audit_trace.get("teacher_pred_action", 0) or 0),
                    "teacher_confidence": float(_alpha2_audit_trace.get("teacher_confidence", 0.0) or 0.0),
                    "teacher_quality": float(_alpha2_audit_trace.get("teacher_quality", 0.0) or 0.0),
                    "teacher_keep_parent": bool(_alpha2_audit_trace.get("keep_parent", False)),
                }
            )
    
            def _risk_price_levels(snapshot: dict, info: dict) -> dict:
                snap = dict(snapshot or {})
                side = str(snap.get("pos") or "").upper()
                entry = float(snap.get("entry_price", 0.0) or 0.0)
                exposure = float(snap.get("notional_exposure", snap.get("total_exposure", 0.0)) or 0.0)
                hold_bars = int(snap.get("hold_bars", 0) or 0)
                if side not in {"LONG", "SHORT"}:
                    return {
                        "take_profit": 0.0,
                        "stop_loss": 0.0,
                        "max_hold_bars": 0,
                        "max_hold_remaining_bars": 0,
                        "take_profit_price": 0.0,
                        "tp_price": 0.0,
                        "stop_price": 0.0,
                        "sl_price": 0.0,
                        "effective_take_profit": 0.0,
                        "effective_stop_loss": 0.0,
                        "risk_source": "flat",
                    }
                trace = dict((info or {}).get("sleeve_trace", {}) or {})
                v31_trace = dict(trace.get("v31", {}) or {})
                is_v31 = str(final_governor.active_lifecycle_v1_scout_model_version or "") == "V31" or bool(v31_trace)
                owner = str((info or {}).get("owner", final_governor.owner or "") or "").lower()
                source = str((info or {}).get("source", "") or "").lower()
                is_omega5 = owner == OMEGA5_OWNER or source.startswith(f"{OMEGA5_OWNER}|")
                if is_omega5:
                    tp = float(final_governor.active_omega5_take_profit or 0.0)
                    sl = float(final_governor.active_omega5_stop_loss or 0.0)
                    max_hold = int(final_governor.active_omega5_max_hold_bars or 0)
                    risk_source = "omega5_active_risk"
                else:
                    tp = float(final_governor.active_lifecycle_v1_take_profit or 0.0)
                    sl = float(final_governor.active_lifecycle_v1_stop_loss or 0.0)
                    max_hold = int(final_governor.active_lifecycle_v1_max_hold_bars or 0)
                    risk_source = "parent_policy"
    
                if is_v31 and not is_omega5:
                    tp = float(v31_trace.get("effective_tp", tp) or tp)
                    sl = float(v31_trace.get("effective_sl", sl) or sl)
                    cfg = dict(v31_trace.get("selected_config", final_governor.v31_cfg or {}) or {})
                    max_hold = int(cfg.get("base_hold", max_hold or final_governor._v31_cfg_int("base_hold", 48)) or 0)
                    risk_source = "v31_effective_overlay"
    
                def _price_from_threshold(threshold: float, take_profit: bool) -> float:
                    threshold = float(threshold or 0.0)
                    if side not in {"LONG", "SHORT"} or entry <= 0.0 or exposure <= 0.0 or threshold <= 0.0:
                        return 0.0
                    raw_move = threshold / max(exposure, 1e-8)
                    if side == "LONG":
                        price = entry * (1.0 + raw_move) if take_profit else entry * max(0.0, 1.0 - raw_move)
                    else:
                        price = entry * max(0.0, 1.0 - raw_move) if take_profit else entry * (1.0 + raw_move)
                    return float(price if price > 0.0 else 0.0)
    
                out = {
                    "take_profit": float(tp),
                    "stop_loss": float(sl),
                    "max_hold_bars": int(max_hold),
                    "max_hold_remaining_bars": int(max(0, max_hold - hold_bars)) if max_hold > 0 else 0,
                    "take_profit_price": _price_from_threshold(tp, True),
                    "tp_price": _price_from_threshold(tp, True),
                    "stop_price": _price_from_threshold(sl, False),
                    "sl_price": _price_from_threshold(sl, False),
                    "effective_take_profit": float(tp),
                    "effective_stop_loss": float(sl),
                    "risk_source": risk_source,
                }
                if is_omega5:
                    out["omega5_source_roundtrip_cost"] = float(
                        (info or {}).get("omega5_source_roundtrip_cost", final_governor.active_omega5_roundtrip_cost) or 0.0
                    )
                    out["omega5_source_exit_reason"] = str(
                        (info or {}).get("omega5_source_exit_reason", final_governor.active_omega5_source_exit_reason) or ""
                    )
                    out["omega5_source_exit_price_move"] = float(
                        (info or {}).get("omega5_source_exit_price_move", final_governor.active_omega5_source_exit_price_move)
                        or 0.0
                    )
                if is_v31 and not is_omega5:
                    out["entry_vol_anchor"] = float(v31_trace.get("entry_vol_anchor", final_governor.active_v31_entry_vol_anchor or 0.0) or 0.0)
                return out
    
            _prev_risk_fields = _risk_price_levels(_prev_trade_snapshot, _active_info)
            _new_risk_fields = _risk_price_levels(_new_trade_snapshot, _active_info)
            if _prev_meta_pos is not None:
                _prev_trade_snapshot.update(_prev_risk_fields)
            if _new_pos is not None:
                _new_trade_snapshot.update(_new_risk_fields)
            _decision_bar_audit_context.update(_new_risk_fields if _new_pos is not None else _prev_risk_fields)
            _transition_label = _pos_transition_label(prev_meta_pos, _new_pos)
            _resized = (
                prev_meta_pos is not None
                and _new_pos == prev_meta_pos
                and (
                    abs(float(_new_trade_snapshot.get("position_fraction", 0.0) or 0.0) - float(_prev_trade_snapshot.get("position_fraction", 0.0) or 0.0)) > 1e-9
                    or abs(float(_new_trade_snapshot.get("execution_leverage", 1.0) or 1.0) - float(_prev_trade_snapshot.get("execution_leverage", 1.0) or 1.0)) > 1e-9
                    or abs(float(_new_trade_snapshot.get("total_exposure", 0.0) or 0.0) - float(_prev_trade_snapshot.get("total_exposure", 0.0) or 0.0)) > 1e-9
                )
            )
            _trade_journal_rows: list[dict] = []
            _position_accounting_audit_rows: list[dict] = []
            _close_trade_payload: dict | None = None
            _open_trade_payload: dict | None = None
            _audit_equity_cursor = _accounting_equity_from_history(getattr(meta_router, "trade_history", []))
            _position_closed = (_prev_meta_pos is not None and _new_pos != _prev_meta_pos)
            if _position_closed:
                _close_reason = str(_block_reason or _hold_reason or _active_info.get("position_reason", "") or _governor_source)
                _close_trade_payload = meta_router.build_close_trade_payload(
                    snapshot=_prev_trade_snapshot,
                    current_price=current_price,
                    timestamp_kst=current_time_kst,
                    event=_transition_label,
                    regime_name=regime_name,
                    source=_governor_source,
                    reason=_close_reason,
                    next_side=_new_pos,
                    audit_context=_decision_bar_audit_context,
                )
                realized = float(_close_trade_payload.get("pnl_frac", 0.0))
                trade_pnl_pct = float(_close_trade_payload.get("pnl_pct", 0.0))
                _close_equity_before = float(_audit_equity_cursor)
                _close_equity_after = float(_close_equity_before * max(0.0, 1.0 + realized))
                _position_accounting_audit_rows.append(_build_position_accounting_audit_row(
                    trade_row=_close_trade_payload,
                    equity_before=_close_equity_before,
                    equity_after=_close_equity_after,
                    prev_snapshot=_prev_trade_snapshot,
                    new_snapshot=_new_trade_snapshot,
                    current_price=current_price,
                    from_pos=prev_meta_pos,
                    to_pos=_new_pos,
                    decision_info=_active_info,
                    fee_rate=float(meta_router.trade_fee),
                    slippage_rate=float(meta_router.trade_slip),
                ))
                _audit_equity_cursor = _close_equity_after
                enhanced_engine.on_trade_close(realized)
                meta_router.record_outcome(realized)
                meta_router.append_trade_history(current_time_kst, realized, payload=_close_trade_payload)
                _trade_journal_rows.append(dict(_close_trade_payload))
    
            if _new_pos is not None and _new_pos != _prev_meta_pos:
                enhanced_engine.on_position_open()
                _open_reason = str(_active_info.get("position_reason", "") or _hold_reason or _governor_source)
                _open_trade_payload = meta_router.build_open_trade_payload(
                    snapshot=_new_trade_snapshot,
                    timestamp_kst=current_time_kst,
                    event=_transition_label,
                    regime_name=regime_name,
                    source=_governor_source,
                    reason=_open_reason,
                    audit_context=_decision_bar_audit_context,
                )
                _position_accounting_audit_rows.append(_build_position_accounting_audit_row(
                    trade_row=_open_trade_payload,
                    equity_before=float(_audit_equity_cursor),
                    equity_after=float(_audit_equity_cursor),
                    prev_snapshot=_prev_trade_snapshot,
                    new_snapshot=_new_trade_snapshot,
                    current_price=current_price,
                    from_pos=prev_meta_pos,
                    to_pos=_new_pos,
                    decision_info=_active_info,
                    fee_rate=float(meta_router.trade_fee),
                    slippage_rate=float(meta_router.trade_slip),
                ))
                _trade_journal_rows.append(dict(_open_trade_payload))
            elif _resized:
                _resize_reason = str(_active_info.get("position_reason", "") or _hold_reason or _governor_source)
                _resize_trade_payload = meta_router.build_resize_trade_payload(
                    prev_snapshot=_prev_trade_snapshot,
                    new_snapshot=_new_trade_snapshot,
                    current_price=current_price,
                    timestamp_kst=current_time_kst,
                    regime_name=regime_name,
                    source=_governor_source,
                    reason=_resize_reason,
                    audit_context=_decision_bar_audit_context,
                )
                _resize_realized = float(_resize_trade_payload.get("pnl_frac", 0.0) or 0.0)
                _resize_equity_before = float(_audit_equity_cursor)
                _resize_equity_after = float(_resize_equity_before * max(0.0, 1.0 + _resize_realized))
                _position_accounting_audit_rows.append(_build_position_accounting_audit_row(
                    trade_row=_resize_trade_payload,
                    equity_before=float(_resize_equity_before),
                    equity_after=float(_resize_equity_after),
                    prev_snapshot=_prev_trade_snapshot,
                    new_snapshot=_new_trade_snapshot,
                    current_price=current_price,
                    from_pos=prev_meta_pos,
                    to_pos=_new_pos,
                    decision_info=_active_info,
                    fee_rate=float(meta_router.trade_fee),
                    slippage_rate=float(meta_router.trade_slip),
                ))
                if abs(_resize_realized) > 1e-12:
                    meta_router.append_trade_history(current_time_kst, _resize_realized, payload=_resize_trade_payload)
                    _audit_equity_cursor = _resize_equity_after
                _trade_journal_rows.append(dict(_resize_trade_payload))
            if _prev_meta_pos is None and _new_pos is not None and trade_pnl_pct is None:
                trade_pnl_pct = 0.0
            if trade_pnl_pct is not None: meta_result["trade_pnl_pct"] = float(trade_pnl_pct)
            meta_result["position_entry_price"] = float(_new_trade_snapshot.get("entry_price", 0.0) or 0.0)
            meta_result["position_unrealized_pnl_pct"] = float(meta_router.unrealized_pnl(current_price) if meta_router.pos else 0.0)
    
            if prev_meta_pos != _new_pos:
                if prev_meta_pos is None and _new_pos: _tg_code = f"ENTER_{_new_pos}"
                elif prev_meta_pos and _new_pos is None: _tg_code = f"EXIT_{prev_meta_pos}"
                elif prev_meta_pos and _new_pos: _tg_code = f"FLIP_{prev_meta_pos}_TO_{_new_pos}"
                else: _tg_code = None
                if _tg_code:
                    task_supervisor.create(
                        tg_notifier.notify(
                            _tg_trade_msg(_tg_code, current_price, current_time_kst, regime_name, meta_result)
                        ),
                        name="telegram-trade",
                    )
    
            _prev_meta_pos = _new_pos
    
            _print_final_trade_summary(
                timestamp_kst=current_time_kst, current_price=current_price,
                regime_name=regime_name, rl_action=rl_action, rl_info=_active_info,
                meta_result=meta_result, prev_pos=prev_meta_pos, cur_pos=meta_router.pos,
            )
    
            if not CONSOLE_LOG_COMPACT:
                meta_router.print_meta_dashboard(meta_result, current_price)
                if "enhanced_diag" in info:
                    enhanced_engine.print_enhanced_dashboard({
                        "action": _fa, "kelly": _kelly, "source": _governor_source,
                        "diagnostics": info.get("enhanced_diag", {}),
                    })
            _perf_metrics = meta_router.performance_metrics(current_time_kst)
            try:
                _ms = _ms_exec
                _tr_shadow = dict(getattr(tr_interceptor, "_shadow_state", {}) or {})
                _tr_bucket = str(_tr_shadow.get("shadow_risk_bucket", "normal"))
                if _tr_bucket == "high":
                    _tr_reco = "HOLD"
                elif _tr_bucket == "watch":
                    _tr_reco = "REDUCE"
                else:
                    _tr_reco = "FOLLOW"
                _tr_pb = _tr_pb_exec
                _pb = _pb_exec
                _pb_hft = _pb_exec_hft
                _pb_mft = _pb_exec_mft
                _pb_overall = dict(_pb_exec_eval.get("winner", {}) or {})
                _pb_list = _pb_exec_list
                _ms_for_llm = dict(ms_signal or {})
                _matched = [x for x in _pb_list if bool(x.get("matched", False))]
                if _matched:
                    _top_consensus = _matched
                else:
                    _top_consensus = [{
                        "name": "NO_ACTIVE_PLAYBOOK",
                        "matched": False,
                        "reason": "현재 시장은 횡보 중이거나 HFT/MFT 특이 조건이 발동되지 않은 평온한 상태입니다.",
                    }]
                _llm_payload = {
                    "portfolio_state": {
                        "current_position": str(meta_router.pos or "NONE"),
                        "unrealized_pnl_pct": float(meta_router.unrealized_pnl(current_price) if meta_router.pos else 0.0),
                    },
                    "market_environment": {
                        "funding_rate": float(_ms_for_llm.get("funding_rate", 0.0)),
                        "eai_energy": float(_ms_for_llm.get("eai", 0.0)),
                        "whale_cvd_30m": float(_ms_for_llm.get("nif_whale_sum_30m", 0.0)),
                        "toxicity": float(_ms_for_llm.get("shadow_toxicity_score", 0.0)),
                        "price_volatility_30m": float(_ms_for_llm.get("price_volatility_30m", 0.0)),
                    },
                    "playbook_consensus": {
                        "winner": {
                            "name": str(_pb_overall.get("name", "NONE")),
                            "matched": bool(_pb_overall.get("matched", False)),
                            "action": int(_pb_overall.get("action", 0)),
                            "kelly": float(_pb_overall.get("kelly", 0.0)),
                            "reason": str(_pb_overall.get("reason", "")),
                        },
                        "top_playbooks": _top_consensus,
                    },
                }
                _llm_advice = {
                    "enabled": False,
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                    "decision": "HOLD",
                    "confidence_score": 0,
                    "kelly_weight": 0.0,
                    "reasoning": "LLM 미실행",
                }
                _trades_tail = _build_trades_tail_from_router(meta_router)
                _closed_trade_equity = _router_closed_trade_equity(meta_router)
                _open_mark_pnl_frac = _router_open_mark_pnl_frac(meta_router, current_price)
                _strategy_equity = _router_strategy_equity(meta_router, current_price)
    
                _sess_flags_live = _session_flags_from_timestamp(current_time_kst)
                _quant_formula = dict(_QUANT_CARD_CACHE.get("payload", {}) or {})
                if not _quant_formula:
                    _quant_formula = _build_quant_formula_card(
                        eth_df=eth_buffer,
                        current_price=float(current_price),
                        current_time_kst=current_time_kst,
                    )
                _dashboard_state = {
                    "schema_version": "live.dashboard.v2",
                    "governor_mode": True,
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                    "cycle_timestamp_kst": str(current_time_kst),
                    "account": {**dict(fetcher.account_status()), "execution": dict(_live_execution_result)},
                    "execution_alert": dict(_execution_alert),
                    "session": {
                        "session_asia": float(_sess_flags_live.get("session_asia", 0.0)),
                        "session_europe": float(_sess_flags_live.get("session_europe", 0.0)),
                        "session_us": float(_sess_flags_live.get("session_us", 0.0)),
                    },
                    "price": float(current_price),
                    "asset_decisions": dict(latest_asset_decisions),
                    "asset_states": dict(latest_asset_decisions),
                    "regime": str(regime_name),
                    "position": {
                        "current": meta_router.pos or "NONE",
                        "entry_price": float(meta_router.entry_price or 0.0),
                        "decision_at": str(meta_router.decision_at or ""),
                        "opened_at": str(meta_router.opened_at or ""),
                        "hold_bars": int(meta_router.hold_count or 0),
                        "position_fraction": float(meta_router.position_fraction or 0.0),
                        "margin_fraction": float(meta_router.position_fraction or 0.0),
                        "execution_leverage": float(meta_router.execution_leverage or 1.0),
                        "notional_exposure": float(meta_router.current_leverage or 0.0),
                        "total_exposure": float(meta_router.current_leverage or 0.0),
                        "unrealized_pnl_pct": float(meta_router.unrealized_pnl(current_price) if meta_router.pos else 0.0),
                        "position_realized_pnl_frac": float(meta_router.position_realized_pnl_frac or 0.0),
                        "position_realized_pnl_pct": float((meta_router.position_realized_pnl_frac or 0.0) * 100.0),
                        "last_resize_realized_pnl_frac": float(meta_router.last_resize_realized_pnl_frac or 0.0),
                        "strategy_equity": float(_strategy_equity),
                        "closed_trade_equity": float(_closed_trade_equity),
                        "deployed_equity": float(_strategy_equity * float(meta_router.position_fraction or 0.0)),
                        "gross_exposure_equity": float(_strategy_equity * float(meta_router.current_leverage or 0.0)),
                        "unrealized_pnl_amount": float(_strategy_equity * _open_mark_pnl_frac),
                        "trade_pnl_pct": float(trade_pnl_pct) if trade_pnl_pct is not None else None,
                        "take_profit": float(_new_risk_fields.get("take_profit", 0.0) or 0.0),
                        "stop_loss": float(_new_risk_fields.get("stop_loss", 0.0) or 0.0),
                        "max_hold_bars": int(_new_risk_fields.get("max_hold_bars", 0) or 0),
                        "max_hold_remaining_bars": int(_new_risk_fields.get("max_hold_remaining_bars", 0) or 0),
                        "take_profit_price": float(_new_risk_fields.get("take_profit_price", 0.0) or 0.0),
                        "tp_price": float(_new_risk_fields.get("tp_price", 0.0) or 0.0),
                        "stop_price": float(_new_risk_fields.get("stop_price", 0.0) or 0.0),
                        "sl_price": float(_new_risk_fields.get("sl_price", 0.0) or 0.0),
                        "effective_take_profit": float(_new_risk_fields.get("effective_take_profit", 0.0) or 0.0),
                        "effective_stop_loss": float(_new_risk_fields.get("effective_stop_loss", 0.0) or 0.0),
                        "risk_source": str(_new_risk_fields.get("risk_source", "") or ""),
                    },
                    "signal": {
                        "rl_action": int(rl_action),
                        "final_action": int(_fa),
                        "source": str(_governor_source),
                        "unified_kelly": float(_target_exposure),
                        "position_fraction": float(_target_fraction),
                        "margin_fraction": float(_target_fraction),
                        "execution_leverage": float(_target_exec_leverage),
                        "notional_exposure": float(_target_fraction * _target_exec_leverage),
                        "governor_owner": str(_active_info.get("owner", final_governor.owner or "")),
                        "governor_regime": str(_active_info.get("regime", regime_name)),
                        "governor_reason": str(_active_info.get("position_reason", _active_info.get("reason", "")) or ""),
                        "governor_score": float(_active_info.get("score", 0.0) or 0.0),
                        "probability_gap": float(_active_info.get("probability_gap", 0.0) or 0.0),
                        "hold_reason": str(_hold_reason),
                        "block_reason": str(_block_reason),
                        "decision_logic": str(_active_info.get("decision_logic", "")),
                        "live_execution": dict(_live_execution_result),
                        "sleeve_trace": dict(_active_info.get("sleeve_trace", {}) or {}),
                        "regime_predictor": dict(_regime_predictor_trace),
                        "take_profit": float(_new_risk_fields.get("take_profit", 0.0) or 0.0),
                        "stop_loss": float(_new_risk_fields.get("stop_loss", 0.0) or 0.0),
                        "max_hold_bars": int(_new_risk_fields.get("max_hold_bars", 0) or 0),
                        "max_hold_remaining_bars": int(_new_risk_fields.get("max_hold_remaining_bars", 0) or 0),
                        "take_profit_price": float(_new_risk_fields.get("take_profit_price", 0.0) or 0.0),
                        "tp_price": float(_new_risk_fields.get("tp_price", 0.0) or 0.0),
                        "stop_price": float(_new_risk_fields.get("stop_price", 0.0) or 0.0),
                        "sl_price": float(_new_risk_fields.get("sl_price", 0.0) or 0.0),
                    },
                    "agents": {
                        "governor": {
                            "action": int(_fa),
                            "source": str(_governor_source),
                            "owner": str(_active_info.get("owner", final_governor.owner or "")),
                            "regime": str(_active_info.get("regime", regime_name)),
                            "score": float(_active_info.get("score", 0.0) or 0.0),
                            "conviction": float(_active_info.get("conviction", 0.0) or 0.0),
                            "probability_gap": float(_active_info.get("probability_gap", 0.0) or 0.0),
                            "notional_exposure": float(_target_exposure),
                            "position_fraction": float(_target_fraction),
                            "execution_leverage": float(_target_exec_leverage),
                        },
                        "regime_predictor": dict(_regime_predictor_trace),
                        "disabled_v13_1": {
                            "enabled": bool(final_governor._v13_1_available()),
                            "model_version": "Disabled V13.1",
                            "model_id": "disabled_v13_1_model",
                            "model": str(final_governor.disabled_v13_1_model_path),
                            "report": str(final_governor.disabled_v13_1_report_path),
                            "cooldown_left": int(final_governor.v13_1_cooldown_left),
                            "active_notional": float(final_governor.active_v13_1_notional),
                            "active_leverage": float(final_governor.active_v13_1_leverage),
                            "active_lane": str(final_governor.active_v13_1_lane),
                            "active_probability": float(final_governor.active_v13_1_probability),
                            "active_threshold": float(final_governor.active_v13_1_threshold),
                            "active_regime": str(final_governor.active_v13_1_regime),
                            "active_regime_multiplier": float(final_governor.active_v13_1_regime_multiplier),
                        },
                        "lifecycle_v1": {
                            "enabled": bool(final_governor._lifecycle_v1_available()),
                            "model": os.path.basename(str(final_governor.lifecycle_v1_model_path)),
                            "base_policy": os.path.basename(str(final_governor.lifecycle_v1_policy_path)),
                            "exit_model": os.path.basename(str(final_governor.lifecycle_v1_exit_model_path)),
                            "cooldown_left": int(final_governor.lifecycle_v1_cooldown_left),
                            "active_base_notional": float(final_governor.active_lifecycle_v1_base_notional),
                            "active_effective_notional": float(final_governor.active_lifecycle_v1_effective_notional),
                            "active_leverage": float(final_governor.active_lifecycle_v1_leverage),
                            "active_edit": str(final_governor.active_lifecycle_v1_edit),
                            "active_entry_bucket": str(final_governor.active_lifecycle_v1_entry_bucket),
                            "active_entry_hazard": float(final_governor.active_lifecycle_v1_entry_hazard),
                            "active_entry_support": int(final_governor.active_lifecycle_v1_entry_support),
                            "deep_gated_gross": {
                                "enabled": bool(final_governor.deep_gated_gross_enabled),
                                "model": str(final_governor.deep_gated_gross_model_path),
                                "report": str(final_governor.deep_gated_gross_report_path),
                                "selected_config": str(
                                    dict(final_governor.deep_gated_gross_cfg or {}).get("name", "")
                                ),
                                "high_notional": float(dict(final_governor.deep_gated_gross_cfg or {}).get("high_notional", 0.0) or 0.0),
                                "mid_notional": float(dict(final_governor.deep_gated_gross_cfg or {}).get("mid_notional", 0.0) or 0.0),
                                "defensive_notional": float(dict(final_governor.deep_gated_gross_cfg or {}).get("defensive_notional", 0.0) or 0.0),
                                "cost3_notional": float(dict(final_governor.deep_gated_gross_cfg or {}).get("cost3_notional", 0.0) or 0.0),
                            },
                            "deep_state_adaptive_calibrator": {
                                "enabled": bool(final_governor.deep_state_adaptive_calibrator_enabled),
                                "model": str(final_governor.deep_state_adaptive_calibrator_model_path),
                                "report": str(final_governor.deep_state_adaptive_calibrator_report_path),
                                "audit": str(final_governor.deep_state_adaptive_calibrator_audit_path),
                                "selected_config": str(
                                    final_governor._adaptive_calibrator_cfg_get(
                                        final_governor.deep_state_adaptive_config,
                                        "name",
                                        "",
                                    )
                                ),
                                "future_rolling_q": (
                                    None
                                    if final_governor.deep_state_adaptive_future_rolling_q is None
                                    else float(final_governor.deep_state_adaptive_future_rolling_q)
                                ),
                            },
                            "scout_layer": {
                                "enabled": bool(final_governor._lifecycle_v22_1_available() or final_governor._lifecycle_v21_available()),
                                "model_version": "V22.1" if final_governor._lifecycle_v22_1_available() else ("V22.1" if final_governor.v21_adapter_version == "v22_1_learned_scout" else "V21"),
                                "model_id": str(final_governor.v22_1_adapter.model_id if final_governor._lifecycle_v22_1_available() and final_governor.v22_1_adapter is not None else final_governor.v21_model_id),
                                "adapter_version": "v22_1_learned_scout" if final_governor._lifecycle_v22_1_available() else str(final_governor.v21_adapter_version),
                                "model": str(final_governor.v22_1_model_path if final_governor._lifecycle_v22_1_available() else final_governor.v21_model_path),
                                "report": str(final_governor.v22_1_report_path if final_governor._lifecycle_v22_1_available() else final_governor.v21_report_path),
                                "audit": str(final_governor.v22_1_audit_path if final_governor._lifecycle_v22_1_available() else final_governor.v21_audit_path),
                                "selected_config": str(
                                    (
                                        dict(final_governor.v22_1_adapter.learned_config or {}).get("name", "")
                                        if final_governor._lifecycle_v22_1_available() and final_governor.v22_1_adapter is not None
                                        else dict(final_governor.v21_scout_config or {}).get("name", "")
                                    )
                                ),
                                "active_sleeve": str(final_governor.active_lifecycle_v1_v21_sleeve),
                                "active_stop_raw": float(final_governor.active_lifecycle_v1_v21_stop_raw),
                            },
                            "deep_constant_gross": {
                                "enabled": bool(final_governor.deep_constant_gross_enabled),
                                "report": str(final_governor.deep_constant_gross_report_path),
                                "selected_config": str(
                                    dict(final_governor.deep_constant_gross_report.get("selected_config", {}) or {}).get("name", "")
                                ),
                                "target_notional": float(final_governor.deep_constant_gross_target_notional),
                                "cost3_notional": float(final_governor.deep_constant_gross_cost3_notional),
                            },
                            "dsac_overlay": {
                                "enabled": bool(final_governor.dsac_overlay_enabled),
                                "checkpoint": str(final_governor.dsac_overlay_ckpt_path),
                                "checkpoint_meta": dict(final_governor.dsac_overlay_ckpt_meta),
                                "mode": str(final_governor.dsac_overlay_mode),
                                "threshold": float(final_governor.dsac_overlay_threshold),
                                "scale": float(final_governor.dsac_overlay_scale),
                                "cost_gate_enabled": bool(final_governor.dsac_overlay_cost_gate_enabled),
                                "cost_buffer": float(final_governor.dsac_overlay_cost_buffer),
                            },
                        },
                        "fully_learned": {
                            "enabled": bool(final_governor.fully_learned_policy_bundle is not None),
                            "model": os.path.basename(str(final_governor.fully_learned_policy_path)),
                            "active_take_profit": float(final_governor.active_fully_learned_take_profit),
                            "active_stop_loss": float(final_governor.active_fully_learned_stop_loss),
                            "active_max_hold_bars": int(final_governor.active_fully_learned_max_hold_bars),
                            "active_cooldown_bars": int(final_governor.active_fully_learned_cooldown_bars),
                            "cooldown_left": int(final_governor.fully_learned_cooldown_left),
                            "active_quality_score": float(final_governor.active_fully_learned_quality_score),
                            "active_confidence": float(final_governor.active_fully_learned_confidence),
                        },
                        "omega4_6_1": {
                            "enabled": bool(final_governor.omega4_6_1_adapter is not None),
                            "model_id": OMEGA4_6_1_MODEL_ID,
                            "model_version": OMEGA4_6_1_MODEL_VERSION,
                            "source_component": str(final_governor.active_omega4_6_1_source_component),
                            "active_take_profit": float(final_governor.active_omega4_6_1_take_profit),
                            "active_stop_loss": float(final_governor.active_omega4_6_1_stop_loss),
                            "active_notional": float(final_governor.active_omega4_6_1_notional),
                            "active_leverage": float(final_governor.active_omega4_6_1_leverage),
                            "active_quality_score": float(final_governor.active_omega4_6_1_quality_score),
                            "active_confidence": float(final_governor.active_omega4_6_1_confidence),
                        },
                        "macro": {
                            "enabled": bool(FINAL_GOVERNOR_MACRO_ENABLE and final_governor.fully_learned_policy_bundle is None),
                            "lookback_bars": int(FINAL_GOVERNOR_MACRO_LOOKBACK_BARS),
                            "threshold": float(FINAL_GOVERNOR_MACRO_THRESHOLD),
                            "persist_updates": int(FINAL_GOVERNOR_MACRO_PERSIST_UPDATES),
                            "update_bars": int(FINAL_GOVERNOR_MACRO_UPDATE_BARS),
                            "notional_cap": float(FINAL_GOVERNOR_MACRO_NOTIONAL),
                            "leverage": float(FINAL_GOVERNOR_MACRO_LEVERAGE),
                            "take_profit": float(FINAL_GOVERNOR_MACRO_TAKE_PROFIT),
                            "stop_loss": float(FINAL_GOVERNOR_MACRO_STOP_LOSS),
                            "trailing_arm": float(FINAL_GOVERNOR_MACRO_TRAILING_ARM),
                            "trailing_gap": float(FINAL_GOVERNOR_MACRO_TRAILING_GAP),
                            "lockout_bars": int(FINAL_GOVERNOR_MACRO_LOCKOUT_BARS),
                            "lockout_on_any_close": bool(FINAL_GOVERNOR_MACRO_LOCKOUT_ON_ANY_CLOSE),
                            "lockout_signal": int(final_governor.macro_lockout_signal),
                            "lockout_bars_left": int(final_governor.macro_lockout_bars_left),
                            "execution_policy": {
                                "enabled": bool(final_governor.execution_policy_bundle is not None),
                                "model": os.path.basename(str(final_governor.execution_policy_path)),
                                "ignore_max_hold": bool(FINAL_GOVERNOR_EXECUTION_POLICY_IGNORE_MAX_HOLD),
                                "quality_overlay": bool(FINAL_GOVERNOR_EXECUTION_POLICY_QUALITY_OVERLAY),
                                "low_quality": float(FINAL_GOVERNOR_EXECUTION_POLICY_LOW_QUALITY),
                                "tail_quality": float(FINAL_GOVERNOR_EXECUTION_POLICY_TAIL_QUALITY),
                                "active_take_profit": float(final_governor.active_macro_take_profit),
                                "active_stop_loss": float(final_governor.active_macro_stop_loss),
                                "active_max_hold_bars": int(final_governor.active_macro_max_hold_bars),
                                "active_quality_score": float(final_governor.active_macro_quality_score),
                            },
                        },
                        "ddh2_ensemble": {
                            "enabled": bool(final_governor.ddh2_ensemble_enabled),
                            "report": str(final_governor.ddh2_report_path),
                            "audit": str(final_governor.ddh2_audit_path),
                            "audit_status": str(final_governor.ddh2_audit.get("status", "")),
                            "fallback_dd_block_active": bool(final_governor.ddh2_fallback_dd_block_active),
                            "trend_cost_gap_buffer": float(final_governor._ddh2_cfg_float("trend_cost_gap_buffer", 0.0)),
                            "micro_cost_gap_buffer": float(final_governor._ddh2_cfg_float("micro_cost_gap_buffer", 0.0)),
                            "fallback_account_dd_block": float(final_governor._ddh2_cfg_float("fallback_account_dd_block", 0.0)),
                            "fallback_account_dd_release": float(final_governor._ddh2_cfg_float("fallback_account_dd_release", 0.0)),
                        },
                        "sniper": {
                            "enabled": bool(FINAL_GOVERNOR_SNIPER_ENABLE and final_governor.fully_learned_policy_bundle is None),
                            "checkpoint": os.path.basename(str(FINAL_GOVERNOR_SNIPER_MODEL_PATH)),
                            "notional_cap": float(FINAL_GOVERNOR_NOTIONAL),
                            "leverage": float(FINAL_GOVERNOR_LEVERAGE),
                        },
                        "trend": {
                            "enabled": bool(FINAL_GOVERNOR_TREND_ENABLE and final_governor.fully_learned_policy_bundle is None),
                            "model": os.path.basename(str(FINAL_GOVERNOR_TREND_MODEL_PATH)),
                            "notional_cap": float(FINAL_GOVERNOR_NOTIONAL),
                            "leverage": float(FINAL_GOVERNOR_LEVERAGE),
                        },
                        "micro": {
                            "enabled": bool(FINAL_GOVERNOR_MICRO_ENABLE and final_governor.fully_learned_policy_bundle is None),
                            "model": os.path.basename(str(FINAL_GOVERNOR_MICRO_MODEL_PATH)),
                            "notional_cap": float(FINAL_GOVERNOR_NOTIONAL),
                            "leverage": float(FINAL_GOVERNOR_LEVERAGE),
                        },
                        "agreement_count": int(_active_info.get("agreement_count", 0)),
                        "net_score": float(_active_info.get("net_score", _active_info.get("score", 0.0))),
                        "conviction": float(_active_info.get("conviction", 0.0)),
                    },
                    "trend": {
                        "prob_up": float((trend_signal or {}).get("prob_up", 0.0) or 0.0),
                        "prob_dn": float((trend_signal or {}).get("prob_dn", 0.0) or 0.0),
                        "strength": float((trend_signal or {}).get("strength", 0.0) or 0.0),
                        "reversal_risk": float((trend_signal or {}).get("rev_prob", 0.0) or 0.0),
                    },
                    "risk": {
                        "hibernation_score": float(_hib_score),
                        "hibernation_th": float(meta_router.hibernation_score_th),
                        "amihud": float(processed_df.iloc[-1].get("amihud_illiquidity_z", 0.0) or 0.0),
                        "cooldown_bars_left": int(meta_router.cooldown_bars_left),
                    },
                    "microstructure": {
                        "updated_at": pd.Timestamp.utcnow().isoformat(),
                        "obi": float(_ms.get("obi", 0.0)),
                        "taker_buy_ratio": float(_ms.get("taker_buy_ratio", 0.5)),
                        "spoofing_score": float(_ms.get("spoofing_score", 0.0)),
                        "spoofing_bias": int(_ms.get("spoofing_bias", 0)),
                        "nif_whale": float(_ms.get("nif_whale", 0.0)),
                        "nif_retail": float(_ms.get("nif_retail", 0.0)),
                        "nif_bias": int(_ms.get("nif_bias", 0)),
                        "eai": float(_ms.get("eai", 0.0)),
                        "eai_bias": int(_ms.get("eai_bias", 0)),
                        "oi_delta_pct": float(_ms.get("oi_delta_pct", 0.0)),
                        "oi_delta_cum_5m": float(_ms.get("oi_delta_cum_5m", 0.0)),
                        "oi_delta_cum_5m_bucket_start_ts": int(_ms.get("oi_delta_cum_5m_bucket_start_ts", 0)),
                        "whale_flow_10s_ratio": float(_ms.get("whale_flow_10s_ratio", 0.0)),
                        "whale_buy_10s_usd": float(_ms.get("whale_buy_10s_usd", 0.0)),
                        "whale_sell_10s_usd": float(_ms.get("whale_sell_10s_usd", 0.0)),
                        "whale_flow_cum_5m_ratio": float(_ms.get("whale_flow_cum_5m_ratio", 0.0)),
                        "whale_buy_cum_5m_usd": float(_ms.get("whale_buy_cum_5m_usd", 0.0)),
                        "whale_sell_cum_5m_usd": float(_ms.get("whale_sell_cum_5m_usd", 0.0)),
                        "whale_flow_cum_5m_bucket_start_ts": int(_ms.get("whale_flow_cum_5m_bucket_start_ts", 0)),
                        "funding_rate": float(_ms.get("funding_rate", 0.0)),
                        "signal_bias": int(_ms.get("signal_bias", 0)),
                        "kelly_mult": float(_ms.get("kelly_mult", 1.0)),
                        "toxicity_score": float(_ms.get("shadow_toxicity_score", 0.0)),
                        "toxicity_regime": str(_ms.get("shadow_toxicity_regime", "normal")),
                        "queue_collapse": float(_ms.get("shadow_queue_collapse", 0.0)),
                        "absorption_score": float(_ms.get("shadow_absorption_score", 0.0)),
                        "queue_bias": int(_ms.get("shadow_queue_bias", 0)),
                        "regime_tag": str(_ms.get("shadow_regime_tag", "normal")),
                        "regime_conf": float(_ms.get("shadow_regime_conf", 0.0)),
                        "price_change_30m": float(_ms.get("price_change_30m", 0.0)),
                        "price_volatility_30m": float(_ms.get("price_volatility_30m", 0.0)),
                        "vwap_gap_15m": float(_ms.get("vwap_gap_15m", 0.0)),
                        "price_breakout_60m": bool(_ms.get("price_breakout_60m", False)),
                        "price_breakdown_60m": bool(_ms.get("price_breakdown_60m", False)),
                        "nif_whale_sum_30m": float(_ms.get("nif_whale_sum_30m", 0.0)),
                        "nif_whale_avg_30m": float(_ms.get("nif_whale_avg_30m", 0.0)),
                        "nif_whale_std_30m": float(_ms.get("nif_whale_std_30m", 0.0)),
                        "whale_short_build_ratio_30m": float(_ms.get("whale_short_build_ratio_30m", 0.0)),
                        "whale_long_close_ratio_30m": float(_ms.get("whale_long_close_ratio_30m", 0.0)),
                        "whale_sell_presence_ratio_30m": float(_ms.get("whale_sell_presence_ratio_30m", 0.0)),
                        "whale_sell_effective_ratio_30m": float(_ms.get("whale_sell_effective_ratio_30m", 0.0)),
                        "whale_long_build_ratio_30m": float(_ms.get("whale_long_build_ratio_30m", 0.0)),
                        "whale_short_cover_ratio_30m": float(_ms.get("whale_short_cover_ratio_30m", 0.0)),
                        "whale_buy_presence_ratio_30m": float(_ms.get("whale_buy_presence_ratio_30m", 0.0)),
                        "whale_buy_effective_ratio_30m": float(_ms.get("whale_buy_effective_ratio_30m", 0.0)),
                        "whale_position_bias_30m": str(_ms.get("whale_position_bias_30m", "중립")),
                        "whale_position_window_min": int(_ms.get("whale_position_window_min", 5)),
                        "whale_position_estimate": str(_ms.get("whale_position_estimate", "NEUTRAL")),
                        "whale_position_confidence": int(_ms.get("whale_position_confidence", 0)),
                        "whale_position_score": float(_ms.get("whale_position_score", 0.0)),
                        "absorption_avg_30m": float(_ms.get("absorption_avg_30m", 0.0)),
                        "bias_avg_30m": float(_ms.get("bias_avg_30m", 0.0)),
                        "toxicity_avg_30m": float(_ms.get("toxicity_avg_30m", 0.0)),
                        "eai_delta_15m": float(_ms.get("eai_delta_15m", 0.0)),
                        "data_stale": bool(_ms.get("data_stale", False)),
                        "depth_connected": bool(_ms.get("depth_connected", False)),
                        "trade_connected": bool(_ms.get("trade_connected", False)),
                        "poll_connected": bool(_ms.get("poll_connected", False)),
                        "depth_age_sec": (float(_ms.get("depth_age_sec")) if _ms.get("depth_age_sec") is not None else None),
                        "trade_age_sec": (float(_ms.get("trade_age_sec")) if _ms.get("trade_age_sec") is not None else None),
                        "poll_age_sec": (float(_ms.get("poll_age_sec")) if _ms.get("poll_age_sec") is not None else None),
                        "recent_trade_count_5m": int(_ms.get("recent_trade_count_5m", 0)),
                        "recent_trade_notional_5m": float(_ms.get("recent_trade_notional_5m", 0.0)),
                        "recent_whale_count_5m": int(_ms.get("recent_whale_count_5m", 0)),
                        "valid_taker_flow": bool(_ms.get("valid_taker_flow", False)),
                        "valid_nif": bool(_ms.get("valid_nif", False)),
                        "status_line": str(ms_scanner.status_line()),
                    },
                    "tail_risk": {
                        "updated_at": pd.Timestamp.utcnow().isoformat(),
                        "aftershock_prob": float(_tr_shadow.get("shadow_aftershock_prob", 0.0)),
                        "half_life_min": float(_tr_shadow.get("shadow_decay_half_life", 0.0)),
                        "risk_bucket": str(_tr_bucket),
                        "z_long": float(_tr_pb.get("z_long", 0.0)),
                        "z_short": float(_tr_pb.get("z_short", 0.0)),
                        "lai": float(_tr_pb.get("lai", 0.0)),
                        "long_usd_1m": float(_tr_pb.get("long_usd_1m", 0.0)),
                        "short_usd_1m": float(_tr_pb.get("short_usd_1m", 0.0)),
                        "liq_event_count_1m": int(_tr_pb.get("liq_event_count_1m", 0)),
                        "ws_connected": bool(_tr_pb.get("ws_connected", False)),
                        "ws_age_sec": (float(_tr_pb.get("ws_age_sec")) if _tr_pb.get("ws_age_sec") is not None else None),
                        "valid_liq_stream": bool(_tr_pb.get("valid_liq_stream", False)),
                        "hawkes_active": bool(_tr_pb.get("hawkes_active", False)),
                        "hawkes_decay_level": float(_tr_pb.get("hawkes_decay_level", 0.0)),
                        "crisis_type": str(_tr_pb.get("crisis_type", "")),
                        "liq_cluster_direction": int(_tr_pb.get("liq_cluster_direction", 0)),
                        "liq_cluster_strength": float(_tr_pb.get("liq_cluster_strength", 0.0)),
                        "distance_to_cluster_pct": float(_tr_pb.get("distance_to_cluster_pct", 1.0)),
                        "liq_cluster_price": float(_tr_pb.get("liq_cluster_price", 0.0)),
                        "z_bias": int(-1 if float(_tr_pb.get("z_long", 0.0)) > float(_tr_pb.get("z_short", 0.0)) else (1 if float(_tr_pb.get("z_short", 0.0)) > float(_tr_pb.get("z_long", 0.0)) else 0)),
                        "recommendation": _tr_reco,
                        "status_line": str(tr_interceptor.status_line()),
                    },
                    "playbook": {
                        "updated_at": pd.Timestamp.utcnow().isoformat(),
                        "matched": bool(_pb.get("matched", False)),
                        "name": str(_pb.get("name", "NONE")),
                        "priority": int(_pb.get("priority", 0)),
                        "action": int(_pb.get("action", _fa)),
                        "kelly": float(_pb.get("kelly", _kelly)),
                        "reason": str(_pb.get("reason", "")),
                        "emergency_exit": bool(_pb.get("emergency_exit", False)),
                        "widen_trailing_stop": bool(_pb.get("widen_trailing_stop", False)),
                        "meta": dict(_pb.get("meta", {}) or {}),
                        "hft": _pb_hft,
                        "mft": _pb_mft,
                        "evaluations": _pb_list,
                    },
                    "performance": {
                        "pnl_24h": float(_perf_metrics.get("pnl_24h", 0.0)),
                        "wr_24h": float(_perf_metrics.get("wr_24h", 0.0)),
                        "pnl_7d": float(_perf_metrics.get("pnl_7d", 0.0)),
                        "wr_7d": float(_perf_metrics.get("wr_7d", 0.0)),
                        "pnl_all": float(_perf_metrics.get("pnl_all", 0.0)),
                        "pnl_24h_sum": float(_perf_metrics.get("pnl_24h_sum", _perf_metrics.get("pnl_24h", 0.0))),
                        "pnl_7d_sum": float(_perf_metrics.get("pnl_7d_sum", _perf_metrics.get("pnl_7d", 0.0))),
                        "pnl_all_sum": float(_perf_metrics.get("pnl_all_sum", _perf_metrics.get("pnl_all", 0.0))),
                        "wr_all": float(_perf_metrics.get("wr_all", 0.0)),
                    },
                    "llm": dict(_llm_advice or {}),
                    "trades_tail": _trades_tail,
                    "quant_formula": dict(_quant_formula or {}),
                    "ensembles": (lambda _e: {**_e, "tracker": _ensemble_tracker_summary(_load_ensemble_tracker_state())})(
                        _build_ensemble_runtime(
                            pb_list=_pb_list,
                            base_action=_fa,
                            base_kelly=_kelly,
                            ms=_ms,
                            tr=_tr_pb,
                        )
                    ),
                }
                _loop = asyncio.get_running_loop()
                await _loop.run_in_executor(None, _atomic_write_json, DASHBOARD_STATE_PATH, _dashboard_state)
                if _trade_journal_rows:
                    await journal_writer.append_many(TRADE_JOURNAL_PATH, _trade_journal_rows)
                if _position_accounting_audit_rows:
                    await journal_writer.append_many(POSITION_ACCOUNTING_AUDIT_PATH, _position_accounting_audit_rows)
                if _position_closed or (_prev_meta_pos is None and _new_pos is not None):
                    await journal_writer.append(DASHBOARD_EVENTS_PATH, {
                        "ts": str(current_time_kst),
                        "event": _transition_label,
                        "from": prev_meta_pos,
                        "to": _new_pos,
                        "price": float(current_price),
                        "kelly": float(_target_exposure),
                        "pnl_pct": (float(trade_pnl_pct) if trade_pnl_pct is not None else 0.0),
                        "regime": str(regime_name),
                        "close_trade": _close_trade_payload,
                        "open_trade": _open_trade_payload,
                    })
                elif _resized:
                    await journal_writer.append(DASHBOARD_EVENTS_PATH, {
                        "ts": str(current_time_kst),
                        "event": "RESIZE",
                        "from": _prev_meta_pos,
                        "to": _new_pos,
                        "price": float(current_price),
                        "kelly": float(_target_exposure),
                        "pnl_pct": 0.0,
                        "regime": str(regime_name),
                        "resize_trade": _trade_journal_rows[-1] if _trade_journal_rows else None,
                    })
            except Exception as _dash_e:
                if CONSOLE_LOG_COMPACT:
                    logger.warning("DATA store=BAD main_dashboard=%s", _dash_e)
                else:
                    logger.debug("dashboard state write skip: %s", _dash_e)
            if not CONSOLE_LOG_COMPACT:
                logger.info("📊 %s", meta_router.performance_summary(current_time_kst))
    
        if use_local:
            eth_buffer, btc_buffer = fetcher.load_local_data()
        else:
            logger.info("초기 캔들 데이터 수집 중...")
            try:
                eth_buffer, btc_buffer = await fetcher.fetch_initial_data()
            except Exception as e:
                logger.error("❌ 초기 캔들 수집 실패: %s", e)
                _notify_execution_alert(
                    build_execution_alert(
                        {"requested_enabled": True, "enabled": True, "blocking": True, "status": "initial_data_error", "error": str(e)},
                        observed_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                    )
                )
                return

        if eth_buffer is None: return
        try:
            _boot_process_t0 = time.perf_counter()
            _boot_eth_buffer = eth_buffer
            _boot_btc_buffer = btc_buffer
            if FINAL_GOVERNOR_LIVE_PROCESS_BARS > 0:
                _tail_n = int(max(600, FINAL_GOVERNOR_LIVE_PROCESS_BARS))
                _boot_eth_buffer = eth_buffer.tail(_tail_n)
                _boot_btc_buffer = btc_buffer.tail(_tail_n)
            processed_boot_full = fe_engine.process(_boot_eth_buffer, _boot_btc_buffer)
            processed_boot = processed_boot_full
            if FINAL_GOVERNOR_LIVE_MODEL_BARS > 0:
                processed_boot = processed_boot_full.tail(int(max(600, FINAL_GOVERNOR_LIVE_MODEL_BARS))).copy()
            if FINAL_GOVERNOR_TIMING_LOG_ENABLE:
                logger.info(
                    "TIMING boot_process rows_eth=%d rows_processed=%d rows_feature_source=%d process_tail=%d model_tail=%d process_sec=%.2f",
                    len(eth_buffer),
                    len(processed_boot),
                    len(processed_boot_full),
                    int(FINAL_GOVERNOR_LIVE_PROCESS_BARS),
                    int(FINAL_GOVERNOR_LIVE_MODEL_BARS),
                    time.perf_counter() - _boot_process_t0,
                )
        except Exception as e:
            logger.error("❌ 초기 피처 처리 실패: %s", e)
            _notify_execution_alert(
                build_execution_alert(
                    {"requested_enabled": True, "enabled": True, "blocking": True, "status": "initial_feature_error", "error": str(e)},
                    observed_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                )
            )
            return
            
        logger.info(
            "SYSTEM governor=READY stack=%s lifecycle=%s adaptive=%s fully_learned=%s legacy_macro_sniper=%s ai=%s",
            FINAL_GOVERNOR_ALPHA43_STICKY_MODEL_ID
            if final_governor.fully_learned_policy_bundle is not None
            else (
                "disabled_v13_1"
                if final_governor._v13_1_available()
                else ("ddh2_v22_1_sniper_trend_micro_full_1x" if final_governor.ddh2_ensemble_enabled else "lifecycle_v1_v17_adaptive_calibrator")
            ),
            os.path.basename(str(FINAL_GOVERNOR_LIFECYCLE_V1_MODEL_PATH)),
            os.path.basename(str(FINAL_GOVERNOR_DEEP_STATE_ADAPTIVE_CALIBRATOR_MODEL_PATH)),
            os.path.basename(str(FINAL_GOVERNOR_FULLY_LEARNED_POLICY_PATH)),
            bool(FINAL_GOVERNOR_SNIPER_ENABLE or FINAL_GOVERNOR_MACRO_ENABLE),
            ",".join(FINAL_GOVERNOR_AI_FEATURE_GROUPS),
        )
        logger.info(
            "SYSTEM governor_legacy trend=%s micro=%s",
            bool(FINAL_GOVERNOR_TREND_ENABLE),
            bool(FINAL_GOVERNOR_MICRO_ENABLE),
        )

        if not use_local and fetcher.account_status().get("ready"):
            account_snapshot = await fetcher.fetch_account_snapshot()
            account_pos = dict(account_snapshot.get("position") or {})
            logger.info(
                "SYSTEM binance_account balance=%s position=%s position_sync=%s",
                "OK" if account_snapshot.get("balance_ok") else "BAD",
                str(account_pos.get("type", "NONE") or "NONE"),
                bool(fetcher.account_status().get("position_sync_enabled")),
            )
	            
        if not use_local:
            exchange_position_state, restored = await _fetch_exchange_position()
            _last_exchange_reconcile_ts = time.time()
            # Covers a resting TP/SL fill that happened while the process was down: the reloaded
            # runtime state may still show an open omega4_6_1 position/order ids even though the
            # exchange is now flat. Same logic as the 15s loop's reconcile check.
            _boot_went_flat = exchange_position_went_flat(exchange_position_state, meta_router.pos)
            _boot_tp_sl_fill_info = None
            _boot_tp_order_id = ""
            _boot_sl_order_id = ""
            if _boot_went_flat and final_governor.owner == OMEGA4_6_1_OWNER:
                _boot_tp_order_id = str(getattr(final_governor, "active_omega4_6_1_tp_order_id", "") or "")
                _boot_sl_order_id = str(getattr(final_governor, "active_omega4_6_1_sl_order_id", "") or "")
                if _boot_tp_order_id or _boot_sl_order_id:
                    _boot_tp_sl_fill_info = await live_executor.poll_tp_sl_orders(
                        tp_order_id=_boot_tp_order_id, sl_order_id=_boot_sl_order_id
                    )
            if restored or _boot_went_flat:
                meta_router.reconcile_external_position(
                    restored.get("type") if restored else None,
                    float(restored.get("entry_price", 0.0)) if restored else 0.0,
                    float(restored.get("leverage", 0.0)) if restored else 0.0,
                    notional=float(restored.get("notional", 0.0)) if restored else 0.0,
                    account_equity=float(restored.get("account_equity_usdt", 0.0)) if restored else 0.0,
                    notional_exposure=float(restored.get("notional_exposure", 0.0)) if restored else 0.0,
                    tp_sl_fill_info=_boot_tp_sl_fill_info,
                    timestamp_kst=pd.Timestamp.now(tz="Asia/Seoul"),
                    regime_name=str(getattr(final_governor, "owner_regime", "") or ""),
                    governor_source=f"{OMEGA4_6_1_OWNER}|exchange_reconcile_boot",
                )
                _boot_reconcile_close_payload = getattr(meta_router, "_last_reconcile_close_payload", None)
                if _boot_reconcile_close_payload:
                    await journal_writer.append_many(
                        TRADE_JOURNAL_PATH, [dict(_boot_reconcile_close_payload)]
                    )
                    if final_governor.owner == OMEGA4_6_1_OWNER:
                        if _boot_tp_order_id or _boot_sl_order_id:
                            await live_executor.cancel_tp_sl_orders(
                                tp_order_id=_boot_tp_order_id, sl_order_id=_boot_sl_order_id
                            )
                        final_governor._reset_omega4_6_1_position_state()
                        final_governor.owner = ""
                        final_governor.owner_regime = ""
                        final_governor._save_runtime_state()
            # Defensive cleanup: if the reloaded runtime state still carries TP/SL order ids while
            # the local position is already flat (e.g. the process crashed after a normal close but
            # before _save_runtime_state cleared them), those ids can't correspond to any current
            # position. Leaving them non-empty would make execute_to_target's race guard (see
            # BinanceFuturesExecutionAdapter.execute_to_target) permanently block every future
            # entry, mistaking the stale ids for an unreconciled external close.
            if meta_router.pos not in {"LONG", "SHORT"}:
                _stale_tp_id = str(getattr(final_governor, "active_omega4_6_1_tp_order_id", "") or "")
                _stale_sl_id = str(getattr(final_governor, "active_omega4_6_1_sl_order_id", "") or "")
                if _stale_tp_id or _stale_sl_id:
                    logger.warning(
                        "SYSTEM stale omega4_6_1 tp/sl order ids found at boot with no local position "
                        "tp_id=%s sl_id=%s -- cancelling and clearing", _stale_tp_id, _stale_sl_id
                    )
                    await live_executor.cancel_tp_sl_orders(tp_order_id=_stale_tp_id, sl_order_id=_stale_sl_id)
                    final_governor._reset_omega4_6_1_position_state()
                    final_governor._save_runtime_state()
        # 재시작 직후 기존 포지션을 "현재 기준점"으로 고정한다.
        # (None으로 두면 첫 사이클에서 기존 포지션도 신규 진입처럼 기록될 수 있음)
        _prev_meta_pos = meta_router.pos
        if _bars_stale(eth_buffer):
            logger.warning("⚠️ stale candle 상태로 첫 사이클 스킵")
            return

        await _execute_pending_next_open(eth_buffer)
        _boot_cycle_t0 = time.perf_counter()
        await _run_cycle(processed_boot, eth_buffer)
        if FINAL_GOVERNOR_TIMING_LOG_ENABLE:
            logger.info("TIMING boot_run_cycle sec=%.2f", time.perf_counter() - _boot_cycle_t0)
        _dashboard_shadow_task = task_supervisor.create(
            _dashboard_shadow_loop(), name="dashboard-shadow-loop"
        )
        _daily_trade_report_task = task_supervisor.create(
            _daily_trade_report_loop(), name="daily-trade-report-loop"
        )

        first_run = True
        while not use_local:
            _main_cycle_t0 = time.perf_counter()
            _fetch_sec = 0.0
            _process_sec = 0.0
            _run_cycle_sec = 0.0
            if not first_run:
                now = time.time()
                next_cycle_ts = now - (now % 300) + 300 + max(0.0, FINAL_GOVERNOR_BAR_FETCH_DELAY_SEC)
                while True:
                    remaining = float(next_cycle_ts - time.time())
                    if remaining <= 0.0:
                        break
                    await asyncio.sleep(min(1.0, remaining))

                if CONSOLE_LOG_REFRESH:
                    logger.info("DATA refresh=START")
                try:
                    _fetch_t0 = time.perf_counter()
                    new_eth, new_btc = await fetcher.fetch_latest_patch()
                    _fetch_sec = time.perf_counter() - _fetch_t0
                except Exception as e:
                    logger.warning("⚠️ 최신 캔들 갱신 실패(이번 사이클 스킵): %s", e)
                    _notify_execution_alert(
                        build_execution_alert(
                            {"requested_enabled": True, "enabled": True, "blocking": True, "status": "market_data_refresh_error", "error": str(e)},
                            observed_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                        )
                    )
                    continue
                _keep_bars = max(7000, int(FINAL_GOVERNOR_BUFFER_BARS), int(FINAL_GOVERNOR_MACRO_LOOKBACK_BARS) + 512)
                eth_buffer = pd.concat([eth_buffer, new_eth]).drop_duplicates('timestamp').tail(_keep_bars)
                btc_buffer = pd.concat([btc_buffer, new_btc]).drop_duplicates('timestamp').tail(_keep_bars)
                if _bars_stale(eth_buffer):
                    logger.warning("⚠️ 데이터 지연으로 이번 사이클 판단 스킵")
                    continue
            else:
                logger.info(
                    "SYSTEM live_loop=START mode=%s governor=%s legacy_macro_sniper=%s trend=%s micro=%s",
                    "local" if use_local else "exchange",
                    "disabled_v13_1"
                    if final_governor._v13_1_available()
                    else ("ddh2_full_1x" if final_governor.ddh2_ensemble_enabled else "lifecycle_v1_clean_base"),
                    bool(FINAL_GOVERNOR_SNIPER_ENABLE or FINAL_GOVERNOR_MACRO_ENABLE),
                    bool(FINAL_GOVERNOR_TREND_ENABLE),
                    bool(FINAL_GOVERNOR_MICRO_ENABLE),
                )
                first_run = False

            async with transition_gate.transition("pending_next_open"):
                await _execute_pending_next_open(eth_buffer)
            _process_eth_buffer = eth_buffer
            _process_btc_buffer = btc_buffer
            if FINAL_GOVERNOR_LIVE_PROCESS_BARS > 0:
                _tail_n = int(max(600, FINAL_GOVERNOR_LIVE_PROCESS_BARS))
                _process_eth_buffer = eth_buffer.tail(_tail_n)
                _process_btc_buffer = btc_buffer.tail(_tail_n)
            _process_t0 = time.perf_counter()
            processed_full_df = fe_engine.process(_process_eth_buffer, _process_btc_buffer)
            processed_df = processed_full_df
            if FINAL_GOVERNOR_LIVE_MODEL_BARS > 0:
                processed_df = processed_full_df.tail(int(max(600, FINAL_GOVERNOR_LIVE_MODEL_BARS))).copy()
            _process_sec = time.perf_counter() - _process_t0
            _run_t0 = time.perf_counter()
            async with transition_gate.transition("bar_cycle"):
                await _run_cycle(processed_df, eth_buffer)
            _run_cycle_sec = time.perf_counter() - _run_t0
            if FINAL_GOVERNOR_TIMING_LOG_ENABLE:
                try:
                    _latest_bar_kst = pd.Timestamp(eth_buffer["timestamp"].iloc[-1]) + pd.Timedelta(hours=9)
                except Exception:
                    _latest_bar_kst = ""
                logger.info(
                    "TIMING main_cycle fetch=%.2fs process=%.2fs run_cycle=%.2fs total=%.2fs rows_eth=%d rows_process=%d rows_feature_source=%d process_tail=%d model_tail=%d latest_bar=%s",
                    float(_fetch_sec),
                    float(_process_sec),
                    float(_run_cycle_sec),
                    time.perf_counter() - _main_cycle_t0,
                    len(eth_buffer),
                    len(processed_df),
                    len(processed_full_df),
                    int(FINAL_GOVERNOR_LIVE_PROCESS_BARS),
                    int(FINAL_GOVERNOR_LIVE_MODEL_BARS),
                    str(_latest_bar_kst),
                )

    finally:
        shadow_fetchers = [
            ctx.get("fetcher")
            for ctx in omega461_shadow_assets.values()
            if ctx.get("fetcher") is not None
        ]
        await shutdown_runtime_resources(
            task_supervisor=task_supervisor,
            journal_writer=journal_writer,
            scanners=(ms_scanner, ms_scanner_sol, ms_scanner_btc),
            tail_interceptor=tr_interceptor,
            fetchers=(*shadow_fetchers, fetcher),
            on_error=lambda stage, resource, error: logger.error(
                "resource_close_failed stage=%s resource=%s error=%s",
                stage,
                type(resource).__name__,
                error,
            ),
        )


if __name__ == "__main__":
    _TRADING_BOT_PROCESS_LOCK_FH = acquire_trading_bot_process_lock(
        journal_path=os.getenv("TRADE_JOURNAL_PATH", "data/live/trade_journal.jsonl"),
        lock_path=os.getenv("TRADING_BOT_PROCESS_LOCK_PATH") or None,
    )
    try:
        asyncio.run(main(use_local=False))
    except Exception as _fatal_error:
        _fatal_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
        _fatal_reason = f"{type(_fatal_error).__name__}: {_fatal_error}"
        _fatal_alert = {
            "active": True,
            "severity": "error",
            "title": "트레이딩봇 프로세스 오류",
            "reason": _fatal_reason,
            "occurred_at": _fatal_at,
            "status": "fatal_runtime_error",
        }
        logger.exception("SYSTEM trading_bot=BAD fatal_error=%s", _fatal_error)
        try:
            _fatal_state = _read_json_safe(DASHBOARD_STATE_PATH)
            if not isinstance(_fatal_state, dict):
                _fatal_state = {}
            _fatal_state["execution_alert"] = dict(_fatal_alert)
            _atomic_write_json(DASHBOARD_STATE_PATH, _fatal_state)
        except Exception as _fatal_state_error:
            logger.error("fatal dashboard alert write failed: %s", _fatal_state_error)
        try:
            asyncio.run(
                TelegramNotifier().notify(
                    "<b>[트레이딩봇 치명적 오류]</b>\n"
                    f"time: <code>{html.escape(_fatal_at)}</code>\n"
                    f"error: <code>{html.escape(_fatal_reason[:1000])}</code>"
                )
            )
        except Exception as _fatal_telegram_error:
            logger.error("fatal Telegram alert failed: %s", _fatal_telegram_error)
        raise
