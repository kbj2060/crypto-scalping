"""Fast-twitch v5 binding for the non-executing fresh-forward observer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_eval_eth_micro_scalp_fast_twitch_v5_20260718 as v5  # noqa: E402


FRESH_START_UTC = pd.Timestamp("2026-07-18 03:55:00")
OBSERVER_DIR = v5.ARTIFACT_DIR / "fresh_forward_observer"
DEFAULT_FEATURE_STREAM = ROOT / "data/live/eth_micro_scalp_v5_features_1m.csv"

OBSERVER_SCRIPT = SCRIPT_DIR / "run_eth_micro_scalp_v3_fresh_forward_observer_20260718.py"
OBSERVER_SPEC = importlib.util.spec_from_file_location(
    "eth_micro_scalp_v3_observer_isolated_for_v5", OBSERVER_SCRIPT
)
if OBSERVER_SPEC is None or OBSERVER_SPEC.loader is None:
    raise RuntimeError("cannot load isolated v5 observer runtime")
observer = importlib.util.module_from_spec(OBSERVER_SPEC)
sys.modules[OBSERVER_SPEC.name] = observer
OBSERVER_SPEC.loader.exec_module(observer)

core_view = SimpleNamespace(
    BASE_FEATURES=v5.v4.SOURCE_STABLE_FEATURES,
    MICRO_FEATURES=v5.v3.core.MICRO_FEATURES,
    HEALTH_FEATURES=v5.v3.core.HEALTH_FEATURES,
    ACTIONS=v5.v3.core.ACTIONS,
    apply_scaler=v5.v3.core.apply_scaler,
)
model_view = SimpleNamespace(
    __file__=v5.__file__,
    MODEL_PATH=v5.MODEL_PATH,
    MODEL_ID=v5.MODEL_ID,
    ARTIFACT_DIR=v5.ARTIFACT_DIR,
    core=core_view,
    OpportunityConfig=v5.FastTwitchConfig,
    OpportunityPolicy=v5.v3.OpportunityPolicy,
    OpportunityCostMoE=v5.FastTwitchOpportunityMoE,
    infer=v5.infer,
    aggregate_seed_predictions=v5.v3.aggregate_seed_predictions,
)

observer.v3 = model_view
observer.MODEL_PATH = v5.MODEL_PATH
observer.MODEL_ID = v5.MODEL_ID
observer.FRESH_START_UTC = FRESH_START_UTC
observer.DEFAULT_FEATURE_STREAM = DEFAULT_FEATURE_STREAM
observer.OBSERVER_DIR = OBSERVER_DIR
observer.DEFAULT_OBSERVER_DB = OBSERVER_DIR / "observer.duckdb"
observer.DEFAULT_READINESS_REPORT = OBSERVER_DIR / "readiness.json"
observer.FEATURE_BUILD_REPORT = OBSERVER_DIR / "feature_stream_build.json"


if __name__ == "__main__":
    raise SystemExit(observer.main())
