"""Source-stable v4 binding for the non-executing fresh-forward observer."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_eth_micro_scalp_v3_fresh_forward_observer_20260718 as observer  # noqa: E402
import train_eval_eth_micro_scalp_source_stable_v4_20260718 as v4  # noqa: E402


FRESH_START_UTC = pd.Timestamp("2026-07-18 02:45:00")
OBSERVER_DIR = v4.ARTIFACT_DIR / "fresh_forward_observer"
DEFAULT_FEATURE_STREAM = ROOT / "data/live/eth_micro_scalp_v4_features_1m.csv"

core_view = SimpleNamespace(
    BASE_FEATURES=v4.SOURCE_STABLE_FEATURES,
    MICRO_FEATURES=v4.v3.core.MICRO_FEATURES,
    HEALTH_FEATURES=v4.v3.core.HEALTH_FEATURES,
    ACTIONS=v4.v3.core.ACTIONS,
    apply_scaler=v4.v3.core.apply_scaler,
)
model_view = SimpleNamespace(
    __file__=v4.__file__,
    MODEL_PATH=v4.MODEL_PATH,
    MODEL_ID=v4.MODEL_ID,
    ARTIFACT_DIR=v4.ARTIFACT_DIR,
    core=core_view,
    OpportunityConfig=v4.v3.OpportunityConfig,
    OpportunityPolicy=v4.v3.OpportunityPolicy,
    OpportunityCostMoE=v4.v3.OpportunityCostMoE,
    infer=v4.v3.infer,
    aggregate_seed_predictions=v4.v3.aggregate_seed_predictions,
)

observer.v3 = model_view
observer.MODEL_PATH = v4.MODEL_PATH
observer.MODEL_ID = v4.MODEL_ID
observer.FRESH_START_UTC = FRESH_START_UTC
observer.DEFAULT_FEATURE_STREAM = DEFAULT_FEATURE_STREAM
observer.OBSERVER_DIR = OBSERVER_DIR
observer.DEFAULT_OBSERVER_DB = OBSERVER_DIR / "observer.duckdb"
observer.DEFAULT_READINESS_REPORT = OBSERVER_DIR / "readiness.json"
observer.FEATURE_BUILD_REPORT = OBSERVER_DIR / "feature_stream_build.json"


if __name__ == "__main__":
    raise SystemExit(observer.main())
