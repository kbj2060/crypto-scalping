import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_eth_micro_scalp_v3_fresh_forward_observer_20260718.py"
SPEC = importlib.util.spec_from_file_location("eth_micro_scalp_v3_observer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _prediction(rows: int, experts: int = 3) -> dict[str, np.ndarray]:
    return {
        "q": np.zeros((rows, 3, 3), dtype=np.float32),
        "expert_q": np.zeros((rows, experts, 3, 3), dtype=np.float32),
        "continuation": np.zeros((rows, 3), dtype=np.float32),
        "expert_continuation": np.zeros((rows, experts, 3), dtype=np.float32),
    }


def test_five_minute_feature_stream_fails_fast(tmp_path: Path) -> None:
    required = ["timestamp", "close", *MODULE.v3.core.BASE_FEATURES, *MODULE.v3.core.MICRO_FEATURES]
    rows = []
    for timestamp in pd.date_range("2026-07-18", periods=2, freq="5min"):
        row = {name: 0.0 for name in required}
        row["timestamp"] = timestamp
        row["close"] = 1800.0
        rows.append(row)
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="cadence violation"):
        MODULE.load_feature_stream(path)


def test_missing_unavailable_micro_payload_is_allowed(tmp_path: Path) -> None:
    required = ["timestamp", "close", *MODULE.v3.core.BASE_FEATURES, *MODULE.v3.core.MICRO_FEATURES]
    rows = []
    for timestamp in pd.date_range("2026-07-18", periods=2, freq="1min"):
        row = {name: 0.0 for name in required}
        row["timestamp"] = timestamp
        row["close"] = 1800.0
        row["micro_available"] = 0.0
        row["book_available"] = 0.0
        row["micro_data_stale"] = np.nan
        row["micro_depth_connected"] = np.nan
        row["micro_warmup_30m_ready"] = np.nan
        row["micro_age_min"] = np.nan
        row["book_age_min"] = np.nan
        row["book_spread_bps"] = np.nan
        rows.append(row)
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    loaded = MODULE.load_feature_stream(path)
    assert len(loaded) == 2


def test_stateful_single_step_matches_batch_policy() -> None:
    prediction = _prediction(7)
    prediction["q"][0, 1, 2] = 4.0
    prediction["q"][1, 2, 2] = 2.0
    prediction["q"][2, 2, 1] = 3.0
    prediction["q"][3, 1, 0] = 4.0
    prediction["q"][4:6, 0, 0] = 2.0
    prediction["q"][6, 0, 1] = 3.0
    prediction["expert_q"][:] = prediction["q"][:, None]
    policy = MODULE.v3.OpportunityPolicy(True, 0.0, 1, False, 0.0, 3)
    batch, _ = MODULE.v3.decide_positions(
        prediction, np.ones(7, dtype=bool), policy
    )
    previous = 0
    step = []
    for index in range(7):
        previous, _ = MODULE.decide_next(prediction, index, True, previous, policy)
        step.append(previous)
    assert step == batch.tolist()


def test_execution_observation_provenance_is_strict() -> None:
    counterfactual = MODULE.validate_execution_observation(
        {
            "decision_timestamp": "2026-07-18 00:00:00",
            "observation_type": "orderbook_counterfactual",
            "observation_id": "book-1",
            "observed_at_utc": "2026-07-18 00:01:00",
            "execution_status": "not_submitted",
            "source": "unit-test-book",
        }
    )
    assert counterfactual["performance_eligible"] is False
    with pytest.raises(ValueError, match="order_id"):
        MODULE.validate_execution_observation(
            {
                "decision_timestamp": "2026-07-18 00:00:00",
                "observation_type": "actual_exchange_fill",
                "observation_id": "fill-1",
                "observed_at_utc": "2026-07-18 00:01:00",
                "execution_status": "filled",
                "source": "unit-test-exchange",
                "requested_quantity": 1.0,
                "filled_quantity": 1.0,
                "average_fill_price": 1800.0,
            }
        )
    partial = MODULE.validate_execution_observation(
        {
            "decision_timestamp": "2026-07-18 00:00:00",
            "observation_type": "actual_exchange_fill",
            "observation_id": "fill-2",
            "observed_at_utc": "2026-07-18 00:01:00",
            "execution_status": "partial",
            "source": "unit-test-exchange",
            "order_id": "order-2",
            "requested_quantity": 1.0,
            "filled_quantity": 0.4,
            "average_fill_price": 1800.0,
        }
    )
    assert partial["performance_eligible"] is False


def test_only_full_actual_fill_is_performance_eligible() -> None:
    filled = MODULE.validate_execution_observation(
        {
            "decision_timestamp": "2026-07-18 00:00:00",
            "observation_type": "actual_exchange_fill",
            "observation_id": "fill-3",
            "observed_at_utc": "2026-07-18 00:01:00",
            "execution_status": "filled",
            "source": "unit-test-exchange",
            "order_id": "order-3",
            "requested_quantity": 1.0,
            "filled_quantity": 1.0,
            "average_fill_price": 1800.0,
        }
    )
    assert filled["performance_eligible"] is True


def test_observer_source_has_no_order_submission_dependency() -> None:
    source = inspect.getsource(MODULE)
    for forbidden in ("import ccxt", "import requests", ".create_order(", "import trading_bot"):
        assert forbidden not in source


def test_decision_commit_is_idempotent(tmp_path: Path) -> None:
    class Runtime:
        model_sha256 = "abc"

    runtime = Runtime()
    decision = {
        "timestamp": pd.Timestamp("2026-07-18 00:00:00"),
        "model_id": MODULE.MODEL_ID,
        "model_sha256": "abc",
        "feature_hash_sha256": "feature",
        "close": 1800.0,
        "available": True,
        "previous_position": 0,
        "target_position": 1,
        "position_change": 1,
        "intent_id": "intent",
        "intent_side": "BUY",
        "notional_change": 1.0,
        "diagnostics": {},
    }
    database = tmp_path / "observer.duckdb"
    assert MODULE.commit_decisions(database, runtime, [decision]) == 1
    assert MODULE.commit_decisions(database, runtime, [decision]) == 0
    summary = MODULE.observer_summary(database)
    assert summary["decision_count"] == 1
    assert summary["performance_eligible"] is False
