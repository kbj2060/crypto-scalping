import importlib.util
import inspect
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_btc_sol_micro_scalp_shadow_bot_20260718.py"
SPEC = importlib.util.spec_from_file_location("btc_sol_micro_scalp_shadow_bot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cross_asset_shadow_has_no_exchange_execution_capability() -> None:
    source = inspect.getsource(MODULE)
    for forbidden in (
        "binance_execution",
        "BinanceFuturesExecutionAdapter",
        "create_order",
        "cancel_order",
        "place_order",
        "submit_order",
    ):
        assert forbidden not in source
    assert '"activation_allowed": False' in source
    assert '"order_submission_supported": False' in source


def test_frozen_adapter_contracts_are_bound_to_exact_parent() -> None:
    runtime = MODULE.binding.observer.load_runtime(device_name="cpu")
    for asset in MODULE.ASSETS:
        artifact = MODULE.load_asset_artifact(asset, runtime)
        assert artifact["parent_model_sha256"] == runtime.model_sha256
        assert artifact["training_performed"] is False
        assert artifact["parameter_updates"] == 0
        assert artifact["activation_allowed"] is False
        assert artifact["order_submission_supported"] is False


def test_disabled_btc_research_policy_is_cash_fail_closed() -> None:
    runtime = MODULE.binding.observer.load_runtime(device_name="cpu")
    artifact = MODULE.load_asset_artifact("btc", runtime)
    assert artifact["selected_research_policy"]["enabled"] is False
    shadow_runtime = MODULE._shadow_runtime(runtime, artifact)
    assert shadow_runtime.policy.enabled is True
    assert shadow_runtime.policy.switch_margin_bp == 1_000_000_000.0


def test_cross_asset_decisions_settle_only_on_following_close(tmp_path: Path) -> None:
    database = tmp_path / "btc-shadow.duckdb"
    artifact = {
        "model_id": "fixture-btc-adapter",
        "artifact_sha256": "adapter-hash",
        "parent_model_id": "fixture-parent",
        "parent_model_sha256": "parent-hash",
        "fresh_start": pd.Timestamp("2026-07-18 06:41:00"),
        "selected_research_policy": {"enabled": True},
    }
    decisions = [
        {
            "timestamp": pd.Timestamp("2026-07-18 06:41:00"),
            "feature_hash_sha256": "a",
            "close": 100.0,
            "available": True,
            "previous_position": 0,
            "target_position": 1,
            "position_change": 1,
            "intent_id": "intent-a",
            "intent_side": "BUY",
            "notional_change": 1.0,
            "diagnostics": {},
        },
        {
            "timestamp": pd.Timestamp("2026-07-18 06:42:00"),
            "feature_hash_sha256": "b",
            "close": 101.0,
            "available": True,
            "previous_position": 1,
            "target_position": 1,
            "position_change": 0,
            "intent_id": None,
            "intent_side": None,
            "notional_change": 0.0,
            "diagnostics": {},
        },
    ]
    assert MODULE.commit_decisions(database, "btc", artifact, decisions) == 2
    assert MODULE.eth_shadow.settle_shadow_pnl(database, (4.5,)) == 1
    connection = duckdb.connect(str(database), read_only=True)
    try:
        row = connection.execute(
            "SELECT decision_timestamp, settlement_timestamp, net_return FROM shadow_pnl"
        ).fetchone()
    finally:
        connection.close()
    assert str(row[0]) == "2026-07-18 06:41:00"
    assert str(row[1]) == "2026-07-18 06:42:00"
    assert row[2] == pytest.approx(0.01 - 0.00045)
