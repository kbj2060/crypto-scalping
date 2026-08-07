import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_micro_scalp_reuse_shadow_bot_20260718.py"
SPEC = importlib.util.spec_from_file_location("micro_scalp_reuse_shadow_bot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_reuse_shadow_has_no_exchange_execution_capability() -> None:
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


def test_selected_modes_are_exact_report_bound_and_post_report() -> None:
    runtime = MODULE.binding.observer.load_runtime(device_name="cpu")
    contract = MODULE.load_contract(runtime)
    report_end = pd.Timestamp(contract["report"]["common_interval"]["end_utc"])
    assert contract["modes"]["eth_lifecycle"]["dynamic_exit"] is True
    assert contract["modes"]["sol_entry"]["dynamic_exit"] is False
    for config in contract["modes"].values():
        assert config["fresh_start"] == report_end + pd.Timedelta(minutes=1)
        assert config["policy"].entry_margin_bp < 1e8


def test_lifecycle_can_continue_from_persisted_position() -> None:
    data = {
        "asset": "eth",
        "timestamps": pd.date_range("2026-07-18 08:00:00", periods=2, freq="1min"),
        "available": np.ones(2, dtype=bool),
        "high_risk": np.zeros(2, dtype=bool),
        "desired": np.zeros(2, dtype=np.int8),
        "edge_bp": np.zeros(2),
        "prediction": {
            "q": np.zeros((2, 3, 3)),
            "expert_q": np.zeros((2, 6, 3, 3)),
            "continuation": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
            "expert_continuation": np.asarray(
                [[[0.0, 0.0, 1.0]] * 6, [[0.0, 0.0, -1.0]] * 6]
            ),
        },
    }
    policy = MODULE.reuse.LifecyclePolicy(0.0, 3, 0.0, 3, 0.0, False)
    positions, counters, _ = MODULE.reuse.lifecycle_positions(
        data, policy, dynamic_exit=True, initial_position=1
    )
    assert positions.tolist() == [1, 0]
    assert counters["extended_parent_cash_bars"] == 1
    assert counters["early_exit_triggers"] == 1
