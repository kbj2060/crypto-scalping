import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_micro_scalp_reuse_layers_20260718.py"
SPEC = importlib.util.spec_from_file_location("micro_scalp_reuse_layers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _lifecycle_fixture() -> dict:
    q = np.zeros((3, 3, 3), dtype=float)
    q[0, :, 2] = 2.0
    q[1:, :, 1] = 2.0
    expert_q = np.repeat(q[:, None, :, :], 6, axis=1)
    continuation = np.zeros((3, 3), dtype=float)
    continuation[1, 2] = 1.0
    continuation[2, 2] = -1.0
    expert_continuation = np.repeat(continuation[:, None, :], 6, axis=1)
    return {
        "asset": "eth",
        "timestamps": pd.date_range("2026-07-18 00:00:00", periods=3, freq="1min"),
        "returns": np.zeros(3),
        "available": np.ones(3, dtype=bool),
        "liquidity_healthy": np.ones(3, dtype=bool),
        "desired": np.asarray([1, 0, 0], dtype=np.int8),
        "edge_bp": np.asarray([2.0, 0.0, 0.0]),
        "agreement": np.asarray([6, 6, 6]),
        "uncertainty": np.zeros(3),
        "gate_entropy": np.zeros(3),
        "risk_score": np.zeros(3),
        "high_risk": np.zeros(3, dtype=bool),
        "close": np.ones(3),
        "source_rows": 3,
        "prediction": {
            "q": q,
            "expert_q": expert_q,
            "continuation": continuation,
            "expert_continuation": expert_continuation,
        },
    }


def test_lifecycle_head_can_extend_and_then_exit_without_fixed_hold() -> None:
    policy = MODULE.LifecyclePolicy(0.0, 4, 0.0, 4, 0.0, False)
    data = _lifecycle_fixture()
    entry_only, _, _ = MODULE.lifecycle_positions(data, policy, dynamic_exit=False)
    lifecycle, counters, _ = MODULE.lifecycle_positions(data, policy, dynamic_exit=True)
    assert entry_only.tolist() == [1, 0, 0]
    assert lifecycle.tolist() == [1, 1, 0]
    assert counters["extended_parent_cash_bars"] == 1
    assert counters["early_exit_triggers"] == 1


def test_allocator_holds_current_asset_inside_switch_margin() -> None:
    candidates = np.asarray([[1, 1, 0], [1, 1, 0], [0, -1, 1]], dtype=float)
    priorities = np.asarray([[2.0, 1.0, 0.0], [1.0, 1.2, 0.0], [0.0, 1.0, 0.5]])
    weights = MODULE.allocator_weights(candidates, priorities, switch_margin=0.5)
    assert weights.tolist() == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    assert np.all(np.sum(np.abs(weights) > 0.0, axis=1) <= 1)


def test_reuse_experiment_does_not_consume_forbidden_historical_inputs() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "read_csv" not in called
    assert "backward" not in called
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"trade_ledgers_used_as_input": False' in source
    assert '"saved_parent_exit_timestamps_used": False' in source
    assert '"future_rows_used_for_entry": False' in source
    assert '"order_submission_supported": False' in source


def test_generated_report_is_research_only_and_chronological() -> None:
    if not MODULE.REPORT_PATH.exists():
        return
    report = json.loads(MODULE.REPORT_PATH.read_text())
    assert report["fresh_forward_bar_by_bar"] is True
    assert report["trade_ledgers_used_as_input"] is False
    assert report["saved_parent_exit_timestamps_used"] is False
    assert report["future_rows_used_for_entry"] is False
    assert report["fixed_holding_period_used"] is False
    assert report["activation_allowed"] is False
    assert report["order_submission_supported"] is False
    assert report["selection_uses_only_tune_split"] is True
    assert report["validation_used_for_selection"] is False
