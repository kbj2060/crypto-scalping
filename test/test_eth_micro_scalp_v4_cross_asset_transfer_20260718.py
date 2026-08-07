from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/test_eth_micro_scalp_v4_cross_asset_transfer_20260718.py"
SPEC = importlib.util.spec_from_file_location("eth_v4_cross_asset_transfer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_transfer_script_has_no_training_or_selection_calls() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    called_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_names.add(node.func.attr)
    forbidden = {
        "backward",
        "fit",
        "select_policy",
        "step",
        "train",
        "train_joint_model",
        "train_model",
    }
    assert called_names.isdisjoint(forbidden)


def test_replay_charges_entry_and_exit_cost_on_following_close() -> None:
    decisions = [
        {
            "timestamp": "2026-07-18 00:00:00",
            "close": 100.0,
            "previous_position": 0,
            "target_position": 1,
        },
        {
            "timestamp": "2026-07-18 00:01:00",
            "close": 101.0,
            "previous_position": 1,
            "target_position": 0,
        },
        {
            "timestamp": "2026-07-18 00:02:00",
            "close": 102.0,
            "previous_position": 0,
            "target_position": 0,
        },
    ]
    result = MODULE.replay_decisions(decisions, 4.5)
    expected = (1.0 + 0.01 - 0.00045) * (1.0 - 0.00045) - 1.0
    assert result["settled_intervals"] == 2
    assert result["turnover"] == 2.0
    assert result["additive_cost_pct"] == pytest.approx(0.09)
    assert result["compounded_return_pct"] == pytest.approx(expected * 100.0)
    assert result["holding"]["completed_count"] == 1
    assert result["holding"]["median_minutes"] == 1.0
