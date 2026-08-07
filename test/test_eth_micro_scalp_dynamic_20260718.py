import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_eval_eth_micro_scalp_dynamic_20260718.py"
SPEC = importlib.util.spec_from_file_location("eth_micro_scalp_dynamic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_dynamic_policy_has_variable_holding_periods() -> None:
    policy = MODULE.DynamicPolicy(enabled=True, entry_threshold_bp=10.0, exit_threshold_bp=0.0)
    scores = np.array([12.0, 5.0, 4.0, -1.0, -12.0, -4.0, 1.0, 0.0])
    positions = MODULE.decide_positions(scores, np.ones(len(scores), dtype=bool), policy)
    assert positions.tolist() == [1, 1, 1, 0, -1, -1, 0, 0]
    assert MODULE.holding_lengths(positions) == [3, 2]


def test_position_can_remain_open_without_a_max_hold() -> None:
    policy = MODULE.DynamicPolicy(enabled=True, entry_threshold_bp=10.0, exit_threshold_bp=0.0)
    scores = np.r_[12.0, np.full(99, 1.0)]
    positions = MODULE.decide_positions(scores, np.ones(len(scores), dtype=bool), policy)
    assert positions.tolist() == [1] * 100
    assert MODULE.holding_lengths(positions) == [100]


def test_unavailable_microstructure_forces_cash_not_a_time_exit() -> None:
    policy = MODULE.DynamicPolicy(enabled=True, entry_threshold_bp=10.0, exit_threshold_bp=0.0)
    scores = np.array([12.0, 12.0, 12.0, 12.0])
    available = np.array([True, True, False, True])
    positions = MODULE.decide_positions(scores, available, policy)
    assert positions.tolist() == [1, 1, 0, 1]


def test_reversal_charges_close_and_reopen_turnover() -> None:
    timestamps = pd.date_range("2026-01-01", periods=2, freq="min")
    positions = np.array([1, -1], dtype=np.int8)
    metrics, ledger = MODULE.replay_positions(
        positions,
        np.zeros(2),
        timestamps,
        fee_per_notional_change=0.001,
    )
    assert ledger["turnover"].tolist() == [1.0, 3.0]
    assert np.isclose(metrics["additive_cost_pct"], 0.4)


def test_split_mask_purges_forward_target_at_boundary() -> None:
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=20, freq="min")
    mask = MODULE.purged_interval_mask(
        timestamps,
        "2026-01-01 00:00:00",
        "2026-01-01 00:15:00",
        horizon_min=5,
    )
    assert timestamps[mask].max() == pd.Timestamp("2026-01-01 00:09:00")


def test_disabled_policy_is_explicit_all_cash() -> None:
    policy = MODULE.DynamicPolicy(enabled=False, entry_threshold_bp=0.0, exit_threshold_bp=0.0)
    positions = MODULE.decide_positions(np.array([100.0, -100.0]), np.array([True, True]), policy)
    assert positions.tolist() == [0, 0]
