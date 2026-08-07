import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_eval_eth_micro_scalp_inventory_moe_20260718.py"
SPEC = importlib.util.spec_from_file_location("eth_micro_scalp_inventory_moe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cost_aware_teacher_prefers_holding_profitable_inventory() -> None:
    returns = np.full(20, 0.001, dtype=np.float64)
    q, action = MODULE.build_cost_aware_teacher(
        returns,
        np.ones(20, dtype=bool),
        np.zeros(20),
        fee_per_notional_change=0.00045,
        gamma=1.0,
        inventory_vol_weight=0.0,
        advantage_clip_bp=50.0,
    )
    assert q.shape == (20, 3, 3)
    assert action.shape == (20, 3)
    assert action[1, 2] == 2


def test_teacher_forces_cash_when_market_state_is_unavailable() -> None:
    _, action = MODULE.build_cost_aware_teacher(
        np.array([0.01]),
        np.array([False]),
        np.array([0.0]),
        fee_per_notional_change=0.00045,
        gamma=1.0,
        inventory_vol_weight=0.0,
        advantage_clip_bp=50.0,
    )
    assert action.tolist() == [[1, 1, 1]]


def test_q_policy_has_no_max_holding_period() -> None:
    q = np.zeros((500, 3, 3), dtype=np.float32)
    q[:, 1, 2] = 5.0
    q[:, 2, 2] = 2.0
    positions = MODULE.decide_q_positions(q, np.ones(500, dtype=bool), MODULE.QPolicy(True, 0.0, 1))
    assert positions.tolist() == [1] * 500


def test_q_policy_can_close_and_reverse_at_different_durations() -> None:
    q = np.zeros((7, 3, 3), dtype=np.float32)
    q[0, 1, 2] = 4.0
    q[1, 2, 2] = 2.0
    q[2, 2, 1] = 3.0
    q[3, 1, 0] = 4.0
    q[4, 0, 0] = 2.0
    q[5, 0, 0] = 2.0
    q[6, 0, 1] = 3.0
    positions = MODULE.decide_q_positions(q, np.ones(7, dtype=bool), MODULE.QPolicy(True, 0.0, 1))
    assert positions.tolist() == [1, 1, 0, -1, -1, -1, 0]


def test_q_replay_records_the_actual_availability_gate() -> None:
    q = np.zeros((3, 3, 3), dtype=np.float32)
    available = np.array([True, False, True])
    _, ledger = MODULE.replay_q_policy(
        q,
        available,
        np.zeros(3),
        pd.date_range("2026-01-01", periods=3, freq="min"),
        MODULE.QPolicy(False, 0.0, 3),
        fee=0.00045,
    )
    assert ledger["available"].tolist() == available.tolist()


def test_model_emits_all_inventory_action_values_and_regime_weights() -> None:
    config = MODULE.Config(window=8, base_channels=12, micro_channels=8, latent_dim=16, experts=3)
    model = MODULE.InventoryMoEQPolicy(n_base=5, n_micro=4, n_aux=7, config=config)
    q, auxiliary, gate, expert_q = model(torch.randn(3, 8, 5), torch.randn(3, 8, 4))
    assert q.shape == (3, 3, 3)
    assert auxiliary.shape == (3, 7)
    assert gate.shape == (3, 3)
    assert expert_q.shape == (3, 3, 3, 3)
    assert torch.allclose(gate.sum(dim=1), torch.ones(3), atol=1e-6)


def test_consensus_can_block_a_disputed_position_change() -> None:
    mixed = np.zeros((1, 3, 3), dtype=np.float32)
    mixed[0, 1, 2] = 3.0
    experts = np.zeros((1, 3, 3, 3), dtype=np.float32)
    experts[0, 0, 1, 2] = 3.0
    experts[0, 1:, 1, 1] = 3.0
    position = MODULE.decide_q_positions(
        mixed,
        np.array([True]),
        MODULE.QPolicy(True, 0.0, 2),
        experts,
    )
    assert position.tolist() == [0]


def test_feature_contract_excludes_btc_and_rule_outputs() -> None:
    forbidden = {"kelly_mult", "signal_bias", "eai"}
    assert not any("btc" in name.lower() for name in MODULE.BASE_FEATURES + MODULE.MICRO_FEATURES)
    assert forbidden.isdisjoint(MODULE.BASE_FEATURES + MODULE.MICRO_FEATURES)
