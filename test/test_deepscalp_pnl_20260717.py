import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_eval_deepscalp_pnl_20260717.py"
SPEC = importlib.util.spec_from_file_location("deepscalp_pnl", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_futures_sizing_contract() -> None:
    margin_fraction = torch.tensor([0.30])
    signed_notional = margin_fraction * 3.0
    net, gross, turnover = MODULE.account_return(
        signed_notional=signed_notional,
        previous_notional=torch.tensor([0.0]),
        price_return=torch.tensor([0.04]),
        fee_per_notional=0.0,
    )
    assert torch.allclose(signed_notional, torch.tensor([0.90]))
    assert torch.allclose(gross, torch.tensor([0.036]))
    assert torch.allclose(net, gross)
    assert torch.allclose(turnover, torch.tensor([0.90]))


def test_reversal_charges_close_and_reopen_turnover() -> None:
    net, gross, turnover = MODULE.account_return(
        signed_notional=torch.tensor([-0.9]),
        previous_notional=torch.tensor([0.9]),
        price_return=torch.tensor([0.0]),
        fee_per_notional=0.00045,
    )
    assert torch.allclose(turnover, torch.tensor([1.8]))
    assert torch.allclose(gross, torch.tensor([0.0]))
    assert torch.allclose(net, torch.tensor([-0.00081]))


def test_causal_windows_do_not_cross_timestamp_gap() -> None:
    minute = 60_000_000_000
    timestamps = np.array([0, minute, 2 * minute, 10 * minute, 11 * minute, 12 * minute], dtype=np.int64)
    targets = np.ones((6, 7), dtype=np.float32)
    returns = np.ones(6, dtype=np.float32)
    indices = MODULE.causal_window_end_indices(timestamps, targets, returns, window=3)
    assert indices.tolist() == [2, 5]


def test_policy_emits_discrete_side_and_bounded_margin() -> None:
    config = MODULE.Config(window=8, base_channels=8, base_hidden=12, micro_channels=6)
    model = MODULE.DeepScalpPolicy(n_base=5, n_micro=4, config=config)
    base = torch.randn(3, 8, 5)
    micro = torch.randn(3, 8, 4)
    market = model.encode_market(base, micro)
    state = torch.zeros(3, 4)
    notional, side, margin, _ = model.policy_step(market, state, action_mode="argmax")
    assert set(side.tolist()).issubset({-1.0, 0.0, 1.0})
    assert torch.all(margin >= 0.0)
    assert torch.all(margin <= config.max_margin_fraction)
    assert torch.all(torch.abs(notional) <= config.max_margin_fraction * config.leverage + 1e-7)


def test_rule_outputs_are_not_model_inputs() -> None:
    forbidden = {"kelly_mult", "signal_bias", "eai", "scalp_action", "scalp_tp_move", "scalp_sl_move"}
    inputs = set(MODULE.BASE_FEATURES) | set(MODULE.MICRO_COLUMNS) | set(MODULE.BOOK_COLUMNS)
    assert forbidden.isdisjoint(inputs)


def test_btc_open_timestamp_features_are_excluded() -> None:
    assert not any("btc" in name.lower() for name in MODULE.BASE_FEATURES)
