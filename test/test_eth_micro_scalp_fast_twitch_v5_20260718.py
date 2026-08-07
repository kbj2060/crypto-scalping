import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_eval_eth_micro_scalp_fast_twitch_v5_20260718.py"
SPEC = importlib.util.spec_from_file_location("eth_micro_scalp_fast_twitch_v5", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _models():
    checkpoint = torch.load(MODULE.v4.MODEL_PATH, map_location="cpu", weights_only=False)
    state = checkpoint["seed_model_states"]["18"]
    config = MODULE.FastTwitchConfig()
    parent = MODULE.v3.OpportunityCostMoE(36, 24, 7, config)
    parent.load_state_dict(state, strict=True)
    fast = MODULE.FastTwitchOpportunityMoE(36, 24, 7, config)
    missing = MODULE.load_v4_weights(fast, state)
    return parent, fast, missing


def test_zero_initialized_fast_head_is_parent_equivalent() -> None:
    parent, fast, missing = _models()
    assert missing and all(name.startswith("fast_q_head.") for name in missing)
    parent.eval()
    fast.eval()
    base = torch.randn(3, 60, 36)
    micro = torch.randn(3, 60, 24)
    with torch.no_grad():
        parent_output = parent(base, micro)
        fast_output = fast(base, micro)
    assert torch.equal(fast_output["fast_q_residual"], torch.zeros_like(fast_output["fast_q_residual"]))
    assert torch.equal(parent_output["q"], fast_output["q"])
    assert torch.equal(parent_output["expert_q"], fast_output["expert_q"])


def test_fast_inputs_are_current_and_strictly_past_only() -> None:
    _, fast, _ = _models()
    base = torch.arange(60 * 36, dtype=torch.float32).reshape(1, 60, 36)
    micro = torch.arange(60 * 24, dtype=torch.float32).reshape(1, 60, 24)
    inputs = fast._fast_inputs(base, micro)
    assert inputs.shape == (1, 180)
    expected_current = torch.cat([base[:, -1], micro[:, -1]], dim=-1)
    assert torch.equal(inputs[:, :60], expected_current)


def test_causal_encoders_are_frozen_during_adapter_training() -> None:
    _, fast, _ = _models()
    trainable = MODULE.configure_adapter_training(fast)
    assert trainable
    for name, parameter in fast.named_parameters():
        assert parameter.requires_grad == name.startswith(MODULE.ADAPTER_TRAINABLE_PREFIXES)


def test_switch_targets_receive_larger_loss_weight() -> None:
    q = torch.zeros(1, 3, 3)
    q[0, 0, 1] = -2.0
    q[0, 1, 2] = -2.0
    q[0, 2, 2] = 2.0
    target = torch.tensor([[1, 2, 2]])
    loss_one, switch = MODULE.weighted_action_loss(q, target, 1.0)
    loss_two, _ = MODULE.weighted_action_loss(q, target, 2.0)
    assert switch.tolist() == [[True, True, False]]
    assert torch.isfinite(loss_one) and torch.isfinite(loss_two)
    assert loss_two > loss_one


def test_v5_has_no_fixed_holding_or_order_path() -> None:
    source = SCRIPT.read_text()
    assert '"fixed_holding_period_used": False' in source
    assert '"activation_allowed": False' in source
    for forbidden in ("create_order", "cancel_order", "trading_bot"):
        assert forbidden not in source
