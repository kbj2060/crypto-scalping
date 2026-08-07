import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_eval_eth_micro_scalp_opportunity_moe_20260718.py"
SPEC = importlib.util.spec_from_file_location("eth_micro_scalp_opportunity_moe", SCRIPT)
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


def test_opportunity_target_is_hold_value_minus_best_alternative() -> None:
    q = np.array(
        [[[-2.0, 0.0, -4.0], [-1.0, -3.0, 0.0], [-5.0, -1.0, 0.0]]],
        dtype=np.float32,
    )
    continuation, exit_target = MODULE.build_opportunity_targets(q)
    assert continuation.tolist() == [[-2.0, -3.0, 1.0]]
    assert exit_target.tolist() == [[1.0, 1.0, 0.0]]


def test_opportunity_exit_closes_without_a_holding_time_rule() -> None:
    prediction = _prediction(3)
    prediction["q"][0, 1, 2] = 3.0
    prediction["q"][1:, 2, 2] = 2.0
    prediction["q"][1:, 2, 1] = 1.0
    prediction["q"][2, 1, 1] = 1.0
    prediction["expert_q"][:, :, :, :] = prediction["q"][:, None, :, :]
    prediction["continuation"][1, 2] = -2.0
    prediction["expert_continuation"][1, :, 2] = -2.0
    policy = MODULE.OpportunityPolicy(True, 0.0, 1, True, -1.0, 2)
    positions, triggers = MODULE.decide_positions(prediction, np.ones(3, dtype=bool), policy)
    assert positions.tolist() == [1, 0, 0]
    assert triggers.tolist() == [False, True, False]


def test_positive_continuation_can_hold_indefinitely() -> None:
    prediction = _prediction(500)
    prediction["q"][0, 1, 2] = 3.0
    prediction["q"][:, 2, 2] = 2.0
    prediction["expert_q"][:, :, :, :] = prediction["q"][:, None, :, :]
    prediction["continuation"][:, 2] = 2.0
    prediction["expert_continuation"][:, :, 2] = 2.0
    policy = MODULE.OpportunityPolicy(True, 0.0, 1, True, 0.0, 2)
    positions, triggers = MODULE.decide_positions(prediction, np.ones(500, dtype=bool), policy)
    assert positions.tolist() == [1] * 500
    assert not triggers.any()


def test_expert_uncertainty_can_veto_a_fragile_entry() -> None:
    prediction = _prediction(1)
    prediction["q"][0, 1, 1] = 1.0
    prediction["q"][0, 1, 2] = 2.0
    prediction["expert_q"][0, :, 1, 1] = 1.0
    prediction["expert_q"][0, :, 1, 2] = np.array([10.0, 2.0, -8.0])
    plain = MODULE.OpportunityPolicy(True, 0.0, 1, False, 0.0, 3)
    conservative = MODULE.OpportunityPolicy(True, 0.0, 1, False, 0.0, 3, 1.0)
    plain_position, _ = MODULE.decide_positions(prediction, np.ones(1, dtype=bool), plain)
    conservative_position, _ = MODULE.decide_positions(
        prediction, np.ones(1, dtype=bool), conservative
    )
    assert plain_position.tolist() == [1]
    assert conservative_position.tolist() == [0]


def test_parent_warm_start_only_leaves_new_heads_uninitialized() -> None:
    config = MODULE.OpportunityConfig(
        window=8, base_channels=12, micro_channels=8, latent_dim=16, experts=3
    )
    parent_model = MODULE.core.InventoryMoEQPolicy(5, 4, 7, config)
    model = MODULE.OpportunityCostMoE(5, 4, 7, config)
    missing = MODULE.load_parent_weights(model, parent_model.state_dict())
    assert missing
    assert all(
        key.startswith(("continuation_head.", "exit_hazard_head.")) for key in missing
    )


def test_only_new_opportunity_heads_are_trainable() -> None:
    config = MODULE.OpportunityConfig(
        window=8, base_channels=12, micro_channels=8, latent_dim=16, experts=3
    )
    model = MODULE.OpportunityCostMoE(5, 4, 7, config)
    trainable = MODULE.freeze_parent_parameters(model)
    assert trainable
    assert all(
        name.startswith(("continuation_head.", "exit_hazard_head.")) for name in trainable
    )
    assert not model.q_head[0].weight.requires_grad


def test_model_emits_inventory_specific_opportunity_values() -> None:
    config = MODULE.OpportunityConfig(
        window=8, base_channels=12, micro_channels=8, latent_dim=16, experts=3
    )
    output = MODULE.OpportunityCostMoE(5, 4, 7, config)(
        torch.randn(2, 8, 5), torch.randn(2, 8, 4)
    )
    assert output["q"].shape == (2, 3, 3)
    assert output["continuation"].shape == (2, 3)
    assert output["expert_continuation"].shape == (2, 3, 3)
    assert output["exit_logit"].shape == (2, 3)


def test_seed_aggregation_preserves_all_opportunity_experts() -> None:
    first = {
        "q": np.ones((2, 3, 3), dtype=np.float32),
        "continuation": np.ones((2, 3), dtype=np.float32),
        "exit_logit": np.ones((2, 3), dtype=np.float32),
        "gate": np.ones((2, 3), dtype=np.float32),
        "expert_q": np.ones((2, 3, 3, 3), dtype=np.float32),
        "expert_continuation": np.ones((2, 3, 3), dtype=np.float32),
        "expert_exit_logit": np.ones((2, 3, 3), dtype=np.float32),
    }
    second = {key: value * 3 for key, value in first.items()}
    result = MODULE.aggregate_seed_predictions([first, second])
    assert np.allclose(result["continuation"], 2.0)
    assert result["expert_continuation"].shape == (2, 6, 3)
