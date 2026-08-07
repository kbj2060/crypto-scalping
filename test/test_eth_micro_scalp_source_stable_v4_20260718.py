import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_eval_eth_micro_scalp_source_stable_v4_20260718.py"
SPEC = importlib.util.spec_from_file_location("eth_micro_scalp_source_stable_v4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_source_stable_feature_contract_is_exact() -> None:
    assert len(MODULE.SOURCE_UNSTABLE_FEATURES) == 7
    assert len(MODULE.SOURCE_STABLE_FEATURES) == 36
    assert not set(MODULE.SOURCE_UNSTABLE_FEATURES) & set(MODULE.SOURCE_STABLE_FEATURES)
    assert set(MODULE.SOURCE_UNSTABLE_FEATURES) | set(MODULE.SOURCE_STABLE_FEATURES) == set(
        MODULE.v3.core.BASE_FEATURES
    )


def test_source_stability_audit_passes_only_retained_boundary() -> None:
    audit = MODULE.audit_source_stability()
    assert audit["pass"] is True
    assert audit["selection_uses_return_metrics"] is False
    assert audit["source_report_original_parity_pass"] is False
    assert audit["retained_worst_max_scaled_error"] <= audit["thresholds"]["max_scaled_error"]
    assert audit["retained_worst_p99_scaled_error"] <= audit["thresholds"]["p99_scaled_error"]


def test_pruned_warm_start_changes_only_input_projection_width() -> None:
    checkpoint = torch.load(MODULE.v3.MODEL_PATH, map_location="cpu", weights_only=False)
    state = checkpoint["seed_model_states"][str(checkpoint["seeds"][0])]
    pruned = MODULE.prune_v3_state(state, checkpoint["base_feature_names"])
    key = "base_encoder.projection.weight"
    assert pruned[key].shape[1] == len(MODULE.SOURCE_STABLE_FEATURES)
    assert state[key].shape[1] == len(MODULE.v3.core.BASE_FEATURES)
    assert set(pruned) == set(state)
    for name in state:
        if name != key:
            assert torch.equal(pruned[name], state[name])


def test_pruned_state_strictly_loads_source_stable_model() -> None:
    checkpoint = torch.load(MODULE.v3.MODEL_PATH, map_location="cpu", weights_only=False)
    config = MODULE.SourceStableConfig()
    model = MODULE.v3.OpportunityCostMoE(
        len(MODULE.SOURCE_STABLE_FEATURES),
        len(MODULE.v3.core.MICRO_FEATURES),
        7,
        config,
    )
    MODULE.load_pruned_v3_weights(
        model,
        checkpoint["seed_model_states"][str(checkpoint["seeds"][0])],
        checkpoint["base_feature_names"],
    )


def test_v4_artifact_policy_is_designed_fail_safe() -> None:
    source = SCRIPT.read_text()
    assert '"activation_allowed": False' in source
    assert "execution_policy = v3.OpportunityPolicy(" in source
    assert "create_order" not in source


def test_seed_subset_search_requires_two_seeds_for_full_ensemble() -> None:
    source = SCRIPT.read_text()
    assert "for size in range(2, len(seeds) + 1)" in source
    assert "itertools.combinations(seeds, size)" in source
    assert "selection_score" in source
