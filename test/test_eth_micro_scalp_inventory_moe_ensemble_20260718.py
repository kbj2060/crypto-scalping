import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_eval_eth_micro_scalp_inventory_moe_ensemble_20260718.py"
SPEC = importlib.util.spec_from_file_location("eth_micro_scalp_inventory_moe_ensemble", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_seed_aggregation_averages_q_and_preserves_all_experts() -> None:
    first = {
        "q": np.ones((2, 3, 3), dtype=np.float32),
        "expert_q": np.ones((2, 3, 3, 3), dtype=np.float32),
        "gates": np.full((2, 3), 1 / 3, dtype=np.float32),
    }
    second = {
        "q": np.full((2, 3, 3), 3.0, dtype=np.float32),
        "expert_q": np.full((2, 3, 3, 3), 3.0, dtype=np.float32),
        "gates": np.full((2, 3), 1 / 3, dtype=np.float32),
    }
    result = MODULE.aggregate_seed_predictions([first, second])
    assert result["q"].shape == (2, 3, 3)
    assert result["expert_q"].shape == (2, 6, 3, 3)
    assert result["gates"].shape == (2, 6)
    assert np.allclose(result["q"], 2.0)


def test_seed_aggregation_requires_a_model() -> None:
    try:
        MODULE.aggregate_seed_predictions([])
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty ensemble must fail-fast")
