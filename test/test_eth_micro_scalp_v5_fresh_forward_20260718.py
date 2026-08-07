import importlib.util
import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _load(
    "eth_micro_scalp_v5_feature_builder",
    ROOT / "scripts/build_eth_micro_scalp_v5_feature_stream_20260718.py",
)
OBSERVER = _load(
    "eth_micro_scalp_v5_observer",
    ROOT / "scripts/run_eth_micro_scalp_v5_fresh_forward_observer_20260718.py",
)


def test_v5_builder_uses_exact_source_stable_contract() -> None:
    assert len(BUILDER.v5.v4.SOURCE_STABLE_FEATURES) == 36
    assert BUILDER.FRESH_START_UTC == OBSERVER.FRESH_START_UTC
    assert BUILDER.DEFAULT_OUTPUT == OBSERVER.DEFAULT_FEATURE_STREAM


def test_v5_observer_strictly_loads_all_three_fast_twitch_models() -> None:
    runtime = OBSERVER.observer.load_runtime(device_name="cpu")
    assert len(runtime.models) == 3
    assert runtime.checkpoint["selected_ensemble_seeds"] == [18, 29, 41]
    assert runtime.checkpoint["activation_allowed"] is False
    assert all(
        type(model) is BUILDER.v5.FastTwitchOpportunityMoE
        for model in runtime.models
    )


def test_v5_fresh_forward_sources_have_no_order_submission() -> None:
    for module in (BUILDER, OBSERVER):
        source = inspect.getsource(module)
        for forbidden in ("create_order", "cancel_order", "import ccxt", "trading_bot"):
            assert forbidden not in source
