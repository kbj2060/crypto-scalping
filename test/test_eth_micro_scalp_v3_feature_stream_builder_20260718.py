import importlib.util
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_eth_micro_scalp_v3_feature_stream_20260718.py"
SPEC = importlib.util.spec_from_file_location("eth_micro_scalp_v3_feature_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_public_endpoint_allowlist_contains_no_trade_path() -> None:
    assert MODULE.PUBLIC_ENDPOINTS
    assert all("order" not in endpoint.lower() for endpoint in MODULE.PUBLIC_ENDPOINTS.values())
    assert all(endpoint.startswith(("/fapi/v1/", "/futures/data/")) for endpoint in MODULE.PUBLIC_ENDPOINTS.values())


def test_metric_mapping_matches_training_column_semantics() -> None:
    assert (
        "top_position", "sum_toptrader_long_short_ratio", "longShortRatio"
    ) in (
        ("top_position", "sum_toptrader_long_short_ratio", "longShortRatio"),
        ("top_account", "count_toptrader_long_short_ratio", "longShortRatio"),
    )


def test_builder_source_has_no_account_or_order_client() -> None:
    source = inspect.getsource(MODULE)
    for forbidden in ("API_KEY", "SECRET", "create_order", "cancel_order", "trading_bot"):
        assert forbidden not in source


def test_lookback_is_bounded_by_public_metric_retention(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 7 and 29"):
        MODULE.build(tmp_path / "x.csv", tmp_path / "x.json", lookback_days=30, end=pd.Timestamp("2026-07-18"))


def test_causal_context_covers_prior_funding_interval() -> None:
    assert MODULE.CAUSAL_CONTEXT >= pd.Timedelta(hours=8)
    assert pd.Timedelta(days=29) + MODULE.CAUSAL_CONTEXT < pd.Timedelta(days=30)


def test_parity_timestamp_conversion_uses_nanoseconds() -> None:
    timestamp = pd.Series(pd.to_datetime(["2026-07-12 09:00:00"]).astype("datetime64[us]"))
    expected_ns = timestamp.astype("datetime64[ns]").astype("int64").iloc[0]
    assert expected_ns == 1_783_846_800_000_000_000


def test_canonical_parity_interval_precedes_stale_cache_tail() -> None:
    assert MODULE.CANONICAL_PARITY_END_UTC < MODULE.observer.FRESH_START_UTC
