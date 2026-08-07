import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import native_alpha_v1 as native  # noqa: E402


def _bars(n=96):
    ts = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = np.linspace(100, 110, n)
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.2,
        "low": close - 0.2, "close": close, "volume": 1000.0,
        "last_funding_rate": 0.0001, "sum_open_interest_value": 100000.0,
    })


def test_asset_contract_is_fail_fast():
    with pytest.raises(ValueError):
        native._check_asset("eth")


def test_hourly_features_are_trailing_and_numeric():
    out = native.build_hourly_features(_bars(24 * 12))
    assert len(out) == 24
    assert out["timestamp"].is_monotonic_increasing
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in native.feature_columns(out))
    assert not out[native.feature_columns(out)].isna().any().any()


def test_labels_do_not_change_features():
    spec = native.ASSET_SPECS["sol"]
    features = native.build_hourly_features(_bars(24 * 12 * 4))
    labels = native.build_labels(features, spec)
    cols = native.feature_columns(features)
    pd.testing.assert_frame_equal(features[cols], labels[cols])


def test_regime_fit_does_not_use_rows_after_training_cutoff():
    features = native.build_hourly_features(_bars(24 * 12 * 8))
    labels = native.build_labels(features, native.ASSET_SPECS["sol"])
    train_idx = np.arange(50, 150)
    base, _ = native.attach_regime_states(labels, train_idx=train_idx)
    changed = labels.copy()
    changed.loc[120:, "ret_24"] = 100.0
    changed.loc[120:, "trend_strength"] = 0.0
    changed_out, _ = native.attach_regime_states(changed, train_idx=train_idx)
    np.testing.assert_allclose(
        base.loc[train_idx, ["regime_stability", "regime_chop_prob"]].to_numpy(),
        changed_out.loc[train_idx, ["regime_stability", "regime_chop_prob"]].to_numpy(),
    )


def test_loader_rejects_missing_contract_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "DATA_DIR", tmp_path)
    pd.DataFrame({"timestamp": ["2024-01-01"], "close": [1]}).to_csv(tmp_path / "sol_features_2024.csv", index=False)
    with pytest.raises(RuntimeError, match="missing required columns"):
        native.load_5m("sol", (2024,))
