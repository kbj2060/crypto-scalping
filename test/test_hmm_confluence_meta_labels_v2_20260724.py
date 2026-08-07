from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.build_hmm_confluence_meta_labels_20260724 as v1
import scripts.build_hmm_confluence_meta_labels_v2_20260724 as v2
import scripts.train_hmm_confluence_meta_filter_v2_20260724 as meta


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02", periods=8, freq="5min"),
            "open": [100.0] * 8,
            "high": [100.5] * 8,
            "low": [99.5] * 8,
            "close": [100.0] * 8,
            "regime3_transition_h6_risk_prob": [0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )


def _candidate() -> pd.Series:
    return pd.Series(
        {
            "decision_index": 0,
            "decision_timestamp": pd.Timestamp("2026-01-02"),
            "candidate_side": 1,
            "horizon_bars": 4,
            "planned_target_price": 103.0,
            "planned_stop_price": 98.0,
        }
    )


def _empty_funding() -> v1.FundingTape:
    return v1.FundingTape(np.array([], dtype=np.int64), np.array([0.0]))


def test_transition_exit_is_scheduled_for_next_bar_open() -> None:
    result = v2.simulate_candidate(
        _frame(),
        _candidate(),
        _empty_funding(),
        transition_exit_threshold=0.75,
    )

    assert result["label_outcome"] == "REGIME_EXIT"
    assert result["event_end_index"] == 2
    assert result["event_end_timestamp"] == pd.Timestamp("2026-01-02 00:10:00")


def test_transition_exit_can_be_disabled() -> None:
    result = v2.simulate_candidate(
        _frame(),
        _candidate(),
        _empty_funding(),
        transition_exit_threshold=None,
    )

    assert result["label_outcome"] == "TIMEOUT"


def test_multi_label_fields_are_finite_for_valid_label() -> None:
    result = v2.simulate_candidate(
        _frame(),
        _candidate(),
        _empty_funding(),
        transition_exit_threshold=0.75,
    )

    assert result["label_class"] in {"positive", "neutral", "negative"}
    assert np.isfinite(result["label_net_r"])
    assert np.isfinite(result["label_path_quality"])


def test_meta_features_are_side_aligned_and_drop_absolute_scale_columns() -> None:
    row = {column: 0.2 for column in meta.BASE_FEATURES}
    row.update(
        {
            "candidate_side": -1,
            "context_vwma288_slope12": -4.0,
            "context_atr192": 2.0,
            "context_volume_confirm": 0.3,
            "context_volume_imbalance": -0.4,
            "context_funding_z": 1.5,
        }
    )
    features = meta.feature_frame(pd.DataFrame([row]))

    assert features.loc[0, "slope_atr"] == -2.0
    assert features.loc[0, "aligned_volume"] == -0.3
    assert features.loc[0, "aligned_imbalance"] == 0.4
    assert features.loc[0, "aligned_funding"] == 1.5
    assert "context_atr192" not in features


def test_parameter_grid_contains_no_oos_dependent_field() -> None:
    forbidden = {"oos", "fresh", "oos_return", "fresh_return"}
    for params in v2.parameter_grid():
        assert forbidden.isdisjoint(vars(params))

