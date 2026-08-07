from __future__ import annotations

import pandas as pd

from scripts.build_eth_oracle_trajectory_labels_20260724 import build_trajectory


def test_build_trajectory_uses_half_open_non_overlapping_intervals() -> None:
    labels = pd.DataFrame(
        {
            "decision_index": range(7),
            "decision_timestamp": pd.date_range("2025-01-01", periods=7, freq="5min"),
            "oracle_dp_selected": [1, 0, 0, 1, 0, 1, 0],
            "oracle_side": [1, 0, 0, -1, 0, 1, 0],
            "oracle_event_end_index": [3, 0, 0, 5, 0, 7, 0],
        }
    )
    result = build_trajectory(labels)
    assert result["zigzag_action"].tolist() == [1, 1, 1, 2, 2, 1, 1]


def test_build_trajectory_rejects_overlapping_trades() -> None:
    labels = pd.DataFrame(
        {
            "decision_index": range(5),
            "decision_timestamp": pd.date_range("2025-01-01", periods=5, freq="5min"),
            "oracle_dp_selected": [1, 0, 1, 0, 0],
            "oracle_side": [1, 0, -1, 0, 0],
            "oracle_event_end_index": [4, 0, 5, 0, 0],
        }
    )
    try:
        build_trajectory(labels)
    except RuntimeError as exc:
        assert "overlapping oracle trades" in str(exc)
    else:
        raise AssertionError("expected overlapping oracle trades to fail")
