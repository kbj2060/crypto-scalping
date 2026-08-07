from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import build_eth_split_oracle_strategy_labels_20260724 as split_oracle


def test_split_contract_is_chronological_and_non_overlapping() -> None:
    ordered = list(split_oracle.SPLITS.values())
    assert ordered[0][0] == pd.Timestamp("2024-01-01")
    assert ordered[0][1] == ordered[1][0]
    assert ordered[1][1] == ordered[2][0]
    assert ordered[2][1] == pd.Timestamp("2026-07-21")


def test_trajectory_masks_invalid_rows_and_preserves_nonoverlap() -> None:
    labels = pd.DataFrame(
        {
            "decision_index": np.arange(8),
            "decision_timestamp": pd.date_range("2026-01-01", periods=8, freq="5min"),
            "oracle_dp_selected": [0, 1, 0, 0, 1, 0, 0, 0],
            "oracle_side": [0, 1, 0, 0, -1, 0, 0, 0],
            "oracle_event_end_index": [-1, 4, -1, -1, 7, -1, -1, -1],
            "label_evaluable": [1, 1, 1, 1, 1, 1, 0, 0],
            "label_invalid_reason": ["", "", "", "", "", "", "right", "right"],
        }
    )
    result = split_oracle.build_trajectory(labels, split="validation")
    assert result["zigzag_action"].tolist() == [0, 1, 1, 1, 2, 2, 0, 0]
    assert result["oracle_label_valid"].tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    assert set(result["oracle_split"]) == {"validation"}


def test_trajectory_rejects_crossing_or_overlapping_interval() -> None:
    labels = pd.DataFrame(
        {
            "decision_index": np.arange(5),
            "decision_timestamp": pd.date_range("2026-01-01", periods=5, freq="5min"),
            "oracle_dp_selected": [1, 0, 1, 0, 0],
            "oracle_side": [1, 0, -1, 0, 0],
            "oracle_event_end_index": [4, -1, 5, -1, -1],
            "label_evaluable": np.ones(5, dtype=np.int8),
            "label_invalid_reason": [""] * 5,
        }
    )
    try:
        split_oracle.build_trajectory(labels, split="train")
    except RuntimeError as exc:
        assert "overlapping" in str(exc)
    else:
        raise AssertionError("overlapping oracle intervals must fail")
