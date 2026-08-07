from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.build_eth_full_oracle_strategy_labels_20260724 as oracle
import scripts.build_hmm_confluence_meta_labels_20260724 as base


def _empty_funding() -> base.FundingTape:
    return base.FundingTape(np.array([], dtype=np.int64), np.array([0.0]))


def test_same_bar_tp_sl_action_is_invalid() -> None:
    rows = 110
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="5min"),
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.1),
            "low": np.full(rows, 99.9),
            "close": np.full(rows, 100.0),
            "oracle_context_atr192": np.ones(rows),
        }
    )
    frame.loc[1, ["high", "low"]] = [101.0, 99.0]
    evaluation = oracle.evaluate_actions(frame, _empty_funding())
    action_id = evaluation.specs.index(oracle.ActionSpec(1, 0.5, 1.0, 12))

    assert evaluation.outcome[0, action_id] == 3
    assert np.isneginf(evaluation.returns[0, action_id])


def test_dynamic_program_can_prefer_two_shorter_non_overlapping_trades() -> None:
    spec = oracle.ActionSpec(1, 1.0, 1.0, 12)
    evaluation = oracle.ActionEvaluation(
        specs=[spec],
        returns=np.array([[0.15], [0.08], [0.10]], dtype=np.float32),
        next_index=np.array([[3], [2], [3]], dtype=np.int32),
        outcome=np.array([[2], [2], [2]], dtype=np.uint8),
        exit_at_open=np.ones((3, 1), dtype=np.uint8),
        local_best_action=np.zeros(3, dtype=np.int16),
        local_best_return=np.array([0.15, 0.08, 0.10]),
        local_second_return=np.zeros(3),
        evaluable_rows=3,
    )
    value, selected, selected_action = oracle.dynamic_program(evaluation, n_rows=4)

    assert selected.tolist() == [0, 1, 1, 0]
    assert selected_action.tolist() == [-1, 0, 0, -1]
    assert value[0] == value[1]
    assert value[0] > np.log1p(0.15)


def test_reconstructed_dp_path_never_overlaps() -> None:
    spec = oracle.ActionSpec(-1, 0.75, 1.5, 24)
    evaluation = oracle.ActionEvaluation(
        specs=[spec],
        returns=np.full((6, 1), 0.02, dtype=np.float32),
        next_index=np.array([[2], [3], [4], [5], [6], [6]], dtype=np.int32),
        outcome=np.full((6, 1), 2, dtype=np.uint8),
        exit_at_open=np.ones((6, 1), dtype=np.uint8),
        local_best_action=np.zeros(6, dtype=np.int16),
        local_best_return=np.full(6, 0.02),
        local_second_return=np.zeros(6),
        evaluable_rows=6,
    )
    _, selected, selected_action = oracle.dynamic_program(evaluation, n_rows=7)
    selected_rows = np.flatnonzero(selected)

    for current, following in zip(selected_rows, selected_rows[1:]):
        assert following >= evaluation.next_index[current, selected_action[current]]


def test_action_grid_is_fixed_and_contains_skip_separately() -> None:
    actions = oracle.action_grid()

    assert len(actions) == 2 * 4 * 4 * 4
    assert len(set((a.side, a.stop_atr, a.reward_r, a.horizon_bars) for a in actions)) == len(actions)
    assert all(a.side in {-1, 1} for a in actions)

