from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_btc_v1_label_family_suite_20260715 as suite


def test_fixed_horizon_labels_use_only_requested_forward_offset() -> None:
    close = np.array([100.0, 101.0, 99.0, 102.0, 98.0])
    label, eligible = suite.fixed_horizon_labels(close, horizon=2)
    assert label.tolist() == [2, 1, 2, 0, 0]
    assert eligible.tolist() == [True, True, True, False, False]


def test_directional_change_event_is_emitted_on_confirmation_bar() -> None:
    close = np.array([100.0, 100.4, 101.1, 100.8, 99.9, 100.0, 101.0])
    label, eligible = suite.directional_change_events(close, threshold=0.01)
    assert np.flatnonzero(eligible).tolist() == [2, 4, 6]
    assert label[eligible].tolist() == [1, 2, 1]


def test_dollar_events_depend_on_accumulated_activity() -> None:
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    activity = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    train_mask = np.ones(len(close), dtype=bool)
    label, eligible, threshold = suite.dollar_event_labels(close, activity, train_mask, horizon_events=1)
    assert threshold == 5.0
    assert np.flatnonzero(eligible).tolist() == [0, 1, 2, 3, 4]
    assert label[eligible].tolist() == [1, 1, 1, 1, 1]


def test_replay_uses_next_bar_and_futures_notional_contract() -> None:
    timestamps = pd.date_range("2026-01-01", periods=30, freq="h")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.full(30, 100.0),
            "close": np.full(30, 110.0),
        }
    )
    signal = np.zeros(30, dtype=np.int8)
    signal[0] = 1
    metrics, ledger, _ = suite.fresh_forward_replay(frame, signal, np.ones(30, dtype=bool))
    expected = suite.NOTIONAL * (0.10 - suite.ROUND_TRIP_COST)
    assert len(ledger) == 1
    assert ledger.iloc[0]["entry_timestamp"] == timestamps[1]
    assert np.isclose(ledger.iloc[0]["account_return"], expected)
    assert np.isclose(metrics["pnl"], expected)
