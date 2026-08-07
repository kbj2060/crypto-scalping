from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_hmm_confluence_meta_labels_20260724.py"
SPEC = importlib.util.spec_from_file_location("hmm_confluence_labels", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LABELS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LABELS
SPEC.loader.exec_module(LABELS)


def _frame(start: str, bars: int = 8) -> pd.DataFrame:
    timestamp = pd.date_range(start, periods=bars, freq="5min")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": np.full(bars, 100.0),
            "high": np.full(bars, 100.5),
            "low": np.full(bars, 99.5),
            "close": np.full(bars, 100.0),
            "volume": np.ones(bars),
        }
    )


def _candidate(timestamp: pd.Timestamp, *, horizon: int = 3) -> pd.Series:
    return pd.Series(
        {
            "decision_index": 0,
            "decision_timestamp": timestamp,
            "candidate_side": 1,
            "horizon_bars": horizon,
            "planned_target_price": 102.0,
            "planned_stop_price": 98.0,
        }
    )


def _empty_funding():
    return LABELS.FundingTape(np.array([], dtype=np.int64), np.array([0.0]))


def test_next_open_entry_and_tp_first_label() -> None:
    frame = _frame("2026-01-02")
    frame.loc[1, ["open", "high", "low", "close"]] = [101.0, 102.5, 100.5, 102.0]
    result = LABELS.simulate_candidate(frame, _candidate(frame.loc[0, "timestamp"]), _empty_funding())

    assert result["entry_index"] == 1
    assert result["entry_fill_price"] == 101.0 * (1.0 + LABELS.SLIPPAGE_RATE)
    assert result["label_outcome"] == "TP"
    assert result["label_valid"] == 1


def test_same_bar_target_and_stop_is_invalid_not_silently_ordered() -> None:
    frame = _frame("2026-01-02")
    frame.loc[1, ["high", "low"]] = [103.0, 97.0]
    result = LABELS.simulate_candidate(frame, _candidate(frame.loc[0, "timestamp"]), _empty_funding())

    assert result["label_outcome"] == "AMBIGUOUS"
    assert result["label_valid"] == 0
    assert result["label_invalid_reason"] == "same_bar_tp_sl"


def test_label_is_censored_at_split_boundary() -> None:
    frame = _frame("2026-03-31 23:45:00", bars=10)
    result = LABELS.simulate_candidate(
        frame,
        _candidate(frame.loc[0, "timestamp"], horizon=4),
        _empty_funding(),
    )

    assert result == {
        "split": "oos",
        "label_valid": 0,
        "label_invalid_reason": "split_boundary_censored",
    }


def test_vpvr_uses_only_completed_history_and_is_prefix_invariant() -> None:
    bars = 40
    base = pd.DataFrame(
        {
            "high": np.linspace(101.0, 110.0, bars),
            "low": np.linspace(99.0, 108.0, bars),
            "close": np.linspace(100.0, 109.0, bars),
            "volume": np.linspace(1.0, 3.0, bars),
        }
    )
    prefix = base.iloc[:31].copy()
    full = base.copy()
    full.loc[30, ["high", "low", "close", "volume"]] = [1000.0, 900.0, 950.0, 1e9]
    prefix_result = LABELS.compute_rolling_vpvr(prefix, window=12, n_bins=6)
    base_result = LABELS.compute_rolling_vpvr(base, window=12, n_bins=6)
    changed_result = LABELS.compute_rolling_vpvr(full, window=12, n_bins=6)

    for prefix_values, base_values in zip(prefix_result[:3], base_result[:3]):
        np.testing.assert_allclose(prefix_values, base_values[:31], equal_nan=True)
    for base_values, changed_values in zip(base_result[:3], changed_result[:3]):
        assert changed_values[30] == base_values[30]


def test_non_overlapping_replay_orders_years_by_timestamp_not_local_index() -> None:
    labels = pd.DataFrame(
        {
            "decision_index": [100, 1],
            "decision_timestamp": pd.to_datetime(["2025-12-01", "2026-01-01"]),
            "event_end_index": [102, 2],
            "event_end_timestamp": pd.to_datetime(["2025-12-01 00:10", "2026-01-01 00:05"]),
            "label_valid": [1, 1],
            "label_outcome": ["TP", "SL"],
            "label_net_return_per_notional": [0.01, -0.01],
            "context_sample_weight": [0.5, 0.5],
            "split": ["validation", "oos"],
        }
    )
    trades = LABELS.replay_non_overlapping(labels, cooldown_bars=0)

    assert trades["decision_timestamp"].tolist() == list(pd.to_datetime(["2025-12-01", "2026-01-01"]))
