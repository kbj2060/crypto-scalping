from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import btc_v2_research_core_20260716 as core


def test_causal_regime_boundaries_are_frozen_on_training_rows() -> None:
    frame = pd.DataFrame(
        {
            "rvol_24": np.r_[np.linspace(0.01, 0.02, 100), 99.0],
            "logret_24": np.r_[np.linspace(-0.03, 0.03, 100), 99.0],
        }
    )
    train = np.r_[np.ones(100, dtype=bool), False]
    regime, detail = core.causal_regime_ids(frame, train)
    assert np.isclose(detail["volatility_median"], 0.015)
    assert regime[-1] == 3


def test_wave_balanced_weights_give_each_run_equal_mass() -> None:
    action = np.array([1, 1, 1, 2, 2, 0, 1, 1], dtype=np.int8)
    weight = core.wave_balanced_weights(action)
    runs = (slice(0, 3), slice(3, 5), slice(5, 6), slice(6, 8))
    masses = [float(weight[run].sum()) for run in runs]
    assert np.allclose(masses, masses[0])
    assert np.isclose(weight.mean(), 1.0)


def test_confirmed_events_wait_for_causal_persistence() -> None:
    action = np.array([0, 1, 2, 2, 2, 1, 1, 0, 1, 1], dtype=np.int8)
    event = core.confirmed_events(action, confirmation_hours=2)
    assert np.flatnonzero(event).tolist() == [3, 6, 9]


def test_cadenced_events_add_reentry_without_overlapping_the_horizon() -> None:
    action = np.array([1] * 20, dtype=np.int8)
    event = core.cadenced_events(action, confirmation_hours=2, reentry_hours=6)
    assert np.flatnonzero(event).tolist() == [1, 7, 13, 19]


def test_terminal_meta_target_uses_next_five_minute_bar_and_72_bar_exit() -> None:
    timestamps = pd.date_range("2025-01-01", periods=100, freq="5min")
    tape = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.full(100, 100.0),
            "high": np.full(100, 100.0),
            "low": np.full(100, 100.0),
            "close": np.full(100, 100.0),
            "atr_pct": np.full(100, 0.001),
        }
    )
    tape.loc[85, "close"] = 102.0
    hourly = pd.DataFrame({"timestamp": [timestamps[0]]})
    probability = np.array([[0.05, 0.90, 0.05]])
    target, net, eligible = core.terminal_meta_targets(hourly, probability, np.array([True]), tape)
    assert eligible.tolist() == [True]
    assert target.tolist() == [1]
    assert np.isclose(net[0], 0.02 - core.ROUND_TRIP_COST)


def test_replay_is_next_bar_stop_first_and_notional_is_not_double_levered() -> None:
    timestamps = pd.date_range("2025-01-01", periods=90, freq="5min")
    tape = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.full(90, 100.0),
            "high": np.full(90, 100.0),
            "low": np.full(90, 100.0),
            "close": np.full(90, 100.0),
            "atr_pct": np.full(90, 0.001),
        }
    )
    tape.loc[1, ["high", "low"]] = [101.0, 99.0]
    signal = np.zeros(90, dtype=np.int8)
    signal[0] = 1
    metrics, ledger, _ = core.replay(tape, signal, timestamps[0], timestamps[-1])
    assert len(ledger) == 1
    assert ledger.iloc[0]["entry_timestamp"] == timestamps[1]
    assert ledger.iloc[0]["exit_reason"] == "stop_loss"
    expected = core.NOTIONAL * (-0.005 - core.ROUND_TRIP_COST)
    assert np.isclose(ledger.iloc[0]["account_return"], expected)
    assert np.isclose(metrics["pnl_pct"], 100.0 * expected)


def test_execution_meta_target_matches_stop_first_replay_contract() -> None:
    timestamps = pd.date_range("2025-01-01", periods=90, freq="5min")
    tape = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.full(90, 100.0),
            "high": np.full(90, 100.0),
            "low": np.full(90, 100.0),
            "close": np.full(90, 100.0),
            "atr_pct": np.full(90, 0.001),
        }
    )
    tape.loc[13, ["high", "low"]] = [101.0, 99.0]
    hourly = pd.DataFrame({"timestamp": [timestamps[0]]})
    probability = np.array([[0.05, 0.90, 0.05]])
    target, net, eligible = core.execution_meta_targets(
        hourly, probability, np.array([True]), tape
    )
    assert eligible.tolist() == [True]
    assert target.tolist() == [0]
    assert np.isclose(net[0], -0.005 - core.ROUND_TRIP_COST)


def test_direction_oof_has_required_embargo_gap() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1400, 4))
    y = np.tile(np.array([0, 1, 2, 1, 2], dtype=np.int8), 280)
    train_mask = np.ones(len(x), dtype=bool)
    _, _, oof, folds = core.fit_direction_oof(
        x,
        y,
        train_mask,
        balance_waves=False,
        min_samples_leaf=10,
    )
    assert all(row["gap_rows"] == core.PURGE_HOURS for row in folds)
    assert np.isfinite(oof).all(axis=1).sum() > 0
