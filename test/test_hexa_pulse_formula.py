from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot_modules.hexa_pulse_formula import (
    HexaPulseConfig,
    HexaPulseState,
    compute_formula_values,
    prepare_live_formula_frame,
    reconstruct_whale_position_score,
    step_formula,
)


def _frame(rows: int = 180) -> pd.DataFrame:
    idx = pd.date_range("2026-07-01", periods=rows, freq="min")
    wave = np.sin(np.arange(rows) / 9.0)
    return pd.DataFrame(
        {
            "nif_whale": 0.4 * wave,
            "obi": 0.5 * np.cos(np.arange(rows) / 11.0),
            "whale_position_score": 0.7 * wave,
            "eai": 1.0 + 0.5 * np.sin(np.arange(rows) / 13.0),
            "shadow_toxicity_score": 0.2,
            "shadow_aftershock_prob": 0.0,
            "data_stale": False,
            "valid_nif": True,
            "warmup_30m_ready": True,
            "valid_liq_stream": True,
            "micro_schema_version": 3,
            "tail_schema_version": 3,
        },
        index=idx,
    )


def test_whale_position_reconstruction_matches_live_numeric_formula() -> None:
    nif = pd.Series([0.20, -0.20, 0.05])
    oi = pd.Series([0.001, 0.001, -0.001])
    score = reconstruct_whale_position_score(nif, oi)
    assert score.iloc[0] == pytest.approx(1.0)
    assert score.iloc[1] == pytest.approx(-1.0)
    assert score.iloc[2] == pytest.approx(0.245)


def test_formula_is_causal_and_bounded() -> None:
    frame = _frame()
    original = compute_formula_values(frame)
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("nif_whale")] = 100.0
    revised = compute_formula_values(changed)
    pd.testing.assert_series_equal(original["score"].iloc[:-1], revised["score"].iloc[:-1])
    assert revised["score"].dropna().between(-1.0, 1.0).all()


def test_live_contract_requires_v3_streams() -> None:
    frame = _frame()
    assert prepare_live_formula_frame(frame)["available"].iloc[-1]
    frame["tail_schema_version"] = 2
    assert not prepare_live_formula_frame(frame)["available"].iloc[-1]


def test_threshold_hysteresis_enters_holds_and_exits_without_time_limit() -> None:
    state = HexaPulseState()
    cfg = HexaPulseConfig()
    first = step_formula(state, score=0.70, toxicity=0.1, tail_risk=0.0, available=True, config=cfg)
    second = step_formula(state, score=0.70, toxicity=0.1, tail_risk=0.0, available=True, config=cfg)
    held = step_formula(state, score=0.20, toxicity=0.1, tail_risk=0.0, available=True, config=cfg)
    exited = step_formula(state, score=0.14, toxicity=0.1, tail_risk=0.0, available=True, config=cfg)
    assert first.action == "CASH"
    assert second.action == "ENTER_LONG"
    assert held.action == "HOLD_LONG"
    assert exited.action == "EXIT"


def test_risk_threshold_forces_exit() -> None:
    state = HexaPulseState(position=-1)
    decision = step_formula(
        state,
        score=-0.9,
        toxicity=0.81,
        tail_risk=0.0,
        available=True,
    )
    assert decision.position == 0
    assert decision.reason == "TOXICITY_EXIT"
