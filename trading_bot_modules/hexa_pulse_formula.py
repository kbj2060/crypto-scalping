"""HexaPulse-R v1: deterministic six-signal microstructure trading formula.

This module contains no fitted parameters or learned model.  It converts the six dashboard
signals into a signed score in [-1, 1] and applies entry/exit hysteresis.  A microstructure row
labelled T is shifted to decision time T+2 minutes, matching the live persistence contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


AVAIL_SHIFT_MIN = 2
FORMULA_ID = "hexa_pulse_r_v1_20260718"
REQUIRED_SIGNAL_COLUMNS = (
    "nif_whale",
    "obi",
    "whale_position_score",
    "eai",
    "shadow_toxicity_score",
    "shadow_aftershock_prob",
)


@dataclass(frozen=True)
class HexaPulseConfig:
    entry_threshold: float = 0.65
    exit_threshold: float = 0.15
    confirmation_bars: int = 2
    toxicity_exit_threshold: float = 0.80
    tail_risk_exit_threshold: float = 0.45


@dataclass
class HexaPulseState:
    position: int = 0
    long_streak: int = 0
    short_streak: int = 0


@dataclass(frozen=True)
class HexaPulseDecision:
    position: int
    action: str
    reason: str
    long_streak: int
    short_streak: int


def reconstruct_whale_position_score(
    nif_whale: pd.Series,
    oi_delta_pct: pd.Series,
    *,
    flow_threshold: float = 0.10,
    oi_threshold: float = 0.0001,
) -> pd.Series:
    """Reproduce the numeric live whale score from its two persisted inputs."""
    flow = pd.to_numeric(nif_whale, errors="coerce").astype(float)
    oi = pd.to_numeric(oi_delta_pct, errors="coerce").astype(float)
    flow_strength = (flow.abs() / max(float(flow_threshold), 1e-8)).clip(0.0, 1.5)
    oi_strength = (oi.abs() / max(float(oi_threshold), 1e-8)).clip(0.0, 1.5)
    sign_flow = pd.Series(np.where(flow >= 0.0, 1.0, -1.0), index=flow.index)
    oi_dir_weight = pd.Series(
        np.where(oi > oi_threshold, 1.0, np.where(oi < -oi_threshold, -0.35, 0.0)),
        index=oi.index,
    )
    score = (
        0.70 * sign_flow * flow_strength.clip(upper=1.0)
        + 0.30 * sign_flow * oi_dir_weight * oi_strength.clip(upper=1.0)
    )
    return score.clip(-1.0, 1.0).rename("whale_position_score")


def _rolling_robust_z(series: pd.Series, window: int = 60) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    past = values.shift(1)
    median = past.rolling(window, min_periods=window).median()
    mad = past.rolling(window, min_periods=window).apply(
        lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
    )
    z = (values - median) / (1.4826 * mad + 1e-8)
    return z.clip(-3.0, 3.0) / 3.0


def _rolling_percentile(series: pd.Series, window: int = 120) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.rolling(window, min_periods=window).apply(
        lambda x: float(np.mean(x[:-1] <= x[-1])) if len(x) > 1 else np.nan,
        raw=True,
    )


def compute_formula_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the immutable formula.  Callers own data-quality contract enforcement."""
    missing = [column for column in REQUIRED_SIGNAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"HexaPulse input contract missing columns: {missing}")

    out = pd.DataFrame(index=frame.index)
    out["flow_z"] = _rolling_robust_z(frame["nif_whale"])
    out["obi_z"] = _rolling_robust_z(frame["obi"])
    out["whale_position"] = pd.to_numeric(
        frame["whale_position_score"], errors="coerce"
    ).clip(-1.0, 1.0)
    out["energy_percentile"] = _rolling_percentile(frame["eai"])
    out["toxicity"] = pd.to_numeric(
        frame["shadow_toxicity_score"], errors="coerce"
    ).clip(0.0, 1.0)
    out["tail_risk"] = pd.to_numeric(
        frame["shadow_aftershock_prob"], errors="coerce"
    ).clip(0.0, 1.0)

    out["pressure"] = (
        0.45 * out["flow_z"]
        + 0.40 * out["obi_z"]
        + 0.15 * out["whale_position"]
    )
    magnitude = ((out["pressure"].abs() - 0.20) / 0.80).clip(0.0, 1.0)
    out["extreme_pressure"] = np.sign(out["pressure"]) * magnitude
    out["risk_multiplier"] = (1.0 - out["toxicity"]) ** 2 * (1.0 - out["tail_risk"]) ** 3
    out["raw_score"] = (
        -out["extreme_pressure"]
        * (0.50 + 0.50 * out["energy_percentile"])
        * out["risk_multiplier"]
    )
    out["smoothed_raw_score"] = out["raw_score"].ewm(span=3, adjust=False, min_periods=3).mean()
    # Fixed unit calibration: raw products naturally occupy a narrow interval even at signal
    # extremes.  tanh maps that interval to the documented [-1, 1] decision-score scale without
    # fitting labels, returns, or trade outcomes.
    out["score"] = np.tanh(5.0 * out["smoothed_raw_score"]).clip(-1.0, 1.0)
    return out


def prepare_live_formula_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the strict live contract and shift rows to their first causal decision minute."""
    required_contract = (
        "data_stale",
        "valid_nif",
        "warmup_30m_ready",
        "valid_liq_stream",
        "micro_schema_version",
        "tail_schema_version",
    )
    missing = [column for column in required_contract if column not in frame.columns]
    if missing:
        raise ValueError(f"HexaPulse live contract missing columns: {missing}")

    out = compute_formula_values(frame)
    finite = np.isfinite(frame[list(REQUIRED_SIGNAL_COLUMNS)].astype(float)).all(axis=1)
    out["available"] = (
        finite
        & ~frame["data_stale"].astype(bool)
        & frame["valid_nif"].astype(bool)
        & frame["warmup_30m_ready"].astype(bool)
        & frame["valid_liq_stream"].astype(bool)
        & (pd.to_numeric(frame["micro_schema_version"], errors="coerce") >= 3)
        & (pd.to_numeric(frame["tail_schema_version"], errors="coerce") >= 3)
    )
    out.index = out.index + pd.Timedelta(minutes=AVAIL_SHIFT_MIN)
    return out


def step_formula(
    state: HexaPulseState,
    *,
    score: float,
    toxicity: float,
    tail_risk: float,
    available: bool,
    config: HexaPulseConfig = HexaPulseConfig(),
) -> HexaPulseDecision:
    """Advance the threshold state machine by one bar; no fixed holding period is used."""
    if not available or not np.isfinite(score):
        action = "EXIT" if state.position else "CASH"
        reason = "DATA_INVALID_EXIT" if state.position else "DATA_INVALID"
        state.position = 0
        state.long_streak = 0
        state.short_streak = 0
        return HexaPulseDecision(0, action, reason, 0, 0)

    if state.position and (
        toxicity >= config.toxicity_exit_threshold
        or tail_risk >= config.tail_risk_exit_threshold
    ):
        reason = "TOXICITY_EXIT" if toxicity >= config.toxicity_exit_threshold else "TAIL_RISK_EXIT"
        state.position = 0
        state.long_streak = 0
        state.short_streak = 0
        return HexaPulseDecision(0, "EXIT", reason, 0, 0)

    if state.position > 0:
        if score < config.exit_threshold:
            state.position = 0
            return HexaPulseDecision(0, "EXIT", "LONG_SCORE_DECAY", 0, 0)
        return HexaPulseDecision(1, "HOLD_LONG", "LONG_SCORE_HOLD", 0, 0)

    if state.position < 0:
        if score > -config.exit_threshold:
            state.position = 0
            return HexaPulseDecision(0, "EXIT", "SHORT_SCORE_DECAY", 0, 0)
        return HexaPulseDecision(-1, "HOLD_SHORT", "SHORT_SCORE_HOLD", 0, 0)

    state.long_streak = state.long_streak + 1 if score >= config.entry_threshold else 0
    state.short_streak = state.short_streak + 1 if score <= -config.entry_threshold else 0
    if state.long_streak >= config.confirmation_bars:
        state.position = 1
        state.long_streak = 0
        state.short_streak = 0
        return HexaPulseDecision(1, "ENTER_LONG", "LONG_THRESHOLD_CONFIRMED", 0, 0)
    if state.short_streak >= config.confirmation_bars:
        state.position = -1
        state.long_streak = 0
        state.short_streak = 0
        return HexaPulseDecision(-1, "ENTER_SHORT", "SHORT_THRESHOLD_CONFIRMED", 0, 0)
    return HexaPulseDecision(
        0,
        "CASH",
        "ENTRY_THRESHOLD_NOT_CONFIRMED",
        state.long_streak,
        state.short_streak,
    )
