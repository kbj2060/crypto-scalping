"""Deterministic microstructure control decisions for research and shadow use.

The controller does not create a trading direction.  A parent supplies the side;
microstructure decides whether the opportunity is active, how urgently to execute,
how much margin to allocate, and whether an existing position should exit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


CONTROLLER_ID = "microstructure_operating_controller_v1_20260719"


@dataclass(frozen=True)
class ControllerConfig:
    min_opportunity: float = 0.65
    max_entry_risk: float = 0.80
    min_entry_alignment: float = -0.75
    urgent_execution: float = 0.75
    urgent_exit_risk: float = 0.85
    passive_exit_opportunity: float = 0.25
    passive_exit_alignment: float = -0.50
    base_margin_fraction: float = 0.30
    min_margin_fraction: float = 0.075
    leverage: float = 3.0


@dataclass(frozen=True)
class EntryDecision:
    allow: bool
    execution_mode: str
    margin_fraction: float
    leverage: float
    notional: float
    reason: str


@dataclass(frozen=True)
class ExitDecision:
    action: str
    reason: str


def _unit(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be finite and inside [0, 1]: {value}")
    return number


def _alignment(value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < -1.0 or number > 1.0:
        raise ValueError(f"alignment must be finite and inside [-1, 1]: {value}")
    return number


def opportunity_score(
    *,
    trade_notional: float,
    whale_activity: float,
    energy: float,
    queue_collapse: float,
) -> float:
    """Combine causal percentile-like inputs into expected movement opportunity."""
    return (
        0.45 * _unit(trade_notional, "trade_notional")
        + 0.30 * _unit(whale_activity, "whale_activity")
        + 0.15 * _unit(energy, "energy")
        + 0.10 * _unit(queue_collapse, "queue_collapse")
    )


def fragility_score(
    *,
    model_risk: float,
    toxicity: float,
    queue_collapse: float,
    aftershock: float,
    spoofing: float,
) -> float:
    """Combine adverse-selection and market-integrity inputs into risk."""
    return (
        0.30 * _unit(model_risk, "model_risk")
        + 0.25 * _unit(toxicity, "toxicity")
        + 0.20 * _unit(queue_collapse, "queue_collapse")
        + 0.15 * _unit(aftershock, "aftershock")
        + 0.10 * _unit(spoofing, "spoofing")
    )


def margin_for_entry(
    opportunity: float,
    risk: float,
    config: ControllerConfig = ControllerConfig(),
) -> tuple[float, float]:
    """Return (margin_fraction, notional) under fixed-leverage futures sizing."""
    opportunity = _unit(opportunity, "opportunity")
    risk = _unit(risk, "risk")
    raw = (
        config.base_margin_fraction
        * (0.55 + 0.45 * opportunity)
        * (1.0 - 0.75 * risk)
    )
    margin = min(config.base_margin_fraction, max(config.min_margin_fraction, raw))
    return float(margin), float(margin * config.leverage)


def decide_entry(
    *,
    side: int,
    opportunity: float,
    risk: float,
    alignment: float,
    config: ControllerConfig = ControllerConfig(),
) -> EntryDecision:
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or 1: {side}")
    opportunity = _unit(opportunity, "opportunity")
    risk = _unit(risk, "risk")
    alignment = _alignment(alignment)
    if risk >= config.max_entry_risk:
        return EntryDecision(False, "BLOCK", 0.0, config.leverage, 0.0, "FRAGILITY_BLOCK")
    if opportunity < config.min_opportunity:
        return EntryDecision(False, "BLOCK", 0.0, config.leverage, 0.0, "INSUFFICIENT_MOVEMENT")
    if alignment < config.min_entry_alignment:
        return EntryDecision(False, "BLOCK", 0.0, config.leverage, 0.0, "HOSTILE_FLOW")
    urgency = 0.55 * opportunity + 0.25 * max(alignment, 0.0) + 0.20 * risk
    mode = "MARKETABLE_LIMIT" if urgency >= config.urgent_execution else "MAKER_JOIN"
    margin, notional = margin_for_entry(opportunity, risk, config)
    return EntryDecision(True, mode, margin, config.leverage, notional, "ENTRY_ALLOWED")


def decide_exit(
    *,
    opportunity: float,
    risk: float,
    alignment: float,
    config: ControllerConfig = ControllerConfig(),
) -> ExitDecision:
    opportunity = _unit(opportunity, "opportunity")
    risk = _unit(risk, "risk")
    alignment = _alignment(alignment)
    if risk >= config.urgent_exit_risk:
        return ExitDecision("URGENT_EXIT", "FRAGILITY_EXIT")
    if (
        opportunity <= config.passive_exit_opportunity
        and alignment <= config.passive_exit_alignment
    ):
        return ExitDecision("PASSIVE_EXIT", "OPPORTUNITY_EXHAUSTED")
    return ExitDecision("HOLD", "MICROSTRUCTURE_SUPPORTS_HOLD")
