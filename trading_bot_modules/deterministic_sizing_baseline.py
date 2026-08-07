"""Fixed-leverage sizing baseline for shadow comparison with learned sidecars."""

from __future__ import annotations

from dataclasses import dataclass
import math

from trading_bot_modules.omega4_6_1_runtime_contract import SizingDecision, finalize_sizing


@dataclass(frozen=True)
class DeterministicSizingConfig:
    base_margin_fraction: float
    fixed_leverage: float
    max_margin_fraction: float
    max_leverage: float
    max_notional: float

    def __post_init__(self) -> None:
        values = (
            self.base_margin_fraction,
            self.fixed_leverage,
            self.max_margin_fraction,
            self.max_leverage,
            self.max_notional,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("deterministic sizing values must be finite and positive")
        if self.base_margin_fraction > self.max_margin_fraction:
            raise ValueError("base_margin_fraction exceeds max_margin_fraction")
        if self.fixed_leverage > self.max_leverage:
            raise ValueError("fixed_leverage exceeds max_leverage")


def deterministic_sizing(
    config: DeterministicSizingConfig,
    *,
    margin_multiplier: float = 1.0,
) -> SizingDecision:
    """Return sizing with a reduction-only multiplier and the canonical final cap pass."""
    multiplier = float(margin_multiplier)
    if not math.isfinite(multiplier) or not 0.0 < multiplier <= 1.0:
        raise ValueError("margin_multiplier must be finite and in (0, 1]")
    margin_fraction = min(
        config.base_margin_fraction * multiplier,
        config.max_margin_fraction,
    )
    return finalize_sizing(
        margin_fraction=margin_fraction,
        requested_notional=margin_fraction * config.fixed_leverage,
        max_leverage=config.max_leverage,
        max_notional=config.max_notional,
    )
