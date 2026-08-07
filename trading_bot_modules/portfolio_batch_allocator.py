"""Order-independent portfolio target allocation for research and shadow evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PortfolioTarget:
    asset: str
    side: int
    requested_notional: float

    def __post_init__(self) -> None:
        if not self.asset:
            raise ValueError("asset must be non-empty")
        if self.side not in (-1, 1):
            raise ValueError("side must be -1 or 1")
        if not math.isfinite(self.requested_notional) or self.requested_notional < 0.0:
            raise ValueError("requested_notional must be finite and non-negative")


@dataclass(frozen=True)
class PortfolioAllocation:
    asset: str
    side: int
    requested_notional: float
    approved_notional: float
    asset_cap_applied: bool
    direction_cap_applied: bool
    gross_cap_applied: bool


def allocate_portfolio_targets(
    targets: list[PortfolioTarget],
    *,
    asset_caps: dict[str, float],
    same_direction_cap: float,
    gross_cap: float,
) -> list[PortfolioAllocation]:
    """Allocate simultaneous targets without depending on their input order.

    The allocation first applies each asset's fixed cap, then proportionally scales
    each side to ``same_direction_cap``, and finally proportionally scales the whole
    batch to ``gross_cap``. Missing asset caps fail instead of silently receiving zero.
    """
    if not math.isfinite(same_direction_cap) or same_direction_cap <= 0.0:
        raise ValueError("same_direction_cap must be finite and positive")
    if not math.isfinite(gross_cap) or gross_cap <= 0.0:
        raise ValueError("gross_cap must be finite and positive")
    if len({target.asset for target in targets}) != len(targets):
        raise ValueError("targets must contain at most one row per asset")

    normalized_caps: dict[str, float] = {}
    for asset, cap in asset_caps.items():
        value = float(cap)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"asset cap for {asset!r} must be finite and non-negative")
        normalized_caps[asset] = value

    state: dict[str, dict[str, float | int | bool]] = {}
    for target in sorted(targets, key=lambda item: item.asset):
        if target.asset not in normalized_caps:
            raise ValueError(f"missing asset cap for {target.asset!r}")
        approved = min(target.requested_notional, normalized_caps[target.asset])
        state[target.asset] = {
            "side": target.side,
            "requested": target.requested_notional,
            "approved": approved,
            "asset_cap_applied": approved < target.requested_notional,
            "direction_cap_applied": False,
            "gross_cap_applied": False,
        }

    for side in (-1, 1):
        side_assets = [asset for asset, row in state.items() if row["side"] == side]
        side_total = sum(float(state[asset]["approved"]) for asset in side_assets)
        if side_total > same_direction_cap:
            scale = same_direction_cap / side_total
            for asset in side_assets:
                state[asset]["approved"] = float(state[asset]["approved"]) * scale
                state[asset]["direction_cap_applied"] = True

    gross_total = sum(float(row["approved"]) for row in state.values())
    if gross_total > gross_cap:
        scale = gross_cap / gross_total
        for row in state.values():
            row["approved"] = float(row["approved"]) * scale
            row["gross_cap_applied"] = True

    return [
        PortfolioAllocation(
            asset=asset,
            side=int(row["side"]),
            requested_notional=float(row["requested"]),
            approved_notional=float(row["approved"]),
            asset_cap_applied=bool(row["asset_cap_applied"]),
            direction_cap_applied=bool(row["direction_cap_applied"]),
            gross_cap_applied=bool(row["gross_cap_applied"]),
        )
        for asset, row in state.items()
    ]
