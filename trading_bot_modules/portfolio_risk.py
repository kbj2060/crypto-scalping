"""Portfolio-level notional risk coordination across multiple real-money execution adapters.

Motivation: the ETH/SOL/BTC concurrent-portfolio research line
(scripts/replay_portfolio_concurrent_3asset_native_20260712.py and
docs/model_contracts/portfolio_concurrent_3asset_*_20260712.md) found:

- ETH/SOL/BTC positions are concurrently open 65-70% of the time at these strategies' typical
  holding period -- treating them as independent, uncoordinated sleeves against one shared
  account balance is not safe once more than one asset trades for real.
- A *shared*, first-come-first-served notional budget starves whichever asset is checked last
  (see the v3 SOL-starvation finding in that doc line) -- fixing this requires giving each asset
  its own non-competing budget share, not a pool assets compete for.
- A hard-reject cap on top of that shared pool is *worse* than no cap in some windows (v2 finding:
  rejecting a candidate doesn't defer it, it substitutes a different, sometimes worse, trade) --
  so this module shrinks the requested exposure to fit the budget rather than blocking the trade
  outright, mirroring the validated "scale"/"prealloc" cap_mode design, not "reject".

This module is intentionally standalone and has no dependency on trading_bot.py's live execution
path -- it does not call `execute_to_target` or place any order itself. Wire it in at the call
site (compute the per-asset target_exposure, call `scale_to_budget` before passing the result to
`execute_to_target`).

Units: `target_exposure` here uses the SAME convention as `execute_to_target`'s `target_exposure`
argument -- notional as a fraction/multiple of account equity (matching the backtest's
`notional = margin_fraction * leverage` convention), not raw USDT. Caps are expressed the same way
(e.g. `total_notional_cap=3.0` means "up to 3x account equity in combined notional exposure"),
directly comparable to the values swept in
docs/model_contracts/portfolio_concurrent_3asset_v4_sweep_duration_gate_20260712.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioRiskConfig:
    """`asset_shares` need not sum to 1.0 -- they are normalized on construction. Set
    `total_notional_cap` to `None` for no cap at all (per-asset shares are then ignored)."""

    total_notional_cap: float | None
    asset_shares: dict[str, float] = field(default_factory=dict)
    min_notional: float = 0.05


@dataclass
class PortfolioRiskDecision:
    allowed: bool
    reason: str
    asset_budget: float | None
    requested_exposure: float
    approved_exposure: float


class PortfolioRiskManager:
    """Coordinates a per-asset notional budget across multiple assets trading concurrently
    against one shared account balance.

    Each asset gets a FIXED, non-competing budget = `total_notional_cap * asset_shares[asset]`,
    evaluated independently of what any other asset is currently doing -- no cross-asset lookup,
    so one asset's activity can never crowd out another's budget (this is what fixes the v3
    SOL-starvation pathology). Only this asset's own currently-open exposure is relevant to its
    own check, and even that isn't needed here since `target_exposure` from the governor already
    represents the desired TOTAL exposure for that position, not an incremental add-on.

    Usage (not yet wired into trading_bot.py -- call this immediately before `execute_to_target`
    for each asset):

        risk = PortfolioRiskManager(PortfolioRiskConfig(
            total_notional_cap=3.0,
            asset_shares={"eth": 0.5, "btc": 0.3, "sol": 0.2},
        ))
        approved_exposure = risk.scale_to_budget("sol", target_exposure)
        if approved_exposure < risk.config.min_notional:
            return  # skip this cycle, do not call execute_to_target
        await live_executor.execute_to_target(..., target_exposure=approved_exposure, ...)
    """

    def __init__(self, config: PortfolioRiskConfig):
        self.config = config
        if config.total_notional_cap is not None and config.asset_shares:
            share_sum = sum(config.asset_shares.values())
            if share_sum <= 0:
                raise ValueError("asset_shares must sum to a positive value when total_notional_cap is set")
            self._shares = {k: v / share_sum for k, v in config.asset_shares.items()}
        else:
            self._shares = dict(config.asset_shares)

    def asset_budget(self, asset: str) -> float | None:
        """Returns None if uncapped (no total_notional_cap set)."""
        if self.config.total_notional_cap is None:
            return None
        return self.config.total_notional_cap * self._shares.get(asset, 0.0)

    def check(self, asset: str, requested_exposure: float) -> PortfolioRiskDecision:
        """Reports whether `requested_exposure` fits within `asset`'s own budget, without
        modifying it. Use `scale_to_budget` if you want the shrunk value directly."""
        budget = self.asset_budget(asset)
        if budget is None:
            return PortfolioRiskDecision(True, "uncapped", None, requested_exposure, requested_exposure)
        if requested_exposure <= budget + 1e-9:
            return PortfolioRiskDecision(True, "within_asset_budget", budget, requested_exposure, requested_exposure)
        return PortfolioRiskDecision(
            False,
            f"requested_exposure={requested_exposure:.4f} exceeds {asset}'s own budget={budget:.4f}",
            budget,
            requested_exposure,
            budget,
        )

    def scale_to_budget(self, asset: str, requested_exposure: float) -> float:
        """Shrinks `requested_exposure` down to `asset`'s own budget if it would otherwise
        exceed it; never rejects outright (that's a deliberate choice -- see module docstring).
        Returns the value unchanged if uncapped or already within budget."""
        budget = self.asset_budget(asset)
        if budget is None:
            return requested_exposure
        return min(requested_exposure, max(0.0, budget))
