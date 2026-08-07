"""P1-2 of docs/pipeline_integrity_and_research_redesign_20260730.md: sizing-chain parity check.

trading_bot.py's live path (as of 2026-07-29/30) computes final notional/leverage by calling
trading_bot_modules.omega4_6_1_runtime_contract.finalize_sizing() and
trading_bot_modules.portfolio_risk.PortfolioRiskManager.scale_to_budget() in sequence:
component notional -> asset NOTIONAL_MULTIPLIER -> scale_to_budget (portfolio share cap) ->
finalize_sizing (component leverage/notional cap, applied again after the portfolio scale-down).

tmp/research_20260728/three_asset_bar_level_mdd.py (research-only, predates the shared functions
existing) reimplements the same sequence by hand in `_apply_live_sizing_layers`: notional *=
multiplier; notional = min(notional, portfolio_cap); block if < min_notional -- but it does NOT
re-apply the component leverage/notional cap after the portfolio scale-down, unlike the live path.

This test checks whether that difference actually matters for the CURRENT live constants (where
portfolio caps are all tighter than the component cap, so re-applying the component cap after
portfolio scaling is a no-op) and, separately, constructs stress cases where the portfolio cap is
LOOSER than the component cap to show the two approaches diverge in general -- i.e. today's
agreement is a coincidence of current constants, not a structural guarantee.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_bot_modules.omega4_6_1_runtime_contract import finalize_sizing
from trading_bot_modules.portfolio_risk import PortfolioRiskConfig, PortfolioRiskManager

# Current live constants (.env, 2026-07-30): FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP=3.0,
# ETH/BTC/SOL_SHARE=0.5/0.3/0.2, ETH/SOL_NOTIONAL_MULTIPLIER=1.5. Component cap (OMEGA4_6_1_*):
# LEVERAGE_CAP=5.0, NOTIONAL_CAP=1.8 (research script's own hardcoded constants, confirmed against
# the live scale-map contract at the time of that script's writing).
LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8
MIN_NOTIONAL = 0.05
TOTAL_PORTFOLIO_CAP = 3.0
ASSET_SHARES = {"eth": 0.5, "btc": 0.3, "sol": 0.2}
PORTFOLIO_CAP = {k: TOTAL_PORTFOLIO_CAP * v / sum(ASSET_SHARES.values()) for k, v in ASSET_SHARES.items()}
NOTIONAL_MULTIPLIER = {"eth": 1.5, "sol": 1.5, "btc": 1.0}


def legacy_apply_live_sizing_layers(asset: str, margin: float, leverage: float) -> tuple[float, float]:
    """Verbatim port of three_asset_bar_level_mdd.py's `_apply_live_sizing_layers` scalar case."""
    notional = margin * leverage
    notional = notional * NOTIONAL_MULTIPLIER[asset]
    notional = min(notional, PORTFOLIO_CAP[asset])
    if notional < MIN_NOTIONAL:
        return margin, 0.0
    leverage_final = notional / max(margin, 1e-12) if margin > 0.0 else 0.0
    return margin, leverage_final


def real_apply_live_sizing_chain(
    asset: str, margin: float, leverage: float, *, total_notional_cap: float, asset_shares: dict
) -> tuple[float, float]:
    """The ACTUAL sequence trading_bot.py calls: multiplier -> scale_to_budget -> finalize_sizing."""
    notional = margin * leverage * NOTIONAL_MULTIPLIER[asset]
    risk = PortfolioRiskManager(PortfolioRiskConfig(total_notional_cap=total_notional_cap, asset_shares=asset_shares))
    approved = risk.scale_to_budget(asset, notional)
    if approved < risk.config.min_notional:
        return margin, 0.0
    sizing = finalize_sizing(
        margin_fraction=margin, requested_notional=approved, max_leverage=LEVERAGE_CAP, max_notional=NOTIONAL_CAP
    )
    return sizing.margin_fraction, sizing.leverage


class SizingChainParityTests(unittest.TestCase):
    def test_parity_under_current_live_constants(self) -> None:
        """Under TODAY's constants (portfolio caps 0.6/0.9/1.5 all < component cap 1.8), the
        legacy hand-rolled formula and the real shared functions must agree -- if this ever
        fails, the research replay's historical numbers no longer match what live actually does."""
        cases = [
            ("eth", 0.30, 2.0),
            ("eth", 0.50, 3.0),
            ("sol", 0.20, 2.5),
            ("sol", 0.45, 1.0),
            ("btc", 0.35, 2.0),
            ("btc", 0.10, 4.5),
        ]
        for asset, margin, leverage in cases:
            with self.subTest(asset=asset, margin=margin, leverage=leverage):
                legacy_margin, legacy_leverage = legacy_apply_live_sizing_layers(asset, margin, leverage)
                real_margin, real_leverage = real_apply_live_sizing_chain(
                    asset, margin, leverage, total_notional_cap=TOTAL_PORTFOLIO_CAP, asset_shares=ASSET_SHARES
                )
                self.assertAlmostEqual(legacy_margin, real_margin, places=9)
                self.assertAlmostEqual(legacy_leverage, real_leverage, places=9)

    def test_parity_breaks_when_portfolio_cap_exceeds_component_cap(self) -> None:
        """Stress case: loosen the portfolio cap above the component NOTIONAL_CAP (1.8). The
        legacy formula never re-applies the component cap after portfolio scaling, so it can
        return a notional > NOTIONAL_CAP; the real chain re-applies finalize_sizing and correctly
        clips it. This demonstrates today's agreement is a coincidence of current constants, not
        a structural guarantee -- confirming the P1-2 parity gap is real."""
        asset = "eth"
        margin, leverage = 0.60, 4.0  # raw component notional = 2.4, *1.5 multiplier = 3.6
        loose_shares = {"eth": 1.0, "btc": 0.0001, "sol": 0.0001}
        loose_total_cap = 10.0  # eth's portfolio budget ~= 10.0, far looser than NOTIONAL_CAP=1.8

        loose_legacy_notional = min(margin * leverage * NOTIONAL_MULTIPLIER[asset], loose_total_cap * 1.0)
        real_margin, real_leverage = real_apply_live_sizing_chain(
            asset, margin, leverage, total_notional_cap=loose_total_cap, asset_shares=loose_shares
        )
        real_notional = real_margin * real_leverage

        self.assertGreater(loose_legacy_notional, NOTIONAL_CAP, "test setup must actually exceed the component cap")
        self.assertLessEqual(
            real_notional, NOTIONAL_CAP + 1e-9,
            "real chain must clip back to the component NOTIONAL_CAP after portfolio scaling",
        )
        self.assertFalse(
            math.isclose(loose_legacy_notional, real_notional, abs_tol=1e-9),
            "legacy formula and real chain must diverge once the portfolio cap exceeds the "
            "component cap -- if they now agree, the legacy formula must have been fixed to "
            "re-apply the cap",
        )


if __name__ == "__main__":
    unittest.main()
