from __future__ import annotations

import unittest

from trading_bot_modules.portfolio_batch_allocator import (
    PortfolioTarget,
    allocate_portfolio_targets,
)


class PortfolioBatchAllocatorTests(unittest.TestCase):
    def test_allocation_is_order_independent(self) -> None:
        targets = [
            PortfolioTarget("sol", 1, 1.0),
            PortfolioTarget("eth", 1, 2.0),
            PortfolioTarget("btc", -1, 1.5),
        ]
        kwargs = {
            "asset_caps": {"eth": 1.5, "btc": 1.0, "sol": 0.5},
            "same_direction_cap": 1.6,
            "gross_cap": 2.0,
        }

        forward = allocate_portfolio_targets(targets, **kwargs)
        reverse = allocate_portfolio_targets(list(reversed(targets)), **kwargs)

        self.assertEqual(forward, reverse)
        self.assertEqual([row.asset for row in forward], ["btc", "eth", "sol"])

    def test_caps_are_enforced_proportionally(self) -> None:
        result = allocate_portfolio_targets(
            [
                PortfolioTarget("eth", 1, 2.0),
                PortfolioTarget("sol", 1, 1.0),
                PortfolioTarget("btc", -1, 1.5),
            ],
            asset_caps={"eth": 1.5, "btc": 1.0, "sol": 0.5},
            same_direction_cap=1.6,
            gross_cap=2.0,
        )
        by_asset = {row.asset: row for row in result}

        self.assertAlmostEqual(sum(row.approved_notional for row in result), 2.0)
        self.assertAlmostEqual(
            sum(row.approved_notional for row in result if row.side == 1),
            1.2307692307692308,
        )
        self.assertAlmostEqual(by_asset["eth"].approved_notional, 0.9230769230769231)
        self.assertAlmostEqual(by_asset["sol"].approved_notional, 0.3076923076923077)
        self.assertAlmostEqual(by_asset["btc"].approved_notional, 0.7692307692307693)
        self.assertTrue(by_asset["eth"].asset_cap_applied)
        self.assertTrue(by_asset["eth"].direction_cap_applied)
        self.assertTrue(by_asset["eth"].gross_cap_applied)

    def test_missing_asset_cap_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing asset cap"):
            allocate_portfolio_targets(
                [PortfolioTarget("sol", 1, 0.5)],
                asset_caps={"eth": 1.0},
                same_direction_cap=1.0,
                gross_cap=1.0,
            )

    def test_duplicate_asset_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "one row per asset"):
            allocate_portfolio_targets(
                [PortfolioTarget("eth", 1, 0.5), PortfolioTarget("eth", -1, 0.5)],
                asset_caps={"eth": 1.0},
                same_direction_cap=1.0,
                gross_cap=1.0,
            )


if __name__ == "__main__":
    unittest.main()
