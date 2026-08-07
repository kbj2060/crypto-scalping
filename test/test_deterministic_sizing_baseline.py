from __future__ import annotations

import unittest

from trading_bot_modules.deterministic_sizing_baseline import (
    DeterministicSizingConfig,
    deterministic_sizing,
)


class DeterministicSizingBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DeterministicSizingConfig(
            base_margin_fraction=0.30,
            fixed_leverage=3.0,
            max_margin_fraction=1.0,
            max_leverage=3.0,
            max_notional=0.75,
        )

    def test_notional_contract_survives_final_cap(self) -> None:
        result = deterministic_sizing(self.config)

        self.assertAlmostEqual(result.margin_fraction, 0.30)
        self.assertAlmostEqual(result.notional, 0.75)
        self.assertAlmostEqual(result.leverage, 2.5)
        self.assertAlmostEqual(result.notional, result.margin_fraction * result.leverage)

    def test_multiplier_reduces_margin_before_notional_derivation(self) -> None:
        result = deterministic_sizing(self.config, margin_multiplier=0.5)

        self.assertAlmostEqual(result.margin_fraction, 0.15)
        self.assertAlmostEqual(result.leverage, 3.0)
        self.assertAlmostEqual(result.notional, 0.45)

    def test_multiplier_cannot_increase_or_zero_risk(self) -> None:
        for multiplier in (0.0, 1.01, float("nan")):
            with self.subTest(multiplier=multiplier):
                with self.assertRaisesRegex(ValueError, "margin_multiplier"):
                    deterministic_sizing(self.config, margin_multiplier=multiplier)

    def test_invalid_base_contract_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "base_margin_fraction"):
            DeterministicSizingConfig(
                base_margin_fraction=1.1,
                fixed_leverage=3.0,
                max_margin_fraction=1.0,
                max_leverage=3.0,
                max_notional=3.0,
            )


if __name__ == "__main__":
    unittest.main()
