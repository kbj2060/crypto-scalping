from __future__ import annotations

import math
import unittest

from trading_bot_modules.omega4_6_1_runtime_contract import (
    EntryOverlayStatus,
    direction_overlay_status,
    finalize_sizing,
)


class Omega461SizingContractTests(unittest.TestCase):
    def test_final_cap_is_applied_after_notional_multiplier(self) -> None:
        decision = finalize_sizing(
            margin_fraction=0.30,
            requested_notional=0.60 * 4.5,
            max_leverage=5.0,
            max_notional=1.8,
        )

        self.assertAlmostEqual(decision.margin_fraction, 0.30)
        self.assertAlmostEqual(decision.leverage, 5.0)
        self.assertAlmostEqual(decision.notional, 1.5)
        self.assertAlmostEqual(
            decision.notional,
            decision.margin_fraction * decision.leverage,
        )

    def test_notional_cap_applies_when_leverage_has_room(self) -> None:
        decision = finalize_sizing(
            margin_fraction=0.50,
            requested_notional=2.70,
            max_leverage=5.0,
            max_notional=1.8,
        )

        self.assertAlmostEqual(decision.leverage, 3.6)
        self.assertAlmostEqual(decision.notional, 1.8)

    def test_invalid_sizing_fails_instead_of_being_silently_repaired(self) -> None:
        for value in (-0.1, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    finalize_sizing(
                        margin_fraction=value,
                        requested_notional=0.5,
                        max_leverage=5.0,
                        max_notional=1.8,
                    )


class Omega461OverlayContractTests(unittest.TestCase):
    def test_unavailable_overlay_blocks_new_entry(self) -> None:
        self.assertIs(
            direction_overlay_status(entry_side=1, predicted_direction=None),
            EntryOverlayStatus.UNAVAILABLE,
        )

    def test_opposite_direction_vetoes_entry(self) -> None:
        self.assertIs(
            direction_overlay_status(entry_side=-1, predicted_direction=1),
            EntryOverlayStatus.VETO,
        )

    def test_matching_or_neutral_direction_passes(self) -> None:
        self.assertIs(
            direction_overlay_status(entry_side=1, predicted_direction=1),
            EntryOverlayStatus.PASS,
        )
        self.assertIs(
            direction_overlay_status(entry_side=1, predicted_direction=0),
            EntryOverlayStatus.PASS,
        )


if __name__ == "__main__":
    unittest.main()
