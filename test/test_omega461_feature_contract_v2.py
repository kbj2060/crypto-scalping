from __future__ import annotations

import math
import unittest

from trading_bot_modules.omega4_6_1_runtime_contract import strict_feature_values


class Omega461FeatureContractTests(unittest.TestCase):
    def test_returns_values_in_artifact_column_order(self) -> None:
        self.assertEqual(
            strict_feature_values(
                ["feature_b", "feature_a"],
                {"feature_a": 1.0, "feature_b": 2.0},
            ),
            [2.0, 1.0],
        )

    def test_missing_feature_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            strict_feature_values(["feature_a", "feature_b"], {"feature_a": 1.0})

    def test_duplicate_artifact_column_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            strict_feature_values(
                ["feature_a", "feature_a"],
                {"feature_a": 1.0},
            )

    def test_non_finite_feature_fails(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    strict_feature_values(["feature_a"], {"feature_a": value})


if __name__ == "__main__":
    unittest.main()
