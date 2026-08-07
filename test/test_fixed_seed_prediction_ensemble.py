from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_fixed_seed_prediction_ensemble_20260729 import (
    ENSEMBLE_SCOPE,
    PROMOTION_BLOCKERS,
    ensemble_prediction_frames,
)


PREFIX = "omega1_regime3_expertdq_oof_"


def _frame(*, timestamp: str, long_prob: float, long_quality: float) -> pd.DataFrame:
    short_prob = 0.2
    cash_prob = 1.0 - long_prob - short_prob
    return pd.DataFrame(
        {
            "timestamp": [timestamp],
            PREFIX + "router_expert": ["bull"],
            PREFIX + "router_confidence": [0.8],
            PREFIX + "router_margin": [0.6],
            PREFIX + "dir_p_cash": [cash_prob],
            PREFIX + "dir_p_long": [long_prob],
            PREFIX + "dir_p_short": [short_prob],
            PREFIX + "dir_confidence": [max(cash_prob, long_prob, short_prob)],
            PREFIX + "dir_side_edge": [long_prob - short_prob],
            PREFIX + "dir_trade_prob": [long_prob + short_prob],
            PREFIX + "dir_action": [1],
            PREFIX + "quality_p_cash": [0.1],
            PREFIX + "quality_p_long": [long_quality],
            PREFIX + "quality_p_short": [0.9 - long_quality],
            PREFIX + "quality_for_action": [long_quality],
            PREFIX + "quality_threshold": [0.6],
            PREFIX + "final_action": [1],
        }
    )


class FixedSeedPredictionEnsembleTests(unittest.TestCase):
    def test_averages_probabilities_before_recomputing_action(self) -> None:
        out = ensemble_prediction_frames(
            {
                17: _frame(timestamp="2026-01-01", long_prob=0.7, long_quality=0.7),
                29: _frame(timestamp="2026-01-01", long_prob=0.5, long_quality=0.6),
            },
            quality_threshold=0.60,
        )

        self.assertAlmostEqual(out.loc[0, PREFIX + "dir_p_long"], 0.6)
        self.assertAlmostEqual(out.loc[0, PREFIX + "quality_p_long"], 0.65)
        self.assertEqual(out.loc[0, PREFIX + "dir_action"], 1)
        self.assertEqual(out.loc[0, PREFIX + "final_action"], 1)

    def test_scope_explicitly_blocks_live_promotion(self) -> None:
        self.assertEqual(
            ENSEMBLE_SCOPE,
            "entry_direction_and_quality_probabilities_only",
        )
        self.assertIn("exit_head_not_ensembled", PROMOTION_BLOCKERS)
        self.assertIn("live_parent_bundle_not_built", PROMOTION_BLOCKERS)

    def test_timestamp_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "timestamp"):
            ensemble_prediction_frames(
                {
                    17: _frame(timestamp="2026-01-01", long_prob=0.7, long_quality=0.7),
                    29: _frame(timestamp="2026-01-02", long_prob=0.5, long_quality=0.6),
                },
                quality_threshold=0.60,
            )

    def test_duplicate_or_single_seed_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            ensemble_prediction_frames(
                {17: _frame(timestamp="2026-01-01", long_prob=0.7, long_quality=0.7)},
                quality_threshold=0.60,
            )


if __name__ == "__main__":
    unittest.main()
