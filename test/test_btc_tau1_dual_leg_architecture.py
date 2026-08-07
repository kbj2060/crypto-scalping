from __future__ import annotations

import unittest

import pandas as pd
import torch

from scripts.btc_tau1_dual_leg_architecture_20260805 import CausalBlock, LegANet, LegBNet, purged_splits


class Tau1DualLegArchitectureTest(unittest.TestCase):
    def test_leg_a_is_causal_and_has_three_class_head(self):
        torch.manual_seed(7)
        model = LegANet().eval()
        market, regime = torch.randn(2, 48, 111), torch.randn(2, 48, 24)
        baseline = model(market, regime)
        changed = market.clone(); changed[:, -1, :] += 3.0
        self.assertEqual(tuple(baseline.shape), (2, 3))
        self.assertFalse(torch.equal(baseline, model(changed, regime)))
        prefix = market[:, :-1, :]
        self.assertEqual(tuple(model(prefix, regime[:, :-1]).shape), (2, 3))

    def test_causal_block_prefix_is_unchanged_by_future_rows(self):
        torch.manual_seed(11)
        block = CausalBlock(4, 2).eval()
        prefix = torch.randn(1, 4, 12)
        extended = torch.cat([prefix, torch.randn(1, 4, 5)], dim=2)
        self.assertTrue(torch.equal(block(prefix), block(extended)[:, :, :12]))

    def test_leg_b_has_three_class_head(self):
        model = LegBNet().eval()
        self.assertEqual(tuple(model(torch.randn(2, 192, 111), torch.randn(2, 192, 24)).shape), (2, 3))

    def test_split_masks_do_not_cross_their_declared_time_boundaries(self):
        timestamps = pd.date_range("2025-08-01", "2026-04-10", freq="1D", tz="UTC")
        groups = purged_splits(timestamps, horizon_bars=1)
        self.assertFalse((timestamps[groups["train"]] >= pd.Timestamp("2025-09-01", tz="UTC")).any())
        self.assertFalse((timestamps[groups["checkpoint"]] >= pd.Timestamp("2025-11-01", tz="UTC")).any())
        self.assertFalse((timestamps[groups["calibration"]] >= pd.Timestamp("2026-01-01", tz="UTC")).any())
        self.assertFalse((timestamps[groups["oos"]] >= pd.Timestamp("2026-04-01", tz="UTC")).any())
