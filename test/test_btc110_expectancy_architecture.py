from __future__ import annotations

import unittest
import torch

from scripts.train_eval_btc110_expectancy_causal_20260804 import ExpectancyNet


class BTC110ExpectancyArchitectureTest(unittest.TestCase):
    def test_dual_head_shape(self):
        long, short = ExpectancyNet()(torch.randn(5,110))
        self.assertEqual(long.shape,(5,))
        self.assertEqual(short.shape,(5,))
