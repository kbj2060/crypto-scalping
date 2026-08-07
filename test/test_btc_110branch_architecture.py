from __future__ import annotations

import unittest
import torch

from scripts.train_eval_btc_110branch_causal_20260804 import BTC110Branch


class BTC110BranchArchitectureTest(unittest.TestCase):
    def test_output_contract(self):
        direction, quantiles = BTC110Branch()(torch.randn(3, 110))
        self.assertEqual(direction.shape, (3,))
        self.assertEqual(quantiles.shape, (3, 7))
        self.assertTrue(torch.all(quantiles[:, 1:] >= quantiles[:, :-1]))
