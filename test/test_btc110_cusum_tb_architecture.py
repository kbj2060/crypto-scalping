from __future__ import annotations

import unittest
import torch

from scripts.train_eval_btc110_cusum_tb_causal_20260804 import EventNet


class BTC110CusumTBArchitectureTest(unittest.TestCase):
    def test_three_class_output_shape(self):
        self.assertEqual(EventNet()(torch.randn(4, 110)).shape, (4, 3))
