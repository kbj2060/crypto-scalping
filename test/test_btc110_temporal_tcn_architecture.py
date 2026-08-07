from __future__ import annotations

import unittest

import torch

from scripts.train_eval_btc110_temporal_tcn_causal_20260805 import CausalTCNBlock, TemporalTCNNet


class BTC110TemporalTCNArchitectureTest(unittest.TestCase):
    def test_three_class_output_shape(self):
        self.assertEqual(TemporalTCNNet()(torch.randn(4, 24, 110)).shape, (4, 3))

    def test_causal_block_does_not_change_past_outputs_when_future_input_changes(self):
        model = CausalTCNBlock(48, dilation=4).eval()
        one = torch.randn(1, 48, 24)
        two = one.clone()
        two[:, :, -1] += 1.0
        with torch.no_grad():
            self.assertTrue(torch.equal(model(one)[:, :, :-1], model(two)[:, :, :-1]))
