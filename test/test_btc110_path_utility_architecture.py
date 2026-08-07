from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from scripts.train_eval_btc110_path_utility_causal_20260805 import PathUtilityNet, path_utility_labels


class BTC110PathUtilityArchitectureTest(unittest.TestCase):
    def test_three_class_output_shape(self):
        self.assertEqual(PathUtilityNet()(torch.randn(4, 110)).shape, (4, 3))

    def test_path_utility_prefers_long_or_short_without_first_touch_logic(self):
        long_frame = pd.DataFrame({"open": [100.0, 100.0, 100.0], "high": [100.0, 104.0, 103.0], "low": [100.0, 99.8, 100.0]})
        short_frame = pd.DataFrame({"open": [100.0, 100.0, 100.0], "high": [100.0, 100.2, 100.0], "low": [100.0, 96.0, 97.0]})
        self.assertEqual(path_utility_labels(long_frame, horizon_bars=2)[0][0], 2)
        self.assertEqual(path_utility_labels(short_frame, horizon_bars=2)[0][0], 1)
        self.assertTrue(np.all(path_utility_labels(long_frame, horizon_bars=2)[0][1:] == -1))
