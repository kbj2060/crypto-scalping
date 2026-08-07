from __future__ import annotations

import unittest

import numpy as np

from scripts.build_btc_tau1_continuation_labels_20260805 import trend_scan


class BTCtau1ContinuationLabelsTest(unittest.TestCase):
    def test_trend_scan_is_causal(self):
        values = np.linspace(1.0, 2.0, 80)
        original = trend_scan(values)[0]
        changed = values.copy()
        changed[-1] = 100.0
        self.assertTrue(np.array_equal(original[:-1], trend_scan(changed)[0][:-1]))
