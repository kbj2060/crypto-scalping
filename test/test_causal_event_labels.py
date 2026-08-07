from __future__ import annotations

import unittest
import numpy as np

from core.causal_event_labels import causal_cusum_events, triple_barrier_direction


class CausalEventLabelTests(unittest.TestCase):
    def test_cusum_prefix_is_invariant_to_future_prices(self):
        close = np.array([100., 100.1, 100.4, 100.2, 100.3])
        vol = np.full(len(close), .001)
        changed = close.copy(); changed[-1] = 200.
        before_last = lambda values: [i for i in causal_cusum_events(values, vol, 2.) if i < len(values) - 1]
        self.assertEqual(before_last(close), before_last(changed))

    def test_dual_touch_is_flat(self):
        self.assertEqual(triple_barrier_direction(entry=100., high=[102.], low=[98.], close=[100.], move=.01), 0)
        self.assertEqual(triple_barrier_direction(entry=100., high=[101.1], low=[99.5], close=[100.], move=.01), 2)
        self.assertEqual(triple_barrier_direction(entry=100., high=[100.5], low=[98.9], close=[100.], move=.01), 1)
