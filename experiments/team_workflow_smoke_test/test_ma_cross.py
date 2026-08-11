"""Unit tests for ma_cross.detect_ma_cross (stdlib unittest, no third-party deps)."""

import unittest

from .ma_cross import detect_ma_cross


class TestDetectMaCross(unittest.TestCase):
    def test_golden_cross(self):
        fast = [1.0, 2.0]
        slow = [2.0, 1.5]
        self.assertEqual(detect_ma_cross(fast, slow), "golden_cross")

    def test_death_cross(self):
        fast = [2.0, 1.0]
        slow = [1.0, 1.5]
        self.assertEqual(detect_ma_cross(fast, slow), "death_cross")

    def test_none_no_cross_fast_stays_above(self):
        fast = [3.0, 3.5]
        slow = [1.0, 1.2]
        self.assertEqual(detect_ma_cross(fast, slow), "none")

    def test_none_no_cross_fast_stays_below(self):
        fast = [1.0, 1.2]
        slow = [3.0, 3.5]
        self.assertEqual(detect_ma_cross(fast, slow), "none")

    def test_golden_cross_from_equal(self):
        fast = [1.0, 2.0]
        slow = [1.0, 1.5]
        self.assertEqual(detect_ma_cross(fast, slow), "golden_cross")

    def test_death_cross_from_equal(self):
        fast = [1.0, 0.5]
        slow = [1.0, 1.5]
        self.assertEqual(detect_ma_cross(fast, slow), "death_cross")

    def test_longer_sequence_uses_last_two_bars(self):
        fast = [5.0, 4.0, 3.0, 1.0, 2.0]
        slow = [1.0, 1.0, 1.0, 2.0, 1.5]
        self.assertEqual(detect_ma_cross(fast, slow), "golden_cross")

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            detect_ma_cross([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            detect_ma_cross([1.0], [1.0])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            detect_ma_cross([], [])


if __name__ == "__main__":
    unittest.main()
