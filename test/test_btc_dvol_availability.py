from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from scripts.build_btc_dvol_features_20260804 import load_dvol


class DvolAvailabilityTests(unittest.TestCase):
    @patch("scripts.build_btc_dvol_features_20260804.pd.read_csv")
    def test_hourly_close_is_available_one_hour_after_candle_timestamp(self, read_csv) -> None:
        read_csv.return_value = pd.DataFrame(
            {"timestamp": ["2026-01-01 00:00:00"], "close": [50.0]}
        )

        result = load_dvol("BTC")

        self.assertEqual(result.iloc[0]["timestamp"], pd.Timestamp("2026-01-01 01:00:00"))
        self.assertEqual(result.iloc[0]["dvol_btc"], 50.0)


if __name__ == "__main__":
    unittest.main()
