from __future__ import annotations

import unittest

import pandas as pd

from features.session_contract_v2 import SESSION_CONTRACT_V2_COLUMNS, build_session_contract_v2


class SessionContractV2Tests(unittest.TestCase):
    def test_dst_aware_us_open(self) -> None:
        timestamps = pd.DatetimeIndex(
            ["2026-01-05 14:30:00+00:00", "2026-06-01 13:30:00+00:00"]
        )
        result = build_session_contract_v2(timestamps)

        self.assertEqual(result["is_us_cash_session"].tolist(), [1, 1])
        self.assertEqual(result["is_us_open_30m"].tolist(), [1, 1])
        self.assertEqual(result["minutes_from_us_open"].tolist(), [0.0, 0.0])

    def test_holiday_and_weekend_are_distinct(self) -> None:
        timestamps = pd.DatetimeIndex(
            ["2026-07-03 15:00:00+00:00", "2026-07-04 15:00:00+00:00"]
        )
        result = build_session_contract_v2(timestamps)

        self.assertEqual(result["is_us_market_holiday"].tolist(), [1, 0])
        self.assertEqual(result["is_weekend"].tolist(), [0, 1])
        self.assertEqual(result["is_us_cash_session"].tolist(), [0, 0])

    def test_naive_timestamp_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            build_session_contract_v2(pd.DatetimeIndex(["2026-01-05 14:30:00"]))

    def test_contract_has_no_legacy_alias(self) -> None:
        self.assertNotIn("session_us", SESSION_CONTRACT_V2_COLUMNS)


if __name__ == "__main__":
    unittest.main()
