"""Canonical DST- and holiday-aware US session features for train and live paths."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


SESSION_CONTRACT_V2 = "session_contract_v2"
SESSION_CONTRACT_V2_COLUMNS = (
    "is_us_cash_session",
    "minutes_from_us_open",
    "is_us_open_30m",
    "is_weekend",
    "is_us_market_holiday",
)


def build_session_contract_v2(
    timestamps: pd.DatetimeIndex | Iterable[pd.Timestamp],
) -> pd.DataFrame:
    index = pd.DatetimeIndex(timestamps)
    if index.tz is None:
        raise ValueError("session_contract_v2 requires timezone-aware timestamps")
    if index.hasnans:
        raise ValueError("session_contract_v2 does not accept missing timestamps")
    if len(index) == 0:
        return pd.DataFrame(columns=SESSION_CONTRACT_V2_COLUMNS, index=index)

    utc = index.tz_convert("UTC")
    new_york = index.tz_convert("America/New_York")
    start = new_york.min().date() - pd.Timedelta(days=1)
    end = new_york.max().date() + pd.Timedelta(days=1)
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    schedule_by_date = {
        pd.Timestamp(session_date).date(): (
            pd.Timestamp(row["market_open"]).tz_convert("UTC"),
            pd.Timestamp(row["market_close"]).tz_convert("UTC"),
        )
        for session_date, row in schedule.iterrows()
    }

    rows: list[dict[str, float | int]] = []
    for ts_utc, ts_ny in zip(utc, new_york):
        session_date = ts_ny.date()
        weekend = session_date.weekday() >= 5
        session = schedule_by_date.get(session_date)
        if session is None:
            minutes_from_open = np.nan
            is_cash = 0
            is_open_30m = 0
        else:
            market_open, market_close = session
            minutes_from_open = float((ts_utc - market_open).total_seconds() / 60.0)
            is_cash = int(market_open <= ts_utc < market_close)
            is_open_30m = int(0.0 <= minutes_from_open < 30.0)
        rows.append(
            {
                "is_us_cash_session": is_cash,
                "minutes_from_us_open": minutes_from_open,
                "is_us_open_30m": is_open_30m,
                "is_weekend": int(weekend),
                "is_us_market_holiday": int(not weekend and session is None),
            }
        )
    return pd.DataFrame(rows, columns=SESSION_CONTRACT_V2_COLUMNS, index=index)
