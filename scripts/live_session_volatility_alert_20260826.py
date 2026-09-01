#!/usr/bin/env python3
"""Session-open volatility risk alert for the Snapshot tab's evidence-signal chip row, 2026-08-26.

Grounded in same-day empirical research (NOT a trading rule -- purely a discretionary risk-
awareness display, same "dashboard exposure = informativeness, not economic viability" precedent
as feedback_dashboard_indicators_ic_bar_not_pnl_bar):
- research_eth_nyse_open_volatility_window_20260826.py: ETH realized volatility (range_pct) spikes
  to 2.2-2.3x baseline right at NYSE open (9:30 ET) and stays >=1.5x through roughly +85min.
- research_eth_multi_market_open_volatility_comparison_20260826.py: LSE (08:00 London) and JPX
  (09:00 JST) opens show only a marginal bump (peak 1.1-1.4x) -- nowhere near NYSE's effect.

User chose simple, memorable alert windows off those findings (not the literal statistically-
elevated boundary) for this display:
- JPX / LSE (Asia / Europe): [0, +30] minutes after open only (pre-open showed no elevation in
  either market, so there is nothing to warn about before the bell).
- NYSE (US): [-60, +60] minutes around open (covers both the open-print spike and the 8:30am ET
  US econ-data-release spike found at bucket -60).

Trading-day/holiday calendars and DST come from pandas_market_calendars (same library already used
by the repo's session-split research, eth_session_split_edge_2023utc_20260817 -- us=NYSE/
europe=LSE/asia=JPX), so weekends and exchange holidays are automatically excluded: on those days
there is no real "open" event, so no alert fires.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pandas_market_calendars as mcal

# (mcal calendar name, Korean label, (window_start_min, window_end_min) relative to that day's open)
ALERT_MARKETS = [
    ("JPX", "일본장", (0, 30)),
    ("LSE", "유럽장", (0, 30)),
    ("NYSE", "미국장", (-60, 60)),
]

_SCHEDULE_CACHE: dict = {"day": None, "opens": {}}


def _opens_around(day: pd.Timestamp) -> dict[str, list[pd.Timestamp]]:
    """Cached per UTC calendar day -- pandas_market_calendars is pure/offline (no network), but
    there is no reason to recompute the schedule on every ~60s evidence-signal refresh when it
    only ever changes once a day."""
    if _SCHEDULE_CACHE["day"] == day:
        return _SCHEDULE_CACHE["opens"]
    opens: dict[str, list[pd.Timestamp]] = {}
    start = (day - pd.Timedelta(days=2)).date()
    end = (day + pd.Timedelta(days=2)).date()
    for cal_name, _, _ in ALERT_MARKETS:
        cal = mcal.get_calendar(cal_name)
        sched = cal.schedule(start_date=start, end_date=end)
        opens[cal_name] = list(sched["market_open"])
    _SCHEDULE_CACHE["day"] = day
    _SCHEDULE_CACHE["opens"] = opens
    return opens


def compute_session_volatility_alert(now: datetime | None = None) -> dict:
    """Returns {"active": [{"code","label","minutes_from_open","window":[lo,hi]}, ...]} for every
    market currently inside its alert window (normally 0 or 1 entries; JPX/LSE windows never
    overlap NYSE's in practice, so >1 is theoretical). Never raises -- any calendar-lookup failure
    degrades to {"active": []} so a bug here can never take down the rest of the evidence-signal
    payload."""
    try:
        now_ts = pd.Timestamp(now.astimezone(timezone.utc)) if now is not None else pd.Timestamp.now(tz="UTC")
        opens = _opens_around(now_ts.normalize())
        active = []
        for cal_name, label, (win_lo, win_hi) in ALERT_MARKETS:
            candidates = opens.get(cal_name) or []
            if not candidates:
                continue
            nearest = min(candidates, key=lambda o: abs((now_ts - o).total_seconds()))
            minutes = (now_ts - nearest).total_seconds() / 60.0
            if win_lo <= minutes <= win_hi:
                active.append({
                    "code": cal_name,
                    "label": label,
                    "minutes_from_open": round(minutes, 1),
                    "window": [win_lo, win_hi],
                })
        return {"active": active}
    except Exception as exc:  # noqa: BLE001 -- display-only, must never break the evidence payload
        print(f"session-volatility-alert computation failed (alert will read as inactive this cycle): {exc}", flush=True)
        return {"active": []}


if __name__ == "__main__":
    import json
    print(json.dumps(compute_session_volatility_alert(), indent=2, default=str))
