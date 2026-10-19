"""FOMC 결정일 표(L6_FOMC_DECISION_DATES) -- live_macro_calendar 가 읽는다.

2026-10-19: Omega6 연구 어댑터(TabM·TCN·사이징·배리어)는 쓰는 곳이 없어 지웠다(git 이력 참조).
"""

from __future__ import annotations

# ---- L6 event-risk governor contract (ported from trading_bot_modules/omega5_live.py;
# pure calendar/feature function only, no ledger-derived selection state) ----
L6_FOMC_DECISION_DATES = {
    # 2026 source: trading_bot_modules/omega5_live.py OMEGA5_EVENT_RISK_FOMC_DECISION_DATES.
    # 2025 source: federalreserve.gov/monetarypolicy/fomccalendars.htm (fetched 2026-07-03),
    # eight 2025 meetings: Jan 28-29, Mar 18-19, May 6-7, Jun 17-18, Jul 29-30, Sep 16-17,
    # Oct 28-29, Dec 9-10 -- date used below is each meeting's second/announcement day
    # (2pm ET statement release), matching the 2026 list's day-of-decision convention.
    2025: ("2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"),
    2026: ("2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"),
}

