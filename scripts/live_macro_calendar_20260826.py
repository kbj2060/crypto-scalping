#!/usr/bin/env python3
"""US macro/corporate event calendar for the Snapshot tab, 2026-08-26 -- follow-up to the same-day
research_eth_nyse_open_volatility_window_20260826.py / research_eth_multi_market_open_volatility_
comparison_20260826.py finding (ETH volatility spikes hardest right at NYSE open, with a distinct
extra spike at -60min matching the 8:30am ET US data-release slot). User wants to see the actual
scheduled events (not just a blanket time window), matching a screenshot of a commercial calendar
widget (PCE/durable goods/GDP/FOMC/EIA/Treasury auctions/earnings/political schedule) -- this
module covers the sourceable subset: hard economic data (FRED), FOMC (already-verified static
calendar elsewhere in this repo), scheduled Federal Reserve Chair appearances (official monthly
HTML calendar), EIA petroleum (fixed weekly cadence, rule-based), and a small corporate-earnings
watchlist (Finnhub). Political schedule (e.g. presidential appearances) has no official government
API and is deliberately NOT covered here.

=== Sources, verified live 2026-08-26 ===
- FRED release/dates (fred/release/dates, NOT the plural fred/releases/dates -- that ignores
  release_id and returns every release mixed together, a real gotcha hit while building this),
  include_release_dates_with_no_data=true so future scheduled dates are returned, not just past
  ones with data already published. Release IDs confirmed via fred/releases listing:
    CPI=10, Employment Situation(NFP)=50, GDP=53, Personal Income and Outlays(PCE)=54,
    Advance Economic Indicators(durable goods)=435 -- NOT the M3 survey(95), which only carries
    the fuller monthly durable-goods report weeks later; 435 is the one that lines up with the
    "오늘" cluster the user's screenshot showed (PCE/GDP/durable-goods same day).
  The five hard-data releases conventionally release at 8:30am ET (matches the -60min spike's own
  mechanism).
- Michigan Surveys of Consumers: the official survey homepage exposes the next release note in
  HTML (including the preliminary/final label and 10:00am ET time). This is used instead of FRED
  release_id=95 because FRED's release-date feed can differ from the survey's published schedule.
- FOMC: trading_bot_modules/omega6_live.py::L6_FOMC_DECISION_DATES -- official Fed calendar,
  already verified/covers all of 2026, 2:00pm ET announcement.
- Federal Reserve Chair appearances: official monthly HTML pages at
  federalreserve.gov/newsevents/YYYY-month.htm. This is scraped for scheduled
  Speech/Testimony/Discussion rows whose speaker is labelled Chair/Chairman; the Fed RSS feed is
  a publication feed and does not provide a future schedule.
- EIA Weekly Petroleum Status Report: EIA's API has no "next scheduled release" field (it serves
  data, not a release calendar), so this uses the well-known fixed cadence instead -- every
  Wednesday 10:30am ET, shifted to Thursday when that week's Monday is a US market holiday
  (pandas_market_calendars NYSE calendar, same library already used in this repo's session-split
  research). This is a RULE, not a fetched date -- same category of caveat as this repo's existing
  ISM/PMI rule-based approximations (see trading_bot_modules/omega5_live.py), flagged as such in
  the returned event's "source" field.
- PMI (S&P Global Final Manufacturing/Services, S&P Global Flash Composite, ISM Manufacturing/
  Services -- 5 releases, PMI_RULES): confirmed 2026-09-01 that FRED carries none of these (searched
  fred/releases for pmi/purchasing/manufactur/ism/markit -- only regional Fed surveys come back,
  e.g. Empire State/Philly Fed, not the national S&P Global or ISM PMI), and S&P Global's own
  press-release pages return HTTP 403 to automated fetches with no public forward calendar found.
  Same rule-based category as EIA above -- all 5 follow day-of-month conventions (1st NYSE trading
  day of the month for Manufacturing: S&P Global Final 9:45am ET + ISM 10:00am ET, same day; 3rd
  trading day for Services, same two sources/times; first trading day on/after the 23rd for S&P
  Global's mid-month Flash Composite, 9:45am ET), holiday-adjusted via pandas_market_calendars (same
  NYSE calendar eia_events() already uses) -- an earlier version matched trading_bot_modules/
  omega5_live.py's own plain-weekday-counting event-risk-governor rule (no holiday shift) for
  consistency with that already-relied-upon convention, but omega5_live.py is no longer used
  elsewhere in this repo (2026-09-01, user-confirmed), so this was switched to the more correct
  holiday-aware form instead of preserving that limitation for its own sake.
- 9 additional FRED releases added 2026-09-01 (same fred/release/dates mechanism as the original 5,
  each confirmed live to return near-term future dates before adding): Producer Price Index(PPI)=46,
  Advance Retail Sales=9 (NOT the derived "Selected Real Retail Sales Series"=92), Industrial
  Production & Capacity Utilization(G.17)=13 (9:15am ET, the Fed's own G.17 convention), New
  Residential Construction(housing starts+permits)=27, JOLTS=192 (10:00am ET), Unemployment
  Insurance Weekly Claims=180 (WEEKLY, not monthly -- see fetch_fred_events()'s 2026-09-01 query-
  pattern fix below), Existing Home Sales=291 (10:00am ET), New Residential Sales=97 (10:00am ET),
  International Trade in Goods and Services=51. Conference Board Consumer Confidence was searched
  for and NOT found on FRED (same private-survey gap as PMI) -- not added this round for lack of a
  verified schedule source.
- Finnhub earnings calendar (finnhub.io/api/v1/calendar/earnings), filtered to EARNINGS_WATCHLIST
  (NVDA only by default -- the one example in the user's screenshot; extend the list to widen).
- Treasury upcoming-auctions (api.fiscaldata.treasury.gov/.../v1/accounting/od/upcoming_auctions,
  no key needed) -- filtered to security_type in (Note, Bond) only. Confirmed live: this endpoint
  shows a 5-Year Note auction on 2026-08-26, exact match to the user's screenshot. Routine weekly
  Bill auctions (4/8/13/17/26-week) are deliberately excluded -- they auction almost every business
  day and would flood the calendar; the screenshot itself only shows the one benchmark Note/Bond
  auction per day, never the Bills. Auction time (1:00pm ET) is the standard bid-close convention
  for Notes/Bonds (Bills close earlier, 11:30am ET, moot since Bills are excluded).

Every source is wrapped so a single failure (bad key, network error, rate limit) degrades to an
empty list for that source only -- never raises, never blanks out the other sources' events.
"""
from __future__ import annotations

import calendar
import re
import sys
from datetime import date, datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_bot_modules.omega6_live import L6_FOMC_DECISION_DATES  # noqa: E402

ET = ZoneInfo("America/New_York")

FRED_URL = "https://api.stlouisfed.org/fred/release/dates"
# (release_id, title_ko, hour_et, minute_et) -- 2026-09-01 expansion added 9 more releases past the
# original 5, all confirmed live via fred/releases search + a fred/release/dates future-date probe
# (今日=2026-09-01) before adding. Times are each release's own standard convention (BLS/Census 8:30am
# ET for most, Fed G.17 9:15am ET, JOLTS/home-sales 10:00am ET per BLS/NAR/Census practice) -- FRED's
# release/dates endpoint gives dates only, never times, same caveat as the original 5.
FRED_RELEASES = {
    "cpi": (10, "소비자물가지수(CPI)", 8, 30),
    "nfp": (50, "고용보고서(비농업고용, NFP)", 8, 30),
    "gdp": (53, "실질GDP성장률", 8, 30),
    "pce": (54, "PCE 가격지수", 8, 30),
    "durable_goods": (435, "내구재수주(잠정)", 8, 30),
    "ppi": (46, "생산자물가지수(PPI)", 8, 30),
    "retail_sales": (9, "소매판매(잠정)", 8, 30),
    "industrial_production": (13, "산업생산·설비가동률(G.17)", 9, 15),
    "housing_starts": (27, "신규주택착공·건축허가", 8, 30),
    "jolts": (192, "구인·이직보고서(JOLTS)", 10, 0),
    "jobless_claims": (180, "신규실업수당청구건수(주간)", 8, 30),
    "existing_home_sales": (291, "기존주택판매", 10, 0),
    "new_home_sales": (97, "신규주택판매", 10, 0),
    "trade_balance": (51, "미국 무역수지", 8, 30),
}
MICHIGAN_NEXT_RELEASE_URL = "https://www.sca.isr.umich.edu/"

FOMC_HOUR_ET, FOMC_MINUTE_ET = 14, 0
EIA_HOUR_ET, EIA_MINUTE_ET = 10, 30
FED_CALENDAR_URL = "https://www.federalreserve.gov/newsevents/{year}-{month}.htm"
FED_CALENDAR_MONTHS = 2  # current month plus next month covers the 21-day event horizon

EIA_URL = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"  # ping-only, confirms key validity
FINNHUB_EARNINGS_URL = "https://finnhub.io/api/v1/calendar/earnings"
EARNINGS_WATCHLIST = {"NVDA": "엔비디아"}  # extend here to widen coverage

TREASURY_AUCTIONS_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/upcoming_auctions"
TREASURY_NOTABLE_TYPES = {"Note", "Bond"}  # excludes routine weekly Bills (see module docstring)
TREASURY_HOUR_ET, TREASURY_MINUTE_ET = 13, 0

LOOKAHEAD_DAYS = 21  # how far ahead events are fetched before merging/sorting -- wide enough that
                     # a monthly-cadence release landing just past a 14-day window isn't cut off


def _et_to_utc(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET).astimezone(timezone.utc)


def fetch_fred_events(api_key: str | None, today: date) -> list[dict]:
    """2026-09-01: switched from sort_order=desc+limit=8 (no date bound) to sort_order=asc+
    realtime_start=today (limit=6) -- the desc form silently broke for release_id=180 (weekly
    Unemployment Insurance Claims, added this round): FRED already has WEEKLY dates scheduled out
    to December, so "latest 8" grabbed Nov-Dec and skipped the near-term Sep dates entirely (verified
    live before switching). asc+realtime_start asks FRED directly for the nearest dates from today
    forward, correct for both monthly and weekly cadences -- also applied to the original 5 monthly
    releases for consistency (their prior behavior is unchanged, just no longer relying on FRED
    happening not to have far-future monthly dates queued)."""
    if not api_key:
        return []
    events: list[dict] = []
    horizon = today + timedelta(days=LOOKAHEAD_DAYS)
    for _key, (release_id, title_ko, hour_et, minute_et) in FRED_RELEASES.items():
        try:
            resp = requests.get(
                FRED_URL,
                params={
                    "release_id": release_id, "api_key": api_key, "file_type": "json",
                    "include_release_dates_with_no_data": "true",
                    "sort_order": "asc", "limit": 6, "realtime_start": today.isoformat(),
                },
                timeout=10,
            )
            resp.raise_for_status()
            for row in resp.json().get("release_dates", []):
                d = date.fromisoformat(row["date"])
                if today <= d <= horizon:
                    events.append({
                        "time_utc": _et_to_utc(d, hour_et, minute_et).isoformat(),
                        "category": "econ",
                        "title_ko": title_ko,
                        "detail": f"{row.get('release_name', title_ko)} 발표 (FRED 공식 캘린더, release_id={release_id})",
                        "importance": "high",
                        "source": "FRED",
                    })
        except Exception as exc:  # noqa: BLE001 -- one release failing must not blank the rest
            print(f"macro-calendar: FRED release_id={release_id} fetch failed: {exc}", flush=True)
    return events


def fetch_michigan_events(today: date) -> list[dict]:
    """Fetch the next official Michigan Surveys of Consumers release from its homepage HTML."""
    horizon = today + timedelta(days=LOOKAHEAD_DAYS)
    try:
        resp = requests.get(
            MICHIGAN_NEXT_RELEASE_URL,
            headers={"User-Agent": "crypto-scalping-macro-calendar/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        text = " ".join(unescape(re.sub(r"<[^>]+>", " ", resp.text)).split())
        match = re.search(
            r"Next data release:\s*\w+,\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})\s+for\s+(.+?)\s+at\s+"
            r"(\d{1,2})(?::(\d{2}))?\s*(a\.?m?\.?|p\.?m?\.?)\s+ET",
            text,
            re.I,
        )
        if not match:
            print("macro-calendar: Michigan official next-release note not found", flush=True)
            return []
        event_day = datetime.strptime(match.group(1), "%B %d, %Y").date()
        if not (today <= event_day <= horizon):
            return []
        hour = int(match.group(3))
        minute = int(match.group(4) or 0)
        meridiem = match.group(5).replace(".", "").lower()
        if meridiem.startswith("p") and hour != 12:
            hour += 12
        elif meridiem.startswith("a") and hour == 12:
            hour = 0
        return [{
            "time_utc": _et_to_utc(event_day, hour, minute).isoformat(),
            "category": "econ",
            "title_ko": "미시간대 소비자심리지수·기대인플레이션 발표",
            "detail": f"{match.group(2).strip()} 발표 (미시간대 Surveys of Consumers 공식 일정)",
            "importance": "high",
            "source": MICHIGAN_NEXT_RELEASE_URL,
        }]
    except Exception as exc:  # noqa: BLE001 -- this source must not blank other calendar events
        print(f"macro-calendar: Michigan official calendar fetch failed: {exc}", flush=True)
        return []


def fomc_events(today: date) -> list[dict]:
    events = []
    horizon = today + timedelta(days=LOOKAHEAD_DAYS)
    try:
        for year in (today.year, today.year + 1):
            for raw in L6_FOMC_DECISION_DATES.get(year, ()):
                d = date.fromisoformat(raw)
                if today <= d <= horizon:
                    events.append({
                        "time_utc": _et_to_utc(d, FOMC_HOUR_ET, FOMC_MINUTE_ET).isoformat(),
                        "category": "fomc",
                        "title_ko": "FOMC 금리결정 발표",
                        "detail": "연준 성명 발표 (2pm ET) -- omega6_live.py의 공식 연준 캘린더 기준",
                        "importance": "high",
                        "source": "Fed (static, verified)",
                    })
    except Exception as exc:  # noqa: BLE001
        print(f"macro-calendar: FOMC calendar failed: {exc}", flush=True)
    return events


class _FedCalendarParser(HTMLParser):
    """Extract the time/content/date columns from Fed calendar event rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict[str, str]] = []
        self._div_depth = 0
        self._row: dict[str, list[str]] | None = None
        self._row_depth = 0
        self._column: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "div":
            return
        self._div_depth += 1
        classes = set((dict(attrs).get("class") or "").split())
        if "row" in classes:
            # Event rows are sometimes wrapped in a panel row, so prefer the nested row that
            # actually contains the three time/content/date columns.
            self._row = {"time": [], "content": [], "date": []}
            self._row_depth = self._div_depth
            self._column = None
            return
        if self._row is not None and self._div_depth == self._row_depth + 1:
            if "col-xs-2" in classes:
                self._column = "time"
            elif "col-xs-7" in classes:
                self._column = "content"
            elif "col-xs-3" in classes:
                self._column = "date"
            else:
                self._column = None

    def handle_endtag(self, tag: str) -> None:
        if tag != "div":
            return
        if self._row is not None and self._div_depth == self._row_depth:
            self.rows.append({key: " ".join(value) for key, value in self._row.items()})
            self._row = None
            self._column = None
        elif self._row is not None and self._div_depth == self._row_depth + 1:
            self._column = None
        self._div_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._row is not None and self._column is not None:
            self._row[self._column].append(data)


def _month_with_offset(day: date, offset: int) -> tuple[int, int]:
    month_index = day.year * 12 + day.month - 1 + offset
    return month_index // 12, month_index % 12 + 1


def _parse_fed_time(value: str) -> tuple[int, int] | None:
    normalized = " ".join(value.lower().replace(".", "").split())
    if "noon" in normalized:
        return 12, 0
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m?\.?|p\.?m?\.?)\b", normalized)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3).replace(".", "")
    if meridiem.startswith("p") and hour != 12:
        hour += 12
    elif meridiem.startswith("a") and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _parse_fed_chair_events(html: str, year: int, month: int, source_url: str,
                             today: date, horizon: date) -> list[dict]:
    parser = _FedCalendarParser()
    parser.feed(html)
    events: list[dict] = []
    chair_row = re.compile(r"\b(Speech|Testimony|Discussion)\s*-\s*(?:Chair|Chairman)\b", re.I)
    for row in parser.rows:
        content = " ".join(row["content"].split())
        match = chair_row.search(content)
        if not match:
            continue
        day_match = re.search(r"\b([1-9]|[12]\d|3[01])\b", row["date"])
        parsed_time = _parse_fed_time(row["time"])
        if not day_match or not parsed_time:
            continue
        try:
            event_day = date(year, month, int(day_match.group(1)))
        except ValueError:
            continue
        if not (today <= event_day <= horizon):
            continue
        hour, minute = parsed_time
        kind = match.group(1).lower()
        title = {
            "speech": "연준 의장 연설",
            "testimony": "연준 의장 증언",
            "discussion": "연준 의장 대담",
        }[kind]
        short_content = re.split(r"\b(?:Watch Live|At the)\b", content, maxsplit=1, flags=re.I)[0].strip()
        events.append({
            "time_utc": _et_to_utc(event_day, hour, minute).isoformat(),
            "category": "fed_speech",
            "title_ko": title,
            "detail": f"{short_content} -- 미 연준 공식 월별 HTML 캘린더",
            "importance": "high",
            "source": source_url,
        })
    return events


def fetch_fed_chair_events(today: date) -> list[dict]:
    """Fetch scheduled Chair speech/testimony/discussion rows from the Fed's monthly HTML pages."""
    horizon = today + timedelta(days=LOOKAHEAD_DAYS)
    events: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for offset in range(FED_CALENDAR_MONTHS):
        year, month_number = _month_with_offset(today, offset)
        month_name = calendar.month_name[month_number].lower()
        url = FED_CALENDAR_URL.format(year=year, month=month_name)
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "crypto-scalping-macro-calendar/1.0"},
                timeout=10,
            )
            resp.raise_for_status()
            for event in _parse_fed_chair_events(resp.text, year, month_number, url, today, horizon):
                key = (event["time_utc"], event["detail"])
                if key not in seen:
                    seen.add(key)
                    events.append(event)
        except Exception as exc:  # noqa: BLE001 -- Fed page failure must not blank other sources
            print(f"macro-calendar: Fed monthly calendar {url} fetch failed: {exc}", flush=True)
    return events


def eia_events(today: date) -> list[dict]:
    """Rule-based (not fetched): every Wednesday 10:30am ET, shifted to Thursday when that week's
    Monday is a US market holiday. EIA's API serves data, not a release-schedule endpoint."""
    events = []
    try:
        nyse = mcal.get_calendar("NYSE")
        d = today
        while d <= today + timedelta(days=LOOKAHEAD_DAYS):
            if d.weekday() == 2:  # Wednesday
                monday = d - timedelta(days=2)
                # Shift Wed -> Thu only when that week's Monday is NOT a trading day (holiday).
                monday_is_trading_day = nyse.valid_days(str(monday), str(monday)).size > 0
                release_day = d if monday_is_trading_day else d + timedelta(days=1)
                events.append({
                    "time_utc": _et_to_utc(release_day, EIA_HOUR_ET, EIA_MINUTE_ET).isoformat(),
                    "category": "eia",
                    "title_ko": "EIA 주간 원유재고",
                    "detail": "고정 주간 일정(매주 수요일, 월요일 휴장시 목요일로 이동) 기반 추정 -- EIA API는 발표 일정 자체를 제공하지 않음",
                    "importance": "medium",
                    "source": "rule-based (EIA weekly cadence)",
                })
            d += timedelta(days=1)
    except Exception as exc:  # noqa: BLE001
        print(f"macro-calendar: EIA schedule computation failed: {exc}", flush=True)
    return events


# (day_rule, hour_et, minute_et, title_ko, detail) -- day_rule is a (kind, arg) pair: ("nth", n) ->
# nth NYSE trading day of the month, ("on_or_after", day) -> first NYSE trading day on/after that
# day-of-month. All 5 share the same "no FRED release, no fetchable official calendar" gap this
# module already handles rule-based for EIA -- verified 2026-09-01: FRED releases search
# (pmi/purchasing/manufactur/ism/markit keywords) returns none of these; S&P Global's own
# pmi.spglobal.com press-release pages return HTTP 403 to automated fetch.
# 2026-09-01: holiday-adjusted via pandas_market_calendars (same NYSE calendar eia_events() already
# uses), replacing an earlier plain weekday-counting version that matched trading_bot_modules/
# omega5_live.py's own (deliberately simplified, no holiday shift) event-risk-governor rule --
# omega5_live.py is no longer relied upon elsewhere in this repo, so there was no reason left to
# keep that limitation. Concretely fixes e.g. 2027-01-01 (a Friday): plain weekday-counting would
# call that "the 1st business day", but it's New Year's Day -- NYSE closed -- so the real PMI slot
# is the next actual trading day.
PMI_RULES = [
    (("nth", 1), 9, 45, "S&P Global 미국 제조업 PMI(확정치)", "전월 제조업 PMI 확정치 -- 매월 첫 거래일"),
    (("nth", 1), 10, 0, "ISM 제조업 PMI", "전월 제조업 PMI(ISM) -- 매월 첫 거래일, S&P Global 확정치와 같은 날"),
    (("nth", 3), 9, 45, "S&P Global 미국 서비스업 PMI(확정치)", "전월 서비스업 PMI 확정치 -- 매월 3번째 거래일"),
    (("nth", 3), 10, 0, "ISM 서비스업 PMI", "전월 서비스업 PMI(ISM) -- 매월 3번째 거래일, S&P Global 확정치와 같은 날"),
    (("on_or_after", 23), 9, 45, "S&P Global 플래시 PMI(제조업+서비스업)", "이번 달 속보치(잠정) -- 매월 23일 이후 첫 거래일"),
]


def pmi_events(today: date) -> list[dict]:
    """Rule-based (not fetched) -- see PMI_RULES docstring above for why. Every month without
    exception (web-confirmed 2026-09-01 for the S&P Global Final Manufacturing time/cadence; the
    other 4 share the same day-of-month conventions). Holiday-aware -- see PMI_RULES docstring."""
    events = []
    try:
        nyse = mcal.get_calendar("NYSE")
        horizon = today + timedelta(days=LOOKAHEAD_DAYS)
        year, month = today.year, today.month
        for _ in range(3):  # this month + up to 2 more, covers the 21-day horizon
            month_start = date(year, month, 1)
            next_month_start = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
            # query a week past month-end too: a late-month holiday cluster could in principle push
            # the "on_or_after 23rd" search (or, in a pathological month, even the 3rd trading day)
            # past the calendar month-end -- 7 days of margin comfortably covers that.
            window_end = next_month_start + timedelta(days=7)
            trading_days = [d.date() for d in nyse.valid_days(str(month_start), str(window_end))]
            trading_days_this_month = [d for d in trading_days if d < next_month_start]

            for day_rule, hour, minute, title, detail in PMI_RULES:
                kind, arg = day_rule
                if kind == "nth":
                    release_day = (trading_days_this_month[arg - 1]
                                   if len(trading_days_this_month) >= arg else None)
                else:
                    threshold = date(year, month, arg)
                    candidates = [d for d in trading_days if d >= threshold]
                    release_day = candidates[0] if candidates else None
                if release_day is not None and today <= release_day <= horizon:
                    events.append({
                        "time_utc": _et_to_utc(release_day, hour, minute).isoformat(),
                        "category": "econ",
                        "title_ko": title,
                        "detail": f"{detail}, 거래일 기준(휴장시 다음 거래일로 이동) -- FRED에 릴리스 없음, 규칙기반 추정",
                        "importance": "high",
                        "source": "rule-based (monthly PMI cadence, NYSE-holiday-adjusted)",
                    })
            month += 1
            if month > 12:
                month = 1
                year += 1
    except Exception as exc:  # noqa: BLE001
        print(f"macro-calendar: PMI schedule computation failed: {exc}", flush=True)
    return events


def fetch_finnhub_events(api_key: str | None, today: date) -> list[dict]:
    if not api_key:
        return []
    events = []
    horizon = today + timedelta(days=LOOKAHEAD_DAYS)
    try:
        resp = requests.get(
            FINNHUB_EARNINGS_URL,
            params={"from": today.isoformat(), "to": horizon.isoformat(), "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        hour_map = {"bmo": (8, 0), "amc": (16, 30), "dmh": (12, 0)}
        for row in resp.json().get("earningsCalendar", []):
            symbol = row.get("symbol")
            if symbol not in EARNINGS_WATCHLIST:
                continue
            d = date.fromisoformat(row["date"])
            hour, minute = hour_map.get(row.get("hour"), (12, 0))
            events.append({
                "time_utc": _et_to_utc(d, hour, minute).isoformat(),
                "category": "earnings",
                "title_ko": f"{EARNINGS_WATCHLIST[symbol]} 실적 발표",
                "detail": f"{symbol} 분기 실적 (EPS 예상 {row.get('epsEstimate', '-')}) -- Finnhub 실적 캘린더",
                "importance": "medium",
                "source": "Finnhub",
            })
    except Exception as exc:  # noqa: BLE001
        print(f"macro-calendar: Finnhub earnings fetch failed: {exc}", flush=True)
    return events


def fetch_treasury_events(today: date) -> list[dict]:
    """No API key needed. Filtered to Note/Bond auctions only -- see module docstring for why
    routine weekly Bill auctions are excluded."""
    events = []
    horizon = today + timedelta(days=LOOKAHEAD_DAYS)
    try:
        resp = requests.get(
            TREASURY_AUCTIONS_URL,
            params={
                "filter": f"auction_date:gte:{today.isoformat()}",
                "sort": "auction_date",
                "page[size]": 50,
            },
            timeout=10,
        )
        resp.raise_for_status()
        for row in resp.json().get("data", []):
            if row.get("security_type") not in TREASURY_NOTABLE_TYPES:
                continue
            d = date.fromisoformat(row["auction_date"])
            if not (today <= d <= horizon):
                continue
            term = row.get("security_term", "")
            events.append({
                "time_utc": _et_to_utc(d, TREASURY_HOUR_ET, TREASURY_MINUTE_ET).isoformat(),
                "category": "treasury",
                "title_ko": f"{term} 국채 입찰",
                "detail": f"{row.get('security_type')} {term} 입찰 마감(1pm ET) -- 미 재무부 공식 입찰 캘린더",
                "importance": "medium",
                "source": "US Treasury",
            })
    except Exception as exc:  # noqa: BLE001
        print(f"macro-calendar: Treasury auctions fetch failed: {exc}", flush=True)
    return events


def compute_macro_calendar(fred_key: str | None, eia_key: str | None, finnhub_key: str | None,
                            now: datetime | None = None) -> dict:
    """Never raises. Returns {"generated_at", "events": [...]} sorted soonest-first, each event
    {"time_utc","category","title_ko","detail","importance","source"}. eia_key is accepted for
    interface symmetry/future use (a real EIA schedule endpoint, if one appears) but the current
    rule-based eia_events() doesn't need it -- the key is verified reachable separately at
    integration time, not required for this pure-calendar computation."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ET).date()
    events: list[dict] = []
    events += fetch_fred_events(fred_key, today)
    events += fetch_michigan_events(today)
    events += fomc_events(today)
    events += fetch_fed_chair_events(today)
    events += eia_events(today)
    events += pmi_events(today)
    events += fetch_finnhub_events(finnhub_key, today)
    events += fetch_treasury_events(today)
    events.sort(key=lambda e: e["time_utc"])
    return {"generated_at": now.isoformat(), "events": events}


MACRO_EVENT_ALERT_WINDOW_MIN = 30  # user request 2026-08-26: "발표 30분 전" -- symmetric +-30min
                                    # around the release, same "surround the event" shape as the
                                    # NYSE-open badge, just narrower (a single data print's shock
                                    # is expected to decay faster than a whole session's elevated-
                                    # volume open) -- NOT separately re-validated per-release the
                                    # way the NYSE/LSE/JPX windows were; treat as a UI default.


def compute_macro_event_alert(events: list[dict], now: datetime | None = None,
                               window_minutes: int = MACRO_EVENT_ALERT_WINDOW_MIN) -> dict:
    """Same {"active": [...]} shape as live_session_volatility_alert_20260826.compute_session_
    volatility_alert() -- deliberately parallel so the frontend can reuse the same render pattern.
    Only "high" importance events (econ releases + FOMC) count; EIA/earnings/treasury (medium) are
    excluded since they were never part of the researched NYSE-open volatility effect this whole
    calendar feature grew out of. Must be called with a FRESH `now` every time (never cached) --
    the events list itself can be hours old (compute_macro_calendar()'s 6h cache), but whether
    "now" currently sits inside any event's window changes minute to minute."""
    now = now or datetime.now(timezone.utc)
    active = []
    for e in events:
        if e.get("importance") != "high":
            continue
        try:
            event_time = datetime.fromisoformat(e["time_utc"])
        except (KeyError, ValueError):
            continue
        minutes = (now - event_time).total_seconds() / 60.0
        if -window_minutes <= minutes <= window_minutes:
            active.append({"title_ko": e["title_ko"], "minutes_from_event": round(minutes, 1)})
    return {"active": active}


if __name__ == "__main__":
    import json
    import os

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    out = compute_macro_calendar(os.getenv("FRED_API_KEY"), os.getenv("EIA_API_KEY"), os.getenv("FINNHUB_API_KEY"))
    print(json.dumps(out, indent=2, ensure_ascii=False))
