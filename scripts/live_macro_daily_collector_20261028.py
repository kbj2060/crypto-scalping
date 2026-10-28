#!/usr/bin/env python3
"""**매크로 일·주 수집기** — FRED 계열 · 재무부 일일 TGA · DefiLlama 스테이블코인 · 순유동성 (2026-10-28, 사용자 «TGA·RRP·연준BS·
순유동성·STLFSI 는 하루·주 1번이라도 수집기 만들어줘»).

왜: Macro Trend 카드(dashboard/macro_trend.py)는 1시간마다 FRED 를 새로 받아 계산하고 버린다 -- 쌓아 둔 것이 없다. FRED 는
  지난 값을 **고쳐 쓴다**(개정·평활: NFCI 누수 사고가 그 예, docs/experiments/macro_crypto_correlation_20261027.md §4). 그래서
  값만이 아니라 **언제 그 값을 처음 봤는지**를 남겨 «그 시점에 알 수 있던 값»(point-in-time)을 다시 꺼낼 수 있게 한다.
무엇: hot = data/hot/macro_daily.sqlite(WAL) -- 하루 수십 행이라 lake 로 봉인하지 않고 hot 에 영구 보관(백업은
  backup_live_data.sh 의 hot 사본). 🔴봉인하지 않는 이유: 개정 행이 «관측 날짜»가 아니라 «본 시각»에 붙어 날짜 파티션과 안 맞는다.
  obs(series, date = 'YYYY-MM-DD' 관측 날짜, value, first_seen = 이 값을 처음 본 epoch 초, backfill)
    -- (series, date) 의 값이 바뀔 때만 새 행이 붙는다(append-only). 같은 값이면 아무것도 안 쓴다.
    -- ponytail: 원천이 관측을 지우면(FRED «.») 기록하지 않아 obs_latest 에 옛 값이 남는다 -- 필요하면 value NULL 행으로.
    -- 🔴backfill=1 행의 first_seen 은 «처음 수집한 시각»이지 공표 시각이 아니다. 그 행들로 공표 시점을 흉내 내면 미래참조다.
       point-in-time 은 backfill=0 행(수집 시작 뒤)만 믿는다. 공표 지연 규약(1주 등)은 쓰는 쪽이 따로 둔다.
  obs_latest(view) = (series, date) 마다 마지막으로 본 값 -- 카드·연구가 현재 빈티지로 읽을 때.
  runs(ts_ms, source, status, n_new) = 원천 호출 한 번(감시기는 fred 의 ok 를 본다).
계열:
  FRED -- macro_trend.SOURCES 의 FRED 계열 전부(판정표 지표) + 순유동성 재료 WALCL·WTREGEN·RRPONTSYD + 2년 금리 DGS2.
    첫 실행(그 계열 행이 없을 때)은 전체 이력, 그다음은 최근 400일만 받는다(그보다 오래된 개정은 못 잡는다 --
    ponytail: 400일 창, 오래된 개정까지 필요하면 ALFRED vintage API 로).
  DTS_TGA -- 재무부 Daily Treasury Statement 의 TGA 마감 잔고(백만$, 영업일마다 · 다음 영업일 오후 공표). FRED WTREGEN 은 수요일
    주간치라 이쪽이 더 촘촘하다. 표의 항목 이름이 두 번 바뀌었다(10-28 실측): ~2021-09-30 «Federal Reserve Account» close_today_bal
    → ~2022-04-14 «Treasury General Account (TGA)» close_today_bal → «Treasury General Account (TGA) Closing Balance» open_today_bal
    (마감 잔고가 open 열에 들어 있다 -- 다음 날 Opening Balance 와 같음을 확인). 첫 실행 2005-10~, 그다음은 최근 30일.
  STABLES_LLAMA -- DefiLlama USD 스테이블코인 전체 공급(카드와 같은 원천·함수). 매번 전체를 받는다. 오늘 날짜 값은 하루 중 바뀔 수 있다.
  NETLIQ -- macro_trend.netliq(카드와 같은 식)을 저장된 최신 WALCL·WTREGEN·RRPONTSYD 로 다시 계산해 같은 규칙으로 붙인다.
cron 매시(서버). 키는 .env 의 FRED_API_KEY. 주문·바이낸스 호출 없음.
  python scripts/live_macro_daily_collector_20261028.py              # cron(서버)
  python scripts/live_macro_daily_collector_20261028.py --selftest   # 네트워크 없이
"""
from __future__ import annotations

import datetime as dt
import math
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import macro_trend as mt  # noqa: E402
from scripts.data_store import ROOT, rw_connect, sqlite_init  # noqa: E402

DB = Path(os.environ.get("MACRO_DAILY_DB", ROOT / "data/hot/macro_daily.sqlite"))
FRED_IDS = sorted({s[5:] for s, *_ in mt.SOURCES.values() if s.startswith("fred:")} | {"WALCL", "WTREGEN", "RRPONTSYD", "DGS2"})
DTS_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance"
TGA_COL = {"Federal Reserve Account": "close_today_bal", "Treasury General Account (TGA)": "close_today_bal",
           "Treasury General Account (TGA) Closing Balance": "open_today_bal"}
FRED_WINDOW_D, DTS_WINDOW_D = 400, 30
DDL = ("CREATE TABLE IF NOT EXISTS obs(series TEXT NOT NULL, date TEXT NOT NULL, value REAL NOT NULL, first_seen INTEGER NOT NULL, "
       "backfill INTEGER NOT NULL, PRIMARY KEY(series, date, first_seen))",
       "CREATE TABLE IF NOT EXISTS runs(ts_ms INTEGER NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL, n_new INTEGER)",
       "CREATE VIEW IF NOT EXISTS obs_latest AS SELECT series, date, value, first_seen, backfill FROM obs o "
       "WHERE first_seen = (SELECT max(first_seen) FROM obs WHERE series = o.series AND date = o.date)")


def latest(con: sqlite3.Connection, series: str) -> dict[str, float]:
    return dict(con.execute("SELECT date, value FROM obs_latest WHERE series = ?", (series,)).fetchall())


def merge(con: sqlite3.Connection, series: str, s: pd.Series, now: int) -> int:
    """관측 날짜 → 값 Series 를 붙인다. (series, date) 의 마지막 값과 다를 때만 새 행. 그 계열 첫 저장이면 backfill=1."""
    prev = latest(con, series)
    bf = int(not prev)
    new = [(series, d, v, now, bf) for d, v in ((pd.Timestamp(i).strftime("%Y-%m-%d"), float(x)) for i, x in s.items())
           if math.isfinite(v) and (d not in prev or not math.isclose(prev[d], v, rel_tol=1e-12, abs_tol=0.0))]
    con.executemany("INSERT OR IGNORE INTO obs VALUES(?,?,?,?,?)", new)
    return len(new)


def tga_series(rows: list[dict]) -> pd.Series:
    """DTS operating_cash_balance 행들 → 날짜별 TGA 마감 잔고(백만$). 항목 이름에 따라 열이 다르다(머리 설명)."""
    out = {}
    for r in rows:
        col = TGA_COL.get(r.get("account_type"))
        v = r.get(col) if col else None
        if v not in (None, "", "null"):
            out[r["record_date"]] = float(v)
    return pd.Series(out, dtype="float64").sort_index()


def fetch_tga(start: str | None) -> pd.Series:
    rows, page = [], 1
    while True:
        params = {"fields": "record_date,account_type,open_today_bal,close_today_bal", "sort": "record_date",
                  "page[size]": 10000, "page[number]": page, **({"filter": f"record_date:gte:{start}"} if start else {})}
        r = requests.get(DTS_URL, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        rows += js["data"]
        if page >= js["meta"]["total-pages"]:
            return tga_series(rows)
        page += 1


def run_source(con: sqlite3.Connection, source: str, fn) -> bool:
    """원천 하나 -- 실패해도 다른 원천은 계속(다음 시간에 다시). 오류 문자열에서 FRED 키를 지운다."""
    t0 = int(time.time())
    try:
        con.execute("BEGIN")
        n = fn(t0)
        con.execute("INSERT INTO runs VALUES(?,?,?,?)", (int(time.time() * 1000), source, "ok", n))
        con.execute("COMMIT")
        if n:
            print(f"{time.strftime('%F %T')} {source} 새 값 {n}행", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        con.execute("ROLLBACK")
        msg = re.sub(r"api_key=[^&\s]+", "api_key=***", f"error:{type(e).__name__}: {e}")[:200]
        con.execute("INSERT INTO runs VALUES(?,?,?,?)", (int(time.time() * 1000), source, msg, 0))
        print(f"{time.strftime('%F %T')} {source} {msg}", file=sys.stderr, flush=True)
        return False


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")          # cron 은 .env 를 안 읽는다 -- 키가 비면 FRED 계열이 전부 실패로 runs 에 남는다
    sqlite_init(DB)
    today = dt.date.today()
    ok = 0
    with rw_connect(DB) as con:
        for ddl in DDL:
            con.execute(ddl)

        def fred(sid):
            return lambda now: merge(con, sid, mt.fetch_fred(sid, None if not latest(con, sid) else
                                                              (today - dt.timedelta(days=FRED_WINDOW_D)).isoformat(), timeout=60), now)
        for sid in FRED_IDS:
            ok += run_source(con, "fred", fred(sid))
        ok += run_source(con, "dts", lambda now: merge(con, "DTS_TGA", fetch_tga(
            None if not latest(con, "DTS_TGA") else (today - dt.timedelta(days=DTS_WINDOW_D)).isoformat()), now))
        ok += run_source(con, "llama", lambda now: merge(con, "STABLES_LLAMA", mt.fetch_stables(), now))

        def netliq(now):
            raw = {k: pd.Series(list(v.values()), index=pd.to_datetime(list(v)), dtype="float64").sort_index()
                   for k in ("WALCL", "WTREGEN", "RRPONTSYD") if (v := latest(con, k))}
            nl = mt.netliq(raw)
            return 0 if nl is None else merge(con, "NETLIQ", nl, now)
        ok += run_source(con, "netliq", netliq)
    return 0 if ok else 1


def selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        con = sqlite3.connect(Path(tmp) / "t.sqlite", isolation_level=None)
        for ddl in DDL:
            con.execute(ddl)
        s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-02", "2026-01-09"]))
        assert merge(con, "X", s, 100) == 2                                   # 첫 저장 = backfill
        assert merge(con, "X", s, 200) == 0                                   # 같은 값은 안 붙는다
        assert merge(con, "X", pd.Series([2.5, 3.0], index=pd.to_datetime(["2026-01-09", "2026-01-16"])), 300) == 2   # 개정 1 + 새 날 1
        assert merge(con, "X", pd.Series([float("nan")], index=pd.to_datetime(["2026-01-23"])), 400) == 0
        assert con.execute("SELECT date, value, backfill FROM obs_latest WHERE series='X' ORDER BY date").fetchall() == \
            [("2026-01-02", 1.0, 1), ("2026-01-09", 2.5, 0), ("2026-01-16", 3.0, 0)]
        # point-in-time: 250 시점에 알 수 있던 01-09 값은 2.0(개정 2.5 는 300 에 봤다)
        assert con.execute("SELECT value FROM obs WHERE series='X' AND date='2026-01-09' AND first_seen <= 250 "
                           "ORDER BY first_seen DESC LIMIT 1").fetchone() == (2.0,)
        assert merge(con, "Y", pd.Series([5.0], index=pd.to_datetime(["2026-01-02"])), 500) == 1 and \
            con.execute("SELECT backfill FROM obs WHERE series='Y'").fetchone() == (1,)   # 다른 계열은 따로 backfill
        con.close()
    tga = tga_series([
        {"record_date": "2021-09-30", "account_type": "Federal Reserve Account", "open_today_bal": "1", "close_today_bal": "215160"},
        {"record_date": "2021-09-30", "account_type": "Short-Term Cash Investments (Table V)", "open_today_bal": "0", "close_today_bal": "0"},
        {"record_date": "2022-01-03", "account_type": "Treasury General Account (TGA)", "open_today_bal": "406108", "close_today_bal": "366304"},
        {"record_date": "2026-10-26", "account_type": "Treasury General Account (TGA) Opening Balance", "open_today_bal": "982489",
         "close_today_bal": "null"},
        {"record_date": "2026-10-26", "account_type": "Treasury General Account (TGA) Closing Balance", "open_today_bal": "1020700",
         "close_today_bal": "null"}])
    assert tga.to_dict() == {"2021-09-30": 215160.0, "2022-01-03": 366304.0, "2026-10-26": 1020700.0}, tga
    assert {"STLFSI4", "WALCL", "WTREGEN", "RRPONTSYD", "DEXJPUS", "DGS2"} <= set(FRED_IDS), FRED_IDS
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else sys.exit(main())
