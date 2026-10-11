#!/usr/bin/env python3
"""DuckDB persistence worker.

Keeps live microstructure and tail-risk collectors running.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from microstructure_scanner import MicrostructureScanner
from oi_lsratio_collector import OiLsRatioCollector
from tail_risk_interceptor import TailRiskInterceptor


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("duckdb_persist_worker")

_DEFAULT_SYMBOLS = "ETHUSDT,BTCUSDT,SOLUSDT"
BOT_SYMBOLS = [s.strip().lower() for s in os.getenv("BOT_SYMBOLS", _DEFAULT_SYMBOLS).split(",") if s.strip()]
# Server-side deployment (2026-08-17) runs this worker alongside trading_bot.py, which already
# owns microstructure collection for every symbol (its own ms_scanner_btc/sol plus the primary
# ETH scanner) -- a second MicrostructureScanner writer for the same symbol on the same duckdb
# file would fight trading_bot.py for the write lock. These flags let a deployment collect only
# what it actually owns; both default on so this script's original single-machine behavior is
# unchanged unless a caller opts out.
COLLECT_MICROSTRUCTURE = os.getenv("COLLECT_MICROSTRUCTURE", "true").strip().lower() not in ("false", "0", "")
COLLECT_TAIL_RISK = os.getenv("COLLECT_TAIL_RISK", "true").strip().lower() not in ("false", "0", "")
# New 2026-08-22, defaults OFF: existing deployments (e.g. the server-side BTC/SOL tail-risk
# worker) must opt in explicitly rather than silently picking up a new collector on their next
# git pull. See docs/experiments/eth_candidate_liquidation_heatmap_magnet_signal_scoping_20260822.md.
COLLECT_OI_LSRATIO = os.getenv("COLLECT_OI_LSRATIO", "false").strip().lower() not in ("false", "0", "")
# 2026-10-12 trading_bot.py(Omega4.6.1 섀도우) 은퇴 -- 봇이 품고 있던 수집을 이 워커가 넘겨받는다.
# TAIL_SYMBOLS: 청산 요격기를 돌릴 심볼만 따로(미시구조는 ETH·BTC·SOL, 청산은 ETH 만 -- BTC/SOL
#   청산은 09-19 에 은퇴). 기본값 = BOT_SYMBOLS 라 기존 배포는 그대로다.
# WRITE_DASHBOARD_STATE: 봇이 10초마다 쓰던 data/live/dashboard_state.json 을 최소판으로 쓴다.
#   화면은 그 파일이 «바뀔 때» SSE 로 render() 를 돌리고(게이지·주문 버튼 글자·청산 목록) 실제로
#   읽는 값은 session 세 칸뿐이다(app.js buildSessionHtml). ops_watchdog check_dashboard 도 updated_at 을 본다.
TAIL_SYMBOLS = [s.strip().lower() for s in os.getenv("TAIL_SYMBOLS", ",".join(BOT_SYMBOLS)).split(",") if s.strip()]
WRITE_DASHBOARD_STATE = os.getenv("WRITE_DASHBOARD_STATE", "false").strip().lower() not in ("false", "0", "")
_STATE_PATH = Path(__file__).resolve().parents[1] / "data/live/dashboard_state.json"
_KST = timezone(timedelta(hours=9))


def session_flags(ts_utc: pd.Timestamp) -> dict[str, float]:
    """trading_bot.py 의 세션 플래그 그대로 -- 거래소 달력(JPX/LSE/NYSE, 휴장·점심 반영), 없으면 UTC 시각."""
    try:
        import pandas_market_calendars as mcal
        flags = {}
        ts_min = ts_utc.floor("min")
        for name, cal_name in (("session_asia", "JPX"), ("session_europe", "LSE"), ("session_us", "NYSE")):
            sched = mcal.get_calendar(cal_name).schedule(start_date=ts_utc.date(), end_date=ts_utc.date())
            active = False
            if not sched.empty:
                row = sched.iloc[0]
                active = pd.Timestamp(row["market_open"]) <= ts_min <= pd.Timestamp(row["market_close"])
                bs, be = row.get("break_start", pd.NaT), row.get("break_end", pd.NaT)
                if pd.notna(bs) and pd.notna(be) and pd.Timestamp(bs) <= ts_min < pd.Timestamp(be):
                    active = False
            flags[name] = 1.0 if active else 0.0
        return flags
    except Exception:
        hour = ts_utc.hour + ts_utc.minute / 60.0
        return {"session_asia": 1.0 if 0.0 <= hour < 8.0 else 0.0,
                "session_europe": 1.0 if 8.0 <= hour < 16.0 else 0.0,
                "session_us": 1.0 if 14.5 <= hour < 21.0 else 0.0}


def write_dashboard_state() -> None:
    now = pd.Timestamp.now(tz="UTC")
    tmp = _STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"session": session_flags(now), "updated_at": now.tz_convert(_KST).isoformat(),
                               "source": "duckdb_persist_worker"}))
    tmp.replace(_STATE_PATH)


async def main() -> None:
    logger.info("starting duckdb persistence worker (symbols=%s, microstructure=%s, tail_risk=%s[%s], oi_lsratio=%s, dashboard_state=%s)",
                ",".join(s.upper() for s in BOT_SYMBOLS), COLLECT_MICROSTRUCTURE, COLLECT_TAIL_RISK,
                ",".join(s.upper() for s in TAIL_SYMBOLS), COLLECT_OI_LSRATIO, WRITE_DASHBOARD_STATE)

    ms_scanners = [MicrostructureScanner(symbol=sym) for sym in BOT_SYMBOLS] if COLLECT_MICROSTRUCTURE else []
    tr_interceptors = [TailRiskInterceptor(symbol=sym) for sym in TAIL_SYMBOLS] if COLLECT_TAIL_RISK else []
    oi_collectors = [OiLsRatioCollector(symbol=sym) for sym in BOT_SYMBOLS] if COLLECT_OI_LSRATIO else []

    # Staggered start (2026-08-17): each start() kicks off a DB bootstrap (CREATE TABLE IF NOT
    # EXISTS + ALTER TABLE for missing columns) on a background task. When a brand-new duckdb
    # file doesn't exist yet, two instances bootstrapping concurrently race on that DDL --
    # confirmed live (a fresh tail_risk_btc_sol.duckdb hit "table does not have column
    # liq_event_count_1m" because the BTC and SOL bootstraps overlapped). A 1s gap lets each
    # bootstrap finish before the next one opens a connection to the same file.
    for scanner in ms_scanners:
        scanner.start()
        await asyncio.sleep(1.0)
    for interceptor in tr_interceptors:
        interceptor.start()
        await asyncio.sleep(1.0)
    for collector in oi_collectors:
        collector.start()
        await asyncio.sleep(1.0)

    try:
        while True:
            if WRITE_DASHBOARD_STATE:
                try:
                    await asyncio.get_running_loop().run_in_executor(None, write_dashboard_state)
                except Exception as e:      # 화면 갱신 신호일 뿐 -- 실패해도 수집은 계속
                    logger.warning("dashboard_state write failed: %s", e)
            await asyncio.sleep(10.0 if WRITE_DASHBOARD_STATE else 5.0)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("shutdown signal received")
    finally:
        for scanner in ms_scanners:
            scanner.stop()
        for interceptor in tr_interceptors:
            interceptor.stop()
        for collector in oi_collectors:
            collector.stop()
        logger.info("duckdb persistence worker stopped")


if __name__ == "__main__" and "--selftest" in sys.argv:
    f = session_flags(pd.Timestamp("2026-10-12 15:00", tz="UTC"))           # 월요일 15:00 UTC = 런던장
    assert f["session_europe"] == 1.0 and f["session_asia"] == 0.0, f
    assert set(session_flags(pd.Timestamp("2026-10-11 03:00", tz="UTC"))) == {"session_asia", "session_europe", "session_us"}
    print("selftest ok")
elif __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
