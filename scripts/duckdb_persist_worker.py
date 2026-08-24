#!/usr/bin/env python3
"""DuckDB persistence worker.

Keeps live microstructure and tail-risk collectors running.
"""

from __future__ import annotations

import asyncio
import logging
import os

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


async def main() -> None:
    logger.info("starting duckdb persistence worker (symbols=%s, microstructure=%s, tail_risk=%s, oi_lsratio=%s)",
                ",".join(s.upper() for s in BOT_SYMBOLS), COLLECT_MICROSTRUCTURE, COLLECT_TAIL_RISK, COLLECT_OI_LSRATIO)

    ms_scanners = [MicrostructureScanner(symbol=sym) for sym in BOT_SYMBOLS] if COLLECT_MICROSTRUCTURE else []
    tr_interceptors = [TailRiskInterceptor(symbol=sym) for sym in BOT_SYMBOLS] if COLLECT_TAIL_RISK else []
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
            await asyncio.sleep(5.0)
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


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
