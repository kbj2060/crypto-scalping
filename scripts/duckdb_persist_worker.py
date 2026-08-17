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
from tail_risk_interceptor import TailRiskInterceptor


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("duckdb_persist_worker")

_DEFAULT_SYMBOLS = "ETHUSDT,BTCUSDT,SOLUSDT"
BOT_SYMBOLS = [s.strip().lower() for s in os.getenv("BOT_SYMBOLS", _DEFAULT_SYMBOLS).split(",") if s.strip()]


async def main() -> None:
    logger.info("starting duckdb persistence worker (symbols=%s)", ",".join(s.upper() for s in BOT_SYMBOLS))

    ms_scanners = [MicrostructureScanner(symbol=sym) for sym in BOT_SYMBOLS]
    tr_interceptors = [TailRiskInterceptor(symbol=sym) for sym in BOT_SYMBOLS]

    for scanner in ms_scanners:
        scanner.start()
    for interceptor in tr_interceptors:
        interceptor.start()

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
        logger.info("duckdb persistence worker stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
