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

BOT_SYMBOL = os.getenv("BOT_SYMBOL", "ETHUSDT").strip().lower()


async def main() -> None:
    logger.info("starting duckdb persistence worker (symbol=%s)", BOT_SYMBOL.upper())

    ms_scanner = MicrostructureScanner(symbol=BOT_SYMBOL)
    tr_interceptor = TailRiskInterceptor(symbol=BOT_SYMBOL)

    ms_scanner.start()
    tr_interceptor.start()

    try:
        while True:
            await asyncio.sleep(5.0)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("shutdown signal received")
    finally:
        ms_scanner.stop()
        tr_interceptor.stop()
        logger.info("duckdb persistence worker stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
