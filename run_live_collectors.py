#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microstructure_scanner import MicrostructureScanner
from polymarket_engine import (
    POLYMARKET_ENABLE,
    append_polymarket_snapshot_to_duckdb,
    get_polymarket_snapshot_cached,
)
from tail_risk_interceptor import TailRiskInterceptor


logger = logging.getLogger("live_collectors_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

DEFAULT_POLY_DB_PATH = str(ROOT / "data/live/polymarket.duckdb")
PRICE_URL = "https://fapi.binance.com/fapi/v1/ticker/price"


def fetch_binance_price(symbol: str) -> float:
    query = urlencode({"symbol": symbol.upper()})
    req = Request(
        url=f"{PRICE_URL}?{query}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Connection": "close",
        },
    )
    with urlopen(req, timeout=3.0) as resp:
        payload = resp.read().decode("utf-8")
    import json

    data = json.loads(payload)
    return float(data["price"])


async def run_polymarket_loop(symbol: str, db_path: str, stop_event: asyncio.Event, interval_sec: float):
    last_price = 0.0
    while not stop_event.is_set():
        started = time.time()
        try:
            last_price = await asyncio.to_thread(fetch_binance_price, symbol)
            snapshot = get_polymarket_snapshot_cached(current_price=last_price)
            append_polymarket_snapshot_to_duckdb(
                db_path=db_path,
                snapshot=snapshot,
                current_price=last_price,
                raw_payload=snapshot.get("raw_payload"),
                logger=logger,
            )
            logger.info(
                "Polymarket snapshot saved | price=%.2f | signal=%s | risk=%s",
                last_price,
                snapshot.get("signal", "-"),
                snapshot.get("risk_state", "-"),
            )
        except (URLError, TimeoutError, ValueError, KeyError) as exc:
            logger.warning("Polymarket loop skipped: %s", exc)
        except Exception as exc:
            logger.exception("Polymarket loop error: %s", exc)

        elapsed = time.time() - started
        sleep_for = max(0.5, interval_sec - elapsed)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)
        except asyncio.TimeoutError:
            pass


async def main():
    parser = argparse.ArgumentParser(
        description="Run MicrostructureScanner, TailRiskInterceptor, and Polymarket DuckDB persistence together."
    )
    parser.add_argument("--symbol", default="ETHUSDT", help="Binance futures symbol for the Polymarket price anchor.")
    parser.add_argument("--scanner-symbol", default="ethusdt", help="Symbol passed to MicrostructureScanner and TailRiskInterceptor.")
    parser.add_argument("--polymarket-db-path", default=DEFAULT_POLY_DB_PATH)
    parser.add_argument("--polymarket-interval-sec", type=float, default=10.0)
    args = parser.parse_args()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    ms = MicrostructureScanner(symbol=args.scanner_symbol)
    tr = TailRiskInterceptor(symbol=args.scanner_symbol)

    ms.start()
    tr.start()
    logger.info("MicrostructureScanner started")
    logger.info("TailRiskInterceptor started")

    poly_task = None
    if POLYMARKET_ENABLE:
        poly_task = asyncio.create_task(
            run_polymarket_loop(
                symbol=args.symbol,
                db_path=args.polymarket_db_path,
                stop_event=stop_event,
                interval_sec=max(1.0, float(args.polymarket_interval_sec)),
            )
        )
        logger.info("Polymarket persistence loop started")
    else:
        logger.info("Polymarket loop disabled by POLYMARKET_ENABLE=false")

    try:
        await stop_event.wait()
    finally:
        ms.stop()
        tr.stop()
        if poly_task is not None:
            poly_task.cancel()
            try:
                await poly_task
            except asyncio.CancelledError:
                pass
        logger.info("Collectors stopped")


if __name__ == "__main__":
    asyncio.run(main())
