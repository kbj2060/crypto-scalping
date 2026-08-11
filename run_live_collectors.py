#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microstructure_scanner import MicrostructureScanner
from tail_risk_interceptor import TailRiskInterceptor


logger = logging.getLogger("live_collectors_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

async def main():
    parser = argparse.ArgumentParser(
        description="Run MicrostructureScanner and TailRiskInterceptor together."
    )
    parser.add_argument("--scanner-symbol", default="ethusdt", help="Symbol passed to MicrostructureScanner and TailRiskInterceptor.")
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

    try:
        await stop_event.wait()
    finally:
        ms.stop()
        tr.stop()
        logger.info("Collectors stopped")


if __name__ == "__main__":
    asyncio.run(main())
