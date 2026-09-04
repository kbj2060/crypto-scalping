"""LiqMagnetCollector -- forward-only poller that logs the Snapshot tab's "청산 자석" (liquidation
magnet) reading over time, so it can eventually be backtested.

Context: tail_risk_interceptor.py::_compute_liq_cluster() derives liq_cluster_price/direction/
strength from self._liq_events, an in-memory deque(maxlen=10_000) of real @forceOrder liquidation
events -- see class docstring there. That deque is NEVER persisted, and the derived cluster/magnet
value itself is not written to tail_risk.duckdb either (only the aggregated long_usd_1m/short_usd_1m
$ totals are, with no price attached) -- so there is currently zero historical record of what the
magnet pointed at, at any past moment. Confirmed live 2026-08-25: dashboard_state.json (trading_bot.
py's _dashboard_shadow_loop rewrites its micro/tail/playbook fields every ~10s) is the cheapest
already-computed source to read this from, without touching tail_risk_interceptor.py or
trading_bot.py at all -- same "read-only, don't touch the live bot" posture as oi_lsratio_collector.py
and every live_*_signal_*.py script in scripts/.

Pure collect-and-persist, no signal computation -- same separation oi_lsratio_collector.py's module
docstring describes (z-scoring / hold-rate / magnet-accuracy analysis happens downstream in research
scripts against the accumulated table, once enough of it exists).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_STATE_PATH = _ROOT / "data" / "live" / "dashboard_state.json"
_DB_PATH = str(os.getenv("QUANT_LIQ_MAGNET_DB_PATH", str(_ROOT / "data/live/liq_magnet_history.duckdb")))
_TABLE = "liq_magnet_1m"


class LiqMagnetCollector:
    """Poll dashboard_state.json on a fixed cadence and append one row per minute with the
    market price and the current liq_cluster_* reading. External interface mirrors
    OiLsRatioCollector/TailRiskInterceptor: start() / stop()."""

    def __init__(self) -> None:
        self.enabled = os.getenv("LIQ_MAGNET_ENABLE", "true").strip().lower() in ("1", "true", "yes", "on")
        self.poll_interval_sec = float(os.getenv("LIQ_MAGNET_POLL_INTERVAL_SEC", "60"))
        self._running = False
        self._poll_task: asyncio.Task | None = None
        self._last_written_minute: int | None = None
        self._last_poll_ts = 0.0
        self._last_poll_ok = False

    # -- DuckDB --------------------------------------------------------------------------------

    def _db_init(self) -> None:
        import duckdb
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        con = duckdb.connect(_DB_PATH)
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                ts TIMESTAMPTZ,
                market_price DOUBLE,
                liq_cluster_price DOUBLE,
                liq_cluster_direction INTEGER,
                liq_cluster_strength DOUBLE,
                distance_to_cluster_pct DOUBLE,
                valid_liq_stream BOOLEAN,
                tail_risk_updated_at VARCHAR,
                collected_at TIMESTAMPTZ,
                schema_version INTEGER
            )
            """
        )
        row = con.execute(f"SELECT MAX(ts) FROM {_TABLE}").fetchone()
        if row and row[0] is not None:
            logger.info("liq_magnet bootstrap: resuming after ts=%s", row[0])
        else:
            logger.info("liq_magnet bootstrap: empty table, starting fresh")
        con.close()

    def _db_insert_row(self, row: dict) -> None:
        import duckdb
        try:
            con = duckdb.connect(_DB_PATH)
            con.execute(
                f"""
                INSERT INTO {_TABLE} (
                    ts, market_price, liq_cluster_price, liq_cluster_direction, liq_cluster_strength,
                    distance_to_cluster_pct, valid_liq_stream, tail_risk_updated_at, collected_at, schema_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    row["ts"], row["market_price"], row["liq_cluster_price"], row["liq_cluster_direction"],
                    row["liq_cluster_strength"], row["distance_to_cluster_pct"], row["valid_liq_stream"],
                    row["tail_risk_updated_at"], datetime.now(timezone.utc), 1,
                ],
            )
            con.close()
        except Exception as e:
            logger.error("liq_magnet DB insert error: %s", e, exc_info=True)

    # -- read dashboard_state.json --------------------------------------------------------------

    def _read_state_blocking(self) -> dict | None:
        """Blocking; run via executor. Returns None on any read/parse problem (including a
        concurrent rewrite mid-read by trading_bot.py's shadow loop) so the caller just skips
        this cycle -- there is always another poll 60s later, no retry-with-backoff needed for a
        plain JSON file the way duckdb's write-lock contention needs one."""
        try:
            with open(_STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("liq_magnet state read failed: %s", e)
            return None
        tail = state.get("tail_risk") or {}
        price = state.get("price")
        cluster_price = tail.get("liq_cluster_price")
        if not (isinstance(price, (int, float)) and price > 0):
            return None
        return {
            "market_price": float(price),
            "liq_cluster_price": float(cluster_price) if isinstance(cluster_price, (int, float)) else None,
            "liq_cluster_direction": int(tail.get("liq_cluster_direction") or 0),
            "liq_cluster_strength": float(tail.get("liq_cluster_strength") or 0.0),
            "distance_to_cluster_pct": float(tail.get("distance_to_cluster_pct")) if isinstance(tail.get("distance_to_cluster_pct"), (int, float)) else None,
            "valid_liq_stream": bool(tail.get("valid_liq_stream")) if tail.get("valid_liq_stream") is not None else None,
            "tail_risk_updated_at": tail.get("updated_at"),
        }

    # -- lifecycle -------------------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                now = time.time()
                sleep_sec = self.poll_interval_sec - (now % self.poll_interval_sec) + 2.0
                await asyncio.sleep(sleep_sec)
                if not self._running:
                    break

                loop = asyncio.get_running_loop()
                parsed = await loop.run_in_executor(None, self._read_state_blocking)
                self._last_poll_ts = time.time()
                self._last_poll_ok = parsed is not None
                if parsed is not None:
                    now_utc = datetime.now(timezone.utc)
                    minute_key = int(now_utc.timestamp() // 60)
                    if minute_key != self._last_written_minute:
                        row = {"ts": now_utc.replace(second=0, microsecond=0), **parsed}
                        await loop.run_in_executor(None, self._db_insert_row, row)
                        self._last_written_minute = minute_key
                logger.info(
                    "liq_magnet poll: ok=%s cluster_price=%s dir=%s strength=%s",
                    self._last_poll_ok,
                    parsed.get("liq_cluster_price") if parsed else None,
                    parsed.get("liq_cluster_direction") if parsed else None,
                    parsed.get("liq_cluster_strength") if parsed else None,
                )
            except Exception as e:
                logger.error("liq_magnet poll loop error: %s", e, exc_info=True)
                await asyncio.sleep(15.0)

    def start(self) -> None:
        if not self.enabled:
            logger.info("LiqMagnetCollector disabled (LIQ_MAGNET_ENABLE=false)")
            return
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self._db_init)
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("LiqMagnetCollector started (interval=%.0fs, db=%s)", self.poll_interval_sec, _DB_PATH)

    def stop(self) -> None:
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()

    def status_line(self) -> str:
        if not self.enabled:
            return "[liq_magnet] disabled"
        age = time.time() - self._last_poll_ts if self._last_poll_ts else None
        age_txt = f"{age:.0f}s ago" if age is not None else "never"
        return f"[liq_magnet] last_poll={age_txt} ok={self._last_poll_ok}"


async def _main() -> None:
    import signal

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    collector = LiqMagnetCollector()
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

    collector.start()
    try:
        await stop_event.wait()
    finally:
        collector.stop()


if __name__ == "__main__":
    asyncio.run(_main())
