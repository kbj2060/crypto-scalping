"""OiLsRatioCollector -- forward-only poller for Binance USDS-M futures open interest and
long/short account ratio history.

Context: docs/experiments/eth_candidate_liquidation_heatmap_magnet_signal_scoping_20260822.md.
`/futures/data/openInterestHist` and its long/short-ratio siblings retain exactly 500 data
points regardless of `period` -- 5m gives ~1.7 days, 1h gives ~21 days, both measured live on
2026-08-22 (see the scoping doc for the exact boundary tests). That free rolling window does not
grow with calendar time, so the only way to accumulate a real multi-week, multi-regime sample is
to poll it ourselves starting now and keep every point.

Pure collect-and-persist, no signal computation -- z-scoring / IC / gating happens downstream in
research scripts against the accumulated table, the same separation tail_risk_interceptor.py
draws between raw liquidation persistence and its own (separate, ETH-only) shadow-signal logic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent

_BASE_URL = "https://fapi.binance.com"
_HEADERS = {"User-Agent": "crypto-scalping-oi-lsratio-collector/1.0"}

_DB_PATH = str(os.getenv("QUANT_OI_LSRATIO_DB_PATH", str(_ROOT / "data/live/oi_lsratio.duckdb")))
_TABLE = "oi_lsratio_5m"

# Real API retention is ~500 points regardless of period (see module docstring) -- 5m is the
# project's standard live cadence, so that is what we poll, accepting the short free lookback.
_PERIOD = "5m"
_PERIOD_SECONDS = 300


class OiLsRatioCollector:
    """Poll openInterestHist / globalLongShortAccountRatio / topLongShortPositionRatio on a
    schedule and upsert into a trailing window (self-healing: each poll re-fetches enough
    history to cover a missed cycle or two, and re-merges every row in that window against
    whatever is already stored so a straggling source can still fill in a column a later
    poll finds that an earlier one missed).

    External interface mirrors TailRiskInterceptor: start() / stop().
    """

    def __init__(self, symbol: str = "ethusdt"):
        self.symbol = symbol.lower()
        self._api_symbol = symbol.upper()
        self._table = _TABLE if self.symbol == "ethusdt" else f"{_TABLE}_{self.symbol.replace('usdt', '')}"
        self._running = False
        self._poll_task: asyncio.Task | None = None

        self.enabled = os.getenv("OI_ENABLE", "true").strip().lower() in ("1", "true", "yes", "on")
        self.poll_interval_sec = float(os.getenv("OI_POLL_INTERVAL_SEC", str(_PERIOD_SECONDS)))
        # Trailing points fetched per poll (self-heal window), not a backfill knob -- the API
        # itself only ever has ~500 points total (see module docstring).
        self.fetch_limit = int(os.getenv("OI_FETCH_LIMIT", "12"))

        self._last_stored_ts_ms: int | None = None
        self._last_poll_ok_sources = 0
        self._last_poll_ts = 0.0

    # -- DuckDB --------------------------------------------------------------------------------

    def _db_init(self) -> None:
        import duckdb
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        con = duckdb.connect(_DB_PATH)
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                ts TIMESTAMPTZ,
                symbol VARCHAR,
                sum_open_interest DOUBLE,
                sum_open_interest_value DOUBLE,
                global_ls_ratio DOUBLE,
                global_ls_long_account DOUBLE,
                global_ls_short_account DOUBLE,
                top_pos_ls_ratio DOUBLE,
                top_pos_ls_long_account DOUBLE,
                top_pos_ls_short_account DOUBLE,
                sources_ok INTEGER,
                collected_at TIMESTAMPTZ,
                schema_version INTEGER
            )
            """
        )
        row = con.execute(f"SELECT MAX(ts) FROM {self._table}").fetchone()
        if row and row[0] is not None:
            self._last_stored_ts_ms = int(row[0].timestamp() * 1000)
            logger.info("oi_lsratio(%s) bootstrap: resuming after ts=%s", self._api_symbol, row[0])
        else:
            logger.info("oi_lsratio(%s) bootstrap: empty table, starting fresh", self._api_symbol)
        con.close()

    _MERGE_COLS = (
        "sum_open_interest", "sum_open_interest_value",
        "global_ls_ratio", "global_ls_long_account", "global_ls_short_account",
        "top_pos_ls_ratio", "top_pos_ls_long_account", "top_pos_ls_short_account",
    )

    def _db_upsert_rows(self, rows: list[dict]) -> None:
        """Read-merge-delete-insert upsert keyed on ts. Needed because the three source
        endpoints do not all publish a given 5m bucket at the same instant (observed live
        2026-08-22: BTC got sources_ok=3/3 on every poll, but ETH/SOL each had 2 polls land
        at sources_ok=1 or 2 when one source hadn't published that bucket yet). The previous
        strict-watermark INSERT-only design silently and permanently lost whichever columns
        weren't ready on the FIRST poll to see a given ts, since later polls skip everything
        at or before the last-stored ts. COALESCE(new, existing) here lets a later poll fill
        in columns a straggling source missed the first time, without ever overwriting a
        real value with a fresh NULL."""
        import duckdb
        if not rows:
            return
        try:
            con = duckdb.connect(_DB_PATH)
            ts_list = [r["ts"] for r in rows]
            placeholders = ",".join(["?"] * len(ts_list))
            existing = con.execute(
                f"SELECT ts, sources_ok, {', '.join(self._MERGE_COLS)} FROM {self._table} WHERE ts IN ({placeholders})",
                ts_list,
            ).fetchall()
            existing_by_ts = {row[0]: row[1:] for row in existing}

            merged = []
            for r in rows:
                ex = existing_by_ts.get(r["ts"])
                ex_sources_ok = ex[0] if ex is not None else 0
                ex_cols = dict(zip(self._MERGE_COLS, ex[1:])) if ex is not None else {}
                merged_row = [r["ts"], self._api_symbol]
                for c in self._MERGE_COLS:
                    new_v = r.get(c)
                    merged_row.append(new_v if new_v is not None else ex_cols.get(c))
                merged_row.append(max(int(ex_sources_ok or 0), int(r.get("sources_ok", 0))))
                merged_row.append(datetime.now(timezone.utc))
                merged_row.append(2)  # schema_version 2: upsert/COALESCE fill-in (2026-08-22)
                merged.append(merged_row)

            con.execute(f"DELETE FROM {self._table} WHERE ts IN ({placeholders})", ts_list)
            con.executemany(
                f"""
                INSERT INTO {self._table} (
                    ts, symbol, sum_open_interest, sum_open_interest_value,
                    global_ls_ratio, global_ls_long_account, global_ls_short_account,
                    top_pos_ls_ratio, top_pos_ls_long_account, top_pos_ls_short_account,
                    sources_ok, collected_at, schema_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                merged,
            )
            con.close()
            self._last_stored_ts_ms = max(int(r["ts"].timestamp() * 1000) for r in rows)
        except Exception as e:
            logger.error("oi_lsratio(%s) DB upsert error: %s", self._api_symbol, e, exc_info=True)

    # -- Binance REST ----------------------------------------------------------------------------

    @staticmethod
    def _http_get_json(url: str, timeout: float = 15.0):
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _fetch_one(self, path: str) -> list[dict] | None:
        """Blocking; run via executor. Returns None (not []) on failure so callers can tell
        'source unreachable this cycle' apart from 'source reachable, genuinely no new points'."""
        url = f"{_BASE_URL}{path}?" + urllib.parse.urlencode(
            {"symbol": self._api_symbol, "period": _PERIOD, "limit": self.fetch_limit}
        )
        try:
            data = self._http_get_json(url)
            if not isinstance(data, list):
                logger.warning("oi_lsratio(%s) unexpected response shape from %s: %r", self._api_symbol, path, data)
                return None
            return data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            logger.warning("oi_lsratio(%s) fetch failed for %s: %s", self._api_symbol, path, e)
            return None

    def _poll_once_blocking(self) -> list[dict]:
        """Fetch all three endpoints, merge by timestamp, return every row in the fetched
        trailing window (not just ones newer than what is stored) so _db_upsert_rows can patch
        columns a straggling source missed on an earlier poll. Blocking; run via executor."""
        oi = self._fetch_one("/futures/data/openInterestHist")
        gls = self._fetch_one("/futures/data/globalLongShortAccountRatio")
        tls = self._fetch_one("/futures/data/topLongShortPositionRatio")

        by_ts: dict[int, dict] = {}

        def _merge(rows: list[dict] | None, mapper):
            if rows is None:
                return
            for row in rows:
                ts_ms = int(row["timestamp"])
                entry = by_ts.setdefault(ts_ms, {"sources_ok": 0})
                entry.update(mapper(row))
                entry["sources_ok"] += 1

        _merge(oi, lambda r: {
            "sum_open_interest": float(r["sumOpenInterest"]),
            "sum_open_interest_value": float(r["sumOpenInterestValue"]),
        })
        _merge(gls, lambda r: {
            "global_ls_ratio": float(r["longShortRatio"]),
            "global_ls_long_account": float(r["longAccount"]),
            "global_ls_short_account": float(r["shortAccount"]),
        })
        _merge(tls, lambda r: {
            "top_pos_ls_ratio": float(r["longShortRatio"]),
            "top_pos_ls_long_account": float(r["longAccount"]),
            "top_pos_ls_short_account": float(r["shortAccount"]),
        })

        self._last_poll_ok_sources = sum(x is not None for x in (oi, gls, tls))

        out = []
        for ts_ms, entry in sorted(by_ts.items()):
            entry["ts"] = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            out.append(entry)
        return out

    # -- lifecycle -------------------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                # Sync to _PERIOD_SECONDS boundary + 90s buffer so Binance has published the
                # bucket before we ask for it, mirroring _agg_loop's boundary-sync pattern in
                # tail_risk_interceptor.py. (Fixed 2026-08-22: the previous formula's "if >
                # interval: subtract interval" wraparound could collapse this to a few seconds
                # instead of guaranteeing >=90s -- confirmed live, first poll fired 26s after
                # start. Not a correctness bug given the upsert fix above always lets a later
                # poll patch an early one, but still worth landing consistently past the
                # boundary rather than relying on that safety net every cycle.)
                now = time.time()
                sleep_sec = (self.poll_interval_sec - (now % self.poll_interval_sec)) + 90.0
                await asyncio.sleep(sleep_sec)
                if not self._running:
                    break

                loop = asyncio.get_running_loop()
                rows = await loop.run_in_executor(None, self._poll_once_blocking)
                self._last_poll_ts = time.time()
                if rows:
                    await loop.run_in_executor(None, self._db_upsert_rows, rows)
                logger.info(
                    "oi_lsratio(%s) poll: sources_ok=%d/3 upserted_rows=%d",
                    self._api_symbol, self._last_poll_ok_sources, len(rows),
                )
            except Exception as e:
                logger.error("oi_lsratio(%s) poll loop error: %s", self._api_symbol, e, exc_info=True)
                await asyncio.sleep(15.0)

    def start(self) -> None:
        if not self.enabled:
            logger.info("OiLsRatioCollector(%s) disabled (OI_ENABLE=false)", self._api_symbol)
            return
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self._db_init)
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("OiLsRatioCollector started (%s, interval=%.0fs, db=%s)", self._api_symbol, self.poll_interval_sec, _DB_PATH)

    def stop(self) -> None:
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()

    def status_line(self) -> str:
        if not self.enabled:
            return f"[oi_lsratio {self._api_symbol}] disabled"
        age = time.time() - self._last_poll_ts if self._last_poll_ts else None
        age_txt = f"{age:.0f}s ago" if age is not None else "never"
        return f"[oi_lsratio {self._api_symbol}] last_poll={age_txt} sources_ok={self._last_poll_ok_sources}/3"
