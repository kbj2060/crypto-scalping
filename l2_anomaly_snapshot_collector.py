#!/usr/bin/env python3
"""L2 anomaly-triggered snapshot collector.

Standalone, independent of trading_bot.py -- opens its OWN @depth20@100ms / @aggTrade /
@forceOrder connections (does not reuse microstructure_scanner.py's or tail_risk_interceptor.py's
live connections), writes to its own duckdb file. This can be started/stopped/crashed without
ever touching the live trading bot's process (matches this repo's established pattern for every
other dashboard-adjacent collector -- oi_lsratio_collector.py, liq_magnet_collector.py -- of
staying fully decoupled from the bot's critical path).

Design (user, 2026-08-26): continuously persisting depth20@100ms + aggTrade would be ~860k depth
messages/day/symbol alone -- too much, and most of it is uninteresting. Instead: keep only a
small in-memory ring buffer (RING_SECONDS of raw messages) and flush it to disk ONLY when a
compound anomaly trigger fires -- liquidation-burst z-score AND concurrent price-move z-score both
crossing threshold ("청산이 많아지면서 폭등/폭락"). On trigger: write the pre-trigger ring buffer,
then keep streaming straight to disk for POST_SECONDS to capture the cascade unfolding, then
re-arm after COOLDOWN_SECONDS. Storage now scales with event COUNT, not wall-clock time.

Both z-scores use a slow EWMA baseline (mean + mean-of-squares) updated on a fixed TICK_INTERVAL,
not a fixed-$ or fixed-% cutoff -- matches this repo's established "relative/expanding threshold,
not absolute" convention (GEX Tier1/2, §13 crowding tercile, etc.), so thresholds stay meaningful
as volatility/liquidation-activity regimes change. All thresholds/windows below are STARTING
DEFAULTS, not tuned -- treat as adjustable until real captures accumulate and can be reviewed.

Ties back to this session's research question: the original ask was whether order-flow/depth
data can tell you if a liquidation-driven move breaks or holds a level. Continuous L2 collection
was ruled out on cost; this collector instead builds a curated dataset of exactly the moments a
cascade might be forming, at a storage budget proportional to how often that actually happens.

quant_ai conda env (websockets, duckdb). Read-only w.r.t. every other table/file in this repo.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger("L2AnomalyCollector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent
# env-overridable so multiple symbols can each get their own file (DuckDB is single-writer-per-file
# in this repo's convention) -- default unchanged for backward compat with ETH's existing file/consumers.
DB_PATH = Path(os.getenv("L2_ANOMALY_DB_PATH", str(ROOT / "data" / "live" / "l2_anomaly_snapshots.duckdb")))

_DEPTH_WS_URL = "wss://fstream.binance.com/ws/{symbol}@depth20@100ms"
_TRADE_WS_URL = "wss://fstream.binance.com/ws/{symbol}@aggTrade"
_FORCE_ORDER_WS_URL = "wss://fstream.binance.com/market/ws/{symbol}@forceOrder"
# REST fallback for the aggTrade WS (2026-08-28) -- l2_anomaly_trades sat at 0 rows for 24h+ despite
# depth/forceOrder working fine over the identical connect/reconnect pattern. Diffed this file's
# _trade_loop against microstructure_scanner.py's byte-for-byte: same URL, same
# ping_interval/ping_timeout, same asyncio.wait_for(35.0) stall guard -- and that script's own
# aggTrade WS logs the identical ~45-50s reconnect churn right now, live-confirmed on the server.
# So this is very likely a genuine upstream/Binance-side issue with this raw stream (not a bug
# specific to this collector), and microstructure_scanner.py has been silently compensating for it
# via _poll_recent_agg_trades() this whole time -- which is exactly why microstructure_1m's trade-
# derived columns (nif_whale, taker_buy_ratio, recent_whale_count_5m) never showed a gap despite
# sharing the same broken WS. Same fix here: poll recent trades over REST, dedup by trade id, feed
# them through the existing _on_trade() so the ring buffer / capture-write path is unchanged.
_AGG_TRADES_URL = "https://fapi.binance.com/fapi/v1/aggTrades?symbol={symbol}&limit={limit}"
TRADE_REST_FALLBACK_INTERVAL_SECONDS = 10.0  # frequent enough to keep RING_SECONDS=90 populated
TRADE_REST_FALLBACK_LIMIT = 500

# -- adjustable defaults, see module docstring --
RING_SECONDS = 90            # pre-trigger context kept in memory at all times
POST_SECONDS = 120           # continue writing straight to disk this long after a trigger
COOLDOWN_SECONDS = 300       # minimum gap between two trigger episodes
LIQ_WINDOW_SECONDS = 60      # trailing window for the liquidation-burst sum
PRICE_WINDOW_SECONDS = 60    # trailing window for the price-move magnitude
TICK_INTERVAL_SECONDS = 5.0  # how often the trigger condition is re-evaluated
EWMA_HALFLIFE_SECONDS = 1800.0  # 30min slow-adapting "what's normal right now" baseline
WARMUP_TICKS = 360           # ~30min at TICK_INTERVAL=5s before triggering is allowed
LIQ_Z_THRESH = 2.0
PRICE_Z_THRESH = 2.0

# Liquidity-withdrawal leg (2026-08-26 addition, per arXiv:2608.03616's finding that the
# in-cascade signature lives in the order-book/liquidity sector, not position density).
# LOGGED ONLY -- does not gate the trigger yet. Top-20 bid+ask notional is a magnitude that
# DROPS during withdrawal, so a "matches literature" event has liquidity_z <= -LIQ_WITHDRAWAL_Z_THRESH
# (negative), unlike the liq/price legs which fire on large positive z. Kept separate rather than
# folded into the AND condition until real captures show whether it actually co-occurs on this
# collector's own data -- promoting it to a hard gate before that would just be assuming the
# literature transfers unchanged from BTC's largest-ever cascade to ETH's day-to-day bursts.
LIQ_WITHDRAWAL_Z_THRESH = 2.0


def _depth_notional(levels: list) -> float:
    """Sum(price*qty) across whatever levels are present (up to top-20, per the depth20 stream)."""
    return float(sum(p * q for p, q in levels))


class _RunningStats:
    """EWMA mean + mean-of-squares, fixed per-tick decay (ticks arrive on a regular clock here,
    so a fixed alpha is correct -- no need for irregular-interval-aware decay math)."""

    def __init__(self, halflife_sec: float, tick_sec: float) -> None:
        self._alpha = 1.0 - 0.5 ** (tick_sec / halflife_sec)
        self.mean = 0.0
        self.mean_sq = 0.0
        self.n = 0

    def update(self, x: float) -> None:
        if self.n == 0:
            self.mean, self.mean_sq = x, x * x
        else:
            self.mean += self._alpha * (x - self.mean)
            self.mean_sq += self._alpha * (x * x - self.mean_sq)
        self.n += 1

    def zscore(self, x: float) -> float:
        var = max(self.mean_sq - self.mean * self.mean, 1e-12)
        return (x - self.mean) / (var ** 0.5)


def _init_db(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS l2_anomaly_events (
            event_id VARCHAR, symbol VARCHAR, triggered_at_kst TIMESTAMPTZ,
            liq_burst_usd_60s DOUBLE, liq_z DOUBLE,
            price_move_pct_60s DOUBLE, price_z DOUBLE,
            liquidity_notional_usd DOUBLE, liquidity_z DOUBLE, liquidity_withdrawal_matched BOOLEAN,
            schema_version VARCHAR
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS l2_anomaly_depth (
            event_id VARCHAR, symbol VARCHAR, phase VARCHAR, ts_ms BIGINT,
            bids_json VARCHAR, asks_json VARCHAR
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS l2_anomaly_trades (
            event_id VARCHAR, symbol VARCHAR, phase VARCHAR, ts_ms BIGINT,
            price DOUBLE, qty DOUBLE, is_buyer_maker BOOLEAN
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS l2_anomaly_liquidations (
            event_id VARCHAR, symbol VARCHAR, phase VARCHAR, ts_ms BIGINT,
            side VARCHAR, qty_usd DOUBLE, price DOUBLE
        )""")


class L2AnomalySnapshotCollector:
    def __init__(self, symbol: str = "ethusdt") -> None:
        self.symbol = symbol.lower()
        self._running = False

        self._depth_ring: deque[tuple[int, list, list]] = deque()
        self._trade_ring: deque[tuple[int, float, float, bool]] = deque()
        self._liq_events: deque[tuple[int, str, float, float]] = deque()  # (ts_ms, side, qty_usd, price)
        self._mid_ring: deque[tuple[int, float]] = deque()           # (ts_ms, mid) for price-move calc

        self._liq_stats = _RunningStats(EWMA_HALFLIFE_SECONDS, TICK_INTERVAL_SECONDS)
        self._price_stats = _RunningStats(EWMA_HALFLIFE_SECONDS, TICK_INTERVAL_SECONDS)
        self._liquidity_stats = _RunningStats(EWMA_HALFLIFE_SECONDS, TICK_INTERVAL_SECONDS)
        self._latest_bids: list = []
        self._latest_asks: list = []
        self._tick_count = 0
        self._last_agg_trade_id: int | None = None  # dedup cursor for the REST trade fallback

        self._capturing = False
        self._capture_event_id: str | None = None
        self._capture_ends_at = 0.0
        self._cooldown_until = 0.0

        # Pending rows batched here and flushed periodically via a short-lived connection (open ->
        # write -> close, same pattern as orderbook_recorder.py::_append_row_duckdb) rather than
        # holding one connection open for the process lifetime -- a persistently-open connection
        # would lock this file for the entire run, so nothing (not even a read-only check) could
        # query it while the collector is up. Only matters during a capture episode; the rest of
        # the time these stay empty.
        self._pending_depth: list = []
        self._pending_trade: list = []
        self._pending_liq: list = []

        import duckdb
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(DB_PATH))
        try:
            _init_db(con)
        finally:
            con.close()

    # -- WS ingestion loops (reconnect-with-backoff, same pattern as microstructure_scanner.py /
    #    tail_risk_interceptor.py, verbatim style for consistency) --

    async def _depth_loop(self) -> None:
        import websockets
        url = _DEPTH_WS_URL.format(symbol=self.symbol)
        delay = 3.0
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("depth WS connected: %s", url)
                    delay = 3.0
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            ts_ms = int(msg.get("E", 0)) or int(time.time() * 1000)
                            bids = [[float(p), float(q)] for p, q in msg.get("b", [])]
                            asks = [[float(p), float(q)] for p, q in msg.get("a", [])]
                            self._on_depth(ts_ms, bids, asks)
                        except Exception:
                            pass
            except Exception as e:
                if self._running:
                    logger.warning("depth WS disconnected, reconnecting (%.0fs): %s", delay, e)
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, 60.0)

    async def _trade_loop(self) -> None:
        import websockets
        url = _TRADE_WS_URL.format(symbol=self.symbol)
        delay = 3.0
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("trade WS connected: %s", url)
                    delay = 3.0
                    while self._running:
                        # A WS ping/pong can keep the transport "connected" even after the
                        # aggTrade stream itself has stopped delivering events -- `async for raw
                        # in ws` would then just sit silently forever with zero trades captured
                        # (this is exactly what happened on the first deploy: connected once at
                        # startup, logged nothing wrong, and never produced a single trade row for
                        # 34+ minutes on ETHUSDT, which is not a plausible real trade lull). Bound
                        # recv so a stall raises and the outer except reconnects -- same fix
                        # microstructure_scanner.py's _trade_loop already uses for this reason.
                        raw = await asyncio.wait_for(ws.recv(), timeout=35.0)
                        try:
                            msg = json.loads(raw)
                            ts_ms = int(msg.get("T", 0))
                            price, qty = float(msg.get("p", 0.0)), float(msg.get("q", 0.0))
                            is_buyer_maker = bool(msg.get("m", False))
                            if price > 0 and qty > 0:
                                self._on_trade(ts_ms, price, qty, is_buyer_maker)
                        except Exception:
                            pass
            except Exception as e:
                if self._running:
                    logger.warning("trade WS disconnected, reconnecting (%.0fs): %s", delay, e)
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, 60.0)

    async def _trade_rest_fallback_loop(self) -> None:
        """Backfill trades over REST regardless of WS health -- see _AGG_TRADES_URL's comment for
        why this exists. Self-deduplicating via aggTrade id (_last_agg_trade_id), so harmless
        overlap with the WS path on the rare window it IS delivering -- same tradeoff
        microstructure_scanner.py::_poll_recent_agg_trades() already accepts (matched deliberately,
        not an oversight). Feeds the existing _on_trade() so the ring buffer and in-capture direct-
        write path behave identically to a trade arriving over WS."""
        import aiohttp
        url = _AGG_TRADES_URL.format(symbol=self.symbol.upper(), limit=TRADE_REST_FALLBACK_LIMIT)
        while self._running:
            await asyncio.sleep(TRADE_REST_FALLBACK_INTERVAL_SECONDS)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                        rows = await r.json(content_type=None)
                if not isinstance(rows, list) or not rows:
                    continue
                max_id = self._last_agg_trade_id
                for row in rows:
                    try:
                        trade_id = int(row.get("a", 0))
                        if self._last_agg_trade_id is not None and trade_id <= self._last_agg_trade_id:
                            continue
                        price, qty = float(row.get("p", 0.0)), float(row.get("q", 0.0))
                        ts_ms = int(row.get("T", 0))
                        is_buyer_maker = bool(row.get("m", False))
                        if price > 0 and qty > 0:
                            self._on_trade(ts_ms, price, qty, is_buyer_maker)
                        max_id = trade_id if max_id is None else max(max_id, trade_id)
                    except Exception:
                        continue
                if max_id is not None:
                    self._last_agg_trade_id = max_id
            except Exception as e:
                logger.debug("trade REST fallback failed: %s", e)

    async def _force_order_loop(self) -> None:
        import websockets
        url = _FORCE_ORDER_WS_URL.format(symbol=self.symbol)
        delay = 3.0
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("forceOrder WS connected: %s", url)
                    delay = 3.0
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            o = msg.get("o", {})
                            side = str(o.get("S", ""))
                            qty_usd = float(o.get("l", 0.0)) * float(o.get("ap", 0.0))
                            ts_ms = int(msg.get("E", 0))
                            if qty_usd > 0 and side in ("BUY", "SELL"):
                                self._on_liquidation(ts_ms, side, qty_usd, float(o.get("ap", 0.0)))
                        except Exception:
                            pass
            except Exception as e:
                if self._running:
                    logger.warning("forceOrder WS disconnected, reconnecting (%.0fs): %s", delay, e)
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, 60.0)

    # -- per-message handlers: always maintain the ring; if currently capturing, also write --

    def _on_depth(self, ts_ms: int, bids: list, asks: list) -> None:
        self._depth_ring.append((ts_ms, bids, asks))
        self._prune(self._depth_ring, ts_ms)
        self._latest_bids, self._latest_asks = bids, asks
        if bids and asks:
            mid = (bids[0][0] + asks[0][0]) / 2.0
            self._mid_ring.append((ts_ms, mid))
            self._prune(self._mid_ring, ts_ms)
        if self._capturing:
            self._write_depth_row(self._capture_event_id, "post", ts_ms, bids, asks)

    def _on_trade(self, ts_ms: int, price: float, qty: float, is_buyer_maker: bool) -> None:
        self._trade_ring.append((ts_ms, price, qty, is_buyer_maker))
        self._prune(self._trade_ring, ts_ms)
        if self._capturing:
            self._write_trade_row(self._capture_event_id, "post", ts_ms, price, qty, is_buyer_maker)

    def _on_liquidation(self, ts_ms: int, side: str, qty_usd: float, price: float) -> None:
        self._liq_events.append((ts_ms, side, qty_usd, price))
        self._prune(self._liq_events, ts_ms)
        if self._capturing:
            self._write_liq_row(self._capture_event_id, "post", ts_ms, side, qty_usd, price)

    def _prune(self, ring: deque, now_ts_ms: int) -> None:
        cutoff = now_ts_ms - int(RING_SECONDS * 1000)
        while ring and ring[0][0] < cutoff:
            ring.popleft()

    # -- trigger evaluation, on a fixed clock --

    async def _tick_loop(self) -> None:
        while self._running:
            await asyncio.sleep(TICK_INTERVAL_SECONDS)
            try:
                self._evaluate_trigger()
            except Exception:
                logger.exception("trigger evaluation failed")

    def _evaluate_trigger(self) -> None:
        now_ms = int(time.time() * 1000)
        liq_cutoff = now_ms - int(LIQ_WINDOW_SECONDS * 1000)
        liq_burst_usd = sum(q for ts, _side, q, _price in self._liq_events if ts >= liq_cutoff)

        # 2026-08-26 fix: this used to be net endpoint-to-endpoint displacement (recent mid vs mid
        # from PRICE_WINDOW_SECONDS ago), which MISSES a sharp spike-and-recover ("V-shape") --
        # exactly the shape a liquidation-driven flash move typically has (forced selling drives
        # price down fast, then the move partly/fully reverses once the selling exhausts -- the
        # same contrarian-reversal mechanism this session's whole liquidation research is about).
        # Confirmed missed for real on 2026-08-26 22:25: a $213,574 burst (bigger than the one that
        # DID trigger) produced a 1m bar open 2439.53 -> low 2434.05 -> close 2440.79 -- a 0.34%
        # intrabar swing but only ~0.05% net displacement, so the old metric never saw it. Now uses
        # high-low range over the trailing window instead, matching the same range-based target
        # already used in this session's OI-delta volatility screen (fwd_range_pct-style).
        price_cutoff = now_ms - int(PRICE_WINDOW_SECONDS * 1000)
        window_mids = [m for ts, m in self._mid_ring if ts >= price_cutoff]
        price_move_pct = 0.0
        if len(window_mids) >= 2 and window_mids[-1] > 0:
            price_move_pct = (max(window_mids) - min(window_mids)) / window_mids[-1] * 100.0

        liquidity_notional = _depth_notional(self._latest_bids) + _depth_notional(self._latest_asks)

        self._liq_stats.update(liq_burst_usd)
        self._price_stats.update(price_move_pct)
        if liquidity_notional > 0:
            self._liquidity_stats.update(liquidity_notional)
        self._tick_count += 1

        self._flush_pending()

        if self._capturing and time.time() >= self._capture_ends_at:
            logger.info("capture %s ended (post-window elapsed)", self._capture_event_id)
            self._capturing = False
            self._capture_event_id = None
            self._cooldown_until = time.time() + COOLDOWN_SECONDS
            return
        if self._capturing or self._tick_count < WARMUP_TICKS or time.time() < self._cooldown_until:
            return

        liq_z = self._liq_stats.zscore(liq_burst_usd)
        price_z = self._price_stats.zscore(price_move_pct)
        liquidity_z = self._liquidity_stats.zscore(liquidity_notional) if liquidity_notional > 0 else 0.0
        if liq_z >= LIQ_Z_THRESH and price_z >= PRICE_Z_THRESH:
            self._fire_trigger(liq_burst_usd, liq_z, price_move_pct, price_z, liquidity_notional, liquidity_z)

    def _fire_trigger(self, liq_burst_usd: float, liq_z: float, price_move_pct: float, price_z: float,
                       liquidity_notional: float, liquidity_z: float) -> None:
        event_id = str(uuid.uuid4())
        withdrawal_matched = bool(liquidity_z <= -LIQ_WITHDRAWAL_Z_THRESH)
        logger.warning(
            "TRIGGER symbol=%s event=%s liq_burst_usd_60s=%.0f liq_z=%.2f price_move_pct_60s=%.3f price_z=%.2f "
            "liquidity_notional_usd=%.0f liquidity_z=%.2f withdrawal_matched=%s",
            self.symbol, event_id, liq_burst_usd, liq_z, price_move_pct, price_z,
            liquidity_notional, liquidity_z, withdrawal_matched,
        )
        import duckdb
        import pandas as pd
        con = duckdb.connect(str(DB_PATH))
        try:
            con.execute(
                "INSERT INTO l2_anomaly_events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [event_id, self.symbol, pd.Timestamp.now(tz="Asia/Seoul"),
                 liq_burst_usd, liq_z, price_move_pct, price_z,
                 liquidity_notional, liquidity_z, withdrawal_matched, "l2_anomaly.v2"],
            )
        finally:
            con.close()

        for ts, bids, asks in list(self._depth_ring):
            self._write_depth_row(event_id, "pre", ts, bids, asks)
        for ts, price, qty, ibm in list(self._trade_ring):
            self._write_trade_row(event_id, "pre", ts, price, qty, ibm)
        for ts, side, qty_usd, price in list(self._liq_events):
            self._write_liq_row(event_id, "pre", ts, side, qty_usd, price)

        self._capturing = True
        self._capture_event_id = event_id
        self._capture_ends_at = time.time() + POST_SECONDS

    # -- writes: append to an in-memory pending batch; _flush_pending() does the actual disk I/O
    #    via a short-lived connection (open -> executemany -> close) so the file isn't locked open
    #    for the whole process lifetime. --

    def _write_depth_row(self, event_id, phase, ts_ms, bids, asks) -> None:
        self._pending_depth.append((event_id, self.symbol, phase, ts_ms, json.dumps(bids), json.dumps(asks)))

    def _write_trade_row(self, event_id, phase, ts_ms, price, qty, is_buyer_maker) -> None:
        self._pending_trade.append((event_id, self.symbol, phase, ts_ms, price, qty, is_buyer_maker))

    def _write_liq_row(self, event_id, phase, ts_ms, side, qty_usd, price) -> None:
        self._pending_liq.append((event_id, self.symbol, phase, ts_ms, side, qty_usd, price))

    def _flush_pending(self) -> None:
        if not (self._pending_depth or self._pending_trade or self._pending_liq):
            return
        depth, trade, liq = self._pending_depth, self._pending_trade, self._pending_liq
        self._pending_depth, self._pending_trade, self._pending_liq = [], [], []
        import duckdb
        con = duckdb.connect(str(DB_PATH))
        try:
            if depth:
                con.executemany("INSERT INTO l2_anomaly_depth VALUES (?,?,?,?,?,?)", depth)
            if trade:
                con.executemany("INSERT INTO l2_anomaly_trades VALUES (?,?,?,?,?,?,?)", trade)
            if liq:
                con.executemany("INSERT INTO l2_anomaly_liquidations VALUES (?,?,?,?,?,?,?)", liq)
        finally:
            con.close()
        logger.info("flushed pending rows: depth=%d trade=%d liq=%d", len(depth), len(trade), len(liq))

    async def run(self) -> None:
        self._running = True
        await asyncio.gather(
            self._depth_loop(), self._trade_loop(), self._trade_rest_fallback_loop(),
            self._force_order_loop(), self._tick_loop(),
        )


async def main() -> None:
    import os
    symbol = os.getenv("L2_ANOMALY_SYMBOL", "ethusdt")
    collector = L2AnomalySnapshotCollector(symbol=symbol)
    await collector.run()


if __name__ == "__main__":
    asyncio.run(main())
