#!/usr/bin/env python3
"""
MicrostructureScanner — 선행 레이더 (Pre-Crash Radar) + MFT 누적 연산기
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from trading_bot_modules.duckdb_access import serialized_duckdb_access

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent

# ── WebSocket / REST URL ───────────────────────────────────────────────────────
_DEPTH_WS_URL = "wss://fstream.binance.com/ws/{symbol}@depth20@100ms"
_TRADE_WS_URL = "wss://fstream.binance.com/ws/{symbol}@aggTrade"
_OI_URL       = "https://fapi.binance.com/fapi/v1/openInterest?symbol={SYMBOL}"
_FUND_URL     = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol={SYMBOL}"
_AGG_TRADES_URL = "https://fapi.binance.com/fapi/v1/aggTrades?symbol={SYMBOL}&limit={LIMIT}"

_DB_PATH = str(_ROOT / "data/live/microstructure.duckdb")
_TABLE   = "microstructure_1m"

class MicrostructureScanner:
    def __init__(self, symbol: str = "ethusdt"):
        self.symbol  = symbol.lower()
        # ETH (the default/original symbol) keeps the original unsuffixed table name so existing
        # consumers (audit scripts, feature views) keep working unchanged; other symbols get their
        # own table so BTC/SOL microstructure never mixes into ETH's rows (the table had no symbol
        # column to disambiguate by).
        self._table = _TABLE if self.symbol == "ethusdt" else f"{_TABLE}_{self.symbol.replace('usdt', '')}"
        self._running = False

        self._depth_task: asyncio.Task | None = None
        self._trade_task: asyncio.Task | None = None
        self._poll_task:  asyncio.Task | None = None
        self._scan_task:  asyncio.Task | None = None
        self._depth_connected = False
        self._trade_connected = False
        self._poll_connected = False
        self._depth_last_msg_ts = 0.0
        self._trade_last_msg_ts = 0.0
        self._poll_last_msg_ts = 0.0

        self._bids: list[tuple[float, float]] = []
        self._asks: list[tuple[float, float]] = []
        self._ob_lock = asyncio.Lock()

        self._trades: deque[tuple[int, bool, float]] = deque(maxlen=6_000)
        self._trade_notional_qty: deque[tuple[int, float, float]] = deque(maxlen=6_000)  # (ts_ms, notional_usd, qty)
        self._trade_lock = asyncio.Lock()

        # ── 💡 MFT 연산용 60분 메모리 버퍼 (새로 추가됨) ─────────────────
        self._price_hist: deque[float] = deque(maxlen=60)
        self._oi_hist:    deque[float] = deque(maxlen=60)
        self._price_obs: deque[tuple[int, float]] = deque(maxlen=120)
        self._oi_obs: deque[tuple[int, float]] = deque(maxlen=120)
        self._nif_obs: deque[tuple[int, float]] = deque(maxlen=720)
        self._oi_delta_obs: deque[tuple[int, float]] = deque(maxlen=720)
        self._nif_hist:   deque[float] = deque(maxlen=60)
        self._oi_delta_hist: deque[float] = deque(maxlen=60)
        self._abs_hist:   deque[float] = deque(maxlen=60)
        self._tox_hist:   deque[float] = deque(maxlen=60)
        self._bias_hist:  deque[int]   = deque(maxlen=60)
        self._eai_hist:   deque[float] = deque(maxlen=60)

        self._fund_rate:  float = 0.0
        self._last_agg_trade_id: int | None = None
        self._shadow_state: dict[str, float | str] = {
            "shadow_toxicity_score": 0.0,
            "shadow_toxicity_regime": "normal",
            "shadow_queue_collapse": 0.0,
            "shadow_absorption_score": 0.0,
            "shadow_queue_bias": 0,
            "shadow_regime_tag": "normal",
            "shadow_regime_conf": 0.0,
        }
        self._prev_bid_vol10: float = 0.0
        self._prev_ask_vol10: float = 0.0
        self._cached: dict = {}
        self._db_path = str(os.getenv("QUANT_MICRO_DB_PATH", str(_ROOT / "data/live/microstructure.duckdb")))
        self._warmup_30m_ready: bool = False
        self._bootstrap_loaded_rows: int = 0
        self._last_hist_bucket_min: int | None = None
        self._last_oi_tick: float | None = None
        self._oi_delta_cum_5m: float = 0.0
        self._oi_delta_5m_bucket: int = -1
        self._whale_flow_5m_bucket: int = -1
        self._whale_buy_cum_5m_usd: float = 0.0
        self._whale_sell_cum_5m_usd: float = 0.0
        self._whale_flow_state: int = 0  # -1 short, 0 neutral, +1 long

        # ── 임계값 ───────────────────────
        self.obi_wall_th      = float(os.getenv("MS_OBI_WALL_TH",      "-0.40"))
        self.taker_buy_th     = float(os.getenv("MS_TAKER_BUY_TH",     "0.60"))
        self.whale_usd_th     = float(os.getenv("MS_WHALE_USD_TH",     "100000"))
        self.eai_threshold    = float(os.getenv("MS_EAI_THRESHOLD",    "2.0"))
        self.poll_interval_sec = float(os.getenv("MS_POLL_INTERVAL_SEC", "10"))
        self.trade_rest_fallback_enabled = os.getenv("MS_TRADE_REST_FALLBACK_ENABLE", "true").strip().lower() in ("1", "true", "yes", "on")
        self.trade_rest_fallback_limit = int(float(os.getenv("MS_TRADE_REST_FALLBACK_LIMIT", "500")))
        self.whale_intent_nif_th = float(os.getenv("MS_WHALE_INTENT_NIF_TH", "0.10"))
        self.whale_flow_entry_th = float(os.getenv("MS_WHALE_FLOW_ENTRY_TH", "0.12"))
        self.whale_flow_exit_th = float(os.getenv("MS_WHALE_FLOW_EXIT_TH", "0.08"))
        self.whale_intent_oi_th = float(os.getenv("MS_WHALE_INTENT_OI_TH", "0.0001"))
        self.whale_intent_window_min = int(float(os.getenv("MS_WHALE_INTENT_WINDOW_MIN", "5")))
        self.whale_pos_long_short_th = float(os.getenv("MS_WHALE_POS_LONG_SHORT_TH", "0.25"))
        self.enabled          = os.getenv("MS_ENABLE", "true").strip().lower() in ("1", "true", "yes", "on")
        self.dsac_intercept_enabled = os.getenv("MS_DSAC_INTERCEPT_ENABLE", "false").strip().lower() in ("1", "true", "yes", "on")

    @serialized_duckdb_access(lambda self: self._db_path)
    def _db_init(self) -> None:
        import duckdb
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        con = duckdb.connect(self._db_path)
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                ts TIMESTAMPTZ, obi DOUBLE, taker_buy_ratio DOUBLE, spoofing_score DOUBLE,
                nif_whale DOUBLE, nif_retail DOUBLE, eai DOUBLE, oi_delta_pct DOUBLE,
                funding_rate DOUBLE, kelly_mult DOUBLE, signal_bias INTEGER,
                shadow_toxicity_score DOUBLE, shadow_toxicity_regime VARCHAR,
                shadow_queue_collapse DOUBLE, shadow_absorption_score DOUBLE,
                shadow_queue_bias INTEGER, shadow_regime_tag VARCHAR, shadow_regime_conf DOUBLE
            )
        """)
        existing_cols = {str(r[1]) for r in con.execute(f"PRAGMA table_info('{self._table}')").fetchall()}
        extra_cols = [
            ("data_stale", "BOOLEAN"),
            ("depth_connected", "BOOLEAN"),
            ("trade_connected", "BOOLEAN"),
            ("poll_connected", "BOOLEAN"),
            ("depth_age_sec", "DOUBLE"),
            ("trade_age_sec", "DOUBLE"),
            ("poll_age_sec", "DOUBLE"),
            ("recent_trade_count_5m", "INTEGER"),
            ("recent_trade_notional_5m", "DOUBLE"),
            ("recent_whale_count_5m", "INTEGER"),
            ("valid_taker_flow", "BOOLEAN"),
            ("valid_nif", "BOOLEAN"),
            ("warmup_30m_ready", "BOOLEAN"),
            ("schema_version", "INTEGER"),
            ("mark_price", "DOUBLE"),
            ("whale_position_score", "DOUBLE"),
        ]
        for col_name, col_type in extra_cols:
            if col_name not in existing_cols:
                con.execute(f"ALTER TABLE {self._table} ADD COLUMN {col_name} {col_type}")
        try:
            rows = con.execute(
                f"""
                SELECT
                    ts,
                    nif_whale,
                    oi_delta_pct,
                    shadow_absorption_score,
                    shadow_toxicity_score,
                    shadow_queue_bias,
                    eai,
                    mark_price
                FROM {self._table}
                WHERE ts >= now() - INTERVAL '60 minutes'
                ORDER BY ts ASC
                """
            ).fetchall()
            rows_5m = con.execute(
                f"""
                SELECT oi_delta_pct
                FROM {self._table}
                WHERE ts >= now() - INTERVAL '5 minutes'
                ORDER BY ts ASC
                """
            ).fetchall()
            for (
                ts,
                nif_whale,
                oi_delta_pct,
                shadow_absorption_score,
                shadow_toxicity_score,
                shadow_queue_bias,
                eai,
                mark_price,
            ) in rows:
                ts_ms = int(ts.timestamp() * 1000) if ts is not None else 0
                self._nif_hist.append(float(nif_whale or 0.0))
                self._oi_delta_hist.append(float(oi_delta_pct or 0.0))
                self._abs_hist.append(float(shadow_absorption_score or 0.0))
                self._tox_hist.append(float(shadow_toxicity_score or 0.0))
                self._bias_hist.append(int(shadow_queue_bias or 0))
                self._eai_hist.append(float(eai or 0.0))
                price = float(mark_price or 0.0)
                if price > 0.0:
                    self._price_hist.append(price)
                    self._price_obs.append((ts_ms, price))
                    self._last_hist_bucket_min = ts_ms // 60_000
            self._bootstrap_loaded_rows = len(rows)
            # 재시작 직후 OI 원시 시계열(self._oi_obs)이 비어도 OIΔ5m를 즉시 사용 가능하도록 시드
            if rows_5m:
                oi5 = 0.0
                for (dlt,) in rows_5m:
                    v = float(dlt or 0.0)
                    oi5 = (1.0 + oi5) * (1.0 + v) - 1.0
                self._cached["oi_delta_pct"] = float(oi5)
                self._cached["oi_delta_cum_5m"] = float(oi5)
                self._cached["oi_delta_cum_5m_bucket_start_ts"] = int((int(time.time()) // 300) * 300)
        except Exception:
            pass
        self._warmup_30m_ready = self._is_30m_ready()
        logger.info(
            "MS bootstrap: rows=%d, warmup_30m_ready=%s (price=%d nif=%d abs=%d tox=%d bias=%d)",
            self._bootstrap_loaded_rows,
            self._warmup_30m_ready,
            len(self._price_hist),
            len(self._nif_hist),
            len(self._abs_hist),
            len(self._tox_hist),
            len(self._bias_hist),
        )
        con.close()

    @serialized_duckdb_access(lambda self, *_args, **_kwargs: self._db_path)
    def _db_insert(self, bucket_ts: datetime, sig: dict) -> None:
        import duckdb
        try:
            con = duckdb.connect(self._db_path)
            valid_taker_flow = bool(sig.get("valid_taker_flow", False))
            valid_nif = bool(sig.get("valid_nif", False))
            con.execute(f"""
                INSERT INTO {self._table} (
                    ts, obi, taker_buy_ratio, spoofing_score,
                    nif_whale, nif_retail, eai, oi_delta_pct,
                    funding_rate, kelly_mult, signal_bias,
                    shadow_toxicity_score, shadow_toxicity_regime,
                    shadow_queue_collapse, shadow_absorption_score,
                    shadow_queue_bias, shadow_regime_tag, shadow_regime_conf,
                    data_stale, depth_connected, trade_connected, poll_connected,
                    depth_age_sec, trade_age_sec, poll_age_sec,
                    recent_trade_count_5m, recent_trade_notional_5m, recent_whale_count_5m,
                    valid_taker_flow, valid_nif, warmup_30m_ready, schema_version,
                    mark_price, whale_position_score
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, [
                    bucket_ts, sig.get("obi", 0.0),
                    (sig.get("taker_buy_ratio", 0.5) if valid_taker_flow else None),
                    sig.get("spoofing_score", 0.0),
                    (sig.get("nif_whale", 0.0) if valid_nif else None),
                    (sig.get("nif_retail", 0.0) if valid_taker_flow else None),
                    sig.get("eai", 0.0), sig.get("oi_delta_pct", 0.0),
                    sig.get("funding_rate", 0.0), sig.get("kelly_mult", 1.0), int(sig.get("signal_bias", 0)),
                    sig.get("shadow_toxicity_score", 0.0), sig.get("shadow_toxicity_regime", "normal"),
                    sig.get("shadow_queue_collapse", 0.0), sig.get("shadow_absorption_score", 0.0),
                    int(sig.get("shadow_queue_bias", 0)), sig.get("shadow_regime_tag", "normal"), sig.get("shadow_regime_conf", 0.0),
                    bool(sig.get("data_stale", True)),
                    bool(sig.get("depth_connected", False)),
                    bool(sig.get("trade_connected", False)),
                    bool(sig.get("poll_connected", False)),
                    sig.get("depth_age_sec"),
                    sig.get("trade_age_sec"),
                    sig.get("poll_age_sec"),
                    int(sig.get("recent_trade_count_5m", 0)),
                    float(sig.get("recent_trade_notional_5m", 0.0)),
                    int(sig.get("recent_whale_count_5m", 0)),
                    valid_taker_flow,
                    valid_nif,
                    bool(sig.get("warmup_30m_ready", False)),
                    3,
                    float(sig.get("mark_price", 0.0) or 0.0),
                    float(sig.get("whale_position_score", 0.0) or 0.0),
                ]
            )
            con.close()
        except Exception as e:
            logger.error("MS duckdb insert failed: %s", e, exc_info=True)

    async def _depth_loop(self) -> None:
        import websockets
        url = _DEPTH_WS_URL.format(symbol=self.symbol)
        delay = 3.0
        _logged_connected = False
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    delay = 3.0
                    self._depth_connected = True
                    if not _logged_connected:
                        logger.info("📡 MS Depth WS 연결: %s", url)
                        _logged_connected = True
                    async for raw in ws:
                        if not self._running: break
                        try:
                            msg  = json.loads(raw)
                            self._depth_last_msg_ts = time.time()
                            bids = [(float(p), float(q)) for p, q in msg.get("b", [])]
                            asks = [(float(p), float(q)) for p, q in msg.get("a", [])]
                            async with self._ob_lock:
                                self._bids, self._asks = bids, asks
                        except Exception: pass
            except Exception as e:
                self._depth_connected = False
                _logged_connected = False
                logger.warning("MS depth WS disconnected; reconnecting... (%s)", e)
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 60.0)

    async def _trade_loop(self) -> None:
        import websockets
        url = _TRADE_WS_URL.format(symbol=self.symbol)
        delay = 3.0
        _logged_connected = False
        outage_open = False
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    delay = 3.0
                    self._trade_connected = True
                    if outage_open:
                        logger.info("MS trade WS recovered: %s", url)
                        outage_open = False
                    elif not _logged_connected:
                        logger.info("📡 MS Trade WS 연결: %s", url)
                    _logged_connected = True
                    while self._running:
                        # A WebSocket ping/pong can keep the transport open even when
                        # the aggTrade stream has stopped delivering market events.
                        # Bound recv so the outer reconnect loop repairs that state.
                        raw = await asyncio.wait_for(ws.recv(), timeout=35.0)
                        try:
                            msg = json.loads(raw)
                            self._trade_last_msg_ts = time.time()
                            is_buyer_maker = bool(msg.get("m", False))
                            qty_usd = float(msg.get("q", 0.0)) * float(msg.get("p", 0.0))
                            ts_ms = int(msg.get("T", 0))
                            if qty_usd > 0:
                                async with self._trade_lock:
                                    self._trades.append((ts_ms, is_buyer_maker, qty_usd))
                                    self._trade_notional_qty.append((ts_ms, qty_usd, float(msg.get("q", 0.0))))
                        except Exception: pass
            except Exception as e:
                self._trade_connected = False
                _logged_connected = False
                if not outage_open:
                    logger.warning(
                        "MS trade WS disconnected; reconnecting... (%s: %s)",
                        type(e).__name__, e,
                    )
                    outage_open = True
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 60.0)

    async def _poll_recent_agg_trades(self, session) -> None:
        """Backfill trade flow when the websocket connects but does not deliver messages."""
        if not self.trade_rest_fallback_enabled:
            return
        try:
            limit = int(np.clip(self.trade_rest_fallback_limit, 1, 1000))
            url = _AGG_TRADES_URL.format(SYMBOL=self.symbol.upper(), LIMIT=limit)
            async with session.get(url, timeout=5) as r:
                rows = await r.json(content_type=None)
            if not isinstance(rows, list) or not rows:
                return
            appended = 0
            max_trade_id = self._last_agg_trade_id
            async with self._trade_lock:
                for row in rows:
                    try:
                        trade_id = int(row.get("a", 0))
                        if self._last_agg_trade_id is not None and trade_id <= self._last_agg_trade_id:
                            continue
                        price = float(row.get("p", 0.0))
                        qty = float(row.get("q", 0.0))
                        qty_usd = price * qty
                        ts_ms = int(row.get("T", 0))
                        is_buyer_maker = bool(row.get("m", False))
                        if qty_usd <= 0 or ts_ms <= 0:
                            continue
                        self._trades.append((ts_ms, is_buyer_maker, qty_usd))
                        self._trade_notional_qty.append((ts_ms, qty_usd, qty))
                        appended += 1
                        max_trade_id = trade_id if max_trade_id is None else max(max_trade_id, trade_id)
                    except Exception:
                        continue
                if max_trade_id is not None:
                    self._last_agg_trade_id = int(max_trade_id)
            if appended > 0:
                self._trade_last_msg_ts = time.time()
                self._trade_connected = True
        except Exception as e:
            logger.debug("MS aggTrades REST fallback failed: %s", e)

    async def _poll_loop(self) -> None:
        import aiohttp
        oi_url = _OI_URL.format(SYMBOL=self.symbol.upper())
        fund_url = _FUND_URL.format(SYMBOL=self.symbol.upper())
        timeout = aiohttp.ClientTimeout(total=5)
        _logged_connected = False
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    self._poll_connected = True
                    if not _logged_connected:
                        logger.info("📡 MS Poll REST 연결: oi=%s fund=%s", oi_url, fund_url)
                        _logged_connected = True
                    async with session.get(oi_url, timeout=timeout) as r:
                        data = await r.json(content_type=None)
                        oi = float(data.get("openInterest", 0.0))
                        now_ms = int(time.time() * 1000)
                        now_s = time.time()
                        prev_tick = self._last_oi_tick if self._last_oi_tick is not None else oi
                        self._last_oi_tick = oi
                        self._oi_obs.append((now_ms, oi))
                        oi_delta_tick = float((oi - prev_tick) / (abs(prev_tick) + 1e-8))
                        self._cached["oi_delta_pct"] = oi_delta_tick
                        bucket_5m = int(now_s // 300)
                        if bucket_5m != self._oi_delta_5m_bucket:
                            self._oi_delta_5m_bucket = bucket_5m
                            self._oi_delta_cum_5m = 0.0
                        # 10초 단위 변화율을 5분 버킷 내 복리 누적
                        self._oi_delta_cum_5m = (1.0 + self._oi_delta_cum_5m) * (1.0 + oi_delta_tick) - 1.0
                        self._cached["oi_delta_cum_5m"] = float(self._oi_delta_cum_5m)
                        self._cached["oi_delta_cum_5m_bucket_start_ts"] = int(bucket_5m * 300)
                    async with session.get(fund_url, timeout=timeout) as r:
                        data = await r.json(content_type=None)
                        self._fund_rate = float(data.get("lastFundingRate", 0.0))
                        mark = float(data.get("markPrice", 0.0))
                        if mark > 0:
                            self._price_obs.append((now_ms, mark))

                        # 30분 지표용 hist는 1분 버킷으로만 누적 (폴링 주기와 분리)
                        bucket_min = int(time.time() // 60)
                        if self._last_hist_bucket_min != bucket_min:
                            self._last_hist_bucket_min = bucket_min
                            if oi > 0:
                                self._oi_hist.append(oi)
                            if mark > 0:
                                self._price_hist.append(mark)
                    await self._poll_recent_agg_trades(session)
                    self._poll_last_msg_ts = time.time()
            except Exception as e:
                self._poll_connected = False
                _logged_connected = False
                logger.warning("MS poll REST failed; retrying... (%s)", e)
            await asyncio.sleep(max(3.0, self.poll_interval_sec))

    def _compute_obi(self) -> float:
        bids, asks = self._bids[:10], self._asks[:10]
        if not bids or not asks: return 0.0
        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        return float((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8))

    def _compute_nif_and_taker(self, window_sec: int = 300) -> tuple[float, float, float, int, float, int]:
        cutoff = int(time.time() * 1000) - window_sec * 1000
        wb = ws_ = rb = rs_ = tb = ts = 0.0
        trade_count = 0
        whale_count = 0
        total_notional = 0.0
        for t_ms, is_buyer_maker, usd in self._trades:
            if t_ms < cutoff: continue
            trade_count += 1
            total_notional += float(usd)
            is_taker_buy = not is_buyer_maker
            if is_taker_buy: tb += usd
            else: ts += usd
            if usd >= self.whale_usd_th:
                whale_count += 1
                if is_taker_buy: wb += usd
                else: ws_ += usd
            else:
                if is_taker_buy: rb += usd
                else: rs_ += usd
        nif_whale = (wb - ws_) / (wb + ws_ + 1e-8)
        nif_retail = (rb - rs_) / (rb + rs_ + 1e-8)
        taker_buy_ratio = tb / (tb + ts + 1e-8)
        return float(nif_whale), float(nif_retail), float(taker_buy_ratio), int(trade_count), float(total_notional), int(whale_count)

    def _compute_whale_flow_window(self, window_sec: int = 10) -> tuple[float, float]:
        cutoff = int(time.time() * 1000) - window_sec * 1000
        whale_buy = 0.0
        whale_sell = 0.0
        for t_ms, is_buyer_maker, usd in self._trades:
            if t_ms < cutoff:
                continue
            if usd < self.whale_usd_th:
                continue
            is_taker_buy = not is_buyer_maker
            if is_taker_buy:
                whale_buy += usd
            else:
                whale_sell += usd
        return float(whale_buy), float(whale_sell)

    def _compute_eai(self) -> float:
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - 5 * 60 * 1000
        prices = [p for ts, p in self._price_obs if ts >= cutoff]
        ois = [v for ts, v in self._oi_obs if ts >= cutoff]
        if len(prices) < 2 or len(ois) < 2:
            return 0.0
        p_arr = np.array(prices, dtype=np.float64)
        price_range = (p_arr.max() - p_arr.min()) / (p_arr.mean() + 1e-8)
        oi_delta = abs(ois[-1] - ois[0]) / (abs(ois[0]) + 1e-8)
        return float(np.clip(oi_delta / (price_range + 1e-6), 0.0, 20.0))

    def _compute_shadow_toxicity(self, obi: float, taker_buy_ratio: float) -> tuple[float, str]:
        mismatch = abs(obi - (2.0 * taker_buy_ratio - 1.0))
        burst = 0.0
        if self._trades:
            recent = [usd for _, _, usd in list(self._trades)[-120:]]
            if recent:
                arr = np.array(recent, dtype=np.float64)
                burst = float(np.clip(np.percentile(arr, 95) / (arr.mean() + 1e-8), 0.0, 5.0))
        score = float(np.clip(0.7 * mismatch + 0.3 * (burst / 5.0), 0.0, 1.0))
        return score, "toxic" if score >= 0.75 else "watch" if score >= 0.50 else "normal"

    def _compute_shadow_queue_absorption(self, taker_buy_ratio: float) -> tuple[float, float, int]:
        bid_vol = sum(q for _, q in self._bids[:10]) if self._bids else 0.0
        ask_vol = sum(q for _, q in self._asks[:10]) if self._asks else 0.0
        prev_bid = self._prev_bid_vol10 if self._prev_bid_vol10 > 0 else bid_vol
        prev_ask = self._prev_ask_vol10 if self._prev_ask_vol10 > 0 else ask_vol
        queue_collapse = float(np.clip(max((prev_bid - bid_vol) / (prev_bid + 1e-8), (prev_ask - ask_vol) / (prev_ask + 1e-8), 0.0), 0.0, 1.0))
        flow_sign = 2.0 * taker_buy_ratio - 1.0
        book_sign = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        # Divergence가 클수록(시장가 체결 방향 vs 호가 방향 충돌) 흡수 점수를 높게 부여
        absorption = float(np.clip(abs(flow_sign - book_sign) / 2.0, 0.0, 1.0))
        queue_bias = int(np.sign(flow_sign)) if abs(flow_sign) > 0.10 else 0
        self._prev_bid_vol10, self._prev_ask_vol10 = bid_vol, ask_vol
        return queue_collapse, absorption, queue_bias

    def _compute_shadow_regime(self, eai: float, oi_delta_pct: float) -> tuple[str, float]:
        funding = self._fund_rate
        if eai > self.eai_threshold and abs(funding) > 0.0008:
            return "squeeze", float(np.clip((eai / max(self.eai_threshold, 1e-6)) * min(abs(funding) / 0.002, 1.0), 0.0, 1.0))
        if abs(oi_delta_pct) > 0.01 and eai < 1.2:
            return "trend", float(np.clip(abs(oi_delta_pct) / 0.03, 0.0, 1.0))
        return "normal", float(np.clip(1.0 - min(eai / max(self.eai_threshold, 1e-6), 1.0), 0.0, 1.0))

    def _detect_spoofing(self, obi: float, taker_buy_ratio: float) -> tuple[float, int]:
        wall_th, buy_th, sell_th = abs(self.obi_wall_th), self.taker_buy_th, 1.0 - self.taker_buy_th
        if obi <= -wall_th and taker_buy_ratio >= buy_th:
            return float(np.clip((abs(obi) - wall_th) * (taker_buy_ratio - buy_th) * 10.0, 0.0, 1.0)), 1
        elif obi >= wall_th and taker_buy_ratio <= sell_th:
            return float(np.clip((obi - wall_th) * (sell_th - taker_buy_ratio) * 10.0, 0.0, 1.0)), -1
        return 0.0, 0

    def compute_signal(self) -> dict:
        now_ts = time.time()
        depth_age_sec = (now_ts - self._depth_last_msg_ts) if self._depth_last_msg_ts > 0 else None
        trade_age_sec = (now_ts - self._trade_last_msg_ts) if self._trade_last_msg_ts > 0 else None
        poll_age_sec = (now_ts - self._poll_last_msg_ts) if self._poll_last_msg_ts > 0 else None
        depth_stale = (depth_age_sec is None or depth_age_sec > 10.0 or not self._depth_connected)
        trade_stale = (trade_age_sec is None or trade_age_sec > 30.0 or not self._trade_connected)
        poll_stale = (poll_age_sec is None or poll_age_sec > 130.0 or not self._poll_connected)

        obi = self._compute_obi()
        (
            nif_whale,
            nif_retail,
            taker_buy_ratio_raw,
            recent_trade_count_5m,
            recent_trade_notional_5m,
            recent_whale_count_5m,
        ) = self._compute_nif_and_taker(300)
        valid_taker_flow = bool((not trade_stale) and recent_trade_count_5m > 0)
        valid_nif = bool((not trade_stale) and recent_whale_count_5m > 0)
        # 결측/스테일 체결 흐름을 0.0 매도 압력으로 위장하지 않도록 중립값을 사용한다.
        taker_buy_ratio = float(taker_buy_ratio_raw if valid_taker_flow else 0.5)
        eai = self._compute_eai()
        spoofing_score, spoof_bias = self._detect_spoofing(obi, taker_buy_ratio)
        # 5분(300초) 롤링 OI 변화율: NIF(300초)와 타임프레임 동기화
        now_ms = int(time.time() * 1000)
        cutoff_5m = now_ms - 5 * 60 * 1000
        valid_ois = [v for ts, v in self._oi_obs if ts >= cutoff_5m]
        if len(valid_ois) >= 2:
            oi_delta_pct = float((valid_ois[-1] - valid_ois[0]) / (abs(valid_ois[0]) + 1e-8))
        else:
            oi_delta_pct = float(self._cached.get("oi_delta_pct", 0.0))

        # 고래 거래량 10초 누적 -> 5분 버킷 누적
        whale_buy_10s_usd, whale_sell_10s_usd = self._compute_whale_flow_window(10)
        whale_flow_10s_ratio = float((whale_buy_10s_usd - whale_sell_10s_usd) / (whale_buy_10s_usd + whale_sell_10s_usd + 1e-8))
        bucket_5m = int(time.time() // 300)
        if bucket_5m != self._whale_flow_5m_bucket:
            self._whale_flow_5m_bucket = bucket_5m
            self._whale_buy_cum_5m_usd = 0.0
            self._whale_sell_cum_5m_usd = 0.0
        self._whale_buy_cum_5m_usd += whale_buy_10s_usd
        self._whale_sell_cum_5m_usd += whale_sell_10s_usd
        whale_flow_cum_5m_ratio = float(
            (self._whale_buy_cum_5m_usd - self._whale_sell_cum_5m_usd)
            / (self._whale_buy_cum_5m_usd + self._whale_sell_cum_5m_usd + 1e-8)
        )

        # 💡 MFT 연산 (새로 추가된 데이터 통계 처리)
        price_change_30m, price_volatility_30m = 0.0, 0.0
        price_breakout_60m, price_breakdown_60m = False, False
        if len(self._price_hist) >= 30:
            p_now = self._price_hist[-1]
            p_30m = self._price_hist[-30]
            price_change_30m = (p_now - p_30m) / (p_30m + 1e-8)
            p_30m_arr = list(self._price_hist)[-30:]
            price_volatility_30m = (max(p_30m_arr) - min(p_30m_arr)) / (p_now + 1e-8)
            
        if len(self._price_hist) >= 15:
            p_now = self._price_hist[-1]
            p_past = list(self._price_hist)[:-5] # 직전 5분 제외한 과거 55분
            if p_past:
                price_breakout_60m = p_now > max(p_past)
                price_breakdown_60m = p_now < min(p_past)

        # 15분 VWAP 이격도: (현재가 - VWAP) / VWAP
        vwap_gap_15m = 0.0
        if self._trade_notional_qty and self._price_hist:
            cutoff = int(time.time() * 1000) - 15 * 60 * 1000
            notional = 0.0
            qty_sum = 0.0
            for t_ms, ntl, qty in reversed(self._trade_notional_qty):
                if t_ms < cutoff:
                    break
                notional += float(ntl)
                qty_sum += float(qty)
            if qty_sum > 0:
                vwap_15m = notional / qty_sum
                p_now = float(self._price_hist[-1])
                vwap_gap_15m = float((p_now - vwap_15m) / (vwap_15m + 1e-8))

        nif_30m = list(self._nif_hist)[-30:]
        nif_whale_sum_30m = sum(nif_30m) if nif_30m else 0.0
        nif_whale_avg_30m = float(np.mean(nif_30m)) if nif_30m else 0.0
        nif_whale_std_30m = float(np.std(nif_30m)) if nif_30m else 0.0

        abs_30m = list(self._abs_hist)[-30:]
        absorption_avg_30m = float(np.mean(abs_30m)) if abs_30m else 0.0

        bias_30m = list(self._bias_hist)[-30:]
        bias_avg_30m = float(np.mean(bias_30m)) if bias_30m else 0.0

        tox_30m = list(self._tox_hist)[-30:]
        toxicity_avg_30m = float(np.mean(tox_30m)) if tox_30m else 0.0

        eai_delta_15m = (self._eai_hist[-1] - self._eai_hist[-15]) if len(self._eai_hist) >= 15 else 0.0

        # 고래 의도 판별: NIF(300초 롤링) + OIΔ5m 분해
        # 히스테리시스 적용: 진입(0.12) / 이탈(0.08) 분리로 오락가락 감소
        flow = float(nif_whale)
        flow_th = max(0.05, self.whale_intent_nif_th)
        entry_th = max(self.whale_flow_entry_th, self.whale_flow_exit_th + 1e-6)
        exit_th = min(self.whale_flow_exit_th, entry_th - 1e-6)
        if self._whale_flow_state == 0:
            if flow >= entry_th:
                self._whale_flow_state = 1
            elif flow <= -entry_th:
                self._whale_flow_state = -1
        elif self._whale_flow_state == 1:
            if flow <= exit_th:
                self._whale_flow_state = 0
        elif self._whale_flow_state == -1:
            if flow >= -exit_th:
                self._whale_flow_state = 0
        flow_state = int(self._whale_flow_state)
        oi5 = float(oi_delta_pct)

        whale_short_build_ratio_30m = 0.0
        whale_long_close_ratio_30m = 0.0
        whale_sell_presence_ratio_30m = 0.0
        whale_sell_effective_ratio_30m = 0.0
        whale_long_build_ratio_30m = 0.0
        whale_short_cover_ratio_30m = 0.0
        whale_buy_presence_ratio_30m = 0.0
        whale_buy_effective_ratio_30m = 0.0
        intent_window_n = max(1, int(self.whale_intent_window_min))

        if flow_state > 0:
            whale_buy_presence_ratio_30m = 1.0
            if oi5 > self.whale_intent_oi_th:
                whale_long_build_ratio_30m = 1.0
                whale_buy_effective_ratio_30m = 1.0
                whale_position_bias_30m = "🟢 신규 롱 구축 우세"
            elif oi5 < -self.whale_intent_oi_th:
                whale_short_cover_ratio_30m = 1.0
                whale_buy_effective_ratio_30m = 1.0
                whale_position_bias_30m = "🔵 숏 커버링(항복) 우세"
            else:
                whale_position_bias_30m = "🟣 매수 우세(의도 불명)"
        elif flow_state < 0:
            whale_sell_presence_ratio_30m = 1.0
            if oi5 > self.whale_intent_oi_th:
                whale_short_build_ratio_30m = 1.0
                whale_sell_effective_ratio_30m = 1.0
                whale_position_bias_30m = "🔴 신규 숏 구축 우세"
            elif oi5 < -self.whale_intent_oi_th:
                whale_long_close_ratio_30m = 1.0
                whale_sell_effective_ratio_30m = 1.0
                whale_position_bias_30m = "🟡 기존 롱 청산 우세"
            else:
                whale_position_bias_30m = "🟣 매도 우세(의도 불명)"
        else:
            whale_position_bias_30m = "관망/중립"

        # 고래 현재 포지션 추정(연속 점수): flow + OIΔ5m 결합
        flow_strength = float(np.clip(abs(flow) / (flow_th + 1e-8), 0.0, 1.5))
        oi_strength = float(np.clip(abs(oi5) / (self.whale_intent_oi_th + 1e-8), 0.0, 1.5))
        sign_flow = 1.0 if flow >= 0 else -1.0
        # OI가 증가하면 flow 방향 포지션 "신규 구축", 감소면 flow 방향 포지션 "축소/청산"
        # 신규 구축의 정보력을 더 크게 반영
        oi_dir_weight = 1.0 if oi5 > self.whale_intent_oi_th else (-0.35 if oi5 < -self.whale_intent_oi_th else 0.0)
        pos_score = float(np.clip((0.7 * sign_flow * min(flow_strength, 1.0)) + (0.3 * sign_flow * oi_dir_weight * min(oi_strength, 1.0)), -1.0, 1.0))
        if pos_score >= self.whale_pos_long_short_th:
            whale_position_estimate = "LONG"
        elif pos_score <= -self.whale_pos_long_short_th:
            whale_position_estimate = "SHORT"
        else:
            whale_position_estimate = "NEUTRAL"
        whale_position_confidence = int(np.clip(abs(pos_score) * 100.0, 0.0, 99.0))

        nif_bias = -1 if nif_whale < -0.30 and nif_retail > 0.10 else (1 if nif_whale > 0.30 and nif_retail < -0.10 else 0)
        eai_bias = -1 if eai > self.eai_threshold and self._fund_rate > 0.0010 else (1 if eai > self.eai_threshold and self._fund_rate < -0.0010 else 0)

        kelly_mult = 1.0
        if nif_bias == -1: kelly_mult *= 0.40
        if spoofing_score > 0.3:
            # 스푸핑 방향과 최종 신호가 일치할 때만 증폭, 불일치면 방어적으로 축소
            predicted_bias = spoof_bias + nif_bias + eai_bias
            if predicted_bias != 0 and np.sign(predicted_bias) == np.sign(spoof_bias):
                kelly_mult *= 1.20
            else:
                kelly_mult *= 0.70
        if eai > self.eai_threshold: kelly_mult *= 1.30
        kelly_mult = float(np.clip(kelly_mult, 0.30, 2.0))

        raw_bias = spoof_bias + nif_bias + eai_bias
        signal_bias = int(np.sign(raw_bias)) if raw_bias != 0 else 0

        shadow_toxicity_score, shadow_toxicity_regime = self._compute_shadow_toxicity(obi, taker_buy_ratio)
        shadow_queue_collapse, shadow_absorption_score, shadow_queue_bias = self._compute_shadow_queue_absorption(taker_buy_ratio)
        shadow_regime_tag, shadow_regime_conf = self._compute_shadow_regime(eai, oi_delta_pct)
        
        self._shadow_state = {
            "shadow_toxicity_score": shadow_toxicity_score,
            "shadow_toxicity_regime": shadow_toxicity_regime,
            "shadow_queue_collapse": shadow_queue_collapse,
            "shadow_absorption_score": shadow_absorption_score,
            "shadow_queue_bias": shadow_queue_bias,
            "shadow_regime_tag": shadow_regime_tag,
            "shadow_regime_conf": shadow_regime_conf,
        }

        out = {
            "obi": float(obi), "taker_buy_ratio": float(taker_buy_ratio), "spoofing_score": float(spoofing_score),
            "spoofing_bias": int(spoof_bias), "nif_whale": float(nif_whale), "nif_retail": float(nif_retail),
            "nif_bias": int(nif_bias), "eai": float(eai), "eai_bias": int(eai_bias),
            "oi_delta_pct": float(oi_delta_pct), "funding_rate": float(self._fund_rate),
            "mark_price": float(self._price_obs[-1][1]) if self._price_obs else 0.0,
            "oi_delta_cum_5m": float(self._cached.get("oi_delta_cum_5m", 0.0)),
            "oi_delta_cum_5m_bucket_start_ts": int(self._cached.get("oi_delta_cum_5m_bucket_start_ts", 0)),
            "whale_flow_10s_ratio": float(whale_flow_10s_ratio),
            "whale_buy_10s_usd": float(whale_buy_10s_usd),
            "whale_sell_10s_usd": float(whale_sell_10s_usd),
            "whale_flow_cum_5m_ratio": float(whale_flow_cum_5m_ratio),
            "whale_buy_cum_5m_usd": float(self._whale_buy_cum_5m_usd),
            "whale_sell_cum_5m_usd": float(self._whale_sell_cum_5m_usd),
            "whale_flow_cum_5m_bucket_start_ts": int(bucket_5m * 300),
            "kelly_mult": float(kelly_mult), "signal_bias": int(signal_bias),
            
            # MFT 메트릭 반환
            "price_change_30m": float(price_change_30m),
            "price_volatility_30m": float(price_volatility_30m),
            "vwap_gap_15m": float(vwap_gap_15m),
            "price_breakout_60m": bool(price_breakout_60m),
            "price_breakdown_60m": bool(price_breakdown_60m),
            "nif_whale_sum_30m": float(nif_whale_sum_30m),
            "nif_whale_avg_30m": float(nif_whale_avg_30m),
            "nif_whale_std_30m": float(nif_whale_std_30m),
            "absorption_avg_30m": float(absorption_avg_30m),
            "bias_avg_30m": float(bias_avg_30m),
            "toxicity_avg_30m": float(toxicity_avg_30m),
            "eai_delta_15m": float(eai_delta_15m),
            "whale_short_build_ratio_30m": float(whale_short_build_ratio_30m),
            "whale_long_close_ratio_30m": float(whale_long_close_ratio_30m),
            "whale_sell_presence_ratio_30m": float(whale_sell_presence_ratio_30m),
            "whale_sell_effective_ratio_30m": float(whale_sell_effective_ratio_30m),
            "whale_long_build_ratio_30m": float(whale_long_build_ratio_30m),
            "whale_short_cover_ratio_30m": float(whale_short_cover_ratio_30m),
            "whale_buy_presence_ratio_30m": float(whale_buy_presence_ratio_30m),
            "whale_buy_effective_ratio_30m": float(whale_buy_effective_ratio_30m),
            "whale_position_bias_30m": str(whale_position_bias_30m),
            "whale_position_window_min": int(intent_window_n),
            "whale_position_estimate": str(whale_position_estimate),
            "whale_position_confidence": int(whale_position_confidence),
            "whale_position_score": float(pos_score),
        }
        ws_stale = bool(depth_stale or trade_stale or poll_stale)
        out["data_stale"] = bool(ws_stale)
        out["depth_connected"] = bool(self._depth_connected)
        out["trade_connected"] = bool(self._trade_connected)
        out["poll_connected"] = bool(self._poll_connected)
        out["depth_age_sec"] = float(depth_age_sec) if depth_age_sec is not None else None
        out["trade_age_sec"] = float(trade_age_sec) if trade_age_sec is not None else None
        out["poll_age_sec"] = float(poll_age_sec) if poll_age_sec is not None else None
        out["recent_trade_count_5m"] = int(recent_trade_count_5m)
        out["recent_trade_notional_5m"] = float(recent_trade_notional_5m)
        out["recent_whale_count_5m"] = int(recent_whale_count_5m)
        out["valid_taker_flow"] = bool(valid_taker_flow)
        out["valid_nif"] = bool(valid_nif)
        self._warmup_30m_ready = self._is_30m_ready()
        out["warmup_30m_ready"] = bool(self._warmup_30m_ready)
        out["warmup_price_samples"] = int(min(len(self._price_hist), 30))
        out["warmup_nif_samples"] = int(min(len(self._nif_hist), 30))
        out["warmup_abs_samples"] = int(min(len(self._abs_hist), 30))
        out["warmup_tox_samples"] = int(min(len(self._tox_hist), 30))
        out["warmup_bias_samples"] = int(min(len(self._bias_hist), 30))
        out["warmup_bootstrap_rows"] = int(self._bootstrap_loaded_rows)
        out.update(self._shadow_state)
        return out

    def _is_30m_ready(self) -> bool:
        return (
            len(self._price_hist) >= 30
            and len(self._nif_hist) >= 30
            and len(self._abs_hist) >= 30
            and len(self._tox_hist) >= 30
            and len(self._bias_hist) >= 30
        )

    async def _scan_loop(self) -> None:
        last_saved_minute = -1
        while self._running:
            now = time.time()
            await asyncio.sleep(10.0 - (now % 10.0) + 0.05)
            if not self._running: break
            try:
                sig = self.compute_signal()
                self._cached = sig
                logger.info("%s", self.status_line())

                dt_now = datetime.now()
                if dt_now.minute != last_saved_minute and dt_now.second < 15:
                    last_saved_minute = dt_now.minute
                    
                    # 💡 매분 정각마다 MFT 연산용 덱(Deque)에 데이터 추가
                    self._nif_hist.append(sig["nif_whale"])
                    self._oi_delta_hist.append(sig["oi_delta_pct"])
                    self._abs_hist.append(sig["shadow_absorption_score"])
                    self._tox_hist.append(sig["shadow_toxicity_score"])
                    self._bias_hist.append(sig["shadow_queue_bias"])
                    self._eai_hist.append(sig["eai"])

                    from datetime import timezone, timedelta
                    bucket_ts = (dt_now - timedelta(minutes=1)).replace(second=0, microsecond=0)
                    await asyncio.get_running_loop().run_in_executor(None, self._db_insert, bucket_ts, sig)
            except Exception as e:
                logger.warning("MS scan_loop 오류: %s", e)

    def start(self) -> None:
        if not self.enabled: return
        self._db_init()
        self._running  = True
        self._depth_task = asyncio.create_task(self._depth_loop())
        self._trade_task = asyncio.create_task(self._trade_loop())
        self._poll_task  = asyncio.create_task(self._poll_loop())
        self._scan_task  = asyncio.create_task(self._scan_loop())

    def stop(self) -> None:
        self._running = False
        for t in (self._depth_task, self._trade_task, self._poll_task, self._scan_task):
            if t and not t.done(): t.cancel()

    def get_kelly_multiplier(self) -> float:
        if not self.dsac_intercept_enabled: return 1.0
        return float(self._cached.get("kelly_mult", 1.0))

    def get_signal(self) -> dict:
        out = self._cached.copy() if self._cached else {}
        if out.get("data_stale", False):
            out["signal_bias"] = 0
            out["kelly_mult"] = 1.0
        if not self.dsac_intercept_enabled and out:
            out["signal_bias"] = 0
            out["kelly_mult"] = 1.0
        return out

    def status_line(self) -> str:
        if not self.enabled: return "MS: 비활성화됨"
        if not self._cached: return "MS: 데이터 수집 중..."
        stale = bool(self._cached.get("data_stale", False))
        stale_txt = "STALE" if stale else "LIVE"
        if not bool(self._cached.get("warmup_30m_ready", False)):
            w_p = int(self._cached.get("warmup_price_samples", 0))
            w_n = int(self._cached.get("warmup_nif_samples", 0))
            return (
                f"📡 MS 1분 스캔({stale_txt}) | 30분 안정화 누적중 "
                f"(price {w_p}/30, nif {w_n}/30) | 펀딩비: {self._fund_rate:+.5f} | EAI: {self._cached.get('eai', 0):.1f}"
            )
        return f"📡 MS 1분 스캔 완료({stale_txt}) | 펀딩비: {self._fund_rate:+.5f} | EAI: {self._cached.get('eai', 0):.1f}"
