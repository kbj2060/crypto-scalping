from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot_modules.duckdb_access import serialized_duckdb_access

logger = logging.getLogger("LiveBot")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def _depth_notional(levels: list[list[float]], n: int) -> tuple[float, float]:
    qty = 0.0
    notional = 0.0
    for price, amount, *_rest in levels[: max(0, int(n))]:
        p = _safe_float(price)
        a = _safe_float(amount)
        qty += a
        notional += p * a
    return float(qty), float(notional)


def _imbalance(bid_notional: float, ask_notional: float) -> float:
    denom = abs(bid_notional) + abs(ask_notional)
    if denom <= 1e-12:
        return 0.0
    return float((bid_notional - ask_notional) / denom)


class OrderBookRecorder:
    """Append-only L2 order book recorder for live/backtest execution parity audits.

    The alpha1.4 soft execution proxy used OHLCV/microstructure columns to infer
    maker-like fills. This recorder captures the actual Binance futures book
    around live decisions so later replays can validate maker/taker assumptions
    against L2 depth instead of candle proxies.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        path: str | Path | None = None,
        db_path: str | Path | None = None,
        table: str | None = None,
        depth: int | None = None,
        throttle_sec: float | None = None,
    ) -> None:
        self.enabled = _env_flag("ORDERBOOK_RECORDER_ENABLED", True) if enabled is None else bool(enabled)
        self.path = Path(path or os.getenv("ORDERBOOK_RECORDER_PATH", "data/live/orderbook_snapshots.jsonl"))
        self.db_path = Path(
            db_path
            or os.getenv("ORDERBOOK_RECORDER_DUCKDB_PATH")
            or os.getenv("QUANT_MICRO_DB_PATH", "data/live/microstructure.duckdb")
        )
        self.table = str(table or os.getenv("ORDERBOOK_RECORDER_TABLE", "orderbook_decision_snapshots"))
        self.storage = str(os.getenv("ORDERBOOK_RECORDER_STORAGE", "duckdb")).strip().lower()
        self.depth = int(depth if depth is not None else os.getenv("ORDERBOOK_RECORDER_DEPTH", "20"))
        self.throttle_sec = float(
            throttle_sec if throttle_sec is not None else os.getenv("ORDERBOOK_RECORDER_THROTTLE_SEC", "2.0")
        )
        self._last_write_monotonic = 0.0

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "storage": str(self.storage),
            "path": str(self.path),
            "db_path": str(self.db_path),
            "table": str(self.table),
            "depth": int(self.depth),
            "throttle_sec": float(self.throttle_sec),
        }

    @staticmethod
    def summarize_orderbook(orderbook: dict[str, Any], *, symbol: str, timestamp_kst: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        bids = [[_safe_float(x[0]), _safe_float(x[1])] for x in list(orderbook.get("bids") or []) if len(x) >= 2]
        asks = [[_safe_float(x[0]), _safe_float(x[1])] for x in list(orderbook.get("asks") or []) if len(x) >= 2]
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
        spread = max(0.0, best_ask - best_bid) if best_bid > 0 and best_ask > 0 else 0.0
        best_bid_qty = bids[0][1] if bids else 0.0
        best_ask_qty = asks[0][1] if asks else 0.0
        microprice = (
            (best_ask * best_bid_qty + best_bid * best_ask_qty) / max(best_bid_qty + best_ask_qty, 1e-12)
            if best_bid > 0 and best_ask > 0
            else 0.0
        )
        row: dict[str, Any] = {
            "schema_version": "orderbook_snapshot.v1",
            "recorded_at_kst": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "timestamp_kst": str(timestamp_kst),
            "exchange_timestamp": orderbook.get("timestamp"),
            "datetime": orderbook.get("datetime"),
            "symbol": str(symbol),
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "mid": float(mid),
            "spread": float(spread),
            "spread_bps": float(spread / mid * 10000.0) if mid > 0 else 0.0,
            "microprice": float(microprice),
            "microprice_edge_bps": float((microprice - mid) / mid * 10000.0) if mid > 0 else 0.0,
            "levels_bid": int(len(bids)),
            "levels_ask": int(len(asks)),
        }
        for n in (1, 5, 10, 20):
            b_qty, b_notional = _depth_notional(bids, n)
            a_qty, a_notional = _depth_notional(asks, n)
            row[f"bid_qty_{n}"] = b_qty
            row[f"ask_qty_{n}"] = a_qty
            row[f"bid_notional_{n}"] = b_notional
            row[f"ask_notional_{n}"] = a_notional
            row[f"imbalance_{n}"] = _imbalance(b_notional, a_notional)
        if context:
            row["context"] = dict(context)
        row["bids_json"] = json.dumps(bids, default=_json_default)
        row["asks_json"] = json.dumps(asks, default=_json_default)
        return row

    def _append_row(self, row: dict[str, Any]) -> None:
        if self.storage in {"duckdb", "both"}:
            self._append_row_duckdb(row)
        if self.storage in {"jsonl", "both"}:
            self._append_row_jsonl(row)

    def _append_row_jsonl(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    @serialized_duckdb_access(lambda self, *_args, **_kwargs: self.db_path)
    def _append_row_duckdb(self, row: dict[str, Any]) -> None:
        import duckdb

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self.db_path))
        try:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    recorded_at_kst TIMESTAMPTZ,
                    timestamp_kst VARCHAR,
                    exchange_timestamp BIGINT,
                    exchange_datetime VARCHAR,
                    symbol VARCHAR,
                    best_bid DOUBLE,
                    best_ask DOUBLE,
                    mid DOUBLE,
                    spread DOUBLE,
                    spread_bps DOUBLE,
                    microprice DOUBLE,
                    microprice_edge_bps DOUBLE,
                    levels_bid INTEGER,
                    levels_ask INTEGER,
                    bid_qty_1 DOUBLE,
                    ask_qty_1 DOUBLE,
                    bid_notional_1 DOUBLE,
                    ask_notional_1 DOUBLE,
                    imbalance_1 DOUBLE,
                    bid_qty_5 DOUBLE,
                    ask_qty_5 DOUBLE,
                    bid_notional_5 DOUBLE,
                    ask_notional_5 DOUBLE,
                    imbalance_5 DOUBLE,
                    bid_qty_10 DOUBLE,
                    ask_qty_10 DOUBLE,
                    bid_notional_10 DOUBLE,
                    ask_notional_10 DOUBLE,
                    imbalance_10 DOUBLE,
                    bid_qty_20 DOUBLE,
                    ask_qty_20 DOUBLE,
                    bid_notional_20 DOUBLE,
                    ask_notional_20 DOUBLE,
                    imbalance_20 DOUBLE,
                    context_json VARCHAR,
                    schema_version VARCHAR,
                    bids_json VARCHAR,
                    asks_json VARCHAR
                )
                """
            )
            # WS-E E1 migration: tables created before 2026-08-17 lack bids_json/asks_json.
            # ALTER TABLE ADD COLUMN IF NOT EXISTS backfills them as NULL on existing rows.
            existing_cols = {c[1] for c in con.execute(f"PRAGMA table_info('{self.table}')").fetchall()}
            for col in ("bids_json", "asks_json"):
                if col not in existing_cols:
                    con.execute(f"ALTER TABLE {self.table} ADD COLUMN {col} VARCHAR")
            con.execute(
                f"""
                INSERT INTO {self.table} (
                    recorded_at_kst, timestamp_kst, exchange_timestamp, exchange_datetime, symbol,
                    best_bid, best_ask, mid, spread, spread_bps, microprice, microprice_edge_bps,
                    levels_bid, levels_ask,
                    bid_qty_1, ask_qty_1, bid_notional_1, ask_notional_1, imbalance_1,
                    bid_qty_5, ask_qty_5, bid_notional_5, ask_notional_5, imbalance_5,
                    bid_qty_10, ask_qty_10, bid_notional_10, ask_notional_10, imbalance_10,
                    bid_qty_20, ask_qty_20, bid_notional_20, ask_notional_20, imbalance_20,
                    context_json, schema_version, bids_json, asks_json
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                [
                    row.get("recorded_at_kst"),
                    str(row.get("timestamp_kst", "")),
                    int(_safe_float(row.get("exchange_timestamp"), 0.0)) if row.get("exchange_timestamp") is not None else None,
                    str(row.get("datetime", "")),
                    str(row.get("symbol", "")),
                    _safe_float(row.get("best_bid")),
                    _safe_float(row.get("best_ask")),
                    _safe_float(row.get("mid")),
                    _safe_float(row.get("spread")),
                    _safe_float(row.get("spread_bps")),
                    _safe_float(row.get("microprice")),
                    _safe_float(row.get("microprice_edge_bps")),
                    int(_safe_float(row.get("levels_bid"), 0.0)),
                    int(_safe_float(row.get("levels_ask"), 0.0)),
                    _safe_float(row.get("bid_qty_1")),
                    _safe_float(row.get("ask_qty_1")),
                    _safe_float(row.get("bid_notional_1")),
                    _safe_float(row.get("ask_notional_1")),
                    _safe_float(row.get("imbalance_1")),
                    _safe_float(row.get("bid_qty_5")),
                    _safe_float(row.get("ask_qty_5")),
                    _safe_float(row.get("bid_notional_5")),
                    _safe_float(row.get("ask_notional_5")),
                    _safe_float(row.get("imbalance_5")),
                    _safe_float(row.get("bid_qty_10")),
                    _safe_float(row.get("ask_qty_10")),
                    _safe_float(row.get("bid_notional_10")),
                    _safe_float(row.get("ask_notional_10")),
                    _safe_float(row.get("imbalance_10")),
                    _safe_float(row.get("bid_qty_20")),
                    _safe_float(row.get("ask_qty_20")),
                    _safe_float(row.get("bid_notional_20")),
                    _safe_float(row.get("ask_notional_20")),
                    _safe_float(row.get("imbalance_20")),
                    json.dumps(row.get("context", {}) or {}, ensure_ascii=False, default=_json_default),
                    str(row.get("schema_version", "orderbook_snapshot.v1")),
                    str(row.get("bids_json", "[]")),
                    str(row.get("asks_json", "[]")),
                ],
            )
        finally:
            con.close()

    async def record_decision_snapshot(
        self,
        fetcher: Any,
        *,
        timestamp_kst: Any,
        context: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "recorded": False, "reason": "disabled"}
        now = asyncio.get_running_loop().time()
        if not force and now - self._last_write_monotonic < self.throttle_sec:
            return {"enabled": True, "recorded": False, "reason": "throttled", "path": str(self.path)}
        symbol = str(getattr(fetcher, "account_symbol", "") or "")
        raw_symbol = str(getattr(fetcher, "symbol", "") or "")
        if not symbol:
            symbol = raw_symbol
        try:
            exchange = getattr(fetcher, "exchange")
            try:
                orderbook = await fetcher._call_with_retry(
                    f"fetch_order_book[{symbol}]",
                    lambda: exchange.fetch_order_book(symbol, limit=int(self.depth)),
                )
            except Exception:
                orderbook = await fetcher._call_with_retry(
                    f"fetch_order_book[{raw_symbol}]",
                    lambda: exchange.fetch_order_book(raw_symbol, limit=int(self.depth)),
                )
            row = self.summarize_orderbook(orderbook, symbol=symbol, timestamp_kst=timestamp_kst, context=context)
            await asyncio.get_running_loop().run_in_executor(None, self._append_row, row)
            self._last_write_monotonic = now
            return {
                "enabled": True,
                "recorded": True,
                "storage": str(self.storage),
                "path": str(self.path),
                "db_path": str(self.db_path),
                "table": str(self.table),
                "symbol": symbol,
                "best_bid": row["best_bid"],
                "best_ask": row["best_ask"],
                "spread_bps": row["spread_bps"],
                "imbalance_5": row["imbalance_5"],
                "imbalance_10": row["imbalance_10"],
                "microprice_edge_bps": row["microprice_edge_bps"],
            }
        except Exception as exc:
            logger.warning("SYSTEM orderbook_snapshot=FAILED reason=%s", exc)
            return {"enabled": True, "recorded": False, "reason": str(exc), "path": str(self.path)}
