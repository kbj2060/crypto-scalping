"""Live Binance USD-M futures account readout for the dashboard (2026-09-10, user request).

Answers "지금 포지션 / 진입 시점 / 청산 시점 / 과거 데이터" straight from the exchange, which is the
only source that also sees manual trades -- data/live/trade_journal.jsonl only ever holds what
trading_bot.py itself decided, and that bot currently runs with account.enabled=false (paper).

Three signed GETs per refresh (weight 5 each): balances, open positions, and per-symbol fills.
The fills are folded into round trips locally, so entry/exit timestamps need no extra endpoint.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode

FAPI = "https://fapi.binance.com"
RECV_WINDOW_MS = 5000
QTY_EPS = 1e-9
DEFAULT_TRADE_LIMIT = 200


def _sign(params: dict[str, Any], secret: str) -> str:
    query = urlencode({**params, "recvWindow": RECV_WINDOW_MS, "timestamp": int(time.time() * 1000)})
    return f"{query}&signature={hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()}"


def round_trips(fills: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold userTrades fills into round trips: a trip opens when net position leaves 0 and closes
    when it returns to 0. realizedPnl/commission are summed over every fill in between, so a
    scaled-in or partially-closed position still reports one entry time and one exit time.

    ponytail: one-way mode only (BUY=+qty, SELL=-qty). Hedge mode holds LONG and SHORT open at the
    same time and would need a net per positionSide -- split on f["positionSide"] if the account
    ever switches (futures_change_position_mode).
    """
    trips: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    net = 0.0
    for fill in sorted(fills, key=lambda f: (int(f["time"]), int(f["id"]))):
        signed_qty = float(fill["qty"]) * (1.0 if fill["side"] == "BUY" else -1.0)
        if current is None:
            current = {
                "symbol": fill["symbol"],
                "side": "LONG" if signed_qty > 0 else "SHORT",
                "entry_time": int(fill["time"]),
                "entry_price": float(fill["price"]),
                "max_qty": 0.0,
                "realized_pnl": 0.0,
                "commission": 0.0,
                "fills": 0,
                "exit_time": None,
                "exit_price": None,
                "closed": False,
            }
        net += signed_qty
        current["max_qty"] = max(current["max_qty"], abs(net))
        current["realized_pnl"] += float(fill["realizedPnl"])
        current["commission"] += float(fill["commission"])
        current["fills"] += 1
        if abs(net) < QTY_EPS:
            current.update(exit_time=int(fill["time"]), exit_price=float(fill["price"]), closed=True)
            current["net_pnl"] = current["realized_pnl"] - current["commission"]
            trips.append(current)
            current, net = None, 0.0
    if current is not None:
        current["net_pnl"] = current["realized_pnl"] - current["commission"]
        trips.append(current)
    return trips


async def _get(session, path: str, params: dict[str, Any], key: str, secret: str) -> Any:
    """Signed GET. Returns the decoded JSON, or {"__error__": ...} rather than raising -- a revoked
    key or a futures-disabled key must degrade the panel, never take the whole dashboard down."""
    url = f"{FAPI}{path}?{_sign(params, secret)}"
    try:
        async with session.get(url, headers={"X-MBX-APIKEY": key}) as response:
            payload = await response.json()
            if response.status != 200:
                return {"__error__": f"{response.status} {payload.get('msg', payload)}"}
            return payload
    except Exception as exc:  # network/TLS/JSON -- same degradation as an API error
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def _iso(ms: Any) -> str | None:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).isoformat() if ms else None


async def fetch_account(session, symbols: Sequence[str], *, trade_limit: int = DEFAULT_TRADE_LIMIT) -> dict[str, Any]:
    key, secret = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_SECRET_KEY", "")
    if not (key and secret):
        return {"ok": False, "error": "BINANCE_API_KEY/BINANCE_SECRET_KEY가 .env에 없습니다."}

    balance = await _get(session, "/fapi/v2/account", {}, key, secret)
    if "__error__" in balance:
        return {"ok": False, "error": balance["__error__"],
                "hint": "API 키에 Futures 읽기 권한(Enable Futures)이 켜져 있는지 확인하세요."}

    risk = await _get(session, "/fapi/v2/positionRisk", {}, key, secret)
    positions = [] if isinstance(risk, dict) else [
        {
            "symbol": p["symbol"],
            "side": "LONG" if float(p["positionAmt"]) > 0 else "SHORT",
            "qty": abs(float(p["positionAmt"])),
            "entry_price": float(p["entryPrice"]),
            "mark_price": float(p["markPrice"]),
            "liquidation_price": float(p["liquidationPrice"]),
            "leverage": float(p["leverage"]),
            "notional": abs(float(p["notional"])),
            "unrealized_pnl": float(p["unRealizedProfit"]),
            "updated_at": _iso(p.get("updateTime")),
        }
        for p in risk if float(p["positionAmt"]) != 0.0
    ]

    trips: list[dict[str, Any]] = []
    for symbol in symbols:
        fills = await _get(session, "/fapi/v1/userTrades", {"symbol": symbol, "limit": trade_limit}, key, secret)
        if isinstance(fills, dict) or not fills:
            continue
        trips.extend(round_trips(fills))
    for trip in trips:
        trip["entry_at"] = _iso(trip["entry_time"])
        trip["exit_at"] = _iso(trip["exit_time"])
    trips.sort(key=lambda t: t["entry_time"], reverse=True)

    # An open position's entry time: the still-open round trip for that symbol knows the first fill,
    # which positionRisk's updateTime does not (that moves on every scale-in and funding settlement).
    open_entry = {t["symbol"]: t["entry_at"] for t in trips if not t["closed"]}
    for position in positions:
        position["entry_at"] = open_entry.get(position["symbol"]) or position["updated_at"]

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "balance": {
            "wallet": float(balance["totalWalletBalance"]),
            "margin": float(balance["totalMarginBalance"]),
            "available": float(balance["availableBalance"]),
            "unrealized": float(balance["totalUnrealizedProfit"]),
        },
        "positions": positions,
        "trades": trips,
    }


def _self_check() -> None:
    fills = [  # scale-in long, partial close, full close -- one round trip, then a short
        {"symbol": "ETHUSDT", "id": 1, "time": 1000, "side": "BUY", "price": "2000", "qty": "1", "realizedPnl": "0", "commission": "0.8"},
        {"symbol": "ETHUSDT", "id": 2, "time": 2000, "side": "BUY", "price": "1900", "qty": "1", "realizedPnl": "0", "commission": "0.76"},
        {"symbol": "ETHUSDT", "id": 3, "time": 3000, "side": "SELL", "price": "2100", "qty": "1", "realizedPnl": "150", "commission": "0.84"},
        {"symbol": "ETHUSDT", "id": 4, "time": 4000, "side": "SELL", "price": "2200", "qty": "1", "realizedPnl": "250", "commission": "0.88"},
        {"symbol": "ETHUSDT", "id": 5, "time": 5000, "side": "SELL", "price": "2200", "qty": "3", "realizedPnl": "0", "commission": "2.64"},
    ]
    trips = round_trips(fills)
    assert len(trips) == 2, trips
    closed, opened = trips
    assert closed["closed"] and closed["side"] == "LONG"
    assert (closed["entry_time"], closed["exit_time"]) == (1000, 4000), closed
    assert closed["entry_price"] == 2000.0 and closed["max_qty"] == 2.0, closed
    assert abs(closed["realized_pnl"] - 400.0) < 1e-9 and abs(closed["net_pnl"] - 396.72) < 1e-9, closed
    assert not opened["closed"] and opened["side"] == "SHORT", opened
    assert opened["max_qty"] == 3.0 and opened["exit_time"] is None, opened
    assert round_trips([]) == []
    print("self-check ok:", len(trips), "round trips")


if __name__ == "__main__":
    import asyncio
    import sys

    _self_check()
    if "--live" in sys.argv:
        import aiohttp
        from dotenv import load_dotenv

        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

        async def main() -> None:
            async with aiohttp.ClientSession() as session:
                result = await fetch_account(session, ["ETHUSDT", "BTCUSDT", "SOLUSDT", "XRPUSDT"])
            print(result if not result.get("ok") else
                  {"balance": result["balance"], "positions": result["positions"], "trades": len(result["trades"])})

        asyncio.run(main())
