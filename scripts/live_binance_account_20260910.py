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
DEFAULT_TRADE_LIMIT = 1000  # Binance max; 잘리면 payload.trades_truncated=True


def _sign(params: dict[str, Any], secret: str, offset_ms: int = 0) -> str:
    stamp = int(time.time() * 1000) + offset_ms
    query = urlencode({**params, "recvWindow": RECV_WINDOW_MS, "timestamp": stamp})
    return f"{query}&signature={hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()}"


async def _clock_offset(session) -> int:
    """Binance rejects any request whose timestamp runs even ~1s AHEAD of its own clock, no matter
    how large recvWindow is. WSL2 clocks drift (2026-09-10: measured 1000ms ahead), so anchor every
    signature to /fapi/v1/time instead of the local clock. Unsigned, weight 1."""
    try:
        async with session.get(f"{FAPI}/fapi/v1/time") as response:
            return int((await response.json())["serverTime"]) - int(time.time() * 1000)
    except Exception:
        return 0


def _split_flips(fills: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Split any fill that closes the open position AND opens the opposite one in one go.

    One-way mode allows a reversal in a single order (short 2 -> BUY 5 -> long 3), and such a fill
    steps over net==0 instead of landing on it. Without this the fold never ends the trip and merges
    every later trade into one bogus round trip -- 2026-09-10 first run reported the live ETH LONG
    as a 98-fill SHORT. realizedPnl belongs entirely to the closing half; commission is prorated.
    """
    net = 0.0
    for fill in fills:
        qty = float(fill["qty"])
        direction = 1.0 if fill["side"] == "BUY" else -1.0
        closing = min(qty, abs(net)) if net * direction < 0 else 0.0
        # 양 끝을 QTY_EPS로 막는다: 0.3-0.1=0.19999999999999998 같은 잔여분 때문에 완전 청산이
        # "거의 청산 + 1e-17 신규진입"으로 쪼개져 수량 0짜리 유령 왕복이 생겼다(2026-09-10 실계좌 5건).
        if QTY_EPS < closing < qty - QTY_EPS:
            for part, realized in ((closing, fill["realizedPnl"]), (qty - closing, "0")):
                yield {**fill, "qty": str(part), "realizedPnl": realized,
                       "commission": str(float(fill["commission"]) * part / qty)}
        else:
            yield fill
        net += qty * direction
        if abs(net) < QTY_EPS:
            net = 0.0  # 잔여 1e-16이 다음 체결을 가짜 뒤집기로 쪼개 유령 왕복을 만든다


def round_trips(fills: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold userTrades fills into round trips: a trip opens when net position leaves 0 and closes
    when it returns to 0. realizedPnl/commission are summed over every fill in between, so a
    scaled-in or partially-closed position still reports one entry time and one exit time.

    Folded per (symbol, positionSide): this account runs in HEDGE mode, where a LONG and a SHORT
    position are open at the same time and netting them together is meaningless (2026-09-10: doing
    that reported the live ETH LONG as a 98-fill SHORT). "BOTH" is one-way mode, where a single
    fill can flip the position instead -- see _split_flips.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for fill in sorted(fills, key=lambda f: (int(f["time"]), int(f["id"]))):
        groups.setdefault((fill["symbol"], fill.get("positionSide", "BOTH")), []).append(fill)
    trips: list[dict[str, Any]] = []
    for group in groups.values():
        trips.extend(_fold(group))
    trips.sort(key=lambda t: t["entry_time"])
    return trips


def _fold(fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One (symbol, positionSide) stream, oldest first. BUY=+qty / SELL=-qty throughout: a hedge
    LONG stream stays >=0 and a hedge SHORT stream stays <=0, so the same zero-crossing test ends
    a trip in every mode."""
    trips: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    net = 0.0
    for fill in _split_flips(fills):
        qty = float(fill["qty"])
        price = float(fill["price"])
        signed_qty = qty * (1.0 if fill["side"] == "BUY" else -1.0)
        if current is None:
            current = {
                "symbol": fill["symbol"],
                "side": "LONG" if signed_qty > 0 else "SHORT",
                "entry_time": int(fill["time"]),
                "max_qty": 0.0,
                "qty_in": 0.0, "qty_out": 0.0, "_in": 0.0, "_out": 0.0,
                "realized_pnl": 0.0,
                "commission": 0.0,
                "fills": 0,
                "exit_time": None,
                "exit_price": None,
                "closed": False,
            }
        # 진입가/청산가는 **각 다리의 VWAP** 이다. 2026-09-12 이전에는 첫 체결가/마지막 체결가를
        # 그대로 실었는데, 물타기나 분할청산이 섞이면 그 둘은 그 왕복의 평균단가가 아니다 --
        # 실계좌 19왕복 중 6건에서 (청산가−진입가)×방향 의 부호가 realizedPnl 과 어긋났다.
        # 포지션을 **키우는** 체결이 진입 다리, **줄이는** 체결이 청산 다리다(_split_flips 가
        # 0을 넘는 체결을 미리 쪼개므로 한 체결이 양쪽에 걸치지 않는다).
        opening = net == 0.0 or (signed_qty > 0) == (net > 0)
        if opening:
            current["qty_in"] += qty; current["_in"] += price * qty
        else:
            current["qty_out"] += qty; current["_out"] += price * qty
        net += signed_qty
        current["max_qty"] = max(current["max_qty"], abs(net))
        current["realized_pnl"] += float(fill["realizedPnl"])
        current["commission"] += float(fill["commission"])
        current["fills"] += 1
        if abs(net) < QTY_EPS:
            current.update(exit_time=int(fill["time"]), closed=True)
            trips.append(_finish(current))
            current, net = None, 0.0
    if current is not None:
        trips.append(_finish(current))
    return trips


def _finish(trip: dict[str, Any]) -> dict[str, Any]:
    """VWAP 을 확정하고 **자기검증 값**을 싣는다.

    완전히 닫힌 왕복은 진입 수량 == 청산 수량이므로 회계 항등식이 성립한다:
        realizedPnl 합 = (청산VWAP − 진입VWAP) × 수량 × 방향부호
    (펀딩은 realizedPnl 이 아니라 income 에 따로 잡히므로 이 식에 안 들어간다.)
    어긋난 폭을 `pnl_check_bp` 로 남겨 두면, 원장을 읽는 쪽이 그 줄을 믿어도 되는지 스스로 안다.
    """
    trip["entry_price"] = trip["_in"] / trip["qty_in"] if trip["qty_in"] else None
    trip["exit_price"] = trip["_out"] / trip["qty_out"] if trip["qty_out"] else None
    trip["net_pnl"] = trip["realized_pnl"] - trip["commission"]
    trip["price_basis"] = "leg_vwap"
    sign = 1.0 if trip["side"] == "LONG" else -1.0
    if trip["closed"] and trip["entry_price"] and trip["exit_price"] and trip["qty_out"]:
        implied = (trip["exit_price"] - trip["entry_price"]) * trip["qty_out"] * sign
        notional = trip["entry_price"] * trip["qty_out"]
        trip["pnl_check_bp"] = round((trip["realized_pnl"] - implied) / notional * 1e4, 4) if notional else None
    else:
        trip["pnl_check_bp"] = None
    for k in ("_in", "_out"):
        trip.pop(k, None)
    return trip


async def _get(session, path: str, params: dict[str, Any], key: str, secret: str, offset_ms: int = 0) -> Any:
    """Signed GET. Returns the decoded JSON, or {"__error__": ...} rather than raising -- a revoked
    key or a futures-disabled key must degrade the panel, never take the whole dashboard down."""
    url = f"{FAPI}{path}?{_sign(params, secret, offset_ms)}"
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

    offset = await _clock_offset(session)
    balance = await _get(session, "/fapi/v2/account", {}, key, secret, offset)
    if "__error__" in balance:
        error = balance["__error__"]
        # 힌트는 실제로 그 오류일 때만. 예전엔 모든 실패에 "선물 권한 확인"을 붙여
        # 시계 드리프트(-1021)까지 권한 문제로 오인하게 만들었다.
        hint = ("API 키에 Futures 읽기 권한(Enable Futures)이 켜져 있는지, 그리고 IP 접근 제한에"
                " 이 서버 주소가 들어 있는지 확인하세요." if "-2015" in error or "permissions" in error
                else "서버 시계가 바이낸스와 어긋났습니다(NTP 동기화 확인)." if "-1021" in error or "Timestamp" in error
                else None)
        return {"ok": False, "error": error, **({"hint": hint} if hint else {})}

    risk = await _get(session, "/fapi/v2/positionRisk", {}, key, secret, offset)
    positions = [] if isinstance(risk, dict) else [
        {
            "symbol": p["symbol"],
            "side": p.get("positionSide") if p.get("positionSide") in ("LONG", "SHORT")
                    else ("LONG" if float(p["positionAmt"]) > 0 else "SHORT"),
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
    # 심볼별 설정 레버리지. **포지션이 없어도** 읽을 수 있어야 한다 -- 위 목록은 수량 0 을
    # 걸러내므로, 진입 미리보기가 거기서 레버리지를 찾으면 «증거금 0» 이 나온다(2026-09-13 실측).
    leverage_by_symbol = {} if isinstance(risk, dict) else {
        p["symbol"]: float(p.get("leverage") or 0.0) for p in risk if p.get("leverage")}

    trips: list[dict[str, Any]] = []
    truncated: list[str] = []
    for symbol in symbols:
        fills = await _get(session, "/fapi/v1/userTrades", {"symbol": symbol, "limit": trade_limit}, key, secret, offset)
        if isinstance(fills, dict) or not fills:
            continue
        # 정확히 limit개면 그 앞이 잘렸다는 뜻 -- 창 밖에서 열린 포지션은 중간부터 접히므로
        # 가장 오래된 왕복 하나는 진입가/방향이 틀릴 수 있다. 숨기지 말고 알린다.
        if len(fills) >= trade_limit:
            truncated.append(symbol)
        trips.extend(round_trips(fills))
    for trip in trips:
        trip["entry_at"] = _iso(trip["entry_time"])
        trip["exit_at"] = _iso(trip["exit_time"])
    trips.sort(key=lambda t: t["entry_time"], reverse=True)

    # An open position's entry time: the still-open round trip for that symbol knows the first fill,
    # which positionRisk's updateTime does not (that moves on every scale-in and funding settlement).
    open_entry = {(t["symbol"], t["side"]): t["entry_at"] for t in trips if not t["closed"]}
    for position in positions:
        position["entry_at"] = open_entry.get((position["symbol"], position["side"])) or position["updated_at"]

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "balance": {
            "wallet": float(balance["totalWalletBalance"]),
            # 🔴`margin` 은 **순자산**(지갑+미실현)이지 사용 증거금이 아니다. 2026-09-11 에
            # 대시보드가 이걸 "증거금 사용"으로 읽어 98.6%(실제 39.6%)를 띄웠다.
            # 사용/유지 증거금은 별도 필드라 아래에 이름 그대로 싣는다.
            "margin": float(balance["totalMarginBalance"]),
            "initial_margin": float(balance["totalInitialMargin"]),
            "maint_margin": float(balance["totalMaintMargin"]),
            "available": float(balance["availableBalance"]),
            "unrealized": float(balance["totalUnrealizedProfit"]),
        },
        "positions": positions,
        "leverage_by_symbol": leverage_by_symbol,
        "trades": trips,
        "trades_truncated": truncated,
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
    # 2000 과 1900 에 한 개씩 담았으니 진입 VWAP 은 1950, 2100/2200 에 하나씩 풀었으니 청산 2150.
    # 첫 체결가(2000)를 쓰던 옛 판은 여기서 2000 을 줬고, 그래서 (청산−진입) 이 손익과 안 맞았다.
    assert closed["entry_price"] == 1950.0 and closed["exit_price"] == 2150.0, closed
    assert closed["qty_in"] == 2.0 and closed["qty_out"] == 2.0 and closed["max_qty"] == 2.0, closed
    assert abs(closed["realized_pnl"] - 400.0) < 1e-9 and abs(closed["net_pnl"] - 396.72) < 1e-9, closed
    # 회계 항등식: (2150 − 1950) × 2 = 400 = realizedPnl 합 → 어긋남 0
    assert abs(closed["pnl_check_bp"]) < 1e-6, closed
    assert closed["price_basis"] == "leg_vwap"
    assert not opened["closed"] and opened["side"] == "SHORT", opened
    assert opened["max_qty"] == 3.0 and opened["exit_time"] is None, opened
    assert round_trips([]) == []

    # 실계좌에서 실제로 나온 모양의 회귀 시험: 물타기로 담았다가 **첫 체결가보다 낮은 값에**
    # 다 풀었는데 손익은 양수다. 첫/마지막 체결가를 쓰던 옛 판은 LONG 인데 (청산 2459 < 진입 2489)
    # 라 부호가 손익과 어긋났다(2026-09-03 건). VWAP 으로 보면 진입 2429.9 < 청산 2459 로 맞는다.
    scaled = [
        {"symbol": "ETHUSDT", "id": 1, "time": 1000, "side": "BUY", "price": "2489.79", "qty": "1", "realizedPnl": "0", "commission": "0", "positionSide": "LONG"},
        {"symbol": "ETHUSDT", "id": 2, "time": 2000, "side": "BUY", "price": "2400.00", "qty": "2", "realizedPnl": "0", "commission": "0", "positionSide": "LONG"},
        {"symbol": "ETHUSDT", "id": 3, "time": 3000, "side": "SELL", "price": "2459.06", "qty": "3", "realizedPnl": "87.45", "commission": "0", "positionSide": "LONG"},
    ]
    t = round_trips(scaled)[0]
    assert abs(t["entry_price"] - 2429.93) < 0.01 and t["exit_price"] == 2459.06, t
    assert (t["exit_price"] - t["entry_price"]) > 0 and t["realized_pnl"] > 0, "부호가 손익과 같아야 한다"
    assert abs(t["pnl_check_bp"]) < 1.0, t          # 항등식 오차 1bp 이내

    # 한 체결로 숏 2 -> 롱 3 뒤집기. 쪼개지 않으면 두 거래가 한 왕복으로 뭉친다.
    flip = [
        {"symbol": "ETHUSDT", "id": 1, "time": 1000, "side": "SELL", "price": "2000", "qty": "2", "realizedPnl": "0", "commission": "1.6"},
        {"symbol": "ETHUSDT", "id": 2, "time": 2000, "side": "BUY", "price": "1900", "qty": "5", "realizedPnl": "200", "commission": "3.8"},
        {"symbol": "ETHUSDT", "id": 3, "time": 3000, "side": "SELL", "price": "2000", "qty": "3", "realizedPnl": "300", "commission": "2.4"},
    ]
    a, b = round_trips(flip)
    assert a["side"] == "SHORT" and (a["entry_time"], a["exit_time"]) == (1000, 2000), a
    assert abs(a["realized_pnl"] - 200.0) < 1e-9 and abs(a["commission"] - 3.12) < 1e-9, a
    assert b["side"] == "LONG" and (b["entry_time"], b["exit_time"]) == (2000, 3000), b
    assert b["max_qty"] == 3.0 and abs(b["realized_pnl"] - 300.0) < 1e-9, b
    # 헤지 모드: 롱과 숏이 동시에 열린다. 합치면 net이 0을 안 밟아 한 덩어리가 된다.
    hedge = [
        {"symbol": "ETHUSDT", "id": 1, "time": 1000, "side": "BUY", "price": "2000", "qty": "2", "realizedPnl": "0", "commission": "0", "positionSide": "LONG"},
        {"symbol": "ETHUSDT", "id": 2, "time": 1500, "side": "SELL", "price": "2000", "qty": "2", "realizedPnl": "0", "commission": "0", "positionSide": "SHORT"},
        {"symbol": "ETHUSDT", "id": 3, "time": 2000, "side": "SELL", "price": "2100", "qty": "2", "realizedPnl": "200", "commission": "0", "positionSide": "LONG"},
    ]
    h = round_trips(hedge)
    assert len(h) == 2, h
    long_trip = next(t for t in h if t["side"] == "LONG")
    short_trip = next(t for t in h if t["side"] == "SHORT")
    assert long_trip["closed"] and (long_trip["entry_time"], long_trip["exit_time"]) == (1000, 2000), long_trip
    assert not short_trip["closed"] and short_trip["max_qty"] == 2.0, short_trip
    # 부동소수 잔여분(0.3-0.1=0.19999999999999998)이 완전 청산을 가짜 뒤집기로 쪼개면 안 된다.
    residue = [
        {"symbol": "ETHUSDT", "id": 1, "time": 1000, "side": "BUY", "price": "2000", "qty": "0.3", "realizedPnl": "0", "commission": "0", "positionSide": "LONG"},
        {"symbol": "ETHUSDT", "id": 2, "time": 2000, "side": "SELL", "price": "2100", "qty": "0.1", "realizedPnl": "10", "commission": "0", "positionSide": "LONG"},
        {"symbol": "ETHUSDT", "id": 3, "time": 3000, "side": "SELL", "price": "2100", "qty": "0.2", "realizedPnl": "20", "commission": "0", "positionSide": "LONG"},
    ]
    r = round_trips(residue)
    assert len(r) == 1 and r[0]["closed"] and r[0]["exit_time"] == 3000, r
    assert not any(t["max_qty"] <= QTY_EPS for t in r), r
    print("self-check ok:", len(trips), "round trips, flip split into 2, hedge split into", len(h),
          ", residue folds into", len(r))


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
