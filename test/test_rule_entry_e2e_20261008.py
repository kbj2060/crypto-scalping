"""4분할 규칙 진입 종단 시험 (2026-10-08) -- 미리보기 → 제출 → 1분할 체결 → 3σ 손절 스탑 → 2~4분할 지정가.
    python -m pytest -q test/test_rule_entry_e2e_20261008.py

🔴실주문 0 이 계약이다(사용자 «수정 중에 실제 주문은 절대 하지 마»):
  ① 주문 서명 요청은 실행기 `signed` 를 가짜 거래소로 갈아끼운다(서버 import 경로가 전부 이 모듈 전역을 부른다).
  ② 공개 조회(ClientSession.get)는 바이낸스 주소를 가짜로 답한다.
  ③ 그 밖에 이 프로세스가 localhost 가 아닌 곳으로 HTTP 를 열려 하면 **즉시 예외**(_request 가드) -- 새 경로가 생겨도 새지 않는다.
  ④ API 키는 가짜 문자열이다(새어 나가도 거래소가 거부한다).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import aiohttp
from yarl import URL

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app  # noqa: E402

from dashboard import server  # noqa: E402
import scripts.live_manual_peg_execute_20260912 as ex  # noqa: E402

SYMBOL = server.MANUAL_EXEC_SYMBOLS["eth"]
FILTERS = {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0}
STATE = {"ok": True, "available": True, "generated_at": "2099-01-01T00:00:00+00:00", "price": 2500.0, "atr_pct": 0.001,
         "atr_pct_percentile": 0.5, "atr_pct_ref": 0.0012, "base_qty": 2.7, "vol_equivalent_qty": 3.0,
         "vol_equivalent_qty_formula": 3.0, "sizing_model": {"used": True}, "horizons": {}, "touch_prob": {},
         "risk_mae": {str(h): {s: {"safe_mae_pct": 3.0, "max_leverage": 33.3} for s in ("LONG", "SHORT")}
                      for h in (60, 120, 240, 480, 1440)}}


class FakeExchange:
    """주문 장부. 진입 지정가는 첫 조회에서 전량 체결, 분할 지정가(dbld)는 대기(NEW)."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []
        self.orders: dict[int, dict] = {}
        self.pos_qty = {"LONG": 0.0, "SHORT": 0.0}
        self.next_id = 100

    async def signed(self, session, method, path, params, key, secret, offset):
        assert key == "fake-key" and secret == "fake-secret", "가짜 키가 아니다"
        self.calls.append((method, path, dict(params)))
        if path == "/fapi/v2/positionRisk":
            return [{"symbol": SYMBOL, "positionSide": s, "positionAmt": str(q), "leverage": "20"}
                    for s, q in self.pos_qty.items()]
        if path == "/fapi/v1/leverage":
            return {"leverage": params["leverage"]}
        if path == "/fapi/v1/order" and method == "POST":
            self.next_id += 1
            self.orders[self.next_id] = {**params, "status": "NEW", "executedQty": "0"}
            return {"orderId": self.next_id, "status": "NEW", "executedQty": "0"}
        if path == "/fapi/v1/order" and method == "GET":
            o = self.orders[params["orderId"]]
            if not str(o.get("newClientOrderId") or "").startswith("dbld"):        # 진입 peg 만 체결
                o.update(status="FILLED", executedQty=str(o["quantity"]))
                self.pos_qty[o["positionSide"]] = float(o["quantity"])
            return {"orderId": params["orderId"], "status": o["status"], "executedQty": o["executedQty"]}
        if path == "/fapi/v1/order" and method == "DELETE":
            self.orders.get(params.get("orderId"), {})["status"] = "CANCELED"
            return {"orderId": params.get("orderId"), "status": "CANCELED"}
        if path in ("/fapi/v1/openOrders", "/fapi/v1/openAlgoOrders"):
            return []
        if path == "/fapi/v1/algoOrder" and method == "POST":
            return {"algoId": 7}
        raise AssertionError(f"예상 밖 거래소 호출 {method} {path}")

    def posts(self, path):
        return [p for m, pth, p in self.calls if m == "POST" and pth == path]


def _book(url, params):
    if "bookTicker" in str(url):
        return {"symbol": params.get("symbol"), "bidPrice": "2500.00", "askPrice": "2500.01"}
    if "klines" in str(url):
        n = int(params.get("limit") or 500)
        return [[1700000000000 + i * 60000, "2500", "2501", "2499", "2500", "10", 0, "25000", 100, "5", "12500", "0"] for i in range(n)]
    return None


class _Resp:
    def __init__(self, d):
        self._d, self.status, self.headers = d, (200 if d is not None else 404), {}

    async def json(self, **_k):
        return self._d

    async def text(self):
        return json.dumps(self._d)


class _Ctx:
    def __init__(self, r):
        self._r = r

    async def __aenter__(self):
        return self._r

    async def __aexit__(self, *_a):
        return False


_REAL_GET, _REAL_REQ = aiohttp.ClientSession.get, aiohttp.ClientSession._request


def _get(self, url, *a, params=None, **k):
    if "binance.com" in str(url):
        return _Ctx(_Resp(_book(url, params or {})))
    return _REAL_GET(self, url, *a, params=params, **k)


async def _guard(self, method, url, *a, **k):
    if URL(str(url)).host not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(f"🔴시험 중 외부 요청 차단: {method} {url}")
    return await _REAL_REQ(self, method, url, *a, **k)


@contextlib.contextmanager
def offline(fx: FakeExchange, positions=()):
    with tempfile.TemporaryDirectory() as tmp:
        live, dash = Path(tmp) / "live", Path(tmp) / "dashboard"
        live.mkdir(); dash.mkdir()
        (dash / "index.html").write_text("<html></html>", encoding="utf-8")
        (live / "eth_position_sizing_state.json").write_text(json.dumps(STATE), encoding="utf-8")
        account = {"ok": True, "balance": {"margin": 2600.0, "initial_margin": 0.0, "available": 2600.0, "unrealized": 0.0},
                   "positions": list(positions), "leverage_by_symbol": {SYMBOL: 20.0}, "trips": []}

        async def fake_account(*_a, **_k):
            return account

        async def zero(*_a, **_k):
            return 0
        with mock.patch.dict("os.environ", {"DASHBOARD_MANUAL_EXEC_ENABLED": "1",
                                            "BINANCE_API_KEY": "fake-key", "BINANCE_SECRET_KEY": "fake-secret"}), \
             mock.patch.object(server, "LIVE_DIR", live), mock.patch.object(server, "DASHBOARD_DIR", dash), \
             mock.patch.object(server, "POSITION_SIZING_STATE_PATH", live / "eth_position_sizing_state.json"), \
             mock.patch.object(server, "POSITION_SIZING_MAX_AGE_MIN", 10 ** 9), \
             mock.patch.object(server, "produce_account", fake_account, create=True), \
             mock.patch.object(server, "fetch_account", fake_account), \
             mock.patch.object(server, "load_filters", mock.AsyncMock(return_value=FILTERS)), \
             mock.patch.object(aiohttp.ClientSession, "get", _get), \
             mock.patch.object(aiohttp.ClientSession, "_request", _guard), \
             mock.patch.object(ex, "signed", fx.signed), mock.patch.object(ex, "_clock_offset", zero), \
             mock.patch.object(ex, "POLL_SEC", 0.0):
            yield account


async def _client():
    app = offline_app(server)
    app["situation_eth"]["vol_mult"] = {"sigma_bp": 104.7, "mult": 0.87, "bar": time.time() - 60}
    app["situation_eth"]["candles"] = [{"time": 0, "close": 2500.0}]
    c = TestClient(TestServer(app))
    await c.start_server()
    return c


def test_rule_preview_submit_places_entry_stop_and_ladder():
    fx = FakeExchange()

    async def go():
        c = await _client()
        try:
            g = await (await c.get("/api/manual-entry/rule-gauge?lev=20")).json()
            assert g["ok"] and g["flat"] and g["long"]["available"], g
            pv = await (await c.get("/api/manual-entry/preview?side=LONG&asset=eth&pct=25&lev=20&mode=rule")).json()
            assert pv["ok"], pv
            plan, rule = pv["plan"], pv["plan"]["rule"]
            assert len(rule["tranches"]) == 4 and plan["quantity"] == rule["tranches"][0]["qty"]
            assert plan["bracket"]["rule"] and plan["bracket"]["backstop_price"] == rule["stop_price"] and plan["bracket"]["tp_price"] is None
            assert rule["binding"] == "cap" and abs(rule["total_x"] - 10.0) < 0.05, rule          # 20배 × 증거금 50% = 순자산 10배
            assert abs(g["long"]["stop_price"] - rule["stop_price"]) < 1.0                       # 게이지 = 같은 함수(가격만 캔들 vs 호가)
            assert len(fx.calls) == 0, "미리보기가 주문 경로를 탔다"
            r = await c.post("/api/manual-entry/submit?side=LONG&asset=eth&pct=25&lev=20&mode=rule&confirm=1")
            body = await r.json()
            assert body["ok"], body
            st = {}
            for _ in range(300):
                st = (await (await c.get("/api/manual-entry/status")).json())["state"]
                if "ladder" in st:
                    break
                await asyncio.sleep(0.01)
            assert st.get("phase") == "filled_maker" and st["ladder"]["placed"] == 3, st
            return rule, (await (await c.get("/api/manual-entry/status")).json())["bracket_armed"]
        finally:
            await c.close()

    with offline(fx):
        rule, armed = asyncio.run(go())
    orders = fx.posts("/fapi/v1/order")
    entry, ladder = orders[0], orders[1:]
    assert entry["type"] == "LIMIT" and entry["timeInForce"] == "GTX" and float(entry["quantity"]) == rule["tranches"][0]["qty"]
    assert [(o["price"], o["quantity"]) for o in ladder] == [(t["price"], t["qty"]) for t in rule["tranches"][1:]]
    assert all(o["newClientOrderId"].startswith("dbldL") and o["timeInForce"] == "GTX" and o["side"] == "BUY" for o in ladder)
    assert not [o for o in orders if o["type"] == "MARKET"], "테이커 주문이 나갔다"
    stops = fx.posts("/fapi/v1/algoOrder")
    assert len(stops) == 1 and stops[0]["triggerPrice"] == rule["stop_price"] and stops[0]["closePosition"] == "true", stops
    key = f"{SYMBOL}:LONG"
    assert armed[key]["rule"] is True and armed[key]["sl_price"] is None and armed[key]["backstop_price"] == rule["stop_price"]


def test_rule_blocked_when_position_open_and_direct_add_blocked_on_rule_side():
    fx = FakeExchange()
    pos = {"symbol": SYMBOL, "side": "LONG", "qty": 1.0, "entry_price": 2500.0, "mark_price": 2500.0,
           "liquidation_price": 2000.0, "leverage": 20.0, "notional": 2500.0, "unrealized_pnl": 0.0, "updated_at": None}

    async def go():
        c = await _client()
        try:
            r = await c.get("/api/manual-entry/preview?side=LONG&asset=eth&pct=25&lev=20&mode=rule")
            assert r.status == 400 and (await r.json())["error"] == "rule_needs_flat"
            server.LIVE_DIR.joinpath("manual_bracket_state.json").write_text(json.dumps(
                {f"{SYMBOL}:LONG": {"symbol": SYMBOL, "side": "LONG", "rule": True, "armed_at": 1.0, "backstop_price": 2300.0}}))
            d = await (await c.get("/api/manual-entry/preview?side=LONG&asset=eth&pct=10&lev=20")).json()
            assert d["ok"] and "4분할 규칙" in (d["plan"]["blocked"] or ""), d["plan"]["blocked"]
            s = await c.post("/api/manual-entry/submit?side=LONG&asset=eth&pct=10&lev=20&confirm=1")
            assert s.status == 400                                                                  # 직접 물타기 실주문도 막힌다
        finally:
            await c.close()

    with offline(fx, positions=[pos]):
        asyncio.run(go())
    assert fx.calls == [], f"막혀야 할 경로에서 거래소 호출: {fx.calls}"


def test_cancel_bracket_also_clears_ladder_but_not_user_orders():
    fx = FakeExchange()
    opens = [{"orderId": 1, "clientOrderId": "dbldL17000000002"}, {"orderId": 2, "clientOrderId": "dbtpL1700"},
             {"orderId": 3, "clientOrderId": "aos_user_app"}, {"orderId": 4, "clientOrderId": "dbldS1700"}]

    async def signed(session, method, path, params, key, secret, offset):
        if path == "/fapi/v1/openOrders":
            return opens
        if path == "/fapi/v1/openAlgoOrders":
            return []
        return await fx.signed(session, method, path, params, key, secret, offset)

    with mock.patch.object(ex, "signed", signed):
        out = asyncio.run(ex.cancel_bracket(None, SYMBOL, "LONG", "fake-key", "fake-secret", 0))
    deleted = [p["orderId"] for m, pth, p in fx.calls if m == "DELETE"]
    assert deleted == [1, 2] and out["cancelled"] == 2, (deleted, out)       # 롱의 분할·익절만 -- 사용자 주문·숏 분할은 그대로
