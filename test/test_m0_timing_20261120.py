"""M0 측정 장치 고정 (2026-11-20, docs/rtcore_design_20261120.md §7). 네트워크 없음 · 실주문 없음.

① order_timing 시각(contextvars)이 거래소로 나가는 주문 파라미터·URL 에 섞이지 않는다(측정 켬/끔 URL 동일) + 첫 주문 POST 만 한 줄
② 멈춤 감지가 일부러 만든 300ms time.sleep 블로킹을 잡고 스택에 그 함수 이름이 나온다
③ /api/client-timing 이 잘못된 본문(너무 큼·숫자 아님·모르는 키·모르는 심볼)을 거절하고 맞는 본문만 한 줄 찍는다
실행: PYTHONPATH=. python test/test_m0_timing_20261120.py
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import live.execution.live_manual_peg_execute_20260912 as ex  # noqa: E402


class _Resp:
    status, headers = 200, {}

    async def json(self):
        return {"orderId": 1}


class _Session:
    def __init__(self):
        self.calls = []

    def request(self, method, url, headers=None):
        self.calls.append((method, url))

        class _Ctx:
            async def __aenter__(self):
                return _Resp()

            async def __aexit__(self, *_a):
                return False
        return _Ctx()


def check_params_untouched() -> None:
    real_sign = ex._sign
    ex._sign = lambda params, secret, offset: "&".join(f"{k}={v}" for k, v in params.items())   # 시각 없는 고정 서명
    try:
        params = {"symbol": "ETHUSDC", "side": "BUY", "positionSide": "LONG", "type": "LIMIT", "timeInForce": "GTX",
                  "price": 2500.0, "quantity": 1.0, "newClientOrderId": "dbpgL1"}
        snap = dict(params)

        async def go(marked: bool):
            s, out = _Session(), io.StringIO()
            if marked:
                m = ex.order_mark("entry")
                m.update(plan=m["t0"], mode="peg")
            with contextlib.redirect_stdout(out):
                await ex.signed(s, "POST", "/fapi/v1/order", params, "k", "s", 0)
                await ex.signed(s, "POST", "/fapi/v1/order", params, "k", "s", 0)   # 둘째 주문은 안 찍는다
            return s.calls, out.getvalue()

        plain, log0 = asyncio.run(go(False))
        marked, log1 = asyncio.run(go(True))
        assert plain == marked, (plain, marked)
        assert params == snap, "측정이 파라미터 dict 를 바꿨다"
        assert not any(k.startswith("_") or k in ("t0", "sent", "plan", "mode", "kind") for k in params)
        assert log0 == "", log0
        lines = [ln for ln in log1.splitlines() if ln.startswith("order_timing ")]
        assert len(lines) == 1 and "kind=entry mode=peg status=200" in lines[0] and "total=" in lines[0], log1
        for word in ("2500", "ETHUSDC", "quantity"):
            assert word not in log1, f"시간 말고 {word} 가 찍혔다"
    finally:
        ex._sign = real_sign
    print("① 파라미터 불변 · 첫 주문만 한 줄:", lines[0])


def blocker_300ms() -> None:
    import time
    time.sleep(0.3)


def check_loop_stall(server) -> None:
    out, app = io.StringIO(), {}

    async def go():
        await server.start_loop_watch(app)
        await asyncio.sleep(0.15)
        blocker_300ms()
        await asyncio.sleep(0.3)   # 감시 스레드가 풀림을 보고 찍을 시간
        await server.stop_loop_watch(app)

    with contextlib.redirect_stdout(out):
        asyncio.run(go())
    lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("loop_stall ")]
    assert len(lines) == 1, out.getvalue()
    assert "blocker_300ms" in lines[0], lines[0]
    ms = int(lines[0].split("ms=")[1].split()[0])
    assert 200 <= ms <= 400, ms
    print("② 멈춤 감지:", lines[0])


def check_client_timing(server) -> None:
    from aiohttp.test_utils import TestClient, TestServer
    from offline_app import offline_app   # 수집기·주문 감시기를 안 띄운다

    async def go():
        client = TestClient(TestServer(offline_app(server)))
        await client.start_server()
        try:
            async def post(body, raw=None):
                server._client_timing_at[0] = 0.0   # 0.2초 상한은 따로 본다
                r = await client.post("/api/client-timing", data=raw if raw is not None else json.dumps(body))
                return r.status
            assert await post(None, raw="x" * 2000) == 413
            assert await post(None, raw="{nope") == 400
            assert await post(None, raw="[" * 1000) == 400
            assert await post([1, 2]) == 400
            assert await post({"sym": "eth", "fp_med_ms": "12"}) == 400          # 숫자 아님
            assert await post({"sym": "eth", "fp_med_ms": True}) == 400          # bool 은 숫자 아님
            assert await post({"sym": "eth", "fp_med_ms": 1e9}) == 400           # 범위 밖
            assert await post(None, raw='{"sym":"eth","fp_med_ms":NaN}') == 400  # 유한 아님
            assert await post({"sym": "eth", "fp_med_ms": 3, "evil": 1}) == 400  # 모르는 키
            assert await post({"sym": "doge", "fp_med_ms": 3}) == 400            # 모르는 심볼
            assert await post({"sym": ["eth"], "fp_med_ms": 3}) == 400           # 심볼이 문자열 아님
            assert await post({"sym": "eth", "kind": "x", "order_rtt_ms": 3}) == 400
            assert await post({"sym": "eth"}) == 400                             # 숫자 하나도 없음
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert await post({"sym": "eth", "kind": "entry", "order_rtt_ms": 180.4, "order_srv_ms": 120,
                                   "order_net_ms": 60, "fp_n": 40, "fp_med_ms": 9, "fp_max_ms": 31, "px_med_ms": -12}) == 200
                assert (await client.post("/api/client-timing", data=json.dumps({"sym": "eth", "fp_n": 1}))).status == 429
            got = out.getvalue().strip()
            assert got.startswith("client_timing sym=eth kind=entry ") and "order_rtt_ms=180" in got and "px_med_ms=-12" in got, got
            return got
        finally:
            await client.close()
    got = asyncio.run(go())
    print("③ client-timing 검증:", got)


def main() -> int:
    check_params_untouched()
    from dashboard import server   # 기존 시험과 같은 import(서버를 띄우지 않는다)
    check_loop_stall(server)
    check_client_timing(server)
    print("통과 3/3")
    return 0


def test_m0_timing() -> None:   # pytest 로 돌려도 같은 검사
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
