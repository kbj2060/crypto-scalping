"""파이썬 그림자 기록 합성 정답지(2026-11-21): 원본 주문 코드(PY_ROOT = main 판을 git archive 로 푼 폴더 · 점-env 없음)를 127.0.0.1 가짜 거래소로 돌려
실서버에서만 나올 꼴의 order_shadow.jsonl 을 만든다. 값은 전부 합성(키·서명 없음).
  ① 진입 → 체결 → 익절 걸고 무장 → 익절 지정가 끄기(접수 요청만 있는 chain · 작업 아님) → 100% 청산
  ② 감시 틱 발동 청산(익절 지정가 먼저 지우고 → 따라가기 청산 → 정리)
  ③ 무장 파일만 바뀌는 틱(새 계좌 읽기 실패 → 30초 뒤 다시 · 요청 0)
  ④ 시작은 있고 끝이 없는 chain(프로세스 재시작으로 잘림 -- ① 의 진입을 복사해 끝 줄을 뺀다)
  ⑤ 쓰는 중이던 반쪽 줄 · 돌린 파일(.1)은 Rust 시험이 이 파일을 잘라 만든다
실행(저장소 뿌리): PY_ROOT=<main 판 폴더> ( ulimit -v 12000000; timeout 600 unshare -rn bash -c 'ip link set lo up; python3 -I rtorder/tools/gen_py_shadow_fixture.py' )
"""
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

PY_ROOT = Path(os.environ["PY_ROOT"])
sys.path.insert(0, str(PY_ROOT))
sys.path.insert(0, str(PY_ROOT / "test"))
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "py_shadow_cases.jsonl"

import test_order_shadow_log_20261121 as tsl  # noqa: E402
import test_chart_lines_entry_20261028 as cl  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import FAKE_STATE, _fake_binance  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402
from dashboard import orders, server  # noqa: E402
from live.execution import live_manual_peg_execute_20260912 as ex  # noqa: E402
from live.execution import order_shadow as shadow  # noqa: E402
from scripts import live_binance_account_20260910 as acct_mod  # noqa: E402

SYM = cl.SYM
captured: list[str] = []
_orig_write = shadow._write


def spy(path, lines):
    captured.extend(lines)
    _orig_write(path, lines)


class Exchange:
    """상태 있는 가짜 거래소를 따로 도는 루프(스레드)에 띄운다."""

    def __init__(self, st):
        self.loop = asyncio.new_event_loop()
        self.ts = TestServer(tsl.stateful_exchange(st), host="127.0.0.1")
        self.loop.run_until_complete(self.ts.start_server())
        self.th = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.th.start()
        self.base = str(self.ts.make_url("")).rstrip("/")
        assert self.base.startswith("http://127.0.0.1:")

    def close(self):
        asyncio.run_coroutine_threadsafe(self.ts.close(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.th.join(10)
        self.loop.close()


def env(base):
    return [mock.patch.object(ex, "FAPI", base), mock.patch.object(acct_mod, "FAPI", base), mock.patch.object(shadow, "_write", spy),
            mock.patch.dict(os.environ, {"BINANCE_API_KEY": tsl.KEY, "BINANCE_SECRET_KEY": tsl.SECRET, "ORDER_SHADOW_LOG": "1"})]


def case_entry_exit():
    st = {"n": 100, "orders": {}, "pos": {}}
    x = Exchange(st)
    acct = json.loads(json.dumps(cl.FLAT))
    tc = unittest.TestCase()

    async def fn(c):
        r = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&pct=10&tp=2600&sl=2400")).json()
        assert r.get("ok"), r
        for _ in range(300):
            await asyncio.sleep(0.05)
            if (server.LIVE_DIR / "manual_bracket_state.json").exists():
                break
        await asyncio.sleep(0.3)
        # 접수 요청만 있는 chain: 익절 지정가 끄기(핸들러가 직접 거래소에 취소를 보낸다 -- 작업 없음)
        r = await (await c.post(f"/api/manual-bracket/tp?side=LONG&symbol={SYM}&act=off&confirm=1")).json()
        qty = st["pos"].get((SYM, "LONG"), 0.0)
        acct["positions"] = [{**cl.POS["positions"][0], "qty": qty, "notional": qty * 2500.0}]
        r = await (await c.post("/api/manual-exit/submit?side=LONG&confirm=1&pct=100&asset=eth")).json()
        assert r.get("ok"), r
        for _ in range(300):
            await asyncio.sleep(0.05)
            if st["pos"].get((SYM, "LONG"), 0.0) == 0.0:
                break
        await asyncio.sleep(0.3)

    try:
        ps = env(x.base)
        for p in ps:
            p.start()
        try:
            cl.ChartLinesTest._run(tc, fn, acct=acct, run_entry=ex.run_entry, run_exit=ex.run_exit, place_bracket=ex.place_bracket,
                                   clear_bracket=ex.clear_bracket, place_tp_ladder=ex.place_tp_ladder)
            shadow.flush()
        finally:
            for p in reversed(ps):
                p.stop()
    finally:
        x.close()


def case_tick(fire: bool):
    """감시 틱: fire = 1초 가격이 손절선에 닿아 발동 청산까지 · 아니면 새 계좌 읽기 실패로 무장 파일만 바뀌는 틱."""
    st = {"n": 200, "orders": {}, "pos": {(SYM, "LONG"): 0.5}}
    arm = dict(tsl.ETH_TOUCHED)
    if fire:   # 익절 지정가가 거래소에 걸려 있다(발동 = 먼저 지운다)
        st["orders"][150] = {"orderId": 150, "clientOrderId": "dbtpL1799990000000", "symbol": SYM, "side": "SELL", "positionSide": "LONG", "type": "LIMIT",
                             "price": "2550.0", "origQty": "0.5", "executedQty": "0", "status": "NEW", "avgPrice": "0"}
    else:
        arm.update(fired=True, tp_resting=False)
    x = Exchange(st)
    calls = {"n": 0}

    async def fake_account(*_a, **_k):
        calls["n"] += 1
        if not fire and calls["n"] >= 2:   # 감시 틱의 캐시 계좌는 정상 · 청산 조립의 새 계좌 읽기는 실패
            return {"ok": False, "error": "positionRisk: 418 [-1003] banned"}
        q = st["pos"].get((SYM, "LONG"), 0.0)
        return {"ok": True, "generated_at": "2099-01-01T00:00:00+00:00", "positions": [{"symbol": SYM, "side": "LONG", "qty": q, "entry_price": 2500.0}] if q else []}

    async def swr(_k, _ttl, produce, **_kw):
        return await produce()

    async def hist(_a):
        return [{"time": 1000, "close": 2490.0}]

    async def fetch(url, params=None, **kw):
        r = await _fake_binance(url, params or {}, **kw)
        return {} if r is None else r

    async def filters(_s, _symbol):   # 원본 캐시 적중(exchangeInfo 왕복 없음) 꼴
        return {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 5.0}

    async def fees(_s, symbol):   # 원본 1시간 캐시 적중 꼴
        return {"symbol": symbol, "maker_bp": 2.0, "taker_bp": 5.0, "source": "live"}

    async def boom(*_a, **_k):
        raise AssertionError("진입 경로")

    async def go():
        from aiohttp import ClientSession
        sess = ClientSession()   # 진짜 집행 함수가 쓰는 세션(가짜 거래소 127.0.0.1 로만 간다 -- FAPI 를 바꿔 둠)
        with tempfile.TemporaryDirectory() as tmp:
            ps = env(x.base) + [mock.patch.object(orders, "LIVE_DIR", Path(tmp)), mock.patch.object(orders, "fetch_account", fake_account),
                                mock.patch.object(orders, "exec_enabled", lambda: True), mock.patch.object(orders, "load_filters", filters),
                                mock.patch.object(orders, "commission_bp", fees), mock.patch.object(orders, "run_entry", boom)]
            for p in ps:
                p.start()
            try:
                (Path(tmp) / "manual_bracket_state.json").write_text(json.dumps({f"{SYM}:LONG": arm}), "utf-8")
                (Path(tmp) / "eth_position_sizing_state.json").write_text(json.dumps(FAKE_STATE), "utf-8")
                api = orders.register(account_trip_state={"seen": {}}, refresh_tasks={}, swr_locks={}, binance_session=lambda: sess,
                                      fetch_binance_json=fetch, swr_cached=swr, _timed_produce=None, load_market_history=hist,
                                      situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60,
                                      sec_prices=lambda a: [(2001, 2490.0), (2002, 2479.0)] if a == "eth" else None)
                await api.shadow_bracket_tick()
                await api.shadow_bracket_tick()
                shadow.flush()
            finally:
                for p in reversed(ps):
                    p.stop()
                await sess.close()

    try:
        asyncio.run(go())
    finally:
        x.close()
    if fire:
        assert st["pos"][(SYM, "LONG")] == 0.0, st["pos"]


def main():
    case_entry_exit()
    case_tick(True)
    case_tick(False)
    rows = [json.loads(x) for x in captured]
    entry = next(r for r in rows if r["rec"] == "start" and r.get("op") == "entry")
    cut = [{**r, "chain": r["chain"] + "-cut"} for r in rows if r["chain"] == entry["chain"] and r["rec"] != "end"]   # ④ 재시작으로 잘림
    rows += cut
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    from collections import Counter
    print(f"줄 {len(rows)} · " + " · ".join(f"{k} {v}" for k, v in Counter((r['rec'], r.get('op') or '') for r in rows if r['rec'] in ('start', 'end')).items()))
    text = OUT.read_text()
    assert "signature=" not in text and tsl.KEY not in text


if __name__ == "__main__":
    main()
