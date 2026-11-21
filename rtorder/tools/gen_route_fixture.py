"""주문·계좌 주소 14개 HTTP 정답지(2026-11-21): 원본 orders.py 핸들러를 **진짜 aiohttp 앱**(127.0.0.1 임의 포트)에 원본과 같은 등록·미들웨어로 붙이고
요청을 보내 받은 바이트(상태·Content-Type·Cache-Control·Allow·본문 글자)와 접수 기록 줄을 적는다.
- 사례 = gen_desk_fixture 의 핸들러 사례 전부(같은 입력을 HTTP 로) + 새 사례(계좌·상태 조회 · 메서드 틀림 405 · 지운 주소 404 · 주문 잠금 409).
- 라우트 표와 order_request_log 는 server.py 5004–5017 · 1352–1379 를 **글자 그대로 옮겨 적은 것**(server.py 는 import 하지 않는다 -- 무거운 의존과 비밀 설정을 끌어온다).
  ponytail: 원본 등록이 바뀌면 여기도 손으로 -- `ROUTES_SRC` 가 그 사본이다.
실행(저장소 뿌리): ( ulimit -v 12000000; TZ=UTC timeout 900 unshare -rn sh -c 'ip link set lo up && python3 -I rtorder/tools/gen_route_fixture.py' )
"""
import asyncio
import contextlib
import io
import json
import sys
import tempfile
import time
import types
from datetime import datetime
from pathlib import Path
from unittest import mock
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_desk_fixture as gd  # noqa: E402  (원본 orders 를 가짜 의존으로 띄우는 재료 · 사례 표)
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

orders = gd.orders
OUT = gd.ROOT / "rtorder" / "tests" / "fixtures" / "route_fixture.json"
NOCACHE = {"Cache-Control": "no-cache"}   # server.py:1044

# server.py:5004–5017 사본
ROUTES_SRC = '''
app.router.add_get("/api/binance-account", order_api.api_binance_account)
app.router.add_get("/api/manual-entry/preview", order_api.api_manual_entry_preview)
app.router.add_post("/api/manual-entry/submit", lambda r: order_api.busy_or(order_api.api_manual_entry_submit, r))
app.router.add_get("/api/manual-entry/status", order_api.api_manual_entry_status)
app.router.add_post("/api/manual-entry/wait-action", order_api.api_manual_wait_action)
app.router.add_post("/api/manual-entry/cancel", order_api.api_manual_entry_cancel)
app.router.add_post("/api/manual-entry/limit-amend", lambda r: order_api.busy_or(order_api.api_manual_limit_amend, r))
app.router.add_post("/api/manual-bracket/disarm", order_api.api_manual_bracket_disarm)
app.router.add_post("/api/manual-bracket/rearm", order_api.api_manual_bracket_rearm)
app.router.add_post("/api/manual-bracket/tp", lambda r: order_api.busy_or(order_api.api_manual_bracket_tp, r))
app.router.add_post("/api/manual-bracket/tp-legs", lambda r: order_api.busy_or(order_api.api_manual_bracket_tp_legs, r))
app.router.add_post("/api/manual-bracket/add", lambda r: order_api.busy_or(order_api.api_manual_bracket_add, r))
app.router.add_post("/api/manual-bracket/modify", lambda r: order_api.busy_or(order_api.api_manual_bracket_modify, r))
app.router.add_post("/api/manual-exit/submit", lambda r: order_api.busy_or(order_api.api_manual_exit_submit, r))
'''


def make_log_mw(order_log: Path):
    # server.py:1352–1379 사본
    @web.middleware
    async def order_request_log(request, handler):
        if request.method != "POST" or not request.path.startswith("/api/manual-"):
            return await handler(request)
        t0, status, err = time.time(), 500, None
        try:
            response = await handler(request)
            status = response.status
            if isinstance(response, web.Response) and response.body is not None and len(response.body) < 20000:
                try:
                    j = json.loads(response.body)
                    err = None if j.get("ok", True) else (j.get("error") or "")
                except (ValueError, AttributeError):
                    pass
            return response
        except web.HTTPException as exc:
            status = exc.status
            raise
        finally:
            row = {"at": datetime.fromtimestamp(t0, ZoneInfo("Asia/Seoul")).isoformat(timespec="milliseconds"), "path": request.path,
                   "query": dict(request.query), "status": status, "error": err, "ms": round(1000 * (time.time() - t0)),
                   "ua": request.headers.get("User-Agent", "")[:200],
                   "ip": request.headers.get("CF-Connecting-IP") or request.headers.get("X-Forwarded-For") or request.remote}
            try:
                with order_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError as exc:
                print(f"order_request_log: 못 씀 {exc}", flush=True)
    return order_request_log


OP_ROUTE = {"api_manual_entry_submit": ("POST", "/api/manual-entry/submit"), "api_manual_entry_preview": ("GET", "/api/manual-entry/preview"),
            "api_manual_exit_submit": ("POST", "/api/manual-exit/submit"), "api_manual_limit_amend": ("POST", "/api/manual-entry/limit-amend"),
            "api_manual_wait_action": ("POST", "/api/manual-entry/wait-action"), "api_manual_entry_cancel": ("POST", "/api/manual-entry/cancel"),
            "api_manual_bracket_disarm": ("POST", "/api/manual-bracket/disarm"), "api_manual_bracket_rearm": ("POST", "/api/manual-bracket/rearm"),
            "api_manual_bracket_tp_legs": ("POST", "/api/manual-bracket/tp-legs"), "api_manual_bracket_tp": ("POST", "/api/manual-bracket/tp"),
            "api_manual_bracket_add": ("POST", "/api/manual-bracket/add"), "api_manual_bracket_modify": ("POST", "/api/manual-bracket/modify")}


async def tp(_key, fn):
    return await fn()


async def run_http(c):
    tmp = Path(tempfile.mkdtemp())
    gd.setup(c, tmp)
    tasks = gd.Tasks({k: gd.FakeTask(v) for k, v in (c.get("tasks") or {}).items()})
    fa = types.SimpleNamespace(**{**gd.FAKE_ASYNCIO.__dict__, "sleep": asyncio.sleep, "create_task": lambda coro: (coro.close(), gd.FakeTask(False))[1]})
    buf = io.StringIO()
    with mock.patch.multiple(orders, LIVE_DIR=tmp, time=gd.FAKE_TIME, datetime=gd.FakeDT, asyncio=fa, exec_enabled=lambda: gd.Ctl.exec_on,
                             load_filters=gd.f_filters, commission_bp=gd.f_fees, clear_bracket=gd.f_clear,
                             run_exit=gd.run_exit, SIZING_CAP_OVERRIDE_X=c.get("override_x", 0.0),
                             fetch_account=gd.boom, run_entry=gd.boom, place_bracket=gd.f_place_bracket, place_tp_ladder=gd.f_ladder):
        api = orders.register(account_trip_state={"seen": {}}, refresh_tasks=tasks, swr_locks={}, binance_session=lambda: None,
                              fetch_binance_json=gd.f_fetch, swr_cached=gd.f_swr, _timed_produce=tp, load_market_history=gd.boom,
                              situation_by=c.get("situation", {}), NOCACHE=NOCACHE, STALE_GRACE_SECONDS=60,
                              sec_prices=lambda a: None if gd.Ctl.sec.get(a) is None else [tuple(x) for x in gd.Ctl.sec[a]])
        cells = gd.closure_map([api.api_manual_entry_submit, api.api_manual_exit_submit, api.api_manual_limit_amend, api.busy_or, api.bracket_tick,
                                api.api_manual_wait_action, api.api_manual_bracket_rearm, api.start_wait_cleanup, api.api_binance_account,
                                api.api_manual_entry_status, api.api_manual_bracket_tp, api.api_manual_bracket_add, api.api_manual_bracket_modify])
        cells["produce_account"].cell_contents = gd.f_produce
        cells["realized_vol_now"].cell_contents = gd.f_vol
        cells["funding_now"].cell_contents = gd.f_funding
        cells["entry_then_bracket"].cell_contents = gd.f_etb
        cells["settle"].cell_contents = gd.f_settle
        cells["exit_then_rearm"].cell_contents = gd.exit_then_rearm
        cells["user_stream"].cell_contents.update(c.get("user_stream") or {})
        sl = {k: cells[k].cell_contents for k in ("manual_entry_state", "side_exit_states", "manual_wait_states", "bracket_exiting")}
        s0 = c.get("slots0") or {}
        sl["manual_entry_state"].clear()
        sl["manual_entry_state"].update(s0.get("entry") or {"phase": "idle"})
        sl["side_exit_states"].update(s0.get("side") or {})
        sl["manual_wait_states"].update(json.loads(json.dumps(s0.get("waits") or {})))
        sl["bracket_exiting"].update(s0.get("exiting") or [])
        order_log = tmp / "order_requests.jsonl"
        app = web.Application(middlewares=[make_log_mw(order_log)])
        exec(ROUTES_SRC, {"app": app, "order_api": api})
        lock = cells["manual_order_lock"].cell_contents
        async with TestClient(TestServer(app, host="127.0.0.1", port=0)) as client:
            assert str(client.make_url("/")).startswith("http://127.0.0.1:")
            if c.get("locked"):
                await lock.acquire()
            pairs = [tuple(x) for x in c["pairs"]]
            with contextlib.redirect_stdout(buf):
                r = await client.request(c["method"], c["path"] + ("?" + urlencode(pairs) if pairs else ""))
                raw = await r.read()
            if c.get("locked"):
                lock.release()
        rows = [json.loads(x) for x in order_log.read_text("utf-8").splitlines()] if order_log.exists() else []
        log = [{k: x[k] for k in ("path", "query", "status", "error")} for x in rows]
        return {"name": c["name"], "in": c, "status": r.status, "content_type": r.headers.get("Content-Type"),
                "cache_control": r.headers.get("Cache-Control"), "allow": r.headers.get("Allow"), "body": raw.decode("utf-8"), "log": log,
                "slots": {"entry": gd.snap(sl["manual_entry_state"]), "side": gd.snap(sl["side_exit_states"]), "waits": gd.snap(sl["manual_wait_states"])}}


def T(dt):
    return gd.T0 + dt


ARMED = {"ETHUSDC:LONG": {"symbol": "ETHUSDC", "side": "LONG", "sl_price": 2380.0, "sl_level": 2380.0, "tp_price": 2600.0, "tp_level": 2600.0, "lines": True,
                          "sl_name": "선 손절 2380.00", "tp_resting": True, "armed_at": T(-600), "placed": {"placed": True}, "extra": "화면에 안 나감"}}
POS = [gd.pos("ETHUSDC", "LONG", 0.4, 2400.0, 2450.0), gd.pos("SOXLUSDT", "SHORT", 3.0, 31.5, 30.1), gd.pos("XRPUSDC", "LONG", 0.0, 0.5, 0.51)]
NEW = [
    ("acct_cached", {"method": "GET", "path": "/api/binance-account", "pairs": [], "account": gd.ACC(POS), "file0": ARMED,
                     "user_stream": {"connected": True, "seq": 3, "last_event_at": T(-12.5), "error": None}}),
    ("acct_fresh", {"method": "GET", "path": "/api/binance-account", "pairs": [["fresh", "1"]], "account": gd.ACC(), "fresh_account": gd.ACC(POS[:1]),
                    "user_stream": {"connected": False, "seq": 0, "last_event_at": None, "error": "RuntimeError: listenKey 빈 응답"}}),
    ("acct_not_ok", {"method": "GET", "path": "/api/binance-account", "pairs": [], "account": gd.ACC(ok=False)}),
    ("acct_raises", {"method": "GET", "path": "/api/binance-account", "pairs": [], "account": {"__raise__": ["RuntimeError", "가짜 실패"]}}),
    ("status_eth", {"method": "GET", "path": "/api/manual-entry/status", "pairs": [["asset", "eth"]], "file0": ARMED,
                    "slots0": {"entry": {"phase": "working", "side": "LONG"}, "side": {"sol": {"phase": "idle"}},
                               "waits": {"eth#2": {"phase": "waiting", "mode": "limit", "side": "LONG", "wid": 9}, "sol": {"phase": "waiting"}}}}),
    ("status_bad_asset", {"method": "GET", "path": "/api/manual-entry/status", "pairs": [["asset", "doge"]], "exec": False}),
    ("status_no_asset_creates_slot", {"method": "GET", "path": "/api/manual-entry/status", "pairs": [], "slots0": {"waits": {"sol": {"phase": "waiting"}}}}),
    ("get_on_post_submit", {"method": "GET", "path": "/api/manual-entry/submit", "pairs": [["side", "LONG"], ["confirm", "1"]]}),
    ("post_on_get_preview", {"method": "POST", "path": "/api/manual-entry/preview", "pairs": [["side", "LONG"]]}),
    ("post_on_get_status", {"method": "POST", "path": "/api/manual-entry/status", "pairs": []}),
    ("head_status", {"method": "HEAD", "path": "/api/manual-entry/status", "pairs": []}),
    ("removed_exit_preview_get", {"method": "GET", "path": "/api/manual-exit/preview", "pairs": [["side", "LONG"]]}),
    ("removed_exit_preview_post", {"method": "POST", "path": "/api/manual-exit/preview", "pairs": [["side", "LONG"]]}),
    ("removed_submit_all", {"method": "POST", "path": "/api/manual-exit/submit-all", "pairs": [["confirm", "1"]]}),
    ("locked_submit_409", {"method": "POST", "path": "/api/manual-entry/submit", "locked": True, "slots0": {"entry": {"phase": "idle", "note": "x"}},
                           "pairs": [["side", "LONG"], ["confirm", "1"], ["tp", "2600"], ["sl", "2380"]]}),
    ("locked_modify_409", {"method": "POST", "path": "/api/manual-bracket/modify", "locked": True, "pairs": [["side", "LONG"], ["confirm", "1"]]}),
    ("locked_disarm_not_locked", {"method": "POST", "path": "/api/manual-bracket/disarm", "locked": True, "file0": ARMED,
                                  "pairs": [["side", "LONG"], ["confirm", "1"]]}),
    ("locked_exit_409", {"method": "POST", "path": "/api/manual-exit/submit", "locked": True, "pairs": [["side", "LONG"], ["confirm", "1"]]}),
    ("dup_keys_first_value", {"method": "POST", "path": "/api/manual-entry/submit", "pairs": [["side", "SHORT"], ["side", "LONG"], ["confirm", "0"], ["confirm", "1"]]}),
    ("exec_off_cancel", {"method": "POST", "path": "/api/manual-entry/cancel", "exec": False, "pairs": [["confirm", "1"]]}),
]


def main():
    out = []
    for item in gd.API_CASES:
        c = {"name": item[0], **item[1]}
        c.setdefault("account", gd.ACC())
        m, p = OP_ROUTE[c["op"]]
        c.update(method=m, path=p, pairs=[[k, v] for k, v in c["query"].items()])
        out.append(asyncio.run(run_http(c)))
    for name, spec in NEW:
        c = {"name": name, **spec}
        c.setdefault("account", gd.ACC())
        out.append(asyncio.run(run_http(c)))
    # 접수 기록 error 칸 규칙(미들웨어의 그 몇 줄을 본문 꼴마다) -- 핸들러가 내지 않는 꼴(ok 없음·목록·큰 본문)까지
    def err_of(body: bytes):
        err = None
        if body is not None and len(body) < 20000:
            try:
                j = json.loads(body)
                err = None if j.get("ok", True) else (j.get("error") or "")
            except (ValueError, AttributeError):
                pass
        return err
    bodies = ['{"x": 1}', '{"ok": false}', '{"ok": 0, "error": "e"}', '{"ok": false, "error": null}', '[1, 2]', '"s"', '{"ok": true, "error": "x"}',
              '{"ok": [], "error": "빈 목록"}', '{"ok": false, "error": "' + "가" * 7000 + '"}', 'not json']
    log_rules = [[b, err_of(b.encode())] for b in bodies]
    OUT.write_text(json.dumps({"cases": out, "log_error": log_rules}, ensure_ascii=False, separators=(",", ":")), "utf-8")
    from collections import Counter
    print(f"사례 {len(out)} · 상태 {dict(Counter(x['status'] for x in out))} · {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
