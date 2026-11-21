#!/usr/bin/env python3
"""rtserve HTTP 일치 시험의 정답지를 만든다 (2026-11-21).

원본 대시보드 응답 서버의 «공통 동작»(미들웨어 json_compress_etag·static_asset_headers · 정적 파일 add_static ·
index 302 · dashboard_index 버스터 · pwa_asset · 404/405/500)과 «파일만 읽는 엔드포인트» 13개를 **진짜 aiohttp 객체**로 돌려,
연결에 나갈 바이트(상태 줄·헤더·본문)를 그대로 떠 둔다. rtserve/tests/http_parity.rs 가 같은 파일 트리로 Rust 서버를
127.0.0.1 에 띄워 실제 HTTP 로 물어 대조한다.

서버를 띄우지 않는다: 소켓을 열지 않고(make_mocked_request + 바이트를 모으는 가짜 전송) web.Application 의 라우터·미들웨어만 돈다.
대시보드 서버 모듈은 import 하지 않는다(import 만으로 수집기·키를 끌어온다) -- 쓰는 함수 본문을 아래에 **옮겨 적었다**(원본 행 주석,
2026-11-21 판). 원본이 바뀌면 여기도 고쳐야 한다. 별도 모듈(sr_lean·risk_radar·regime_band·acct_notes·worker_state·data_store)은
파일 경로로 그대로 불러 쓴다(표준 라이브러리 + pandas 만 끌어온다 -- 확인함). 시계는 time.time / datetime.now 를 NOW 로 고정한다.

    python3 rtserve/tools/gen_http_fixture.py        # 저장소 루트에서 · rtserve/tests/fixtures/http.json 을 쓴다(몇 초)
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["AIOHTTP_NOSENDFILE"] = "1"   # 가짜 전송에는 sendfile 이 없다 -- 같은 바이트를 write 로 낸다(원본 FileResponse 의 대체 경로)
from aiohttp import web  # noqa: E402
from aiohttp.http_writer import StreamWriter  # noqa: E402
from aiohttp.test_utils import make_mocked_request  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "http.json"
NOW = 1_795_000_123.456789          # 2026-11-17 UTC 무렵 · 소수부가 있어 repr 이 길다
MTIME_NS = 1_794_990_000_123_456_789
DASHBOARD_ASSETS = ["eth", "sol", "xrp"]   # 원본 기본 env 값 "eth,sol,xrp"(server.py:424)
COIN_CONFIG = ["eth", "btc", "sol", "xrp", "hype", "sndk", "soxl", "skhynix", "mu"]   # scripts/coin_config.py 열쇠 순서

_real_time = time.time
time.time = lambda: NOW   # 모듈들이 `import time; time.time()` 으로 부른다


def load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sr_lean = load("sr_lean", "dashboard/sr_lean.py")
risk_radar = load("risk_radar", "dashboard/risk_radar.py")
regime_band = load("regime_band", "dashboard/regime_band.py")
acct_notes = load("acct_notes", "dashboard/acct_notes.py")
worker_state = load("worker_state", "dashboard/worker_state.py")
sys.path.insert(0, str(ROOT))
from scripts.data_store import read_rows  # noqa: E402


class _FixedDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(NOW, tz)


worker_state.datetime = _FixedDT


# ───────────────────────── 원본에서 옮겨 적은 것 (server.py 2026-11-21 판, 행 번호) ─────────────────────────
NOCACHE = {"Cache-Control": "no-cache"}                                      # 1044
ASSET_FILES = ("app.js", "styles.css", "fp_daily.js", "liq_profile.js", "bubble_map.js")   # 1282


def etag_matches(request, etag):                                             # 1060
    candidates = request.headers.get("If-None-Match", "")
    return any(candidate.strip() in {"*", etag} for candidate in candidates.split(","))


@web.middleware
async def static_asset_headers(request, handler):                       # 1287
    response = await handler(request)
    asset_name = Path(request.path).name
    if asset_name in ASSET_FILES:
        response.headers["Cache-Control"] = "no-cache"
        response.enable_compression()
    elif asset_name.endswith(".ttf"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif asset_name.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache"
    return response

@web.middleware
async def json_compress_etag(request, handler):                         # 1325
    response = await handler(request)
    if isinstance(response, web.Response):
        response._zlib_executor_size = 4096
    if (request.method != "GET" or not isinstance(response, web.Response)
            or response.status != 200 or response.body is None
            or (response.content_type or "") != "application/json"):
        return response
    if "ETag" not in response.headers:
        etag = f'W/"body-{hashlib.sha256(response.body).hexdigest()[:16]}"'
        response.headers["ETag"] = etag
        response.headers.setdefault("Cache-Control", "no-cache")
        if etag_matches(request, etag):
            return web.Response(status=web.HTTPNotModified.status_code,
                                headers={"ETag": etag,
                                         "Cache-Control": response.headers["Cache-Control"]})
    response.enable_compression()
    return response


def make_app(R: Path) -> web.Application:
    LIVE_DIR = R / "data" / "live"
    DASHBOARD_DIR = R / "dashboard" / "live"
    TREASURY_DB = R / "data" / "hot" / "treasury_futures.sqlite"             # 593
    XYZ_DB = R / "data" / "hot" / "hl_xyz.sqlite"                            # 618
    XYZ_COIN = "xyz:XYZ100"
    MACRO_5M_DB = R / "data" / "hot" / "macro_5m.sqlite"                     # 643
    MACRO_CALENDAR_STATE_PATH = R / "data" / "live" / "macro_calendar_state.json"   # 370
    _read_only_rows = read_rows                                              # coin_flow.py:113 → data_store.read_rows

    # order_request_log(1352)는 POST /api/manual-* 만 건드린다 -- 이 조각에는 그 주소가 없다(다음 조각).
    app = web.Application(middlewares=[static_asset_headers, json_compress_etag])

    def no_cache(resp):                                                      # 1265
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    async def index(_):                                                      # 3590
        raise web.HTTPFound("/dashboard/live/")

    def _asset_buster():                                                     # 3593
        h = hashlib.sha256()
        for name in ASSET_FILES:
            try:
                h.update((DASHBOARD_DIR / name).read_bytes())
            except OSError:
                h.update(name.encode())
        return h.hexdigest()[:12]

    def _hide_off_assets(html):                                              # 3609
        def mark(m):
            tag, asset = m.group(0), m.group(1)
            if asset in DASHBOARD_ASSETS or " hidden" in tag:
                return tag
            return tag[:-1] + " hidden>"
        return re.sub(r'<button[^>]*\bdata-asset="([a-z]+)"[^>]*>', mark, html)

    async def dashboard_index(_):                                            # 3625
        html = (DASHBOARD_DIR / "index.html").read_text(encoding="utf-8")
        v = _asset_buster()
        html = re.sub(r'(' + "|".join(re.escape(n) for n in ASSET_FILES) + r')\?v=[A-Za-z0-9._-]+',
                      lambda m: f"{m.group(1)}?v={v}", html)
        html = _hide_off_assets(html)
        response = web.Response(text=html, content_type="text/html")
        response.enable_compression()
        return no_cache(response)

    async def pwa_asset(request):                                            # 3635
        name = request.match_info["name"]
        return no_cache(web.FileResponse(DASHBOARD_DIR / name))

    def _query_coin_asset(request):                                          # 4742
        asset = request.query.get("asset", "eth").lower()
        if asset not in COIN_CONFIG:
            raise web.HTTPBadRequest(reason="unsupported_asset")
        return asset

    def treasury_payload(db=TREASURY_DB, hours=12.0, show_h=1.0):            # 596
        try:
            rows = _read_only_rows(db, """SELECT symbol, ts_sec, close, contract FROM treasury_fut_1m
                                      WHERE close IS NOT NULL AND ts_sec >= (SELECT max(ts_sec) FROM treasury_fut_1m) - ?
                                      ORDER BY symbol, ts_sec""", [int(hours * 3600)])
        except (sqlite3.Error, OSError):
            return {"ok": False, "symbols": {}}
        out = {}
        for sym, ts, close, contract in rows:
            out.setdefault(sym, {"contract": contract, "bars": []})["bars"].append([int(ts), float(close)])
        for v in out.values():
            b = v["bars"]
            v["chg12_pct"] = (b[-1][1] / b[0][1] - 1) * 100
            v["bars"] = [x for x in b if x[0] >= b[-1][0] - show_h * 3600]
        return {"ok": bool(out), "symbols": out}

    def xyz100_payload(since=0, db=XYZ_DB, hours=24.0):                      # 622
        try:
            rows = _read_only_rows(db, """SELECT ts_sec, close FROM hl_xyz_1m WHERE coin = ? AND close IS NOT NULL
                                      AND ts_sec >= max(?, (SELECT max(ts_sec) FROM hl_xyz_1m WHERE coin = ?) - ?) ORDER BY ts_sec""",
                                   [XYZ_COIN, int(since), XYZ_COIN, int(hours * 3600)])
            live = _read_only_rows(db, "SELECT recv_sec, mid, mark, oracle FROM hl_xyz_ctx_1s WHERE coin = ? ORDER BY recv_sec DESC LIMIT 1",
                                   [XYZ_COIN])
        except (sqlite3.Error, OSError, OverflowError):
            return {"ok": False, "bars": [], "live": None, "now_sec": int(time.time())}
        lv = None
        if live:
            t, mid, mark, ora = live[0]
            px = mid if mid is not None else mark
            lv = {"recv_sec": int(t), "px": None if px is None else float(px), "oracle": None if ora is None else float(ora)}
        return {"ok": bool(rows or lv), "coin": XYZ_COIN, "full": not since, "bars": [[int(t), float(c)] for t, c in rows], "live": lv,
                "now_sec": int(time.time())}

    def macro_today_payload(xyz_db=XYZ_DB, m5_db=MACRO_5M_DB, hours=24.0):   # 646
        out = {}
        now = int(time.time())
        try:
            b = _read_only_rows(xyz_db, """SELECT ts_sec + 60, close FROM hl_xyz_1m WHERE coin = 'xyz:SP500' AND close IS NOT NULL AND ts_sec + 60 <= ?
                                       AND ts_sec >= (SELECT max(ts_sec) FROM hl_xyz_1m WHERE coin = 'xyz:SP500') - ? ORDER BY ts_sec""",
                                [now, int(hours * 3600)])
            lv = _read_only_rows(xyz_db, "SELECT recv_sec, coalesce(mid, mark) FROM hl_xyz_ctx_1s WHERE coin = 'xyz:SP500' "
                                         "ORDER BY recv_sec DESC LIMIT 1", [])
            if b or lv:
                out["SP"] = {"bars": [[int(t), float(c)] for t, c in b],
                             "live": {"recv_sec": int(lv[0][0]), "px": float(lv[0][1])} if lv and lv[0][1] is not None else None}
        except (sqlite3.Error, OSError):
            pass
        try:
            b = _read_only_rows(m5_db, """SELECT ts_sec + 300, close FROM yahoo_5m WHERE symbol = 'VIX' AND close IS NOT NULL
                                      AND ts_sec >= (SELECT max(ts_sec) FROM yahoo_5m WHERE symbol = 'VIX') - ? ORDER BY ts_sec""",
                               [int(hours * 3600)])
            if b:
                out["VIX"] = {"bars": [[int(t), float(c)] for t, c in b], "live": None}
        except (sqlite3.Error, OSError):
            pass
        return {"ok": bool(out), "symbols": out, "now_sec": now}

    def macro_calendar_payload():                                            # 1150
        return worker_state.worker_payload(MACRO_CALENDAR_STATE_PATH, 90.0, ts_field="generated_at", extra_missing={"events": []})

    def guarded(fn):   # api_regime_band·api_sr_lean·api_risk_radar(2785~2804)의 try/except 모양(swr 캐시는 빼고 바로 부른다)
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}

    async def api_regime_band(request):                                      # 2785
        return web.json_response(guarded(lambda: regime_band.regime_band_payload(LIVE_DIR / "regime_band_forward.jsonl")), headers=NOCACHE)

    async def api_sr_lean(request):                                          # 2792
        return web.json_response(guarded(lambda: sr_lean.sr_lean_payload(LIVE_DIR / "sr_lean")), headers=NOCACHE)

    async def api_risk_radar(request):                                       # 2799 (모듈 ROOT 를 시험 트리로 바꿔 부른다)
        risk_radar.ROOT = R
        risk_radar.PARAMS = R / "dashboard/risk_radar_params_20261116.json"
        return web.json_response(guarded(lambda: risk_radar.risk_radar_payload(R / "data/live/risk_radar_latest.json", risk_radar.PARAMS)),
                                 headers=NOCACHE)

    _liq_tab_files = {}

    async def api_liq_tab(request):                                          # 2808
        asset = "all" if request.query.get("asset") == "all" else _query_coin_asset(request)
        path = LIVE_DIR / f"liq_tab_{asset}.json"

        def read():
            try:
                mtime = path.stat().st_mtime_ns
                hit = _liq_tab_files.get(asset)
                if hit and hit[0] == mtime:
                    return hit[1]
                body = path.read_bytes()
            except OSError:
                return None
            _liq_tab_files[asset] = (mtime, body)
            return body
        body = read()
        if body is None:
            return web.json_response({"ok": False, "reason": "not_built"}, headers=NOCACHE)
        resp = web.Response(body=body, content_type="application/json", headers=NOCACHE)
        resp.enable_compression()
        return resp

    async def api_acct_notes(request):                                       # 2866
        return web.json_response({"notes": acct_notes.load(LIVE_DIR / "acct_notes.json")}, headers=NOCACHE)

    async def api_acct_wrong(request):                                       # 2878
        return web.json_response({"rows": acct_notes.wrong_load(LIVE_DIR / "acct_wrong.json")}, headers=NOCACHE)

    async def api_iceberg(request):                                          # 2958
        asset = request.query.get("asset", "eth")
        if asset not in ("eth", "xrp"):
            raise web.HTTPBadRequest(reason="asset")

        def produce():
            d = LIVE_DIR / "iceberg"
            try:
                st = json.loads((d / f"{asset}_state.json").read_text())
            except (OSError, ValueError):
                return {"ok": False, "reason": "no_state"}
            cut, ev = time.time() - 86400, []
            try:
                for ln in (d / f"{asset}_events.jsonl").read_text().splitlines():
                    e = json.loads(ln)
                    if e["t_end"] >= cut:
                        ev.append(e)
            except FileNotFoundError:
                pass
            return {"ok": True, "n": st["n_forward"], "target": st["target"], "age_s": round(time.time() - st["updated_at"], 1),
                    "valid": st["valid"], "events": ev}
        return web.json_response(produce(), headers=NOCACHE)

    async def api_ob5m(request):                                             # 2982
        asset = request.query.get("asset", "eth")
        if asset not in COIN_CONFIG:
            return web.json_response({"bars": [], "walls": [], "marks": []}, headers=NOCACHE)

        def produce():
            db = LIVE_DIR / "ob5m" / f"{asset}usdt.sqlite"
            if not db.exists():
                return {"bars": [], "walls": [], "marks": []}
            cut = int((time.time() - 7 * 86400) * 1000)
            rows = read_rows(db, "SELECT t, warm, imb50, eat_n_a, wall_px_b, wall_usd_b, wall_px_a, wall_usd_a FROM bars WHERE t >= ? ORDER BY t", (cut,))
            try:
                lv = read_rows(db, "SELECT upd, bar, marks FROM live WHERE k = 0 AND upd >= ?", (int(time.time() * 1000) - 30_000,))
            except sqlite3.Error:
                lv = []
            lb, lmarks = (json.loads(lv[0][1]), json.loads(lv[0][2])) if lv else (None, [])
            live_t = None
            if lb and (not rows or lb["t"] > rows[-1][0]):
                rows.append((lb["t"], lb["warm"], lb["imb50"], None, lb["wall_px_b"], lb["wall_usd_b"], lb["wall_px_a"], lb["wall_usd_a"]))
                live_t = lb["t"]
            bars, walls, run = [], [], {}
            for t, warm, imb, ev, pb, ub, pa, ua in rows:
                bars.append({"t": t, "tilt_pct": None if imb is None else round(imb * 100, 2), "book_ok": not warm and imb is not None,
                             "events_known": ev is not None, **({"live": True} if t == live_t else {})})
                for side, px, usd in (("bid", pb, ub), ("ask", pa, ua)):
                    r = run.get(side)
                    if r and px is not None and r["px"] == px and r["t_end"] == t:
                        r["t_end"], r["usd_max"] = t + 300_000, max(r["usd_max"], usd or 0.0)
                        continue
                    if r:
                        walls.append(run.pop(side))
                    if px is not None:
                        run[side] = {"side": side, "px": px, "t_start": t, "t_end": t + 300_000, "usd_max": usd or 0.0}
            walls += run.values()
            marks = [{"t": t, "bar_t": bt, "side": s, "kind": k, "px": px, "usd": round(u or 0), "dist_bp": round(d, 2)}
                     for t, bt, s, k, px, u, d in read_rows(db, "SELECT t, bar_t, side, kind, px, usd, dist_bp FROM marks WHERE bar_t >= ? ORDER BY t", (cut,))]
            known = {b["t"] for b in bars if b["events_known"]} | {m["bar_t"] for m in marks}
            marks += [{"t": m["t"], "bar_t": m["bar_t"], "side": m["side"], "kind": m["kind"], "px": m["px"], "usd": round(m["usd"]),
                       "dist_bp": round(m["dist_bp"], 2), "live": True} for m in lmarks if m["bar_t"] not in known]
            return {"bars": bars, "walls": walls, "marks": marks}
        return web.json_response(produce(), headers=NOCACHE)

    async def api_treasury_futures(request):                                 # 2110
        if request.query.get("h") == "24":
            payload = treasury_payload(TREASURY_DB, 24.0, 24.0)
        else:
            payload = treasury_payload()
        return web.json_response(payload, headers=NOCACHE)

    async def api_macro_today(request):                                      # 2117
        return web.json_response(macro_today_payload(), headers=NOCACHE)

    async def api_xyz100(request):                                           # 2121
        try:
            since = min(max(0, int(request.query.get("since") or 0)), 2**40)
        except ValueError:
            since = 0
        return web.json_response(xyz100_payload(since) if since else xyz100_payload(), headers=NOCACHE)

    async def api_macro_calendar(request):                                   # 4836 (load_macro_calendar 의 swr 는 빼고)
        return web.json_response(macro_calendar_payload(), headers=NOCACHE)

    # 등록 순서 = 원본(4932~5025) 중 이 조각에 든 것
    app.router.add_get("/", index)
    app.router.add_get("/dashboard/live", dashboard_index)
    app.router.add_get("/dashboard/live/", dashboard_index)
    app.router.add_get("/dashboard/live/index.html", dashboard_index)
    app.router.add_get("/dashboard/live/{name:sw\\.js|manifest\\.webmanifest}", pwa_asset)
    app.router.add_get("/api/regime-band", api_regime_band)
    app.router.add_get("/api/risk-radar", api_risk_radar)
    app.router.add_get("/api/sr-lean", api_sr_lean)
    app.router.add_get("/api/liq-tab", api_liq_tab)
    app.router.add_get("/api/acct-notes", api_acct_notes)
    app.router.add_get("/api/acct-wrong", api_acct_wrong)
    app.router.add_get("/api/iceberg", api_iceberg)
    app.router.add_get("/api/ob5m", api_ob5m)
    app.router.add_get("/api/treasury-futures", api_treasury_futures)
    app.router.add_get("/api/xyz100", api_xyz100)
    app.router.add_get("/api/macro-today", api_macro_today)
    app.router.add_get("/api/macro-calendar", api_macro_calendar)
    app.router.add_static("/dashboard/live/", DASHBOARD_DIR, show_index=False)
    return app


# ───────────────────────── 바이트 뜨기 ─────────────────────────
class _Tr:
    def __init__(self):
        self.buf = bytearray()

    def write(self, b):
        self.buf += b

    def writelines(self, bs):
        for b in bs:
            self.buf += b

    def is_closing(self):
        return False

    def get_extra_info(self, name, default=None):
        return default


class _Proto:
    def __init__(self):
        self.transport = _Tr()
        self._paused = False
        self.writing_paused = False
        self.ssl_context = None
        self.peername = ("127.0.0.1", 50000)
        self.sockname = ("127.0.0.1", 8787)

    async def _drain_helper(self):
        return None

    def resume_reading(self, *a, **k):   # 본문 StreamReader 가 부른다
        return None

    def pause_reading(self, *a, **k):
        return None


def _parse(raw: bytes) -> dict:
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    _, status, reason = lines[0].split(" ", 2)
    status = int(status)
    headers = [list(ln.split(": ", 1)) for ln in lines[1:]]
    chunked = any(k.lower() == "transfer-encoding" and v == "chunked" for k, v in headers)
    if chunked:
        out, rest = bytearray(), body
        while True:
            size, _, rest = rest.partition(b"\r\n")
            n = int(size, 16)
            if n == 0:
                break
            out += rest[:n]
            rest = rest[n + 2:]
        body = bytes(out)
    return {"status": status, "reason": reason, "headers": headers, "chunked": chunked, "body_b64": base64.b64encode(body).decode()}


async def run_one(app: web.Application, method: str, path: str, headers: dict, body: bytes | None = None) -> dict:
    proto = _Proto()
    loop = asyncio.get_running_loop()
    writer = StreamWriter(proto, loop)
    kw = {}
    if body is not None:   # POST 본문 -- aiohttp StreamReader 에 넣고 끝을 알린다
        from aiohttp.streams import StreamReader
        sr = StreamReader(proto, 2 ** 16, loop=loop)
        sr.feed_data(body)
        sr.feed_eof()
        kw["payload"] = sr
    req = make_mocked_request(method, path, headers=headers, app=app, writer=writer, protocol=proto, transport=proto.transport, **kw)
    try:
        resp = await app._handle(req)
    except web.HTTPException as exc:          # web_protocol._handle_request: resp = exc
        resp = exc
    except Exception:                         # web_protocol.handle_error(500) -- Accept 에 text/html 이 없을 때 모양
        resp = web.Response(status=500, text="500 Internal Server Error\n\nServer got itself in trouble", content_type="text/plain")
        resp.force_close()
    await resp.prepare(req)
    await resp.write_eof()
    return _parse(bytes(proto.transport.buf))


# ───────────────────────── 시나리오(파일 트리 + 요청) ─────────────────────────
def sqlite_bytes(ddl_rows: list[tuple[str, list]]) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.sqlite"
        con = sqlite3.connect(p)
        for sql, rows in ddl_rows:
            if rows:
                con.executemany(sql, rows)
            else:
                con.execute(sql)
        con.commit()
        con.close()
        return p.read_bytes()


AE = {"Accept-Encoding": "gzip, deflate, br, zstd"}   # 브라우저 모양 -- 원본은 «deflate» 를 먼저 고른다(부분 글자 일치)
GZ = {"Accept-Encoding": "gzip"}


def j(o, **kw) -> bytes:
    return json.dumps(o, **kw).encode()


def tree_static() -> dict[str, bytes]:
    big = "".join(f"function f{i}(a,b){{return a*{i}+b; /* 한글 주석 {i} */}}\n" for i in range(400))
    html = ('<!doctype html><html><head><link rel="stylesheet" href="styles.css?v=20261101a">'
            '<script src="app.js?v=old.1-x"></script><script src="liq_profile.js?v=z"></script><script src="other.js?v=keep"></script></head>'
            '<body><button class="tab" data-asset="eth">ETH</button><button data-asset="btc" class="tab">BTC</button>'
            '<button data-asset="hype" hidden>HYPE</button><button data-asset="xrp">XRP</button><button data-asset="sol" >SOL</button>'
            '<p>대시보드 · 시험용</p></body></html>\n')
    return {
        "dashboard/live/index.html": html.encode(),
        "dashboard/live/app.js": big.encode(),
        "dashboard/live/styles.css": b"body{color:#111}\n" * 300,
        "dashboard/live/liq_profile.js": b"var x=1;\n",
        "dashboard/live/other.js": b"var other=2;\n" * 50,
        "dashboard/live/sw.js": b"self.addEventListener('push',()=>{});\n",
        "dashboard/live/manifest.webmanifest": j({"name": "dash", "icons": []}),
        "dashboard/live/icons/badge-96.png": bytes(range(256)) * 4,
        "dashboard/live/fonts/a.ttf": b"\x00\x01\x00\x00" + b"ttf" * 100,
        "dashboard/live/data.json": j({"a": 1}),
        "dashboard/live/notes.txt": "글자\n".encode(),
        "dashboard/live/help.html": "<p>도움말</p>\n".encode(),   # 정적 .html 도 미들웨어가 no-cache
        "data/live/secret.json": b'{"secret": true}',
    }


def tree_api() -> dict[str, bytes]:
    now = NOW
    sr_rows = []
    ev = lambda a, look, side, grp: dict(k="ev", id=f"{a}_{look}_brk_{side}", a=a, d=a + 5, side=side, kind="brk", look=look, L=1800.25, s1h=.004, P=1801.5, p=.6, grp=grp)
    a0 = int(now) - 900
    sr_rows = [ev(a0, 60, 1, "top"), ev(a0, 120, 1, "bot"), ev(a0, 240, 1, "top"), ev(a0 + 100, 60, -1, "mid"), ev(a0 + 200, 60, -1, "bot"),
               ev(int(now) - 100, 60, 1, "bot"), dict(ev(int(now) - 50, 60, 1, "top"), lag=7), ev(int(now) - 200000, 60, 1, "top"),
               dict(k="res", id=f"{a0}_60_brk_1", y=1, hit_s=12, r=3.0), dict(k="res", id=f"{a0 + 200}_60_brk_-1", y=0, hit_s=40.5, r=2.0),
               dict(k="skip", a=1, why="warm")]
    sr_log = ("\n".join(json.dumps(r) for r in sr_rows) + "\n{깨진 줄\n[1,2]\n\"str\"\n").encode()
    regime_rows = [dict(ts="2026-11-16 19:00:00", coin=c, p_trend=0.3, regime="횡보", confidence=0.4, support=1.0, resistance=2.5) for c in ("ETH", "XRP")]
    regime_rows += [dict(ts="2026-11-16T20:00:00", coin="ETH", p_trend=0.91, regime="추세", confidence=0.8, support=3.0, resistance=4.0),
                    dict(ts="2026-11-16 21:00:00", coin="SOL", p_trend=0.5),   # 열쇠 빠짐 → 건너뜀
                    dict(ts="2026-11-16 21:00:00", coin="BTC", p_trend="0.25", regime="횡보", confidence=1, support=7, resistance=8)]
    regime = ("\n".join(json.dumps(r, ensure_ascii=False) for r in regime_rows) + "\n{깨진\n").encode()
    iceberg_ev = "\n".join(json.dumps(e) for e in [
        {"t_end": NOW - 90000, "px": 1.0}, {"t_end": NOW - 3600, "px": 3001.5, "side": "bid", "n": 3, "usd": 1.5e5, "note": "한글"},
        {"t_end": NOW - 10, "px": 3002.0, "side": "ask", "x": [1, 2.5, None, True]}]) + "\n"
    ob_ddl = [
        ("CREATE TABLE bars (t INTEGER PRIMARY KEY, warm INTEGER, imb50 REAL, eat_n_a INTEGER, wall_px_b REAL, wall_usd_b REAL, wall_px_a REAL, wall_usd_a REAL)", []),
        ("CREATE TABLE live (k INTEGER PRIMARY KEY, upd INTEGER, bar TEXT, marks TEXT)", []),
        ("CREATE TABLE marks (t INTEGER, bar_t INTEGER, side TEXT, kind TEXT, px REAL, usd REAL, dist_bp REAL)", []),
    ]
    t0 = (int(NOW) // 300 * 300 - 6 * 300) * 1000
    bars = [(t0 - 8 * 86400 * 1000, 0, 0.1, 1, 3000.0, 1e5, 3100.0, 2e5),
            (t0, 1, None, None, 2990.0, 5e4, None, None),
            (t0 + 300_000, 0, 0.12345, 2, 2990.0, 7e4, 3010.5, 1e5),
            (t0 + 600_000, 0, -0.0055, 0, 2990.0, None, 3010.5, 9e4),
            (t0 + 900_000, 0, 0.25, None, 2985.0, 1e4, 3010.5, 3e5),
            (t0 + 1_500_000, 0, 0.005, 1, 2985.0, 2e4, 3011.0, 1e5)]
    live_bar = {"t": t0 + 1_800_000, "warm": False, "imb50": 0.031415, "wall_px_b": 2985.0, "wall_usd_b": 3e4, "wall_px_a": 3011.0, "wall_usd_a": 5e4}
    live_marks = [{"t": t0 + 1_810_000, "bar_t": t0 + 1_800_000, "side": "ask", "kind": "eat", "px": 3011.0, "usd": 12345.5, "dist_bp": 1.234},
                  {"t": t0 + 300_100, "bar_t": t0 + 300_000, "side": "bid", "kind": "fake", "px": 2990.0, "usd": 1.0, "dist_bp": 0.5}]
    ob = sqlite_bytes(ob_ddl + [
        ("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)", bars),
        ("INSERT INTO live VALUES (?,?,?,?)", [(0, int(NOW * 1000) - 5000, json.dumps(live_bar), json.dumps(live_marks))]),
        ("INSERT INTO marks VALUES (?,?,?,?,?,?,?)", [(t0 + 310_000, t0 + 300_000, "ask", "eat", 3010.5, 20000.5, 2.345),
                                                      (t0 + 620_000, t0 + 600_000, "bid", "fake", 2990.0, None, -0.125)]),
    ])
    ob_stale = sqlite_bytes(ob_ddl + [("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)", bars[2:4]),
                                      ("INSERT INTO live VALUES (?,?,?,?)", [(0, int(NOW * 1000) - 60000, json.dumps(live_bar), "[]")])])
    m = int(NOW) // 60 * 60
    tre = sqlite_bytes([("CREATE TABLE treasury_fut_1m (symbol TEXT, ts_sec INTEGER, close REAL, contract TEXT)", []),
                        ("INSERT INTO treasury_fut_1m VALUES (?,?,?,?)",
                         [(s, m - 3600 * 30 - i * 60, 110.0 + k + i / 64, f"{s}Z6") for k, s in enumerate(("ZN", "ZB", "ZF", "ZT")) for i in range(5)]
                         + [(s, m - 3600 * 13 + i * 600, 110.0 + k + i / 32, f"{s}Z6") for k, s in enumerate(("ZN", "ZB", "ZF")) for i in range(80)]
                         + [("ZT", m - 100, None, "ZTZ6")])])
    xyz = sqlite_bytes([("CREATE TABLE hl_xyz_1m (coin TEXT, ts_sec INTEGER, close REAL)", []),
                        ("CREATE TABLE hl_xyz_ctx_1s (coin TEXT, recv_sec INTEGER, mid REAL, mark REAL, oracle REAL)", []),
                        ("INSERT INTO hl_xyz_1m VALUES (?,?,?)",
                         [("xyz:XYZ100", m - 86400 * 2 + i * 3600, 25000.0 + i, ) for i in range(30)]
                         + [("xyz:XYZ100", m - 1800 + i * 60, 25100.125 + i * 0.1) for i in range(31)]
                         + [("xyz:SP500", m - 1200 + i * 60, 6800.0 + i * 0.25) for i in range(21)] + [("xyz:SP500", m - 30, None)]),
                        ("INSERT INTO hl_xyz_ctx_1s VALUES (?,?,?,?,?)",
                         [("xyz:XYZ100", int(NOW) - 3, None, 25103.5, 25102.0), ("xyz:XYZ100", int(NOW) - 10, 25000.0, 1.0, None),
                          ("xyz:SP500", int(NOW) - 2, 6805.75, 6805.5, 6805.0)])])
    m5 = sqlite_bytes([("CREATE TABLE yahoo_5m (symbol TEXT, ts_sec INTEGER, close REAL)", []),
                       ("INSERT INTO yahoo_5m VALUES (?,?,?)", [("VIX", m - 7200 + i * 300, 17.5 + i / 8) for i in range(20)] + [("SPX", m, 1.0)])])
    cal = {"generated_at": datetime.fromtimestamp(NOW - 1800, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "ok": True,
           "events": [{"t": "2026-11-18T13:30:00Z", "name": "CPI", "imp": 3}], "tone": "warn", "subText": "다음 CPI"}
    return {
        "data/live/acct_notes.json": json.dumps({"2026-11-20": {"text": "메모 \"따옴표\" \\ 끝\n둘째 줄 😀", "at": 1795000000000},
                                                 "2026-11-19": {"text": "x", "at": 1.5}}, ensure_ascii=False).encode(),
        "data/live/acct_wrong.json": json.dumps([{"id": 7, "date": "2026-11-20", "text": "가짜 돌파", "tag": "가짜 돌파", "at": 1795000000000},
                                                 {"id": 8, "date": "2026-11-21", "text": "x\u0001y", "tag": "", "at": 1e3}], ensure_ascii=False).encode(),
        "data/live/risk_radar_latest.json": j({"updated": "2026-11-17T09:21:00Z", "params": "risk_radar_params_20261117b.json",
                                               "coins": {"ETH": {"sigma15_pct": 0.3123, "lo": -1.5e-05, "big": 1e16}, "SOXL": {"sigma15_pct": 0.9, "touch": [0.6, 0.4]},
                                                         "eth": {"dup": 1}}}),
        "dashboard/risk_radar_params_20261116.json": j({"touch_k": [0.25, 0.5], "coins": {"ETH": {"touch": [0.7, 0.5]}}}),
        "dashboard/risk_radar_params_20261117b.json": j({"touch_k": [0.25, 0.5, 1.0], "coins": {"ETH": {"touch": [0.71, 0.51, 0.2]}, "BTC": {}}}),
        "data/live/sr_lean/ethusdt_state.json": j({"ts": NOW - 5, "done": NOW - 2.5, "warm_until": NOW - 100, "rate": {"top": [989, 1752], "bot": [1269, 2804]}}),
        "data/live/sr_lean/ethusdt.jsonl": sr_log,
        "data/live/regime_band_forward.jsonl": regime,
        "data/live/liq_tab_eth.json": '{"ok":true,"x":[1,2.50,3e5],"한":"글"}'.encode(),
        "data/live/liq_tab_all.json": b'{"ok": true, "coins": {}}',
        "data/live/iceberg/eth_state.json": j({"n_forward": 12, "target": 150, "updated_at": NOW - 42.25, "valid": True}),
        "data/live/iceberg/eth_events.jsonl": iceberg_ev.encode(),
        "data/live/iceberg/xrp_state.json": j({"n_forward": 3, "target": None, "updated_at": NOW - 7, "valid": False}),
        "data/live/ob5m/ethusdt.sqlite": ob,
        "data/live/ob5m/xrpusdt.sqlite": ob_stale,
        "data/hot/treasury_futures.sqlite": tre,
        "data/hot/hl_xyz.sqlite": xyz,
        "data/hot/macro_5m.sqlite": m5,
        "data/live/macro_calendar_state.json": j(cal, ensure_ascii=False),
    }


def tree_api_variants() -> dict[str, bytes]:
    """같은 주소의 다른 갈래: 파일이 낡음·깨짐·빔."""
    old = {"generated_at": datetime.fromtimestamp(NOW - 7200, timezone.utc).isoformat(), "events": [1], "tone": "x"}
    return {
        "data/live/acct_notes.json": b"[1, 2]",                       # 사전이 아님 → {}
        "data/live/acct_wrong.json": b"{broken",                      # 깨짐 → []
        "data/live/risk_radar_latest.json": j({"updated": "2026-11-17T09:00:00Z", "coins": {"BTC": {"s": 1}}}),   # params 없음 → 기본 파일 · 지연
        "dashboard/risk_radar_params_20261116.json": j({"coins": {}}),
        "data/live/sr_lean/ethusdt_state.json": j({"done": None}),   # 기록 없음 · rate 기본값 · stale
        "data/live/regime_band_forward.jsonl": b"{bad\n",
        "data/live/iceberg/eth_state.json": j({"n_forward": 1, "target": 150, "updated_at": NOW - 1, "valid": True}),
        "data/live/iceberg/eth_events.jsonl": b'{"t_end": 1}\n{broken\n',   # 깨진 줄 → 원본은 500(ValueError 를 안 잡는다)
        "data/live/macro_calendar_state.json": j(old),
        "data/hot/hl_xyz.sqlite": b"not a database at all, just bytes................................................................",
    }


def tree_big() -> dict[str, bytes]:
    """꼬리 읽기(256KB·32KB) 경계: 잘린 첫 줄을 버린다."""
    rows = [json.dumps(dict(k="ev", id=f"{i}", a=int(NOW) - 3000 + i, d=int(NOW) - 2990 + i, side=1 if i % 2 else -1, kind="brk",
                            look=60, L=1800.0 + i / 4, P=1801.0, grp="top" if i % 3 else "bot")) for i in range(2500)]
    rows += [json.dumps(dict(k="res", id=f"{i}", y=i % 2, hit_s=i)) for i in range(0, 2500, 7)]
    row = lambda i, c="BTC": json.dumps(dict(ts=f"2026-10-0{1 + i % 9} 0{i % 10}:00:00", coin=c, p_trend=i / 1000, regime="r", confidence=0.5,
                                            support=1, resistance=2)) + "\n"
    head = "".join(row(i) for i in range(300))
    ada = row(999, "ADA")                                  # 꼬리 32KB 의 첫 줄이 «통째로» 이 줄이 되게 맞춘다 -- 원본은 그래도 버린다
    tail = "".join(row(i) for i in range(300, 700))
    while len(ada.encode()) + len(tail.encode()) < 32768:
        tail += row(1)
    tail = tail[: 32768 - len(ada.encode())]               # ASCII 라 글자 = 바이트
    tail = tail[: tail.rfind("\n") + 1]
    pad = 32768 - len(ada) - len(tail)                      # 모자란 만큼 마지막 줄 앞에 공백 줄(빈 줄은 깨진 줄로 건너뜀)
    reg = head + ada + tail + " " * (pad - 1) + "\n" if pad > 0 else head + ada + tail
    assert len(reg.encode()) - 32768 == len(head.encode()), "자르는 자리가 ADA 줄 머리가 아니다"
    return {
        "data/live/sr_lean/ethusdt_state.json": j({"done": NOW - 100, "warm_until": None, "rate": {"top": [1, 3], "bot": [5, 8]}}),
        "data/live/sr_lean/ethusdt.jsonl": ("\n".join(rows) + "\n").encode(),
        "data/live/regime_band_forward.jsonl": reg.encode(),
    }


REQ_STATIC = [
    ("GET", "/", {}),
    ("GET", "/dashboard/live/", AE),
    ("GET", "/dashboard/live", {}),
    ("GET", "/dashboard/live/index.html", GZ),
    ("GET", "/dashboard/live/app.js", {}),
    ("GET", "/dashboard/live/app.js", AE),
    ("GET", "/dashboard/live/app.js", GZ),
    ("GET", "/dashboard/live/styles.css?v=abc", {"Accept-Encoding": "deflate"}),
    ("GET", "/dashboard/live/other.js", AE),
    ("GET", "/dashboard/live/icons/badge-96.png", AE),
    ("GET", "/dashboard/live/fonts/a.ttf", {}),
    ("GET", "/dashboard/live/data.json", AE),
    ("GET", "/dashboard/live/notes.txt", {}),
    ("GET", "/dashboard/live/help.html", AE),
    ("GET", "/dashboard/live/sw.js", AE),
    ("GET", "/dashboard/live/manifest.webmanifest", {}),
    ("GET", "/dashboard/live/missing.js", {}),
    ("GET", "/dashboard/live/fp_daily.js", AE),   # ASSET_FILES 인데 없는 파일: 404 에도 압축이 걸린다(빈 zlib 흐름)
    ("GET", "/dashboard/live/icons", {}),
    ("GET", "/dashboard/live/icons/", {}),
    ("GET", "/api/nope", {}),
    ("GET", "/favicon.ico", {}),
    ("POST", "/api/sr-lean", {}),
    ("POST", "/dashboard/live/app.js", {}),
]
REQ_API = [
    ("GET", "/api/acct-notes", {}), ("GET", "/api/acct-notes", AE), ("GET", "/api/acct-wrong", GZ),
    ("GET", "/api/risk-radar", {}), ("GET", "/api/sr-lean", AE), ("GET", "/api/regime-band", {}),
    ("GET", "/api/liq-tab", {}), ("GET", "/api/liq-tab?asset=all", AE), ("GET", "/api/liq-tab?asset=BTC", {}), ("GET", "/api/liq-tab?asset=doge", {}),
    ("GET", "/api/iceberg", {}), ("GET", "/api/iceberg?asset=xrp", {}), ("GET", "/api/iceberg?asset=btc", {}),
    ("GET", "/api/ob5m", AE), ("GET", "/api/ob5m?asset=xrp", {}), ("GET", "/api/ob5m?asset=sol", {}), ("GET", "/api/ob5m?asset=doge", {}),
    ("GET", "/api/treasury-futures", {}), ("GET", "/api/treasury-futures?h=24", AE),
    ("GET", "/api/xyz100", {}), ("GET", f"/api/xyz100?since={int(NOW) // 60 * 60 - 300}", {}), ("GET", "/api/xyz100?since=abc", {}),
    ("GET", "/api/xyz100?since=99999999999999999999", {}), ("GET", "/api/xyz100?since=-5", {}),
    ("GET", "/api/macro-today", {}), ("GET", "/api/macro-calendar", {}),
]
REQ_VARIANT = [("GET", p, {}) for p in ("/api/acct-notes", "/api/acct-wrong", "/api/risk-radar", "/api/sr-lean", "/api/regime-band",
                                        "/api/iceberg", "/api/macro-calendar", "/api/xyz100", "/api/macro-today", "/api/treasury-futures",
                                        "/api/liq-tab", "/api/ob5m")]
REQ_BIG = [("GET", "/api/sr-lean", {}), ("GET", "/api/regime-band", {})]


async def scenario(name: str, files: dict[str, bytes], reqs: list) -> dict:
    with tempfile.TemporaryDirectory() as td:
        R = Path(td).resolve()
        (R / "dashboard" / "live").mkdir(parents=True)   # add_static 은 폴더가 없으면 앱을 못 만든다(원본도 같다)
        for rel, data in files.items():
            p = R / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            os.utime(p, ns=(MTIME_NS, MTIME_NS))
        app = make_app(R)
        app.freeze()
        cases = []
        for method, path, headers in reqs:
            out = await run_one(app, method, path, headers)
            cases.append({"method": method, "path": path, "req_headers": headers, **out})
            etag = next((v for k, v in out["headers"] if k.lower() == "etag"), None)
            if etag and out["status"] == 200:   # 같은 요청 + If-None-Match → 304 (본문 해시 ETag · 정적 파일 ETag 둘 다)
                h2 = {**headers, "If-None-Match": f'"other", {etag}'}
                cases.append({"method": method, "path": path, "req_headers": h2, **await run_one(app, method, path, h2)})
    return {"name": name, "files": {k: base64.b64encode(v).decode() for k, v in files.items()}, "cases": cases}


async def main() -> None:
    out = {"now": NOW, "mtime_ns": MTIME_NS, "dashboard_assets": ",".join(DASHBOARD_ASSETS),
           "scenarios": [await scenario("static", tree_static(), REQ_STATIC),
                         await scenario("api", tree_api(), REQ_API),
                         await scenario("api_variants", tree_api_variants(), REQ_VARIANT),
                         await scenario("api_big", tree_big(), REQ_BIG)]}
    # round(x, nd)·round(x) 정답(정확한 반값·이진 오차 근처·큰 값·음수) -- 레이더 age_s·ob5m tilt_pct·sr_lean pct 가 쓴다
    import random
    rnd = random.Random(20261121)
    xs = [k / 8 for k in range(-40, 41)] + [k / 1000 + 0.0005 for k in range(-3000, 3000, 7)] + [2.675, 1.005, 0.125, 0.375, 1e15 + 0.5, 92863.45, -0.05]
    xs += [rnd.uniform(-1e6, 1e6) for _ in range(3000)] + [round(rnd.uniform(-100, 100), 3) for _ in range(3000)]
    out["rounds"] = [[x, nd, round(x, nd)] for x in xs for nd in (1, 2, 6)] + [[x, -1, round(x)] for x in xs]
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    n = sum(len(s["cases"]) for s in out["scenarios"])
    print(f"{OUT.relative_to(ROOT)}: 시나리오 {len(out['scenarios'])} · 요청 {n}")
    for s in out["scenarios"]:
        for c in s["cases"]:
            print(f"  {s['name']:12} {c['method']:4} {c['path'][:60]:60} {c['status']} {len(base64.b64decode(c['body_b64']))}B"
                  f"{' chunked' if c['chunked'] else ''}")


if __name__ == "__main__":
    asyncio.run(main())
