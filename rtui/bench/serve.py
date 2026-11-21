"""실기기 측정 서버(127.0.0.1 만) -- rtui 와 지금 화면(dashboard/live)을 같은 합성 자료·같은 가짜 @trade 소켓으로 띄운다. 바깥 연결 없음.
  /rtui/*            rtui 정적 파일(index.html · dist · bench/probe.js)
  /dashboard/live/   지금 화면 -- index.html 을 읽어 머리에 bench/probe.js 를 끼우고 바깥 글꼴 주소를 로컬로 바꿔 준다(파일은 안 고침) · 나머지는 정적
  /api/footprint · market-history · footprint-week · liquidation-5m-signal·history = rtui/compare/fixture/*.json · 그 밖 /api/* = 404
  /rt                가짜 rtcore 허브(?src=push 일 때 rtui 가 붙는다 · 같은 체결 열을 허브 모양으로 -- h_ws 주석)
  /ws/trade          가짜 @trade: 페이지가 {"cmd":"pre","T"} → 체결 1건 · {"cmd":"start","rate","secs","off"} → 초당 rate 건 secs 초(각 메시지 끝 "_s" = 보낸 시각 ms)
  POST /bench/result 결과 → 표준출력 한 줄 + rtui/bench/results/<page>_<rate>_<시각>.json
python3 -I rtui/bench/serve.py [--port 8790]  (README.md)"""
import argparse, asyncio, gzip, json, pathlib, random, re, time
from aiohttp import web, WSMsgType

RTUI = pathlib.Path(__file__).resolve().parent.parent
WT = RTUI.parent
LIVE, FX, OUT = WT / "dashboard/live", RTUI / "compare/fixture", RTUI / "bench/results"
API = {"/api/footprint": "footprint.json", "/api/market-history": "market-history.json", "/api/footprint-week": "footprint-week.json",
       "/api/liquidation-5m-signal": "liquidation-5m-signal.json", "/api/liquidation-5m-history": "liquidation-5m-history.json"}
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html", ".png": "image/png", ".wasm": "application/wasm",
      ".json": "application/json", ".ttf": "font/ttf", ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml"}
FONT_CSS = "@font-face{font-family:'Pretendard Variable';src:url(/rtui/fonts/Pretendard-Bold.ttf) format('truetype');font-weight:45 920;}"
NOCACHE = {"Cache-Control": "no-store"}


def static(root, rel):
    f = (root / rel).resolve()
    if root.resolve() not in f.parents or not f.is_file():   # 경로 밖(../)·없는 파일
        raise web.HTTPNotFound()
    return web.Response(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream"), headers=NOCACHE)


def live_html():
    """지금 화면 index.html: probe 를 맨 앞에 · 구글 글꼴 줄은 빼고 · jsdelivr Pretendard 는 로컬 글꼴 css 로"""
    h = (LIVE / "index.html").read_text(encoding="utf-8")
    h = "\n".join(l for l in h.split("\n") if "fonts.googleapis.com" not in l and "fonts.gstatic.com" not in l)
    h = re.sub(r'https://cdn\.jsdelivr\.net/[^"]*pretendard[^"]*', "/bench/font.css", h)
    h = h.replace("<head>", '<head>\n  <script src="/rtui/bench/probe.js"></script>', 1)
    return h


async def h_rtui(req):
    return static(RTUI, req.match_info["p"] or "index.html")


async def h_font(req):
    return web.Response(text=FONT_CSS, content_type="text/css")


async def h_live(req):
    rel = req.match_info["p"]
    if rel in ("", "index.html"):
        return web.Response(text=live_html(), content_type="text/html", headers=NOCACHE)
    if rel == "sw.js":
        raise web.HTTPNotFound()
    return static(LIVE, rel)


async def h_api(req):
    f = API.get(req.path)
    if req.method == "GET" and f and (FX / f).is_file():
        since = float(req.query.get("since") or 0)
        if req.path == "/api/footprint" and since > 0:   # 실서버처럼 증분(since 이후 봉 · full=false) -- 늘 전량(100KB)을 주면 묻기 쪽만 부풀려진다(2026-11-21 측정)
            d = json.loads((FX / f).read_text())
            d["bars"], d["full"] = [b for b in d["bars"] if b["time"] >= since], False
            return web.json_response(d, headers=NOCACHE)
        return web.Response(body=(FX / f).read_bytes(), content_type="application/json", headers=NOCACHE)
    return web.Response(status=404, text="{}", content_type="application/json")


async def h_ws(req):
    """/ws/trade = 바이낸스 @trade 모양 · /rt = rtcore 허브 모양(가짜 허브 -- rtcore/src/proto.rs Stamped: 구독 → 스냅숏(풋프린트 = fixture) →
    체결 사건 + 16ms 마다 형성 봉 바뀐 칸). 같은 시드라 두 길의 체결 열이 같다. 명령(pre·start)은 같은 소켓으로 받는다(진짜 허브는 무시하는 글)."""
    ws = web.WebSocketResponse()
    await ws.prepare(req)
    hub = req.path == "/rt"
    if req.path != "/ws/trade" and not hub:   # 다른 소켓은 받자마자 닫는다(페이지가 연 바깥 소켓 = probe 가 기록)
        await ws.close(); return ws
    rng = random.Random(7); tid = [5_000_000]; px = [1745.0]; seq = [0]
    fx = json.loads((FX / "footprint.json").read_text()) if hub else {}
    bucket = float(fx.get("bucket") or 0.5)
    cells, dirty = {}, {}   # (봉, 칸 키) → 칸 [가격, 매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도]
    for b in fx.get("bars", []):
        for l in b["levels"]: cells[(b["time"], round(l[0] / bucket))] = list(l) + [0.0] * (7 - len(l))
    head = {k: fx.get(k) for k in ("bucket", "barSeconds", "barsExpected", "ready", "retailMaxUsd", "whaleMinUsd")}

    def stamp(ty, ex_ms, **kw):   # 허브 한 통(tx_us = 지금 · 받는 쪽 지연의 시작)
        seq[0] += 1; us = int(time.monotonic() * 1e6)   # 단조 시계(벽시계는 WSL 시각 맞추기로 수백 ms 씩 뛴다 -- 2026-11-21 측정 중 실제로 ~750ms)
        return json.dumps({"s": "ethusdt", "seq": seq[0], "ex_ms": int(ex_ms), "rx_us": us, "tx_us": us, "type": ty, **kw}, separators=(",", ":"))

    def trade(T):
        if rng.random() < 0.5: px[0] = round(px[0] + (rng.random() - 0.5) * 0.6, 2)
        tid[0] += 1
        q, sell = float(f"{rng.random() * 2:.3f}"), rng.random() < 0.5
        if not hub:
            return (f'{{"e":"trade","E":{T:.0f},"T":{T:.0f},"s":"ETHUSDT","t":{tid[0]},"p":"{px[0]:.2f}","q":"{q:.3f}",'
                    f'"m":{"true" if sell else "false"},"_s":{time.monotonic() * 1000:.3f}}}')   # _s = 단조 시계(ms)
        bar, key = int(T // 300000) * 300, round(px[0] / bucket)   # 파이썬 round = 동률 짝수(코어와 같은 격자)
        c = cells.setdefault((bar, key), [round(key * bucket, 2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        c[2 if sell else 1] = round(c[2 if sell else 1] + q, 3); dirty[(bar, key)] = c
        return stamp("trade", T, id=tid[0], px=px[0], qty=q, sell=sell)

    async def flush():   # 코어 FLUSH_MS 16 과 같은 묶음 -- 고래·리테일 되묶기는 안 한다(형성 봉은 화면이 브라우저 칸을 그리는 구간이라 값이 안 보인다)
        while True:
            await asyncio.sleep(0.016)
            if dirty:
                by = {}
                for (bar, _), c in sorted(dirty.items()): by.setdefault(bar, []).append(c)
                dirty.clear()
                await ws.send_str(stamp("footprint", time.time() * 1000, full=False, **head, bars=[{"time": t, "levels": v} for t, v in by.items()]))
    task, ft = None, None

    async def stream(rate, secs, off):
        t0, n = time.perf_counter(), 0
        while (el := time.perf_counter() - t0) < secs:
            due = int(el * rate)
            while n < due:
                n += 1; await ws.send_str(trade(time.time() * 1000 + off))
            await asyncio.sleep(0.004)
        print(f"[{'hub' if hub else 'ws'}] 보냄 {n}건 ({rate}/초 × {secs}초)", flush=True)
    async for m in ws:
        if m.type != WSMsgType.TEXT:
            continue
        try:
            c = json.loads(m.data)
        except ValueError:
            continue
        if hub and "subscribe" in c and ft is None:
            fp = {"full": True, **head, "bars": fx.get("bars", [])}
            await ws.send_str(stamp("snapshot", 0, book=None, mark=None, trades=[], footprint=fp))
            ft = asyncio.create_task(flush())
        elif c.get("cmd") == "pre":
            await ws.send_str(trade(float(c["T"])))
        elif c.get("cmd") == "start" and task is None:
            rate, secs = min(int(c["rate"]), 5000), min(float(c["secs"]), 240)
            task = asyncio.create_task(stream(rate, secs, float(c["off"])))
    for t in (task, ft):
        if t: t.cancel()
    return ws


def bundle():
    """묶음 크기(원본·gzip 바이트) -- 서버가 압축 없이 주므로 실기기 전송량 = 원본"""
    def sz(fs): b = [f.read_bytes() for f in fs if f.is_file()]; return {"raw": sum(map(len, b)), "gzip": sum(len(gzip.compress(x, 6)) for x in b)}
    ver = (RTUI / "dist/VERSION").read_text().strip() if (RTUI / "dist/VERSION").is_file() else ""
    return {"rtui": sz([RTUI / f"dist/rtui-{ver}.js", RTUI / f"dist/rtui-{ver}_bg.wasm", RTUI / "index.html"]),
            "js": sz([LIVE / f for f in ("index.html", "app.js", "fp_daily.js", "liq_profile.js", "liq_tab.js", "bubble_map.js", "styles.css")])}


async def h_result(req):
    r = await req.json()
    r["bundle"] = bundle()
    OUT.mkdir(exist_ok=True)
    f = OUT / f"{r.get('page', 'x')}_{r.get('params', {}).get('rate', 'x')}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    s = r.get("summary", {})
    print(f"[결과] {f.name} " + json.dumps(s, ensure_ascii=False), flush=True)
    return web.json_response({"file": str(f)})


async def h_index(req):
    p = req.app["port"]
    rows = "".join(f'<li>{r}/초: <a href="/rtui/index.html?asset=eth&bench=8&rate={r}">rtui</a> · <a href="/rtui/index.html?asset=eth&bench=8&rate={r}&aa=0">rtui aa=0</a>'
                   f' · <a href="/rtui/index.html?asset=eth&bench=8&rate={r}&src=push">rtui 밀기</a> · <a href="/dashboard/live/?bench=8&rate={r}">지금 화면</a></li>' for r in (100, 300, 800))
    return web.Response(text=f"<!doctype html><meta charset=utf-8><title>rtui bench</title><h3>127.0.0.1:{p} 측정</h3><ul>{rows}</ul>", content_type="text/html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    app = web.Application(client_max_size=8 << 20)
    app["port"] = a.port
    app.router.add_get("/", h_index)
    app.router.add_get("/rtui/{p:.*}", h_rtui)
    app.router.add_get("/dashboard/live/{p:.*}", h_live)
    app.router.add_get("/bench/font.css", h_font)
    app.router.add_route("*", "/api/{p:.*}", h_api)
    app.router.add_get("/ws/{p:.*}", h_ws)
    app.router.add_get("/rt", h_ws)
    app.router.add_post("/bench/result", h_result)
    left = sorted(set(re.findall(r"(?:https?|wss?)://[\w.\-]+", live_html())) - {"http://www.w3.org"})
    print(f"http://127.0.0.1:{a.port}/  · 지금 화면 html 에 남은 바깥 주소: {left or '없음'}", flush=True)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None)   # 127.0.0.1 만(0.0.0.0 금지)


if __name__ == "__main__":
    main()
