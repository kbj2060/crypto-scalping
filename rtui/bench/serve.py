"""실기기 측정 서버(127.0.0.1 만) -- rtui 와 지금 화면(dashboard/live)을 같은 합성 자료·같은 가짜 @trade 소켓으로 띄운다. 바깥 연결 없음.
  /rtui/*            rtui 정적 파일(index.html · dist · bench/probe.js)
  /dashboard/live/   지금 화면 -- index.html 을 읽어 머리에 bench/probe.js 를 끼우고 바깥 글꼴 주소를 로컬로 바꿔 준다(파일은 안 고침) · 나머지는 정적
  /api/footprint · market-history · footprint-week = rtui/compare/fixture/*.json · 그 밖 /api/* = 404
  /ws/trade          가짜 @trade: 페이지가 {"cmd":"pre","T"} → 체결 1건 · {"cmd":"start","rate","secs","off"} → 초당 rate 건 secs 초(각 메시지 끝 "_s" = 보낸 시각 ms)
  POST /bench/result 결과 → 표준출력 한 줄 + rtui/bench/results/<page>_<rate>_<시각>.json
python3 -I rtui/bench/serve.py [--port 8790]  (README.md)"""
import argparse, asyncio, gzip, json, pathlib, random, re, time
from aiohttp import web, WSMsgType

RTUI = pathlib.Path(__file__).resolve().parent.parent
WT = RTUI.parent
LIVE, FX, OUT = WT / "dashboard/live", RTUI / "compare/fixture", RTUI / "bench/results"
API = {"/api/footprint": "footprint.json", "/api/market-history": "market-history.json", "/api/footprint-week": "footprint-week.json"}
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
        return web.Response(body=(FX / f).read_bytes(), content_type="application/json", headers=NOCACHE)
    return web.Response(status=404, text="{}", content_type="application/json")


async def h_ws(req):
    ws = web.WebSocketResponse()
    await ws.prepare(req)
    if req.path != "/ws/trade":   # 다른 소켓은 받자마자 닫는다(페이지가 연 바깥 소켓 = probe 가 기록)
        await ws.close(); return ws
    rng = random.Random(7); tid = [5_000_000]; px = [1745.0]

    def trade(T):
        if rng.random() < 0.5: px[0] = round(px[0] + (rng.random() - 0.5) * 0.6, 2)
        tid[0] += 1
        return (f'{{"e":"trade","E":{T:.0f},"T":{T:.0f},"s":"ETHUSDT","t":{tid[0]},"p":"{px[0]:.2f}","q":"{rng.random() * 2:.3f}",'
                f'"m":{"true" if rng.random() < 0.5 else "false"},"_s":{time.time() * 1000:.3f}}}')
    task = None

    async def stream(rate, secs, off):
        t0, n = time.perf_counter(), 0
        while (el := time.perf_counter() - t0) < secs:
            due = int(el * rate)
            while n < due:
                n += 1; await ws.send_str(trade(time.time() * 1000 + off))
            await asyncio.sleep(0.004)
        print(f"[ws] 보냄 {n}건 ({rate}/초 × {secs}초)", flush=True)
    async for m in ws:
        if m.type != WSMsgType.TEXT:
            continue
        try:
            c = json.loads(m.data)
        except ValueError:
            continue
        if c.get("cmd") == "pre":
            await ws.send_str(trade(float(c["T"])))
        elif c.get("cmd") == "start" and task is None:
            rate, secs = min(int(c["rate"]), 5000), min(float(c["secs"]), 240)
            task = asyncio.create_task(stream(rate, secs, float(c["off"])))
    if task: task.cancel()
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
                   f' · <a href="/dashboard/live/?bench=8&rate={r}">지금 화면</a></li>' for r in (100, 300, 800))
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
    app.router.add_post("/bench/result", h_result)
    left = sorted(set(re.findall(r"(?:https?|wss?)://[\w.\-]+", live_html())) - {"http://www.w3.org"})
    print(f"http://127.0.0.1:{a.port}/  · 지금 화면 html 에 남은 바깥 주소: {left or '없음'}", flush=True)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None)   # 127.0.0.1 만(0.0.0.0 금지)


if __name__ == "__main__":
    main()
