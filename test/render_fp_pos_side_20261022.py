#!/usr/bin/env python3
"""풋프린트 «포지션» 보기 롱·숏 분리 -- 시안 A(이어 붙인 막대) 확정판 (2026-10-22 사용자 «포지션을 롱/숏 비율대로 · 시안 3개» → «A 로 확정»)
-- 서버·거래소 없이 실데이터 사본(서버 10-22 00:24 KST 풋프린트·캔들·포지션 + 같은 시점 롱·숏 분리 장부)으로 실제 브라우저에서 그린다.

    python3 -u test/render_fp_pos_side_20261022.py FIXDIR [OUT_DIR]

FIXDIR: footprint_asset_eth_bars_288_since_0.json · market-history_asset_eth.json · position-profile_asset_eth_days_{1,7,30}.json · side_split.json
검사: 1d·7d·30d 에서 초록·빨강 막대가 있고 «롱 위 … · 숏 위 …» 줄이 있으며 JS 오류 0 · 바깥 호스트 요청은 글꼴 CDN 만 통과, 나머지 abort.
"""
import json, pathlib, sys, time
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright

WT = pathlib.Path(__file__).resolve().parents[1]
FIX = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/render_fp_pos_side_20261022"); OUT.mkdir(parents=True, exist_ok=True)
DASH = WT / "dashboard/live"; ORIGIN = "http://dash.local"
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html"}
fp = json.loads((FIX / "footprint_asset_eth_bars_288_since_0.json").read_text())
mh = json.loads((FIX / "market-history_asset_eth.json").read_text())
side = json.loads((FIX / "side_split.json").read_text())
prof = {}
for d in ("1", "7", "30"):
    j = json.loads((FIX / f"position-profile_asset_eth_days_{d}.json").read_text())
    j["win"][d]["long"], j["win"][d]["short"] = side[d]["long"], side[d]["short"]
    prof[d] = j
shift = int(time.time() // 300 * 300) - fp["bars"][-1]["time"]          # 사본 마지막 봉 = 지금 형성 봉
for b in fp["bars"]: b["time"] += shift
for c in mh["candles"]: c["time"] += shift
fp["okxLive"] = {}
fails, errs = [], []
FONT_HOSTS = {"fonts.googleapis.com", "fonts.gstatic.com", "cdn.jsdelivr.net"}


def handle(route):
    u = route.request.url; pr = urlparse(u)
    if route.request.method == "GET" and pr.hostname in FONT_HOSTS: return route.continue_()   # 한글 글꼴(Pretendard) -- 로컬 크로미움엔 한글 글꼴이 없다
    if not u.startswith(ORIGIN) or route.request.method != "GET": return route.abort()
    if pr.path == "/api/footprint": return route.fulfill(body=json.dumps(fp), content_type="application/json")
    if pr.path == "/api/market-history": return route.fulfill(body=json.dumps(mh), content_type="application/json")
    if pr.path == "/api/position-profile":
        d = parse_qs(pr.query).get("days", ["7"])[0]
        return route.fulfill(body=json.dumps(prof.get(d, prof["7"])), content_type="application/json")
    if pr.path.startswith("/api/"): return route.fulfill(status=503, body="{}")
    name = "index.html" if pr.path.rstrip("/") == "/dashboard/live" else pr.path.split("/dashboard/live/", 1)[-1]
    f = DASH / name
    return route.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream")) if f.is_file() else route.fulfill(status=404, body="")


MEASURE = """() => { const svg = document.getElementById('candleSvgSnapshot');
  const rs = [...svg.querySelectorAll('rect')], ts = [...svg.querySelectorAll('text')].map((t) => t.textContent);
  return { good: rs.filter((r) => r.getAttribute('fill') === 'var(--good)').length, bad: rs.filter((r) => r.getAttribute('fill') === 'var(--bad)').length,
           head: ts.filter((t) => /^(롱|숏) 위 /.test(t)).length, avg: ts.filter((t) => /^(↑ |↓ )?(롱|숏) 평균/.test(t)).length }; }"""

with sync_playwright() as p:
    b = p.chromium.launch()
    for sty in "A":
        for win in (1, 7, 30):
            ctx = b.new_context(viewport={"width": 1920, "height": 1080})
            pg = ctx.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.route(lambda u: True, handle); pg.route_web_socket(lambda u: True, lambda ws: None)
            pg.add_init_script(f"try {{ localStorage.clear(); localStorage.setItem('fpCol','pos'); localStorage.setItem('fpPosWin','{win}'); }} catch (e) {{}}"
)
            pg.goto(f"{ORIGIN}/dashboard/live/", wait_until="load"); pg.wait_for_timeout(3500)
            pg.evaluate("() => { const f = document.getElementById('ofab'); if (f) f.style.visibility = 'hidden'; }")
            m = pg.evaluate(MEASURE); tag = f"{sty}_{win}d"
            print(f"[{tag}] 초록 {m['good']} · 빨강 {m['bad']} · 머리 줄 {m['head']} · 평균 줄 {m['avg']}", flush=True)
            if m["good"] + m["bad"] < 10 or m["head"] != 2: fails.append(f"[{tag}] {m}")
            pg.locator("#candleSvgSnapshot").screenshot(path=str(OUT / f"pos_{tag}.png"))
            ctx.close()
    b.close()
if errs: fails.append(f"JS 오류 {errs[:3]}")
print("FAIL\n  " + "\n  ".join(fails) if fails else "PASS", flush=True)
sys.exit(1 if fails else 0)
