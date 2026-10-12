#!/usr/bin/env python3
"""휴대폰 풋프린트 = 체결 칸 없이 5분봉만(2026-10-12 사용자 «모바일에서는 5분봉에 있는 체결박스들만 모두 제거 · 데스크탑은 수정하지 마», 시안 B)
-- 서버·거래소 **없이** 실데이터 사본(test/fixtures/fp_eth_quiet_20261010.json)으로 실제 브라우저에서 잰다.

    python3 -u test/render_fp_mobile_no_boxes_20261012.py [OUT_DIR] [--dash DIR]     # --dash = 다른 dashboard/live 로(음성 대조: 옛 판은 실패해야)

1h 창(12봉): 390 폭 = 체결 칸(«· 매수/매도» 툴팁 rect)·POC 테두리 0개 · 5분봉 몸통 중앙값 > 8px(칸 폭의 60%)
            1920 폭 = 체결 칸 있음 · 몸통 ≤ 8px(옛 옆 캔들 그대로). JS 오류 0 · 바깥 호스트 요청은 전부 abort.
"""
import argparse, copy, json, pathlib, sys, time
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

WT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("out", nargs="?", default="/tmp/render_fp_mobile_no_boxes_20261012"); ap.add_argument("--dash")
A = ap.parse_args()
DASH = pathlib.Path(A.dash) if A.dash else WT / "dashboard/live"
OUT = pathlib.Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
FIX = json.loads((WT / "test/fixtures/fp_eth_quiet_20261010.json").read_text("utf-8"))
ORIGIN = "http://dash.local"
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html"}
fp, cs = copy.deepcopy(FIX["footprint"]), copy.deepcopy(FIX["candles"])
d = int(time.time() // 300 * 300) - fp["bars"][-1]["time"]         # 사본 마지막 봉 = 지금 형성 봉
for b in fp["bars"]: b["time"] += d
for c in cs: c["time"] += d
fp["okxLive"] = {}
MEASURE = """() => { const svg = document.getElementById('candleSvgSnapshot'); const rs = [...svg.querySelectorAll('rect')];
  const cells = rs.filter((r) => /· (매수|매도) /.test(r.textContent)).length;
  const poc = rs.filter((r) => r.getAttribute('stroke') === 'var(--amber)' && r.getAttribute('fill') === 'none').length;
  const bodies = rs.filter((r) => /--(good|bad)/.test(r.getAttribute('fill') || '') && r.getAttribute('rx') !== null).map((r) => +r.getAttribute('width')).sort((a, b) => a - b);
  return { cells, poc, n: bodies.length, med: bodies.length ? bodies[bodies.length >> 1] : 0 }; }"""
fails, errs = [], []


def handle(route):
    u = route.request.url; pr = urlparse(u)
    if not u.startswith(ORIGIN): return route.abort()
    if route.request.method != "GET": return route.abort()
    if pr.path == "/api/footprint": return route.fulfill(body=json.dumps(fp), content_type="application/json")
    if pr.path == "/api/market-history": return route.fulfill(body=json.dumps({"asset": "eth", "candles": cs}), content_type="application/json")
    if pr.path.startswith("/api/"): return route.fulfill(status=503, body="{}")
    name = "index.html" if pr.path.rstrip("/") == "/dashboard/live" else pr.path.split("/dashboard/live/", 1)[-1]
    f = DASH / name
    return route.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream")) if f.is_file() else route.fulfill(status=404, body="")


with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h, mobile in ((390, 844, True), (1920, 1080, False)):
        ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=2 if mobile else 1)
        pg = ctx.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route(lambda u: True, handle); pg.route_web_socket(lambda u: True, lambda ws: None)
        pg.add_init_script("try { localStorage.clear(); } catch (e) {}")
        pg.goto(f"{ORIGIN}/dashboard/live/", wait_until="load"); pg.wait_for_timeout(2500)
        pg.evaluate("() => { const f = document.getElementById('ofab'); if (f) f.style.visibility = 'hidden'; }")
        pg.click("#chartWindowTabs [data-bars='12']"); pg.wait_for_timeout(1200)
        m = pg.evaluate(MEASURE); tag = f"{w}"
        print(f"[{tag}] 체결 칸 {m['cells']} · POC 테두리 {m['poc']} · 5분봉 몸통 {m['n']}개 중앙 폭 {m['med']:.1f}px", flush=True)
        if m["n"] < 10: fails.append(f"[{tag}] 5분봉 몸통을 못 찾음 {m['n']}")
        if mobile and (m["cells"] or m["poc"]): fails.append(f"[{tag}] 휴대폰에 체결 칸 {m['cells']} · POC 테두리 {m['poc']} 남음")
        if mobile and m["med"] <= 8: fails.append(f"[{tag}] 휴대폰 5분봉이 안 커짐 {m['med']:.1f}px")
        if not mobile and not m["cells"]: fails.append(f"[{tag}] 데스크탑 체결 칸이 사라짐")
        if not mobile and m["med"] > 8: fails.append(f"[{tag}] 데스크탑 옆 5분봉 폭이 바뀜 {m['med']:.1f}px")
        pg.locator("#candleSvgSnapshot").screenshot(path=str(OUT / f"fp_{tag}.png"))
        ctx.close()
    b.close()
if errs: fails.append(f"JS 오류 {errs[:3]}")
print("FAIL\n  " + "\n  ".join(fails) if fails else "PASS", flush=True)
sys.exit(1 if fails else 0)
