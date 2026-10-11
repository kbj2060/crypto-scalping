#!/usr/bin/env python3
"""청산맵 «지나간 칸 지우기»가 풋프린트 밀도 층과 같은 칸을 지우는가(2026-10-11 사용자 «업데이트할 때마다 봉이 안 채워져 ·
풋프린트엔 연한 파랑인데 청산맵엔 청산추정 0») -- 서버·거래소 **없이** 실제 브라우저로 잰다.

    python3 -u test/render_liqmap_sweep_20261011.py [OUT_DIR] [--dash DIR]     # --dash = 다른 dashboard/live 로(음성 대조)

원천 = 서버 실응답 사본 test/fixtures/liqmap_eth_20261011.json(10-11 05:01:09 UTC · 지도 생성 05:00:11 · 현재가 2505.9).
대상 칸 = 996(가운데 2495.88 · 범위 2494.63~2497.13, 서버 값 > 0). 시계를 05:02:00 에 고정해 흘리고 가짜 WS 체결로 가격을 민다:
  ① 지도 뒤 체결 2496.6(칸 위쪽 가장자리만 스침 · 가운데 안 지남) → 풋프린트 밀도 층은 안 끊김(liqSweepIdx = -1) · 청산맵 칸 > 0 이어야
  ② 대조: 지도 뒤 체결 2495.6(가운데 지남) → 둘 다 지움
  ③ 새 지도(생성 05:03:11 · 서버가 그 지나감을 반영해 칸 값 40% 남김) → 청산맵 = 서버 값. 지도 «앞» 고·저를 품은 형성 5분봉이
     그 칸을 다시 0 으로 만들면 안 된다(사용자 «업데이트할 때마다 안 채워짐»)
  ④ 대조: 새 지도 뒤 체결 2494.0(칸 전체 지남) → 청산맵 칸 0
1920 캡처(청산맵 1d · 풋프린트 1h/4h/12h). JS 오류 0 · 바깥 호스트 요청은 전부 abort.
"""
import argparse, copy, json, pathlib, sys
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

WT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("out", nargs="?", default="/tmp/render_liqmap_sweep_20261011"); ap.add_argument("--dash")
A = ap.parse_args()
DASH = pathlib.Path(A.dash) if A.dash else WT / "dashboard/live"
OUT = pathlib.Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
FIX = json.loads((WT / "test/fixtures/liqmap_eth_20261011.json").read_text("utf-8"))
ORIGIN = "http://dash.local"
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html"}
T0 = 1791694920                       # 05:02:00 UTC
K, CENTER = 996, 996 * FIX["liqmap"]["tier_profile"]["bin_width"]
SINCE = 1791694800                    # 마지막 스냅샷(04:00) + 1h = 밀도 층 끊기 시작
cur = {"map": FIX["liqmap"]}
fails, errs, log = [], [], []


def handle(route):
    u = route.request.url; pr = urlparse(u); log.append(u)
    if not u.startswith(ORIGIN): return route.abort()
    if "/api/manual-" in pr.path and route.request.method != "GET": return route.abort()
    if pr.path == "/api/liquidation-map": return route.fulfill(body=json.dumps(cur["map"]), content_type="application/json")
    if pr.path == "/api/market-history": return route.fulfill(body=json.dumps({"asset": "eth", "candles": FIX["candles"]}), content_type="application/json")
    if pr.path.startswith("/api/"): return route.fulfill(status=503, body="{}")
    name = "index.html" if pr.path.rstrip("/") == "/dashboard/live" else pr.path.split("/dashboard/live/", 1)[-1]
    f = DASH / name
    return route.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream")) if f.is_file() else route.fulfill(status=404, body="")


def bin_val(pg):
    return pg.evaluate("(k) => { const tp = window.fpLiqTp && window.fpLiqTp(); return tp ? tp.tiers.reduce((s, t) => s + (t.values[k - tp.lo] || 0), 0) : null; }", K)


def fp_swept(pg):
    return pg.evaluate("([s, p]) => liqSweepIdx(candleHistoryByAsset.eth, s, p) >= 0", [SINCE, CENTER])


def trade(pg, wss, px, i):
    t = pg.evaluate("Date.now()")
    c = FIX["candles"][-1]; c["low"], c["high"], c["close"] = min(c["low"], px), max(c["high"], px), px   # 서버 형성 봉도 같이(다시 받아도 같은 고·저)
    for w in wss: w.send(json.dumps({"e": "trade", "p": f"{px:.2f}", "q": "0.1", "T": t, "m": False, "t": 100 + i}))
    pg.wait_for_timeout(1600)        # 청산맵 1초 주기 + 그리기


def check(tag, val, swept, want_zero, want_swept):
    print(f"[{tag}] 청산맵 칸 {K} = {val} · 풋프린트 밀도 층 끊김 = {swept}", flush=True)
    if val is None: fails.append(f"[{tag}] 청산맵 값을 못 읽음")
    elif (val == 0) != want_zero: fails.append(f"[{tag}] 청산맵 칸 {'0 이어야' if want_zero else '> 0 이어야'} 하는데 {val}")
    if swept != want_swept: fails.append(f"[{tag}] 풋프린트 밀도 층 끊김 {swept} ≠ 기대 {want_swept}")


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1920, "height": 1080})
    pg = ctx.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    wss = []
    pg.route(lambda u: True, handle); pg.route_web_socket(lambda u: True, lambda ws: wss.append(ws))
    pg.clock.install(time=T0)
    pg.add_init_script("try { localStorage.clear(); } catch (e) {}")
    pg.goto(f"{ORIGIN}/dashboard/live/", wait_until="load"); pg.wait_for_timeout(2500)
    pg.evaluate("() => { const f = document.getElementById('ofab'); if (f) f.style.visibility = 'hidden'; }")
    for bars in (12, 48, 144):
        pg.click(f"#chartWindowTabs [data-bars='{bars}']"); pg.wait_for_timeout(900)
        pg.locator("#fpCard").screenshot(path=str(OUT / f"fp_{bars}.png"))
    pg.click("#fpViewTabs [data-view='liq']"); pg.wait_for_timeout(1500)
    v0 = bin_val(pg)
    print(f"[시작] 서버 칸 {K}(가운데 {CENTER:.2f}) = {v0}", flush=True)
    if not v0: fails.append(f"[시작] 사본의 칸 {K} 이 0 -- 시험 전제 깨짐")

    trade(pg, wss, 2496.6, 1)
    check("① 지도 뒤 가장자리 스침", bin_val(pg), fp_swept(pg), False, False)
    pg.locator("#fpCard").screenshot(path=str(OUT / "liq_1d_edge.png"))
    trade(pg, wss, 2495.6, 2)
    check("② 대조: 지도 뒤 가운데 지남", bin_val(pg), fp_swept(pg), True, True)
    trade(pg, wss, 2505.0, 3)

    # 새 지도 = 서버가 그 지나감을 이미 반영(칸 안 청산가 단위라 가운데 아래 몫 40% 남음)
    m2 = copy.deepcopy(FIX["liqmap"]); m2["generated_at"] = "2026-10-11T05:03:11+00:00"
    tp2 = m2["tier_profile"]
    for t in tp2["tiers"]: t["values"][K - tp2["lo"]] = round(t["values"][K - tp2["lo"]] * 0.4, 2)
    cur["map"] = m2
    pg.clock.fast_forward(72000)     # 05:03:12+ · 청산맵 폴링(30초) 두 번
    pg.wait_for_timeout(2500)
    got = pg.evaluate("latestLiquidationMap && latestLiquidationMap.generated_at")
    if got != m2["generated_at"]: fails.append(f"[③] 새 지도를 못 받음 {got}")
    v3 = bin_val(pg)
    check("③ 새 지도 -- 지도 앞 고·저 품은 형성 5분봉", v3, fp_swept(pg), False, True)
    if v3 and abs(v3 - v0 * 0.4) > 1: fails.append(f"[③] 청산맵 칸 = 서버 값이어야 {v0 * 0.4:.0f} ≠ {v3}")
    pg.locator("#fpCard").screenshot(path=str(OUT / "liq_1d_update.png"))

    trade(pg, wss, 2494.0, 4)
    check("④ 대조: 새 지도 뒤 칸 전체 지남", bin_val(pg), fp_swept(pg), True, True)
    pg.locator("#fpCard").screenshot(path=str(OUT / "liq_1d_crossed.png"))
    b.close()

out = [u for u in log if not u.startswith(ORIGIN)]
print("바깥 호스트 요청(전부 abort)", len(out), "· JS 오류", errs[:3])
if errs: fails.append(f"JS 오류 {errs[:3]}")
print("FAIL" if fails else "PASS", *fails, sep="\n")
sys.exit(1 if fails else 0)
