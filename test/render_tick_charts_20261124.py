"""«틱» 보기(펜 메모 11-23) 렌더 검증 (2026-11-24).
  공개 대시보드에 이 작업 사본의 index.html·app.js·styles.css·tick_charts.js·fp_daily.js·liq_profile.js 를 끼우고(쓰기 요청 전부 차단),
  서버에 아직 없는 /api/tick-bars 는 같은 시간대를 덮는 합성 봉(지금가 근처 무작위 걸음)으로 흉내 낸다 -- 그림·동작 확인용, 값 검증 아님.
  (1) 틱 탭 → 다섯 칸이 교체 영역에 그려짐 (2) 지표 팝업 단추·LONG/SHORT 줄 보임 (3) 한 칸에 커서 → 다섯 칸이 같은 시각 (4) SMMA 를 켜면 틱 칸에도 선
  (5) LONG 누르면 풋프린트로 돌아감 · JS 오류 0 · 데스크톱 1920 · 휴대폰 390. 실행: python test/render_tick_charts_20261124.py <스크린숏 폴더>"""
import json, random, re, sys, time, pathlib
from playwright.sync_api import sync_playwright

W = str(pathlib.Path(__file__).resolve().parents[1] / "dashboard/live") + "/"
SP = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
LOCAL = {"app.js": "application/javascript", "styles.css": "text/css", "tick_charts.js": "application/javascript",
         "fp_daily.js": "application/javascript", "liq_profile.js": "application/javascript"}
SIZES = [100, 500, 1000, 2000, 5000]
PX = {"p": 0.0}


def fake_ticks():
    random.seed(7)
    now = int(time.time() * 1000)
    out = {}
    for n in SIZES:
        dt = n * 8                      # ms per bar(초당 ~125 체결)
        rows, p = [], PX["p"] or 2000.0
        for i in range(200):
            t0 = now - (200 - i) * dt
            o = p; c = p + random.gauss(0, 0.05 * n ** 0.5)
            rows.append([i, t0, t0 + dt - 1, round(o, 2), round(max(o, c) + abs(random.gauss(0, 0.3)), 2), round(min(o, c) - abs(random.gauss(0, 0.3)), 2), round(c, 2),
                         round(n * 0.4, 3), round(random.gauss(0, n * 0.05), 3), n])
            p = c
        out[str(n)] = {"k": 200, "bars": rows, "cur": None}
    return {"symbol": "ETHUSDT", "sizes": SIZES, "ticks": out}


def handle(route):
    req = route.request
    if req.method != "GET":
        return route.abort()
    path = re.sub(r"\?.*$", "", req.url)
    for f, ct in LOCAL.items():
        if "/dashboard/live/" in path and path.endswith("/" + f):
            return route.fulfill(body=open(W + f, encoding="utf-8").read(), content_type=ct)
    if path.rstrip("/").endswith("/dashboard/live") or path.endswith("/dashboard/live/index.html"):
        return route.fulfill(body=open(W + "index.html", encoding="utf-8").read(), content_type="text/html")
    if path.endswith("/api/tick-bars"):
        return route.fulfill(body=json.dumps(fake_ticks()), content_type="application/json")
    return route.continue_()


JS = r"""() => {
  const cells = [...document.querySelectorAll('#tickCharts .tkc-cell')].map((c) => { const cv = c.querySelector('canvas'), r = cv.getBoundingClientRect();
    return { n: +c.dataset.n, w: Math.round(r.width), h: Math.round(r.height), sub: c.querySelector('.tkc-sub').textContent, geo: !!c._geo }; });
  const box = document.getElementById('tickCharts'), sw = document.getElementById('fpLineSwitch'), lb = document.querySelector('#fpCard .fp-line-btn');
  const vis = (e) => !!e && getComputedStyle(e).visibility !== 'hidden' && getComputedStyle(e).display !== 'none' && e.getBoundingClientRect().width > 0;
  return { on: window.fpTickActive && window.fpTickActive(), boxShown: vis(box), cells, indBtn: vis(sw), lineBtn: vis(lb),
           tab: document.querySelector('#fpViewTabs .asset-tab.active')?.dataset.view };
}"""

fails = []
def check(c, msg):
    print(("PASS " if c else "FAIL ") + msg)
    if not c:
        fails.append(msg)


with sync_playwright() as p:
    b = p.chromium.launch()
    for vp, name in (({"width": 1920, "height": 1080}, "desktop"), ({"width": 390, "height": 844}, "mobile")):
        ctx = b.new_context(viewport=vp, device_scale_factor=2 if name == "mobile" else 1)
        pg = ctx.new_page(); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route("**/*", handle)
        pg.goto("https://thesan.xyz/dashboard/live/", wait_until="domcontentloaded"); pg.wait_for_timeout(8000)
        PX["p"] = pg.evaluate("Number(latestLivePriceByAsset[activeSnapshotAsset]) || 2000")
        pg.evaluate("() => { Object.assign(mcLine, { smma21: true, smma50: true }); }")
        pg.evaluate("() => document.querySelector('#fpViewTabs .asset-tab[data-view=tick]').click()")
        pg.wait_for_timeout(3000)
        r = pg.evaluate(JS); print(name, json.dumps(r, ensure_ascii=False))
        check(r["on"] and r["boxShown"] and r["tab"] == "tick", f"{name}: 틱 보기 켜짐")
        check(len(r["cells"]) == 5 and all(c["w"] > 100 and c["h"] > 80 and c["geo"] for c in r["cells"]), f"{name}: 다섯 칸 그려짐")
        if name == "desktop":
            check(r["indBtn"], f"{name}: 지표 단추 줄 보임")
            check(r["lineBtn"], f"{name}: LONG·SHORT 줄 보임")
        # 커서: 첫 칸 가운데에 올리면 다섯 칸 모두 그 시각 봉에 세로선(봉 정보 줄이 칸 위에 생김 -- 그림이라 스크린숏으로 본다)
        cv = pg.locator('#tickCharts .tkc-cell[data-n="1000"] canvas').bounding_box()
        pg.mouse.move(cv["x"] + cv["width"] * 0.6, cv["y"] + cv["height"] * 0.4); pg.wait_for_timeout(400)
        pg.screenshot(path=f"{SP}/tick_{name}.png", full_page=False)
        if name == "desktop":
            pg.evaluate("() => document.querySelector('#fpCard .fp-line-btn button')?.click()")
            pg.wait_for_timeout(1500)
            r2 = pg.evaluate(JS)
            check(r2["tab"] == "fp" and not r2["on"], f"{name}: LONG 누르면 풋프린트로 돌아감")
        check(not errs, f"{name}: JS 오류 0 {errs[:2]}")
        ctx.close()
    b.close()
print("FAILS", fails)
sys.exit(1 if fails else 0)
