"""혼자 뜨는 시험 페이지 확인: rtui/index.html?asset=eth 를 1600×900 창에 꽉 채워 띄우고(합성 자료 · 1주일 늘 받음 · 가짜 @trade 몇 초) 5m → 1h 를 눌러 캡처.
python -I rtui/compare/page.py <worktree>  →  rtui/compare/page_5m.png · page_1h.png · 표준출력 = 받은 주소 목록(바깥 접속 없음 확인)"""
import pathlib, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import INIT, LAUNCH, TZ, make_route

WT = pathlib.Path(sys.argv[1]).resolve()
FX, HERE = WT / "rtui/compare/fixture", WT / "rtui/compare"
route = make_route(WT, {"/api/footprint": FX / "footprint.json", "/api/market-history": FX / "market-history.json", "/api/footprint-week": FX / "footprint-week.json"})
urls = []
with sync_playwright() as p:
    br = p.chromium.launch(args=LAUNCH)
    try:
        ctx = br.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=1, timezone_id=TZ)
        ctx.route("**/*", route); ctx.add_init_script(INIT)
        pg = ctx.new_page()
        pg.on("request", lambda r: urls.append(r.url))
        pg.on("pageerror", lambda e: print("pageerror", str(e)[:300]))
        pg.goto("http://fx.invalid/rtui/index.html?asset=eth")
        pg.wait_for_function("window.RESULT", timeout=30_000, polling=100)
        pg.evaluate("""() => new Promise((res) => { let n = 0, px = 1745; const iv = setInterval(() => { const T = Date.now(); px += (Math.random() - .5) * .4;
            window.__wsAll[window.__wsAll.length - 1].onmessage({ data: JSON.stringify({ e: 'trade', E: T, T, t: 9e6 + n, p: px.toFixed(2), q: '0.8', m: n % 2 === 0 }) });
            if (++n > 300) { clearInterval(iv); res(); } }, 5); })""")
        pg.wait_for_timeout(800); pg.screenshot(path=str(HERE / "page_5m.png"))
        pg.click("#tf button[data-k='12']"); pg.wait_for_timeout(1200); pg.screenshot(path=str(HERE / "page_1h.png"))
        print("state", pg.evaluate("() => window.rtui.state()"))
        print("ws", pg.evaluate("() => window.__wsAll.map((w) => w.url)"))
    finally:
        br.close()
print("요청", sorted({u.split("?")[0] for u in urls}))
