"""나란히 비교 캡처: 지금 화면(dashboard/live, 자바스크립트) 과 rtui(웹어셈블리) 를 같은 헤드리스 크로미움 · 1920×1080 · 같은 합성 자료로.
네트워크 없음: 모든 요청을 가로채 디스크에서 준다(rtui_proto/bench_js.py 방식). 계좌·주문·다른 /api/* 는 전부 404 -- 화면엔 시세·풋프린트만.
python -I rtui/compare/capture.py <worktree> <js|rtui|compare> [bars=36]   (먼저 make_fixture.py · rtui 웹어셈블리 빌드 → rtui/dist)
단계마다 브라우저 하나 · 페이지 하나 · 끝나면(예외여도) 닫는다. js → rtui → compare 순서(rtui 는 js 가 잰 상자 자리를 읽는다).
출력: rtui/compare/{js,rtui,side,diff}_<bars>.png · coords_<bars>.json(좌표 대조) · 표준출력에 요약."""
import json, os, pathlib, sys
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from PIL import Image, ImageChops, ImageDraw

WT = pathlib.Path(sys.argv[1]).resolve()
STAGE = sys.argv[2]
BARS = int(sys.argv[3]) if len(sys.argv) > 3 else 36
HERE, LIVE, FX = WT / "rtui/compare", WT / "dashboard/live", WT / "rtui/compare/fixture"
FONT = WT / "rtui/fonts/Pretendard-Bold.ttf"
T_FX = 1795156790   # 하네스 시계(합성 마지막 봉 1795156500 안)
TZ = "Asia/Seoul"
API = {"/api/footprint": FX / "footprint.json", "/api/market-history": FX / "market-history.json"}
# 지금 화면이 웹 글꼴(jsdelivr Pretendard Variable)을 받는 자리에 같은 글꼴의 Bold 한 벌만 준다(오프라인 -- 굵기 400·800 도 Bold 로 그려진다)
FONT_CSS = "@font-face{font-family:'Pretendard Variable';src:url(http://fx.invalid/font/p.ttf) format('truetype');font-weight:45 920;}"
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html", ".png": "image/png", ".wasm": "application/wasm", ".json": "application/json"}

INIT = r"""(() => {
  const OFF = %d * 1000 - Date.now(); const RD = Date;
  class D extends RD { constructor(...a) { if (a.length) super(...a); else super(RD.now() + OFF); } static now() { return RD.now() + OFF; } }
  window.Date = D;
  window.WebSocket = class { constructor() { this.readyState = 3; } send() {} close() {} addEventListener() {} };
  window.EventSource = class { constructor(u) { this.url = u; this.readyState = 1; } addEventListener() {} removeEventListener() {} close() { this.readyState = 2; } };
  // 겹침 선(VWAP·SMA·스윙·OB 등)은 이번 범위 밖 -- 사용자 스위치(카드 «지표»)로 끈다. POC 선만 켠다.
  try { localStorage.setItem('mcLine', JSON.stringify({ poc: true, vwap: false, veto: false, sma50: false, sw1h: false, sw30m: false, sw30mOn: 1, ob: false, fvg: false, orb: false, cci: false, rsi2: false, rdiv: false, srl: false, bb: false, wvwap: false })); } catch (e) {}
  // 캔버스 그리기 기록(좌표 대조용) -- window.__rec 이 켜져 있을 때만
  window.__rec = null; const P = CanvasRenderingContext2D.prototype;
  const tag = (ctx) => ctx.canvas.classList.contains('fp-cv') ? (ctx.canvas === ctx.canvas.parentNode.querySelector('.fp-cv') ? 'bg' : 'fg') : null;
  for (const k of ['fillRect', 'fillText', 'strokeRect', 'stroke', 'fill']) { const o = P[k]; P[k] = function (...a) {
    const L = window.__rec && tag(this);
    if (L) { const r = { k, L, a: a.map((v) => typeof v === 'number' ? Math.round(v * 1000) / 1000 : (typeof v === 'string' ? v : null)), s: this.fillStyle, st: this.strokeStyle, al: this.globalAlpha, f: this.font, lw: this.lineWidth, ta: this.textAlign }; window.__rec.push(r); }
    return o.apply(this, a); }; }
})();""" % T_FX


def route(r):
    u = urlparse(r.request.url)
    if r.request.method != "GET":
        return r.abort()
    if u.hostname == "cdn.jsdelivr.net" and "pretendard" in u.path:
        return r.fulfill(body=FONT_CSS, content_type="text/css", headers={"Access-Control-Allow-Origin": "*"})
    if u.hostname == "fonts.googleapis.com":
        return r.fulfill(body="", content_type="text/css")
    if u.hostname != "fx.invalid":
        return r.abort()
    if u.path == "/font/p.ttf":
        return r.fulfill(body=FONT.read_bytes(), content_type="font/ttf")
    if u.path.startswith("/dashboard/live/"):
        f = LIVE / (u.path[len("/dashboard/live/"):] or "index.html")
        if f.is_file() and f.name != "sw.js":
            return r.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream"))
        return r.fulfill(status=404, body="")
    if u.path.startswith("/rtui/"):
        f = WT / "rtui" / u.path[len("/rtui/"):]
        if f.is_file():
            return r.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream"))
        return r.fulfill(status=404, body="")
    if u.path in API:
        return r.fulfill(body=API[u.path].read_bytes(), content_type="application/json")
    return r.fulfill(status=404, body="{}", content_type="application/json")


# 지금 화면: 창을 BARS 로 맞추고 한 번 다시 그리며 캔버스 명령·svg 글자·기하를 모은다
JS_DUMP = r"""async (bars) => {
  chartWindowBars = bars; chartPanRight = 0; chartPanY = 0;
  await new Promise((r) => requestAnimationFrame(r));
  // 알약 폭은 «지금까지 본 최대»(app.js 4418 pillWMax)라 웹 글꼴이 늦게 온 첫 렌더(대체 글꼴) 폭이 남는다 -- 글꼴이 이미 있는 세션처럼 비운다
  Object.keys(pillWMax).forEach((k) => delete pillWMax[k]);
  window.__rec = [];
  renderSnapshotChart();
  const rec = window.__rec; window.__rec = null;
  const svg = document.getElementById('candleSvgSnapshot'), b = svg.getBoundingClientRect();
  const cvs = [...svg.parentNode.querySelectorAll('.fp-cv')].map((c) => c.getBoundingClientRect().toJSON());
  const texts = [...svg.querySelectorAll('text')].filter((t) => getComputedStyle(t).display !== 'none' && t.getBoundingClientRect().width > 0).map((t) => ({
    s: t.textContent, x: +t.getAttribute('x'), y: +t.getAttribute('y'), fs: t.getAttribute('font-size'), fw: t.getAttribute('font-weight'), anchor: t.getAttribute('text-anchor'),
    fill: getComputedStyle(t).fill, bb: t.getBoundingClientRect().toJSON() }));
  const pill = svg.querySelector('rect[data-pill]'), ptx = [...svg.querySelectorAll('text')].find((t) => t.getAttribute('font-size') === '14');
  return { geo: { ...chartGeo, bars: undefined, box: b.toJSON(), cvs, cp: latestLivePriceByAsset[activeSnapshotAsset] || null, fw: getComputedStyle(svg).fontWeight, fam: getComputedStyle(svg).fontFamily, tzOff: new Date().getTimezoneOffset() },
           rec, texts, pill: pill && { x: +pill.getAttribute('x'), y: +pill.getAttribute('y'), w: +pill.getAttribute('width'), h: +pill.getAttribute('height'), tw: ptx && ptx.getComputedTextLength() },
           deltaStats: renderCandleSvg._deltaStats, candles: chartGeo.bars.map((c) => [c.time, c.open, c.high, c.low, c.close]) };
}"""


def quant(im):   # 500KB 아래로: 256색 팔레트 PNG(차트는 색 수가 적다)
    return im.convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


LAUNCH = ["--host-resolver-rules=MAP * ~NOTFOUND", "--proxy-server=127.0.0.1:9", "--disable-background-networking"]
full_js, full_rt = HERE / f"_js_full_{BARS}.png", HERE / f"_rtui_full_{BARS}.png"
cj, cr = HERE / f"_coords_js_{BARS}.json", HERE / f"_coords_rtui_{BARS}.json"

if STAGE in ("js", "rtui"):
    with sync_playwright() as p:
        br = p.chromium.launch(args=LAUNCH)
        try:
            if STAGE == "js":
                ctx = br.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, service_workers="block", timezone_id=TZ, locale="ko-KR")
                ctx.route("**/*", route)
                ctx.add_init_script(INIT)
                pg = ctx.new_page()
                errs = []
                pg.on("console", lambda m: m.type == "error" and errs.append(m.text[:160]))
                pg.goto("http://fx.invalid/dashboard/live/", wait_until="domcontentloaded")
                pg.wait_for_timeout(7000)
                js = pg.evaluate(JS_DUMP, BARS)
                pg.wait_for_timeout(300)
                pg.screenshot(path=str(full_js))
                # 바탕판: 차트 층(svg·캔버스 두 장)만 숨긴 같은 화면 -- rtui 를 그 위에 그려 카드 면(세로 그라디언트)까지 같게 비교
                pg.evaluate("() => { const s = document.getElementById('candleSvgSnapshot'); s.style.visibility = 'hidden'; s.parentNode.querySelectorAll('.fp-cv').forEach((c) => (c.style.visibility = 'hidden')); }")
                pg.wait_for_timeout(200)
                pg.screenshot(path=str(HERE / f"_plate_{BARS}.png"))
                g = js["geo"]
                print("js geo", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items() if k not in ("cvs",)})
                print("js console errors", len(errs), errs[:3])
                json.dump(js, open(cj, "w"), ensure_ascii=False)
            else:
                g = json.load(open(cj))["geo"]
                box = g["box"]
                # 바탕 = 지금 화면 svg 상자 안 빈 자리(가격 플롯 위 여백 왼쪽 끝)의 실제 픽셀(카드 그라디언트 위 -- 한 색으로 근사)
                bgpx = Image.open(full_js).convert("RGB").getpixel((int(box["x"] + 4), int(box["y"] + g["mt"] - 3)))
                q = f"x={box['x']}&y={box['y']}&w={g['wAll']}&h={g['hAll']}&bars={BARS}&tz={g['tzOff']}&bg=none&plate=/rtui/compare/_plate_{BARS}.png"
                print("bg sample", bgpx)
                ctx = br.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, timezone_id=TZ)
                ctx.route("**/*", route)
                pr = ctx.new_page()
                pr.on("console", lambda m: print("rtui console", m.type, m.text[:200]))
                pr.on("pageerror", lambda e: print("rtui pageerror", str(e)[:300]))
                pr.on("requestfailed", lambda r: print("rtui reqfail", r.url))
                pr.goto(f"http://fx.invalid/rtui/index.html?mode=shot&{q}")
                pr.wait_for_function("window.RESULT", timeout=20_000, polling=100)
                rt = pr.evaluate("window.RESULT")
                pr.wait_for_timeout(300)
                pr.screenshot(path=str(full_rt))
                json.dump(rt, open(cr, "w"), ensure_ascii=False)
        finally:
            br.close()
    sys.exit(0)

js, rt = json.load(open(cj)), json.load(open(cr))
g, box = js["geo"], js["geo"]["box"]
# 자르기: svg 왼쪽 열(가격 플롯 + 프로파일 자리 + 현재가 알약) · 플롯 위 6px ~ 플롯 바닥
x0, y0 = int(box["x"]), int(box["y"] + g["mt"] - 6)
x1, y1 = int(box["x"] + rt["geo"]["w"]), int(box["y"] + g["pb"] + 1)
A = Image.open(full_js).convert("RGB").crop((x0, y0, x1, y1))
B = Image.open(full_rt).convert("RGB").crop((x0, y0, x1, y1))
W, H = A.size
side = Image.new("RGB", (W * 2 + 8, H), (255, 255, 255))
side.paste(A, (0, 0)); side.paste(B, (W + 8, 0))
d = ImageChops.difference(A, B).convert("L")
# 차이 칠하기: 회색 바탕(지금 화면) 위에 차이 크기만큼 빨강 · 범위 밖 구역(레인·프로파일 기둥)은 파란 테두리
base = A.convert("L").point(lambda v: v // 3).convert("RGB")
red = Image.new("RGB", A.size, (255, 40, 40))
diff = Image.composite(red, base, d.point(lambda v: 255 if v > 24 else (v * 8)))
dr = ImageDraw.Draw(diff)
ml, cw, mt, pb = g["ml"], g["cw"], g["mt"], g["pb"]
lane_top = rt["geo"]["lane_top"]
X = lambda v: v - (x0 - box["x"]); Y = lambda v: v - (y0 - box["y"])
dr.rectangle([X(ml), Y(lane_top), X(ml + cw), Y(pb)], outline=(80, 140, 255))
dr.rectangle([X(ml + cw), Y(mt), X(rt["geo"]["w"] - rt["geo"]["mr"]), Y(pb)], outline=(80, 140, 255))
for nm, im in (("js", A), ("rtui", B), ("side", side), ("diff", diff)):
    quant(im).save(HERE / f"{nm}_{BARS}.png", optimize=True)
# 범위 안(가격 플롯 중 레인 위) 차이 통계
inr = d.crop((int(X(ml)), int(Y(mt)), int(X(ml + cw)), int(Y(lane_top))))
hist = inr.histogram(); n = sum(hist)
big = sum(hist[25:])
print(f"in-scope diff: px {n} · >24 {big} ({100 * big / n:.2f}%) · mean {sum(i * c for i, c in enumerate(hist)) / n:.2f}")
print("sizes KB", {f.name: f.stat().st_size // 1024 for f in HERE.glob(f"*_{BARS}.png")})
