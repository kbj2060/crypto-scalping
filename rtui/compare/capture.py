"""나란히 비교 캡처: 지금 화면(dashboard/live, 자바스크립트) 과 rtui(웹어셈블리) 를 같은 헤드리스 크로미움 · 1920×1080 · 같은 합성 자료로.
네트워크 없음: 모든 요청을 가로채 디스크에서 준다(rtui_proto/bench_js.py 방식). 계좌·주문·다른 /api/* 는 전부 404 -- 화면엔 시세·풋프린트만.
python -I rtui/compare/capture.py <worktree> <js|rtui|compare|hover|jsbench> [bars=36] [nohint]   (먼저 make_fixture.py · rtui 웹어셈블리 빌드 → rtui/dist)
단계마다 브라우저 하나 · 페이지 하나 · 끝나면(예외여도) 닫는다. js → rtui → compare → hover 순서(rtui 는 js 가 잰 상자 자리를 읽는다).
출력: rtui/compare/{js,rtui,side,diff}_<bars>.png(카드 영역: 차트 + 레인 + 프로파일) · side_hover_<bars>_<n>.png · 기록 JSON(_*) · 표준출력에 요약."""
import json, pathlib, sys
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from PIL import Image, ImageChops, ImageDraw

WT = pathlib.Path(sys.argv[1]).resolve()
STAGE = sys.argv[2]
BARS = int(sys.argv[3]) if len(sys.argv) > 3 else 36
# nohint = 크로미움 --font-render-hinting=none(글자 진행 폭을 정수로 안 붙인다 -- 기본 헤드리스는 붙인다: probe_tnum.py)
NOHINT = "nohint" in sys.argv[4:]
HERE, LIVE, FX = WT / "rtui/compare", WT / "dashboard/live", WT / "rtui/compare/fixture"
FONT = WT / "rtui/fonts/Pretendard-Bold.ttf"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # -I 는 스크립트 폴더를 안 넣는다
from harness import T_FX, TZ, FONT_CSS, CT, INIT, LAUNCH as _L, make_route
API = {"/api/footprint": FX / "footprint.json", "/api/market-history": FX / "market-history.json"}
route = make_route(WT, API, FONT)

# 지금 화면: 창을 BARS 로 맞추고 한 번 다시 그리며 캔버스 명령·svg 요소·기하를 모은다
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
  const vis = (e) => { const cs = getComputedStyle(e); return cs.display !== 'none' && cs.visibility !== 'hidden' && e.closest('svg') === svg && !e.closest('defs'); };
  const texts = [...svg.querySelectorAll('text')].filter((t) => vis(t) && t.getBoundingClientRect().width > 0).map((t) => ({
    s: t.textContent, x: +t.getAttribute('x'), y: +t.getAttribute('y'), fs: t.getAttribute('font-size'), fw: t.getAttribute('font-weight'), anchor: t.getAttribute('text-anchor'),
    fill: getComputedStyle(t).fill, bb: t.getBoundingClientRect().toJSON(), tl: t.getComputedTextLength() }));
  // svg 요소 전부(속성 그대로) -- 프로파일 기둥·레인 글자 대조용
  const els = [...svg.querySelectorAll('rect,line,circle,text,path,polyline')].filter(vis).map((e) => ({
    t: e.tagName, a: Object.fromEntries([...e.attributes].map((x) => [x.name, x.value])), s: e.tagName === 'text' ? e.textContent : undefined,
    tl: e.tagName === 'text' ? e.getComputedTextLength() : undefined, cls: e.closest('g') ? e.closest('g').getAttribute('class') : null }));
  const pill = svg.querySelector('rect[data-pill]'), ptx = [...svg.querySelectorAll('text')].find((t) => t.getAttribute('font-size') === '14');
  return { geo: { ...chartGeo, bars: undefined, box: b.toJSON(), cvs, cp: latestLivePriceByAsset[activeSnapshotAsset] || null, fw: getComputedStyle(svg).fontWeight, fam: getComputedStyle(svg).fontFamily, tzOff: new Date().getTimezoneOffset() },
           rec, texts, els, pill: pill && { x: +pill.getAttribute('x'), y: +pill.getAttribute('y'), w: +pill.getAttribute('width'), h: +pill.getAttribute('height'), tw: ptx && ptx.getComputedTextLength() },
           deltaStats: renderCandleSvg._deltaStats, candles: chartGeo.bars.map((c) => [c.time, c.open, c.high, c.low, c.close]) };
}"""

# 지금 화면 호버 상태: 십자선·가격 배지(svg .hover-layer) + 툴팁(#chartTooltip) 상자·글자
JS_HOVER = r"""() => {
  const svg = document.getElementById('candleSvgSnapshot'), hl = svg.querySelector('.hover-layer'), t = document.getElementById('chartTooltip');
  const shown = (e) => e.style.display !== 'none' && getComputedStyle(e).display !== 'none';
  const items = [...hl.children].filter(shown).map((e) => ({ t: e.tagName, a: Object.fromEntries([...e.attributes].map((x) => [x.name, x.value])), s: e.tagName === 'text' ? e.textContent : undefined,
    tl: e.tagName === 'text' ? e.getComputedTextLength() : undefined }));
  const vis = t.classList.contains('visible');
  // 툴팁 줄: 글자 노드마다 화면 상자(줄 위치 대조)
  const lines = [];
  if (vis) { const w = document.createTreeWalker(t, NodeFilter.SHOW_TEXT); let n; while ((n = w.nextNode())) { if (!n.nodeValue.trim()) continue;
    const r = document.createRange(); r.selectNodeContents(n); [...r.getClientRects()].forEach((q) => lines.push({ s: n.nodeValue.trim().slice(0, 40), x: q.x, y: q.y, w: q.width, h: q.height, col: getComputedStyle(n.parentNode).color })); } }
  return { items, tip: vis ? { box: t.getBoundingClientRect().toJSON(), text: t.innerText, lines } : null };
}"""


def quant(im):   # 500KB 아래로: 256색 팔레트 PNG(차트는 색 수가 적다)
    return im.convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def card_box(g, rg):
    """자르기(화면 px): svg 왼쪽 열 전체 폭(가격 플롯 + 프로파일 기둥 + 알약) · 기둥 머리 글자 위 ~ 플롯 바닥(레인은 플롯 안 바닥)"""
    box = g["box"]
    return int(box["x"]), int(box["y"] + g["mt"] - 24), int(box["x"] + rg["w"]), int(box["y"] + g["pb"] + 2)


def prof_x(rg):
    """프로파일 칸 x(app.js 8584-8588 · 9320 -- 넓은 2단 · 호가 칸 0) -- 호버 점과 범위 밖 상자(청산 기둥)용"""
    dt_w, trade_w = round(64 * 0.75), round(88 * 0.75)
    c_tpo = rg["w"] - rg["mr"] + 6 - 6 - dt_w
    c_del = c_tpo - 6 - dt_w
    d_x0 = c_del - 6 - trade_w
    return {"trade_x0": rg["ml"] + rg["cw"], "d_x0": d_x0, "d_mid": d_x0 + (c_del + dt_w - d_x0) / 2, "t_x0": c_tpo, "t_x1": c_tpo + dt_w}


def hover_points(js, rg):
    """정해진 호버 자리(svg 좌표): ① 봉 칸 위 ② 레인 위 ③ 체결·델타 칸 ④ TPO 칸 아래쪽 ⑤ 플롯 위 여백(가격 배지 없음 · 봉 툴팁)"""
    g, c = js["geo"], js["candles"]
    slot = g["cw"] / len(c)
    y_at = lambda v: g["mt"] + (g["ymax"] - v) * (g["pb"] - g["mt"]) / (g["ymax"] - g["ymin"])
    i1, i2 = len(c) * 2 // 7, len(c) * 4 // 7
    px = prof_x(rg)
    return [(g["ml"] + slot * (i1 + 0.4), y_at((c[i1][2] + c[i1][3]) / 2) + 0.3),
            (g["ml"] + slot * (i2 + 0.5), rg["lane_top"] + 60),
            (px["d_mid"] + 6, y_at(rg["cp"]) - 9),
            (px["t_x0"] + 8, g["pb"] - 70),
            (g["ml"] + slot * (len(c) - 1.5), g["mt"] + 20)]


LAUNCH = _L + (["--font-render-hinting=none"] if NOHINT else [])
full_js, full_rt = HERE / f"_js_full_{BARS}.png", HERE / f"_rtui_full_{BARS}.png"
cj, cr, ch_ = HERE / f"_coords_js_{BARS}.json", HERE / f"_coords_rtui_{BARS}.json", HERE / f"_hover_{BARS}.json"


def open_js(br):
    ctx = br.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, service_workers="block", timezone_id=TZ, locale="ko-KR")
    ctx.route("**/*", route)
    ctx.add_init_script(INIT)
    pg = ctx.new_page()
    errs = []
    pg.on("console", lambda m: m.type == "error" and errs.append(m.text[:160]))
    pg.goto("http://fx.invalid/dashboard/live/", wait_until="domcontentloaded")
    pg.wait_for_timeout(7000)
    return pg, errs


def open_rt(br, g):
    box = g["box"]
    q = f"x={box['x']}&y={box['y']}&w={g['wAll']}&h={g['hAll']}&bars={BARS}&tz={g['tzOff']}&bg=none&plate=/rtui/compare/_plate_{BARS}.png"
    ctx = br.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, timezone_id=TZ)
    ctx.route("**/*", route)
    ctx.add_init_script(INIT)   # 같은 시계 · 가짜 WebSocket(rtui 도 이제 /api/* 와 @trade 를 스스로 부른다 -- 셋째 조각)
    pr = ctx.new_page()
    pr.on("console", lambda m: print("rtui console", m.type, m.text[:200]))
    pr.on("pageerror", lambda e: print("rtui pageerror", str(e)[:300]))
    pr.on("requestfailed", lambda r: print("rtui reqfail", r.url))
    pr.goto(f"http://fx.invalid/rtui/index.html?mode=shot&{q}")
    pr.wait_for_function("window.RESULT", timeout=20_000, polling=100)
    return pr


# 지금 화면 측정(rtui_proto/bench_js.py 와 같은 식 · 같은 합성 자료): 형성 봉 칸을 매 프레임 바꾸고 renderCandleSvg 한 번(M0 구간 m0.fp)
JS_BENCH = r"""async ([N, bars]) => {
  chartWindowBars = bars; chartPanRight = 0; renderSnapshotChart();
  const last = latestFootprint.bars[latestFootprint.bars.length - 1], out = [];
  for (let f = 0; f < N + 30; f++) {
    await new Promise((r) => requestAnimationFrame(r));
    last.levels.forEach((l, j) => { l[1] += 0.37 + ((f + j) % 5) * 0.11; l[2] += 0.29 + ((f * 3 + j) % 7) * 0.07; });
    m0.fp.length = 0; renderSnapshotChart();
    if (f >= 30) out.push(m0.fp[0]);
  }
  return out;
}"""

if STAGE == "jsbench":
    with sync_playwright() as p:
        br = p.chromium.launch(args=LAUNCH)
        try:
            pg, _ = open_js(br)
            ms = sorted(pg.evaluate(JS_BENCH, [300, BARS]))
        finally:
            br.close()
    print(f"JS renderCandleSvg {BARS}봉 n={len(ms)} med={ms[len(ms) // 2]:.3f} p99={ms[int(0.99 * len(ms))]:.3f} max={ms[-1]:.3f} ms")
    sys.exit(0)

if STAGE in ("js", "rtui", "hover"):
    with sync_playwright() as p:
        br = p.chromium.launch(args=LAUNCH)
        try:
            if STAGE == "js":
                pg, errs = open_js(br)
                js = pg.evaluate(JS_DUMP, BARS)
                pg.wait_for_timeout(300)
                pg.screenshot(path=str(full_js))
                # 바탕판: 차트 층(svg·캔버스 두 장)만 숨긴 같은 화면 -- rtui 를 그 위에 그려 카드 면(세로 그라디언트)·HTML 단추까지 같게 비교
                pg.evaluate("() => { const s = document.getElementById('candleSvgSnapshot'); s.style.visibility = 'hidden'; s.parentNode.querySelectorAll('.fp-cv').forEach((c) => (c.style.visibility = 'hidden')); }")
                pg.wait_for_timeout(200)
                pg.screenshot(path=str(HERE / f"_plate_{BARS}.png"))
                g = js["geo"]
                print("js geo", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items() if k not in ("cvs",)})
                print("js console errors", len(errs), errs[:3])
                json.dump(js, open(cj, "w"), ensure_ascii=False)
            elif STAGE == "rtui":
                pr = open_rt(br, json.load(open(cj))["geo"])
                rt = pr.evaluate("window.RESULT")
                pr.wait_for_timeout(300)
                pr.screenshot(path=str(full_rt))
                json.dump(rt, open(cr, "w"), ensure_ascii=False)
            else:
                # 호버: 두 쪽에 같은 화면 좌표로 마우스를 올리고 캡처(페이지 하나씩 차례로)
                js, rt = json.load(open(cj)), json.load(open(cr))
                g, rg = js["geo"], rt["geo"]
                pts = hover_points(js, rg)
                pg, _ = open_js(br)
                jb = pg.evaluate(JS_DUMP, BARS)["geo"]["box"]   # 다시 띄운 지금 화면은 상자가 1px 어긋날 수 있다 -- svg 좌표로 같은 점에
                out = {"pts": pts, "js": [], "rt": [], "jbox": jb, "rbox": g["box"]}
                for n, (x, y) in enumerate(pts):
                    pg.mouse.move(jb["x"] + x, jb["y"] + y)
                    pg.wait_for_timeout(400)
                    out["js"].append(pg.evaluate(JS_HOVER))
                    pg.screenshot(path=str(HERE / f"_hjs_{BARS}_{n}.png"))
                pg.context.close()
                pr = open_rt(br, g)
                for n, (x, y) in enumerate(pts):
                    pr.mouse.move(g["box"]["x"] + x, g["box"]["y"] + y)
                    pr.wait_for_timeout(400)
                    out["rt"].append(json.loads(pr.evaluate("window.rtui_scene()") or "{}").get("hover"))
                    pr.screenshot(path=str(HERE / f"_hrt_{BARS}_{n}.png"))
                json.dump(out, open(ch_, "w"), ensure_ascii=False)
                # 나란히: 마우스 점 둘레 + 두 쪽 툴팁 상자를 다 담는 창(svg 상자 기준으로 맞춘다 -- 두 화면의 상자 자리가 1px 다를 수 있다)
                dx, dy = jb["x"] - g["box"]["x"], jb["y"] - g["box"]["y"]
                for n, (x, y) in enumerate(pts):
                    bx = [g["box"]["x"] + x - 60, g["box"]["y"] + y - 40, g["box"]["x"] + x + 60, g["box"]["y"] + y + 40]
                    tj, tr = out["js"][n]["tip"], (out["rt"][n] or {}).get("tip")
                    if tj: b = tj["box"]; bx = [min(bx[0], b["x"] - dx), min(bx[1], b["y"] - dy), max(bx[2], b["x"] + b["width"] - dx), max(bx[3], b["y"] + b["height"] - dy)]
                    if tr: bx = [min(bx[0], tr["x"]), min(bx[1], tr["y"]), max(bx[2], tr["x"] + tr["w"]), max(bx[3], tr["y"] + tr["h"])]
                    c = [int(max(0, bx[0] - 12)), int(max(0, bx[1] - 12)), int(min(1920, bx[2] + 12)), int(min(1080, bx[3] + 12))]
                    A = Image.open(HERE / f"_hjs_{BARS}_{n}.png").convert("RGB").crop((c[0] + int(dx), c[1] + int(dy), c[2] + int(dx), c[3] + int(dy)))
                    B = Image.open(HERE / f"_hrt_{BARS}_{n}.png").convert("RGB").crop(tuple(c))
                    side = Image.new("RGB", (A.size[0] * 2 + 8, A.size[1]), (255, 255, 255))
                    side.paste(A, (0, 0)); side.paste(B, (A.size[0] + 8, 0))
                    quant(side).save(HERE / f"side_hover_{BARS}_{n}.png", optimize=True)
                print("hover", len(pts), "points saved")
        finally:
            br.close()
    sys.exit(0)

js, rt = json.load(open(cj)), json.load(open(cr))
g, box, rg = js["geo"], js["geo"]["box"], rt["geo"]
x0, y0, x1, y1 = card_box(g, rg)
A = Image.open(full_js).convert("RGB").crop((x0, y0, x1, y1))
B = Image.open(full_rt).convert("RGB").crop((x0, y0, x1, y1))
W, H = A.size
side = Image.new("RGB", (W * 2 + 8, H), (255, 255, 255))
side.paste(A, (0, 0)); side.paste(B, (W + 8, 0))
d = ImageChops.difference(A, B).convert("L")
# 차이 칠하기: 회색 바탕(지금 화면) 위에 차이 크기만큼 빨강 · 범위 밖 구역(청산 기둥 -- 청산맵 응답이 필요)은 파란 테두리
base = A.convert("L").point(lambda v: v // 3).convert("RGB")
red = Image.new("RGB", A.size, (255, 40, 40))
diff = Image.composite(red, base, d.point(lambda v: 255 if v > 24 else (v * 8)))
dr = ImageDraw.Draw(diff)
X = lambda v: v - (x0 - box["x"]); Y = lambda v: v - (y0 - box["y"])
px = prof_x(rg)
out_box = [X(px["trade_x0"]), Y(g["mt"] - 24), X(px["d_x0"] - 2), Y(g["pb"])]   # 청산 기둥(머리 «청산» 글자는 범위 안이지만 같은 칸)
dr.rectangle(out_box, outline=(80, 140, 255))
for nm, im in (("js", A), ("rtui", B), ("side", side), ("diff", diff)):
    quant(im).save(HERE / f"{nm}_{BARS}.png", optimize=True)
# 범위 안 차이 통계: 카드 영역에서 청산 기둥 칸을 뺀 곳
m = Image.new("L", A.size, 255)
ImageDraw.Draw(m).rectangle(out_box, fill=0)
hist = d.histogram(mask=m); n = sum(hist)
big = sum(hist[25:])
print(f"in-scope diff: px {n} · >24 {big} ({100 * big / n:.2f}%) · mean {sum(i * c for i, c in enumerate(hist)) / n:.2f}")
print("sizes KB", {f.name: f.stat().st_size // 1024 for f in HERE.glob(f"*_{BARS}.png")})
