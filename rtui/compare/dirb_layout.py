"""왼쪽 위 «지표 방향» 판 정답지(2/2): dirb_js.mjs 가 자른 원본 코드(fixture/dirb_src.js)를 헤드리스 크로미움 1920×1080 + 원본 styles.css 위 진짜 <svg id="candleSvgSnapshot"> 에서
입력(fixture/dirb_in.json)마다 renderDirBoard() 로 돌려 #dirBoard 안 svg 요소 전부(태그 · 속성 · 글 · 계산된 fill/stroke 색)와 measureTextW 가 잰 글 폭(글 · font · 폭)을 적는다 → fixture/dirb_js.json.
네트워크 없음(글꼴은 디스크에서 · 바깥 주소는 abort). 실행: unshare -rn bash -c 'ip link set lo up; python3 -I rtui/compare/dirb_layout.py <worktree>' · 변이 시험: DIRB_SRC · STYLES_CSS · DIRB_OUT"""
import json, os, pathlib, sys
from playwright.sync_api import sync_playwright

WT = pathlib.Path(sys.argv[1]).resolve()
FX, FONTS = WT / "rtui/compare/fixture", WT / "rtui/fonts"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import FONT_CSS, MONO_CSS, LAUNCH

# 원본 코드가 읽는 전역(가짜) · measureTextW 는 원본 그대로 재고 적기만 한다
PRE = r"""
let chartGeo = null, fpBoard = null, dirFold = false, dirFoldFocus = false, latestSituation = null, activeSnapshotAsset = "eth", measureCtx2d = null, NOW = 0;
const el = (id) => document.getElementById(id), fpAltOn = () => false, isMobileChartMode = () => false, showTooltip = () => {}, hideTooltip = () => {}, renderSnapshotChart = () => {}, helpAttrs = () => "";
Date.now = () => NOW;
"""
POST = r"""
const LOG = [], measure0 = measureTextW;
measureTextW = (t, f) => { const w = measure0(t, f); LOG.push([t, f, w]); return w; };
window.RUN = (s) => {
  chartGeo = s.G; fpBoard = { rows: s.rows, cond: s.cond, top: s.top }; dirFold = s.fold; latestSituation = s.S; activeSnapshotAsset = s.asset; NOW = s.now * 1000; LOG.length = 0;
  renderDirBoard();
  const col = (v) => { if (!v || v === "none") return null; const m = /^rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)$/.exec(v); if (!m) throw v; return [+m[1], +m[2], +m[3], m[4] == null ? 1 : +m[4]]; };
  const b = document.querySelector("#dirBoard"), els = [];
  if (b) for (const e of b.querySelectorAll("*")) {
    if (e.tagName === "g") continue;
    const at = {}; for (const x of e.attributes) at[x.name] = x.value;
    const cs = getComputedStyle(e);
    els.push({ tag: e.tagName, at, s: e.tagName === "text" ? e.textContent : null, fill: col(cs.fill), stroke: col(cs.stroke) });
  }
  return { els, measured: LOG.slice() };
};
"""

src = pathlib.Path(os.environ.get("DIRB_SRC", FX / "dirb_src.js")).read_text()
inp = json.loads((FX / "dirb_in.json").read_text())
css = pathlib.Path(os.environ.get("STYLES_CSS", WT / "dashboard/live/styles.css")).read_text()
with sync_playwright() as p:
    b = p.chromium.launch(args=LAUNCH)
    try:
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        def route(r):
            n = r.request.url.rsplit("/", 1)[-1]
            if "fx.invalid/font/" in r.request.url: return r.fulfill(body=(FONTS / n).read_bytes(), content_type="font/ttf")
            r.abort()
        pg.route("**/*", route)
        pg.set_content(f'<!doctype html><html class="fit1"><head><style>{FONT_CSS}{MONO_CSS}</style><style>{css}</style></head><body>'
                       '<svg id="candleSvgSnapshot" viewBox="0 0 1400 880" width="1400" height="880"></svg></body></html>')
        pg.evaluate("document.fonts.ready")
        pg.evaluate("""() => Promise.all(['400 12px "Pretendard Variable"', '700 12px "Pretendard Variable"'].map((f) => document.fonts.load(f)))""")
        pg.evaluate("() => Promise.all([...document.fonts].map((f) => f.load()))")
        pg.add_script_tag(content=PRE + src + POST)
        out = [pg.evaluate("(s) => RUN(s)", s) for s in inp]
    finally:
        b.close()
pathlib.Path(os.environ.get("DIRB_OUT", FX / "dirb_js.json")).write_text(json.dumps(out, ensure_ascii=False) + "\n")
for k, o in enumerate(out): print(k, "요소", len(o["els"]), "잰 글", len(o["measured"]))
