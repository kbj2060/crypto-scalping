"""⑤ 예측 줄 자리 정답지: fc_js.mjs 가 원본 함수로 지은 HTML(fixture/fc_html.json)을 원본 styles.css 로 헤드리스 크로미움 1920×1080(넓은 화면 #mcBody.mc-cmp · 폭 W)에 띄워
글자(바닥선 대신 상자) · 칠한 상자 · 테두리 선 · 가상 요소 선(.pr-sp::before 점선 · .pr-h::after · .xs-c::before 0선)의 자리·색을 .mc-fc 왼쪽 위 기준으로 잰다 → fixture/fc_js.json.
네트워크 없음(글꼴은 디스크에서 · 바깥 주소는 abort). 실행: unshare -rn bash -c 'ip link set lo up; python3 -I rtui/compare/fc_layout.py <worktree>'"""
import json, os, pathlib, sys
from playwright.sync_api import sync_playwright

WT = pathlib.Path(sys.argv[1]).resolve()
FX, FONTS = WT / "rtui/compare/fixture", WT / "rtui/fonts"
W = 404
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import FONT_CSS, MONO_CSS, LAUNCH

DUMP = r"""() => {
  const root = document.querySelector(".mc-fc"), o = root.getBoundingClientRect();
  const R = (r) => [r.left - o.left, r.top - o.top, r.width, r.height].map((v) => Math.round(v * 1000) / 1000);
  const col = (s) => {   // rgb(a)(…) · color(srgb r g b / a) → [r, g, b, a] (0~255 · 0~1)
    let m = /^rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)$/.exec(s);
    if (m) return [+m[1], +m[2], +m[3], m[4] == null ? 1 : +m[4]];
    m = /^color\(srgb ([\d.e-]+) ([\d.e-]+) ([\d.e-]+)(?: \/ ([\d.]+))?\)$/.exec(s);
    if (m) return [m[1] * 255, m[2] * 255, m[3] * 255, m[4] == null ? 1 : +m[4]].map((v, i) => (i < 3 ? Math.round(v * 100) / 100 : v));
    throw s;
  };
  const texts = [], boxes = [], lines = [];
  const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n; (n = tw.nextNode());) {
    if (!n.textContent.trim()) continue;
    const e = n.parentElement, cs = getComputedStyle(e), rg = document.createRange();
    rg.selectNodeContents(n);
    const rs = [...rg.getClientRects()].filter((r) => r.width > 0);
    if (!rs.length) continue;
    const r = rs.reduce((a, b) => ({ left: Math.min(a.left, b.left), top: Math.min(a.top, b.top), right: Math.max(a.right, b.right), bottom: Math.max(a.bottom, b.bottom) }));
    texts.push({ s: n.textContent, r: R({ left: r.left, top: r.top, width: r.right - r.left, height: r.bottom - r.top }), size: parseFloat(cs.fontSize), wt: +cs.fontWeight,
      mono: cs.fontFamily.startsWith('"JetBrains'), c: col(cs.color), cls: e.className });
  }
  for (const e of root.querySelectorAll("*")) {
    const cs = getComputedStyle(e), r = e.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const bg = col(cs.backgroundColor);
    // 넘치면 잘림(.pr-sp overflow hidden) -- 보이는 몫만
    let v = r;
    for (let p = e.parentElement; p && p !== root; p = p.parentElement) if (getComputedStyle(p).overflow === "hidden") { const q = p.getBoundingClientRect(); v = { left: Math.max(v.left, q.left), top: v.top, width: Math.min(v.left + v.width, q.right) - Math.max(v.left, q.left), height: v.height }; }
    if (bg[3] > 0 && v.width > 0) boxes.push({ r: R(v), c: bg, rad: parseFloat(cs.borderTopLeftRadius), cls: e.className, f: cs.filter });
    const bw = parseFloat(cs.borderTopWidth);
    if (bw > 0 && cs.borderTopStyle !== "none") lines.push({ r: R(r), c: col(cs.borderTopColor), style: cs.borderTopStyle, all: parseFloat(cs.borderLeftWidth) > 0, rad: cs.borderTopLeftRadius, cls: e.className });
  }
  for (const e of root.querySelectorAll(".pr-sp")) { const r = e.getBoundingClientRect(); lines.push({ r: R({ left: r.left, top: r.bottom - 9, width: r.width, height: 1 }), c: col(getComputedStyle(e, "::before").borderTopColor), style: "dashed", all: false, cls: "pr-sp::before" }); }
  for (const e of root.querySelectorAll(".xs-c")) { const r = e.getBoundingClientRect(); lines.push({ r: R({ left: r.left, top: r.top + 26, width: r.width, height: 1 }), c: col(getComputedStyle(e, "::before").borderTopColor), style: "solid", all: false, cls: "xs-c::before" }); }
  for (const e of root.querySelectorAll(".pr-h")) {
    const r = e.getBoundingClientRect(), rg = document.createRange(); rg.selectNodeContents(e.firstChild);
    const x = rg.getBoundingClientRect().right + 8;
    boxes.push({ r: R({ left: x, top: r.top + r.height / 2 - 0.5, width: r.right - x, height: 1 }), c: col(getComputedStyle(e, "::after").backgroundColor), rad: 0, cls: "pr-h::after" });
  }
  return { h: o.height, texts, boxes, lines };
}"""

# 변이 시험: FC_HTML(fc_js.mjs OUT) · STYLES_CSS(styles.css 사본) · FC_OUT(쓸 곳)
html = json.loads(pathlib.Path(os.environ.get("FC_HTML", FX / "fc_html.json")).read_text())
css = pathlib.Path(os.environ.get("STYLES_CSS", WT / "dashboard/live/styles.css")).read_text()
out = []
with sync_playwright() as p:
    b = p.chromium.launch(args=LAUNCH)
    try:
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        def route(r):
            n = r.request.url.rsplit("/", 1)[-1]
            if "fx.invalid/font/" in r.request.url: return r.fulfill(body=(FONTS / n).read_bytes(), content_type="font/ttf")
            r.abort()
        pg.route("**/*", route)
        for h in html:
            pg.set_content(f'<!doctype html><html class="fit1"><head><style>{FONT_CSS}{MONO_CSS}</style><style>{css}</style></head><body>'
                           f'<div id="mcBody" class="mc-body mc-cmp" style="position:static;width:{W}px;padding:0;left:0;top:0">{h}</div></body></html>')
            pg.evaluate("document.fonts.ready")
            pg.wait_for_function("document.fonts.check('700 15px \"Pretendard Variable\"') && document.fonts.check('600 11px \"JetBrains Mono\"')")
            pg.evaluate("() => Promise.all([...document.fonts].map((f) => f.load()))")
            out.append(pg.evaluate(DUMP))
    finally:
        b.close()
pathlib.Path(os.environ.get("FC_OUT", FX / "fc_js.json")).write_text(json.dumps({"w": W, "cases": out}, ensure_ascii=False) + "\n")
for k, o in enumerate(out): print(k, "h", o["h"], "글자", len(o["texts"]), "상자", len(o["boxes"]), "선", len(o["lines"]))
