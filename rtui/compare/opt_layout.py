"""Option 카드 정답지 2/2: opt_js.mjs 가 자른 원본 함수 묶음(fixture/opt_src.json)을 원본 styles.css · 진짜 #optCard 틀(index.html 과 같은 요소)로 헤드리스 크로미움 1920×1080(html.fit1 · 서울 시간)에 띄워
renderOptions 를 두 번 돌리고(첫 판은 사다리 폭을 짐작 -- 지금 화면도 같다) 잰다 → fixture/opt_js.json:
  HTML = 글자(상자) · 칠한 상자 · 테두리 선(위 · 아래) -- 카드 왼쪽 위 기준(fc_layout.py 와 같은 꼴) · ::before 이름표(«지금»·«가격대»·«거래 · 심리»)는 같은 모양 div 로 바꿔 끼워 잰다
  SVG = 그림 요소 속성 그대로(카드 안 자리 · 배율) -- rtui 그림과 1e-6 으로 맞댄다.
네트워크 없음(글꼴은 디스크 · 바깥 주소는 abort). 실행: unshare -rn bash -c 'ip link set lo up; python3 -I rtui/compare/opt_layout.py <worktree>'"""
import json, os, pathlib, sys
from playwright.sync_api import sync_playwright

WT = pathlib.Path(sys.argv[1]).resolve()
FX, FONTS = WT / "rtui/compare/fixture", WT / "rtui/fonts"
W, LAD, CK = 1838, 514, 0.9   # 카드 폭 · 사다리 높이(fitLadderFormula 1080 창) · 아래 줄 그림 배율(fitOptChartK)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import FONT_CSS, MONO_CSS, LAUNCH

CARD = ('<section id="optCard" class="panel ops-health-panel opt-card" style="width:%dpx;position:relative">'
        '<div class="ops-health-head"><div class="opt-card-head"><h3 class="card-title">Option</h3><span id="optCovHead" class="muted">Deribit · 참고, 신호 아님</span></div></div>'
        '<div id="optBody" class="opt-body"></div><div id="optRow" class="opt-row" hidden><div id="optLaneBody" class="opt-lane-body"></div>'
        '<div id="optBlkBody" class="opt-lane-body opt-blk-col"></div><div id="optFlowBody" class="opt-lane-body"></div></div></section>') % W

RUN = r"""([body, s, lad, ck]) => {
  const now = s.now;
  class D extends Date { constructor(...a) { super(...(a.length ? a : [now])); } static now() { return now; } }
  const env = { latestGex: s.gex, latestSmile: s.smile, latestLivePriceByAsset: s.price ? { [s.asset]: s.price } : {}, activeSnapshotAsset: s.asset, candleHistoryByAsset: { [s.asset]: s.candles },
    latestBinanceAccount: null, fitOptLadderH: lad, fitOptChartK: ck, helpKey: null, optLadderScope: "week", optBlkOpen: false, fpProfileSpan: null, renderPaper: () => {}, el: (id) => document.getElementById(id), Date: D };
  const px = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  new Function("px", `with (px) { ${body}\nrenderOptions(); renderOptions(); }`)(px);
  // ::before 이름표 → 같은 모양 div(레이아웃 같음 -- 글자를 잴 수 있게)
  const card = document.getElementById("optCard");
  for (const e of card.querySelectorAll(".opt-sec, #optBlkBody")) {
    const c = getComputedStyle(e, "::before").content;
    if (!c || c === "none" || c === "normal" || getComputedStyle(e, "::before").display === "none") continue;
    e.classList.add("nb");
    const d = document.createElement("div");
    d.className = "pb"; d.textContent = JSON.parse(c);
    e.insertBefore(d, e.firstChild);
  }
}"""

DUMP = r"""() => {
  const root = document.getElementById("optCard"), o = root.getBoundingClientRect();
  const R = (r) => [r.left - o.left, r.top - o.top, r.width, r.height].map((v) => Math.round(v * 1000) / 1000);
  const col = (s) => {
    let m = /^rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)$/.exec(s);
    if (m) return [+m[1], +m[2], +m[3], m[4] == null ? 1 : +m[4]];
    m = /^color\(srgb ([\d.e-]+) ([\d.e-]+) ([\d.e-]+)(?: \/ ([\d.]+))?\)$/.exec(s);
    if (m) return [m[1] * 255, m[2] * 255, m[3] * 255, m[4] == null ? 1 : +m[4]].map((v, i) => (i < 3 ? Math.round(v * 100) / 100 : v));
    throw s;
  };
  const vis = (e) => { for (let p = e; p && p !== root; p = p.parentElement) if (getComputedStyle(p).display === "none") return false; return true; };
  const texts = [], boxes = [], lines = [], svgs = [];
  const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n; (n = tw.nextNode());) {
    if (!n.textContent.trim() || n.parentElement.closest("svg") || !vis(n.parentElement)) continue;
    const e = n.parentElement, cs = getComputedStyle(e), rg = document.createRange();
    rg.selectNodeContents(n);
    const rs = [...rg.getClientRects()].filter((r) => r.width > 0);
    if (!rs.length) continue;
    // 줄마다 따로(꺾인 글) -- 한 줄이면 하나
    const r = rs.reduce((a, b) => ({ left: Math.min(a.left, b.left), top: Math.min(a.top, b.top), right: Math.max(a.right, b.right), bottom: Math.max(a.bottom, b.bottom) }));
    texts.push({ s: n.textContent, r: R({ left: r.left, top: r.top, width: r.right - r.left, height: r.bottom - r.top }), size: parseFloat(cs.fontSize), wt: +cs.fontWeight,
      mono: cs.fontFamily.startsWith('"JetBrains'), c: col(cs.color), op: +cs.opacity, cls: e.className, lines: rs.length });
  }
  for (const e of root.querySelectorAll("*")) {
    if (e.closest("svg") || !vis(e)) continue;
    const cs = getComputedStyle(e), r = e.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const bg = col(cs.backgroundColor);
    if (bg[3] > 0) boxes.push({ r: R(r), c: bg, rad: parseFloat(cs.borderTopLeftRadius), cls: e.className, f: cs.filter });
    const bw = parseFloat(cs.borderTopWidth);
    if (bw > 0 && cs.borderTopStyle !== "none") lines.push({ r: R(r), c: col(cs.borderTopColor), style: cs.borderTopStyle, all: parseFloat(cs.borderLeftWidth) > 0, rad: cs.borderTopLeftRadius, cls: e.className, op: +cs.opacity });
    else if (parseFloat(cs.borderBottomWidth) > 0 && cs.borderBottomStyle !== "none") boxes.push({ r: R({ left: r.left, top: r.bottom - 1, width: r.width, height: 1 }), c: col(cs.borderBottomColor), rad: 0, cls: e.className + "::bottom", f: "none" });
  }
  for (const g of root.querySelectorAll("svg")) {
    if (!vis(g)) continue;
    const r = g.getBoundingClientRect(), vb = g.viewBox.baseVal, els = [];
    for (const e of g.querySelectorAll("*")) {
      if (["g", "title"].includes(e.tagName) || e.closest("title")) continue;
      const a = {}; for (const at of e.attributes) a[at.name] = at.value;
      els.push({ tag: e.tagName, a, s: e.tagName === "text" ? e.textContent : null });
    }
    svgs.push({ r: R(r), vb: [vb.width, vb.height], cls: g.getAttribute("class") || "", els });
  }
  return { h: o.height, texts, boxes, lines, svgs };
}"""

body = json.loads(pathlib.Path(os.environ.get("OPT_SRC", FX / "opt_src.json")).read_text())
inputs = json.loads((FX / "opt_in.json").read_text())
css = pathlib.Path(os.environ.get("STYLES_CSS", WT / "dashboard/live/styles.css")).read_text()
out = []
with sync_playwright() as p:
    b = p.chromium.launch(args=LAUNCH)
    try:
        pg = b.new_page(viewport={"width": 1920, "height": 1080}, timezone_id="Asia/Seoul")
        def route(r):
            n = r.request.url.rsplit("/", 1)[-1]
            if "fx.invalid/font/" in r.request.url: return r.fulfill(body=(FONTS / n).read_bytes(), content_type="font/ttf")
            r.abort()
        pg.route("**/*", route)
        for s in inputs:
            pg.set_content(f'<!doctype html><html class="fit1"><head><style>{FONT_CSS}{MONO_CSS}</style><style>{css}</style>'
                           '<style>#optCard .nb::before{content:none!important} #optCard .pb{display:block;margin-bottom:6px;font:800 11px var(--font-sans);letter-spacing:.08em;color:var(--muted)}</style>'
                           f'</head><body style="margin:0">{CARD}</body></html>')
            pg.evaluate("() => Promise.all([...document.fonts].map((f) => f.load()))")
            pg.wait_for_function("document.fonts.check('700 15px \"Pretendard Variable\"') && document.fonts.check('600 11px \"JetBrains Mono\"')")
            pg.evaluate(RUN, [body, s, LAD, CK])
            pg.evaluate("document.fonts.ready")
            out.append(pg.evaluate(DUMP))
    finally:
        b.close()
pathlib.Path(os.environ.get("OPT_OUT", FX / "opt_js.json")).write_text(json.dumps({"w": W, "lad": LAD, "ck": CK, "cases": out}, ensure_ascii=False) + "\n")
for k, o in enumerate(out): print(k, "h", o["h"], "글자", len(o["texts"]), "상자", len(o["boxes"]), "선", len(o["lines"]), "svg", [(g["cls"], len(g["els"])) for g in o["svgs"]])
