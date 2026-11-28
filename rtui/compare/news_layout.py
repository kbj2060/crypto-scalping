"""News 카드 + ⑤ 다음 24시간 자리 정답지: news_js.mjs 가 원본 코드로 지은 HTML(fixture/news_html.json)을 원본 styles.css 로 헤드리스 크로미움 1920×1080(html.fit1)에 띄워 잰다 → fixture/news_js.json.
카드 = 지금 화면과 같은 구조(section#newsCard > 머리 + #newsBody · ⑤ #mcWhen = .nw-main 끝 · body.not-eth/data-coin) · 기준 = #newsBody 왼쪽 위.
잰 것은 mt_layout.py 와 같다(글 · 칠 · 테 · 고리 · 잘림 상자) + svg 자식 전부(글 = 내용·크기·굵기·정렬 · 원 · 선 · 계산된 색·불투명도). 같은 구조의 #mtBody 를 같이 재 높이 기준(1838×917 = mt_layout)을 맞춘다.
네트워크 없음(글꼴은 디스크에서 · 바깥 주소는 abort). 실행: unshare -rn bash -c 'ip link set lo up; python3 -I rtui/compare/news_layout.py <worktree>'"""
import json, os, pathlib, sys
from playwright.sync_api import sync_playwright

WT = pathlib.Path(sys.argv[1]).resolve()
FX, FONTS = WT / "rtui/compare/fixture", WT / "rtui/fonts"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import FONT_CSS, MONO_CSS, LAUNCH

DUMP = r"""(sel) => {
  const root = document.querySelector(sel), o = root.getBoundingClientRect();
  const q = (v) => Math.round(v * 1000) / 1000;
  const R = (r) => [r.left - o.left, r.top - o.top, r.width, r.height].map(q);
  const col = (s) => {   // rgb(a)(…) · color(srgb r g b / a) → [r, g, b, a] (0~255 · 0~1)
    if (s === "none" || !s) return null;
    let m = /^rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)$/.exec(s);
    if (m) return [+m[1], +m[2], +m[3], m[4] == null ? 1 : +m[4]];
    m = /^color\(srgb ([\d.e-]+) ([\d.e-]+) ([\d.e-]+)(?: \/ ([\d.]+))?\)$/.exec(s);
    if (m) return [m[1] * 255, m[2] * 255, m[3] * 255, m[4] == null ? 1 : +m[4]].map((v, i) => (i < 3 ? Math.round(v * 100) / 100 : v));
    throw s;
  };
  const op = (e) => { let a = 1; for (let p = e; p && p !== root.parentElement; p = p.parentElement) a *= +getComputedStyle(p).opacity; return a; };
  const ca = (c, a) => (c ? [c[0], c[1], c[2], q(c[3] * a)] : null);
  const clipOf = (e) => {   // overflow 가 visible 아닌 조상들의 겹침(기준 상자 밖은 안 봄)
    let c = null;
    for (let p = e.parentElement; p && p !== root; p = p.parentElement) {
      const cs = getComputedStyle(p);
      if (cs.overflowX !== "visible" || cs.overflowY !== "visible") { const r = p.getBoundingClientRect(); const v = [r.left, r.top, r.left + p.clientWidth + (cs.overflowX === "visible" ? 1e4 : 0), r.top + p.clientHeight];
        c = c ? [Math.max(c[0], v[0]), Math.max(c[1], v[1]), Math.min(c[2], v[2]), Math.min(c[3], v[3])] : v; }
    }
    return c && [c[0] - o.left, c[1] - o.top, c[2] - c[0], c[3] - c[1]].map(q);
  };
  const texts = [], boxes = [], lines = [], rings = [], svgs = [];
  const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n; (n = tw.nextNode());) {
    if (!n.textContent.trim()) continue;
    const e = n.parentElement;
    if (e.closest("svg")) continue;
    const cs = getComputedStyle(e), rg = document.createRange();
    rg.selectNodeContents(n);
    const rs = [...rg.getClientRects()].filter((r) => r.width > 0);
    if (!rs.length) continue;
    const base = { size: parseFloat(cs.fontSize), wt: +cs.fontWeight, mono: cs.fontFamily.startsWith('"JetBrains'), ls: cs.letterSpacing === "normal" ? 0 : parseFloat(cs.letterSpacing),
      c: ca(col(cs.color), op(e)), cls: e.className && e.className.baseVal === undefined ? e.className : "", k: clipOf(n.parentElement.firstChild ? e.firstChild : e), ell: e.scrollWidth > e.clientWidth && cs.textOverflow === "ellipsis" };
    base.k = clipOf(e.firstChild || e);
    // 줄 조각: 글자마다 상자 → 위(top)로 묶음
    const t = n.textContent, ch = [];
    for (let i = 0; i < t.length; i++) { const r1 = document.createRange(); r1.setStart(n, i); r1.setEnd(n, i + 1); const b = [...r1.getClientRects()].filter((r) => r.width > 0 || t[i] === " ")[0]; ch.push(b ? [t[i], b.left, b.top, b.width] : [t[i], null, null, 0]); }
    // 말줄임(한 줄 text-overflow · 여러 줄 line-clamp): 크로미움은 보이는 조각 상자를 따로 준다 -- 보이는 조각만 남기고 그 끝에 «…»(그려지지만 글 마디가 아니다)
    const clamp = cs.webkitLineClamp !== "none" && e.scrollHeight > e.clientHeight + 0.5;
    if (base.ell || clamp) {   // 같은 줄(위)에 상자가 둘이면 좁은 쪽 = 잘린 보이는 조각 · line-clamp 아래 숨은 줄은 버린다
      const eb = e.getBoundingClientRect().bottom, keep = [];
      for (const r of rs) { if (clamp && r.top >= eb - 0.5) continue; const j = keep.findIndex((x) => Math.abs(x.top - r.top) < 1); if (j < 0) keep.push(r); else { if (r.width < keep[j].width) keep[j] = r; keep[j].cut = true; } }
      rs.splice(0, rs.length, ...keep.map((r) => Object.assign({ left: r.left, top: r.top, right: r.right, width: r.width, height: r.height }, { cut: !!r.cut })));
    }
    rs.forEach((r) => {
      const mine = ch.filter((c) => c[2] != null && Math.abs(c[2] - r.top) < 1 && c[1] >= r.left - 0.5 && (r.cut ? c[1] + c[3] <= r.right + 0.5 : c[1] < r.right + 0.5));
      const s = mine.map((c) => c[0]).join("");
      const item = { ...base, s, r: R(r) };
      item.ch = (r.cut ? ch.filter((c) => c[2] != null) : mine).map((c) => [c[0], q(c[3])]);   // 글자별 폭(잘린 글은 숨은 글자까지 -- 말줄임 자리 찾기)
      texts.push(item);
      if (r.cut) texts.push({ ...base, s: "…", r: R({ left: r.right, top: r.top, width: 0, height: r.height }), synth: true });
    });
  }
  const all = [root, ...root.querySelectorAll("*")];
  for (const e of all) {
    const cs = getComputedStyle(e), r = e.getBoundingClientRect();
    if (e.closest("svg") && e.tagName !== "svg") continue;
    if (!r.width && !r.height) continue;
    if (cs.display === "contents") continue;
    const a = op(e), k = clipOf(e), cls = typeof e.className === "string" ? e.className : (e.className.baseVal || "");
    const bg = col(cs.backgroundColor);
    if (bg && bg[3] > 0) boxes.push({ r: R(r), c: ca(bg, a), rad: cs.borderTopLeftRadius, cls, k });
    const sides = ["Top", "Right", "Bottom", "Left"].map((s) => [s, parseFloat(cs[`border${s}Width`]), cs[`border${s}Style`], cs[`border${s}Color`]]).filter((x) => x[1] > 0 && x[2] !== "none");
    if (sides.length === 4 && sides.every((x) => x[3] === sides[0][3] && x[1] === sides[0][1])) lines.push({ side: "all", r: R(r), w: sides[0][1], c: ca(col(sides[0][3]), a), rad: cs.borderTopLeftRadius, cls, k });
    else for (const [s, w, st, c] of sides) {
      const rr = s === "Top" ? { left: r.left, top: r.top, width: r.width, height: w } : s === "Bottom" ? { left: r.left, top: r.bottom - w, width: r.width, height: w }
        : s === "Left" ? { left: r.left, top: r.top, width: w, height: r.height } : { left: r.right - w, top: r.top, width: w, height: r.height };
      lines.push({ side: s, r: R(rr), w, c: ca(col(c), a), style: st, cls, k });
    }
    if (cs.boxShadow !== "none") for (const m of cs.boxShadow.matchAll(/((?:rgba?|color)\([^)]*\))\s+(-?[\d.]+)px\s+(-?[\d.]+)px\s+([\d.]+)px\s+(-?[\d.]+)px(\s+inset)?/g))
      rings.push({ r: R(r), c: ca(col(m[1]), a), dx: +m[2], dy: +m[3], blur: +m[4], spread: +m[5], inset: !!m[6], rad: cs.borderTopLeftRadius, cls, k });
    if (e.classList.contains("mt-g")) {   // ::before ±0.5σ 띠 · ::after 가운데 선
      const b = getComputedStyle(e, "::before"), f = getComputedStyle(e, "::after");
      boxes.push({ r: R({ left: r.left + r.width * .4167, top: r.top, width: r.width * .1666, height: r.height }), c: ca(col(b.backgroundColor), a), rad: "0px", cls: "mt-g::before", k });
      boxes.push({ r: R({ left: r.left + r.width * .5, top: r.top - 2, width: 1, height: r.height + 4 }), c: ca(col(f.backgroundColor), a), rad: "0px", cls: "mt-g::after", k });
    }
    if (e.tagName === "svg") {
      const vb = e.viewBox.baseVal, els = [];
      for (const c of e.children) {
        if (c.tagName === "title") continue;
        const s = getComputedStyle(c), at = {};
        for (const x of c.attributes) at[x.name] = x.value;
        const it = { tag: c.tagName, at, stroke: col(s.stroke), fill: col(s.fill), so: +s.strokeOpacity, fo: +s.fillOpacity, op: +s.opacity, sw: parseFloat(s.strokeWidth), dash: s.strokeDasharray, ve: s.vectorEffect };
        if (c.tagName === "text") { const b = c.getBBox(); Object.assign(it, { s: c.textContent, size: parseFloat(s.fontSize), wt: +s.fontWeight, anchor: s.textAnchor, fam: s.fontFamily, bw: q(b.width) }); }
        els.push(it);
      }
      svgs.push({ r: R(r), vb: [vb.x, vb.y, vb.width, vb.height], par: e.getAttribute("preserveAspectRatio"), cls, a, color: col(cs.color), k, els });
    }
  }
  return { w: q(o.width), h: q(o.height), texts, boxes, lines, rings, svgs };
}"""

html = json.loads(pathlib.Path(os.environ.get("NEWS_HTML", FX / "news_html.json")).read_text())
inp = json.loads((FX / "news_in.json").read_text())
css = pathlib.Path(os.environ.get("STYLES_CSS", WT / "dashboard/live/styles.css")).read_text()
APPH = int(os.environ.get("APPH", 64))   # 앱바 아래 끝(app.js fitLayout --apph) -- #mtBody 가 1838×917(mt_layout 기준)이 되는 값
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
        for s, (hidden, body, when) in zip(inp, html):
            k = body.rfind("</div>")
            if body: body = body[:k] + f'<div id="mcWhen" class="mc-when">{when}</div>' + body[k:]   # renderNews 가 ⑤ 를 새 .nw-main 끝에 다시 붙인다
            pg.set_content(f'<!doctype html><html class="fit1" style="--apph:{APPH}px"><head><style>{FONT_CSS}{MONO_CSS}</style><style>{css}</style></head>'
                           f'<body class="{"" if s["asset"] == "eth" else "not-eth"}" data-coin="{s["asset"]}">'
                           f'<section id="mtCard" class="panel ops-health-panel mt-card" style="width:1872px"><div class="ops-health-head"><div class="opt-card-head"><h3 class="card-title">Macro Trend</h3><span class="muted">x</span></div></div><div id="mtBody" class="mt-body"></div></section>'
                           f'<section id="newsCard" class="panel ops-health-panel news-card" style="width:1872px"{" hidden" if hidden else ""}><div class="ops-health-head"><div class="opt-card-head"><h3 class="card-title">News</h3><span id="newsMeta" class="muted">x</span></div></div>'
                           f'<div id="newsBody" class="news-body">{body}</div></section></body></html>')
            pg.evaluate("document.fonts.ready")
            pg.evaluate("""() => Promise.all(['400 12px "Pretendard Variable"', '700 12px "Pretendard Variable"', '800 12px "Pretendard Variable"'].map((f) => document.fonts.load(f)))""")
            pg.evaluate("() => Promise.all([...document.fonts].map((f) => f.load()))")
            mt = pg.evaluate("() => { const r = document.querySelector('#mtBody').getBoundingClientRect(); return [r.width, r.height]; }")
            nb = pg.evaluate("() => { const r = document.querySelector('#newsBody').getBoundingClientRect(); return [r.width, r.height]; }")
            print("mtBody", mt, "newsBody", nb)
            assert mt == [1838, 917], mt
            if hidden: out.append(None); continue
            main = pg.evaluate("() => document.querySelector('#newsBody .nw-main').clientWidth")
            assert main == 1426, main   # news_js.mjs MAINW
            if os.environ.get("NEWS_DBG"):   # 칸 상자(자리 계산 확인용)
                print(pg.evaluate("""(qs) => qs.map((q) => [q, [...document.querySelectorAll(q)].map((e) => { const r = e.getBoundingClientRect(), o = document.querySelector('#newsBody').getBoundingClientRect(); return [r.top - o.top, r.height].map((v) => Math.round(v * 64) / 64); })])""",
                                  os.environ["NEWS_DBG"].split("|")))
            d = pg.evaluate(DUMP, "#newsBody")
            assert (d["w"], d["h"]) == (s["w"], s["h"]), (d["w"], d["h"])
            out.append(d)
        probe = pg.evaluate("""(ps) => ps.map(([s, z, w, mono]) => { const e = document.createElement("span");
            e.style.cssText = `font: ${w} ${z}px ${mono ? '"JetBrains Mono"' : '"Pretendard Variable"'}; white-space: pre; position: absolute`; e.textContent = s; document.body.append(e);
            const r = e.getBoundingClientRect().width; e.remove(); return [s, z, w, mono, Math.round(r * 1000) / 1000]; })""",
            [[" ", 12, 400, False], [" ", 12, 700, False], [" ", 12, 800, False], [" ", 11.5, 400, False], [" ", 11.5, 600, False], ["…", 12, 400, False], ["…", 12, 700, False], ["…", 12, 800, False]])
    finally:
        b.close()
pathlib.Path(os.environ.get("NEWS_OUT", FX / "news_js.json")).write_text(json.dumps({"cases": out, "probe": probe}, ensure_ascii=False) + "\n")
for k, o in enumerate(out):
    print(k, "숨김" if o is None else f'{o["w"]}x{o["h"]} 글 {len(o["texts"])} 상자 {len(o["boxes"])} 선 {len(o["lines"])} 고리 {len(o["rings"])} svg {len(o["svgs"])}')