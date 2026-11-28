// «청산» 탭 «남은 매물 일봉별» 정답지: 지금 화면의 실제 함수(liq_tab.js CH.remain · heatUrl · remView · remainKpi · 끌기·휠·두 번 클릭 손잡이 init 555~603 · 십자 가격선 el._xh)를
// 가짜 전역에서 돌린 SVG + 히트맵 캔버스 fillRect → 그리기 목록 → fixture/rem_js.json. rtui/src/rem.rs 시험 js_parity 가 같은 입력(fixture/rem_in.json)으로 맞댄다. 숫자는 전부 지어낸 값.
// 실행: node rtui/compare/rem_js.mjs   (바깥 접속 없음) · LIQ_JS=<liq_tab.js 사본> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src0 = fs.readFileSync(process.env.LIQ_JS || here + "../../dashboard/live/liq_tab.js", "utf8");
const hook = "  // ── 조립";
if (!src0.includes(hook)) throw "hook";
// densityColor = app.js 그 부분을 그대로 잘라 온다(clamp01 · DENSITY_STOPS_* · densityStops · densityColor)
const app = fs.readFileSync(process.env.APP_JS || here + "../../dashboard/live/app.js", "utf8");
const cut = (a, b) => { const i = app.indexOf(a), j = app.indexOf(b, i); if (i < 0 || j < 0) throw "app.js " + a; return app.slice(i, j); };
const dens = cut("function clamp01(v)", "\n}\n") + "\n}\n" + cut("const DENSITY_STOPS_DARK", "// 가격 꼬리표 세로 배치");
const src = dens + src0.replace(hook, "  window.__CH = CH; window.__S = S; window.__K = remainKpi;\n" + hook);   // 닫힌 함수 밖으로 그리기·상태·지표만 꺼낸다(나머지는 원본 그대로)

// ── 지어낸 입력 ──
let seed = 11;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const T0 = 1795000000;
const iso = (s) => new Date(s * 1000).toISOString().slice(0, 10);
const days = (n, m0) => { let m = m0; return Array.from({ length: n }, (_, i) => { m *= 1 + (rnd() - 0.5) * 0.05; const o = m * (1 + (rnd() - 0.5) * 0.02), c = m * (1 + (rnd() - 0.5) * 0.02), hi = Math.max(o, c) * (1 + rnd() * 0.02), lo = Math.min(o, c) * (1 - rnd() * 0.02); return [iso(T0 - (n - i) * 86400), o, hi, lo, c, Math.round(rnd() * 1e4) / 1e4, Math.round(rnd() * 1e4) / 1e4]; }); };
const cols = (n, nr) => Array.from({ length: n }, () => { if (rnd() < 0.08) return []; const a = Math.floor(rnd() * nr * 0.3), len = Math.floor(nr * (0.3 + 0.5 * rnd())); return [a, ...Array.from({ length: len }, () => (rnd() < 0.3 ? 0 : Math.round(rnd() * 1000)))]; });
const prof = (P) => Array.from({ length: 40 }, (_, i) => [Math.round(P * (0.8 + i * 0.01) * 100) / 100, i < 20 ? 0 : 1, Math.round(rnd() * 1e5) / 1e5]);
const rem = (view, lo, row, n, nr, P, kpi) => ({ view, lo, row, cols: cols(n, nr), ...(kpi ? { prof: prof(P), ll: 1e6 * rnd(), lm: 2e6, ul: 1e6 * rnd(), um: 3e6, lb: kpi === 2 ? null : P * 0.93, ub: P * 1.071, near3: rnd(), old7: rnd() } : {}) });
const tab = (dd, price, remain, ageH) => ({ ok: true, v: 1, coin: "eth", generated_ms: (T0 - (ageH || 0) * 3600) * 1000, price, days: dd, remain });
const D = days(90, 3000);
const ev = [["down", 300, 200], ["move", 302, 201], ["move", 340, 230], ["move", 380, 150], ["up"], ["wheel", 400, 200, 0, 120, false], ["wheel", 400, 200, 0, -300, false], ["wheel", 20, 120, 0, 200, false], ["wheel", 20, 300, 0, -900, false],
  ["wheel", 400, 200, 0, 80, true], ["wheel", 400, 200, 150, 10, false], ["down", 200, 100], ["move", 2000, 100], ["up"], ["dbl"], ["wheel", 400, 200, 0, -3000, false], ["down", 100, 300], ["move", 100, 700], ["up"]];
const inputs = [
  { win: 60, w: 760, h: 420, price: 3000, tab: tab(D, 2990, { 60: rem(70, 2400, 5, 70, 160, 3000, 1) }), xh: [[300, 200], [50, 30], [700, 100], [10, 10]] },
  { win: 7, w: 760, h: 420, price: 3010, tab: tab(D, 2990, { 7: rem(30, 2600, 4, 30, 200, 3010, 1) }) },
  { win: 365, w: 280, h: 220, price: 2950, tab: tab(D, 2990, { 365: rem(80, 2000, 10, 80, 100, 2950, 2) }, 5) },
  { win: 60, w: 900, h: 500, price: 0, tab: tab(D, 9000, { 60: rem(60, 2400, 5, 60, 120, 9000, 1) }) },
  { win: 30, w: 640, h: 360, price: 3000, tab: tab(D, 2990, { 30: rem(40, 2500, 5, 25, 250, 3000, 0) }) },
  { win: 60, w: 760, h: 420, price: 3000, tab: tab(D, 2990, { 60: { view: 70, lo: 2400, row: 5, cols: [] } }) },
  { win: 60, w: 760, h: 420, price: 3000, tab: null },
  { win: 90, w: 820, h: 460, price: 3000, tab: tab(D, 2990, { 90: rem(90, 2300, 5, 90, 320, 3000, 1) }), ev, xh: [[420, 260], [300, 50]] },
  { win: 180, w: 700, h: 400, price: 3000, tab: tab(D.slice(-20), 2990, { 180: rem(20, 2600, 5, 20, 200, 3000, 1) }), ev: ev.slice(5, 11) },
  { win: 30, w: 700, h: 400, price: 3000, tab: tab(D, 2990, { 30: rem(31, 2600, 5, 31, 120, 3000, 1) }), ev: [["wheel", 400, 200, 3000, 0, false]], xh: [[690, 200], [600, 300]] },   // 기간 = 날 수 − 1(시작선 경계) · 오른쪽 끝으로 민 뒤 빈 날 위 십자선
];
fs.writeFileSync(here + "fixture/rem_in.json", JSON.stringify(inputs) + "\n");

// ── 원본을 가짜 전역에서 ──
const fake = (attrs) => ({ a: attrs, setAttribute(k, v) { this.a[k] = String(v); }, textContent: "" });
function run(c) {
  const heats = {}; let hn = 0, handlers = {};
  const canvas = () => { const rec = []; const g = { scale() {}, beginPath() {}, rect(...q) { rec.clip = q; }, clip() {}, globalAlpha: 1, fillStyle: "", fillRect(x, y, w, h) { rec.push(["H", x, y, w, h, this.fillStyle, this.globalAlpha]); } };
    return { getContext: () => g, toDataURL: () => { const k = "heat" + hn++; heats[k] = rec; return k; } }; };
  const host = { addEventListener: (k, f) => { handlers[k] = f; }, querySelectorAll: () => [] };
  let el;
  const env = {
    activeSnapshotAsset: "eth", latestLivePriceByAsset: { eth: c.price }, latestMarketCtx: null, fetch: () => new Promise(() => {}), localStorage: { getItem: () => null, setItem: () => {} },
    document: { readyState: "complete", getAttribute: () => null, getElementById: (id) => (id === "lqTab" ? host : null), querySelectorAll: () => [], querySelector: () => el, documentElement: { dataset: {}, getAttribute: () => null }, addEventListener: () => {}, createElement: canvas },
    requestAnimationFrame: (f) => { f(); return 0; }, setInterval: () => 0, setTimeout: () => 0, addEventListener: () => {}, matchMedia: () => ({ matches: false }),
  };
  env.window = env;
  const p = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  new Function("px", `with (px) { ${src} }`)(p);
  const S = env.__S; S.data = c.tab; S.win = c.win;
  el = { dataset: { d: "remain" }, getBoundingClientRect: () => ({ left: 0, top: 0, width: c.w, height: c.h }), classList: { add() {}, remove() {} }, setPointerCapture() {}, closest() { return el; },
    set innerHTML(v) { el.svg = v; }, get innerHTML() { return el.svg; } };
  const draw = () => { const s = env.__CH.remain(el, c.w, c.h); el.svg = s; return s; };
  draw();
  const rv = () => (S.rv ? [S.rv.n, S.rv.i0, S.rv.span, S.rv.yOff, S.rv.yk] : null);
  const steps = [];
  for (const q of c.ev || []) {
    const e = { target: el, pointerId: 1, pointerType: "mouse", button: 0, clientX: q[1], clientY: q[2], deltaX: q[3], deltaY: q[4], shiftKey: q[5], preventDefault() {} };
    handlers[{ down: "pointerdown", move: "pointermove", up: "pointerup", wheel: "wheel", dbl: "dblclick" }[q[0]]](e);
    steps.push(rv());
  }
  const out = el.svg == null ? null : parse(el.svg, heats);
  const xh = (c.xh || []).map(([x, y]) => xhOf(el, x, y));
  return { prims: out && out.prims, clip: out && out.clip, kpi: kpiOf(env.__K()), steps, xh };
}

// ── SVG → 그리기 목록(rem.rs 시험과 같은 모양) ──
// styles.css: .lq-ax = --muted 500 10px mono · .t = --text 700 · .o = 바탕색 테 · .lq-pxt = --on-fill 700 10.5px mono · .lq-gl = --soft-line 1 · .lq-z = --neutral 38% 1 · .lq-px = --accent 1.2 «4 3» ·
//   .lq-wk = --muted 1 · .lq-cu/.lq-cd = good/bad 60% + chart-bg · .lq-sh = good · .lq-lg = bad · .lq-xl = --muted 1 «2 3» .75
const col = (v) => { const m = /^var\(--([\w-]+)\)$/.exec(v || ""); if (!m) throw "색 " + v; return m[1]; };
const lineSt = (a) => { const cl = (a.class || "").split(" ");
  return cl.includes("lq-gl") ? ["soft-line", 1, 1, null] : cl.includes("lq-z") ? ["neutral", 0.38, 1, null] : cl.includes("lq-px") ? ["accent", 1, 1.2, [4, 3]] : cl.includes("lq-wk") ? ["muted", 1, 1, null]
    : cl.includes("lq-xl") ? ["muted", 0.75, 1, [2, 3]] : a.stroke ? [col(a.stroke), +(a.opacity || 1), +(a["stroke-width"] || 1), a["stroke-dasharray"] ? a["stroke-dasharray"].split(" ").map(Number) : null] : null; };
const txtOf = (a, s) => { const cl = (a.class || "").split(" "), ax = cl.includes("lq-ax"), t = cl.includes("t"); if (!ax && !cl.includes("lq-pxt")) throw "글 " + a.class;
  return ["T", +a.x, +a.y, s, ax ? 10 : 10.5, ax ? (t ? 700 : 500) : 700, ax ? (t ? "text" : "muted") : "on-fill", a["text-anchor"] || "start", cl.includes("o")]; };
const rectCol = (a) => { const cl = a.class || ""; return cl === "lq-cu" ? "cu" : cl === "lq-cd" ? "cd" : cl === "lq-sh" ? "good" : cl === "lq-lg" ? "bad" : col(a.fill); };
const attrs = (s) => { const a = {}; for (const q of s.matchAll(/([\w-]+)="([^"]*)"/g)) a[q[1]] = q[2]; return a; };
const parse = (svg, heats) => {
  const o = []; let clip = null;
  const xh = svg.indexOf('<g class="lq-xh"'), body = xh >= 0 ? svg.slice(0, xh) : svg, g0 = body.indexOf("<g clip-path"), g1 = body.indexOf("</g>", g0);
  for (const m of body.matchAll(/<(line|text|rect|image|clipPath)\b([^>]*?)(?:\/>|>([^<]*)<\/text>|>)/g)) {
    const a = attrs(m[2]), cp = m.index > g0 && m.index < g1 ? "lane" : "no";
    if (m[1] === "clipPath") continue;
    if (m[1] === "rect" && body.lastIndexOf("<clipPath", m.index) > body.lastIndexOf("</clipPath>", m.index)) { clip = [+a.x, +a.y, +a.width, +a.height]; continue; }
    if (m[1] === "image") { for (const h of heats[a.href]) { const r = /^rgb\((\d+),(\d+),(\d+)\)$/.exec(h[5]); o.push(["H", h[1], h[2], h[3], h[4], [+r[1], +r[2], +r[3]], h[6], "plot"]); } continue; }
    if (m[1] === "line") { const st = lineSt(a); if (!st) throw "선 " + a.class; o.push(["L", +a.x1, +a.y1, +a.x2, +a.y2, ...st, cp]); }
    else if (m[1] === "text") o.push([...txtOf(a, m[3]), cp]);
    else o.push(["R", +a.x, +a.y, +a.width, +a.height, rectCol(a), +(a.opacity || 1), +(a.rx || 0), cp]);
  }
  return { prims: o, clip };
};
// 십자 가격선: 숨은 묶음(svg 끝)을 가짜 요소로 만들어 el._xh 를 부르고, 보이면 그 속성으로 목록을
function xhOf(el, x, y) {
  const s = el.svg, part = s.slice(s.indexOf('<g class="lq-xh"')), g = fake(attrs(/<g ([^>]*)>/.exec(part)[1]));
  const ls = [...part.matchAll(/<line ([^>]*)\/>/g)].map((m) => fake(attrs(m[1]))), r = fake(attrs(/<rect ([^>]*)\/>/.exec(part)[1])), t = fake(attrs(/<text ([^>]*)>/.exec(part)[1]));
  el.querySelector = (q) => (q === ".lq-xh" ? g : q === "rect" ? r : t);
  g.querySelectorAll = () => ls; g.querySelector = el.querySelector;
  el._xh(x, y);
  if (g.a.visibility !== "visible") return [];
  return [...ls.map((l) => ["L", +l.a.x1, +l.a.y1, +l.a.x2, +l.a.y2, ...lineSt(l.a), "no"]), ["R", +r.a.x, +r.a.y, +r.a.width, +r.a.height, col(r.a.fill), 1, +r.a.rx, "no"], [...txtOf(t.a, t.textContent), "no"]];
}
const kpiOf = (html) => [...html.matchAll(/<div class="lq-kp"><span>(.*?)<\/span><b class="(.*?)">(.*?)<\/b>(?:<small>(.*?)<\/small>)?<\/div>/g)].map((m) => [m[1], m[3], m[4] || "", m[2]]);

const res = inputs.map(run);
fs.writeFileSync(here + "fixture/rem_js.json", JSON.stringify(res) + "\n");
for (const [i, r] of res.entries()) console.log(i, r.prims ? `${r.prims.length}개 · 칸 ${r.prims.filter((q) => q[0] === "H").length} · 지표 ${r.kpi.length} · 걸음 ${r.steps.length} · 십자 ${r.xh.map((q) => q.length)}` : "null(빈 상태)");
