// «청산» 탭 청산맵(지도 1·7·30·60·90·180·365일) 정답지: 지금 화면의 실제 함수(liq_tab.js CH.ladder · mapOf · ladRange + liq_profile.js fpLiqColTp·liveTp·sweep)를
// 가짜 전역에서 돌린 SVG → 그리기 목록 → fixture/liq_js.json. rtui/src/liq.rs 시험 js_parity 가 같은 입력(fixture/liq_in.json)으로 맞댄다. 숫자는 전부 지어낸 값.
// 실행: node rtui/compare/liq_js.mjs   (바깥 접속 없음) · LIQ_JS=<liq_tab.js 사본> · LIQ_PROFILE_JS=<liq_profile.js 사본> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const tabSrc0 = fs.readFileSync(process.env.LIQ_JS || here + "../../dashboard/live/liq_tab.js", "utf8");
const profSrc = fs.readFileSync(process.env.LIQ_PROFILE_JS || here + "../../dashboard/live/liq_profile.js", "utf8");
const hook = "  // ── 조립";
if (!tabSrc0.includes(hook)) throw "hook";
const tabSrc = tabSrc0.replace(hook, "  window.__CH = CH; window.__S = S;\n" + hook);   // 닫힌 함수 밖으로 그리기·상태만 꺼낸다(나머지는 원본 그대로)

// ── 지어낸 입력 ──
let seed = 7;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const P0 = 3000, T0 = 1795000000;   // 지어낸 가격·시각(초)
const iso = (s) => new Date(s * 1000).toISOString().replace(".000Z", "+00:00");
const tp = (cp, bw, n, unit, gap) => {
  const lo = Math.round(cp * 0.72 / bw);
  const tiers = ["10", "20~25", "50", "75~100"].map((name, ti) => ({ name, values: Array.from({ length: n }, (_, i) => {
    const px = (lo + i) * bw, d = Math.abs(px / cp - 1);
    if (gap && d < gap) return 0;
    const v = rnd() < 0.35 ? 0 : rnd() * 1e6 * Math.exp(-d * (8 + 6 * ti));
    return unit === "share" ? Math.round(v / 3e8 * 1e8) / 1e8 : Math.round(v * 100) / 100;
  }) }));
  return { bin_width: bw, lo, current_price: cp, tiers, unit, oi_usd: unit === "share" ? null : 9.1e9, basis: "추정" };
};
const days = (n, lo, hi) => Array.from({ length: n }, (_, i) => { const m = lo + (hi - lo) * rnd(), r = m * 0.03 * rnd(); return [iso(T0 - (n - i) * 86400).slice(0, 10), m, m + r, m - r, m + r * (rnd() - 0.5), 0.4, 0.5]; });
const tab = (dd, price) => ({ ok: true, v: 1, coin: "eth", generated_ms: T0 * 1000, price, days: dd, remain: {} });
const ext = (d, t) => ({ days: d, tier_profile: t, decay: d < 60, generated_at: iso(T0 - 30) });
const map1 = (t, ku) => ({ warmed_up: true, tier_profile: { ...t, known_until: ku }, generated_at: iso(T0 - 20) });
const candles = (from, n, lo, hi) => Array.from({ length: n }, (_, i) => ({ time: from + i * 300, open: lo, high: hi, low: lo, close: lo }));
// 갈래: 7일(넓은 화면 · 오른쪽 글 있음) · 가격 셋이 지나간 칸 지움 · 1일(known_until 뒤 봉 고저로 지움 · 그 전 봉 무시) · 좁은 판(<420) · 몫 단위(OI 없음) ·
//   실시간 가격 없음 = liq-tab 가격 · 760 미만(오른쪽 글 없음) · 지도 없음(빈 상태) · 30일(일봉 범위 넓음 → 칸 2% 묶음) · 365일(범위 끝 ±100%) · 칸이 전부 0 · 지금가 0 · liq-tab 없음(범위 ±6%+1)
const inputs = [
  { days: 7, w: 900, h: 520, prices: [3000], tab: tab(days(40, 2850, 3150), 2999), map: ext(7, tp(3000, 1, 900, "oi_usd")) },
  { days: 7, w: 900, h: 520, prices: [3000, 3012.4, 2991.6, 3001], tab: tab(days(40, 2850, 3150), 2999), map: ext(7, tp(3000, 1, 900, "oi_usd")) },
  { days: 1, w: 1000, h: 560, prices: [3005], tab: tab(days(10, 2950, 3050), 3004), map: map1(tp(2998, 0.5, 3400, "oi_usd"), iso(T0 - 600)), candles: [...candles(T0 - 1200, 2, 2000, 3900), ...candles(T0 - 750, 1, 2100, 3800), ...candles(T0 - 300, 2, 2960, 3040)] },
  { days: 7, w: 400, h: 380, prices: [3000], tab: tab(days(40, 2850, 3150), 2999), map: ext(7, tp(3000, 1, 900, "oi_usd")) },
  { days: 30, w: 820, h: 500, prices: [3000], tab: tab(days(60, 2600, 3400), 2999), map: ext(30, tp(3000, 2, 600, "share")) },
  { days: 7, w: 700, h: 500, prices: [0], tab: tab(days(40, 2850, 3150), 3021.5), map: ext(7, tp(3000, 1, 900, "oi_usd", 0.004)) },
  { days: 7, w: 900, h: 520, prices: [3000], tab: tab(days(40, 2850, 3150), 2999), map: { days: 7, tier_profile: null, generated_at: iso(T0) } },
  { days: 30, w: 900, h: 520, prices: [3000], tab: tab(days(40, 1500, 3200), 2999), map: ext(30, tp(3000, 5, 600, "oi_usd")) },
  { days: 365, w: 1100, h: 600, prices: [3000], tab: tab([[iso(T0).slice(0, 10), 3000, 9000, 50, 3000, 0, 0], ...days(30, 2900, 3100)], 2999), map: ext(365, tp(3000, 10, 600, "oi_usd")) },
  { days: 7, w: 900, h: 520, prices: [3000], tab: tab(days(40, 2850, 3150), 2999), map: ext(7, { ...tp(3000, 1, 50, "oi_usd"), tiers: ["10", "20~25", "50", "75~100"].map((name) => ({ name, values: Array(50).fill(0) })) }) },
  { days: 7, w: 900, h: 520, prices: [0], tab: null, map: ext(7, tp(3000, 1, 900, "oi_usd")) },
  { days: 7, w: 600, h: 450, prices: [3000], tab: null, map: ext(7, tp(3000, 1, 900, "oi_usd")) },
  { days: 90, w: 1300, h: 640, prices: [2875.25], tab: tab(days(120, 2500, 3300), 2870), map: ext(90, tp(2870, 1, 1500, "oi_usd")) },
];
fs.writeFileSync(here + "fixture/liq_in.json", JSON.stringify(inputs) + "\n");

// ── 원본을 가짜 전역에서 ──
const realTimeout = setTimeout;
async function run(c) {
  const fetchFake = (url) => Promise.resolve({ ok: true, json: () => Promise.resolve(url.includes("/tiers") ? c.map : null) });
  const env = {
    activeSnapshotAsset: "eth", latestLivePriceByAsset: { eth: 0 }, latestMarketCtx: null, candleHistoryByAsset: { eth: c.candles || [] },
    latestLiquidationMap: c.days === 1 ? c.map : null, fetch: fetchFake, localStorage: { getItem: () => null, setItem: () => {} },
    document: { readyState: "complete", getElementById: () => null, querySelectorAll: () => [], documentElement: { dataset: {} }, addEventListener: () => {} },
    setInterval: () => 0, setTimeout: () => 0, CustomEvent: function () {}, dispatchEvent: () => {}, addEventListener: () => {}, matchMedia: () => ({ matches: false }),
  };
  env.window = env;
  const p = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  new Function("px", `with (px) { ${profSrc}\n${tabSrc} }`)(p);
  env.__S.data = c.tab; env.__S.lad = String(c.days);
  if (c.days > 1) { env.fpLiqColTp(c.days); await new Promise((r) => realTimeout(r, 0)); }   // 7일 이상 = 첫 부름이 /tiers 를 묻는다
  let out = null;
  for (const px of c.prices) { env.latestLivePriceByAsset = { eth: px }; out = env.__CH.ladder({}, c.w, c.h); }
  return out;
}

// ── SVG → 그리기 목록(liq.rs 시험과 같은 모양) ──
// styles.css: .lq-ax = --muted 500 10px mono · .t = --text 700 · .lq-axh = --text 600 · .o = 바탕색 테 · .lq-pxt = --on-fill 700 10.5px mono · .lq-gl = --soft-line 1 · .lq-z = --neutral 38% 1 · .lq-px = --accent 1.2 «4 3»
const col = (v) => { const m = /^var\(--([\w-]+)\)$/.exec(v || ""); if (!m) throw "색 " + v; return m[1]; };
const pts = (d) => [...d.matchAll(/[ML]([-\d.]+) ([-\d.]+)/g)].map((m) => [+m[1], +m[2]]);
const parse = (svg) => {
  const o = [];
  for (const m of svg.matchAll(/<(line|text|rect|path)\b([^>]*?)(?:\/>|>([^<]*)<\/text>)/g)) {
    const a = {}; for (const q of m[2].matchAll(/([\w-]+)="([^"]*)"/g)) a[q[1]] = q[2];
    const cl = (a.class || "").split(" ");
    if (m[1] === "line") {
      const st = cl.includes("lq-gl") ? ["soft-line", 1, 1, null] : cl.includes("lq-z") ? ["neutral", 0.38, 1, null] : cl.includes("lq-px") ? ["accent", 1, 1.2, [4, 3]] : null;
      if (!st) throw "선 " + a.class;
      o.push(["L", +a.x1, +a.y1, +a.x2, +a.y2, ...st]);
    } else if (m[1] === "text") {
      const ax = cl.includes("lq-ax"), t = cl.includes("t"), h = cl.includes("lq-axh");
      if (!ax && !cl.includes("lq-pxt")) throw "글 " + a.class;
      o.push(["T", +a.x, +a.y, m[3], ax ? 10 : 10.5, ax ? (t ? 700 : h ? 600 : 500) : 700, ax ? (t || h ? "text" : "muted") : "on-fill", a["text-anchor"] || "start", cl.includes("o"), true]);
    } else if (m[1] === "rect") {
      o.push(["R", +a.x, +a.y, +a.width, +a.height, col(a.fill), 1, +(a.rx || 0)]);
    } else if (a.fill === "none") {
      o.push(["P", pts(a.d), col(a.stroke), +a["stroke-width"]]);
    } else {
      if (!a.d.endsWith("Z")) throw "면";
      o.push(["A", pts(a.d), col(a.fill), +a.opacity]);
    }
  }
  return o;
};
const res = [];
for (const c of inputs) { const s = await run(c); res.push(s == null ? null : parse(s)); }
fs.writeFileSync(here + "fixture/liq_js.json", JSON.stringify(res) + "\n");
for (const [i, r] of res.entries()) console.log(i, r ? `${r.length}개 · 막대 ${r.filter((q) => q[0] === "R").length} · 머리 ${r.find((q) => q[0] === "T" && q[5] === 600)[3]}` : "null(빈 상태)");
