// «청산» 탭 아랫줄 세 칸(오늘 누적 · 최대 고통 · 코인별 최대 고통) 정답지: 지금 화면의 실제 함수(liq_tab.js CH.today · today · toBar · CH.pain · painOf · sweptRange ·
// CH.quad · coinRows · painKpi · coinTable · draw 의 lqPainLeg)를 가짜 전역에서 돌린 SVG·HTML → 그리기 목록·글 → fixture/liqx_js.json.
// rtui/src/liqx.rs 시험 js_parity 가 같은 입력(fixture/liqx_in.json)으로 맞댄다. 숫자는 전부 지어낸 값.
// 실행: node rtui/compare/liqx_js.mjs   (바깥 접속 없음) · LIQ_JS=<liq_tab.js 사본> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src0 = fs.readFileSync(process.env.LIQ_JS || here + "../../dashboard/live/liq_tab.js", "utf8");
const hook = "  // ── 조립";
if (!src0.includes(hook)) throw "hook";
const src = src0.replace(hook, "  window.__CH = CH; window.__S = S; window.__toBar = toBar; window.__draw = draw;\n" + hook);   // 닫힌 함수 밖으로 그리기·상태만 꺼낸다(나머지는 원본 그대로)

// ── 지어낸 입력 ──
let seed = 23;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const D0 = 1795046400;   // UTC 자정(지어낸 날)
const iso = (s) => new Date(s * 1000).toISOString().replace(".000Z", "+00:00");
const r2 = (x) => Math.round(x * 100) / 100;
const venue = () => (rnd() < 0.25 ? undefined : { long_usd: r2(rnd() * 3e4), short_usd: r2(rnd() * 3e4), n: 2 });
// 5분봉: from(초)부터 n개 · 금액은 지어낸 값 · 거래소 몫은 가끔 없음
const bars = (from, n) => Array.from({ length: n }, (_, i) => ({ ts: iso(from + i * 300), long_usd: r2(rnd() * 2e5), short_usd: r2(rnd() * 2e5), events: Math.floor(rnd() * 40), okx: rnd() < 0.5, okx_detail: venue(), bybit: venue(), hl: venue() }));
const band = (n, k) => { let a = 0, b = 0, c = 0; const p25 = [], p50 = [], p75 = []; for (let i = 0; i < 288; i++) { a += rnd() * 4e4 * k; b = a * (1.2 + rnd() * 0.1); c = b * (1.3 + rnd() * 0.2); p25.push(Math.round(a)); p50.push(Math.round(b)); p75.push(Math.round(c)); } return { n, p25, p50, p75 }; };
const tab = (price, b) => ({ ok: true, v: 1, coin: "eth", generated_ms: (D0 - 1800) * 1000, price, days: [], remain: {}, band: b });
const PERS = ["12H", "24H", "48H", "3D", "1W", "2W", "1M"];
// 1시간봉 72개(마지막 = t1 시작) · 종가가 m 근처를 걷는다
const k3 = (t1, m, n = 72) => { let c = m; return Array.from({ length: n }, (_, i) => { const o = c; c = o * (1 + (rnd() - 0.5) * 0.012); const hi = Math.max(o, c) * (1 + rnd() * 0.004), lo = Math.min(o, c) * (1 - rnd() * 0.004); return [t1 - (n - 1 - i) * 3600, r2(o), r2(hi), r2(lo), r2(c)]; }); };
// 주기마다 쪽별 봉우리 3개(아래 = 가격 밑 · 위 = 가격 위) · 금액 단위 몫이면 합 1 쪽 작은 수
const bins = (P, below, sc, share) => Array.from({ length: 3 }, (_, i) => [r2(P * (below ? 1 - (0.004 + rnd() * 0.06 * sc) : 1 + 0.004 + rnd() * 0.06 * sc)), share ? Math.round(rnd() * 1e4) / 1e5 : Math.round(rnd() * 5e7)]);
const coin = (P, gen, opt = {}) => {
  const per = {};
  PERS.forEach((k, j) => { if (opt.skip && opt.skip.includes(k)) return; per[k] = { dn: opt.noDn && opt.noDn.includes(k) ? [] : bins(P, true, 1 + j * 0.6, opt.share), up: bins(P, false, 1 + j * 0.6, opt.share) }; });
  if (opt.fix) opt.fix(per, P);
  return { generated_ms: gen, price: P, unit: opt.share ? "share" : "oi_usd", sigma_d: opt.sigma || 0.03, per, k3: opt.k3 || k3(Math.floor(gen / 3600000) * 3600 - 3600, P) };
};
const NOW = (D0 + 15 * 3600 + 13 * 60 + 7) * 1000;   // 그날 15:13:07 UTC
const g0 = NOW - 20 * 60000;   // 20분 전 계산
const coins = (P) => ({ btc: coin(91000, g0), eth: coin(P, g0), sol: coin(140, g0, { sigma: 0.045 }), xrp: coin(2.1, NOW - 3 * 3600e3, { sigma: 0.05 }) });
// 가격이 지나간(5분봉 고저가 닿은) 봉우리를 빼는 갈래: 계산 뒤 봉 = k3 끝 + 1시간부터
const candles = (t0, n, lo, hi) => Array.from({ length: n }, (_, i) => ({ time: t0 + i * 300, open: lo, high: hi, low: lo, close: lo }));
const nestFix = (per, P) => { ["12H", "24H", "48H", "3D", "1W"].forEach((k, i) => { per[k].up[0] = [r2(P * 1.02 * (1 + i * 0.0005)), 9e7 + i]; }); per["24H"].dn = [[r2(P * 0.995), 6e7], [r2(P * 0.97), 5.8e7], [r2(P * 0.95), 1e6]]; };
const sweepFix = (per, P) => { per["1W"].dn.push([r2(P * 0.985), 9e9]); per["1W"].up.push([r2(P * 1.01), 9e9]); };   // 큰 봉우리를 지나간 범위 안에
const tightFix = (per, P) => { per["24H"].dn = [[r2(P * 0.99), 4e7]]; per["24H"].up = [[r2(P * 1.012), 3e7]]; };
const inputs = [
  // 0 넓은 판 · 4코인 · 띠·오늘 · 24H
  { now: NOW, asset: "eth", pp: "24H", live: { eth: 3000, btc: 91200, sol: 0 }, tod: [340, 300], pain: [760, 340], quad: [380, 300], tab: tab(2995, band(14, 1)), bars: bars(D0 - 3600, 12 + 183), all: { ok: true, coins: coins(3000) }, candles: [] },
  // 1 좁은 판(사다리 w<440 · 오늘 w<280) · 1주 · 지나간 봉우리(5분봉 고저) · 몫 단위 코인
  { now: NOW, asset: "eth", pp: "1W", live: { eth: 3010 }, tod: [260, 220], pain: [420, 300], quad: [300, 260], tab: tab(2995, band(9, 2)), bars: bars(D0, 120),
    all: { ok: true, coins: { eth: coin(3010, g0, { share: true, fix: sweepFix }), btc: coin(90500, g0), doge: coin(0.31, g0, { share: true, sigma: 0.06 }) } }, candles: candles(Math.floor(g0 / 3600000) * 3600, 4, 2960, 3075) },
  // 2 띠 없음(liq-tab 없음 → 가격 = 실시간) · 오늘 봉만 · 다른 코인 하나뿐(사분면 없음) · 12시간
  { now: NOW, asset: "eth", pp: "12H", live: { eth: 2988.5 }, tod: [340, 300], pain: [760, 340], quad: [380, 300], tab: null, bars: bars(D0, 60), all: { ok: true, coins: { eth: coin(2988.5, g0), btc: coin(91000, g0, { noDn: ["12H"] }) } }, candles: [] },
  // 3 오늘 봉 없음(띠만) · 아래 봉우리 없음(1달) · 협공·1% 이내(1일) · 실시간 가격 없음 → liq-tab 가격
  { now: NOW, asset: "eth", pp: "24H", live: {}, tod: [340, 300], pain: [760, 340], quad: [380, 300], tab: tab(3002, band(20, 1)), bars: bars(D0 - 7200, 20),
    all: { ok: true, coins: { eth: coin(3002, g0, { noDn: ["1M"], fix: tightFix, sigma: 0.04 }), sol: coin(141, g0), bnb: coin(610, g0, { fix: tightFix }) } }, candles: [] },
  // 4 전부 없음 → 세 칸 빈 상태
  { now: NOW, asset: "eth", pp: "24H", live: { eth: 3000 }, tod: [340, 300], pain: [760, 340], quad: [380, 300], tab: null, bars: [], all: null, candles: [] },
  // 5 오래된 계산(사분면 흐림·표 old·범례 경고) · 중첩 5/7 · 비슷한 2등 · 끝 글이 오른쪽에 닿음(늦은 시각) · 1달
  { now: (D0 + 23 * 3600 + 58 * 60) * 1000, asset: "sol", pp: "1M", live: { sol: 138.2, eth: 3001 }, tod: [300, 260], pain: [700, 360], quad: [420, 320], tab: tab(138, band(30, 3)), bars: bars(D0, 287),
    all: { ok: true, coins: { sol: coin(138.2, NOW - 4 * 3600e3, { fix: nestFix, sigma: 0.05 }), eth: coin(3000, NOW - 5 * 3600e3), btc: coin(91000, g0), sui: coin(3.4, NOW - 9 * 3600e3, { sigma: 0.07 }) } }, candles: [] },
  // 6 1시간봉 1개뿐 → 사다리 없음 · 표·사분면은 그린다 · 2주
  { now: NOW, asset: "eth", pp: "2W", live: { eth: 3000 }, tod: [340, 300], pain: [760, 340], quad: [380, 300], tab: tab(2995, band(14, 1)), bars: bars(D0, 3),
    all: { ok: true, coins: { eth: coin(3000, g0, { k3: k3(D0, 3000, 1) }), btc: coin(91000, g0), sol: coin(140, g0) } }, candles: [] },
  // 7 넓은 오늘 칸 · 3일 · 48H 없는 다른 코인
  { now: NOW, asset: "btc", pp: "3D", live: { btc: 90900, eth: 2999 }, tod: [520, 340], pain: [980, 400], quad: [460, 360], tab: tab(90880, band(25, 40)), bars: bars(D0, 180),
    all: { ok: true, coins: { btc: coin(90900, g0), eth: coin(3000, g0, { skip: ["48H"] }), sol: coin(140, g0), xrp: coin(2.1, g0, { sigma: 0.05 }) } }, candles: candles(Math.floor(g0 / 3600000) * 3600 + 300, 3, 90000, 91900) },
  // 8 좁은 오늘 칸(끝 글이 오른쪽 60px 안 · 30px 밖) · 3시간 전 계산 코인이 지금 코인(범례 경고 90분~5시간)
  { now: NOW, asset: "xrp", pp: "24H", live: { xrp: 2.11, eth: 3000 }, tod: [170, 200], pain: [600, 320], quad: [360, 300], tab: tab(2.1, band(14, 1)), bars: bars(D0 - 3600, 12 + 183), all: { ok: true, coins: coins(3000) }, candles: [] },
];
for (const c of inputs) c.allText = JSON.stringify(c.all);   // 응답 글 그대로(코인 차례 = 넣은 차례 -- 시험은 이 글을 읽는다)
fs.writeFileSync(here + "fixture/liqx_in.json", JSON.stringify(inputs) + "\n");

// ── 원본을 가짜 전역에서 ──
function run(c) {
  class FD extends Date { constructor(...a) { super(...(a.length ? a : [c.now])); } static now() { return c.now; } }
  const els = {}, host = { innerHTML: "", querySelectorAll: () => [], addEventListener: () => {} };
  const env = {
    activeSnapshotAsset: c.asset, latestLivePriceByAsset: c.live, latestMarketCtx: null, candleHistoryByAsset: { [c.asset]: c.candles }, Date: FD,
    fetch: () => new Promise(() => {}), localStorage: { getItem: () => null, setItem: () => {} },
    document: { readyState: "complete", getElementById: (id) => (id === "lqTab" ? host : els[id] || (els[id] = { innerHTML: "" })), querySelectorAll: () => [], documentElement: { dataset: {} }, addEventListener: () => {} },
    requestAnimationFrame: () => 0, setInterval: () => 0, setTimeout: () => 0, addEventListener: () => {}, matchMedia: () => ({ matches: false }),
  };
  env.window = env;
  const p = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  new Function("px", `with (px) { ${src} }`)(p);
  const S = env.__S;
  Object.assign(S, { on: true, pp: c.pp, data: c.tab, all: c.all, bars: c.bars.map(env.__toBar) });
  env.__draw();
  const ch = (k, wh) => { const s = env.__CH[k]({}, wh[0], wh[1]); return s == null ? null : parse(s); };
  return { today: ch("today", c.tod), pain: ch("pain", c.pain), quad: ch("quad", c.quad), kpi: kpiOf(els.lqPainKpi.innerHTML), table: tableOf(els.lqCoTbl.innerHTML), leg: legOf(els.lqPainLeg.innerHTML) };
}

// ── SVG → 그리기 목록(liqx.rs 시험과 같은 모양) ──
// styles.css: .lq-ax = --muted 500 10px mono · .t = --text 700 · .o = 바탕색 테 · .lq-pxt = --on-fill 700 10.5px mono · .lq-gl = --soft-line 1 · .lq-z = --neutral 38% 1 · .lq-px = --accent 1.2 «4 3» ·
//   .lq-wk = --muted 1 · .lq-ln = --accent 1.8 · .lq-ln2 = --muted 1.2 «3 3» · .lq-bd = --neutral 15% 면 · .lq-dt = --accent · .lq-cu/.lq-cd = good/bad 60% + chart-bg
const col = (v) => { const m = /^var\(--([\w-]+)\)$/.exec(v || ""); if (!m) throw "색 " + v; return m[1]; };
const fill = (v) => { let m = /^color-mix\(in srgb,var\(--([\w-]+)\) (\d+)%,transparent\)$/.exec(v); if (m) return [m[1], +m[2] / 100]; m = /^rgb\(var\(--lift\) \/ ([.\d]+)\)$/.exec(v); if (m) return ["white", +m[1]]; return [col(v), 1]; };
const attrs = (s) => { const a = {}; for (const q of s.matchAll(/([\w-]+)="([^"]*)"/g)) a[q[1]] = q[2]; return a; };
const pts = (d) => [...d.matchAll(/[ML]([-\d.]+) ([-\d.]+)/g)].map((m) => [+m[1], +m[2]]);
const dash = (a) => (a["stroke-dasharray"] ? a["stroke-dasharray"].split(" ").map(Number) : null);
const parse = (svg) => {
  const o = [];
  for (const m of svg.matchAll(/<(line|text|rect|path|circle)\b([^>]*?)(?:\/>|>([^<]*)<\/text>)/g)) {
    const a = attrs(m[2]), cl = (a.class || "").split(" ");
    if (m[1] === "line") {
      const st = cl.includes("lq-gl") ? ["soft-line", 1, 1, null] : cl.includes("lq-z") ? ["neutral", 0.38, 1, null] : cl.includes("lq-px") ? ["accent", 1, 1.2, [4, 3]] : cl.includes("lq-wk") ? ["muted", 1, 1, null]
        : cl.includes("lq-ln2") ? ["muted", 1, 1.2, [3, 3]] : a.stroke ? [col(a.stroke), +(a.opacity || 1), +a["stroke-width"], dash(a)] : null;
      if (!st) throw "선 " + a.class;
      o.push(["L", +a.x1, +a.y1, +a.x2, +a.y2, ...st]);
    } else if (m[1] === "text") {
      const ax = cl.includes("lq-ax"), t = cl.includes("t");
      if (!ax && !cl.includes("lq-pxt")) throw "글 " + a.class;
      o.push(["T", +a.x, +a.y, m[3], ax ? 10 : 10.5, ax ? (t ? 700 : 500) : 700, ax ? (t ? "text" : "muted") : "on-fill", a["text-anchor"] || "start", cl.includes("o")]);
    } else if (m[1] === "rect") {
      const f = a.class === "lq-cu" ? ["cu", 1] : a.class === "lq-cd" ? ["cd", 1] : fill(a.fill);
      o.push(["R", +a.x, +a.y, +a.width, +a.height, ...f, +(a.rx || 0)]);
    } else if (m[1] === "circle") {
      o.push(["C", +a.cx, +a.cy, +a.r, a.class === "lq-dt" ? "accent" : col(a.fill), +(a.opacity || 1)]);
    } else if (a.class === "lq-bd") {
      if (!a.d.endsWith("Z")) throw "면";
      o.push(["A", pts(a.d), "neutral", 0.15]);
    } else o.push(["P", pts(a.d), ...(a.class === "lq-ln" ? ["accent", 1.8, null] : a.class === "lq-ln2" ? ["muted", 1.2, [3, 3]] : (() => { throw "길 " + a.class; })())]);
  }
  return o;
};
// painKpi: 지표 넷 [제목, 값, 아래 글, 색] · 가까운 쪽 [쪽, 괄호 글, 인력 글] · 꼬리표 [노랑?, 글]
const kpiOf = (h) => {
  if (!h) return null;
  const kp = [...h.matchAll(/<div class="lq-kp"><span>(.*?)<\/span><b class="(.*?)">(.*?)<\/b>(?:<small>(.*?)<\/small>)?<\/div>/g)].map((m) => [m[1], m[3], m[4] || "", m[2]]);
  const pl = /<div class="lq-pl"><span>가까운 쪽 <b>(.*?)<\/b>(.*?)<\/span><span>(.*?)<\/span><\/div>/.exec(h);
  const tags = [...h.matchAll(/<span class="lq-tag( lit)?"(?: title="[^"]*")?>(.*?)<\/span>/g)].map((m) => [!!m[1], m[2]]);
  return { kp, pl: [pl[1], pl[2], pl[3]], tags };
};
// coinTable: 줄마다 [종목, on|old|"", 협공?, 아래 [굵은 글, 작은 글]|null, 위 …, 가까운] · 빈 상태 = null
const cellOf = (s) => { const m = /^<b>(.*?)<\/b> <small>(.*?)<\/small>$/.exec(s); return m ? [m[1], m[2]] : (s === "—" ? null : (() => { throw "칸 " + s; })()); };
const tableOf = (h) => (h.startsWith('<p class="lq-empty">') ? null : [...h.matchAll(/<tr(?: class="(on|old)"(?: title="[^"]*")?)?><td>(\w+)( <span class="lq-tag lit">협공<\/span>)?<\/td><td class="lg">(.*?)<\/td><td class="sh">(.*?)<\/td><td>(.*?)<\/td><\/tr>/g)]
  .map((m) => [m[2], m[1] || "", !!m[3], cellOf(m[4]), cellOf(m[5]), m[6]]));
const legOf = (h) => [...h.matchAll(/<span( class="lq-warn")?>(.*?)<\/span>/g)].map((m) => [m[2], !!m[1]]);

const res = inputs.map(run);
fs.writeFileSync(here + "fixture/liqx_js.json", JSON.stringify(res) + "\n");
const n = (v) => (v ? v.length : "null");
for (const [i, r] of res.entries()) console.log(i, `오늘 ${n(r.today)} · 고통 ${n(r.pain)} · 사분면 ${n(r.quad)} · 지표 ${r.kpi ? r.kpi.kp.length + "/" + r.kpi.tags.length : "null"} · 표 ${n(r.table)} · 범례 ${r.leg.length}`);
