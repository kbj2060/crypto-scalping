// 1초 수급 정답지: 지금 화면의 실제 함수(app.js applySupply1s · mergedSupplySrc · renderSupply1s · fmtFootprintQty)를 떼어
// 가짜 svg 에 그린 결과 → fixture/s1_js.json. rtui/src/s1.rs 시험 js_parity 가 같은 입력(fixture/s1_in.json)으로 맞댄다.
// 실행: TZ=Asia/Seoul node rtui/compare/s1_js.mjs   (바깥 접속 없음 · 브라우저 없음)
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src = fs.readFileSync(here + "../../dashboard/live/app.js", "utf8");
const fn = (name) => { const i = src.indexOf(`\nfunction ${name}(`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n}\n", i) + 3); };
// 입력(결정적 의사난수): 스냅숏 supply 한 덩이 + supply1s 증분 둘(같은 초 덮어쓰기 · partial · 바이낸스 OI 는 SSE 몫이라 "oi" 칸으로)
let seed = 7; const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const r3 = (x) => Math.round(x * 1000) / 1000, T0 = 1795156500, NOW = T0 + 157;   // 봉 시작 15:35 KST · 157초 경과
const row = (s) => { const rb = r3(rnd() * 3), rs = r3(rnd() * 3), wb = rnd() < 0.2 ? r3(rnd() * 40) : 0, ws = rnd() < 0.2 ? r3(rnd() * 40) : 0; return [s, rb, rs, wb, ws, r3(rb + wb + rnd() * 20), r3(rs + ws + rnd() * 20), r3(3000 + rnd() * 5)]; };
const secs = [...Array(150).keys()].map((i) => T0 - 20 + i).filter((s) => !(s > T0 + 40 && s < T0 + 52));   // 11초 빔 하나
const snap = { now: NOW - 10, retailMaxUsd: 1000, whaleMinUsd: 50000, seconds: [...secs.map(row), [T0 + 61, 2, 0, 10, 0, 5, 1, 3001]], partial: { bn: row(NOW - 9), okx: null, spot: null },
  liq: [[T0 + 30, 2.5, 0, 7500, 0], [T0 + 90, 0, 40, 0, 120000]], spot: secs.filter((s) => s % 3 === 0).map(row), spotNow: NOW - 10, spotMeta: { connected: true, tradeAge: 0.4, errors: 0 },
  okx: secs.filter((s) => s % 2 === 0).map(row), okxNow: NOW - 10, okxMeta: { connected: true, tradeAge: 1.2, oiAge: 2, inst: "ETH-USDT-SWAP", errors: 0 },
  okxLiq: [[T0 + 90, 0, 3, 0, 9000]], okxOi: [[T0 - 15, 100000], [T0, 100010], [T0 + 10, 100020], [T0 + 33, 100005], [T0 + 70, 100060], [T0 + 125, 100040]],
  bybitLiq: [[T0 + 30, 1, 0, 3000, 0]], oi: [[T0 - 4, 200000], [T0, 200005], [T0 + 5, 200010], [T0 + 20, 199990], [T0 + 64, 200030], [T0 + 120, 200100]] };
const inc1 = { now: NOW - 3, seconds: [row(NOW - 9), row(NOW - 8), row(NOW - 4)], partial: { bn: row(NOW - 3), okx: row(NOW - 3), spot: null }, okxNow: NOW - 3, okxMeta: { tradeAge: 0.8 }, spotNow: NOW - 10, spotMeta: { tradeAge: 7 } };
const inc2 = { now: NOW, seconds: [row(NOW - 3), row(NOW - 1)], okx: [row(NOW - 3)], partial: { bn: row(NOW), okx: null, spot: null }, okxNow: NOW, okxMeta: { tradeAge: 0.3 }, spotNow: NOW, spotMeta: { tradeAge: 0.5 }, liq: [[NOW - 1, 0.4, 0, 1200, 0]] };
const msgs = [snap, inc1, inc2];
fs.writeFileSync(here + "fixture/s1_in.json", JSON.stringify(msgs) + "\n");
// 가짜 DOM: 요소 = {tag, a(속성), kids, textContent}
const el = (tag) => ({ tag, a: {}, kids: [], style: {}, textContent: "", setAttribute(k, v) { this.a[k] = String(v); }, appendChild(c) { this.kids.push(c); return c; }, remove() { this.gone = true; }, getComputedTextLength() { return 0; } });
const svg = el("svg"); Object.defineProperty(svg, "innerHTML", { set() { svg.kids = []; } }); svg.querySelector = () => svg.kids.find((k) => k.tag === "path") || null;
const document = { createElementNS: (_ns, tag) => el(tag) };
const body = ["fmtFootprintQty", "applySupply1s", "applyOi1s", "mergedSupplySrc", "renderSupply1s"].map(fn).join("\n");
const run = (document, msgs, W, H, svg) => new Function("document", "msgs", "W", "H", "svg", `
  const SUPPLY_1S_SEGMENT = 300;
  let supply1s = new Map(), oi1s = new Map(), liq1s = new Map(), okxSupply1s = new Map(), okxOi1s = new Map(), okxLiq1s = new Map(), spotSupply1s = new Map(), bybitLiq1s = new Map();
  let supply1sSince = 0, oi1sSince = 0, liq1sSince = 0, okxSupply1sSince = 0, okxOi1sSince = 0, okxLiq1sSince = 0, spotSupply1sSince = 0, bybitLiq1sSince = 0, supply1sVer = 0;
  let supply1sMeta = { now: 0 }, okxMeta = { now: 0 }, spotMeta = { now: 0 };
  const latestMarketCtx = null, activeSnapshotAsset = "eth", isMobileChartMode = () => false, measureTextW = (t, f) => t.length * parseFloat(/(\\d+(\\.\\d+)?)px/.exec(f)[1]) * 0.5;
  const fmtUsdCompact = (v) => String(v), coinUnit = () => "ETH", mcLiqBurstSide = () => null, chipScoreLine = () => "";
  ${body}
  for (const m of msgs) applySupply1s(m);
  renderSupply1s({ svg, w: W, h: H });
`)(document, msgs, W, H, svg);
const res = {};
for (const W of [400, 900]) {
const H = 300;
run(document, msgs, W, H, svg);
// 경로 d → 조각별 점 목록(M 이 새 조각) · 한 점 조각은 svg 가 안 그린다
const pathPts = (d) => d.split(/ (?=[ML])/).reduce((acc, t) => { const [x, y] = t.slice(1).split(" ").map(Number); if (t[0] === "M") acc.push([]); acc[acc.length - 1].push([x, y]); return acc; }, []).filter((p) => p.length >= 2);
const out = { texts: [], paths: [], circles: [], rects: [] };
for (const k of svg.kids) {
  if (k.tag === "text") out.texts.push(k.textContent);
  if (k.tag === "path") for (const p of pathPts(k.a.d)) out.paths.push({ pts: p, dash: k.a["stroke-dasharray"] || null, lw: +k.a["stroke-width"] });
  if (k.tag === "circle") out.circles.push([+k.a.cx, +k.a.cy, +k.a.r]);
  if (k.tag === "rect") out.rects.push([+k.a.x, +k.a.y, +k.a.width, +k.a.height]);
}
res["w" + W] = out;
console.log(`w=${W}: texts ${out.texts.length} · paths ${out.paths.length} · circles ${out.circles.length} · rects ${out.rects.length}`);
}
fs.writeFileSync(here + "fixture/s1_js.json", JSON.stringify(res) + "\n");
