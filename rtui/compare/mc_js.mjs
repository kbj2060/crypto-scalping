// 시장 맥락 ①② 정답지: 지금 화면의 실제 함수(app.js renderMarketCtx · mcGfx)를 떼어 넓은 화면(#mcBody.mc-cmp) 분기로 돌린 HTML → fixture/mc_js.json.
// rtui/src/mc.rs 시험 js_parity 가 같은 입력(fixture/mc_in.json)으로 글자·색 갈래·게이지 도형을 맞댄다(자리 배치는 CSS 라 node 로 못 잰다 -- mc.rs 머리 주석).
// 실행: node rtui/compare/mc_js.mjs   (바깥 접속 없음 · 브라우저 없음) · 변이: APP_JS=<app.js 사본> 로 돌린 뒤 cargo test mc:: 가 실패해야 한다(끝나면 원본으로 다시 돌린다)
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src = fs.readFileSync(process.env.APP_JS || here + "../../dashboard/live/app.js", "utf8");   // APP_JS = 변이 시험용 사본
const fn = (name) => { const i = src.indexOf(`\nfunction ${name}(`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n}\n", i) + 3); };
const obj = (name) => { const i = src.indexOf(`\nconst ${name} = {`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n};\n", i) + 4); };
// 입력: 갈래를 고루 밟는 /api/market-context 응답 셋(값 · 빈 값 · 기본값 펀딩 · 합산 없음 · 과열 칩)
const A = { asset: "eth", available: true, lev: { key: "long_crowd", label: "롱 과밀 — 펀딩·OI 같이 높다" },
  funding: { bn: 0.00031, bn_pct180: 0.953, bn_at_base: false }, basis: { bp: -2.345, pct7d: 0.31 },
  oi: { z30m: 0.4, all: { z30m: -1.74, d30m_pct: -0.12, venues: ["binance", "okx"] } },
  flow: { cvd30_all_z: 1.26, cvd30_z: 9, z30h_all: { whale: 2.61, mid: -0.34, retail: null }, venues: ["binance", "okx"] }, quad: { key: "dn_up", label: "x" } };
const B = { asset: "eth", available: true, lev: { key: "normal", label: "보통" }, funding: { bn_at_base: true, bn_pct180: 0.5 }, basis: { bp: 0.04, pct7d: 0.02 },
  oi: { z30m: 3.9 }, flow: { cvd30_z: -0.49, z30h: { whale: -4, mid: 0.5, retail: -0.5 } }, quad: { key: "small" } };
const C = { asset: "eth", available: true, flow: {}, oi: { z30m: 1.2 }, quad: null };   // OI 1.2 = 색 문턱(1.5) 아래 갈래
const inputs = [A, B, C];
fs.writeFileSync(here + "fixture/mc_in.json", JSON.stringify(inputs) + "\n");
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const run = (d, BW) => {
  let html = "";
  const body = { style: {}, classList: { contains: (c) => c === "mc-cmp" }, clientWidth: BW, children: [], set innerHTML(v) { html = v; }, getBoundingClientRect: () => ({ top: 0, bottom: 0 }) };
  // 이 분기가 실제로 부르는 것만 진짜(mcGfx · renderMarketCtx) -- 나머지(다른 카드 · DOM · 팝업)는 빈 값. 모르는 이름 = undefined(부르면 터져서 알려 준다)
  const env = { latestMarketCtx: d, activeSnapshotAsset: "eth", flowOn: () => true, coinUnit: () => "ETH", el: (id) => (id === "mcBody" ? body : null), escapeHtml: esc,
    helpAttrs: () => "", fcRowHtml: () => "", setH: () => {}, mtHeadHtml: () => "", macroSpikeChips: () => "", renderPaper: () => {}, nwWhenFit: () => {},
    keepFocus: (b, f) => f(), mcFoldApply: () => {}, isMobileChartMode: () => false, optFront: () => null, mcUsSession: () => null, isTodayOrTomorrowLocal: () => false,
    lineSt: () => null, LINE_DEF: new Proxy({}, { get: () => [""] }), mcLine: {}, lineCfgHtml: () => "", MC_TIPS: {}, ASSET_CONFIG: { eth: { dp: 2 } }, STAT_KEYS: [], fpIndOpen: false, mcNeedH: 0, document: { activeElement: null } };
  const px = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  new Function("px", `with (px) { ${obj("mcGfx")}\n${fn("renderMarketCtx")}\nrenderMarketCtx(); }`)(px);
  return html;
};
// HTML → 타일(①②) 둘: 칩 글자·경고 · 줄마다 [이름, 값, 색 갈래, 게이지 도형]
const parse = (html) => {
  const s = html.indexOf('<div class="mc-strip mc-strip-wall">');
  if (s < 0) return { note: (/<div class="mc-note">([^<]*)</.exec(html) || [])[1] || null };
  const tiles = html.slice(s).split('<div class="mc-sec').slice(1, 3).map((t) => {
    const chip = /<span class="mc-chip( warn)?">([^<]*)<\/span>/.exec(t);
    const rows = t.split('<div class="mc-gk').slice(1).map((g) => {
      const name = /<span>([^<]*)<\/span>/.exec(g)[1], b = /<b class="([^"]*)">([^<]*)<\/b>/.exec(g);
      const svg = /<svg[^>]*width="([\d.]+)" height="(\d+)"[^>]*>(.*?)<\/svg>/.exec(g);
      const shapes = [...svg[3].matchAll(/<(rect|circle|line|text)([^>]*?)\/?>(?:([^<]*)<\/text>)?/g)].map((m) => {
        const a = Object.fromEntries([...m[2].matchAll(/([\w-]+)="([^"]*)"/g)].map((x) => [x[1], x[2]]));
        return { tag: m[1], a, text: m[3] ?? null };
      });
      return { name, val: b ? b[2] : null, cls: b ? b[1] : null, w: +svg[1], h: +svg[2], shapes };
    });
    return { chip: chip ? chip[2] : null, warn: !!(chip && chip[1]), rows };
  });
  return { tiles };
};
const res = {};
for (const W of [404, 300]) res["w" + W] = inputs.map((d) => parse(run(d, W)));
res.wait = [parse(run(null, 404)), parse(run({ asset: "eth", available: false, error: "HTTP 502" }, 404))];
fs.writeFileSync(here + "fixture/mc_js.json", JSON.stringify(res) + "\n");
for (const [k, v] of Object.entries(res)) console.log(k, JSON.stringify(v).length, v.map((x) => (x.tiles ? x.tiles.map((t) => t.rows.length).join("+") : x.note)).join(" | "));
