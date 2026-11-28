// 계좌 칩(맨 위 전광판) 정답지: 지금 화면의 실제 함수(app.js renderPnlTicker · tickerPnl · entryUsdt · armedBasis · acctCoin)를 가짜 DOM 에서 돌린 칸 글자·색 → fixture/acct_js.json.
// rtui/src/acct.rs 시험 js_parity 가 같은 입력(fixture/acct_in.json)으로 맞댄다. 숫자는 전부 지어낸 값(실계좌 아님).
// 실행: node rtui/compare/acct_js.mjs   (바깥 접속 없음) · APP_JS=<app.js 사본> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src = fs.readFileSync(process.env.APP_JS || here + "../../dashboard/live/app.js", "utf8");
const fn = (name) => { const i = src.indexOf(`\nfunction ${name}(`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n}\n", i) + 3); };
const line = (name) => { const m = new RegExp(`\\nconst ${name} = [^\\n]*`).exec(src); if (!m) throw name; return m[0]; };
// 입력 = [계좌 응답, 코인별 1초 가격] -- 갈래: USDC 무장 손절 basis · 익절 basis · 서버 basis · 표시가 ÷ 차트가 · 가격 없음(미실현) · 수량 0 · 소문자 side · 0 근처 음수 · 문자열 숫자
const inputs = [
  [{ ok: true, positions: [
      { symbol: "ETHUSDC", side: "LONG", qty: "0.5", entry_price: "3000", mark_price: "3010.5", unrealized_pnl: "1" },
      { symbol: "SOLUSDT", side: "SHORT", qty: -10, entry_price: 150, unrealized_pnl: 2 },
      { symbol: "XRPUSDC", side: "SHORT", qty: 100, entry_price: 0.6, mark_price: 0.61, unrealized_pnl: -3.456 },
      { symbol: "BTCUSDC", side: "LONG", qty: 0, entry_price: 90000 },
      { symbol: "SOXLUSDT", side: "LONG", qty: 2, entry_price: 40, unrealized_pnl: 5 },
      { symbol: "MUUSDC", side: "LONG", qty: 1, entry_price: 100, unrealized_pnl: -0.004 },
    ], bracket_armed: { "ETHUSDC:LONG": { sl_price: 2900.3, sl_level: 2900 } } },
   { eth: 3012.34, sol: 151.2, soxl: 40 }],
  [{ ok: true, positions: [
      { symbol: "ETHUSDC", side: "short", qty: "-0.25", entry_price: "3100", mark_price: 3050 },
      { symbol: "BTCUSDC", side: "SHORT", qty: 0.01, entry_price: 91000, mark_price: 90500 },
      { symbol: "SOLUSDC", side: "LONG", qty: 3, entry_price: 140, mark_price: 141 },
      { symbol: "XRPUSDC", side: "LONG", qty: 50, entry_price: 0.6 },
    ], bracket_armed: { "ETHUSDC:SHORT": { sl_price: 0, tp_price: 2995.5, tp_level: 3000 }, "BTCUSDC:SHORT": { basis: "1.0004" }, "XRPUSDC:LONG": { sl_price: 1, sl_level: 0 } } },
   { eth: 3040, btc: 90400.5, sol: 141.37, xrp: 0.6123 }],
  [{ ok: false, error: "x", positions: [{ symbol: "ETHUSDT", side: "LONG", qty: 1, entry_price: 1 }] }, { eth: 2 }],
  [{ ok: true, positions: [] }, {}],
  [{ ok: true, positions: [{ symbol: "ETHUSDT", side: "LONG", qty: 1, entry_price: 3000, unrealized_pnl: "abc" }] }, { eth: 0 }],
];
fs.writeFileSync(here + "fixture/acct_in.json", JSON.stringify(inputs) + "\n");
const run = ([acct, px]) => {
  let html = "", hidden = true;
  const bs = {};   // data-s → 두 벌 b(숫자가 흐르는 두 사본)
  const box = { dataset: {}, style: {}, set hidden(v) { hidden = v; }, get hidden() { return hidden; },
    set innerHTML(v) { html = v; }, parentElement: { getBoundingClientRect: () => ({ left: 0 }) },
    querySelectorAll: (sel) => { const k = /data-s="([^"]*)"/.exec(sel)[1]; return (bs[k] ||= [{}, {}]); } };
  const mid = {}, title = { parentElement: mid, getBoundingClientRect: () => ({ left: 1000 }) }, slot = { getBoundingClientRect: () => ({ right: 0 }) };
  const env = { latestBinanceAccount: acct.ok ? acct : null, latestLivePriceByAsset: px, posChipsSet: () => {},
    el: (id) => ({ pnlTicker: box, topLeftSlot: slot, fpCoinName: title, topCenterSlot: mid })[id] || null };
  const p = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  new Function("px", `with (px) { ${line("acctCoin")}\n${fn("armedBasis")}\n${fn("entryUsdt")}\n${fn("tickerPnl")}\n${fn("renderPnlTicker")}\nrenderPnlTicker(); }`)(p);
  if (hidden) return { hidden: true, chips: [] };
  const chips = [...html.matchAll(/data-s="([^"]*)"><i>([^<]*)<\/i>/g)].map((m) => {
    const [a, b] = bs[m[1]];
    if (a.textContent !== b.textContent || a.className !== b.className) throw "두 사본이 다르다";
    return { label: m[2], val: a.textContent, cls: a.className };
  });
  return { hidden: false, width: box.style.width, chips };
};
const res = inputs.map(run);
fs.writeFileSync(here + "fixture/acct_js.json", JSON.stringify(res) + "\n");
for (const r of res) console.log(r.hidden ? "숨김" : r.chips.map((c) => `${c.label} ${c.val} ${c.cls}`).join(" | "));
