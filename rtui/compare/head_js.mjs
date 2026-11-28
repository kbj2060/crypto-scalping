// 풋프린트 머리 작은 칸 정답지(?card=head): 지금 화면의 실제 함수(buildSessionHtml · renderMacroEventAlert · macroSpikeChips + renderMarketCtx 의 칩 줄 · renderFpPos · posMarginShare ·
// snapshotAccountPosition · execSymbolFor · entryUsdt · armedBasis · lineFmt · ASSET_CONFIG)를 가짜 전역에서 돌린 HTML → fixture/head_html.json.
// 코인 이름(renderSnapshotAssetTabs)과 맨 위 배치(placeTopLeft)는 compare/head_layout.py 가 진짜 index.html 위에서 원본 그대로 돌린다(page 글).
// 🔴계좌·가격·급변 숫자는 전부 지어낸 값 -- 실제 응답을 받거나 저장하지 않는다. 실행: TZ=Asia/Seoul node rtui/compare/head_js.mjs · APP_JS=<사본> OUT=<html json> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src = fs.readFileSync(process.env.APP_JS || here + "../../dashboard/live/app.js", "utf8");
const fn = (name) => { const i = src.indexOf(`\nfunction ${name}(`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n}\n", i) + 3); };
const line = (head) => { const i = src.indexOf("\n" + head); if (i < 0) throw head; return src.slice(i, src.indexOf("\n", i + 1)); };
const block = (head) => { const i = src.indexOf("\n" + head); if (i < 0) throw head; return src.slice(i, src.indexOf("\n};\n", i) + 4); };
const chipLine = (() => { const i = src.indexOf('setH("macroSpikeChips", macroSpikeChips('); if (i < 0) throw "chipLine"; return src.slice(i, src.indexOf("\n", i)); })();
const body = [fn("escapeHtml"), block("const ASSET_CONFIG = {"), line("const lineDig = "), line("const lineFmt = "), fn("fmtUsd"), fn("armedBasis"), fn("execSymbolFor"), fn("snapshotAccountPosition"),
  fn("entryUsdt"), fn("posMarginShare"), fn("renderFpPos"), fn("buildSessionHtml"), fn("renderMacroEventAlert"), line("const SPIKE_SHOW_MIN = "), fn("macroSpikeChips")].join("\n");
// 원본 함수 그대로 브라우저에서 돌릴 것(코인 이름 · 맨 위 자리 옮기기)
const page = [line("const PAGE_TABS_WIDE = "), line("const TOP_TITLE_WIDE = "), fn("assetPins"), fn("renderSnapshotAssetTabs"), fn("placeTopLeft")].join("\n");
if (process.env.TZ !== "Asia/Seoul") throw new Error("TZ=Asia/Seoul 로 돌린다");

const NOW = 1795000000;   // 수 11-18 20:06 KST
const sess = (a, e, u) => ({ session_asia: a, session_europe: e, session_us: u });
const alert = (on) => ({ active: on ? [{ title_ko: "미국 CPI", minutes_from_event: -12.4 }] : [] });
const ev = (ago_min, z, r) => [NOW - 300 - ago_min * 60, z, r];   // [봉 시작, 평소의 배, 수익률 %] -- 끝난 지 ago_min 분
const spk = (o) => ({ ok: true, z: 4, now_sec: NOW + 2, symbols: o });
const pos = (symbol, side, qty, entry, x = {}) => ({ symbol, side, qty, entry_price: entry, mark_price: x.mark ?? entry, leverage: x.lev ?? 10, notional: x.notional ?? qty * entry, unrealized_pnl: 0 });
const acct = (ps, x = {}) => ({ ok: true, positions: ps, balances: x.bal ?? { USDT: { margin: 5000, wallet: 4800, unrealized: 12 }, USDC: { wallet: 3000, unrealized: -40 } }, multi_assets_margin: !!x.multi, balance: x.balance,
  exec_symbol: x.exec ?? "", exec_symbols: x.execs ?? {}, bracket_armed: x.armed ?? {} });
// 갈래: 아무것도 없음 · 다 있음(롱 이익 · 증거금 7.4% · 5·10년 같은 쪽 한 칩 + 나스닥) · USDC 숏 손실(basis 환산 · 다중자산 증거금 ≥10% → 정수 · 5·10년 반대쪽 = 두 칩 + 나스닥 + 경보 → 칸 넘침) ·
//   XRP(소수 넷째 · 잔고 없음 = 증거금 칸 없음 · 40분 지난 급변 = 없음) · BTC(쉼표 · 유럽만 · 10년만 · 0분) · 가격 없음 = 숨김(경보만 · 급변 ok 아님) · SOL 수량 0 = 숨김(나스닥 오름 · 39분 경계) · ETH USDC basis 없음(마크/지금가)
const inputs = [
  { asset: "eth", px: {}, sess: {}, alert: alert(false), spk: null, acct: null },
  { asset: "eth", px: { eth: 3125.4 }, sess: sess(1, 0, 1), alert: alert(false), spk: spk({ NQ: { events: [ev(30, -5.1, -0.41), ev(3, 4.6, -0.38)] }, ZF: { events: [ev(9, -4.4, -0.08)] }, ZN: { events: [ev(7, -4.9, -0.12)] } }),
    acct: acct([pos("ETHUSDT", "LONG", 1.2, 3050.25, { lev: 10, notional: 3750 })]) },
  { asset: "eth", px: { eth: 3125.4 }, sess: sess(0, 1, 1), alert: alert(true), spk: spk({ NQ: { events: [ev(12, 4.2, 0.52)] }, ZF: { events: [ev(5, 4.1, 0.06)] }, ZN: { events: [ev(8, -4.3, -0.09)] } }),
    acct: acct([pos("ETHUSDT", "LONG", 0.5, 3000), pos("ETHUSDC", "SHORT", 2.4, 3101.9, { mark: 3124.1, lev: 5, notional: 7497.84 })], { multi: true, balance: { margin: 9800 }, exec: "ETHUSDC", execs: { eth: "ethusdc" },
      armed: { "ETHUSDC:SHORT": { sl_price: 3200.5, sl_level: 3201.1, tp_price: 0, tp_level: 0 } } }) },
  { asset: "xrp", px: { xrp: 2.48731 }, sess: sess(1, 1, 0), alert: alert(false), spk: spk({ NQ: { events: [ev(41, 5, 0.6)] }, ZF: { events: [] } }), acct: acct([pos("XRPUSDT", "SHORT", 1500, 2.4512)], { bal: {} }) },
  { asset: "btc", px: { btc: 98345.6 }, sess: sess(0, 1, 0), alert: alert(false), spk: spk({ ZN: { events: [ev(0, 4.0, 0.11)] } }), acct: acct([pos("BTCUSDT", "LONG", 0.35, 97123.45, { lev: 20 })]) },
  { asset: "eth", px: {}, sess: sess(0, 0, 0), alert: alert(true), spk: { ok: false, symbols: {} }, acct: acct([pos("ETHUSDT", "LONG", 1, 3000)]) },
  { asset: "sol", px: { sol: 151.2 }, sess: sess(1, 0, 0), alert: alert(false), spk: spk({ NQ: { events: [ev(39, 4.8, 0.27)] } }), acct: acct([pos("SOLUSDT", "LONG", 0, 150)]) },
  { asset: "eth", px: { eth: 3125.4 }, sess: sess(0, 0, 1), alert: alert(false), spk: null, acct: acct([pos("ETHUSDC", "LONG", 0.8, 3130.2, { mark: 3127.7, notional: 2502 })], { exec: "ETHUSDC" }) },
];
for (const s of inputs) s.now = NOW;
fs.writeFileSync(here + "fixture/head_in.json", JSON.stringify(inputs) + "\n");
const run = (s) => {
  const now = s.now * 1000;
  class D extends Date { constructor(...a) { super(...(a.length ? a : [now])); } static now() { return now; } }
  const E = { fpPos: { hidden: true, innerHTML: "" }, macroEventAlertBadge: { style: { display: "none" }, title: "" }, macroSpikeChips: { innerHTML: "" } };
  const env = { Date: D, el: (id) => E[id] || null, latestBinanceAccount: s.acct, lastGoodAccount: s.acct, latestLivePriceByAsset: s.px, activeSnapshotAsset: s.asset,
    macroSpk: s.spk, macroSpkSkew: s.spk && Number.isFinite(s.spk.now_sec) ? s.spk.now_sec - now / 1000 : 0, __S: s };
  env.setH = (id, html) => { const t = env.el(id); if (t && t.innerHTML !== html) t.innerHTML = html; };
  const px = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  return new Function("px", `with (px) { ${body}\nrenderFpPos(); renderMacroEventAlert(__S.alert); ${chipLine}\n`
    + `return { sess: buildSessionHtml(__S.sess), pos: el("fpPos").hidden ? null : el("fpPos").innerHTML, badge: el("macroEventAlertBadge").style.display !== "none", spk: el("macroSpikeChips").innerHTML }; }`)(px);
};
const res = inputs.map(run);
fs.writeFileSync(process.env.OUT || here + "fixture/head_html.json", JSON.stringify({ cases: res, page }) + "\n");
for (const r of res) console.log(`경보 ${r.badge} · 급변 ${r.spk.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim() || "-"} · 포지션 ${r.pos ? r.pos.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim() : "숨김"}`);
