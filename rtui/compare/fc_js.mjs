// ⑤ 예측 줄 전체 정답지: 지금 화면의 실제 함수(app.js fcRowHtml 과 그 안에서 부르는 fcTrendCell · f1Html · prHit · srLeanCell · rrCells · predXsecHtml …)를 가짜 전역에서 돌린 HTML → fixture/fc_html.json.
// 그 HTML 을 compare/fc_layout.py 가 원본 styles.css 로 헤드리스 크로미움에 띄워 글자·상자·선 자리·색을 잰다 → fixture/fc_js.json. rtui/src/fc.rs 시험 js_parity 가 같은 입력(fixture/fc_in.json)으로 맞댄다.
// 숫자는 전부 지어낸 값. 실행: node rtui/compare/fc_js.mjs   (바깥 접속 없음) · APP_JS=<app.js 사본> OUT=<html json> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src = fs.readFileSync(process.env.APP_JS || here + "../../dashboard/live/app.js", "utf8");
const fn = (name) => { const i = src.indexOf(`\nfunction ${name}(`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n}\n", i) + 3); };
const line = (head) => { const i = src.indexOf("\n" + head); if (i < 0) throw head; return src.slice(i, src.indexOf("\n", i + 1)); };
const block = (head, end) => { const i = src.indexOf("\n" + head); if (i < 0) throw head; return src.slice(i, src.indexOf(end, i) + end.length); };
const body = [line("const DIR4H_ASSETS = "), block("const DIR4H_NOTE = {", "] };\n"), line("const DIRSHORT_ASSETS = "), block("const DIRSHORT_NOTE = {", "] };\n"),
  line("const dirShortNow = "), line("const dir4hNow = "), line("const TREND_NOTE = "), line("const STOCK_ASSETS = "), line("const SR_LEAN_POLL_MS = "), line("const srLeanNowS = "),
  line("const helpAttrs = "), line("const mcTipBtn = "),
  ...["prHit", "predXsecHtml", "riskRadarNow", "rrCells", "fcTrendCell", "f1Html", "srLeanRead", "srLeanCell", "fcRowHtml"].map(fn)].join("\n");
const N = 1795000000000;   // 지어낸 지금(ms)
const H = 3600e3;
const trend = (sig, ups, f1) => ({ ok: true, signal: sig, ups, age_days: 6, votes: [7, 14, 28, 56, 90].map((L, i) => ({ L, up: i < ups, ref_now: 3000 + i * 111.4 })), ...(f1 ? { f1 } : {}) });
const f1 = (ss) => ({ signal: Math.sign(ss.reduce((a, b) => a + b, 0)), ups: ss.filter((s) => s > 0).length, downs: ss.filter((s) => s < 0).length, votes: [7, 14, 21, 28].map((L, i) => ({ L, s: ss[i], ref_now: 2900 + i * 50 })) });
const d4 = (o) => ({ ok: true, asset: "eth", decided_at_ms: N - 200e3, signal: 1, pred_bp: 31.4, lo: -22.6, hi: 24.1, model_month: "2026-11", train_window: ["2025-11-01T00", "2026-10-31T00"], nan_feats: [],
  hist: Array.from({ length: 16 }, (_, i) => [N - 200e3 - (15 - i) * 300e3, -20 + i * 3.3]), ...o });
const ds = (o) => ({ ok: true, asset: "eth", decided_at_ms: N - 120e3, model_month: "2026-11", book: "ok", nan_feats: [], thr: { "15m": [-0.31, 0.42], "1h": [-0.5, 0.61] },
  pred: { "15m": { score: 0.51, pct: 92.35, signal: 1 }, "1h": { score: -0.2, pct: 44.6, signal: 0 } }, hist: Array.from({ length: 20 }, (_, i) => [N - (20 - i) * 300e3, 10 + i * 4.1, 90 - i * 3.3]), ...o });
const rec = (bits, o) => ({ n: bits.length, hit: bits.filter(Boolean).length, recent: bits.map((b, i) => [N - 9e6 + i * 300e3, b]), ...o });
const B = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0];
const xsec = (o) => ({ latest: { rel_bp: { BTC: 12.4, ETH: -8.6, SOL: 41.2, XRP: -55.5, DOGE: 3.2, BNB: -0.4, ADA: 19.6, AVAX: -27.1, LINK: 0.5 } }, n: 70, ic: 0.046, ...o });
const rb = (o) => ({ p_trend: 0.78, confidence: 0.56, decided_at_ms: N - 30 * 60e3, support: 2950.123, resistance: 3101.77, ...o });
const rr = (o) => ({ valid: true, price: 3012.4, sigma15_pct: 0.21, band80_pct: 0.34, band95_pct: 0.52, baseline7d_sigma15_pct: 0.18, alert: { active: false }, state60: "보통", ratio60: 1.12, sigma60_pct: 0.44, ...o });
const sr = (o) => ({ ok: true, now: N / 1000, stale: false, warm_until: null, pct: { top: 56, bot: 55 }, late: 0, mid48: 7, rec: rec([1, 0, 1, 1]),
  ev: [{ a: N / 1000 - 60, d: N / 1000 - 60, side: 1, kind: "touch", look: 60, L: 3012.5, P: 3011.9, grp: "top", y: null }], ...o });
const orv = (o) => ({ ok: true, date: "2026-11-21", entry_ms: N - H, exit_due_ms: N + 3 * H, open_pos: -1, f30_bp: 41.2, sigma30_bp: 60.1, ...o });
const S = (o) => ({ asset: "eth", now: N, price: null, trend: null, dir4h: null, openRev: null, dirShort: null, pred: null, rb: null, rr: null, srl: null, ...o });
// 갈래: ETH 다 있음(종합 3/3 · 7일 추세 · 4h 신호 · 레벨 5분 진행 중 · 횡보·추세 추세 · 앞 1시간 게이지 · 상대 강도 9코인) · 전부 없음 ·
//   SOL 하락 추세(7일 추세 없음) · 4h 숏 · 애매 · 급변 경고 · 비율 없음 · BTC(4h 없음 · 판정 지연 · 격함 · 상대 강도 기록 쌓는 중 · 막대 1px) ·
//   XRP 4h 지연(빈 점 없음) · 횡보 · 조용 · 7일 추세 중립(z 칩) · 위험 레이더 묵음 · SOXL 주식 세션 · ETH 종합 중립(09시 판정 전 · 4h 지연) · 모두 ok 아님
const inputs = [
  S({ price: 3020.55, trend: trend(0.6, 4, f1([1, 1, 1, -1])), dir4h: d4({}), openRev: orv({ open_pos: 1 }), dirShort: ds({}),
      pred: { ok: true, coins: { eth: { "15m": rec(B), "1h": rec(B.slice(4)), "4h": rec([1, 0, 1, 1, 1, 0, 1]), band: { n: 30, hit: 21, daily: [["11-20", 4, 3], ["11-21", 6, 2], ["11-22", 10, 5]] } } }, xsec: xsec({}) },
      rb: { ok: true, coins: { eth: rb({}) } }, rr: { ok: true, stale: false, coins: { eth: rr({}) } }, srl: sr({}) }),
  S({}),
  S({ asset: "sol", price: 181.234, trend: trend(-0.6, 1), dir4h: d4({ asset: "sol", signal: -1, pred_bp: -30.2 }), dirShort: ds({ asset: "sol" }),
      pred: { ok: true, coins: { sol: { "4h": { n: 0, hit: null }, band: rec([0, 0, 1, 0, 0]) } }, xsec: xsec({ n: 0 }) },
      rb: { ok: true, coins: { sol: rb({ p_trend: 0.55, confidence: 0.1 }) } }, rr: { ok: true, coins: { sol: rr({ price: 180.5, alert: { active: true, minutes_ago: 4 }, ratio60: null, baseline7d_sigma15_pct: null }) } } }),
  S({ asset: "btc", price: 91234.5, trend: trend(0.2, 3, f1([1, -1, 1, 1])), dir4h: { ok: false, reason: "모델 없음" },
      pred: { ok: true, coins: { btc: { band: rec([1, 1, 1, 0, 1, 1]) } }, xsec: { latest: { rel_bp: { BTC: 0.2, ETH: -0.3 } }, n: 0 } },
      rb: { ok: true, coins: { btc: rb({ decided_at_ms: N - 91 * 60e3 }) } }, rr: { ok: true, coins: { btc: rr({ price: 91000, state60: "격함", ratio60: 2.9, sigma60_pct: 0.88, band80_pct: 0.41, band95_pct: 0.63 }) } } }),
  S({ asset: "xrp", price: 0.61234, trend: trend(-0.2, 2, f1([1, -1, 0, 1])), dir4h: d4({ asset: "xrp", decided_at_ms: N - 15 * 60e3 - 1, signal: 0, pred_bp: -5.5, hist: [] }),
      pred: { ok: true, coins: { xrp: { "4h": rec([1, 1, 0]), band: rec([0, 1]) } } }, rb: { ok: true, coins: { xrp: rb({ p_trend: 0.2, confidence: 0.6 }) } },
      rr: { ok: true, stale: true, coins: { xrp: rr({}) } } }),
  S({ asset: "soxl", price: 41.2, trend: trend(0.6, 5), rr: { ok: true, coins: { soxl: rr({ price: 41.0, session: "정규장", sigma60_pct: 1.21, band80_pct: 1.4, band95_pct: 2.2 }) } } }),
  S({ trend: trend(0.2, 3), dir4h: d4({ decided_at_ms: N - 16 * 60e3, pred_bp: 2.1, signal: 0 }), openRev: orv({ entry_ms: N + H }), dirShort: ds({ book: "warm", pred: { "15m": { score: -0.1, pct: 30, signal: 0 }, "1h": { score: 0.2, pct: 75, signal: 0 } } }),
      pred: { ok: true, coins: { eth: {} }, xsec: xsec({ ic: -0.031 }) }, rb: { ok: true, coins: {} }, rr: { ok: true, coins: { eth: rr({ ratio60: 0.5, state60: "조용", baseline7d_sigma15_pct: 0.3 }) } }, srl: sr({ stale: true }) }),
  S({ trend: { ok: false, reason: "fetch_failed" }, dir4h: { ok: false }, openRev: { ok: false }, dirShort: { ok: false, reason: "x" }, pred: { ok: false }, rb: { ok: false }, rr: { ok: false, reason: "준비 중" }, srl: { ok: false } }),
];
fs.writeFileSync(here + "fixture/fc_in.json", JSON.stringify(inputs) + "\n");
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const run = (s) => {
  const now = s.now;
  class D extends Date { constructor(...a) { super(...(a.length ? a : [now])); } static now() { return now; } }
  const env = { latestTrend: s.trend, latestDirSig: null, activeSnapshotAsset: s.asset, latestVol4h: null, latestDir4h: s.dir4h, latestOpenRev: s.openRev, latestDirShort: s.dirShort,
    latestRegimeBand: s.rb, latestRiskRadar: s.rr, latestPredRecord: s.pred, latestSrLean: s.srl, srLeanSkew: 0, latestLivePriceByAsset: s.price ? { [s.asset]: s.price } : {}, helpKey: null, escapeHtml: esc, Date: D };
  const px = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  return new Function("px", `with (px) { ${body}\nreturn fcRowHtml(null); }`)(px);
};
const res = inputs.map(run);
fs.writeFileSync(process.env.OUT || here + "fixture/fc_html.json", JSON.stringify(res) + "\n");
for (const h of res) console.log(h.replace(/ title="[^"]*"/g, "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").slice(0, 400));
