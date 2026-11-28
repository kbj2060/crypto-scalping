// ⑤ «15분 예측»·«1시간 예측» 두 줄 정답지: 지금 화면의 실제 함수(app.js fcRowHtml 안의 sCell · cell · prHit · dirShortNow)를 가짜 전역에서 돌린 HTML → fixture/dirs_js.json.
// rtui/src/dirs.rs 시험 js_parity 가 같은 입력(fixture/dirs_in.json = [/api/dir-short 응답, /api/pred-record 응답, 지금 코인, Date.now()])으로 글자·색·게이지(점·빈 점·띠)·적중 막대를 맞댄다. 숫자는 전부 지어낸 값.
// 실행: node rtui/compare/dirs_js.mjs   (바깥 접속 없음) · APP_JS=<app.js 사본> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src = fs.readFileSync(process.env.APP_JS || here + "../../dashboard/live/app.js", "utf8");
const fn = (name) => { const i = src.indexOf(`\nfunction ${name}(`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n}\n", i) + 3); };
const line = (head) => { const i = src.indexOf("\n" + head); if (i < 0) throw head; return src.slice(i, src.indexOf("\n", i + 1)); };
const block = (head, end) => { const i = src.indexOf("\n" + head); if (i < 0) throw head; return src.slice(i, src.indexOf(end, i) + end.length); };
const body = [line("const DIR4H_ASSETS = "), line("const DIRSHORT_ASSETS = "), block("const DIRSHORT_NOTE = {", "};\n"), line("const dirShortNow = "), fn("prHit"),
  line("  const cls = (s) =>"), line("  const kst = (ms) =>"), block("  const cell = (hz, c, main, sub, tip, gauge", "\n  };\n"),
  line("  const onS = "), line("  const gs = dirShortNow()"), block("  const sCell = (nm, ko, back, col) => {", "\n  };\n")].join("\n");
// 부르는 자리(fcRowHtml 의 sCell("15m", …) · sCell("1h", …))도 원본에서 떼어 온다
const calls = ["15m", "1h"].map((nm) => { const m = new RegExp(`sCell\\("${nm}"[^)]*\\)`).exec(src); if (!m) throw nm; return m[0]; }).join(", ");
const N = 1795000000000;   // 지어낸 지금(ms)
const hist = (n, f) => Array.from({ length: n }, (_, i) => [N - (n - i) * 300e3, ...f(i)]);
const ds = (o) => ({ ok: true, asset: "eth", decided_at_ms: N - 120e3, model_month: "2026-11", book: "ok", nan_feats: [], thr: { "15m": [-0.31, 0.42], "1h": [-0.5, 0.61] },
  pred: { "15m": { score: 0.51, pct: 92.35, signal: 1 }, "1h": { score: -0.6, pct: 12.25, signal: -1 } }, hist: hist(20, (i) => [10 + i * 4.1, 90 - i * 3.3]), ...o });
const rec = (bits) => ({ n: bits.length, hit: bits.filter(Boolean).length, recent: bits.map((b, i) => [N - 9e6 + i * 300e3, b]) });
const pr = (eth, o = {}) => ({ ok: true, coins: { eth, ...o } });
const B = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0];
// 갈래: 응답 없음 · ok 아님 · 다른 코인 응답 · 15분 지연 경계(900초 = 아님 · 넘으면 지연) · 분위 반올림 동점(12.25 → 12.3 · 87.75 → 87.8) · 0·100 밖 자르기 ·
//   보류(신호 0) · pred 한쪽 없음 · 이력 짧음(빈 점 없음) · 이력 값 null · 적중: 기록 없음(칸 없음) · 쌓는 중 · recent 막대(참/거짓·0/1) · 일별 막대(>·<·= .5) · n=0 이면 막대 + 쌓는 중 · 55%·45% 경계
const inputs = [
  [null, null, "eth", N],
  [{ ok: false, reason: "모델 파일 없음" }, pr({ "15m": rec(B), "1h": rec(B.slice(3)) }), "eth", N],
  [ds({}), pr({ "15m": rec(B), "1h": { n: 0, hit: null } }), "eth", N],
  [ds({ asset: "btc" }), pr({ "15m": rec(B) }), "eth", N],
  [ds({ decided_at_ms: N - 900e3 }), pr({ "15m": { n: 20, hit: 11, daily: [["11-20", 4, 3], ["11-21", 6, 2], ["11-22", 10, 5], ["11-23", 2, 2]] }, "1h": rec([1, 0]) }), "eth", N],
  [ds({ decided_at_ms: N - 900e3 - 1 }), pr({ "15m": rec(B) }), "eth", N],
  [ds({ pred: { "15m": { score: 0.3, pct: 87.75, signal: 1 }, "1h": { score: 0.01, pct: 50, signal: 0 } } }), pr({ "15m": { n: 20, hit: 11, recent: [] , daily: [] }, "1h": { n: 20, hit: 9, recent: [[N, 0], [N, 1], [N, 1]] } }), "eth", N],
  [ds({ asset: "sol", pred: { "15m": { pct: 104, signal: 1 }, "1h": { pct: -3, signal: -1 } }, hist: hist(5, (i) => [i * 20, 100 - i * 20]) }), pr({}, { sol: { "15m": rec([1, 1, 1, 0]) } }), "sol", N],
  [ds({ asset: "xrp", pred: { "15m": null, "1h": { pct: 64.4, signal: 0 } }, hist: hist(14, (i) => [i % 2 ? null : 30, 70]) }), pr({}, { xrp: { "15m": { hit: 3, n: 0, recent: [[N, 1], [N, 0], [N, 1]] }, "1h": { n: 7, hit: 4 } } }), "xrp", N],
  [ds({ pred: { "15m": { score: -0.31, pct: 20, signal: -1 }, "1h": { score: 0.61, pct: 80, signal: 1 } } }), { ok: false }, "eth", N],
  [ds({ pred: { "15m": { score: -1.2, pct: 0.05, signal: -1 }, "1h": { score: 1.4, pct: 99.95, signal: 1 } }, hist: hist(13, (i) => [i, 100 - i]) }), pr({ "15m": rec(Array.from({ length: 60 }, (_, i) => i % 3 !== 1)), "1h": rec(B.concat(B, B)) }), "eth", N],
];
fs.writeFileSync(here + "fixture/dirs_in.json", JSON.stringify(inputs) + "\n");
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const run = ([d, p, asset, now]) => {
  class D extends Date { constructor(...a) { super(...(a.length ? a : [now])); } static now() { return now; } }
  const env = { latestDirShort: d, latestPredRecord: p, activeSnapshotAsset: asset, eth: asset === "eth", escapeHtml: esc, Date: D };
  const px = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  return new Function("px", `with (px) { ${body}\nreturn [${calls}]; }`)(px);
};
// HTML → 칸: 색 갈래 · 이름 · 판정 글자 · 게이지(점·빈 점·띠) · 오른쪽(없음 / 적중 막대 [갈래, 높이] · 적중률 글자·색 · 분모 / 글)
const parse = (h) => {
  const c = /^<div class="fc-c ([^"]*)"/.exec(h)[1], hz = /<span class="hz">([^<]*)</.exec(h)[1];
  const g = /<u style="left:([^"]*)%"><\/u>/.exec(h), ago = /<u class="ago" style="left:([^"]*)%"><\/u>/.exec(h), main = /<b class="gl[^"]*"[^>]*>([^<]*)<\/b>/.exec(h)[1];
  const zones = [...h.matchAll(/<i class="z (dn|up)" style="(width|left):([\d.]+)%"><\/i>/g)].map((m) => [m[1], m[2], +m[3]]);
  const r = /<div class="fz-r">(.*)<\/div><\/div>$/.exec(h);
  if (!r) return { c, hz, main, gauge: g ? g[1] : null, ago: ago ? ago[1] : null, zones, right: null };
  const bars = r[1].includes('class="pr-sp"') ? [...r[1].matchAll(/<i class="(u|d|)" style="height:([^"]*)px"[^>]*><\/i>/g)].map((m) => [m[1], m[2]]) : null;
  const b = /<span class="pr-n"><b class="([^"]*)">([^<]*)<\/b> ([^<]*)<\/span>/.exec(r[1]), n = /<span class="pr-n">([^<]*)<\/span>/.exec(r[1]);
  return { c, hz, main, gauge: g ? g[1] : null, ago: ago ? ago[1] : null, zones, right: { bars, pct: b ? b[2] : null, pcls: b ? b[1] : null, frac: b ? b[3] : null, note: b ? null : n[1] } };
};
const res = inputs.map((x) => run(x).map(parse));
fs.writeFileSync(here + "fixture/dirs_js.json", JSON.stringify(res) + "\n");
for (const rr of res) console.log(rr.map((r) => `${r.hz} ${r.c} ${r.main} g=${r.gauge} o=${r.ago} ${r.right ? (r.right.bars ? `막대${r.right.bars.length} ` : "") + (r.right.pct ? `${r.right.pct}(${r.right.pcls}) ${r.right.frac}` : r.right.note) : "칸 없음"}`).join(" | "));
