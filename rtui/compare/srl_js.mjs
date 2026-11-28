// «레벨 5분» 칸 정답지: 지금 화면의 실제 함수(app.js srLeanCell · srLeanRead · fcRowHtml 안의 cell)를 가짜 전역에서 돌린 HTML → fixture/srl_js.json.
// rtui/src/srl.rs 시험 js_parity 가 같은 입력(fixture/srl_in.json = [/api/sr-lean 응답, 서버 시계 nowS])으로 글자·색·게이지·적중 막대를 맞댄다. 숫자는 전부 지어낸 값.
// 실행: node rtui/compare/srl_js.mjs   (바깥 접속 없음) · APP_JS=<app.js 사본> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src = fs.readFileSync(process.env.APP_JS || here + "../../dashboard/live/app.js", "utf8");
const fn = (name) => { const i = src.indexOf(`\nfunction ${name}(`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n}\n", i) + 3); };
const line = (name) => { const m = new RegExp(`\\nconst ${name} = [^\\n]*`).exec(src); if (!m) throw name; return m[0]; };
// fcRowHtml 안의 칸 그리기(const cell = ...) 를 그대로 떼어 온다
const cellSrc = (() => { const i = src.indexOf("\n  const cell = (hz, c, main, sub, tip, gauge"); if (i < 0) throw "cell"; return src.slice(i, src.indexOf("\n  };\n", i) + 5); })();
const N = 1795000000;   // 지어낸 서버 시각(초)
const ev = (o) => ({ a: N - 100, d: N - 100, side: 1, kind: "touch", look: 60, L: 3012.5, P: 3011.9, grp: "top", y: null, hit_s: null, ...o });
const rec = (bits) => ({ n: bits.length, hit: bits.filter(Boolean).length, recent: bits.map((b, i) => [(N - 9000 + i * 60) * 1000, b]) });
const base = { ok: true, now: N, stale: false, warm_until: null, pct: { top: 56, bot: 55 }, late: 0, mid48: 7 };
// 갈래: 응답 없음 · ok 아님(이유 있음/없음) · 멈춤 · 준비 중 · 대기(진행 중 없음 · 결과 못 냄 경계 305초) · 이어짐 위/아래 · 되돌림 위/아래 · 가장 최근 진행 중 고르기 · 적중률 55%/45%/사이 · 기록 없음/n=0
const inputs = [
  [null, N],
  [{ ok: false, reason: "워커 기록 없음" }, N],
  [{ ok: false }, N],
  [{ ...base, stale: true, ev: [ev({})], rec: { n: 0, hit: 0, recent: [] } }, N],
  [{ ...base, warm_until: N + 1500, ev: [ev({})], rec: rec([1, 1, 0]) }, N],
  [{ ...base, ev: [ev({ d: N - 305 }), ev({ d: N - 400, y: 1 })], rec: rec([1, 1, 0, 1, 1, 0, 1, 1, 0]) }, N],
  [{ ...base, ev: [ev({ d: N - 304.5 })], rec: rec([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]) }, N],
  [{ ...base, ev: [ev({ side: -1, look: 240 })], rec: rec([1, 0]) }, N],
  [{ ...base, ev: [ev({ grp: "bot", look: 1440 })] }, N],
  [{ ...base, ev: [ev({ grp: "bot", side: -1 })], rec: { n: 0, hit: 0, recent: [] } }, N],
  [{ ...base, ev: [ev({ grp: "bot", d: N - 50 }), ev({ grp: "top", side: -1, d: N - 20 }), ev({ d: N - 10, y: 0 })], rec: rec(Array.from({ length: 30 }, (_, i) => i % 3 !== 0)) }, N],
  [{ ...base, pct: { top: 61, bot: 48 }, ev: [ev({ d: N - 1 })], rec: rec([1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]) }, N],
  [{ ...base, ev: [] }, N],
  [{ ...base, ev: [ev({ grp: "bot", side: 1, d: N - 2 })], rec: rec([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]) }, N],
];
fs.writeFileSync(here + "fixture/srl_in.json", JSON.stringify(inputs) + "\n");
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const run = ([r, nowS]) => {
  const env = { latestSrLean: r, srLeanNowS: () => nowS, latestPredRecord: null, escapeHtml: esc };
  const p = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  return new Function("px", `with (px) { ${line("SR_LEAN_POLL_MS")}\n${fn("prHit")}\n${fn("srLeanRead")}\n${fn("srLeanCell")}\n${cellSrc}\nreturn srLeanCell(cell); }`)(p);
};
// HTML → 칸: 색 갈래 · 이름 · 판정 글자 · 게이지 점(없으면 null) · 오른쪽(적중 막대 [맞음..] · 적중률 글자·색 · 분모 / 쌓는 중)
const parse = (h) => {
  const c = /^<div class="fc-c ([^"]*)"/.exec(h)[1], hz = /<span class="hz">([^<]*)</.exec(h)[1];
  const g = /<u style="left:([\d.]+)%"><\/u>/.exec(h), main = /<b class="gl[^"]*"[^>]*>([^<]*)<\/b>/.exec(h)[1];
  const zones = [...h.matchAll(/<i class="z (dn|up)" style="(width|left):([\d.]+)%"><\/i>/g)].map((m) => [m[1], m[2], +m[3]]);
  const r = /<div class="fz-r">(.*)<\/div><\/div>$/.exec(h)[1];
  const bars = r.includes('class="pr-sp"') ? [...r.matchAll(/<i class="(u|d)" style="height:(\d+)px"><\/i>/g)].map((m) => [m[1] === "u", +m[2]]) : null;
  const b = /<b class="([^"]*)">([^<]*)<\/b> ([^<]*)<\/span>/.exec(r), n = /<span class="pr-n">([^<]*)<\/span>/.exec(r);
  return { c, hz, main, gauge: g ? +g[1] : null, zones, bars, pct: b ? b[2] : null, pcls: b ? b[1] : null, frac: b ? b[3] : null, note: b ? null : n[1] };
};
const res = inputs.map((x) => parse(run(x)));
fs.writeFileSync(here + "fixture/srl_js.json", JSON.stringify(res) + "\n");
for (const r of res) console.log(r.c, r.main, r.gauge, r.bars ? `${r.pct}(${r.pcls}) ${r.frac} 막대 ${r.bars.length}` : r.note);
