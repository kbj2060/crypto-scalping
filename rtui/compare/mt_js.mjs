// Macro Trend 카드 정답지: 지금 화면의 실제 코드(app.js «Macro Trend 카드» 구역 통째 = MT_* 상수 · mtFmt · mtDay · mtFeed · mtTodayCell · mtDayChart · mtOverlay · renderMacroTrend · mtHeadHtml
// + xyzClock · xyzNow · escapeHtml · helpAttrs)를 가짜 전역에서 돌린 HTML → fixture/mt_html.json. 그 HTML 을 compare/mt_layout.py 가 원본 styles.css 로 헤드리스 크로미움에 띄워 잰다.
// 숫자·문장은 전부 지어낸 값. 실행: TZ=Asia/Seoul node rtui/compare/mt_js.mjs   (바깥 접속 없음) · APP_JS=<app.js 사본> OUT=<html json> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src = fs.readFileSync(process.env.APP_JS || here + "../../dashboard/live/app.js", "utf8");
const fn = (name) => { const i = src.indexOf(`\nfunction ${name}(`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n}\n", i) + 3); };
const line = (head) => { const i = src.indexOf("\n" + head); if (i < 0) throw head; return src.slice(i, src.indexOf("\n", i + 1)); };
const span = (a, b) => { const i = src.indexOf(a), j = src.indexOf(b, i); if (i < 0 || j < 0) throw a; return src.slice(i, j); };
// 구역 통째(3370 const MT_POLL_MS ~ renderMacroTrend 끝) -- 안의 async 받기 함수는 정의만 되고 안 불린다 · document.addEventListener 줄은 가짜 document 가 받는다
const body = [fn("escapeHtml"), line("const helpAttrs = "), line("const xyzClock = "), fn("xyzNow"), span("\nconst MT_POLL_MS", '\nel("mtBody")?.addEventListener')].join("\n");
if (process.env.TZ !== "Asia/Seoul") throw new Error("TZ=Asia/Seoul 로 돌린다(hm = 브라우저 지역 시각 · rtui 는 tz_off_min -540)");

let seed = 7;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const r2 = (v) => Math.round(v * 1e4) / 1e4;
const NA = 1795000000;      // 수 11-18 20:06 KST(미 동부 06:06 · 시카고 05:06) -- VIX 산출 중 · CME 열림
const NB = 1795330000;      // 일 11-22 15:46 KST(시카고 00:46 일요일) -- 둘 다 쉼
const FAMS = [["나스닥", "%"], ["S&P500", "%"], ["VIX", "%"], ["STLFSI", ""], ["HY스프레드", "bp"], ["Baa스프레드", "bp"], ["10Y금리", "bp"], ["기대인플레", "bp"], ["기간프리미엄", "bp"], ["연방기금", "bp"],
  ["달러지수", "%"], ["USD/JPY", "%"], ["EUR/USD", "%"], ["원/달러", "%"], ["USD/CNY", "%"], ["순유동성", "%"], ["연준BS", "%"], ["지준", "%"], ["TGA", "%"], ["RRP", "$bn"], ["스테이블코인", "%"], ["브렌트", "%"]];
const STATES = ["count", "count", "zero", "obs", "stale", "old", "weak", "off", "unknown", "count"];
const WHY = { count: "셈 — 연구 r +0.41 · 최근 52주 r +0.33 유효 · +1.2σ", zero: "중립(셈) — 13주 변화가 ETF 이후 분포 +0.3σ (|z| < 0.5)",
  obs: "관찰(안 셈) — 연구 «사용 가능» 기준 미달: ETF 이후 |r| 0.22 < 0.30 · 순열 p 0.08 ≥ 0.05 · 52주 굴림 상관 같은 부호 61% < 75%",
  stale: "변동 없음(안 셈) — 최근 26주 13주 변화의 흔들림이 ETF 이후의 18% (< 25%)", old: "지연(안 셈) — 마지막 완결 주 2026-10-23 (26일 전) · 21일 넘음",
  weak: "약화(안 셈) — 최근 52주 r +0.08: 연구 r +0.37 와 같은 쪽이지만 |r| < 0.15", off: "이탈(안 셈) — 최근 52주 r −0.29 가 연구 r +0.37 와 반대 부호이고 |r| ≥ 0.15",
  unknown: "감시 모름(안 셈) — 최근 52주 상관을 낼 겹치는 주가 26주 미만" };
const series = (n, f) => Array.from({ length: n }, (_, i) => f(i));
const research = (verdict) => ({ verdict, vs_2020_23: "같음", r_p3: r2((rnd() - .5) * 1.4), perm_p: rnd() < .2 ? null : r2(rnd() * .2), r_first_half: r2((rnd() - .5) * 1.2), r_second_half: r2((rnd() - .5) * 1.2),
  r_excl_2024h2: r2(rnd() - .5), roll52_same_sign: r2(.5 + rnd() / 2), r_2020_23: rnd() < .15 ? null : r2((rnd() - .5) * 1.2) });
// 줄 하나: name · kind · 단위 · 상태(자산마다) · 선택적 덮어쓰기
const row = (name, unit, kind, k, o = {}) => {
  const assets = {}, la = {};
  for (const a of o.only || ["BTC", "ETH"]) {
    const st = STATES[(k + (a === "ETH" ? 3 : 0)) % STATES.length];
    assets[a] = research(st === "obs" ? "관찰" : "사용 가능");
    la[a] = { bias: st === "zero" ? 0 : (k + a.length) % 3 === 0 ? -1 : 1, roll52: rnd() < .1 ? null : r2((rnd() - .5) * 1.2), watch: "유효", state: st, why: rnd() < .08 ? "" : WHY[st] };
  }
  const z = o.z ?? r2((rnd() - .5) * 6.4), chg = r2((rnd() - .5) * (unit === "bp" ? 120 : unit === "$bn" ? 30 : 12));
  const nul = (i) => (o.holes && i % 9 === 4 ? null : 0);
  const live = o.live === null ? null : { week: o.week || "2026-11-13", chg13: chg, z, stale: false, old: false, p3_mean: 0.1, p3_sd: 1.1,
    spark: o.short ? series(15, (i) => r2(Math.sin(i / 3))) : series(52, (i) => (nul(i) === null ? null : r2(Math.sin(i / 4 + k) * 2 + i / 30))),
    coin13: { BTC: series(52, (i) => r2(Math.cos(i / 5 + k) * 20)), ETH: series(52, (i) => (o.holes && i === 30 ? null : r2(Math.cos(i / 6 + k) * 25))) }, assets: la };
  const fs = ["ok", "ok", "ok", "bad", "wait"][k % 5];
  const feed = o.feed === null ? {} : { short: k % 2 ? "주 1회 공표" : "매일 공표", cadence: k % 2 ? "주 1회 · 목요일(수요일 기준, 연준 H.4.1)" : "영업일 매일 · 다음 영업일 공표",
    ok: fs === "ok" ? true : fs === "bad" ? false : null, last: "2026-11-16", age_d: fs === "wait" ? null : 2 + (k % 4), max_d: 5, why: fs === "bad" ? `X${k} 마지막 값 2026-11-09 (9일 전) -- 한도 5일` : "" };
  return { key: `${kind}:${name}`, name, kind, unit, realtime: "free", realtime_note: k % 3 ? `실시간 경로 설명 ${k} · 무료 시세는 약 10분 지연` : "", note: o.note || "", assets, live, feed };
};
const summarize = (rows) => Object.fromEntries(["BTC", "ETH"].map((a) => {
  const s = { co_up: 0, co_dn: 0, lead_up: 0, lead_dn: 0, neutral: 0, excluded: 0 };
  for (const r of rows) { const l = r.live && r.live.assets[a], st = l && l.state; if (!st || st === "obs") continue; if (st === "zero") s.neutral++; else if (st === "count") s[`${r.kind}_${l.bias > 0 ? "up" : "dn"}`]++; else s.excluded++; }
  return [a, s];
}));
const full = (o = {}) => {
  const rows = [];
  FAMS.forEach(([n, u], k) => {
    rows.push(row(n, u, "co", k, { note: n === "연준BS" ? "양적긴축 기간과 BTC 상승이 겹친 관계 -- 최근 변동이 1/3 로 줄었다(보류 권장)" : n === "RRP" ? "잔고가 이미 0 근처($0.2bn) -- 더 움직일 여지가 없다" : "",
      live: n === "브렌트" ? null : undefined, holes: n === "달러지수", only: n === "HY스프레드" ? ["BTC"] : undefined, week: n === "STLFSI" ? "2026-11-06" : undefined }));
    if (k % 2 === 0 || n === "10Y금리") rows.push(row(n, u, "lead", k + 5, { only: n === "원/달러" ? ["ETH"] : undefined, holes: n === "나스닥" }));
  });
  return { ok: true, asof: NA, rows, summary: summarize(rows), errors: o.errors || {}, method: "…" };
};
const small = () => {
  const rows = [row("VIX", "%", "lead", 3, { z: -2.2, short: true }), row("KOSPI", "%", "co", 1, { z: 0.4 }), row("RRP", "$bn", "co", 2, { z: 3.6, note: "잔고가 이미 0 근처($0.2bn)" }), row("TGA", "%", "lead", 2, { z: -0.1, feed: null })];
  for (const r of rows) for (const a in r.live.assets) r.live.assets[a].state = "weak";
  return { ok: true, asof: NA, rows, summary: summarize(rows), errors: {}, method: "…" };
};
// 장중: 나스닥 1분봉 24시간(xyz100) · S&P 5분봉 + 초 가격 · VIX 5분봉 · ZN 5분봉(국채 24시간)
const bars = (t1, step, n, p0, amp, ph) => series(n, (i) => [t1 - (n - 1 - i) * step, r2(p0 * (1 + amp * Math.sin(i / 37 + ph) + amp * .3 * Math.sin(i / 5)))]);
const xyz = (now, fresh) => ({ bars: bars(now - 60 - 30, 60, 1440, 21000, .006, 1), live: { px: 21033.5, recv_sec: fresh ? now - 3 : now - 200 } });
const intra = (now, o = {}) => ({ at: now - 20, p: { ok: true, now_sec: now - 22, symbols: {
  ...(o.noSP ? {} : { SP: { bars: bars(now - 140, 300, 288, 6050, .004, 2), live: { px: 6061.25, recv_sec: o.spOld ? now - 400 : now - 25 } } }),
  VIX: { bars: bars(now - 1500, 300, 280, 17.2, .05, 3) } } } });
const treas = (now, o = {}) => ({ at: now * 1000 - 60e3, p: { ok: true, symbols: { ZN: { bars: bars(now - (o.ago ?? (o.late ? 2400 : 600)), 300, 280, 112.4, .002, 4) }, ZF: { bars: [] } } } });
// 첫 선택 갈래: |z| 가장 큰 지표(TGA 3.97)는 셈 칸이 없고 그다음(지준 3.95)만 셈 칸 → 셈 칸 중 최대 = 지준
const pick = () => {
  const p = full();
  for (const r of p.rows) {
    if (r.live) r.live.z = r2(r.live.z / 10);
    for (const a in (r.live || {}).assets || {}) if (r.live.assets[a].state === "count" && !["지준", "TGA"].includes(r.name)) r.live.assets[a].state = "weak";
    if (r.name === "지준" && r.live) { r.live.z = 3.95; for (const a in r.live.assets) r.live.assets[a].state = "count"; }
    if (r.name === "TGA" && r.live) { r.live.z = -3.97; for (const a in r.live.assets) r.live.assets[a].state = "zero"; }
  }
  p.summary = summarize(p.rows);
  return p;
};
const ev = (t, nm, imp = "high", cat = "") => ({ time_utc: new Date(t * 1000).toISOString().replace(".000Z", "+00:00"), importance: imp, category: cat, title_ko: nm, title: "x" });
const evs = (now) => [ev(now + 2 * 3600, "미국 CPI"), ev(now + 2.5 * 3600, "근원 CPI"), ev(now + 2.6 * 3600, "미국 CPI"), ev(now + 5 * 3600, "FOMC 의사록", "medium", "fomc"),
  ev(now + 7 * 3600, "실업수당 청구"), ev(now + 8.5 * 3600, "소매판매"), ev(now + 11.5 * 3600, "PMI 속보치 아주 긴 이름의 발표 일정"), ev(now + 30 * 3600, "GDP"), ev(now - 3600, "지난 발표"), ev(now + 3 * 3600, "중요도 낮음", "low")];
const W = 1838, H = 917;
// 갈래: 응답 없음 · ok 아님(오류 글 이스케이프) · 전부 있음(첫 선택 = 셈 칸 중 |z| 최대 · 발표 묶음 · 나스닥 줄 선택 아님) · 10Y금리 선택(ZN 지연 띠 · 원천 실패 · 수신 늦음 · 나스닥 끊김 · S&P 끊김) ·
//   주말(CME·VIX 쉼 · S&P 받는 중 · 나스닥 받는 중 · VIX 선택) · 작은 응답(기타 계열 · 셈 칸 없음 → |z| 최대 · 선행 짧은 이력 = 겹친 그림 없음 · 낮은 높이 = 줄 최소 22px) ·
//   나스닥 선택(실시간 · 앞 12시간 발표 없음) · S&P 선택(좁은 폭 1600 · 발표 2시간 묶음) · 첫 선택 = 셈 칸 중 |z| 최대(전체 최대는 중립만) · ZN 마지막 봉 27.5분(30분 경계 앞)
const inputs = [
  { now: NA, w: W, h: H, trend: null, sel: null, xyz: null, intra: null, treas: null, events: null },
  { now: NA, w: W, h: H, trend: { ok: false, error: "<b>FRED</b> 시간 초과 & 재시도" }, sel: null, xyz: null, intra: null, treas: null, events: [] },
  { now: NA, w: W, h: H, trend: full(), sel: null, xyz: xyz(NA, true), intra: intra(NA), treas: treas(NA), events: evs(NA) },
  { now: NA, w: W, h: H, trend: full({ errors: { DGS10: "ReadTimeout: x", WALCL: "HTTPError: 500" } }), sel: "10Y금리", xyz: xyz(NA, false), intra: intra(NA, { spOld: true }), treas: treas(NA, { late: true }), events: evs(NA) },
  { now: NB, w: W, h: H, trend: full(), sel: "VIX", xyz: { bars: [], live: null }, intra: intra(NB, { noSP: true }), treas: treas(NB), events: [] },
  { now: NA, w: W, h: 560, trend: small(), sel: null, xyz: null, intra: null, treas: null, events: evs(NA) },
  { now: NA, w: W, h: H, trend: full(), sel: "나스닥", xyz: xyz(NA, true), intra: intra(NA), treas: treas(NA), events: [ev(NA + 13 * 3600, "먼 발표")] },
  { now: NA, w: 1600, h: H, trend: full(), sel: "S&P500", xyz: xyz(NA, true), intra: intra(NA), treas: treas(NA), events: evs(NA) },
  { now: NA, w: W, h: H, trend: pick(), sel: null, xyz: xyz(NA, true), intra: intra(NA), treas: treas(NA, { ago: 1650 }), events: evs(NA) },
];
fs.writeFileSync(here + "fixture/mt_in.json", JSON.stringify(inputs) + "\n");
const run = (s) => {
  const now = s.now * 1000;
  class D extends Date { constructor(...a) { super(...(a.length ? a : [now])); } static now() { return now; } }
  const box = { innerHTML: "", querySelector: () => null, addEventListener() {} };
  const env = { Date: D, el: (id) => (id === "mtBody" ? box : null), document: { addEventListener() {}, activeElement: null }, helpKey: null, latestMacroEvents: s.events,
    xyzBars: s.xyz ? s.xyz.bars : [], xyzLive: s.xyz ? s.xyz.live : null, xyzSkew: 0, CSS: { escape: (x) => x }, __S: s };
  const px = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  return new Function("px", `with (px) { ${body}\nlatestMacroTrend = __S.trend; mtSel = __S.sel; mtTreas = __S.treas; mtIntra = __S.intra;\nrenderMacroTrend();\nreturn [el("mtBody").innerHTML, mtHeadHtml(), mtSel]; }`)(px);
};
const res = inputs.map(run);
fs.writeFileSync(process.env.OUT || here + "fixture/mt_html.json", JSON.stringify(res) + "\n");
for (const [b, h, sel] of res) console.log(`선택 ${sel} · 본문 ${b.length}자 · 머리 ${h.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").slice(0, 120)}`);
