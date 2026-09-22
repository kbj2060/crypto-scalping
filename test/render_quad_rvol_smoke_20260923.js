// 사분면 레인의 60분 RVOL 띠 + 상단 세션 배지 검증 (2026-09-23).
//   node test/render_quad_rvol_smoke_20260923.js
// 🔴상태 하나가 아니라 **전부**를 돈다 — 워커 정상/부재 · ETH/타코인 · 세 라벨 · 모바일.
// 🔴이 파일의 역사가 곧 계약이다: RVOL 은 CVD 레인 -> 전용 레인 -> **사분면 위 띠** 로 옮겼고,
//   5분 계열은 제거됐다(흡수가 측정에서 무너졌다). 되살아나면 여기서 실패해야 한다.
const fs = require("fs");
const src = fs.readFileSync("dashboard/live/app.js", "utf8");
let fail = 0;
const ck = (c, what) => { if (!c) { console.log("🔴 " + what); fail++; } else console.log("  ok " + what); };
const NS = "http://www.w3.org/2000/svg";

// ── 계약: 옮긴 자리와 지운 것 ────────────────────────────────────────────────
ck(!/rvolBar5By|\brvol5\b|rv\.bar5/.test(src), "5분 계열(bar5)이 클라에 남아 있지 않다");
ck(!/cachedLayer\("rvolLane"/.test(src) && !/RVOL_H/.test(src), "전용 RVOL 레인이 없다");
ck(/const RVOL_BAND = fpBars\.length/.test(src), "사분면 레인 안 RVOL_BAND 가 있다");
{
  const key = src.match(/cachedLayer\("quadLane",[\s\S]{0,260}?\(g\) => \{/);
  ck(!!key && /rvolSig/.test(key[0]), "quadLane 캐시 키에 rvolSig 포함(안 넣으면 옛 노드 재사용)");
  const cum = src.match(/cachedLayer\("cumLane",[\s\S]{0,300}?\(g\) => \{/);
  const cumStart = src.indexOf('cachedLayer("cumLane"');
  const cumBody = src.slice(cumStart, cumStart + 9000);
  ck(!/rvolBy/.test(cumBody), "누적 CVD 레인은 RVOL 을 더 이상 그리지 않는다");
  ck(!!cum, "cumLane 블록이 있다");
}

// ── 사분면 레인을 잘라 실제로 돌린다 ─────────────────────────────────────────
const a = src.indexOf("const rvolBy = new Map()");
if (a < 0) { console.log("🔴 rvol 블록 앵커를 못 찾음"); process.exit(1); }
const qi = src.indexOf('cachedLayer("quadLane"', a);
const open = src.indexOf("{", src.indexOf("(g) => ", qi));
let depth = 0, end = open;
for (; end < src.length; end++) {
  if (src[end] === "{") depth++;
  else if (src[end] === "}" && --depth === 0) { end++; break; }
}
const PRE = src.slice(a, qi);                       // rvolBy 추출 + 배지
const LANE = src.slice(open + 1, end - 1);

function run({ asset = "eth", worker = true, session = 1.02, label = "보통", mobile = false } = {}) {
  const els = [];
  const mk = (kind) => ({ _a: {}, _kind: kind, kids: [],
    setAttribute(k, v) { this._a[k] = v; }, appendChild(c) { this.kids.push(c); },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; },
    getComputedTextLength() { return String(this._t || "").length * 6; } });
  const document = { createElementNS: (ns, kind) => { const e = mk(kind); els.push(e); return e; } };
  const T0 = 1789430000 - (1789430000 % 300), N = 12;
  const candles = [], fpBars = [], oiBars = [], times = [], bar = [];
  for (let i = 0; i < N; i++) {
    const t = T0 + i * 300;
    candles.push({ time: t });
    fpBars.push({ time: t, levels: [[100, 5, 4]] });
    oiBars.push([t, i % 2 ? 3 : -3]);
    times.push(new Date(t * 1000).toISOString());
    bar.push(1 + i * 0.05);
  }
  const latestBreakoutDetector = worker ? { rvol: {
    base_days: 14, line_minutes: 60, bar, times,
    session, session_label: label, session_bounds: [0.7, 1.37] } } : null;
  const badge = { hidden: true, textContent: "-", title: "" };
  const g = mk("g");
  // 데스크톱 기하: quadY 300 · RVOL_BAND 30 · QUAD_H 162
  new Function("candles", "fpBars", "oiBars", "GROUP", "NS", "document", "xAt", "bw", "ml", "cw",
               "w", "quadY", "QUAD_H", "QUAD_TXT", "QUAD_TEXT_OK", "RVOL_BAND", "mobileChart",
               "objToken", "cachedLayer", "activeSnapshotAsset", "latestBreakoutDetector",
               "supplyFlowOfBar", "fmtFootprintQty", "fmtUsdCompact", "fmtDateTick", "el",
               PRE + "\ncachedLayer('quadLane', '', (g) => {" + LANE + "});")(
    candles, fpBars, oiBars, g, NS, document, (i) => 40 + i * 20, 16, 40, 240, 320,
    300, mobile ? 112 : 162, mobile ? 28 : 34, true, mobile ? 24 : 30, mobile,
    () => "#1", (n, sig, draw) => draw(g), asset, latestBreakoutDetector,
    () => ({ whale: 2, mid: 1, retail: 1 }), (v) => String(Math.round(v)),
    (v) => "$" + Math.round(v), () => "09-23 10:00",
    (id) => (id === "rvolSessionBadge" ? badge : null));
  els.badge = badge;
  return els;
}
const txts = (els) => els.filter((e) => e._kind === "text").map((e) => String(e.textContent));
const rvLine = (els) => els.filter((e) => e._kind === "path" && e._a.stroke === "var(--turnover)");

// ① 정상 — 띠가 사분면 «위»(quadY..quadY+30)에 있고 막대는 그 아래
{
  const els = run();
  const line = rvLine(els);
  ck(line.length === 1, `60분 RVOL 선 1줄 (실제 ${line.length})`);
  const ys = (line[0]._a.d.match(/-?\d+\.\d+/g) || []).filter((_, i) => i % 2 === 1).map(Number);
  ck(Math.max.apply(null, ys) <= 330 && Math.min.apply(null, ys) >= 300,
     `선이 띠 안(300~330)에 있다 (실제 ${Math.min.apply(null, ys).toFixed(0)}~${Math.max.apply(null, ys).toFixed(0)})`);
  const bars = els.filter((e) => e._kind === "rect" && e._a.rx === "3");
  ck(bars.length === 12, `사분면 막대 12개 (실제 ${bars.length})`);
  ck(Math.min.apply(null, bars.map((b) => Number(b._a.y))) >= 330,
     "막대가 띠 아래에서만 그려진다(겹치지 않는다)");
  ck(els.some((e) => e._kind === "line" && e._a["stroke-dasharray"] === "3 4"), "«평소»(1.0) 점선");
  ck(txts(els).some((t) => t === "거래량 60분"), `좌측 라벨 (실제 ${txts(els).filter((t) => t.includes("거래량"))})`);
  ck(txts(els).some((t) => /^\d\.\d\d배$/.test(t)), "우측 현재값");
  ck(!txts(els).some((t) => t.includes("5분")), "5분 표기가 없다");
}
// ② 워커 부재 / 타코인 -> 띠만 빠지고 막대는 그대로
for (const [nm, opt] of [["워커 부재", { worker: false }], ["BTC", { asset: "btc" }]]) {
  const els = run(opt);
  ck(rvLine(els).length === 0, `${nm} -> 띠 없음`);
  ck(els.filter((e) => e._kind === "rect" && e._a.rx === "3").length === 12, `${nm} -> 막대는 그대로`);
  ck(els.badge.hidden === true, `${nm} -> 상단 배지 숨김`);
}
// ③ 세션 배지 세 라벨
for (const [sv, lab] of [[0.55, "적음"], [1.02, "보통"], [2.10, "많음"]]) {
  const els = run({ session: sv, label: lab });
  ck(els.badge.textContent === `오늘 거래량 ${lab} ${sv.toFixed(2)}배` && els.badge.hidden === false,
     `배지 «${lab}» (실제 "${els.badge.textContent}")`);
}
{
  const els = run();
  ck(/q25 \/ q75|분위/.test(els.badge.title), "배지 툴팁이 경계를 분위로 밝힌다");
}
// ④ 모바일에서도 띠가 산다
{
  const els = run({ mobile: true });
  ck(rvLine(els).length === 1, "모바일에서도 60분 띠");
  ck(txts(els).some((t) => t === "거래량"), "모바일 라벨은 짧게");
}
process.exit(fail ? 1 : 0);
