// 사분면 행 RVOL 선 검증 (2026-09-23, 「거래대금 대신 rvol」).
//   node test/render_quad_rvol_smoke_20260923.js
// 🔴상태 하나가 아니라 **전부**를 돈다 — 워커 정상/부재 · ETH/타코인 · 축 잘림 유무 ·
//   세션값 유무. 폴백(거래대금)이 조용히 동작해야 하고, 「빈 선」이 되면 안 된다.
// 🔴캐시 키도 본다: RVOL 은 fpBars/oiBars 와 무관하게 도착하므로 키에 안 들어가면
//   «거래대금»으로 그린 노드가 계속 재사용된다(실제로 그 버그가 있었다).
const fs = require("fs");
const src = fs.readFileSync("dashboard/live/app.js", "utf8");
let fail = 0;
const ck = (c, what) => { if (!c) { console.log("🔴 " + what); fail++; } else console.log("  ok " + what); };

// ── 계약 ① 캐시 키에 RVOL 서명과 코인이 들어 있다 ───────────────────────────────
const keyLine = src.match(/cachedLayer\("quadLane",[\s\S]{0,220}?\(g\) => \{/);
ck(!!keyLine && /rvolSig/.test(keyLine[0]) && /activeSnapshotAsset/.test(keyLine[0]),
   "quadLane 캐시 키에 rvolSig·코인 포함");
ck(!!keyLine && !/objToken\(\s*latestBreakoutDetector/.test(keyLine[0]),
   "키가 objToken(신원)이 아니라 값 기반");

// ── 계약 ②~ 선 그리기 블록을 잘라 실행한다 ──────────────────────────────────────
const a = src.indexOf("const rvolBy = new Map()");
if (a < 0) { console.log("🔴 rvol 블록 앵커를 못 찾음"); process.exit(1); }
let depth = 0, end = src.indexOf("cachedLayer(\"quadLane\"", a);
const open = src.indexOf("{", src.indexOf("(g) => ", end));
for (end = open; end < src.length; end++) {
  if (src[end] === "{") depth++;
  else if (src[end] === "}" && --depth === 0) { end++; break; }
}
const BLOCK = src.slice(a, src.indexOf("(g) => ", src.indexOf("cachedLayer(\"quadLane\"", a)))
                 .replace(/cachedLayer\("quadLane",[\s\S]*$/, "")
            + "\n(function(g){" + src.slice(open + 1, end - 1) + "})(GROUP);";

const NS = "http://www.w3.org/2000/svg";
function run({ asset = "eth", worker = true, spike = false, session = 1.02, label = "보통",
               mobile = false } = {}) {
  const els = [];
  const mk = (kind) => ({ _a: {}, _kind: kind, kids: [],
    setAttribute(k, v) { this._a[k] = v; }, appendChild(c) { this.kids.push(c); },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; },
    getComputedTextLength() { return String(this._t || "").length * 6; } });
  const document = { createElementNS: (ns, kind) => { const e = mk(kind); els.push(e); return e; } };
  const N = 12, T0 = 1789430000 - (1789430000 % 300);
  const candles = [], fpBars = [], oiBars = [], times = [], bar = [];
  for (let i = 0; i < N; i++) {
    const t = T0 + i * 300;
    candles.push({ time: t, close: 100, high: 101, low: 99, open: 100 });
    fpBars.push({ time: t, levels: [[100, 5, 4]] });
    oiBars.push([t, i % 2 ? 3 : -3]);
    times.push(new Date(t * 1000).toISOString());
    bar.push(spike && i === N - 1 ? 9.8 : 1 + i * 0.05);
  }
  const bar5 = bar.map((v, i) => (i === N - 1 && spike ? 9.8 : v));
  const latestBreakoutDetector = worker ? { rvol: {
    base_days: 14, line_minutes: 60, bar, bar5, times,
    session, session_label: label, session_bounds: [0.7, 1.37] } } : null;
  const badge = { hidden: true, textContent: "-", title: "" };
  const fn = new Function(
    "candles", "fpBars", "oiBars", "GROUP", "NS", "document", "xAt", "bw", "ml", "cw", "w",
    "quadY", "QUAD_H", "QUAD_TXT", "QUAD_TEXT_OK", "mobileChart", "objToken", "cachedLayer",
    "activeSnapshotAsset", "latestBreakoutDetector", "supplyFlowOfBar", "fmtFootprintQty",
    "fmtUsdCompact", "fmtDateTick", "el", BLOCK);
  const GROUP = mk("g");
  fn(candles, fpBars, oiBars, GROUP, NS, document, (i) => 40 + i * 20, 16, 40, 240, 320,
     10, mobile ? 88 : 132, mobile ? 28 : 34, true, mobile,
     () => "#1", (n, sig, draw) => draw(GROUP), asset, latestBreakoutDetector,
     () => ({ whale: 2, mid: 1, retail: 1 }), (v) => String(Math.round(v)),
     (v) => "$" + Math.round(v), () => "09-23 10:00",
     (id) => (id === "rvolSessionBadge" ? badge : null));
  els.badge = badge;
  return els;
}
const texts = (els) => els.filter((e) => e._kind === "text").map((e) => String(e.textContent));
const pathD = (els) => (els.find((e) => e._kind === "path") || { _a: {} })._a.d || "";

// ② 워커 정상(ETH) -> RVOL 범례 · 평소 점선 · 선이 그려진다
{
  const els = run();
  ck(texts(els).some((t) => t.includes("─ RVOL 60분")), "ETH·워커정상 -> 범례 «─ RVOL 60분»");
  const dash = els.filter((e) => e._kind === "line" && e._a["stroke-dasharray"] === "3 4");
  ck(dash.length === 1, `«평소»(1.0) 점선 1줄 (실제 ${dash.length})`);
  ck(pathD(els).split("L").length >= 10, "선이 전 봉에 그려짐");
  ck(!texts(els).some((t) => t.includes("오늘")), "세션은 범례에 **없다**(상단 배지로 옮김)");
  ck(els.badge.hidden === false && els.badge.textContent === "오늘 거래량 보통 1.02배",
     `상단 배지 (실제 hidden=${els.badge.hidden} "${els.badge.textContent}")`);
  ck(/q25 \/ q75|분위/.test(els.badge.title), "배지 툴팁이 경계를 분위로 밝힌다");
}
// ③ 워커 부재 -> 조용히 거래대금 폴백. **빈 선이 되면 안 된다**
{
  const els = run({ worker: false });
  ck(texts(els).some((t) => t.includes("─ 거래대금")), "워커 부재 -> 거래대금 폴백");
  ck(pathD(els).split("L").length >= 10, "폴백에서도 선이 비지 않음");
  ck(els.badge.hidden === true, "워커 부재 -> 상단 배지 숨김");
}
// ④ 다른 코인 -> 워커가 살아 있어도 ETH 값을 쓰지 않는다
{
  const els = run({ asset: "btc" });
  ck(texts(els).some((t) => t.includes("─ 거래대금")), "BTC -> ETH RVOL 을 쓰지 않음");
  ck(els.badge.hidden === true, "BTC -> 상단 배지 숨김(ETH 전용 워커)");
}
// ⑤ 축 잘림 -- 9.8배 스파이크가 있어도 축은 5배에서 멈추고 «+» 를 붙인다
{
  const els = run({ spike: true });
  ck(texts(els).some((t) => /축 5\.0배\+/.test(t)), `축 잘림 표기 (실제 ${texts(els).filter((t) => t.includes("축"))})`);
  const dash = els.find((e) => e._kind === "line" && e._a["stroke-dasharray"] === "3 4");
  const ys = pathD(els).match(/-?\d+\.\d+(?=\s*$|\s+L)/g) || [];
  ck(Number(dash._a.y1) > 100, `«평소» 선이 바닥에 깔리지 않음 (y=${dash._a.y1})`);
}
// ⑥ 세 라벨이 그대로 나온다 — 라벨은 워커가 붙이고 화면은 나르기만 한다
for (const [sv, lab] of [[0.55, "적음"], [1.02, "보통"], [2.10, "많음"]]) {
  const els = run({ session: sv, label: lab });
  ck(els.badge.textContent === `오늘 거래량 ${lab} ${sv.toFixed(2)}배`,
     `배지 «${lab}» (실제 "${els.badge.textContent}")`);
  ck(els.badge.hidden === false, `«${lab}» 상태에서 배지가 보인다`);
}
// ⑦ 봉 툴팁이 «이 봉» 값을 따로 말한다 (선이 1시간이라 흡수 판독을 툴팁이 이어받는다)
{
  const els = run({ spike: true });
  const tips = els.filter((e) => e._kind === "title").map((e) => String(e.textContent));
  ck(tips.some((t) => t.includes("이 봉은 평소의 9.80배") && t.includes("선(최근 60분)")),
     "봉 툴팁에 «이 봉» 값과 «선» 값이 둘 다");
}
// ⑧ 모바일 한 줄 범례
{
  const els = run({ mobile: true });
  ck(texts(els).some((t) => t.includes("─ RVOL 60분 축")), "모바일 한 줄 범례에 RVOL");
  ck(els.badge.hidden === false, "모바일에서도 상단 배지");
}
process.exit(fail ? 1 : 0);
