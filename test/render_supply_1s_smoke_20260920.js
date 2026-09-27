// 1초 수급 패널 렌더 검증 (2026-09-20, 5분 리셋 누적판).
// 🔴이 패널은 네 번 바뀌었다: 창시작 누적 -> 30초 롤링 -> 5분 리셋 누적(창은 최근 5분)
//   -> **창 자체가 현재 5분봉**(시안 H). 30초 롤링은 사용자가 실제로 속아서 버렸다
//   (고래 매수가 30초 뒤 «가짜 매도»로 보였다).
//   H 의 핵심은 ①x축 왼쪽 끝 = 봉이 열린 시각 ②누적이 그 경계에서 0부터 쌓임
//   ③아직 안 온 시간이 오른쪽에 남고 그게 «없음»이 아니라 «아직»으로 보이는가.
//   node test/render_supply_1s_smoke_20260920.js
const fs = require("fs");
const src = fs.readFileSync("dashboard/live/app.js", "utf8");
const grab = (re, what) => { const m = src.match(re); if (!m) { console.log(`🔴 못 찾음: ${what}`); process.exit(1); } return m[0]; };
const FN = grab(/function renderSupply1s\(box = null, src = null\) \{[\s\S]*?\n\}\n/, "renderSupply1s");
const PRE = grab(/function fmtFootprintQty\(v\) \{[\s\S]*?\n\}\n/, "fmtFootprintQty")
          + grab(/const SUPPLY_1S_SEGMENT = \d+;/, "SEGMENT")
          + "\nconst qtyScale = () => 1;\n";   // 2026-09-28 계단 눈금(STEPS)은 다섯 줄 거울 막대로 바뀌며 사라졌다

const MERGED = grab(/function mergedSupplySrc\(\) \{[\s\S]*?\n\}\n/, "mergedSupplySrc");
const SEG = Number(PRE.match(/const SUPPLY_1S_SEGMENT = (\d+);/)[1]);
const NOW = 1700000000;   // CI 의 esprima 4 는 숫자 구분자(1_700_000_000)를 모른다
const BOUND = Math.floor(NOW / SEG) * SEG;      // 창 안의 5분 경계
const W = 1318, H = 386;   // 1단 데스크톱 실제 상자(400 − 네 숫자 줄 14)

function draw(label, { qty = 10, hole = null, oi = null } = {}) {
  const els = [];
  const mk = (kind) => ({ _a: {}, _kind: kind, kids: [],
    setAttribute(k, v) { this._a[k] = v; }, appendChild(c) { this.kids.push(c); },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; } });
  global.document = { createElementNS: (ns, kind) => { const e = mk(kind); els.push(e); return e; } };
  const sup = new Map(), oiMap = new Map();
  for (let s = NOW - 650; s <= NOW; s++) {
    if (hole && s > hole[0] && s < hole[1]) continue;
    sup.set(s, [0, 0, qty, 0, qty, 0, 3000]);      // 고래 매수만 qty ETH/초
  }
  if (oi) for (let s = NOW - 650; s <= NOW; s += 5) oiMap.set(s, oi * s);
  global.supply1s = sup;
  global.oi1s = oiMap;
  global.supply1sMeta = { now: NOW, retailMaxUsd: 10000, whaleMinUsd: 100000 };
  // 2026-09-23 렌더러가 기본 출처 객체를 만들 때 세 전역을 **즉시** 읽는다(예전엔 청산 루프
  // 안에서만 읽어서 하네스가 비워둬도 지나갔다). 실제 앱은 셋 다 항상 선언돼 있다.
  global.liq1s = new Map();
  // 2026-09-23 기본 출처가 OKX 레인(합산선용)도 즉시 읽는다. 비어 있으면 overlay=null 이라
  // 합산선을 안 그린다 -- 이 하네스는 바이낸스 단독 렌더를 검사한다.
  global.okxSupply1s = new Map();
  // 2026-09-23 기본 출처가 **합산**이라 mergedSupplySrc 가 읽는 전역이 전부 있어야 한다.
  global.spotSupply1s = new Map();
  global.okxLiq1s = new Map();
  global.okxOi1s = new Map();
  global.okxMeta = { now: 0, connected: true, tradeAge: 0 };
  global.spotMeta = { now: 0, connected: true, tradeAge: 0 };
  const svg = { _a: {}, innerHTML: "", kids: [], setAttribute(k, v) { this._a[k] = v; },
                appendChild(c) { this.kids.push(c); }, parentElement: { clientWidth: W } };
  try {
    eval(PRE + MERGED + FN + "\nrenderSupply1s({ svg, w: W, h: H });");
  } catch (e) { console.log(`🔴 ${label}: ${e.constructor.name} — ${e.message}`); return null; }
  const nums = els.flatMap((e) => ["x", "y", "x1", "x2", "y1", "y2", "width", "height"]
    .filter((k) => k in e._a).map((k) => Number(e._a[k])));
  if (!nums.every(Number.isFinite)) { console.log(`🔴 ${label}: NaN 좌표`); return null; }
  const paths = els.filter((e) => e._kind === "path");
  const ys = paths.flatMap((e) => [...String(e._a.d).matchAll(/[ML](-?[\d.]+) (-?[\d.]+)/g)].map((m) => Number(m[2])));
  if (ys.some((y) => y < -0.5 || y > H + 0.5)) { console.log(`🔴 ${label}: 세로 넘침`); return null; }
  const texts = els.filter((e) => e._kind === "text").map((e) => String(e.textContent || ""));
  return { els, paths, texts, line: paths[paths.length - 1] };
}

let ok = true;
const fail = (m) => { console.log("🔴 " + m); ok = false; };
const subpaths = (d) => String(d).split(/(?=M)/).filter((x) => x.trim());
const ysOf = (sp) => [...sp.matchAll(/[ML][\d.]+ (-?[\d.]+)/g)].map((m) => Number(m[1]));
// 2026-09-28 다섯 줄 거울 막대: 누적선 = stroke var(--ink) · fill none 인 path 넷(CVD·고래·중형·리테일 순서)
const cumLines = (r) => r.paths.filter((p) => p._a.fill === "none" && p._a.stroke === "var(--ink)");
const fmtK = (v) => (v >= 1000 ? (v / 1000).toFixed(1) + "k" : String(v));

// ① 매초 10 ETH 고래 매수만 들어오면 고래 줄 누적선이 봉 안에서 **단조 상승**(y 단조 감소), 값 = 봉 누적.
const a = draw("정상 10 ETH/s", {});
if (!a) ok = false;
else {
  const cl = cumLines(a);
  if (cl.length !== 4) fail(`누적선이 넷(CVD·고래·중형·리테일)이 아니다: ${cl.length}`);
  const sp = subpaths(cl[1]._a.d);
  if (sp.length !== 1) fail(`봉 하나인데 고래 선이 ${sp.length}조각이다`);
  const ys = ysOf(sp[0]);
  if (ys.length < 30) fail(`점이 적다 (${ys.length})`);
  if (!ys.every((y, i) => i === 0 || y < ys[i - 1] + 1e-9)) fail("고래 누적이 단조 상승이 아니다");
  const span = NOW - BOUND;
  for (const name of ["CVD", "고래", "중형", "리테일", "신규"]) if (!a.texts.includes(name)) fail(`줄 이름이 없다: ${name}`);
  if (!a.texts.includes("+" + fmtK(span * 10))) fail(`고래·CVD 누적값(+${fmtK(span * 10)})이 없다: ${JSON.stringify(a.texts.slice(0, 12))}`);
  if (!a.texts.some((t) => t.includes("봉 시작"))) fail("왼쪽 축 라벨(봉 시작)이 없다");
  if (!a.texts.some((t) => /지금 \(\d+초 경과\)/.test(t))) fail("진행 표시(N초 경과)가 없다");
  const xs = [...String(cl[1]._a.d).matchAll(/[ML]([\d.]+) /g)].map((m) => Number(m[1]));
  const rects = a.els.filter((e) => e._kind === "rect");
  const rightEdge = Math.max(...rects.map((r) => Number(r._a.x) + Number(r._a.width)), 0);
  if (!(Math.max(...xs) < rightEdge - 5)) fail("선이 오른쪽 끝까지 갔다 -- x축이 봉 전체가 아니다");
  if (!rects.some((r) => Number(r._a["fill-opacity"]) === 0.05)) fail("남은 시간 음영이 없다");
  if (!a.paths.some((p) => p._a.fill === "var(--good)")) fail("매수 막대(초록 path)가 없다");
  if (a.paths.some((p) => p._a.fill === "var(--bad)")) fail("매도가 0 인데 매도 막대가 있다");
}

// ② 공백은 이어 그리지 않는다(«0» 이 아니라 «모름»).
const b = draw("공백 60초", { hole: [NOW - 120, NOW - 60] });
if (!b) ok = false;
else {
  if (subpaths(cumLines(b)[1]._a.d).length < 2) fail("공백을 가로질러 이었다");
  if (!b.els.some((e) => e._kind === "rect" && e._a.fill === "var(--neutral)")) fail("공백 음영이 없다");
}

// ③ 순수급 0 이어도 값이 부호로 끝나면 안 된다(fmtFootprintQty(0) 은 "" 를 준다).
const z = draw("순수급 0", { qty: 0 });
if (!z) ok = false;
else {
  const bad = z.texts.filter((t) => /[+-]$/.test(t));
  if (bad.length) fail(`값이 부호로 끝난다: ${JSON.stringify(bad)}`);
  if (!z.texts.includes("+0")) fail(`0 일 때 «+0» 이 아니다: ${JSON.stringify(z.texts)}`);
}

// ④ 폭발해도(5000 ETH/s + OI) 상자를 안 넘는다 -- draw() 가 NaN·세로 넘침을 본다.
if (!draw("폭발 5000 ETH/s + OI", { qty: 5000, oi: 0.5 })) ok = false;

console.log(ok ? "✅ 1초 수급 패널 OK" : "");
process.exit(ok ? 0 : 1);
