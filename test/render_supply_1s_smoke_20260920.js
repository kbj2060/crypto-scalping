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
          + "\n" + grab(/const SUPPLY_1S_STEPS = \[[^\]]*\];/, "STEPS") + "\n";

const SHARED = "const supply1sPeaks = { bn: 0, okx: 0 };\n";
const SEG = Number(PRE.match(/const SUPPLY_1S_SEGMENT = (\d+);/)[1]);
const NOW = 1700000000;   // CI 의 esprima 4 는 숫자 구분자(1_700_000_000)를 모른다
const BOUND = Math.floor(NOW / SEG) * SEG;      // 창 안의 5분 경계
const W = 1318, H = 150;

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
  const svg = { _a: {}, innerHTML: "", kids: [], setAttribute(k, v) { this._a[k] = v; },
                appendChild(c) { this.kids.push(c); }, parentElement: { clientWidth: W } };
  try {
    eval(PRE + SHARED + FN + "\nrenderSupply1s({ svg, w: W, h: H });");
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

// ① 매초 10 ETH 가 줄곧 들어오면 봉 안에서 **단조 상승**(y 는 단조 감소)이어야 한다.
//    롤링 시절엔 평평했다 -- 평평하면 누적이 아니라는 뜻이다.
const a = draw("정상 10 ETH/s", {});
if (!a) ok = false;
else {
  const sp = subpaths(a.line._a.d);
  // H 는 봉 하나만 그리므로 공백이 없으면 subpath 도 하나다(경계는 화면 왼쪽 끝이다).
  if (sp.length !== 1) fail(`봉 하나인데 선이 ${sp.length}조각이다 -- 경계가 화면 안에 들어왔나?`);
  const lastYs = ysOf(sp[sp.length - 1]);
  if (lastYs.length < 30) fail(`마지막 구간의 점이 적다 (${lastYs.length})`);
  if (!lastYs.every((y, i) => i === 0 || y < lastYs[i - 1] + 1e-9)) fail("구간 안에서 단조 상승이 아니다 -- 누적이 아니다");
  // 경계 직후 값은 0 근처(한 초분)여야 하고, 끝값은 구간 길이만큼 쌓여야 한다.
  const span = NOW - BOUND + 1;
  if (!a.texts.some((t) => t === "고래 +" + (span * 10 >= 1000 ? (span * 10 / 1000).toFixed(1) + "k" : String(span * 10))))
    fail(`끝 꼬리표가 봉 누적과 다르다 (기대 ${span * 10}): ${JSON.stringify(a.texts.filter((t) => t.startsWith("고래")))}`);
  if (!a.texts.some((t) => t.includes("봉 시작"))) fail("왼쪽 축 라벨(봉 시작)이 없다");
  if (!a.texts.some((t) => /지금 \(\d+초 경과\)/.test(t))) fail("진행 표시(N초 경과)가 없다");
  if (!a.texts.some((t) => t.includes("이번 5분봉 누적 순수급"))) fail("머리글이 없다");
  // ⭐x축은 **봉 전체**다: 아직 안 온 시간이 오른쪽에 남아야 하고, 그 자리가 음영으로 표시돼야 한다.
  const xs = [...String(a.line._a.d).matchAll(/[ML]([\d.]+) /g)].map((m) => Number(m[1]));
  const rects = a.els.filter((e) => e._kind === "rect");
  const rightEdge = Math.max(...rects.map((r) => Number(r._a.x) + Number(r._a.width)), 0);
  if (!(Math.max(...xs) < rightEdge - 5))
    fail(`선이 오른쪽 끝까지 갔다 -- x축이 봉 전체가 아니다 (선끝 ${Math.max(...xs).toFixed(1)} vs ${rightEdge.toFixed(1)})`);
  if (!rects.some((r) => Number(r._a["fill-opacity"]) === 0.05))
    fail("남은 시간 음영이 없다 -- 빈 오른쪽이 «데이터 없음»으로 읽힌다");
  if (!a.paths.some((p) => p._a.fill === "var(--good)") || !a.paths.some((p) => p._a.fill === "var(--bad)"))
    fail("0선 면적이 없다");
}

// ② 공백은 이어 그리지 않는다.
const b = draw("공백 60초", { hole: [NOW - 120, NOW - 60] });
if (!b) ok = false;
else {
  if (subpaths(b.line._a.d).length < 2) fail("공백을 가로질러 이었다");
  if (!b.els.some((e) => e._kind === "rect" && e._a.fill === "var(--neutral)")) fail("공백 음영이 없다");
}

// ③ 순수급이 0 이어도 꼬리표가 부호로 끝나면 안 된다(fmtFootprintQty(0) 은 "" 를 준다).
const z = draw("순수급 0", { qty: 0 });
if (!z) ok = false;
else {
  const bad = z.texts.filter((t) => /[+-]$/.test(t));
  if (bad.length) fail(`꼬리표가 부호로 끝난다: ${JSON.stringify(bad)}`);
  if (!z.texts.includes("고래 +0")) fail(`0 일 때 「고래 +0」이 아니다: ${JSON.stringify(z.texts)}`);
}

// ④ 셋이 동시에 0 근처여도 꼬리표가 안 겹친다(짝짓기로는 셋에서 깨진다).
const t3 = draw("꼬리표 겹침", { qty: 0, oi: 0.0001 });
if (!t3) ok = false;
else {
  const ys = t3.els.filter((e) => e._kind === "text" && /^(고래|리테일|신규계약|OI) /.test(String(e.textContent || "")))
    .map((e) => Number(e._a.y)).sort((x, y) => x - y);
  if (ys.length < 3) fail(`꼬리표가 셋이 아니다 (${ys.length})`);
  for (let i = 1; i < ys.length; i++) if (ys[i] - ys[i - 1] < 11.5) { fail(`꼬리표가 겹친다: ${JSON.stringify(ys)}`); break; }
}

// ⑤ 폭발해도 상자를 안 넘는다.
if (!draw("폭발 5000 ETH/s + OI", { qty: 5000, oi: 0.5 })) ok = false;

console.log(ok ? "✅ 1초 수급 패널 OK" : "");
process.exit(ok ? 0 : 1);
