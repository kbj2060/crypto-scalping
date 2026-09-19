// 1초 수급 패널 렌더 검증(2026-09-20 절대값 개편). node --check 는 문법만 본다 --
// 여기서는 실제로 그려 보고 ①롤링 합 ②계단 눈금 ③상자 경계 ④공백 절단을 검사한다.
//   node test/render_supply_1s_smoke_20260920.js
const fs = require("fs");
const src = fs.readFileSync("dashboard/live/app.js", "utf8");
const grab = (re, what) => { const m = src.match(re); if (!m) { console.log(`🔴 못 찾음: ${what}`); process.exit(1); } return m[0]; };
const FN = grab(/function renderSupply1s\(box = null\) \{[\s\S]*?\n\}\n/, "renderSupply1s");
const PRE = grab(/function fmtFootprintQty\(v\) \{[\s\S]*?\n\}\n/, "fmtFootprintQty")
          + grab(/const SUPPLY_1S_WINDOW = \d+;/, "WINDOW")
          + "\n" + grab(/const SUPPLY_1S_ROLL = \d+;/, "ROLL")
          + "\n" + grab(/const SUPPLY_1S_STEPS = \[[^\]]*\];/, "STEPS") + "\n";

const NOW = 1700000000;   // CI 의 esprima 4 는 숫자 구분자(1_700_000_000)를 모른다
const W = 1318, H = 150;

function draw(label, { qty = 10, hole = null, oi = null } = {}) {
  const els = [];
  const mk = (kind) => ({ _a: {}, _kind: kind, kids: [],
    setAttribute(k, v) { this._a[k] = v; }, appendChild(c) { this.kids.push(c); },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; } });
  global.document = { createElementNS: (ns, kind) => { const e = mk(kind); els.push(e); return e; } };
  const sup = new Map(), oiMap = new Map();
  for (let s = NOW - 350; s <= NOW; s++) {
    if (hole && s > hole[0] && s < hole[1]) continue;
    sup.set(s, [0, 0, qty, 0, qty, 0, 3000]);      // 고래 매수만 qty ETH/초
  }
  if (oi) for (let s = NOW - 350; s <= NOW; s += 5) oiMap.set(s, oi * s);
  global.supply1s = sup;
  global.oi1s = oiMap;
  global.supply1sMeta = { now: NOW, retailMaxUsd: 10000, whaleMinUsd: 100000 };
  const svg = { _a: {}, innerHTML: "", kids: [], setAttribute(k, v) { this._a[k] = v; },
                appendChild(c) { this.kids.push(c); }, parentElement: { clientWidth: W } };
  try {
    eval(PRE + FN + "\nrenderSupply1s({ svg, w: W, h: H });");
  } catch (e) { console.log(`🔴 ${label}: ${e.constructor.name} — ${e.message}`); return null; }
  const nums = els.flatMap((e) => ["x", "y", "x1", "x2", "y1", "y2", "width", "height"]
    .filter((k) => k in e._a).map((k) => Number(e._a[k])));
  if (!nums.every(Number.isFinite)) { console.log(`🔴 ${label}: NaN 좌표`); return null; }
  const ys = els.filter((e) => e._kind === "path").flatMap((e) =>
    [...String(e._a.d).matchAll(/[ML](-?[\d.]+) (-?[\d.]+)/g)].map((m) => Number(m[2])));
  if (ys.some((y) => y < -0.5 || y > H + 0.5)) { console.log(`🔴 ${label}: 세로 넘침`); return null; }
  return { els, svg, paths: els.filter((e) => e._kind === "path") };
}

let ok = true;
const fail = (m) => { console.log("🔴 " + m); ok = false; };

// ① 매초 10 ETH 가 **줄곧** 들어오면 30초 롤링은 화면 왼쪽 끝부터 끝까지 300 으로 평평해야
//    한다. 여기가 기울어지면 창 시작 전 ROLL 초를 안 읽은 것(가짜 램프)이다.
const a = draw("정상 10 ETH/s", {});
if (!a) ok = false;
else {
  const lineYs = [...String(a.paths[a.paths.length - 1]._a.d).matchAll(/[ML][\d.]+ (-?[\d.]+)/g)]
    .map((m) => Number(m[1]));
  if (lineYs.length < 290) fail(`그린 점이 적다 (${lineYs.length})`);
  if (new Set(lineYs.map((y) => y.toFixed(1))).size !== 1) fail(`고래선이 평평하지 않다 — 왼쪽 램프? ${lineYs[0]} .. ${lineYs[lineYs.length - 1]}`);
  // ② 계단 눈금: 봉우리 300 -> 500 칸. 축 라벨에 그 값이 있어야 한다.
  const texts = a.els.filter((e) => e._kind === "text").map((e) => e.textContent);
  if (!texts.includes("+500")) fail(`계단 눈금이 500 이 아니다: ${JSON.stringify(texts)}`);
  if (!texts.some((t) => String(t).includes("30초 순수급"))) fail("머리글이 없다");
  if (!texts.some((t) => String(t).startsWith("고래 +"))) fail("고래 꼬리표가 없다");
  // ③ 면적이 위(초록)에만 칠해져야 한다 -- 아래 면적은 0선에 눌려 납작하다.
  const good = a.paths.find((p) => p._a.fill === "var(--good)");
  const bad = a.paths.find((p) => p._a.fill === "var(--bad)");
  if (!good || !bad) fail("0선 면적이 없다");
}

// ④ 공백은 이어 그리지 않는다 -- 선이 끊겨야(M 이 둘 이상) 한다.
const b = draw("공백 60초", { hole: [NOW - 200, NOW - 140] });
if (!b) ok = false;
else {
  const d = String(b.paths[b.paths.length - 1]._a.d);
  if ((d.match(/M/g) || []).length < 2) fail("공백을 가로질러 이었다");
  if (!b.els.some((e) => e._kind === "rect")) fail("공백 음영이 없다");
}

// ⑤ 폭발해도 상자를 안 넘는다(마지막 계단 초과) + OI 차분선이 붙는다.
if (!draw("폭발 5000 ETH/s + OI", { qty: 5000, oi: 0.5 })) ok = false;

// ⑥ 순수급이 정확히 0 인 창. fmtFootprintQty(0) 이 ""를 주므로 그대로 쓰면 꼬리표가
//    「고래 +」로 숫자 없이 뜬다 -- 2026-09-20 배포본 스크린샷에서 실제로 그랬다.
//    30초 창에서 0 은 드물지 않다(고래 주문이 분당 13건).
const z = draw("순수급 0", { qty: 0 });
if (!z) ok = false;
else {
  const bad = z.els.filter((e) => e._kind === "text")
    .map((e) => String(e.textContent || "")).filter((t) => /[+-]$/.test(t));
  if (bad.length) fail(`꼬리표가 부호로 끝난다(숫자 없음): ${JSON.stringify(bad)}`);
  const texts = z.els.filter((e) => e._kind === "text").map((e) => e.textContent);
  if (!texts.includes("고래 +0")) fail(`0 일 때 「고래 +0」이 아니다: ${JSON.stringify(texts)}`);
}

console.log(ok ? "✅ 1초 수급 패널 OK" : "");
process.exit(ok ? 0 : 1);
