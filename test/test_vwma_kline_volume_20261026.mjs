/* 화면 VWMA 100 = 바이낸스 5분봉 거래량 (2026-10-26). 실행: node test/<이 파일>
 * app.js 의 tfCandles·mcVwma 본문을 떼어 돌린다: 1h 로 묶을 때 거래량은 합 · 1주일 앞 volPre 는 이어질 때만 붙는다. */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const body = (name) => { const i = src.indexOf(`function ${name}(`); assert.ok(i > 0, name); return src.slice(i, src.indexOf("\n}\n", i) + 2); };
let fpWeek = null;
const load = (k) => new Function("fpWeekFor",
  `const chartTfK = ${k}, tfCMemo = {}, tfT = (t) => (chartTfK > 1 ? t - (t % (300 * chartTfK)) : t);
   ${body("tfCandles")} ${body("mcVwma")} return { tfCandles, mcVwma };`)(() => fpWeek);

const T0 = 1792800000;   // 시간 경계
const five = Array.from({ length: 24 }, (_, i) => ({ time: T0 + i * 300, open: 1, high: 1, low: 1, close: 100 + i, volume: 1 }));

// 1h: 5분봉 24개 → 2봉, 거래량 = 합
const H = load(12), h = H.tfCandles(five);
assert.deepEqual(h.map((c) => c.volume), [12, 12], "1h 거래량 = 5분봉 합");
assert.equal(H.mcVwma(h, 3).size, 0, "volPre 없으면 2봉으로 VWMA 3 없음");
fpWeek = { pre: Array.from({ length: 12 }, (_, i) => [T0 - 3600 + i * 300, 50, 2]) };   // 바로 앞 1h(이어짐)
const m = H.mcVwma(h, 3);
assert.equal(m.size, 1);
assert.ok(Math.abs(m.get(T0 + 3600) - (50 * 24 + 111 * 12 + 123 * 12) / 48) < 1e-9, "Σ(종가×거래량)/Σ거래량");
fpWeek = { pre: [[T0 - 7200, 50, 2]] };   // 사이 1시간이 빈 volPre
assert.equal(H.mcVwma(h, 3).size, 0, "끊긴 volPre 는 안 잇는다");

// 5m: 거래량 없는 봉(형성 5분봉)이 창에 들면 그 점은 없다
fpWeek = null;
const F = load(1), g = five.map((c, i) => (i === 23 ? { ...c, volume: undefined } : c));
const v = F.mcVwma(g, 3);
assert.ok(v.has(T0 + 22 * 300) && !v.has(T0 + 23 * 300));
console.log("ok");
