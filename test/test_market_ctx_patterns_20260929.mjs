/* 시장 맥락 카드·차트 겹침의 순수 함수 (2026-09-29). 실행: node test/<이 파일>
 * app.js 에서 함수 본문을 떼어 **실제로 돌린다** -- 패턴 판정(다이버·소진·흡수?) · 미국장 시각 · 볼린저 · RSI. */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const take = (name) => {
  const i = src.indexOf(`function ${name}(`);
  assert.ok(i > 0, `${name} 를 못 찾았다`);
  return src.slice(i, src.indexOf("\n}\n", i) + 2);
};
const fpPatternMarks = eval(`(${take("fpPatternMarks")})`);
const mcUsSession = eval(`(${take("mcUsSession")})`);
const mcBollinger = eval(`(${take("mcBollinger")})`);
const mcRsi = eval(`(${take("mcRsi")})`);

// ── 패턴: 12봉 평평하게 깔고 13번째 봉만 바꾼다 ──
const flat = (d = 1) => Array.from({ length: 12 }, () => ({ high: 101, low: 99, close: 100, delta: d, topBuy: 1, botSell: 1 }));
const kinds = (bars) => fpPatternMarks(bars).map((m) => `${m.i}:${m.kind}:${m.side}`).sort();

// 새 고가인데 누적 CVD 가 직전 고가 때보다 낮다 -> 다이버(위)
let bars = [...flat(), { high: 103, low: 100, close: 102, delta: -50, topBuy: 1, botSell: 1 }];
assert.ok(kinds(bars).includes("12:div:-1"), `다이버 누락: ${kinds(bars)}`);
// 새 고가 + 큰 매수 델타 + 윗꼬리로 닫힘 -> 소진(위). CVD 는 올라가서 다이버는 아니다
bars = [...flat(), { high: 104, low: 100, close: 100.5, delta: 90, topBuy: 1, botSell: 1 }];
assert.deepEqual(kinds(bars), ["12:exh:-1"]);
// 윗꼬리 없이 위에서 닫히면 소진이 아니다
bars = [...flat(), { high: 104, low: 100, close: 103.8, delta: 90, topBuy: 1, botSell: 1 }];
assert.deepEqual(kinds(bars), []);
// 흡수?: 맨 윗줄 매수가 창 상위 10% 이고 다음 봉이 그 고가를 못 넘었다. 다음 봉이 넘으면 없다
const big = { high: 101.5, low: 99, close: 100.5, delta: 1, topBuy: 50, botSell: 1 };   // 새 30분 고가
bars = [...flat(), big, { high: 100.5, low: 99, close: 100, delta: 1, topBuy: 1, botSell: 1 }];
assert.ok(kinds(bars).includes("12:abs:-1"), `흡수 누락: ${kinds(bars)}`);
bars = [...flat(), big, { high: 102, low: 99, close: 101, delta: 1, topBuy: 1, botSell: 1 }];
assert.ok(!kinds(bars).includes("12:abs:-1"), "다음 봉이 넘었는데 흡수로 찍었다");
// 평평한 구간(새 극값 없음)에선 흡수를 안 찍는다 -- 극값 조건이 빠졌던 첫 판의 회귀 방지
assert.deepEqual(kinds([...flat(), ...flat()]), []);
// 마지막 봉은 «다음 봉»이 없어 흡수를 말하지 않는다
assert.ok(!kinds([...flat(), big]).some((k) => k.endsWith("abs:-1")), "다음 봉 없이 흡수를 찍었다");
// 저가 쪽 거울: 새 저가인데 누적 CVD 가 직전 저가 때보다 높다 -> 다이버(아래)
bars = [...flat(), { high: 100, low: 97, close: 98, delta: 50, topBuy: 1, botSell: 1 }];
assert.ok(kinds(bars).includes("12:div:1"), `아래 다이버 누락: ${kinds(bars)}`);
// 판정 창(6봉)보다 짧으면 아무것도 안 찍는다
assert.deepEqual(fpPatternMarks(flat().slice(0, 5)), []);

// ── 미국장: 서머타임 13:30 UTC · 주말 건너뜀 · 표준시 14:30 UTC · 장중 판정 ──
const us = (iso) => { const r = mcUsSession(Date.parse(iso)); return [new Date(r.open).toISOString().slice(0, 16), r.live]; };
assert.deepEqual(us("2026-09-29T12:00:00Z"), ["2026-09-29T13:30", false]);
assert.deepEqual(us("2026-09-29T15:00:00Z"), ["2026-09-29T13:30", true]);
assert.deepEqual(us("2026-10-03T12:00:00Z"), ["2026-10-05T13:30", false]);   // 토요일 -> 월요일
assert.deepEqual(us("2026-12-01T12:00:00Z"), ["2026-12-01T14:30", false]);   // 표준시

// ── 볼린저: 상수 종가면 폭 0, 20봉 전엔 값 없음 ──
const cs = Array.from({ length: 25 }, (_, i) => ({ time: i, close: 100 }));
const bb = mcBollinger(cs);
assert.equal(bb.has(18), false); assert.deepEqual(bb.get(19), [100, 100, 100]);
// ── RSI: 계속 오르면 100, 계속 내리면 0, 모자라면 null ──
assert.equal(mcRsi(Array.from({ length: 30 }, (_, i) => 100 + i)), 100);
assert.equal(mcRsi(Array.from({ length: 30 }, (_, i) => 100 - i)), 0);
assert.equal(mcRsi([1, 2, 3]), null);
console.log("market ctx patterns ok");
