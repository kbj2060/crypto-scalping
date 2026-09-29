/* 시장 맥락 카드·차트 겹침의 순수 함수 (2026-09-29). 실행: node test/<이 파일>
 * app.js 에서 함수 본문을 떼어 **실제로 돌린다** -- 미국장 시각 · 볼린저 · RSI. (패턴 판정은 09-30 표식과 함께 제거) */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const take = (name) => {
  const i = src.indexOf(`function ${name}(`);
  assert.ok(i > 0, `${name} 를 못 찾았다`);
  return src.slice(i, src.indexOf("\n}\n", i) + 2);
};
const mcUsSession = eval(`(${take("mcUsSession")})`);
const mcBollinger = eval(`(${take("mcBollinger")})`);
const mcRsi = eval(`(${take("mcRsi")})`);

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
