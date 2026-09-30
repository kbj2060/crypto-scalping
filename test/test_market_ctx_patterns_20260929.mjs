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
const mcLine = { vsess: false };                                  // mcVwapOf 가 읽는 스위치 -- 직접 eval 이라 이 바인딩을 본다
const mcVwapOf = eval(`(${take("mcVwapOf")})`);

// ── VWAP 기준 스위치: 끄면 하루(UTC 날짜가 seg) · 켜면 시장 세션(개장 초가 seg, 이름) · 없는 쪽은 null ──
const row = { time: 1790640000 + 3600, vwap: 2690, vsd: 3, svwap: 2695, svsd: 1, sstart: 1790640000, sname: "아시아" };
assert.deepEqual(mcVwapOf(row), { vwap: 2690, vsd: 3, seg: Math.floor(row.time / 86400), name: "하루" });
mcLine.vsess = true;
assert.deepEqual(mcVwapOf(row), { vwap: 2695, vsd: 1, seg: 1790640000, name: "아시아" });
assert.equal(mcVwapOf({ time: 1, vwap: 1, vsd: 1 }), null);          // 서버가 아직 세션 VWAP 을 안 실었으면 선을 안 그린다
mcLine.vsess = false;

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
