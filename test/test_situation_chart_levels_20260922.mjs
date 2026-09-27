/* 30분 카드 차트 가격선. 실행: node test/<이 파일>
 * app.js 에서 함수 본문만 떼어 실제로 돌린다 -- 소스 grep 이 아니라 동작을 잰다.
 * 2026-09-28 대상 교체: 옛 A/B/C 목표선(situationTargetLevels, 09-25 카드 폐기 뒤 이미 깨져 있던 테스트) →
 *   일 단위 추세의 뒤집힘 가격선(trendFlipLevels). 보이는 캔들 범위 안의 선만 · ETH 만 · 풋프린트면 삼각형. */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const grab = (sig) => { const i = src.indexOf(sig); assert.ok(i > 0, sig); return src.slice(i, src.indexOf("\n}\n", i) + 2); };
const LV = grab("function trendFlipLevels(");
const run = (body, trend, { asset = "eth" } = {}) => {
  const latestTrend = trend;
  const activeSnapshotAsset = asset;
  return eval(LV + "\n" + body);
};
const CANDLES = [{ low: 2400, high: 2480 }, { low: 2420, high: 2520 }];
const TREND = { ok: true, votes: [
  { L: 7, up: false, flip_price: 2443 }, { L: 14, up: true, flip_price: 1918 }, { L: 28, up: true, flip_price: 2510 },
  { L: 56, up: true, flip_price: 1771 }, { L: 90, up: true, flip_price: 2600 }] };

const r = run("trendFlipLevels(true, CANDLES);", TREND);
assert.deepEqual(r.map((x) => x.val), [2443, 2510], "보이는 범위(2400~2520) 안의 선만");
assert.deepEqual(r.map((x) => x.label), ["추세7일", "추세28일"]);
assert.ok(r.every((x) => x.marker && x.dashed && x.priceLeft), "풋프린트면 삼각형 · 점선 · 왼쪽 가격");
assert.ok(run("trendFlipLevels(false, CANDLES);", TREND).every((x) => !x.marker), "청산맵이면 삼각형 없음");
assert.equal(run("trendFlipLevels(true, CANDLES);", TREND, { asset: "btc" }).length, 0, "비ETH");
assert.equal(run("trendFlipLevels(true, CANDLES);", { ok: false, reason: "x" }).length, 0, "추세 없음");
assert.equal(run("trendFlipLevels(true, CANDLES);", null).length, 0, "아직 안 받음");
assert.equal(run("trendFlipLevels(true, []);", TREND).length, 0, "캔들 없음");
console.log("trend flip levels OK");
