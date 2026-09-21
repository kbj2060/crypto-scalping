/* 상황 시나리오 목표를 풋프린트 차트 가격선으로 (2026-09-22).
 * 실행: node test/test_situation_chart_levels_20260922.mjs
 * app.js 에서 함수 본문만 떼어 실제로 돌린다 -- 소스 grep 이 아니라 동작을 잰다.
 * 지키는 계약 넷: ①선점(null)·옛 띠(배열) 목표는 안 그린다 ②비ETH 에서는 아무것도 안 그린다
 * ③풋프린트면 선이 아니라 삼각형(가로선이 셀 숫자를 덮는다) ④색은 예약색이 아니어야 한다.  */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const i = src.indexOf("function situationTargetLevels(");
assert.ok(i > 0, "situationTargetLevels 를 못 찾았다");
const fn = src.slice(i, src.indexOf("\n}\n", i) + 2);

const call = (now, { asset = "eth", footprint = true } = {}) => {
  const latestSituation = now === null ? null : { now };
  const activeSnapshotAsset = asset;
  return eval(fn + "\nsituationTargetLevels(" + footprint + ");");
};
const TREND = { ok: true, prob: { A: 56, B: 26, C: 18 },
                targets: { A: 2700, B: 2716.15, C: null } };   // C 는 선점됨

let r = call(TREND);
assert.equal(r.length, 2, "선점된 C 가 그려졌다");
assert.deepEqual(r.map((x) => x.label), ["56%", "26%"]);
assert.ok(r.every((x) => x.marker === true), "풋프린트인데 선으로 그린다");
assert.ok(r[0].width > r[1].width, "굵기가 확률을 안 따른다");

assert.ok(call(TREND, { footprint: false }).every((x) => !x.marker), "청산맵 모드인데 삼각형이다");
assert.equal(call(TREND, { asset: "btc" }).length, 0, "비ETH 에 ETH 시나리오를 그린다");
assert.equal(call(null).length, 0, "페이로드가 없는데 그린다");
assert.equal(call({ ok: false }).length, 0, "ok=false 인데 그린다");

// 옛 장부의 «띠»(배열) 목표
assert.equal(call({ ok: true, prob: { A: 40, B: 30, C: 30 },
                    targets: { A: [2600, 2610], B: 0, C: 2590 } }).length, 1, "배열/0 목표를 걸러야 한다");

// 예약색 금지 -- 초록/빨강은 청산 S/R, accent 는 현재가, amber 는 진입가다
for (const l of call(TREND)) {
  assert.ok(!/accent|amber|good|bad|liq-/.test(l.color), `예약색을 썼다: ${l.color}`);
}
console.log("모두 통과");
