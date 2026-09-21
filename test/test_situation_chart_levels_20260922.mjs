/* 시나리오 가격선 (2026-09-22). 실행: node test/<이 파일>
 * app.js 에서 함수 본문만 떼어 실제로 돌린다 -- 소스 grep 이 아니라 동작을 잰다.  */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const grab = (sig) => { const i = src.indexOf(sig); assert.ok(i > 0, sig); return src.slice(i, src.indexOf("\n}\n", i) + 2); };
const LV = grab("function situationTargetLevels(");
const run = (body, now, { asset = "eth", footprint = true } = {}) => {
  const latestSituation = now === null ? null : { now };
  const activeSnapshotAsset = asset;
  return eval(LV + "\n" + body);
};
const TREND = { ok: true, prob: { A: 56, B: 26, C: 18 }, evidence: { range_bp: 47 },
                targets: { A: 2700, B: 2716.15, C: null } };            // C 는 선점됨

// ── 가격선 ──
let r = run("situationTargetLevels(true);", TREND);
assert.equal(r.length, 2, "선점된 C 가 그려졌다");
// 확률은 왼쪽 라벨이 아니라 **오른쪽 배지 안**(sub)에 들어간다 -- 왼쪽은 비운다
assert.deepEqual(r.map((x) => x.sub), ["56%", "26%"]);
assert.ok(r.every((x) => x.label === ""), "왼쪽 라벨이 남아 있다");
assert.deepEqual(r.map((x) => x.scenario), ["A", "B"]);
assert.ok(r.every((x) => x.marker === true), "풋프린트인데 선으로 그린다");
assert.ok(r.every((x) => !/accent|amber|good|bad|liq-/.test(x.color)), "예약색을 썼다");
assert.ok(run("situationTargetLevels(false);", TREND).every((x) => !x.marker), "청산맵인데 삼각형");
assert.equal(run("situationTargetLevels(true);", TREND, { asset: "btc" }).length, 0, "비ETH");
assert.equal(run("situationTargetLevels(true);", { ok: true, prob: { A: 1, B: 1, C: 1 },
  targets: { A: [2600, 2610], B: 0, C: 2590 } }).length, 1, "배열/0 목표를 걸러야 한다");

console.log("모두 통과");
