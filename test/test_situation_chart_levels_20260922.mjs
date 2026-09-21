/* 시나리오 가격선 + 확률 원뿔 (2026-09-22). 실행: node test/<이 파일>
 * app.js 에서 함수 본문만 떼어 실제로 돌린다 -- 소스 grep 이 아니라 동작을 잰다.  */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const grab = (sig) => { const i = src.indexOf(sig); assert.ok(i > 0, sig); return src.slice(i, src.indexOf("\n}\n", i) + 2); };
// 상수 셋은 «const CONE_FIT» 부터 다음 function 선언 앞까지 통째로 (줄 끝 주석 때문에 ;$ 로는 못 끊는다)
const ci = src.indexOf("const CONE_FIT");
assert.ok(ci > 0, "CONE_FIT");
const consts = src.slice(ci, src.indexOf("\nfunction ", ci));
const LV = grab("function situationTargetLevels("), CM = grab("function coneModel(");
const run = (body, now, { asset = "eth", footprint = true } = {}) => {
  const latestSituation = now === null ? null : { now };
  const activeSnapshotAsset = asset;
  return eval(consts + "\n" + LV + "\n" + CM + "\n" + body);
};
const TREND = { ok: true, prob: { A: 56, B: 26, C: 18 }, evidence: { range_bp: 47 },
                targets: { A: 2700, B: 2716.15, C: null } };            // C 는 선점됨

// ── 가격선 ──
let r = run("situationTargetLevels(true);", TREND);
assert.equal(r.length, 2, "선점된 C 가 그려졌다");
assert.deepEqual(r.map((x) => x.label), ["56%", "26%"]);
assert.deepEqual(r.map((x) => x.scenario), ["A", "B"]);
assert.ok(r.every((x) => x.marker === true), "풋프린트인데 선으로 그린다");
assert.ok(r.every((x) => !/accent|amber|good|bad|liq-/.test(x.color)), "예약색을 썼다");
assert.ok(run("situationTargetLevels(false);", TREND).every((x) => !x.marker), "청산맵인데 삼각형");
assert.equal(run("situationTargetLevels(true);", TREND, { asset: "btc" }).length, 0, "비ETH");
assert.equal(run("situationTargetLevels(true);", { ok: true, prob: { A: 1, B: 1, C: 1 },
  targets: { A: [2600, 2610], B: 0, C: 2590 } }).length, 1, "배열/0 목표를 걸러야 한다");

// ── 원뿔 ──
assert.equal(run("coneModel(false);", TREND), null, "풋프린트가 아닌데 원뿔을 낸다");
assert.equal(run("coneModel(true);", TREND, { asset: "btc" }), null, "비ETH 에 원뿔");
assert.equal(run("coneModel(true);", { ok: true, evidence: {} }), null, "창폭 없이 원뿔");
const c = run("coneModel(true);", TREND);
assert.equal(c.bands.length, 3);
assert.deepEqual(c.bands.map((b) => b.pct), [50, 70, 90], "안쪽부터 50/70/90");
for (const b of c.bands) {
  assert.equal(b.lo.length, 6); assert.equal(b.hi.length, 6);
  assert.ok(b.lo.every((v) => v < 0) && b.hi.every((v) => v > 0), "아래/위 부호가 틀렸다");
  for (let j = 1; j < 6; j++) assert.ok(b.hi[j] > b.hi[j - 1], "원뿔이 시간에 따라 안 벌어진다");
}
// 중첩: 바깥 밴드가 안쪽을 포함해야 한다
for (let k = 1; k < 3; k++)
  assert.ok(c.bands[k].hi[5] > c.bands[k - 1].hi[5] && c.bands[k].lo[5] < c.bands[k - 1].lo[5], "밴드가 안 겹친다");
// 크기: rng 47bp 에서 30분 90% 밴드 상단이 실측(약 +56bp) 근처여야 한다
assert.ok(Math.abs(c.bands[2].hi[5] - 56) < 8, `90% 상단이 ${c.bands[2].hi[5].toFixed(1)}bp (실측 ~56)`);
// 시장을 따라간다: 창폭이 4배면 원뿔은 4^0.7 ≈ 2.6배
const wide = run("coneModel(true);", { ...TREND, evidence: { range_bp: 188 } });
const ratio = wide.bands[2].hi[5] / c.bands[2].hi[5];
assert.ok(Math.abs(ratio - Math.pow(4, 0.701)) < 0.05, `창폭 4배에 원뿔 ${ratio.toFixed(2)}배 (기대 2.63)`);
console.log("모두 통과 · rng 47bp 30분 90% 밴드 ±%s / %s bp"
  .replace("%s", c.bands[2].hi[5].toFixed(1)).replace("%s", c.bands[2].lo[5].toFixed(1)));
