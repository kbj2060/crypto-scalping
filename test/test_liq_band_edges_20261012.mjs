/* 풋프린트 «가격축 자»(2026-10-12 시안 A) -- 현재가 → 위·아래 첫 보이는 청산 띠 가장자리. 실행: node test/<이 파일>
 * app.js 에서 liqBandEdges 본문만 떼어 실제로 돌린다. */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const i = src.indexOf("function liqBandEdges(");
assert.ok(i > 0, "liqBandEdges 없음");
const liqBandEdges = eval(`(${src.slice(i, src.indexOf("\n}\n", i) + 2)})`);

const B = (price, weightPct) => ({ price, weightPct });
const bins = [B(2500, 0.5), B(2510, 0.05), B(2520, 0.2), B(2540, 0.1), B(2545, 0.3), B(2550, 0.9)];
// clip 0.4 → t: 2500 1.25 · 2510 .125(안 보임) · 2520 .5 · 2540 .25(경계 = 보임) · 2545 .75 · 2550 2.25
let r = liqBandEdges(bins, 0.4, 2, 2530, new Map());
assert.deepEqual(r, { up: 2539, dn: 2521 }, "가장 가까운 보이는 칸의 가장자리(중심 ∓ 반 칸)");
r = liqBandEdges(bins, 0.4, 2, 2530, new Map([[2540, 3], [2520, 5]]));
assert.deepEqual(r, { up: 2544, dn: 2501 }, "밀도 층이 끊은(쓸린) 칸은 건너뛴다");
r = liqBandEdges(bins, 0.4, 2, 2539.5, new Map());
assert.equal(r.up, 2544, "현재가가 칸 안이면 그 칸은 위·아래 어느 쪽도 아니다");
assert.equal(r.dn, 2521);
assert.deepEqual(liqBandEdges(bins, 0.4, 2, 2600, new Map()), { up: null, dn: 2551 }, "위에 띠 없음 → null");
assert.deepEqual(liqBandEdges(bins, 0, 2, 2530, new Map()), { up: null, dn: null }, "clip 0 → 아무것도 없음");
assert.deepEqual(liqBandEdges(undefined, 0.4, 2, 2530, new Map()), { up: null, dn: null }, "칸 없음");
console.log("liq band edges OK");
