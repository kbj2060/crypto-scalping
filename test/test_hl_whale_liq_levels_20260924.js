// HL 고래 청산 레벨 고르기 (2026-09-24). 본문을 떼어 실제로 돌린다.   node test/test_hl_whale_liq_levels_20260924.js
const assert = require("assert");
const src = require("fs").readFileSync(__dirname + "/../dashboard/live/app.js", "utf8");
for (const re of [/const HL_WHALE_LIQ_MIN_ETH = .*;/, /const HL_WHALE_LIQ_RANGE = .*;/]) eval(src.match(re)[0].replace("const ", "global."));
eval(src.match(/function hlWhaleLiqLevels\([\s\S]*?\n}\n/)[0].replace("function hlWhaleLiqLevels", "global.lv = function"));
global.activeSnapshotAsset = "eth";
const cl = [[2545, 36607.9, 0, 1], [2600, 500, 0, 3], [2610, 1200, 0, 2],   // 아래: 36.6k 가 가장 큼
            [2700, 0, 2000, 4], [2720, 0, 3100, 2], [3335, 0, 16451, 1]];   // 위: 3335 는 +25% 라 범위 밖
global.latestHlWhaleLiq = { ok: true, age_s: 100, clusters: cl };
let r = lv(2667, true);
assert.deepStrictEqual(r.map((x) => [x.label, x.val, x.sub]), [["HL롱", 2545, "36.6k"], ["HL숏", 2720, "3.1k"]], JSON.stringify(r));
assert.ok(r.every((x) => x.marker === true && x.dashed), "풋프린트에선 삼각형");
global.latestHlWhaleLiq = { ok: true, age_s: 900, clusters: cl };
assert.deepStrictEqual(lv(2667, true), [], "10분 넘게 묵은 스냅샷은 안 그린다");
global.latestHlWhaleLiq = { ok: true, age_s: 100, clusters: [[2600, 500, 0, 3]] };
assert.deepStrictEqual(lv(2667, true), [], "1,000 ETH 미만 뭉치는 꼬리표 없음");
global.activeSnapshotAsset = "btc";
global.latestHlWhaleLiq = { ok: true, age_s: 100, clusters: cl };
assert.deepStrictEqual(lv(2667, true), [], "ETH 전용");
console.log("hl whale liq levels OK");
