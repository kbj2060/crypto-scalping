// 합산 수급 패널의 «거래소 빠짐» 판정 (2026-09-23). 본문을 떼어 실제로 돌린다.
// 예전엔 선물 나이가 늘 0 이고 null(한 번도 안 붙음)은 건너뛰어, 선물이 죽거나 OKX 가 안 붙어도
// 합산 CVD 가 절반짜리로 멀쩡해 보였다.   node test/test_merged_supply_stale_20260923.js
const assert = require("assert");
const src = require("fs").readFileSync(__dirname + "/../dashboard/live/app.js", "utf8");
eval(src.match(/function mergedSupplySrc\(\)[\s\S]*?\n}\n/)[0].replace("function mergedSupplySrc", "global.mss = function"));
Object.assign(global, { supply1s: new Map(), okxSupply1s: new Map(), spotSupply1s: new Map(),
  liq1s: new Map(), okxLiq1s: new Map(), oi1s: new Map(), okxOi1s: new Map() });
const run = (bn, okx, okxAge, spot, spotAge) => {
  Object.assign(global, { supply1sMeta: { now: bn }, okxMeta: { now: okx, tradeAge: okxAge },
    spotMeta: { now: spot, tradeAge: spotAge } });
  return mss();
};
assert.strictEqual(run(700, 699, 1.2, 698, 2).stale, false, "셋 다 살아 있으면 정상");
assert.strictEqual(run(100, 700, 0.2, 700, 0.3).stale, true, "선물이 600초 멈췄다");
const never = run(100, 0, null, 0, null);
assert.ok(never.stale && never.age.includes("OKX/현물"), "한 번도 안 붙은 거래소는 빠짐으로 적는다");
console.log("merged supply stale OK");
