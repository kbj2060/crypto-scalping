// 틱 칸 가격선은 지운 선을 안 그린다(2026-11-25 펜 메모). 실행: node test/test_tick_price_lines_20261125.mjs [tick_charts.js 경로]
import fs from "node:fs";
import assert from "node:assert";
const src = fs.readFileSync(process.argv[2] || new URL("../dashboard/live/tick_charts.js", import.meta.url), "utf8");
const lines = (b) => {
  const w = { addEventListener() {}, devicePixelRatio: 1 };
  const doc = { readyState: "loading", addEventListener() {}, getElementById: () => null, querySelectorAll: () => [], documentElement: {} };
  new Function("window", "document", "getComputedStyle", "localStorage", "activeSnapshotAsset", "latestLivePriceByAsset", "snapshotAccountPosition", "entryUsdt", "armedLinesNow", src)(
    w, doc, () => ({ getPropertyValue: () => "#000" }), { getItem: () => null }, "eth", { eth: 1720 }, () => ({ qty: 1 }), () => 1723.5, () => (b ? { b } : null));
  return w.fpTickPriceLines();
};
const B = { tp_level: 1736.36, sl_level: 1712.91 };
assert.deepEqual(lines(B), [["진입", 1723.5], ["익절", 1736.36], ["손절", 1712.91]]);
assert.deepEqual(lines({ ...B, sl_off: { at: 1 } }), [["진입", 1723.5], ["익절", 1736.36]], "지운 손절은 안 그린다");
assert.deepEqual(lines({ ...B, tp_off: { at: 1 } }), [["진입", 1723.5], ["손절", 1712.91]], "지운 익절은 안 그린다");
assert.deepEqual(lines({ ...B, sl_legs: [{ level: 1712 }, { level: 1708 }] }), [["진입", 1723.5], ["익절", 1736.36], ["손절", 1712], ["손절", 1708]], "분할 손절 = 조각마다");
assert.deepEqual(lines({ ...B, sl_off: { at: 1 }, sl_legs: [{ level: 1712 }, { level: 1708 }] }), [["진입", 1723.5], ["익절", 1736.36]]);
assert.deepEqual(lines(null), [["진입", 1723.5]]);
console.log("OK tick price lines");
