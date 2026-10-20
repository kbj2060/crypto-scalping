/* 4시간봉 히트맵 요청 나누기(2026-10-20 리뷰 11) -- 서버 h4_heat_payload 는 한 요청 60칸(t1 − t0 ≤ 60 × 4h)까지. 실행: node test/<이 파일>
 * fp_daily.js 에서 H4_HEAT_SPAN_MS · heatChunks 본문만 떼어 실제로 돌린다. */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/fp_daily.js", import.meta.url), "utf8");
const i = src.indexOf("var H4_HEAT_SPAN_MS");
assert.ok(i > 0, "heatChunks 없음");
const heatChunks = new Function(src.slice(i, src.indexOf("\n  }\n", src.indexOf("function heatChunks(", i)) + 4) + "; return heatChunks;")();

const key = (n) => new Date(Date.UTC(2026, 9, 1) + n * 4 * 3600e3).toISOString().slice(0, 13);   // «2026-10-01T04»
const keys = Array.from({ length: 130 }, (_, n) => key(n));
const parts = heatChunks(keys, true);
const span = (p) => (Date.parse(p[p.length - 1] + ":00:00Z") - Date.parse(p[0] + ":00:00Z")) / (4 * 3600e3);
assert.deepEqual(parts.map((p) => p.length), [61, 61, 8], "61칸(폭 60) 씩 -- 서버 경계 그대로");
assert.ok(parts.every((p) => span(p) <= 60), "모든 요청 폭 ≤ 60칸");
assert.deepEqual(parts.flat(), keys, "빠짐·겹침 없음");
assert.deepEqual(heatChunks([key(0), key(5), key(70)], true), [[key(0), key(5)], [key(70)]], "빈 칸(캐시 있음)을 건너뛰어도 시각 폭으로 자른다");
assert.deepEqual(heatChunks(["2020-01-01", "2026-10-20"], false), [["2020-01-01", "2026-10-20"]], "일봉 = 한 번에");
assert.deepEqual(heatChunks([], true), [], "빈 목록");
assert.deepEqual(heatChunks([], false), [], "빈 목록(일봉)");
console.log("fp daily heat chunks OK");
