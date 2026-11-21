// rtcore /rt 그림자 재기(app.js rtSeen·rtMeasure·rtSse·rtReport). `node test/test_rt_shadow_measure_20261121.mjs`
// 입력 = rtcore 정답지(rtcore/tests/fixtures/supply1s_eth.json -- 원본 supply_1s_payload 의 seconds 행)를 rt 메시지 모양(proto.rs Stamped:
// type "snapshot"(supply.seconds) · "supply1s"(seconds) · ex_ms)과 SSE supply 모양({seconds}) 양쪽에 넣는다.
// 계약: 같은 초를 둘 다 받은 지 10초 지나면 한 번 견준다 · 값이 다르면 어긋남 · 먼저 받은 차(SSE − rt) · 지연 = 받은 시각 − ex_ms(스냅숏 빼고).
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";

const src = readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const i = src.indexOf("const rtM = {");
const j = src.indexOf("function ensureRtStream");
assert.ok(i > 0 && j > i, "app.js 에서 재기 함수를 못 찾았다");
const api = new Function(`${src.slice(i, j)}\nreturn { rtM, rtMeasure, rtSse, rtReport };`)();

const fx = JSON.parse(readFileSync(new URL("../rtcore/tests/fixtures/supply1s_eth.json", import.meta.url), "utf8"));
const rows = fx.final.seconds;
assert.ok(rows.length > 100 && rows[0].length === 8, "정답지 seconds 행 모양");
const m = api.rtM;
const T = 1_763_700_000_000;

// rt: 스냅숏(앞 50초) + 밀기(한 초씩 · 거래소 시각 → 받은 시각 = 30ms) · SSE: 같은 초를 120ms 늦게
api.rtMeasure(m, { s: "ethusdt", seq: 1, ex_ms: T - 5, type: "snapshot", supply: { seconds: rows.slice(0, 50) } }, T);
for (let k = 50; k < 100; k++) api.rtMeasure(m, { s: "ethusdt", seq: k, ex_ms: T + k, type: "supply1s", seconds: [rows[k]], partial: {} }, T + k + 30);
api.rtSse(m, { seconds: rows.slice(0, 50) }, T + 120);
for (let k = 50; k < 100; k++) api.rtSse(m, { seconds: [rows[k]] }, T + k + 150);
api.rtMeasure(m, { s: "ethusdt", seq: 200, ex_ms: T, type: "trade", id: 1, px: 1, qty: 1, sell: false }, T + 7);   // 체결도 지연에 든다
api.rtMeasure(m, "망가진 것", T);                                                                                  // 무시

assert.deepEqual(api.rtReport(m, T + 5000), { rt_lat_n: 51, rt_lat_med_ms: 30, rt_lat_p99_ms: 30 }, "10초 전엔 견주지 않는다");
let r = api.rtReport(m, T + 20000);
assert.equal(r.rt_cmp_n, 100); assert.equal(r.rt_bad_n, 0, "같은 정답지 값이면 어긋남 0");
assert.equal(r.rt_lead_n, 100); assert.equal(r.rt_lead_med_ms, 120, "SSE 가 120ms 늦게 받았다");
assert.equal(m.rt.size + m.sse.size, 0, "견준 초는 지운다");

// 양성 대조: 수급 한 초를 일부러 다르게(총매수 +0.001) → 어긋남 1
const bad = rows[120].slice(); bad[5] = Math.round((bad[5] + 0.001) * 1000) / 1000;
api.rtMeasure(m, { ex_ms: T, type: "supply1s", seconds: [rows[110], bad, rows[130]] }, T);
api.rtSse(m, { seconds: [rows[110], rows[120], rows[130]] }, T);
r = api.rtReport(m, T + 11000);
assert.equal(r.rt_cmp_n, 3); assert.equal(r.rt_bad_n, 1, "한 초 다른 값을 어긋남으로 센다");

// 한쪽에만 온 초는 견주지 않고 2분 뒤 버린다 · 늦은 체결로 같은 초가 다시 오면 값만 갈고 첫 도착 시각은 둔다
api.rtMeasure(m, { ex_ms: T, type: "supply1s", seconds: [rows[140]] }, T);
api.rtMeasure(m, { ex_ms: T, type: "supply1s", seconds: [rows[141]] }, T);
api.rtMeasure(m, { ex_ms: T, type: "supply1s", seconds: [rows[141]] }, T + 900);
api.rtSse(m, { seconds: [rows[141]] }, T + 1000);
r = api.rtReport(m, T + 12000);
assert.equal(r.rt_cmp_n, 1); assert.equal(r.rt_lead_med_ms, 1000, "첫 도착 시각 기준");
assert.equal(m.rt.size, 1); api.rtReport(m, T + 130000); assert.equal(m.rt.size, 0, "2분 넘게 한쪽에만 있으면 버림");
assert.deepEqual(api.rtReport(m, T + 140000), {}, "빈 보고는 칸이 없다(M0 가 안 보낸다)");
console.log(`rt 그림자 재기 ok -- 정답지 ${rows.length}초 중 100초 견줌 · 양성 대조(한 초 다름) 어긋남 1 잡음`);
