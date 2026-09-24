// /api/stream 여닫기 게이트(app.js ensureLiveStream). `node test/test_live_stream_gate_20260924.mjs`
// 계약: 메시지를 받기 전엔 폴링을 안 쉰다 · 10초 침묵/코인 변경이면 새로 연다 · 에러 뒤 5초 쉰다.
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";

const src = readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const i = src.indexOf("const API_STREAM_URL");
const j = src.indexOf("\n}\n", src.indexOf("function ensureLiveStream")) + 3;
let now = 1_000_000;
const opened = [];
class FakeES {
  constructor(url) { this.url = url; this.h = {}; this.closed = false; opened.push(this); }
  addEventListener(ev, fn) { this.h[ev] = fn; }
  close() { this.closed = true; }
  emit(ev, data) { this.h[ev]({ data: JSON.stringify(data) }); }
}
const env = { activePageTab: "snapshot", activeSnapshotAsset: "eth", hidden: false, applied: 0, sit: 0 };
const api = new Function("env", "EventSource", "Date", `
  let supply1sSince = 5, oi1sSince = 0, liq1sSince = 0, okxSupply1sSince = 0, okxOi1sSince = 0,
      okxLiq1sSince = 0, spotSupply1sSince = 0, latestSituation = null;
  const document = { get hidden() { return env.hidden; } };
  const applySupply1s = () => { env.applied++; }, repaintSupply1sPanel = () => {}, renderSituation = () => { env.sit++; };
  Object.defineProperty(globalThis, "activePageTab", { get: () => env.activePageTab, configurable: true });
  Object.defineProperty(globalThis, "activeSnapshotAsset", { get: () => env.activeSnapshotAsset, configurable: true });
  ${src.slice(i, j)}
  return { ensureLiveStream, liveStreamOn };`)(env, FakeES, { now: () => now });

api.ensureLiveStream();
assert.equal(opened.length, 1);
assert.match(opened[0].url, /^\/api\/stream\?supply=1&since=5&/);
assert.equal(api.liveStreamOn(), false, "메시지 전엔 폴링을 쉬면 안 된다");
opened[0].emit("supply", { seconds: [] });
assert.equal(api.liveStreamOn(), true); assert.equal(env.applied, 1);
opened[0].emit("situation", { computed_at: 1 }); assert.equal(env.sit, 1);
now += 3500; assert.equal(api.liveStreamOn(), false, "3초 조용하면 폴링이 대신한다");
now += 7000; api.ensureLiveStream();                        // 10.5초 침묵 -> 새로 연다
assert.ok(opened[0].closed); assert.equal(opened.length, 2);
env.activeSnapshotAsset = "btc"; api.ensureLiveStream();     // 코인 변경 -> 수급 없이 다시
assert.ok(opened[1].closed); assert.equal(opened[2].url, "/api/stream");
opened[2].onerror(); api.ensureLiveStream();
assert.equal(opened.length, 3, "에러 직후엔 다시 안 연다");
now += 5001; api.ensureLiveStream(); assert.equal(opened.length, 4);
env.hidden = true; api.ensureLiveStream(); assert.ok(opened[3].closed, "숨으면 닫는다");
console.log("ok");
