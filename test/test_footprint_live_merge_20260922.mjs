/* 풋프린트 «진행 중 봉» 병합 가드 (2026-09-22). 실행: node test/<이 파일>
 * app.js 에서 함수 본문을 떼어 **실제로 돌린다**.
 *
 * 왜: 이 함수는 이름과 달리 합치지 않고 **서버 봉을 통째로 교체**한다. 교체가 정당한
 * 경우는 «클라가 그 봉을 처음부터 봤고, 실제로 셀이 있을 때» 하나뿐이다. 그 조건이
 * 깨졌을 때 봉이 사라지는 게 5분봉 깜빡임이었다(footprintForChart 가 levels.length 0 을
 * 걸러낸다).  */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const i = src.indexOf("function footprintMergeLive(");
assert.ok(i > 0, "함수를 못 찾았다");
const body = src.slice(i, src.indexOf("\n}\n", i) + 2);

const BAR = 1790085600;                       // 봉 시각(초)
const cells = (n) => new Map(Array.from({ length: n }, (_, k) => [5480 + k, [1, 2, 0, 0, 0, 0]]));
const run = (live, okxLive = null) => {
  const footprintLive = { bucket: 0.5, ...live };
  const latestFootprint = { okxLive };          // app.js 에선 전역 -- 2026-09-24 OKX 몫
  const byTime = new Map([[BAR, [[2740, 9, 9, 0, 0, 0, 0]]]]);   // 서버가 준 멀쩡한 봉
  eval(body + "\nfootprintMergeLive(byTime, 0.5);");
  return byTime.get(BAR);
};
const SERVER = [[2740, 9, 9, 0, 0, 0, 0]];
const same = (x) => JSON.stringify(x) === JSON.stringify(SERVER);

// ── 정상: 봉 처음부터 봤고 셀이 있다 -> 클라 값으로 교체한다 ──
const ok = run({ barStart: BAR, since: (BAR - 10) * 1000, cells: cells(3) });
assert.equal(ok.length, 3, "정상인데 교체를 안 했다");
assert.ok(!same(ok), "정상인데 서버 값 그대로다");

// ── 가드 ①: 봉 중간에 붙었다(재연결 포함) -> 서버 값 유지 ──
assert.ok(same(run({ barStart: BAR, since: (BAR + 30) * 1000, cells: cells(3) })),
          "봉 중간에 붙었는데 클라 값으로 덮었다 — 끊겨 있던 구간이 빠진다");

// ── 가드 ②: 셀이 비었다 -> 서버 값 유지(이게 깜빡임의 원인이었다) ──
const empty = run({ barStart: BAR, since: (BAR - 10) * 1000, cells: new Map() });
assert.ok(same(empty), "빈 셀로 서버 봉을 덮었다 — levels 0 이면 화면에서 봉이 사라진다");
assert.ok(empty.length > 0, "봉이 빈 배열이 됐다");

// ── 가드 ③: 클라가 다른 봉을 보고 있다 -> 손대지 않는다 ──
assert.ok(same(run({ barStart: BAR - 300, since: (BAR - 600) * 1000, cells: cells(3) })),
          "다른 봉을 보고 있는데 이 봉을 덮었다");
assert.ok(same(run({ barStart: 0, since: 0, cells: cells(3) })), "barStart 0 인데 덮었다");

// ── 재연결에서 가드 ①이 실제로 살아나는가: since 를 Infinity 로 되돌리는 코드가 있어야 한다 ──
// (없으면 since 가 최초 연결 시각에 고정돼 재연결 뒤 가드가 영원히 안 먹는다)
assert.match(src, /footprintLive\.since = Infinity;/,
             "새 연결에서 since 를 되돌리는 줄이 없다 — 재연결 뒤 가드가 죽는다");

// ── OKX 몫 (2026-09-24): 같은 봉이면 실시간 바이낸스 셀에 **더한다** ──
const live = { barStart: BAR, since: (BAR - 10) * 1000, cells: cells(3) };   // 2740·2740.5·2741 각 [1,2]
const okxLv = { time: BAR, levels: [[2740, 1, 1, 0, 0, 0, 0], [2745, 5, 0, 5, 0, 0, 0]] };
const summed = run(live, okxLv);
assert.deepEqual(summed.find((l) => l[0] === 2740), [2740, 2, 3, 0, 0, 0, 0], "같은 가격 칸에 안 더했다");
assert.deepEqual(summed.find((l) => l[0] === 2745), [2745, 5, 0, 5, 0, 0, 0], "OKX 에만 있는 칸이 빠졌다");
assert.equal(summed.length, 4);
// 렌더마다 불린다 -- 실시간 셀을 고쳐 쓰면 OKX 몫이 렌더 횟수만큼 쌓인다
assert.deepEqual(run(live, okxLv), summed, "두 번째 렌더에서 값이 불었다 — 실시간 셀을 변형했다");
assert.deepEqual(live.cells.get(5480), [1, 2, 0, 0, 0, 0], "실시간 셀이 바뀌었다");
// 다른 봉의 OKX 몫(롤오버 직후 서버가 아직 옛 봉을 줄 때)은 더하지 않는다
assert.equal(run(live, { time: BAR - 300, levels: [[2740, 99, 99, 0, 0, 0, 0]] })
  .find((l) => l[0] === 2740)[1], 1, "다른 봉의 OKX 몫을 더했다");

console.log("모두 통과 (가드 3 + 재연결 리셋 + OKX 합산 3)");
