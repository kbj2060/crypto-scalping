/* 이력 fetch 가 창을 한 칸 밀지 않는다 (2026-09-23). 실행: node test/<이 파일>
 *
 * 왜: 서버는 **마감봉만** 준다. 형성 중 봉은 updateSnapshotCandleLive() 가 클라에서 밀어
 * 넣는다. 이력 fetch 가 배열을 통째로 교체한 뒤 그 복원 없이 렌더가 돌면 slice(-창) 의
 * 오른쪽 끝이 한 봉 왼쪽으로 가고, 맨 과거 봉이 되살아났다가 다음 틱에 사라진다 -- 그게
 * 사용자가 신고한 5분봉 깜빡임이었다. 여기서는 app.js 의 함수 본문을 떼어 실제로 돌린다. */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");

// ── 계약 1: fetch 가 교체 직후 형성봉을 복원한다(호출자가 아니라 여기서) ──────────
const fi = src.indexOf("async function fetchBinanceHistory(");
assert.ok(fi > 0, "fetchBinanceHistory 를 못 찾았다");
const fetchBody = src.slice(fi, src.indexOf("\n}\n", fi));
const assignAt = fetchBody.indexOf("candleHistoryByAsset[asset] =");
// 🔴주석이 아니라 **문장**을 찾는다. 처음엔 indexOf 로 찾았는데 고침을 설명하는 주석 안에
//   같은 이름이 들어 있어서, 호출을 지운 사본에서도 테스트가 통과했다(음성 대조군이 잡았다).
const m = /^[ \t]*updateSnapshotCandleLive\(\);[ \t]*$/m.exec(fetchBody);
const restoreAt = m ? m.index : -1;
assert.ok(restoreAt > assignAt && assignAt > 0,
  "fetchBinanceHistory 가 배열 교체 뒤 updateSnapshotCandleLive() 를 안 부른다 -- 창이 한 칸 밀린다");

// ── 계약 2: 실제로 돌려 본다 -- 교체 전후로 창의 오른쪽 끝이 같아야 한다 ──────────
const ui = src.indexOf("function updateSnapshotCandleLive(");
assert.ok(ui > 0, "updateSnapshotCandleLive 를 못 찾았다");
const updBody = src.slice(ui, src.indexOf("\n}\n", ui) + 2);

const BAR = 300, MAX = 200, WINDOW = 12;   // 5분봉 · CHART_MAX_CANDLES · 1h 창
const T0 = 1790085600;                      // 마감봉 200개의 마지막
const closed = () => Array.from({ length: MAX }, (_, k) => {
  const t = T0 - (MAX - 1 - k) * BAR;
  return { time: t, open: 100, high: 101, low: 99, close: 100, sma: 100 };
});

const env = {
  activeSnapshotAsset: "eth",
  candleHistoryByAsset: { eth: closed() },
  latestLivePriceByAsset: { eth: 100.5 },
  latestLivePriceTsByAsset: { eth: new Date((T0 + BAR + 30) * 1000).toISOString() },
  CHART_CANDLE_MIN: 5,
  CHART_MAX_CANDLES: MAX,
  lastSnapshotHistoryFetchAt: 1,
};
const run = new Function(...Object.keys(env), updBody + "\nupdateSnapshotCandleLive();");
const call = () => run(...Object.keys(env).map((k) => env[k]));

const right = () => {
  const w = env.candleHistoryByAsset.eth.slice(-WINDOW);
  return { last: w[w.length - 1].time, first: w[0].time, n: w.length };
};

call();                                     // 평소 렌더 경로
const before = right();
assert.equal(before.last, T0 + BAR, "형성 중 봉이 안 들어갔다");
assert.equal(before.n, WINDOW);

env.candleHistoryByAsset.eth = closed();    // 이력 fetch 가 마감봉으로 통째로 교체
const naive = right();
assert.equal(naive.last, T0,
  "전제가 깨졌다 -- 교체만 하면 창이 한 칸 밀려야 정상(이 테스트의 의미가 사라진다)");

call();                                     // fetch 안에서 곧바로 복원
const after = right();
assert.deepEqual(after, before, "이력 fetch 뒤 창이 움직였다 -- 5분봉이 깜빡인다");

console.log("OK  이력 fetch 가 창을 밀지 않는다 (오른쪽 끝 %d 고정, %d봉)", after.last, after.n);
