/* SMA 이력 재시도는 봉당 한 번뿐이다 (2026-09-23 사용자 지시). 실행: node test/<이 파일>
 *
 * 왜: 서버의 마감봉 프레임은 60초 TTL 이라 10초마다 물으면 여섯 번 중 다섯 번이 같은 답이다.
 * 봉당 한 번으로 줄이면 «언제 쏘는가»가 중요해진다 -- 마감 직후는 서버가 아직 안 만들어
 * 한 발을 헛되이 쓴다(전에 «경계에서 한 번»이 실패한 이유). 게이트 본문을 떼어 가짜 시계로
 * 15분을 돌리며 실제 호출 수를 센다. */
import fs from "node:fs";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../dashboard/live/app.js", import.meta.url), "utf8");
const i = src.indexOf("const HISTORY_RETRY_AFTER_CLOSE_S");
assert.ok(i > 0, "HISTORY_RETRY_AFTER_CLOSE_S 를 못 찾았다");
const fi = src.indexOf("async function maybeFetchSnapshotChartHistory(", i);
const block = src.slice(i, src.indexOf("\n}\n", fi) + 2);

const BAR = 300, T0 = 1790000100;   // 봉 경계에 정렬된 시각 (T0 % 300 === 0)

// serverLagS: 마감 뒤 몇 초가 지나야 서버가 그 봉의 sma 를 주는가 (Infinity = 영영 안 준다)
function run(serverLagS, minutes = 15) {
  let clock = (T0 + BAR) * 1000;            // 첫 봉이 막 마감된 직후
  const eth = [{ time: T0, open: 1, high: 1, low: 1, close: 1, sma: 1 },
               { time: T0 + BAR, open: 1, high: 1, low: 1, close: 1 }];   // 형성 중
  const log = [];
  const env = {
    CANDLE_HISTORY_POLL_MS: 300000,
    CHART_CANDLE_MIN: 5,
    candleHistoryByAsset: { eth },
    Date: { now: () => clock },
    fetchBinanceHistory: async () => {
      log.push(clock / 1000);
      // 서버는 «마감 + serverLagS» 를 넘긴 봉에만 sma 를 붙여 준다
      eth.forEach((c, k) => {
        if (k === eth.length - 1) return;                 // 형성 중 봉엔 sma 가 없다
        if (clock / 1000 >= c.time + BAR + serverLagS) c.sma = 1;
      });
    },
  };
  const gate = new Function("env", `
    let lastSnapshotHistoryFetchAt = 0;
    const { CANDLE_HISTORY_POLL_MS, CHART_CANDLE_MIN, candleHistoryByAsset,
            Date, fetchBinanceHistory } = env;
    const activeSnapshotAsset = "eth";
    const scheduleSnapshotChartRender = () => {};
    ${block}
    return { gate: maybeFetchSnapshotChartHistory, retryBar: () => historyRetryDoneBar };
  `)(env);

  const gen = [], retries = [];              // 경계 기록 · 재시도로 쏜 시각
  return (async () => {
    for (let s = 0; s < minutes * 60; s += 1) {
      clock = (T0 + BAR + s) * 1000;
      const ts = Math.floor(clock / 1000 / BAR) * BAR;
      const last = eth[eth.length - 1];
      if (last.time < ts) {                  // updateSnapshotCandleLive 이 하는 일
        eth.push({ time: ts, open: 1, high: 1, low: 1, close: 1 });
        gen.push(ts);
      }
      const before = gate.retryBar();
      await gate.gate();
      if (gate.retryBar() !== before) retries.push(clock / 1000);   // 이번 건 재시도였다
    }
    return { fetches: log, retries, bars: gen.length + 1 };
  })();
}

const perBar = (fetches) => {
  const byBar = new Map();
  fetches.forEach((t) => {
    const b = Math.floor(t / BAR) * BAR;
    byBar.set(b, (byBar.get(b) || 0) + 1);
  });
  return [...byBar.values()];
};

// ── 1. 서버가 안 밀린다: 재시도 0, 정규 폴링(5분)만 ────────────────────────────
// 최초 1회(부트스트랩) + 5분 폴링 3회 = 4. 재시도가 섞이면 이보다 많아진다.
const a = await run(0);
assert.ok(a.fetches.length <= 4,
  `서버가 멀쩡한데 ${a.fetches.length}회 불렀다 -- 부트스트랩1 + 폴링3 = 4 이하여야 한다`);

// ── 2. 서버가 65초 밀린다: 봉마다 정확히 한 번, 그리고 마감+70초 이후에 ────────
const b = await run(65);
assert.ok(b.retries.length >= 2, "재시도가 아예 안 일어났다 -- 테스트가 증상을 못 만들었다");
assert.ok(perBar(b.retries).every((n) => n <= 1),
  `한 봉에서 재시도 ${Math.max(...perBar(b.retries))}회 -- 봉당 한 번이어야 한다`);
b.retries.forEach((t) => {
  const closed = Math.floor(t / BAR) * BAR - BAR;         // 그때 «직전 마감봉»의 open
  assert.ok(t - (closed + BAR) >= 70,
    `마감 후 ${t - closed - BAR}초에 쐈다 -- 서버 TTL(60초) 전이라 헛발이다`);
});

// ── 3. 서버가 영영 안 준다: 그래도 봉당 한 번 (무한 재시도로 돌지 않는다) ──────
const c = await run(Infinity);
assert.ok(perBar(c.retries).every((n) => n <= 1),
  `sma 가 영영 안 와도 봉당 한 번이어야 한다 (최대 ${Math.max(...perBar(c.retries))}회)`);
assert.ok(c.retries.length <= c.bars,
  `15분에 재시도 ${c.retries.length}회 -- 봉 수(${c.bars})를 넘었다`);

console.log("OK  서버 정상: 호출 %d 재시도 %d · 65초 지연: 호출 %d 재시도 %d(봉당 %s) · 영영 없음: 재시도 %d/%d봉",
  a.fetches.length, a.retries.length, b.fetches.length, b.retries.length,
  perBar(b.retries).join(",") || "-", c.retries.length, c.bars);
