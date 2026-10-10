"""형성 중 5분봉 OHLC 가 거래소 봉과 같은가 -- app.js 본문을 떼어 node 로 «서버 박자 그대로» 돌린다 (2026-10-11).

사용자 «PC에서 현재 진행 중인 봉이 실제 봉과 달라». 서버 실측(10-10 19:20 UTC): /api/market-history 는 마감봉만 주고
그 마감봉도 닫힌 뒤 ~50초에야 붙는다. 화면의 형성 봉은 클라이언트가 라이브 가격으로 만든다.
시뮬: 체결 100ms 마다(바이낸스 @trade 와 같은 원천 = 거래소 봉의 정의) · 렌더 틱 400ms · 폴링 틱 500ms ·
서버는 «닫힌 지 50초 지난 봉»만 준다. 매초 화면 형성 봉과 그때까지의 참 OHLC 를 대조한다.
    python3 -m pytest -q test/test_forming_candle_vs_exchange_20261011.py
"""
import json
import pathlib
import re
import subprocess

APP = pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "live" / "app.js"
FNS = ("mergeFormingCandle", "historyRetryDue", "fetchBinanceHistory", "maybeFetchSnapshotChartHistory",
       "updateSnapshotCandleLive", "noteLiveTrade")

SIM = r"""
let NOW = 0; Date.now = () => NOW;
const CHART_CANDLE_MIN = 5, CHART_MAX_CANDLES = 200, CANDLE_HISTORY_POLL_MS = 300000;
const HISTORY_RETRY_AFTER_CLOSE_S = 60, HISTORY_RETRY_EVERY_MS = 15000, LAG_S = 50;
let candleHistoryByAsset = {}, activeSnapshotAsset = "eth", latestLivePriceByAsset = {}, latestLivePriceTsByAsset = {};
let lastSnapshotHistoryFetchAt = 0, historyRetryAt = 0;
const scheduleSnapshotChartRender = () => {};
const B0 = 1791659400, bar = (s) => Math.floor(s / 300) * 300;
const trades = [], truth = new Map();                       // 참: 체결로 만든 봉(거래소 klines 정의)
let p = 2511.4, seed = 7;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
for (let t = (B0 - 1800) * 1000; t < (B0 + 900) * 1000; t += 100) {
  p = Math.round((p + (rnd() - 0.5) * 0.12 + (rnd() < 0.004 ? (rnd() - 0.5) * 3 : 0)) * 100) / 100;   // 가끔 0.1초 스파이크
  trades.push([t, p]);
  const b = bar(t / 1000), o = truth.get(b);
  if (!o) truth.set(b, { time: b, open: p, high: p, low: p, close: p, sma: 1 });
  else { o.high = Math.max(o.high, p); o.low = Math.min(o.low, p); o.close = p; }
}
// 서버: 마감봉만 · 닫힌 지 LAG_S 초 지나야 붙는다
globalThis.fetch = async () => ({ ok: true, json: async () => ({ candles:
  [...truth.values()].filter((c) => c.time + 300 + LAG_S <= NOW / 1000).map((c) => ({ ...c })) }) });
(async () => {
  NOW = (B0 - 600) * 1000 - 100;                             // 경계 직전에 열린 페이지(봉 중간 로드는 시가를 알 길이 없다 -- 대상 밖)
  await fetchBinanceHistory("eth");
  lastSnapshotHistoryFetchAt = NOW;
  let k = 0, out = [];
  for (; NOW < (B0 + 900) * 1000 - 100; NOW += 100) {
    while (k < trades.length && trades[k][0] <= NOW) {
      const [t, px] = trades[k++];
      if (typeof noteLiveTrade === "function") noteLiveTrade("eth", px, t);
      else latestLivePriceByAsset.eth = px;                   // 옛 WS 경로: 가격만(봉 반영은 렌더 틱에서)
    }
    if (NOW % 500 === 0) await maybeFetchSnapshotChartHistory();
    if (NOW % 400 === 0) updateSnapshotCandleLive();          // maybeRenderSnapshotChartNow 400ms 게이트
    latestLivePriceTsByAsset.eth = new Date(NOW - 700).toISOString();   // SSE 시세 시각(1초 주기·지연)
    if (NOW % 1000 === 0 && NOW / 1000 >= B0 - 600 + 2) {
      const cs = candleHistoryByAsset.eth, last = cs[cs.length - 1], tb = truth.get(bar(NOW / 1000));
      const ref = { ...tb };                                  // 지금까지의 참(봉 시작 ~ NOW)
      const seen = trades.filter((r) => bar(r[0] / 1000) === tb.time && r[0] <= NOW).map((r) => r[1]);
      Object.assign(ref, { open: seen[0], high: Math.max(...seen), low: Math.min(...seen), close: seen[seen.length - 1] });
      const gap = cs.slice(-3).some((c, i, a) => i && c.time - a[i - 1].time !== 300);
      out.push({ s: NOW / 1000, t: last.time, want: ref.time, gap, d: ["open", "high", "low", "close"].map((f) => +(last[f] - ref[f]).toFixed(2)) });
    }
  }
  console.log(JSON.stringify(out));
})();
"""


def run():
    src = APP.read_text("utf-8")
    body = ""
    for n in FNS:
        m = re.search(rf"^(async )?function {n}\(.*?^}}\n", src, re.S | re.M)
        if m:
            body += m.group(0)
    return json.loads(subprocess.run(["node", "-e", body + SIM], capture_output=True, text=True, check=True).stdout)


def test_forming_bar_matches_exchange_every_second():
    rows = run()
    bad = [r for r in rows if r["t"] != r["want"] or r["gap"] or any(r["d"])]
    assert len(rows) > 1400
    assert not bad, f"{len(bad)}/{len(rows)}초 어긋남 · 첫 5개 {bad[:5]}"


if __name__ == "__main__":
    rows = run()
    bad = [r for r in rows if r["t"] != r["want"] or r["gap"] or any(r["d"])]
    print(len(bad), "/", len(rows), "초 어긋남")
    for r in bad[:: max(1, len(bad) // 25)]:
        print(r)
