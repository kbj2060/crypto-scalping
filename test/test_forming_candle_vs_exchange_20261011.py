"""형성 중 5분봉 OHLC 가 거래소 봉과 같은가 -- app.js 본문을 떼어 node 로 «서버 박자 그대로» 돌린다 (2026-10-11).

사용자 «PC에서 현재 진행 중인 봉이 실제 봉과 달라» · «실시간으로 그리는 중에 새로고침하면 최신 5분봉이 사라져».
서버 모형(server.py load_chart_klines_frames): REST klines 프레임을 60초 캐시(상황 계산이 매초 불러 60초마다 새로 받는다) --
마감봉은 그 프레임 시각까지 닫힌 것만(실측 «닫힌 뒤 ~50초»), 형성 봉은 partial 로 프레임 시각까지의 값(경계 뒤 5초 안엔 REST 에
새 봉 행이 없다). 체결 100ms(바이낸스 @trade = 거래소 봉의 원천) · 렌더 틱 400ms · 폴링 틱 500ms. 매초 화면 형성 봉 vs 참 OHLC.
새로고침 = 화면 상태를 비우고 서버 응답부터 다시(봉 2분 30초 · 경계 뒤 20초 두 번).
허용: 새로고침한 봉의 고저는 «서버 partial 이 찍힌 시각 ~ 새로고침» 사이 체결을 알 길이 없다(캐시 ≤60초) -- 그 봉만 «알 수 있는 체결»
(마지막으로 받은 partial 까지 ∪ 새로고침 뒤)과 대조하고, 참값과의 차이는 출력만 한다. 경계 뒤 20초 새로고침은 프레임에 그 봉 partial 이
아직 없을 수 있어 시가도 «알 수 있는 값» 기준(남은 한계).
    python3 -m pytest -q test/test_forming_candle_vs_exchange_20261011.py      ·   python3 test/... [--old-server]  (형성 봉 안 주는 옛 서버)
"""
import json
import pathlib
import re
import subprocess
import sys

APP = pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "live" / "app.js"
FNS = ("mergeFormingCandle", "historyRetryDue", "fetchBinanceHistory", "maybeFetchSnapshotChartHistory",
       "updateSnapshotCandleLive", "noteLiveTrade")

SIM = r"""
let NOW = 0; Date.now = () => NOW;
const CHART_CANDLE_MIN = 5, CHART_MAX_CANDLES = 200, CANDLE_HISTORY_POLL_MS = 300000;
const HISTORY_RETRY_AFTER_CLOSE_S = 60, HISTORY_RETRY_EVERY_MS = 15000, PARTIAL = __PARTIAL__;
let candleHistoryByAsset, activeSnapshotAsset = "eth", latestLivePriceByAsset, latestLivePriceTsByAsset;
let lastSnapshotHistoryFetchAt, historyRetryAt;
const scheduleSnapshotChartRender = () => {};
const B0 = 1791659400, bar = (s) => Math.floor(s / 300) * 300;
const trades = [];
let p = 2511.4, seed = 7;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
for (let t = (B0 - 1800) * 1000; t < (B0 + 900) * 1000; t += 100) {
  p = Math.round((p + (rnd() - 0.5) * 0.12 + (rnd() < 0.004 ? (rnd() - 0.5) * 3 : 0)) * 100) / 100;   // 가끔 0.1초 스파이크
  trades.push([t, p]);
}
const ohlc = (rows) => rows.length ? { open: rows[0][1], high: Math.max(...rows.map((r) => r[1])), low: Math.min(...rows.map((r) => r[1])), close: rows[rows.length - 1][1] } : null;
const inBar = (b, t0, t1) => trades.filter((r) => bar(r[0] / 1000) === b && r[0] >= t0 && r[0] <= t1);
const PHASE = 23000, seenPartial = new Map();                 // 프레임 갱신 위상 · 봉 -> 화면이 받은 partial 의 프레임 시각(최대)
globalThis.fetch = async () => {
  const ts = NOW - ((NOW - PHASE) % 60000 + 60000) % 60000;   // 60초 캐시 프레임 시각
  const out = [];
  for (let b = B0 - 1800; b + 300 <= ts / 1000; b += 300) out.push({ time: b, ...ohlc(inBar(b, 0, Infinity)), sma: 1 });
  const fb = bar(ts / 1000);
  if (PARTIAL && ts / 1000 - fb >= 5) {                       // REST 는 새 봉 행을 경계 ~5초 뒤에 붙인다
    out.push({ time: fb, ...ohlc(inBar(fb, 0, ts)), partial: true });
    seenPartial.set(fb, Math.max(seenPartial.get(fb) || 0, ts));
  }
  return { ok: true, json: async () => ({ candles: out.map((c) => ({ ...c })) }) };
};
const RELOADS = [(B0 + 150) * 1000, (B0 + 320) * 1000];
async function load() {                                       // 새 페이지: 상태 없음 → 첫 이력 요청
  candleHistoryByAsset = {}; latestLivePriceByAsset = {}; latestLivePriceTsByAsset = {};
  lastSnapshotHistoryFetchAt = 0; historyRetryAt = 0;
  await fetchBinanceHistory("eth");
  lastSnapshotHistoryFetchAt = NOW;
}
(async () => {
  NOW = (B0 - 600) * 1000 - 100;                              // 경계 직전에 연 페이지
  await load();
  let k = 0, R = NOW;
  const out = [];
  for (; NOW < (B0 + 900) * 1000 - 100; NOW += 100) {
    if (RELOADS.includes(NOW)) { R = NOW; await load(); }
    while (k < trades.length && trades[k][0] <= NOW) {
      const [t, px] = trades[k++];
      if (typeof noteLiveTrade === "function") noteLiveTrade("eth", px, t);
      else latestLivePriceByAsset.eth = px;                   // 옛 WS 경로: 가격만(봉 반영은 렌더 틱에서)
    }
    if (NOW % 500 === 0) await maybeFetchSnapshotChartHistory();
    if (NOW % 400 === 0) updateSnapshotCandleLive();          // maybeRenderSnapshotChartNow 400ms 게이트
    latestLivePriceTsByAsset.eth = new Date(NOW - 700).toISOString();   // SSE 시세 시각(1초 주기·지연)
    if (NOW % 1000 === 0 && NOW / 1000 >= B0 - 600 + 2) {
      const cs = candleHistoryByAsset.eth, last = cs[cs.length - 1], b = bar(NOW / 1000);
      const truth = ohlc(inBar(b, 0, NOW));
      let want = truth, reload = false;
      if (b === bar(R / 1000) && R > (B0 - 600) * 1000) {    // 새로고침한 봉: 알 수 있는 체결만
        reload = true;
        const snap = seenPartial.get(b) || 0;
        want = ohlc([...inBar(b, 0, snap), ...inBar(b, R, NOW)]);
      }
      const gap = cs.slice(-3).some((c, i, a) => i && c.time - a[i - 1].time !== 300);
      const f = ["open", "high", "low", "close"];
      out.push({ s: NOW / 1000 - B0, t: last.time, want: b, gap, reload, seeded: seenPartial.has(b),
                 d: f.map((x) => +(last[x] - want[x]).toFixed(2)), vsTruth: f.map((x) => +(last[x] - truth[x]).toFixed(2)) });
    }
  }
  console.log(JSON.stringify(out));
})();
"""


def run(partial=True, app=None):
    src = (app or APP).read_text("utf-8")
    body = ""
    for n in FNS:
        m = re.search(rf"^(async )?function {n}\(.*?^}}\n", src, re.S | re.M)
        if m:
            body += m.group(0)
    js = body + SIM.replace("__PARTIAL__", "true" if partial else "false")
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


def bad(rows):
    return [r for r in rows if r["t"] != r["want"] or r["gap"] or any(r["d"])]


def test_forming_bar_matches_exchange_every_second():
    rows = run()
    assert len(rows) > 1400
    assert not bad(rows), f"{len(bad(rows))}/{len(rows)}초 어긋남 · 첫 5개 {bad(rows)[:5]}"


def test_reload_mid_bar_keeps_open_and_extremes():
    rows = run()
    mid = [r for r in rows if r["reload"] and 150 <= r["s"] < 300]   # 봉 2분 30초 새로고침 → 서버 partial 이 씨앗
    assert mid and all(r["seeded"] for r in mid)
    assert all(r["vsTruth"][0] == 0 for r in mid), "새로고침 뒤 시가가 참 시가와 다르다"
    assert not bad(mid)


if __name__ == "__main__":
    rows = run(partial="--old-server" not in sys.argv)
    b = bad(rows)
    mid = [r for r in rows if r["reload"] and 150 <= r["s"] < 300]
    print(len(b), "/", len(rows), "초 어긋남 · 2분30초 새로고침 봉 시가 참값과 다름",
          sum(r["vsTruth"][0] != 0 for r in mid), "/", len(mid), "초 · 새로고침 봉 참값 대비 최대 |O,H,L,C|",
          [max(abs(r["vsTruth"][i]) for r in rows if r["reload"]) for i in range(4)])
    for r in b[:: max(1, len(b) // 15)]:
        print(r)
