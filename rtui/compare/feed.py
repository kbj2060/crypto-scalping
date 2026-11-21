"""자료 흐름 일치: 같은 «서버 응답 열 + 체결 열»을 지금 화면의 실제 함수(헤드리스 크로미움 안 app.js -- refreshFootprint · fetchBinanceHistory ·
ws.onmessage(noteLiveTrade · updateLivePriceFast · footprintLiveAdd · maybeRenderSnapshotChartNow) · ensurePriceWs · footprintForChart/footprintMergeLive ·
tfFootprint · tfFpBars · chartTfFull · historyRetryDue)와 rtui feed::Feed(examples/feed_replay.rs)에 넣고 단계마다 봉·칸 상태를 맞댄다.
가짜로 바꾼 것: 시계(Date = 시나리오 시각) · fetch(시나리오 응답 · 그 밖은 404) · WebSocket(죽은 가짜 -- onmessage 를 직접 부른다) · setInterval(끔 -- tick 은 페이지 열 때 한 번).
python -I rtui/compare/feed.py <worktree>   (먼저 cargo build --release --example feed_replay)  →  rtui/compare/feed.md"""
import json, math, pathlib, random, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import INIT, LAUNCH, TZ, make_route

WT = pathlib.Path(sys.argv[1]).resolve()
HERE = WT / "rtui/compare"
R = random.Random(20261124)
B0, B1, B2 = 1795155900, 1795156200, 1795156500   # 형성 봉 셋(5분)
LOAD = (B0 + 140) * 1000                           # 페이지를 연 순간 = B0 중간 → 소켓도 봉 중간에 붙는다
META = {"symbol": "ETHUSDT", "bucket": 0.5, "barSeconds": 300, "barsExpected": 144, "ready": True, "whaleMinUsd": 100000.0, "retailMaxUsd": 10000.0}


def lv(lo, hi):
    out, k = [], round(lo * 2) / 2
    while k <= hi + 1e-9:
        b, s = round(R.lognormvariate(3, 1), 3), round(R.lognormvariate(3, 1), 3)
        out.append([k, b, s, round(b * .2, 3), round(s * .3, 3), round(b * .1, 3), round(s * .1, 3)]); k += 0.5
    return out


def fp(bars, full=False):
    return json.dumps({**META, "full": full, "bars": bars})


def bar(t, lo, hi, dhl=True):
    b = {"time": t, "levels": lv(lo, hi)}
    if dhl: b["dhl"] = [-120.5, 300.25, 88.0]
    return b


def cdl(t, o, h, l, c, **kw):
    return {"time": t, "open": o, "high": h, "low": l, "close": c, "volume": 10.0, **kw}


ev, tid, px = [], 1000, 1745.3
def trades(t0_ms, t1_ms, n, whale_every=0):
    """체결 n 개(시각 오름) -- 같은 가격·연속 체결 ID·100ms 안이면 한 주문(TakerOrderAggregator) · 가끔 가격 0(바이낸스가 섞어 보낸다) · whale_every 마다 고래 주문"""
    global tid, px
    ts = sorted(R.uniform(t0_ms, t1_ms) for _ in range(n))
    for j, t in enumerate(ts):
        t = int(t)
        if R.random() < 0.6: px = round(px + R.choice((-0.01, 0.01, -0.25, 0.25, -0.5, 0.5)) * R.random() * 2, 2)
        q = round(R.lognormvariate(-1, 1.2), 3)
        if whale_every and j % whale_every == 0: q = 70.0 + j % 5
        tid += 1 if R.random() < 0.8 else 3
        p = "0" if R.random() < 0.004 else f"{px:.2f}"
        ev.append({"k": "trade", "now": t + 40, "msg": json.dumps({"e": "trade", "E": t + 5, "T": t, "s": "ETHUSDT", "t": tid, "p": p, "q": f"{q}", "m": R.random() < 0.5})})


D = lambda tf=1: ev.append({"k": "dump", "tf": tf})
# 1 첫 응답(전량) + 캔들 이력(형성 봉 partial · 고가가 라이브보다 낮다)
ev.append({"k": "hist", "now": LOAD + 300, "body": json.dumps({"candles": [cdl(B0 - 300 * i, 1744, 1746, 1743, 1745, sma=1740.0) for i in range(30, 0, -1)] + [cdl(B0, 1745, 1745.6, 1744.9, 1745.2, partial=True)]})})
ev.append({"k": "fp", "now": LOAD + 500, "body": fp([bar(B0 - 300 * i, 1743, 1746) for i in range(8, 0, -1)] + [bar(B0, 1744.5, 1746)], full=True)})
D()
# 2 봉 중간 접속: B0 체결은 칸에 쌓이지만 서버 값을 쓴다(since > 봉 시작)
trades(LOAD + 600, (B0 + 299) * 1000, 120, whale_every=37)
D()
ev.append({"k": "fp", "now": (B0 + 299) * 1000 + 100, "body": fp([bar(B0, 1744, 1746.5)])})
D()
# 3 봉 경계를 걸친 체결: 경계 직후 B1 체결 · 늦게 온 B0 체결 하나(칸을 B0 으로 되돌린다 -- 지금 화면 규칙 그대로) · 다시 B1
trades(B1 * 1000 + 1, B1 * 1000 + 900, 30)
ev.append({"k": "trade", "now": B1 * 1000 + 950, "msg": json.dumps({"e": "trade", "E": B1 * 1000 + 940, "T": B1 * 1000 - 20, "t": tid + 1, "p": f"{px:.2f}", "q": "0.5", "m": True})}); tid += 1
trades(B1 * 1000 + 1000, B1 * 1000 + 60_000, 150, whale_every=41)
D()
# 4 서버 응답이 형성 봉(B1)을 실어 온다 → 브라우저 값으로 바꿔 낀다(소켓이 B1 열리기 전부터 붙어 있었다) · 30m·1h 묶기도
ev.append({"k": "fp", "now": B1 * 1000 + 60_500, "body": fp([bar(B0, 1744, 1746.5), bar(B1, 1744, 1747)])})
D(); D(6); D(12)
# 5 since 증분이 빈 응답
ev.append({"k": "fp", "now": B1 * 1000 + 61_000, "body": fp([])})
D()
# 6 재연결(봉 중간) → since 를 비운다 → B1 은 다시 서버 값
ev.append({"k": "reconnect", "now": B1 * 1000 + 90_000})
trades(B1 * 1000 + 90_100, B1 * 1000 + 200_000, 120, whale_every=29)
ev.append({"k": "fp", "now": B1 * 1000 + 200_500, "body": fp([bar(B1, 1743.5, 1747.5)])})
D()
# 7 새 봉 B2: 재연결 뒤 처음 열린 봉 = 브라우저 값 · 이력 다시(형성 봉 partial · 마감 봉 sma 없음)
trades(B2 * 1000 + 5, B2 * 1000 + 30_000, 120, whale_every=23)
ev.append({"k": "fp", "now": B2 * 1000 + 30_500, "body": fp([bar(B1, 1743.5, 1747.5), bar(B2, 1744, 1748, dhl=False)])})
ev.append({"k": "hist", "now": B2 * 1000 + 31_000, "body": json.dumps({"candles": [cdl(B0 - 300 * i, 1744, 1746, 1743, 1745, sma=1740.0) for i in range(29, -1, -1)] + [cdl(B1, 1745, 1746, 1744, 1745.5), cdl(B2, 1745.5, 1745.9, 1745.1, 1745.4, partial=True)]})})
D(); D(6)
# 8 두 봉 넘게 묵음 → 전량(since=0)
ev.append({"k": "fp", "now": (B2 + 1000) * 1000, "body": fp([bar(B1, 1743.5, 1747.5), bar(B2, 1744, 1748)], full=True)})
D()
# 9 historyRetryDue 경계 사례 [마감 봉 시각, sma, now, 지난 재시도]
ev.append({"k": "retry", "cases": [[B1, None, (B1 + 300 + 59) * 1000, 0], [B1, None, (B1 + 300 + 60) * 1000, 0], [B1, 1740.0, (B1 + 400) * 1000, 0],
                                   [B1, None, (B1 + 900) * 1000, 0], [B1, None, (B1 + 400) * 1000, (B1 + 390) * 1000], [0, None, 1e12, 0]]})
SC = {"load": LOAD, "events": ev}

INIT_FEED = r"""(() => {
  const RD = Date; window.__now = %d;
  class D extends RD { constructor(...a) { if (a.length) super(...a); else super(window.__now); } static now() { return window.__now; } }
  window.Date = D; window.setInterval = () => 0;
  window.__q = {}; window.__urls = [];
  window.fetch = async (u) => { const s = String(u); window.__urls.push(s); const p = s.split('?')[0], b = window.__q[p]; delete window.__q[p];
    return b == null ? new Response('', { status: 404 }) : new Response(b, { status: 200, headers: { 'Content-Type': 'application/json' } }); };
})();""" % LOAD
JS_EV = r"""async (e) => {
  if (e.now) window.__now = e.now;
  const ws = () => window.__wsAll[window.__wsAll.length - 1];
  if (e.k === 'fp') { window.__q['/api/footprint'] = e.body; const n = window.__urls.length; await refreshFootprint();
    const u = window.__urls.slice(n).find((x) => x.includes('/api/footprint?')); return { since: u ? u.split('since=')[1] : null }; }
  if (e.k === 'hist') { window.__q['/api/market-history'] = e.body; await fetchBinanceHistory('eth'); return null; }
  if (e.k === 'trade') { ws().onmessage({ data: e.msg }); return null; }
  if (e.k === 'reconnect') { ws().onclose(); ensurePriceWs(); return null; }
  if (e.k === 'retry') return { retry: e.cases.map((c) => historyRetryDue(c[0], c[1], c[2], c[3])) };
  const k = e.tf; chartTfK = k;
  const f = tfFootprint(footprintForChart()), L = footprintLive;
  const lane = latestFootprint ? (k > 1 ? tfFpBars(fpBarsNow()).map((b) => [b.time, b.levels, null]) : fpBarsNow().map((b) => [b.time, b.levels, b.dhl || null])) : null;
  const out = { tf: k, byTime: f ? [...f.byTime.entries()].sort((a, b) => a[0] - b[0]) : null, lane, candles: chartTfFull().map((c) => [c.time, c.open, c.high, c.low, c.close]),
    live: { barStart: L.barStart, since: L.since === Infinity ? null : L.since, cells: [...L.cells.entries()].sort((a, b) => a[0] - b[0]) },
    price: latestLivePriceByAsset.eth || 0, fast: lastFastPriceAt, full: lastSnapshotChartRenderAt, nbars: footprintBars.size };
  chartTfK = 1;
  return out;
}"""


def js_run():
    out, since = [], []
    with sync_playwright() as p:
        br = p.chromium.launch(args=LAUNCH)
        try:
            ctx = br.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, service_workers="block", timezone_id=TZ, locale="ko-KR")
            ctx.route("**/*", make_route(WT, {}))
            ctx.add_init_script(INIT); ctx.add_init_script(INIT_FEED)
            pg = ctx.new_page()
            pg.goto("http://fx.invalid/dashboard/live/", wait_until="domcontentloaded")
            pg.wait_for_function("typeof footprintLiveAdd === 'function' && window.__wsAll && window.__wsAll.length > 0", timeout=20_000)
            pg.wait_for_timeout(1500)
            for e in SC["events"]:
                r = pg.evaluate(JS_EV, e)
                if e["k"] == "fp": since.append(r["since"])
                elif e["k"] in ("dump", "retry"): out.append(r)
        finally:
            br.close()
    return {"dumps": out, "since": since}


def diff(a, b, path=""):
    """같은가 -- (어긋난 자리 수, 최대 수치 차, 첫 자리)"""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        d = abs(a - b)
        return (0 if d == 0 else 1), d, (None if d == 0 else path)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): return 1, math.inf, f"{path} 길이 {len(a)}/{len(b)}"
        n, m, f = 0, 0.0, None
        for i, (x, y) in enumerate(zip(a, b)):
            k, d, p = diff(x, y, f"{path}[{i}]"); n += k; m = max(m, d); f = f or p
        return n, m, f
    if isinstance(a, dict) and isinstance(b, dict):
        n, m, f = 0, 0.0, None
        for k in sorted(set(a) | set(b)):
            c, d, p = diff(a.get(k), b.get(k), f"{path}.{k}"); n += c; m = max(m, d); f = f or p
        return n, m, f
    return (0, 0.0, None) if a == b else (1, math.inf, f"{path} {str(a)[:60]} / {str(b)[:60]}")


if __name__ == "__main__":
    sp = HERE / "_feed_scenario.json"
    sp.write_text(json.dumps(SC))
    rt = json.loads(subprocess.run([str(WT / "rtui/target/release/examples/feed_replay"), str(sp)], capture_output=True, text=True, timeout=60, check=True).stdout)
    js = js_run()
    json.dump({"js": js, "rt": rt}, open(HERE / "_feed.json", "w"))
    names = ["1 첫 응답(전량)", "2 봉 중간 접속 뒤 체결 120", "2 서버가 B0 을 다시 줌", "3 봉 경계 걸침 + 늦은 B0 체결 + B1 체결 150", "4 서버가 형성 봉 B1 을 실음(5m)", "4 같은 때 30m", "4 같은 때 1h",
             "5 since 증분 빈 응답", "6 재연결(봉 중간) 뒤 체결 120 + 응답", "7 새 봉 B2 체결 120 + 응답 + 이력(5m)", "7 같은 때 30m", "8 두 봉 넘게 묵음 → 전량", "9 historyRetryDue 6사례"]
    n_tr = sum(e["k"] == "trade" for e in ev)
    md = ["## 자료 흐름 일치 (지금 화면 app.js 실제 함수 대 rtui feed::Feed · 같은 시나리오)", "",
          f"시나리오: 체결 {n_tr}건(가격 0 섞임 · 고래 주문 · 같은 주문 묶음) · 풋프린트 응답 {sum(e['k'] == 'fp' for e in ev)} · 캔들 이력 2 · 재연결 1 · 덤프 {len(js['dumps'])}", "",
          f"since 열(지금/rtui): {js['since']} / {rt['since']} → {'같음' if js['since'] == rt['since'] else '다름'}", "",
          "| 단계 | 칸(byTime) | 레인 입력(fpBars) | 캔들(chartTfFull) | 형성 봉 칸(footprintLive) | 현재가·80ms·400ms 시각·봉 수 | 판정 |", "|---|---|---|---|---|---|---|"]
    for nm, a, b in zip(names, js["dumps"], rt["dumps"]):
        cols, bad = [], 0
        for k in (("byTime",), ("lane",), ("candles",), ("live",), ("price", "fast", "full", "nbars")) if "retry" not in a else (("retry",),):
            c, d, f = diff({x: a.get(x) for x in k}, {x: b.get(x) for x in k})
            size = sum(len(a[x]) if isinstance(a.get(x), list) else 1 for x in k)
            cols.append(f"{size}개 · 어긋남 {c}" + (f" ({f})" if f else "")); bad += c
        if len(cols) == 1: cols = cols * 5
        md.append(f"| {nm} | " + " | ".join(cols) + f" | {'같음' if not bad else '다름'} |")
    txt = "\n".join(md)
    print(txt)
    (HERE / "feed.md").write_text(txt + "\n")
