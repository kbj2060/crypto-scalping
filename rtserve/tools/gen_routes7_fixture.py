#!/usr/bin/env python3
"""rtserve 일곱째 묶음 정답지(2026-11-28): /api/chart-markers · /api/position-profile.

chart-markers: 원본 모듈 scripts/live_eth_chart_markers_20260909.py 를 그대로 불러 쓴다(requests.get 자리에 «녹화된 응답 표») +
  server.py load_chart_markers 3213~3227 · api_chart_markers 4710~4712 를 옮겨 적음.
position-profile: 원본 모듈 dashboard/position_profile.py 를 그대로 불러 쓴다 + server.py pos_* 3303~3388 · api_position_profile 4738~4771 을
  옮겨 적음(fetch_binance_json 자리에 녹화 표 · flows 자리에 흐름 엔진 상태 · 1초 OI 링 = coin_flow collect_oi_1s 규칙으로 hot SQLite 행에서).
바깥 접속 없음.
    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR" \\
      $PY rtserve/tools/gen_routes7_fixture.py    # 저장소 루트에서 · 서버(AVX-512 없음)와 같은 np.exp
출력 rtserve/tests/fixtures/gen/routes7/{markers,posprof}.json(커밋 안 함).
"""
from __future__ import annotations

import base64
import json
import math
import os
import random
import sqlite3
import sys
import tempfile
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import live_eth_chart_markers_20260909 as CM  # noqa: E402
from dashboard import position_profile as pprof  # noqa: E402

assert sys.version_info[:2] == (3, 11), sys.version
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "routes7"
NOW = 0.0
REST: dict[str, tuple[int, str]] = {}
SENT: list[str] = []
SWR: dict[str, tuple[float, object]] = {}
time.time = lambda: NOW


class HTTPException(Exception):
    status = 0

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class HTTPBadRequest(HTTPException):
    status = 400


class HTTPNotFound(HTTPException):
    status = 404


class HTTPBadGateway(HTTPException):
    status = 502


class HTTPServiceUnavailable(HTTPException):
    status = 503


def target(url, params):
    return urllib.parse.urlsplit(url).path + "?" + urllib.parse.urlencode(params)


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self.body = status, body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return json.loads(self.body)


def fake_get(url, params=None, timeout=None):   # requests.get 자리(차트 마커의 _fetch)
    t = target(url, params)
    SENT.append(t)
    return FakeResp(*REST.get(t, (404, "[]")))


CM.LD.requests.get = fake_get


def fetch_binance_json(url, params, *, timeout=None, error_reason=None):
    t = target(url, params)
    SENT.append(t)
    status, body = REST.get(t, (404, "[]"))
    if status != 200:
        if error_reason is None:
            return None
        raise HTTPBadGateway(error_reason)
    return json.loads(body)


def swr_cached(key, ttl, produce):   # 신선(나이 < ttl) = 그대로 · 그 밖 = 새로(시험은 묵음 창 600초를 늘 넘겨 뛴다) · 실패는 안 남김
    hit = SWR.get(key)
    if hit is not None and NOW - hit[0] < ttl:
        return hit[1]
    v = produce()
    SWR[key] = (NOW, v)
    return v


# ───────────────────────── chart-markers ─────────────────────────

def api_chart_markers(q):   # 4710 + load_chart_markers 3213
    asset = (q.get("asset", "eth") or "eth").lower()
    if asset != "eth":
        return 200, CM.compute_chart_markers(asset)
    return 200, swr_cached("chart_markers", 60, lambda: CM.compute_chart_markers("eth", None, None, None, None))


def kl(t0, n, c0, rnd, dp, shocks=()):
    out, c = [], c0
    for i in range(n):
        o = c
        sd = 0.004 if i % 97 < 3 else 0.0015
        c = round(c * (1 + rnd.gauss(0, sd)), dp)
        if shocks and n - 60 <= i < n - 48:       # 끝 72봉 안에 급락(바닥 쪽 순위·칼만·수익 z·피보 연장) → 급등(꼭대기 쪽)
            c = round(c * 0.9965, dp)
        elif shocks and n - 20 <= i < n - 8:
            c = round(c * 1.0035, dp)
        if i % 211 < 15 and i < 1200:   # 평평한 구간(고=저 → %K NaN · 864봉 순위 창이 덜 참) -- 끝 72봉 순위에는 안 걸리게 앞쪽만
            c = o
        h, lo = max(o, c), min(o, c)
        if not (i % 211 < 15 and i < 1200):
            h, lo = round(h * (1 + rnd.uniform(0, 0.0015)), dp), round(lo * (1 - rnd.uniform(0, 0.0015)), dp)
        vol = rnd.uniform(50, 900)
        tb = vol * rnd.uniform(0.2, 0.8)
        if i in shocks:
            vol *= 8; tb = vol * (0.97 if (i % 2 if not (n - 60 <= i < n - 8) else i >= n - 20) else 0.03)
        t = t0 + i * 300_000
        out.append([t, f"{o:.{dp}f}", f"{h:.{dp}f}", f"{lo:.{dp}f}", f"{c:.{dp}f}", f"{vol:.3f}", t + 299_999, "0", 10, f"{tb:.3f}", "0", "0"])
    return out


def put_kl(sym, rows, end=None, status=200, body=None):
    p = {"symbol": sym, "interval": "5m", "limit": 1500}
    if end is not None:
        p["endTime"] = end
    REST[target(CM.LD.KLINES, p)] = (status, body if body is not None else json.dumps(rows))


def scen_markers(now, setup, paths):
    global NOW
    NOW = now
    REST.clear(); SWR.clear()
    setup()
    cases = []
    for p in paths:
        SENT.clear()
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(p).query, keep_blank_values=True))
        status, v = api_chart_markers(q)
        cases.append({"path": p, "status": status, "body": json.dumps(v), "sent": list(SENT)})
    return {"now": now, "rest": {k: list(v) for k, v in REST.items()}, "cases": cases}


def markers():
    rnd = random.Random(20261128)
    now = datetime(2026, 11, 20, 10, 2, 3, 500000, tzinfo=timezone.utc).timestamp()
    m_last = int(now * 1000) // 300_000 * 300_000            # 형성 봉 시작
    t0 = m_last - 2999 * 300_000
    shocks = set(range(2930, 3000, 7)) | {2944, 2948, 2951, 2981, 2986, 2990}

    def two_pages(sym, rows):
        put_kl(sym, rows[1500:])
        put_kl(sym, rows[:1500], end=int(rows[1500][0]) - 1)

    def setup_a():
        eth = kl(t0, 3000, 3000.0, rnd, 2, shocks)
        two_pages("ETHUSDT", eth)
        btc = kl(t0, 3000, 90000.0, rnd, 1, shocks)
        del btc[2950]; del btc[100]                           # BTC 빠진 봉(왼쪽 붙이기 NaN)
        btc[2000] = list(btc[2000]); btc.insert(2001, list(btc[2000]))   # 같은 시각 두 번(처음 것)
        put_kl("BTCUSDT", btc[-1500:])
        put_kl("BTCUSDT", btc[:-1500], end=int(btc[-1500][0]) - 1)
    pa = ["/api/chart-markers", "/api/chart-markers?asset=eth", "/api/chart-markers?asset=SOL", "/api/chart-markers?asset=", "/api/chart-markers?asset=ETH"]

    def setup_b():   # ETH 500 → price_fetch_failed
        put_kl("ETHUSDT", None, status=500, body='{"code": -1}')

    def setup_c():   # 1100봉 + 빈 둘째 쪽 → < 1200
        eth = kl(m_last - 1099 * 300_000, 1100, 3000.0, rnd, 2)
        put_kl("ETHUSDT", eth); put_kl("ETHUSDT", [], end=int(eth[0][0]) - 1)

    def setup_d():   # BTC 첫 쪽 503 → btc_fetch_failed
        two_pages("ETHUSDT", kl(t0, 3000, 2900.0, rnd, 2, shocks))
        put_kl("BTCUSDT", None, status=503, body="x")

    def setup_e():   # BTC 첫 쪽만(둘째 쪽 빈 목록) · ETH 마지막 = 방금 닫힌 봉(형성 봉 없음)
        eth = kl(t0 - 300_000, 3000, 3100.0, rnd, 2, shocks)
        two_pages("ETHUSDT", eth)
        b = kl(t0 + 1500 * 300_000, 1500, 88000.0, rnd, 1, shocks)
        put_kl("BTCUSDT", b); put_kl("BTCUSDT", [], end=int(b[0][0]) - 1)
    sc = [scen_markers(now, setup_a, pa), scen_markers(now, setup_b, ["/api/chart-markers", "/api/chart-markers?asset=xrp"]),
          scen_markers(now, setup_c, ["/api/chart-markers"]), scen_markers(now, setup_d, ["/api/chart-markers"]),
          scen_markers(now, setup_e, ["/api/chart-markers?asset=eth"])]
    return {"scenarios": sc}


# ───────────────────────── position-profile ─────────────────────────

FLOW_ASSETS = ("eth", "sol", "xrp", "btc")      # start_with "eth,btc,sol,xrp" 의 흐름 엔진(FLOW_SPECS 순서)
BUCKET = {"eth": 0.5, "sol": 0.02, "xrp": 0.0003, "btc": 15.0}
POS_OI_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"
POS_SEED_BACKOFF_S = 60
COIN_CONFIG = ("eth", "btc", "sol", "xrp", "hype", "sndk", "soxl", "skhynix", "mu")


class Flow:
    def __init__(self, bucket):
        self.footprint_state = {"bars": {}, "sec": {}}
        self.oi_1s = {}
        self.spec = type("Spec", (), {"bucket": bucket})


flows: dict[str, Flow] = {}
pos_books: dict[str, dict] = {}
pos_seed_fail: dict[str, float] = {}
OI_ROWS: list[tuple[str, int, float]] = []


def oi_ring(sym):   # collect_oi_1s: 지금까지 쓰인 행 중 (지금 − 690초, 지금] · 행이 있으면 지금 − 660초 앞을 버림
    lo, hi = int((NOW - 660) * 1000) - 30_000, int(NOW * 1000)
    rows = [(t, v) for s, t, v in OI_ROWS if s == sym and lo < t <= hi]
    ring = {}
    for t, v in rows:
        ring[t // 1000] = v
    if rows:
        cut = int(NOW) - 660
        ring = {k: v for k, v in ring.items() if k >= cut}
    return ring


def pos_build(bars):   # 3313
    book = pprof.Book()
    for b in bars:
        book.push(*b)
    return book


def pos_bars_rest(asset, start_s, end_s):   # 3319
    sym, step = asset.upper() + "USDT", 300_000
    oi = {}
    t = start_s * 1000
    while t < end_s * 1000:
        oi.update(pprof.oi_hist_map(fetch_binance_json(
            POS_OI_HIST_URL, {"symbol": sym, "period": "5m", "limit": 500, "startTime": t, "endTime": min(t + 499 * step, end_s * 1000)},
            error_reason="position_profile_upstream_error")))
        t += 500 * step
    out = []
    t = start_s * 1000
    while t < end_s * 1000:
        rows = fetch_binance_json("https://fapi.binance.com/fapi/v1/klines",
                                  {"symbol": sym, "interval": "5m", "limit": 1500, "startTime": t, "endTime": end_s * 1000 - 1},
                                  error_reason="position_profile_upstream_error")
        if not rows:
            break
        out += pprof.bars_from_klines(rows, oi, end_s)
        t = int(rows[-1][0]) + step
    return out


def pos_offset(asset, st):   # 3341
    f = flows.get(asset)
    live = pprof.live_at(f.oi_1s, st["last"] + 300) if f is not None else None
    if live is not None and st["book"].oi is not None:
        st["off"] = st["book"].oi - live


def pos_ensure(asset):   # 3349
    now = int(time.time()); cur = now // 300 * 300
    st = pos_books.get(asset)
    if st is None:
        if now - pos_seed_fail.get(asset, 0.0) < POS_SEED_BACKOFF_S:
            raise HTTPServiceUnavailable("position_profile_seed_backoff")
        try:
            bars = pos_bars_rest(asset, cur - 30 * 86400 + 300, cur)
        except Exception as exc:  # noqa: BLE001
            pos_seed_fail[asset] = now
            raise HTTPBadGateway("position_profile_upstream_error") from exc
        book = pos_build(bars)
        st = pos_books[asset] = {"book": book, "last": bars[-1][0] if bars else cur - 300, "tried": now, "off": None}
        pos_offset(asset, st)
    elif cur - st["last"] > 300 and now - st["tried"] >= 60:
        st["tried"] = now
        try:
            for b in pos_bars_rest(asset, st["last"] + 300, cur):
                if b[0] > st["last"]:
                    st["book"].push(*b); st["last"] = b[0]
            pos_offset(asset, st)
        except Exception as exc:  # noqa: BLE001
            print(f"position profile advance failed (retry next minute): {exc!r}")
    elif st["off"] is None:
        pos_offset(asset, st)
    return st


def pos_forming(asset, st):   # 3378
    f = flows.get(asset)
    if f is None:
        return None, None
    px = next((float(c[6]) for _, c in sorted(list(f.footprint_state["sec"].items()), reverse=True) if c[6]), None)
    hlv = pprof.forming_from_cells(f.footprint_state["bars"], st["last"] + 300, f.spec.bucket)
    if hlv is None or not f.oi_1s or st.get("off") is None:
        return None, px
    return (int(time.time()) // 300 * 300, hlv[0], hlv[1], hlv[2], float(f.oi_1s[max(f.oi_1s)]) + st["off"]), px


def api_position_profile(q):   # 4738
    asset = q.get("asset", "eth").lower()
    if asset not in COIN_CONFIG:
        raise HTTPBadRequest("unsupported_asset")
    if asset not in flows:
        raise HTTPNotFound("flow_off")
    d = q.get("days")
    wins = (int(d),) if d in ("1", "7", "30") else pprof.WINDOWS
    rng = None
    if q.get("from_s") is not None:
        try:
            f_s, t_s = (int(q.get(k, "")) // 14400 * 14400 for k in ("from_s", "to_s"))
        except ValueError:
            raise HTTPBadRequest("bad_range") from None
        now_i = int(time.time())
        g4 = lambda v: v // 14400 * 14400   # noqa: E731
        f_s, t_s = max(f_s, g4(now_i - 31 * 86400)), min(t_s, g4(now_i + 86400))
        if f_s >= t_s:
            raise HTTPBadRequest("bad_range")
        rng, wins = (f_s, t_s), ()

    def produce():
        st = pos_ensure(asset)
        forming, px = pos_forming(asset, st)
        bw = 1.0 if asset == "eth" else flows[asset].spec.bucket
        return pprof.payload(st["book"], int(time.time()), px, forming, bw=bw, windows=wins, rng=rng)
    try:
        payload = swr_cached(f"position_profile:{asset}:{','.join(map(str, wins))}:{rng}", 1.0, produce)
    except HTTPException:
        return 502, {"error": "position_profile_upstream_error"}
    return 200, payload


def oi_hist_rows(bars_s, rnd, oi0):
    out, oi = [], oi0
    for b in bars_s:
        oi = max(oi + rnd.gauss(0, 800), 1.0)
        out.append({"symbol": "X", "sumOpenInterest": f"{oi:.3f}", "sumOpenInterestValue": "0", "timestamp": (b + 300) * 1000})
    return out, oi


def pk(b, c0, rnd, dp):
    o = c0; c = round(o * (1 + rnd.gauss(0, 0.002)), dp)
    h, lo = round(max(o, c) * (1 + rnd.uniform(0, 0.002)), dp), round(min(o, c) * (1 - rnd.uniform(0, 0.002)), dp)
    return [b * 1000, f"{o:.{dp}f}", f"{h:.{dp}f}", f"{lo:.{dp}f}", f"{c:.{dp}f}", f"{rnd.uniform(500, 5000):.3f}", b * 1000 + 299_999, "0", 1, "0", "0", "0"], c


def put_oi(sym, t, end_ms, rows, status=200):
    REST[target(POS_OI_HIST_URL, {"symbol": sym, "period": "5m", "limit": 500, "startTime": t, "endTime": end_ms})] = (status, json.dumps(rows))


def put_k(sym, t, end_ms, rows, status=200):
    REST[target("https://fapi.binance.com/fapi/v1/klines", {"symbol": sym, "interval": "5m", "limit": 1500, "startTime": t, "endTime": end_ms})] = (status, json.dumps(rows))


def oi_pages(sym, start_s, end_s, rows):   # 모든 쪽 녹화(행은 첫 쪽에 몰아 둠 -- 원본이 사전으로 합친다)
    t, first = start_s * 1000, True
    while t < end_s * 1000:
        put_oi(sym, t, min(t + 499 * 300_000, end_s * 1000), rows if first else [])
        t += 500 * 300_000; first = False


def write_db(rows):
    p = Path(tempfile.mkdtemp()) / "binance_ctx.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE oi_1s (ts_ms INTEGER, symbol TEXT, open_interest REAL)")
    con.executemany("INSERT INTO oi_1s VALUES (?, ?, ?)", [(t, s, v) for s, t, v in rows])
    con.commit(); con.close()
    return base64.b64encode(p.read_bytes()).decode()


def set_hub(hub):
    flows.clear()
    for a in FLOW_ASSETS:
        f = flows[a] = Flow(BUCKET[a])
        h = hub.get(a)
        if h:
            for t, cells in h["bars"]:
                f.footprint_state["bars"][t] = {k: [b, s, 0.0, 0.0, 0.0, 0.0] for k, b, s in cells}
            for s, px in h["secs"]:
                f.footprint_state["sec"][s] = [0.0] * 6 + [px]


def scen_pos(hub, oi_rows, cases_in):
    global NOW
    SWR.clear(); pos_books.clear(); pos_seed_fail.clear(); SENT.clear()   # REST 는 부르는 쪽이 채워 둠
    OI_ROWS[:] = oi_rows
    set_hub(hub)
    cases = []
    for now, p in cases_in:
        NOW = now
        for a in FLOW_ASSETS:
            flows[a].oi_1s = oi_ring(a + "usdt")
        SENT.clear()
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(p).query, keep_blank_values=True))
        try:
            status, v = api_position_profile(q)
            body = json.dumps(v)
        except HTTPException as exc:
            status, body = exc.status, f"{exc.status}: {exc.reason}"
        cases.append({"now": now, "path": p, "status": status, "body": body, "sent": list(SENT)})
    hub_rs = {a + "usdt": {"bucket": BUCKET[a], "bars": [[t, [[round(k * BUCKET[a], 6), b, s] for k, b, s in cells]] for t, cells in h["bars"]],
                           "secs": h["secs"]} for a, h in hub.items()}
    return {"now": cases_in[0][0], "rest": {k: list(v) for k, v in REST.items()}, "files": {"data/hot/binance_ctx.sqlite": write_db(oi_rows)},
            "hub": hub_rs, "cases": cases}


def hub_bars(rnd, ts, mid_key, spread):
    out = []
    for t in ts:
        ks = rnd.sample(range(mid_key - spread, mid_key + spread), rnd.randint(3, 9))
        out.append([t, [[k, rnd.randint(0, 400) / 8, rnd.randint(0, 400) / 8] for k in ks]])
    return out


def oi_secs(sym, now, oi, rnd, n=200):
    return [(sym, int((now - 650 + i * 3.2) * 1000) + rnd.randint(0, 999), round(oi + rnd.gauss(0, 50), 3)) for i in range(n)]


def posprof():
    rnd = random.Random(1128)
    REST.clear()
    # S1: ETH 30일 전부(8,639봉) · 900초씩 두 번 밀기(두 번째에 30일 접힘) · 형성 봉·1초 OI·체결가 있음
    now0 = datetime(2026, 11, 20, 10, 2, 3, 500000, tzinfo=timezone.utc).timestamp()
    cur0 = int(now0) // 300 * 300
    start = cur0 - 30 * 86400 + 300
    bars = list(range(start, cur0 + 1800, 300))                 # 시드 + 밀기 두 번(각 3봉) + 형성 봉
    rows, c = [], 3000.0
    for b in bars:
        r, c = pk(b, c, rnd, 2); rows.append(r)
    oih, _ = oi_hist_rows(bars, rnd, 2.0e6)
    del oih[5000]                                              # 끝 스냅샷 없는 봉 → 빠짐
    oih[6000]["sumOpenInterest"] = "0"                         # OI 0 → push 가 건너뜀
    seed = [r for r in rows if r[0] < cur0 * 1000]
    oi_pages("ETHUSDT", start, cur0, oih[:-6])
    t = start * 1000
    while t < cur0 * 1000:
        page = [r for r in seed if r[0] >= t][:1500]
        put_k("ETHUSDT", t, cur0 * 1000 - 1, page)
        t = page[-1][0] + 300_000
    for k in (1, 2):                                           # 밀기: [cur0 + 900(k−1), cur0 + 900k) · 형성 봉 한 줄 덤(b < end_s 로 빠짐)
        a, e = cur0 + 900 * (k - 1), cur0 + 900 * k
        put_oi("ETHUSDT", a * 1000, min(a * 1000 + 499 * 300_000, e * 1000), [o for o in oih if a + 300 <= o["timestamp"] // 1000 <= e])
        put_k("ETHUSDT", a * 1000, e * 1000 - 1, [r for r in rows if a * 1000 <= r[0] <= e * 1000])
    now1, now2 = now0 + 900, now0 + 1800
    hub = {"eth": {"bars": hub_bars(rnd, range(cur0, cur0 + 2101, 300), int(c / 0.5), 12),
                   "secs": [[int(now2) - 3, round(c + 0.5, 2)], [int(now2) - 1, 0.0], [int(now0) - 5, round(c - 1.0, 2)]]}}
    oi_rows = oi_secs("ethusdt", now0, 2.0e6, rnd) + oi_secs("ethusdt", now1, 2.0e6, rnd) + oi_secs("ethusdt", now2, 2.0e6, rnd)
    P = "/api/position-profile"
    cases1 = [(now0, f"{P}?asset=eth"), (now0, f"{P}?asset=eth&days=7"), (now0, f"{P}?asset=eth&days=2"), (now0, f"{P}?asset=ETH"),
              (now0, f"{P}?asset=eth&from_s={cur0 - 3 * 86400}&to_s={cur0 + 3600}"), (now0, f"{P}?asset=eth&from_s=abc&to_s=1"),
              (now0, f"{P}?asset=eth&from_s=5&to_s=5"), (now0, f"{P}?asset=eth&from_s=5"), (now0, f"{P}?asset=doge"), (now0, f"{P}?asset=hype"),
              (now0, f"{P}?asset=sol"), (now0, f"{P}?asset=sol"),
              (now1, f"{P}?asset=eth"), (now2, f"{P}?asset=eth"), (now2, f"{P}?asset=eth&days=1&from_s=0&to_s=99999999999")]
    s1 = scen_pos(hub, oi_rows, cases1)

    # S2: XRP 첫 적재 실패(502) → 30초 뒤 백오프(안 보냄) → 61초 뒤(새 5분봉) 짧은 적재 성공 · 900초 뒤 밀기 실패(로그만) · 흐름 상태 없음(가격·형성 봉 없음)
    REST.clear()
    t2 = datetime(2026, 3, 9, 15, 0, 0, tzinfo=timezone.utc).timestamp() + 250
    c2 = int(t2) // 300 * 300
    put_oi("XRPUSDT", (c2 - 30 * 86400 + 300) * 1000, (c2 - 30 * 86400 + 300) * 1000 + 499 * 300_000, [], status=502)
    c3 = c2 + 300
    st3 = c3 - 30 * 86400 + 300
    xb = list(range(c3 - 200 * 300, c3, 300))
    xr, xc = [], 2.3456
    for b in xb:
        r, xc = pk(b, xc, rnd, 4); xr.append(r)
    xo, _ = oi_hist_rows(xb, rnd, 3.0e8)
    oi_pages("XRPUSDT", st3, c3, xo)
    put_k("XRPUSDT", st3 * 1000, c3 * 1000 - 1, xr)
    c4 = (int(t2) + 61 + 900) // 300 * 300
    put_oi("XRPUSDT", c3 * 1000, min(c3 * 1000 + 499 * 300_000, c4 * 1000), [])
    put_k("XRPUSDT", c3 * 1000, c4 * 1000 - 1, [], status=502)
    cases2 = [(t2, f"{P}?asset=xrp"), (t2 + 30, f"{P}?asset=xrp"), (t2 + 61, f"{P}?asset=xrp"), (t2 + 961, f"{P}?asset=xrp"),
              (t2 + 961, f"{P}?asset=xrp&days=1")]
    s2 = scen_pos({}, [], cases2)
    return {"scenarios": [s1, s2]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = markers()
    (OUT / "markers.json").write_text(json.dumps(m, ensure_ascii=False))
    p = posprof()
    (OUT / "posprof.json").write_text(json.dumps(p, ensure_ascii=False))
    n = lambda fx: sum(len(s["cases"]) for s in fx["scenarios"])   # noqa: E731
    print(f"{OUT.relative_to(ROOT)}: markers 요청 {n(m)} · posprof 요청 {n(p)} · 파이썬 {sys.version.split()[0]} · numpy {np.__version__} · "
          f"pandas {pd.__version__} · NPY_DISABLE_CPU_FEATURES={'set' if os.environ.get('NPY_DISABLE_CPU_FEATURES') else 'unset'}")


if __name__ == "__main__":
    main()
