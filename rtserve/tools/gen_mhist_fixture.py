#!/usr/bin/env python3
"""rtserve 차트 과거 봉 정답지(2026-11-28): /api/market-history.

원본 핸들러·적재 함수를 옮겨 적었다(dashboard/server.py forming_kline 404~415 · trend_veto_rows 1619~1652 · kline_candles 1654~1667 ·
load_market_history_from_evidence_cache 1671~1722 · load_market_history 1724~1742 · load_chart_klines_frames 3193~3281 · api_market_history 3709~3720
-- Read 로 읽은 줄). VWAP 은 원본 모듈 dashboard/market_ctx.py 를 그대로 불러 쓴다. 바깥 접속 없음: fetch_binance_json 자리에 «녹화된 응답 표».
    $PY rtserve/tools/gen_mhist_fixture.py    # 저장소 루트에서
출력 rtserve/tests/fixtures/gen/mhist/mhist.json(커밋 안 함).
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dashboard import market_ctx as mctx  # noqa: E402

assert sys.version_info[:2] == (3, 11), sys.version
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "mhist" / "mhist.json"
FAPI = "/fapi/v1/klines"
MARKET_SYMBOLS = {a: a.upper() + "USDT" for a in ("eth", "sol", "btc", "xrp", "hype", "sndk", "soxl", "skhynix", "mu")}
EVIDENCE_ASSETS = ("eth", "btc")
TREND_VETO_N, TREND_VETO_K = 144, 1.0
NOW = 0.0


class HTTPException(Exception):
    status = 0

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class HTTPBadRequest(HTTPException):
    status = 400


class HTTPBadGateway(HTTPException):
    status = 502


REST: dict[str, tuple[int, str]] = {}
SENT: list[str] = []
SWR: dict[str, object] = {}
evidence_signal_cache: dict = {"frames": None}
market_rows_memo: dict = {}
time.time = lambda: NOW


def fetch_binance_json(url, params, *, timeout=None, error_reason=None):
    target = urllib.parse.urlsplit(url).path + "?" + urllib.parse.urlencode(params)
    SENT.append(target)
    status, body = REST.get(target, (404, "[]"))
    if status != 200:
        if error_reason is None:
            return None
        raise HTTPBadGateway(error_reason)
    return json.loads(body)


def swr_cached(key, produce):   # 신선 구간 안(같은 키 = 캐시 값 · 실패는 캐시 안 함)
    if key not in SWR:
        SWR[key] = produce()
    return SWR[key]


def forming_kline(raw, now_ms, bar_s=300):   # 404
    if not raw:
        return None
    r = raw[-1]
    t = int(r[0]) // 1000
    if t + bar_s <= now_ms / 1000:
        return None
    return {"time": t, "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "volume": float(r[5]), "partial": True}


def trend_veto_rows(df, tail=TREND_VETO_N * 5):   # 1619
    w = df.tail(tail)
    if len(w) < TREND_VETO_N + 1:
        return {}
    c, h, l = w["close"], w["high"], w["low"]
    sma = c.rolling(TREND_VETO_N).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / TREND_VETO_N, adjust=False).mean()
    dev = ((c - sma) / sma).to_numpy()
    eps = (TREND_VETO_K * atr / c).to_numpy()
    drop = c.shift(TREND_VETO_N - 1).to_numpy()
    out = {}
    cur = 0
    for i, ts in enumerate(w["timestamp"]):
        if not (np.isfinite(dev[i]) and np.isfinite(eps[i])):
            continue
        if dev[i] > eps[i]:
            cur = 1
        elif dev[i] < -eps[i]:
            cur = -1
        row = {"sma": round(float(sma.iloc[i]), 6), "atr": round(float(atr.iloc[i]), 6), "veto": cur}
        if np.isfinite(drop[i]):
            row["drop"], row["vn"] = round(float(drop[i]), 6), TREND_VETO_N
            row["vk"] = TREND_VETO_K
        out[int(ts.timestamp())] = row
    return out


def kline_candles(rows, now_ms, nd, veto_tail=TREND_VETO_N * 5):   # 1654
    closed = [r for r in rows if int(r[6]) < now_ms]
    veto = trend_veto_rows(pd.DataFrame({"timestamp": pd.to_datetime([int(r[0]) for r in closed], unit="ms", utc=True),
                                         "high": [float(r[2]) for r in closed], "low": [float(r[3]) for r in closed],
                                         "close": [float(r[4]) for r in closed]}), veto_tail)
    tss = [int(r[0]) // 1000 for r in rows]
    vwap = mctx.vwap_rows(tss, [float(r[2]) for r in rows], [float(r[3]) for r in rows],
                          [float(r[4]) for r in rows], [float(r[5]) for r in rows], nd=nd)
    return [{"time": t, "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "volume": float(r[5]),
             **veto.get(t, {}), **vwap.get(t, {}),
             **({"partial": True} if int(r[6]) >= now_ms else {})}
            for t, r in zip(tss, rows)]


def load_chart_klines_frames():   # 3193
    def produce():
        raw = fetch_binance_json("https://fapi.binance.com/fapi/v1/klines", {"symbol": "ETHUSDT", "interval": "5m", "limit": 1500},
                                 error_reason="evidence_signal_upstream_error")
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
        df = pd.DataFrame(raw, columns=cols)
        for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
            df[c] = df[c].astype("float64")
        df["close_time"] = df["close_time"].astype("int64")
        df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        now_ms = int(time.time() * 1000)
        if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
            df = df.iloc[:-1].reset_index(drop=True)
        forming = {"eth": forming_kline(raw, now_ms), "btc": None}
        btc_df = None
        try:
            braw = fetch_binance_json("https://fapi.binance.com/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "5m", "limit": 1500})
            if braw is not None:
                bdf = pd.DataFrame(braw, columns=cols)
                for c in ("open", "high", "low", "close"):
                    bdf[c] = bdf[c].astype("float64")
                bdf["close_time"] = bdf["close_time"].astype("int64")
                bdf["timestamp"] = pd.to_datetime(bdf["open_time"].astype("int64"), unit="ms", utc=True)
                bdf = bdf.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
                forming["btc"] = forming_kline(braw, now_ms)
                if len(bdf) and int(bdf.iloc[-1]["close_time"]) >= now_ms:
                    bdf = bdf.iloc[:-1].reset_index(drop=True)
                btc_df = bdf[["timestamp", "open", "high", "low", "close"]]
        except Exception as btc_exc:  # noqa: BLE001
            print(f"BTC leg failed: {btc_exc}")
        evidence_signal_cache["frames"] = (df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]], btc_df, None)
        evidence_signal_cache["forming"] = forming
        return {"available": True}
    return swr_cached("evidence_signal", produce)


def load_market_history_from_evidence_cache(asset):   # 1671
    load_chart_klines_frames()
    frames = evidence_signal_cache["frames"]
    if frames is None:
        raise HTTPBadGateway("market_history_upstream_error")
    closed_df, btc_df, _funding_df = frames
    src = closed_df if asset == "eth" else btc_df
    if src is None or src.empty:
        raise HTTPBadGateway("market_history_upstream_error")
    memo = market_rows_memo.get(asset)
    if memo is not None and memo[0] is src:
        return memo[1]
    veto = trend_veto_rows(src)
    vwap = {}
    if "volume" in src.columns:
        tail = src.tail(600)
        tss = [int(t.timestamp()) for t in tail["timestamp"]]
        vwap = mctx.vwap_rows(tss, tail["high"].tolist(), tail["low"].tolist(), tail["close"].tolist(), tail["volume"].tolist())
    rows = [{"time": int(row["timestamp"].timestamp()), "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]),
             "close": float(row["close"]), **({"volume": float(row["volume"])} if "volume" in src.columns else {}),
             **veto.get(int(row["timestamp"].timestamp()), {}), **vwap.get(int(row["timestamp"].timestamp()), {})}
            for _, row in src.tail(200).iterrows()]
    market_rows_memo[asset] = (src, rows)
    return rows


def load_market_history(asset):   # 1724
    if asset in EVIDENCE_ASSETS:
        return load_market_history_from_evidence_cache(asset)

    def produce():
        rows = fetch_binance_json("https://fapi.binance.com/fapi/v1/klines", {"symbol": MARKET_SYMBOLS[asset], "interval": "5m", "limit": 1000},
                                  timeout=3.0, error_reason="market_history_upstream_error")
        return kline_candles(rows, time.time() * 1000, 6)[-200:]
    return swr_cached(f"market_history:{asset}", produce)


def api_market_history(q):   # 3709
    asset = q.get("asset", "").lower()
    if asset not in MARKET_SYMBOLS:
        raise HTTPBadRequest("unsupported_market_history_asset")
    candles = load_market_history(asset)
    f = (evidence_signal_cache.get("forming") or {}).get(asset) if asset in EVIDENCE_ASSETS else None
    if f and candles and f["time"] > candles[-1]["time"]:
        candles = [*candles, f]
    return 200, {"asset": asset, "candles": candles}


def kl(t0, n, c0, rnd, dp, sd=0.003, zero_vol=0):
    out, c = [], c0
    for i in range(n):
        o = c
        c *= 1 + rnd.gauss(0, sd)
        h, lo = max(o, c) * (1 + rnd.uniform(0, 0.002)), min(o, c) * (1 - rnd.uniform(0, 0.002))
        t = t0 + i * 300_000
        vol = 0.0 if i < zero_vol else rnd.uniform(1, 900)
        out.append([t, f"{o:.{dp}f}", f"{h:.{dp}f}", f"{lo:.{dp}f}", f"{c:.{dp}f}", f"{vol:.3f}", t + 299_999, "0", 10, "1.5", "0", "0"])
    return out


def put(sym, limit, status, body):
    REST[FAPI + "?" + urllib.parse.urlencode({"symbol": sym, "interval": "5m", "limit": limit})] = (status, body if isinstance(body, str) else json.dumps(body))


def scenario(now, setup, paths):
    global NOW
    NOW = now
    REST.clear(); SWR.clear(); market_rows_memo.clear(); evidence_signal_cache.clear(); evidence_signal_cache["frames"] = None
    setup()
    cases = []
    for p in paths:
        SENT.clear()
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(p).query, keep_blank_values=True))
        try:
            status, v = api_market_history(q)
            body = json.dumps(v)
        except HTTPException as exc:
            status, body = exc.status, f"{exc.status}: {exc.reason}"
        cases.append({"path": p, "status": status, "body": body, "sent": list(SENT)})
    return {"now": now, "rest": {k: list(v) for k, v in REST.items()}, "cases": cases}


def main():
    rnd = random.Random(20261128)
    # A: 2026-03-09 15:02:03.4 UTC -- 1500봉(5.2일) 안에 미국 서머타임 시작(03-08) · 형성 봉 · 같은 시각 두 번(같은 값 -- 값이 다르면 원본 정렬이 불안정)
    now_a = datetime(2026, 3, 9, 15, 2, 3, 400000, tzinfo=timezone.utc).timestamp()
    m_last = int(now_a * 1000) // 300_000 * 300_000

    def setup_a():
        eth = kl(m_last - 1499 * 300_000, 1500, 2100.0, rnd, 2)
        eth.insert(700, list(eth[700]))
        put("ETHUSDT", 1500, 200, eth)
        put("BTCUSDT", 1500, 200, kl(m_last - 1499 * 300_000, 1500, 70000.0, rnd, 1))
        put("SOLUSDT", 1000, 200, kl(m_last - 999 * 300_000, 1000, 88.0, rnd, 3))
        put("XRPUSDT", 1000, 500, '{"code": -1}')
        put("HYPEUSDT", 1000, 200, [])
        put("MUUSDT", 1000, 200, kl(m_last - 99 * 300_000, 100, 110.0, rnd, 2))
        put("SOXLUSDT", 1000, 200, kl(m_last - 998 * 300_000, 999, 30.0, rnd, 4, zero_vol=40))   # 마지막 = 방금 닫힌 봉(형성 봉 없음) · 처음 40봉 거래량 0
    paths_a = [f"/api/market-history?asset={a}" for a in ("eth", "ETH", "btc", "sol", "xrp", "hype", "mu", "soxl", "doge", "")] + ["/api/market-history"]

    # B: 런던 서머타임 끝(2026-10-25) · ETH 실패 → 502 evidence · BTC 는 같은 프레임이라 같이 실패
    now_b = datetime(2026, 10, 26, 9, 0, 0, tzinfo=timezone.utc).timestamp()
    mb = int(now_b * 1000) // 300_000 * 300_000

    def setup_b():
        put("ETHUSDT", 1500, 502, "bad")
        put("SNDKUSDT", 1000, 200, kl(mb - 999 * 300_000, 1000, 250.0, rnd, 2))
    paths_b = ["/api/market-history?asset=eth", "/api/market-history?asset=btc", "/api/market-history?asset=sndk"]

    # C: ETH 빈 목록(마감봉 0 → 502 market) · BTC 404(없음 → 502 market) -- 런던 서머타임 시작 주(2026-03-29)
    now_c = datetime(2026, 3, 30, 7, 30, 0, tzinfo=timezone.utc).timestamp()
    mc = int(now_c * 1000) // 300_000 * 300_000

    def setup_c():
        put("ETHUSDT", 1500, 200, [])
        put("SKHYNIXUSDT", 1000, 200, kl(mc - 999 * 300_000, 1000, 9.5, rnd, 4))
    paths_c = ["/api/market-history?asset=eth", "/api/market-history?asset=btc", "/api/market-history?asset=skhynix"]

    # D: ETH 정상 · BTC 실패(404) → eth 만
    def setup_d():
        put("ETHUSDT", 1500, 200, kl(m_last - 1500 * 300_000, 1500, 2050.0, rnd, 2))   # 마지막 = 방금 닫힌 봉 → 형성 봉 없음
    paths_d = ["/api/market-history?asset=btc", "/api/market-history?asset=eth"]

    sc = [scenario(now_a, setup_a, paths_a), scenario(now_b, setup_b, paths_b), scenario(now_c, setup_c, paths_c), scenario(now_a, setup_d, paths_d)]
    # 세션 시작 표(ZoneInfo): 2007~2040 매일 -- Rust session_starts 대조(2007 전 미국 규칙은 안 옮김 · 그 앞 봉은 안 온다)
    sessions = [[d, *mctx.session_starts(d)] for d in range(13514, 25568)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scenarios": sc, "sessions": sessions}, ensure_ascii=False))
    print(f"{OUT.relative_to(ROOT)}: 시나리오 {len(sc)} · 요청 {sum(len(s['cases']) for s in sc)} · 세션 표 {len(sessions)}일 · "
          f"파이썬 {sys.version.split()[0]} · numpy {np.__version__} · pandas {pd.__version__}")


if __name__ == "__main__":
    main()
