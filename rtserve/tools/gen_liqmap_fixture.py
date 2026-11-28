#!/usr/bin/env python3
"""rtserve 청산맵 정답지(2026-11-28): /api/liquidation-map · /api/liquidation-map/tiers.

원본 핸들러·적재 함수를 옮겨 적었다(dashboard/server.py liq_klines_df 3322~3345 · liq_oi_usd 3347~3365 · load_liquidation_tiers 3367~3378 ·
load_liquidation_map 3469~3516 · api_liquidation_map 4804~4813 · api_liquidation_tiers 4877~4883 -- Read 로 읽은 줄). 계산은 원본 모듈
scripts/live_liquidation_map_20260824.py 를 그대로 불러 쓴다. 바깥 접속 없음: fetch_binance_json 자리에 «녹화된 응답 표»(Rust 시험은 같은 표를 127.0.0.1 가짜 REST 로).
    source <(bash rtmodel/tools/server_env.sh <venv> --env) && $PY rtserve/tools/gen_liqmap_fixture.py    # 저장소 루트에서
출력 rtserve/tests/fixtures/gen/liqmap/liqmap.json(커밋 안 함).
"""
from __future__ import annotations

import base64
import json
import random
import sqlite3
import sys
import tempfile
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd  # noqa: E402
from live_liquidation_map_20260824 import (RECENCY_HALFLIFE_HOURS, compute_spliced_heatmap_history_5m, compute_spliced_levels,  # noqa: E402
                                           compute_tier_profile, latest_window_5m)

assert sys.version_info[:2] == (3, 11), sys.version
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "liqmap" / "liqmap.json"
NOW = 1_795_000_123.456789
time.time = lambda: NOW   # 원본의 time.time() 자리(형성 봉 버림 · OI 10분)
FAPI = "/fapi/v1/klines"
COIN_CONFIG = {a: {"binance_symbol": a.upper() + "USDT"} for a in ("eth", "btc", "sol", "xrp", "hype", "sndk", "soxl", "skhynix", "mu")}
LIQUIDATION_MAP_INTERVAL, LIQUIDATION_MAP_LOOKBACK_HOURS, LIQUIDATION_MAP_FETCH_LIMIT = "1h", 24, 48
LIQUIDATION_MAP_CACHE_SECONDS, LIQUIDATION_MAP_HISTORY_SNAPSHOTS, LIQUIDATION_MAP_5M_FETCH_LIMIT, STALE_GRACE_SECONDS = 60, 144, 460, 600
TMP = Path(tempfile.mkdtemp())
BINANCE_CTX_DB, OKX_CTX_DB_PATH, BYBIT_CTX_DB = TMP / "binance_ctx.sqlite", TMP / "okx_ctx.sqlite", TMP / "bybit_ctx.sqlite"


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
liq_hist_memo: dict[str, dict] = {}


def fetch_binance_json(path, params, *, error_reason=None):
    target = path + "?" + urllib.parse.urlencode(params)
    SENT.append(target)
    status, body = REST.get(target, (404, "[]"))
    if status != 200:
        if error_reason is None:
            return None
        raise HTTPBadGateway(error_reason)
    return json.loads(body)


def swr_cached(key, produce):   # 이 시나리오는 신선 구간 안(같은 키 = 캐시 값 · 실패는 캐시 안 함)
    if key not in SWR:
        SWR[key] = produce()
    return SWR[key]


def read_rows(path, sql, params):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


_read_only_rows = read_rows


def liq_klines_df(asset, limit, keep_forming=False, interval=LIQUIDATION_MAP_INTERVAL):   # 3322
    url, q = FAPI, {"symbol": COIN_CONFIG[asset]["binance_symbol"], "interval": interval, "limit": min(limit, 1500)}
    raw = fetch_binance_json(url, q, error_reason="liquidation_map_upstream_error")
    while raw and len(raw) < limit and len(raw) % 1500 == 0:
        more = fetch_binance_json(url, {**q, "limit": min(limit - len(raw), 1500), "endTime": int(raw[0][0]) - 1},
                                  error_reason="liquidation_map_upstream_error")
        if not more:
            break
        raw = more + raw
        if len(more) < 1500:
            break
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    df["close_time"] = df["close_time"].astype("int64")
    df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    now_ms = int(time.time() * 1000)
    if not keep_forming and len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
        df = df.iloc[:-1].reset_index(drop=True)
    return df


def liq_oi_usd(asset, current_price):   # 3347
    sym = COIN_CONFIG[asset]["binance_symbol"]
    try:
        r = read_rows(BINANCE_CTX_DB, "SELECT open_interest FROM oi_1s WHERE symbol = ? ORDER BY ts_ms DESC LIMIT 1", [sym.lower()])
    except Exception:  # noqa: BLE001
        return None
    if not (r and r[0][0]):
        return None
    coins, since = float(r[0][0]), int(time.time() * 1000) - 600_000
    base = sym[:-4] if sym.endswith("USDT") else sym
    for path, tbl, key in ((OKX_CTX_DB_PATH, "okx_oi", f"{base}-USDT-SWAP"), (BYBIT_CTX_DB, "bybit_oi", sym)):
        try:
            x = _read_only_rows(path, f"SELECT oi_base FROM {tbl} WHERE inst = ? AND ts_ms >= ? ORDER BY ts_ms DESC LIMIT 1", [key, since]) if path.exists() else []
        except Exception:  # noqa: BLE001
            x = []
        coins += float(x[0][0]) if x and x[0][0] else 0.0
    return coins * current_price


def now_iso():
    return datetime.fromtimestamp(NOW, timezone.utc).isoformat()


def load_liquidation_tiers(asset, days):   # 3367
    def produce():
        df = (liq_klines_df(asset, days * 24 + 2, keep_forming=True)).tail(days * 24).reset_index(drop=True)
        cp = float(df["close"].iloc[-1]) if len(df) else 0.0
        oi_usd = liq_oi_usd(asset, cp)
        tp = compute_tier_profile(df, cp, oi_usd, None, None if days >= 60 else RECENCY_HALFLIFE_HOURS)
        return {"days": days, "tier_profile": tp, "decay": days < 60, "generated_at": now_iso()}
    return swr_cached(f"liquidation_tiers:{asset}:{days}", produce)


def load_liquidation_map(asset):   # 3469
    def produce():
        df = liq_klines_df(asset, LIQUIDATION_MAP_FETCH_LIMIT)
        df_5m = liq_klines_df(asset, LIQUIDATION_MAP_5M_FETCH_LIMIT, interval="5m")
        current_price = float(df["close"].iloc[-1]) if len(df) else 0.0
        payload = compute_spliced_levels(df.tail(LIQUIDATION_MAP_LOOKBACK_HOURS).reset_index(drop=True), current_price, df_5m)
        payload["heatmap_history"] = compute_spliced_heatmap_history_5m(df, df_5m, current_price, LIQUIDATION_MAP_LOOKBACK_HOURS,
                                                                        LIQUIDATION_MAP_HISTORY_SNAPSHOTS, liq_hist_memo.setdefault(asset, {}))
        win = latest_window_5m(df, df_5m, LIQUIDATION_MAP_LOOKBACK_HOURS)
        tp = None
        if win is not None and len(win[0]):
            end_px = float(win[0]["close"].iloc[-1])
            oi_usd = liq_oi_usd(asset, end_px)
            tp = compute_tier_profile(win[0], current_price, oi_usd, df_5m)
            if tp is not None:
                tp.update(current_price=end_px, known_until=win[1].isoformat())
        if len(df):
            payload.update(levels_price=current_price, levels_known_until=(df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)).isoformat())
        payload["tier_profile"] = tp
        payload["generated_at"] = now_iso()
        return payload
    return swr_cached(f"liquidation_map:{asset}", produce)


def _query_coin_asset(q):
    asset = q.get("asset", "eth").lower()
    if asset not in COIN_CONFIG:
        raise HTTPBadRequest("unsupported_asset")
    return asset


def api_liquidation_map(q):   # 4804
    try:
        return 200, load_liquidation_map(_query_coin_asset(q))
    except HTTPBadGateway:
        return 502, {"error": "liquidation_map_upstream_error", "detail": "Binance klines fetch failed."}


def api_liquidation_tiers(q):   # 4877
    days = int(q.get("days", "1")) if q.get("days", "1") in ("1", "7", "30", "60", "90", "180", "365") else 1
    try:
        return 200, load_liquidation_tiers(_query_coin_asset(q), days)
    except HTTPBadGateway:
        return 502, {"error": "liquidation_map_upstream_error"}


def kl(t0, step, closes, rnd, dp=2, vol=(1, 500)):
    out, prev = [], closes[0]
    for i, c in enumerate(closes):
        o = prev
        h = max(o, c) * (1 + rnd.uniform(0, 0.004))
        lo = min(o, c) * (1 - rnd.uniform(0, 0.004))
        t = t0 + i * step
        out.append([t, f"{o:.{dp}f}", f"{h:.{dp}f}", f"{lo:.{dp}f}", f"{c:.{dp}f}", f"{rnd.uniform(*vol):.3f}", t + step - 1, "0", 10, "0", "0", "0"])
        prev = c
    return out


def walk(rnd, n, c0, sd):
    c, out = c0, []
    for _ in range(n):
        c *= 1 + rnd.gauss(0, sd)
        out.append(c)
    return out


def put(sym, iv, limit, body, end=None):
    q = {"symbol": sym, "interval": iv, "limit": limit}
    if end is not None:
        q["endTime"] = end
    REST[FAPI + "?" + urllib.parse.urlencode(q)] = body


def paged(sym, rows, limit):
    """원본 쪽 나누기 그대로 표를 채운다(마지막 쪽부터 1,500씩 · endTime = 앞 쪽 첫 봉 − 1)."""
    tail = rows[-min(limit, 1500):]
    put(sym, "1h", min(limit, 1500), (200, json.dumps(tail)))
    have = len(tail)
    while have < limit and have % 1500 == 0:
        first = rows[-have][0]
        lim = min(limit - have, 1500)
        more = [r for r in rows if r[0] <= first - 1][-lim:]
        put(sym, "1h", lim, (200, json.dumps(more)), first - 1)
        if not more:
            break
        have += len(more)
        if len(more) < 1500:
            break


def main():
    rnd = random.Random(20261128)
    H, M = 3_600_000, 300_000
    now_ms = int(NOW * 1000)
    h_last, m_last = now_ms // H * H, now_ms // M * M   # 형성 중인 봉 시작
    sc = {"ETHUSDT": (3000, 2), "BTCUSDT": (95000, 1), "SOXLUSDT": (0.6123, 4), "SOLUSDT": (150, 2), "MUUSDT": (110, 2)}
    for sym, (p0, dp) in sc.items():
        n5 = 460
        c5 = walk(rnd, n5, p0, 0.003 if sym != "BTCUSDT" else 0.006)
        m5 = kl(m_last - (n5 - 1) * M, M, c5, rnd, dp)          # 마지막 = 형성 봉(버림)
        if sym == "BTCUSDT":
            del m5[300:303]                                       # 빠진 5분봉 -- 그 시간은 옛 규약(몫 1)
        h1 = kl(h_last - 47 * H, H, walk(rnd, 48, c5[0], 0.008), rnd, dp)
        if sym == "ETHUSDT":
            h1.insert(10, list(h1[10]))                           # 같은 시각 두 번(마지막 남김)
        if sym == "SOLUSDT":
            h1, m5 = h1[-19:], []                                 # 모자람 → insufficient_data · 5분봉 없음
        put(sym, "1h", 48, (200, json.dumps(h1)))
        put(sym, "5m", 460, (200, json.dumps(m5)))
    put("XRPUSDT", "1h", 48, (500, '{"code": -1}'))
    put("HYPEUSDT", "1h", 48, (200, json.dumps(kl(h_last - 47 * H, H, walk(rnd, 48, 30, 0.01), rnd, 3))))
    put("HYPEUSDT", "5m", 460, (502, "bad"))
    # 청산 막대 기간: 1·7·30 일(쪽 하나) · 90일(2,162 = 두 쪽) · 365일(8,762 = 여섯 쪽) -- 형성 봉 남김
    long_eth = kl(h_last - 8800 * H, H, walk(rnd, 8801, 1800, 0.006), rnd, 2)
    for d in (1, 7, 30, 90, 365):
        paged("ETHUSDT", long_eth, d * 24 + 2)
    btc_long = kl(h_last - 200 * H, H, walk(rnd, 201, 90000, 0.004), rnd, 1)
    for d in (7, 60):
        paged("BTCUSDT", btc_long, d * 24 + 2)
    put("SOLUSDT", "1h", 26, (502, "x"))
    # OI 파일: ETH = 바이낸스 + OKX(10분 안) + 바이비트(묵음 → 뺌) · MU = 바이낸스 0(거짓 → 비율) · BTC 없음
    for p, ddl, rows in ((BINANCE_CTX_DB, "CREATE TABLE oi_1s(symbol TEXT, ts_ms INTEGER, open_interest REAL)",
                          [("ethusdt", now_ms - 5000, 2_000_000.5), ("ethusdt", now_ms - 1000, 2_100_000.25), ("muusdt", now_ms - 1000, 0.0),
                           ("soxlusdt", now_ms - 1000, 1234.5)]),
                         (OKX_CTX_DB_PATH, "CREATE TABLE okx_oi(inst TEXT, ts_ms INTEGER, oi_base REAL)",
                          [("ETH-USDT-SWAP", now_ms - 2000, 500_000.0), ("SOXL-USDT-SWAP", now_ms - 700_000, 9.0)]),
                         (BYBIT_CTX_DB, "CREATE TABLE bybit_oi(inst TEXT, ts_ms INTEGER, oi_base REAL)",
                          [("ETHUSDT", now_ms - 900_000, 300_000.0), ("SOXLUSDT", now_ms - 1000, 11.0)])):
        con = sqlite3.connect(p)
        con.execute(ddl)
        con.executemany(f"INSERT INTO {ddl.split()[2].split('(')[0]} VALUES (?,?,?)", rows)
        con.commit()
        con.close()

    paths = [f"/api/liquidation-map?asset={a}" for a in ("eth", "BTC", "soxl", "sol", "xrp", "hype", "mu", "doge")] + ["/api/liquidation-map", "/api/liquidation-map?asset="]
    paths += [f"/api/liquidation-map/tiers?asset=eth&days={d}" for d in ("1", "7", "30", "90", "365", "abc", "")]
    paths += ["/api/liquidation-map/tiers?asset=btc&days=7", "/api/liquidation-map/tiers?asset=btc&days=60", "/api/liquidation-map/tiers?asset=sol",
              "/api/liquidation-map/tiers?asset=doge&days=7", "/api/liquidation-map/tiers?days=1"]
    cases = []
    for p in paths:
        SENT.clear()
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(p).query, keep_blank_values=True))
        h = api_liquidation_tiers if "/tiers" in p else api_liquidation_map
        try:
            status, v = h(q)
            body = json.dumps(v)
        except HTTPException as exc:
            status, body = exc.status, f"{exc.status}: {exc.reason}"
        cases.append({"path": p, "status": status, "body": body, "sent": list(SENT)})
    files = {f"data/hot/{p.name}": base64.b64encode(p.read_bytes()).decode() for p in (BINANCE_CTX_DB, OKX_CTX_DB_PATH, BYBIT_CTX_DB)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"now": NOW, "rest": {k: list(v) for k, v in REST.items()}, "files": files, "cases": cases}, ensure_ascii=False))
    print(f"{OUT.relative_to(ROOT)}: 요청 {len(cases)} · REST 표 {len(REST)} · 파이썬 {sys.version.split()[0]} · pandas {pd.__version__}")


if __name__ == "__main__":
    main()
