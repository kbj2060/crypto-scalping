#!/usr/bin/env python3
"""rtserve 풋프린트 1주일 정답지(2026-11-28): /api/footprint-week.

원본을 옮겨 적었다(dashboard/server.py fp_week_part 4098~4155 · fp_week_sync 4157~4187 · fp_week_candles 4189~4211 · api_footprint_week 4213~4228 ·
dashboard/coin_flow.py okx_tape_footprint 132~206 -- Read 로 읽은 줄). week_bars·oi_deltas 는 원본 모듈 dashboard/footprint_daily.py, 칸 폭·고래 경계는
scripts/live_trade_tape_collector_20260916.py · scripts/live_okx_trade_tape_collector_20260923.py 를 그대로 불러 쓴다. kline_candles = gen_mhist_fixture(같은 원본 옮김).
바깥 접속 없음: klines = «녹화된 응답 표» · hot SQLite 다섯 파일은 여기서 만든다.
    $PY rtserve/tools/gen_fpweek_fixture.py    # 저장소 루트에서
출력 rtserve/tests/fixtures/gen/fpweek/fpweek.json(커밋 안 함).
"""
from __future__ import annotations

import base64
import json
import random
import sqlite3
import sys
import tempfile
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_mhist_fixture as M  # noqa: E402
from dashboard import footprint_daily as fpd  # noqa: E402
from scripts import live_okx_trade_tape_collector_20260923 as OKXC  # noqa: E402
from scripts import live_trade_tape_collector_20260916 as BNC  # noqa: E402

OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "fpweek" / "fpweek.json"
FOOTPRINT_BAR_SECONDS, FP_WEEK_DAYS, FP_WEEK_VWMA_PRE_S = 300, 7, 100 * 3600
TREND_VETO_N = 144
SPECS = {"eth": (0.5, 2), "sol": (0.02, 2), "xrp": (0.0003, 4), "btc": (15.0, 0)}   # coin_flow FLOW_SPECS 272~281(이 시험이 켜는 코인)
TMP = Path(tempfile.mkdtemp())
HOT_TAPE_DB, BINANCE_CTX_DB, BYBIT_LIQ_DB = TMP / "binance_tape.sqlite", TMP / "binance_ctx.sqlite", TMP / "bybit_liq.sqlite"
OKX_HOT_TAPE_DB, OKX_HOT_CTX_DB = TMP / "okx_tape.sqlite", TMP / "okx_ctx.sqlite"
OKX_LIQ_DB_PATH = OKX_HOT_CTX_DB
LIVE_DIR = TMP / "live"
fp_week_days: dict = {}


def read_rows(path, sql, params):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


_read_only_rows = read_rows


class Spec:
    def __init__(self, asset):
        self.asset, self.symbol = asset, asset.upper() + "USDT"
        self.bucket, self.price_dp = SPECS[asset]
        self.okx_inst, self.okx_tape_db, self.okx_ctx_db = f"{asset.upper()}-USDT-SWAP", OKX_HOT_TAPE_DB, OKX_HOT_CTX_DB


class Flow:
    def __init__(self, asset):
        self.spec = Spec(asset)
        self.retail_max, self.whale_min = BNC.size_bands(self.spec.symbol)


def okx_tape_footprint(tape_db, ctx_db, inst, lo_sec, t0_ms, bar_seconds, bucket, tape_bucket=None, liq_db=None):   # coin_flow 132
    t0_sec = t0_ms // 1000
    q = """SELECT ts_sec - ts_sec % ? AS bar, price_bin,
                  sum(buy_qty), sum(sell_qty),
                  sum(coalesce(whale_buy_qty, 0)), sum(coalesce(whale_sell_qty, 0)),
                  sum(coalesce(retail_buy_qty, 0)), sum(coalesce(retail_sell_qty, 0))
           FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY 1, 2"""
    tape_bucket = tape_bucket or OKXC.BUCKETS[inst]
    bars = {}
    key = {}
    for bar, pb, *v in _read_only_rows(tape_db, q, [bar_seconds, inst, lo_sec, t0_sec]):
        k = key.get(pb)
        if k is None:
            k = key[pb] = int(round(round(int(pb) * tape_bucket, 10) / bucket))
        t = bars.setdefault(int(bar), {}).setdefault(k, [0.0] * 6)
        for j in range(6):
            t[j] += float(v[j] or 0.0)
    ver = _read_only_rows(tape_db, "SELECT ts_min, rel_err FROM verify_1m WHERE symbol = ? AND ts_min >= ?", [inst, lo_sec])
    gaps = _read_only_rows(tape_db, "SELECT from_ms, to_ms FROM gaps WHERE symbol = ? AND to_ms > ?", [inst, lo_sec * 1000])
    good = {int(m) for m, r in ver if abs(r) <= 1e-4}
    seen = {int(m) for m, _ in ver}
    now = int(M.time.time())
    ok_min = set(good)
    for m in range(lo_sec, t0_sec + 60, 60):
        if m not in seen and m >= now - 900 and not any(a < (m + 60) * 1000 and b > m * 1000 for a, b in gaps):
            ok_min.add(m)
    oi = {}
    ctx_ok = True
    try:
        for ts, v in _read_only_rows(ctx_db, "SELECT ts_ms, oi_base FROM okx_oi WHERE inst = ? AND ts_ms >= ? "
                                             "AND ts_ms < ? ORDER BY ts_ms", [inst, lo_sec * 1000, t0_ms]):
            bar = int(ts) // 1000 // bar_seconds * bar_seconds
            oi[bar] = (oi[bar][0], float(v)) if bar in oi else (float(v), float(v))
    except sqlite3.Error:
        ctx_ok = False
    liq = []
    try:
        for ts, pos, qty, px in _read_only_rows(liq_db or ctx_db, """
                SELECT ts_ms, pos_side, sz_base, bk_px FROM okx_liquidations
                WHERE inst_id = ? AND ts_ms >= ? AND ts_ms < ? AND sz_base > 0 ORDER BY ts_ms""", [inst, lo_sec * 1000, t0_ms]):
            liq.append({"ts_ms": int(ts), "side": "long" if pos == "long" else "short", "qty": float(qty), "price": float(px or 0.0),
                        "usd": float(qty) * float(px or 0.0), "symbol": inst})
    except sqlite3.Error:
        ctx_ok = False
    tape_max = _read_only_rows(tape_db, "SELECT max(ts_sec) FROM trade_tape_1s WHERE symbol = ?", [inst])
    return {"bars": bars, "ok_min": ok_min, "oi": oi, "liq": liq, "tape_max": int(tape_max[0][0] or 0), "ctx_ok": ctx_ok}


def fp_week_part(f, d0, d1):   # server 4098
    spec, sym = f.spec, f.spec.symbol.lower()
    fails = []

    def rows(path, sql, params):
        if not path.exists():
            return []
        try:
            return read_rows(path, sql, params)
        except sqlite3.Error as exc:
            fails.append(f"{path.name}: {exc!r}")
            return []
    bn = okx_tape_footprint(HOT_TAPE_DB, LIVE_DIR / "_no_ctx.duckdb", sym, d0, d1 * 1000, FOOTPRINT_BAR_SECONDS, spec.bucket, BNC.BUCKETS[sym])
    ox, okx_on = {"bars": {}, "ok_min": set(), "oi": {}, "liq": [], "ctx_ok": True}, spec.okx_tape_db.exists()
    if okx_on:
        try:
            ox = okx_tape_footprint(spec.okx_tape_db, spec.okx_ctx_db, spec.okx_inst, d0, d1 * 1000, FOOTPRINT_BAR_SECONDS, spec.bucket, liq_db=OKX_LIQ_DB_PATH)
            okx_on = ox["tape_max"] > 0
        except sqlite3.Error as exc:
            fails.append(f"okx tape: {exc!r}")
    if not ox["ctx_ok"]:
        fails.append("okx oi/liq")
    bar = lambda ms: int(ms) // 1000 // FOOTPRINT_BAR_SECONDS * FOOTPRINT_BAR_SECONDS   # noqa: E731
    cells = fpd.week_bars(bn, None, d0, d1, FOOTPRINT_BAR_SECONDS)
    oi_bn = {}
    for ts, v in rows(BINANCE_CTX_DB, "SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms >= ? AND ts_ms < ? "
                                      "ORDER BY ts_ms", [sym, d0 * 1000, d1 * 1000]):
        b = bar(ts)
        oi_bn[b] = (oi_bn[b][0], float(v)) if b in oi_bn else (float(v), float(v))
    liq = {}

    def put(ms, is_long, usd, venue):
        a = liq.setdefault(bar(ms), [0.0] * 9)
        for j in (0, venue):
            a[j + (0 if is_long else 1)] += usd
            a[j + 2] += 1
    for ts, sd, usd in rows(BINANCE_CTX_DB, "SELECT ts_ms, side, usd FROM liquidations WHERE symbol = ? AND ts_ms >= ? AND ts_ms < ?",
                            [spec.symbol, d0 * 1000, d1 * 1000]):
        put(ts, sd == "long", float(usd or 0.0), 0)
    for e in ox["liq"]:
        put(e["ts_ms"], e["side"] == "long", e["usd"], 3)
    for ts, sd, q, px in rows(BYBIT_LIQ_DB, "SELECT ts_ms, side, qty, price FROM bybit_liquidations WHERE symbol = ? "
                                            "AND ts_ms >= ? AND ts_ms < ?", [spec.symbol, d0 * 1000, d1 * 1000]):
        put(ts, sd == "Buy", float(q) * float(px), 6)
    dp = spec.price_dp
    return {"bars": [{"time": b, "levels": [[round(k * spec.bucket, dp)] + [round(x, 3) for x in v] for k, v in sorted(c.items())]}
                     for b, c in sorted(cells.items())],
            "oi_bn": oi_bn, "oi_ox": dict(ox["oi"]), "liq": liq, "n": len(cells), "ok": not fails}


def fp_week_sync(f, now):   # server 4157
    today = int(now) // 86400 * 86400
    first = today - FP_WEEK_DAYS * 86400
    parts = []
    for d in range(first, today, 86400):
        key = (f.spec.asset, d)
        p = fp_week_days.get(key) or fp_week_part(f, d, d + 86400)
        if p["n"] == 86400 // FOOTPRINT_BAR_SECONDS and p["ok"]:
            fp_week_days[key] = p
        parts.append(p)
    parts.append(fp_week_part(f, today, int(now) // FOOTPRINT_BAR_SECONDS * FOOTPRINT_BAR_SECONDS))
    for k in [k for k in list(fp_week_days) if k[1] < first]:
        fp_week_days.pop(k, None)
    oi_bn, oi_ox, liq = {}, {}, {}
    for p in parts:
        oi_bn.update(p["oi_bn"]); oi_ox.update(p["oi_ox"]); liq.update(p["liq"])
    dbn, dox = fpd.oi_deltas(oi_bn, FOOTPRINT_BAR_SECONDS), fpd.oi_deltas(oi_ox, FOOTPRINT_BAR_SECONDS)
    oi = [[b, round(d + (dox[b][0] if dox else 0.0), 3), round(c + (dox[b][1] if dox else 0.0), 3), 0, g]
          for b, (d, c, g) in sorted(dbn.items()) if not dox or b in dox]
    det = lambda a, j: {"long_usd": round(a[j]), "short_usd": round(a[j + 1]), "n": int(a[j + 2])} if a[j + 2] else None   # noqa: E731
    liq_out = [{"ts": datetime.fromtimestamp(b, timezone.utc).isoformat(), "long_usd": round(a[0], 2), "short_usd": round(a[1], 2),
                "events": int(a[2]), **{k: v for k, v in (("okx_detail", det(a, 3)), ("bybit", det(a, 6))) if v}}
               for b, a in sorted(liq.items())]
    return {"bars": [b for p in parts for b in p["bars"]], "oi": oi, "liq": liq_out}


def fp_week_candles(asset):   # server 4191
    week0 = (int(M.time.time()) // 86400 - FP_WEEK_DAYS) * 86400
    start_ms = (week0 - max(TREND_VETO_N * 5 * FOOTPRINT_BAR_SECONDS, FP_WEEK_VWMA_PRE_S)) * 1000
    rows = []
    while len(rows) < 4000:
        got = M.fetch_binance_json("https://fapi.binance.com/fapi/v1/klines",
                                   {"symbol": M.MARKET_SYMBOLS[asset], "interval": "5m", "startTime": start_ms, "limit": 1000},
                                   timeout=5.0, error_reason="fp_week_klines")
        rows += got or []
        if not got or len(got) < 1000:
            break
        start_ms = int(got[-1][0]) + FOOTPRINT_BAR_SECONDS * 1000
    now_ms = M.time.time() * 1000
    rows = [r for r in rows if int(r[6]) < now_ms]
    if not rows:
        return [], []
    pre = [[int(r[0]) // 1000, float(r[4]), float(r[5])] for r in rows if week0 - FP_WEEK_VWMA_PRE_S <= int(r[0]) // 1000 < week0]
    return [c for c in M.kline_candles(rows, now_ms, 4 if asset == "eth" else 6, len(rows)) if c["time"] >= week0], pre


FLOWS = {}


def api_footprint_week(q):   # server 4213 (flow_for = 꺼진 코인 404 flow_off)
    a = q.get("asset", "eth").lower() or "eth"
    if a not in FLOWS:
        raise NotFound("flow_off")
    f = FLOWS[a]
    asset = f.spec.asset

    def produce():
        try:
            candles, vol_pre = M.swr_cached(f"fp_week_k5_{asset}", lambda: fp_week_candles(asset))
        except M.HTTPException:
            candles, vol_pre = [], []
        body = fp_week_sync(f, M.time.time())
        return {"symbol": f.spec.symbol, "bucket": f.spec.bucket, "barSeconds": FOOTPRINT_BAR_SECONDS,
                "retailMaxUsd": f.retail_max, "whaleMinUsd": f.whale_min, "days": FP_WEEK_DAYS,
                "candles": candles, "volPre": vol_pre, **body, "at": M.time.time()}
    return 200, M.swr_cached(f"fp_week_{asset}", produce)


class NotFound(M.HTTPException):
    status = 404


def db(path, ddl, rows):
    con = sqlite3.connect(path)
    for d in ddl:
        con.execute(d)
    for tbl, rs in rows.items():
        if rs:
            con.executemany(f"INSERT INTO {tbl} VALUES ({','.join('?' * len(rs[0]))})", rs)
    con.commit()
    con.close()


TAPE_DDL = ["CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL, whale_buy_qty REAL, "
            "whale_sell_qty REAL, retail_buy_qty REAL, retail_sell_qty REAL)",
            "CREATE TABLE verify_1m(symbol TEXT, ts_min INTEGER, rel_err REAL)", "CREATE TABLE gaps(symbol TEXT, from_ms INTEGER, to_ms INTEGER)"]


def main():
    rnd = random.Random(20261128)
    now = datetime(2026, 11, 26, 10, 7, 41, 250000, tzinfo=timezone.utc).timestamp()
    M.NOW = now
    ni = int(now)
    today = ni // 86400 * 86400
    first = today - 7 * 86400
    end = ni // 300 * 300
    tape, ver, gaps = [], [], []
    px0 = {"ethusdt": 3000.0, "btcusdt": 91000.0, "xrpusdt": 2.2}
    for sym, p in px0.items():
        tb = BNC.BUCKETS[sym]
        for b in range(first, end, 300):
            p *= 1 + rnd.gauss(0, 0.002)
            for s in (rnd.randrange(0, 300), rnd.randrange(0, 300)):   # 봉 하나에 두 초 · 칸 둘 이상
                for dk in range(rnd.randrange(1, 4)):
                    pb = int(round(p / tb)) + dk * 3
                    v = [round(rnd.uniform(0, 3), 4) for _ in range(6)]
                    tape.append((sym, b + s, pb, v[0], v[1], v[2] if rnd.random() > 0.2 else None, v[3], v[4], v[5]))
        for m in range(first, ni + 60, 60):
            if ni - m < 840 and m % 120 == 0:
                continue                                      # 최근 14분 검사 전(15분 안이면 공백 대조로 채움)
            if sym == "ethusdt" and first + 2 * 86400 + 3000 <= m < first + 2 * 86400 + 3300:
                ver.append((sym, m, 0.01))                    # 대조 불일치 → 그 날 덜 참(안 굳힘)
            else:
                ver.append((sym, m, 0.0 if rnd.random() > 0.3 else 5e-5))
    gaps.append(("ethusdt", (ni - 300) * 1000, (ni - 240) * 1000))
    db(HOT_TAPE_DB, TAPE_DDL, {"trade_tape_1s": tape, "verify_1m": ver, "gaps": gaps})
    # OKX: ETH 만 테이프 · OI·청산
    otape = [("ETH-USDT-SWAP", first + 10, 30000, 1.0, 2.0, 0, 0, 1.0, 2.0)]
    db(OKX_HOT_TAPE_DB, TAPE_DDL, {"trade_tape_1s": otape, "verify_1m": [], "gaps": []})
    ooi = [("ETH-USDT-SWAP", (first + i * 97) * 1000, 3_000_000 + rnd.uniform(-5000, 5000)) for i in range(0, (ni - first) // 97)]
    oliq = [("ETH-USDT-SWAP", (first + rnd.randrange(0, ni - first)) * 1000, rnd.choice(["long", "short"]), round(rnd.uniform(0.1, 30), 3), rnd.choice([None, 2999.5, 3012.25]))
            for _ in range(40)]
    db(OKX_HOT_CTX_DB, ["CREATE TABLE okx_oi(inst TEXT, ts_ms INTEGER, oi_base REAL)",
                        "CREATE TABLE okx_liquidations(inst_id TEXT, ts_ms INTEGER, pos_side TEXT, sz_base REAL, bk_px REAL)"],
       {"okx_oi": ooi, "okx_liquidations": oliq})
    boi = [(s, (first + i * 61) * 1000, 2_000_000 + i * 0.37) for s in ("ethusdt", "btcusdt") for i in range(0, (ni - first) // 61)]
    boi = [r for r in boi if not (r[0] == "btcusdt" and first + 86400 <= r[1] // 1000 < first + 86400 + 3600)]   # BTC OI 한 시간 빔(공백 봉)
    bliq = [(s, (first + rnd.randrange(0, ni - first)) * 1000, rnd.choice(["long", "short"]), rnd.choice([None, round(rnd.uniform(10, 90000), 2)]))
            for s in ("ETHUSDT", "BTCUSDT") for _ in range(60)]
    db(BINANCE_CTX_DB, ["CREATE TABLE oi_1s(symbol TEXT, ts_ms INTEGER, open_interest REAL)", "CREATE TABLE liquidations(symbol TEXT, ts_ms INTEGER, side TEXT, usd REAL)"],
       {"oi_1s": boi, "liquidations": bliq})
    yliq = [(s, (first + rnd.randrange(0, ni - first)) * 1000, rnd.choice(["Buy", "Sell"]), round(rnd.uniform(0.01, 5), 3), round(rnd.uniform(2900, 3100), 2))
            for s in ("ETHUSDT", "XRPUSDT") for _ in range(30)]
    db(BYBIT_LIQ_DB, ["CREATE TABLE bybit_liquidations(symbol TEXT, ts_ms INTEGER, side TEXT, qty REAL, price REAL)"], {"bybit_liquidations": yliq})

    # klines: ETH = 네 쪽(1000·1000·1000·나머지) · BTC = 둘째 쪽 실패(502 → 캔들 없이) · XRP = 빈 목록
    week0 = (ni // 86400 - 7) * 86400
    k0 = (week0 - 360000) * 1000
    eth = M.kl(k0, (ni * 1000 - k0) // 300_000 + 1, 3000.0, rnd, 2)          # 마지막 = 형성 봉(버림)
    start, i = k0, 0
    while True:
        page = eth[i:i + 1000]
        M.REST[M.FAPI + "?" + urllib.parse.urlencode({"symbol": "ETHUSDT", "interval": "5m", "startTime": start, "limit": 1000})] = (200, json.dumps(page))
        if len(page) < 1000:
            break
        i += 1000
        start = page[-1][0] + 300_000
    btc = M.kl(k0, 1000, 91000.0, rnd, 1)
    M.REST[M.FAPI + "?" + urllib.parse.urlencode({"symbol": "BTCUSDT", "interval": "5m", "startTime": k0, "limit": 1000})] = (200, json.dumps(btc))
    M.REST[M.FAPI + "?" + urllib.parse.urlencode({"symbol": "BTCUSDT", "interval": "5m", "startTime": btc[-1][0] + 300_000, "limit": 1000})] = (502, "x")
    M.REST[M.FAPI + "?" + urllib.parse.urlencode({"symbol": "XRPUSDT", "interval": "5m", "startTime": k0, "limit": 1000})] = (200, "[]")

    for a in ("eth", "btc", "sol", "xrp"):
        FLOWS[a] = Flow(a)
    paths = [f"/api/footprint-week?asset={a}" for a in ("eth", "ETH", "btc", "xrp", "doge")] + ["/api/footprint-week"]
    cases = []
    for p in paths:
        M.SENT.clear()
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(p).query, keep_blank_values=True))
        try:
            status, v = api_footprint_week(q)
            body = json.dumps(v)
        except M.HTTPException as exc:
            status, body = exc.status, f"{exc.status}: {exc.reason}"
        cases.append({"path": p, "status": status, "body": body, "sent": list(M.SENT)})
    files = {f"data/hot/{p.name}": base64.b64encode(p.read_bytes()).decode() for p in (HOT_TAPE_DB, BINANCE_CTX_DB, BYBIT_LIQ_DB, OKX_HOT_TAPE_DB, OKX_HOT_CTX_DB)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scenarios": [{"now": now, "rest": {k: list(v) for k, v in M.REST.items()}, "files": files, "cases": cases}]}, ensure_ascii=False))
    print(f"{OUT.relative_to(ROOT)}: 요청 {len(cases)} · 본문 {[len(c['body']) for c in cases]} · 굳힌 날 {len(fp_week_days)} · 파이썬 {sys.version.split()[0]}")


if __name__ == "__main__":
    main()
