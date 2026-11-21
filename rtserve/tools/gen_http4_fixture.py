#!/usr/bin/env python3
"""rtserve 넷째 조각 정답지(2026-11-22): /api/oi-5m · /api/liquidation-5m-history.

원본 api_oi_5m(server.py:4363~4411)·oi_5m_buckets(983~1019) 를 옮겨 적고(실제 파일을 Read 로 읽은 줄), 흐름 엔진 메모리는 가짜로 둔다:
oi_1s 링 = hot oi_1s 의 마지막 행 하나(rtserve 는 링이 없어 hot 마지막 행을 쓴다 -- 그 몫이 같아지게) · okx_oi_5m(봉 → (첫, 끝)) · okx_fp first_bar.
rtserve 시험은 같은 okx_d(원본 루프가 만든 [봉, Δ, 끝])를 코어 Micro 메시지로 넣는다. 서버를 띄우지 않는다(gen_http_fixture 의 가짜 전송).

    python3 rtserve/tools/gen_http4_fixture.py     # 저장소 루트에서 · rtserve/tests/fixtures/http4.json
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_http_fixture as g1  # noqa: E402
from aiohttp import web  # noqa: E402

NOW, MTIME_NS, ROOT = g1.NOW, g1.MTIME_NS, g1.ROOT
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "http4.json"
read_rows = g1.read_rows
OI_5M_BAR_SECONDS, OI_5M_WINDOW_BARS = 300, 48            # coin_flow.py:105 · server.py:449
FLOW = [a for a in ("eth", "sol", "xrp", "btc", "sndk", "soxl", "skhynix", "mu") if a == "eth" or a in g1.DASHBOARD_ASSETS]


def make_app(R: Path, mem: dict) -> web.Application:
    OI_1S_DB_PATH, OI_1S_TABLE = R / "data" / "hot" / "binance_ctx.sqlite", "oi_1s"

    def oi_5m_buckets(bars, symbol):                                         # 983
        if not OI_1S_DB_PATH.exists():
            return []
        floor = (int(time.time()) // OI_5M_BAR_SECONDS - (bars - 1)) * OI_5M_BAR_SECONDS
        try:
            raw = read_rows(OI_1S_DB_PATH, f"SELECT ts_ms, open_interest FROM {OI_1S_TABLE} WHERE symbol = ? AND ts_ms >= ? "
                                           "ORDER BY ts_ms", [symbol, (floor - OI_5M_BAR_SECONDS) * 1000])
        except Exception as exc:  # noqa: BLE001
            print(f"oi-5m read failed: {exc}", flush=True)
            return []
        agg = {}
        for ts_ms, v in raw:
            bar = int(ts_ms) // 1000 // OI_5M_BAR_SECONDS * OI_5M_BAR_SECONDS
            a = agg.get(bar)
            if a is None:
                agg[bar] = [float(v), float(v), 1]
            else:
                a[1] = float(v); a[2] += 1
        rows = [(bar, o, c, n) for bar, (o, c, n) in sorted(agg.items())]
        out = []
        prev_bar = None
        prev_close = 0.0
        for bar, oi_open, oi_close, n in rows:
            bar = int(bar)
            gap = prev_bar is None or bar - prev_bar > OI_5M_BAR_SECONDS
            delta = (oi_close - oi_open) if gap else (oi_close - prev_close)
            prev_bar, prev_close = bar, oi_close
            if bar >= floor:
                out.append([bar, round(delta, 3), round(oi_close, 3), int(n), 1 if gap else 0])
        return out

    flows = {a: SimpleNamespace(**mem[a]) for a in FLOW}

    def flow_for(q):                                                         # 1725
        f = flows.get(str(q.get("asset") or "eth").lower())
        if f is None:
            raise web.HTTPNotFound(reason="flow_off")
        return f

    async def api_oi_5m(request):                                            # 4363 (swr 빼고)
        try:
            bars = max(1, min(288, int(request.query.get("bars", OI_5M_WINDOW_BARS))))
        except ValueError:
            bars = OI_5M_WINDOW_BARS
        f = flow_for(request.query)
        oi_1s, okx_oi_5m, okx_fp, FOOTPRINT_SYMBOL = f.oi_1s, f.okx_oi_5m, f.okx_fp, f.symbol
        sym = f.symbol.lower()
        buckets = oi_5m_buckets(bars, sym)
        now_oi = oi_1s[max(oi_1s)] if oi_1s else (buckets[-1][2] if buckets else 0.0)
        okx_d = {}
        prev = None
        for ob in sorted(okx_oi_5m):
            o, c = okx_oi_5m[ob]
            okx_d[ob] = ((c - o) if prev is None or ob - prev[0] > OI_5M_BAR_SECONDS else (c - prev[1]), c)
            prev = (ob, c)
        okx_from = okx_fp["first_bar"]
        bars_out = ([[b[0], round(b[1] + okx_d[b[0]][0], 3), round(b[2] + okx_d[b[0]][1], 3)] + b[3:]
                     for b in buckets if b[0] > okx_from and b[0] in okx_d]
                    if okx_d else buckets)
        mem_okx_d[sym] = [[b, d, c] for b, (d, c) in okx_d.items()]
        return web.json_response({
            "symbol": FOOTPRINT_SYMBOL, "barSeconds": OI_5M_BAR_SECONDS, "openInterest": round(float(now_oi), 3),
            "venues": ["binance-perp", "okx-swap"] if okx_d else ["binance-perp"], "bars": bars_out}, headers=g1.NOCACHE)

    mem_okx_d = mem["_okx_d"]
    app = web.Application(middlewares=[g1.static_asset_headers, g1.json_compress_etag])
    app.router.add_get("/api/oi-5m", api_oi_5m)
    app.router.add_static("/dashboard/live/", R / "dashboard" / "live", show_index=False)
    return app


def tree() -> tuple[dict[str, bytes], dict]:
    rnd = random.Random(5)
    now = int(NOW)
    rows = []
    for sym, base in (("ethusdt", 2_300_000.0), ("solusdt", 9_100_000.0)):
        oi = base
        for s in range(now - 20 * 3600, now, 1):
            if sym == "ethusdt" and now - 9 * 3600 < s < now - 8 * 3600 - 1000:   # 수집기 공백 → gap
                continue
            if (now // 300 - 10) * 300 <= s < (now // 300 - 9) * 300:   # 봉 하나만 빔(앞뒤 봉 간격 600초 = 공백 경계)
                continue
            if s % 3:
                continue
            oi += rnd.gauss(0, 15.0)
            rows.append((sym, s * 1000 + rnd.randint(0, 999), round(oi, 3)))
    db = g1.sqlite_bytes([("CREATE TABLE oi_1s (symbol TEXT, ts_ms INTEGER, open_interest REAL)", []),
                          ("INSERT INTO oi_1s VALUES (?,?,?)", rows)])
    last = {s: max((r for r in rows if r[0] == s), key=lambda r: r[1]) for s in ("ethusdt", "solusdt")}
    bar = now // 300 * 300
    okx = {b: (500_000.0 + (b - bar) / 300 * 3.5, 500_010.25 + (b - bar) / 300 * 3.5) for b in range(bar - 30 * 300, bar + 1, 300) if b != bar - 12 * 300}
    mem = {"_okx_d": {},
           "eth": {"symbol": "ETHUSDT", "oi_1s": {last["ethusdt"][1] // 1000: last["ethusdt"][2]}, "okx_oi_5m": okx, "okx_fp": {"first_bar": bar - 20 * 300}},
           "sol": {"symbol": "SOLUSDT", "oi_1s": {last["solusdt"][1] // 1000: last["solusdt"][2]}, "okx_oi_5m": {}, "okx_fp": {"first_bar": 0}},
           "xrp": {"symbol": "XRPUSDT", "oi_1s": {}, "okx_oi_5m": {}, "okx_fp": {"first_bar": 0}}}
    return {"data/hot/binance_ctx.sqlite": db}, mem


# ───────── /api/liquidation-5m-history: liq_5m_add 537前 516~535 · liq_5m_payload 537~555 · merge_bybit_liq 833~853 · okx_liq_hot_bars 856~875 ·
#           merge_okx_liq 878~904 · merge_hl_liq 907~928 · hl_whale_liq_events 504~513 · 핸들러 4413~4473 · footprint_window_bars 437~447 (전부 Read 로 읽은 줄)
from datetime import datetime, timezone  # noqa: E402

BAR, KEEP, FP_BARS = 300, 288, 12
COIN_CONFIG = ("eth", "btc", "sol", "xrp", "hype", "sndk", "soxl", "skhynix", "mu")


def liq_5m_add(store, ev, bar_s=BAR, keep_bars=KEEP):
    bar = int(ev.get("ts_ms") or 0) // 1000 // bar_s * bar_s
    a = store.setdefault(bar, [0.0, 0.0, 0, None, None])
    is_long = ev.get("side") == "long"
    a[0 if is_long else 1] += float(ev.get("usd") or 0.0)
    a[2] += 1
    px = float(ev.get("price") or 0.0)
    if px > 0:
        if is_long:
            a[3] = px if a[3] is None else min(a[3], px)
        else:
            a[4] = px if a[4] is None else max(a[4], px)
    cut = bar - keep_bars * bar_s
    for old in [b for b in store if b < cut]:
        del store[old]


def liq_5m_payload(store, from_s, bars, now_s, bar_s=BAR):
    now_bar = int(now_s) // bar_s * bar_s
    from_bar = int(from_s) // bar_s * bar_s
    out = []
    for k in range(bars - 1, -1, -1):
        b = now_bar - k * bar_s
        if b <= from_bar:
            continue
        a = store.get(b) or (0.0, 0.0, 0, None, None)
        out.append({"ts": datetime.fromtimestamp(b, timezone.utc).isoformat(),
                    "long_usd": round(a[0], 2), "short_usd": round(a[1], 2), "events": int(a[2]),
                    "long_min_px": a[3], "short_max_px": a[4], "partial": b == now_bar})
    return {"warmed_up": True, "bars": out, "error": None, "source": "dashboard-forceorder"}


def merge_bybit_liq(bars, events, start_ms, bar_seconds):
    if start_ms is None:
        return bars
    first = start_ms // 1000 // bar_seconds * bar_seconds
    add = {}
    for e in events:
        b = e["ts_ms"] // 1000 // bar_seconds * bar_seconds
        a = add.setdefault(b, [0.0, 0.0, 0])
        a[0 if e["side"] == "long" else 1] += e["usd"]
        a[2] += 1
    out = []
    for b in bars:
        t = int(datetime.fromisoformat(b["ts"]).timestamp())
        if t > first:
            a = add.get(t, [0.0, 0.0, 0])
            b = {**b, "long_usd": b["long_usd"] + a[0], "short_usd": b["short_usd"] + a[1], "events": b["events"] + a[2],
                 "bybit": {"long_usd": round(a[0]), "short_usd": round(a[1]), "n": a[2]}}
        out.append(b)
    return out


def okx_liq_hot_bars(inst, db, bar_seconds=BAR):
    since = (int(time.time()) // bar_seconds - KEEP) * bar_seconds * 1000
    bar_ms = bar_seconds * 1000
    try:
        start = read_rows(db, "SELECT min(ts_ms) FROM okx_liquidations WHERE inst_id = ?", [inst])[0][0]
        rows = read_rows(db, """SELECT ts_ms - ts_ms % ?, pos_side, sum(sz_base * coalesce(bk_px, 0)), count(*)
                                      FROM okx_liquidations WHERE inst_id = ? AND ts_ms >= ? AND sz_base > 0
                                      GROUP BY 1, 2""", [bar_ms, inst, since])
    except Exception:  # noqa: BLE001 -- 원본 (duckdb.Error, sqlite3.Error, OSError, IndexError)
        return {}, None
    out = {}
    for bar, pos, usd, n in rows:
        a = out.setdefault(int(bar) // 1000, [0.0, 0.0, 0])
        a[0 if pos == "long" else 1] += float(usd or 0.0)
        a[2] += int(n)
    return out, int(start) if start is not None else None


def merge_okx_liq(bars, live_events, live_from, hot_bars, hot_start_ms, bar_seconds):
    add = {}
    for e in live_events if live_from else ():
        bar = int(e["ts_ms"]) // 1000 // bar_seconds * bar_seconds
        if bar > live_from:
            a = add.setdefault(bar, [0.0, 0.0, 0])
            a[0 if e["side"] == "long" else 1] += float(e.get("usd") or 0.0)
            a[2] += 1
    hot_first = None if hot_start_ms is None else hot_start_ms // 1000 // bar_seconds * bar_seconds
    out = []
    for b in bars:
        t = int(datetime.fromisoformat(b["ts"]).timestamp())
        a = add.get(t, [0.0, 0.0, 0]) if live_from and t > live_from else (
            hot_bars.get(t, [0.0, 0.0, 0]) if hot_first is not None and t > hot_first else None)
        if a is not None:
            b = {**b, "okx": True, "long_usd": b["long_usd"] + a[0],
                 "short_usd": b["short_usd"] + a[1], "events": b["events"] + a[2],
                 "okx_detail": {"long_usd": round(a[0]), "short_usd": round(a[1]), "n": a[2]}}
        out.append(b)
    return out


def merge_hl_liq(bars, events, bar_seconds):
    add = {}
    for t, usd, is_long, user in events:
        bar = t // 1000 // bar_seconds * bar_seconds
        a = add.setdefault(bar, [0.0, 0.0, 0, []])
        a[0 if is_long else 1] += usd
        a[2] += 1
        if user not in a[3]:
            a[3].append(user)
    out = []
    for b in bars:
        a = add.get(int(datetime.fromisoformat(b["ts"]).timestamp()))
        if a:
            b = {**b, "long_usd": b["long_usd"] + a[0], "short_usd": b["short_usd"] + a[1],
                 "events": b["events"] + a[2],
                 "hl": {"long_usd": round(a[0]), "short_usd": round(a[1]), "n": a[2], "users": a[3][:5]}}
        out.append(b)
    return out


def hl_whale_liq_events(since_ms, db, coin):
    try:
        rows = read_rows(db, """SELECT fill_ms, px * sz, dir, side, user FROM hl_liquidations
                                      WHERE coin = ? AND fill_ms >= ?""", [coin, since_ms])
    except Exception:  # noqa: BLE001
        return []
    return [(int(t), float(u or 0.0), ("Long" in str(d)) if d else side == "A", str(user)) for t, u, d, side, user in rows]


def make_liq_app(R: Path, mem: dict) -> web.Application:
    HL = {"eth": R / "data/hot/hl_positions.sqlite", "sol": R / "data/hot/hl_positions_sol_xrp.sqlite", "xrp": R / "data/hot/hl_positions_sol_xrp.sqlite",
          **{a: R / f"data/hot/hl_positions_{a}.sqlite" for a in ("btc", "sndk", "soxl", "skhynix", "mu")}}
    OKX_LIQ_DB = R / "data/hot/okx_ctx.sqlite"

    def footprint_window_bars(request):
        try:
            n = int(request.query.get("bars", FP_BARS))
        except (TypeError, ValueError):
            return FP_BARS
        return max(1, min(n, KEEP))

    async def api(request):
        asset = str(request.query.get("asset") or "eth").lower()
        if asset not in COIN_CONFIG:
            raise web.HTTPBadRequest(reason="unsupported_asset")
        bars = footprint_window_bars(request)
        m = mem[asset]
        payload = liq_5m_payload(m["store"], m["from_s"], bars, time.time())
        if asset not in FLOW or not payload.get("bars"):
            return web.json_response(payload, headers=g1.NOCACHE)
        okx_from, okx_liq_events = m["okx_first_bar"], m["okx_events"]
        inst = f"{asset.upper()}-USDT-SWAP"
        okx_hot, okx_hot_start = okx_liq_hot_bars(inst, OKX_LIQ_DB)

        def _done(bars, venues):
            bev, bstart = m["bybit_events"], m["bybit_start"]
            if bstart is not None:
                bars, venues = merge_bybit_liq(bars, bev, bstart, BAR), venues + ["bybit-perp"]
            if asset in HL:
                since = int((time.time() - KEEP * BAR) * 1000) if bars else 0
                ev = hl_whale_liq_events(since, HL[asset], asset.upper())
                merged = merge_hl_liq(bars, ev, BAR)
                if any("hl" in b for b in merged):
                    venues = venues + ["hyperliquid-whales"]
                return {**payload, "bars": merged, "venues": venues}
            return {**payload, "bars": bars, "venues": venues}
        if not okx_from and okx_hot_start is None:
            return web.json_response(_done(payload["bars"], ["binance-perp"]), headers=g1.NOCACHE)
        bars = merge_okx_liq(payload["bars"], list(okx_liq_events), okx_from, okx_hot, okx_hot_start, BAR)
        return web.json_response(_done(bars, ["binance-perp", "okx-swap"]), headers=g1.NOCACHE)

    app = web.Application(middlewares=[g1.static_asset_headers, g1.json_compress_etag])
    app.router.add_get("/api/liquidation-5m-history", api)
    app.router.add_static("/dashboard/live/", R / "dashboard" / "live", show_index=False)
    return app


def core_liq5m(m: dict) -> dict:
    """원본 메모리 → 코어가 미는 Liq5m(rtcore supply.rs liq5m_all · venue_liq5m 과 같은 규칙: 기록 시작 봉 뒤만 · r2 · OKX first_bar 뒤 · Bybit 시작 봉 뒤)"""
    from_bar = int(m["from_s"]) // BAR * BAR
    bars = [[b, round(a[0], 2), round(a[1], 2), a[2], a[3], a[4]] for b, a in sorted(m["store"].items()) if b > from_bar]
    okx = {}
    if m["okx_first_bar"]:
        for e in m["okx_events"]:
            b = e["ts_ms"] // 1000 // BAR * BAR
            if b > m["okx_first_bar"]:
                a = okx.setdefault(b, [0.0, 0.0, 0])
                a[0 if e["side"] == "long" else 1] += e["usd"]
                a[2] += 1
    bf = None if m["bybit_start"] is None else m["bybit_start"] // 1000 // BAR * BAR
    by = {}
    for e in m["bybit_events"] if bf is not None else ():
        b = e["ts_ms"] // 1000 // BAR * BAR
        a = by.setdefault(b, [0.0, 0.0, 0])
        a[0 if e["side"] == "long" else 1] += e["usd"]
        a[2] += 1
    return {"from_s": m["from_s"], "bars": bars, "okx": {"from": m["okx_first_bar"] or None, "bars": [[b, *a] for b, a in sorted(okx.items())]},
            "bybit": {"from": bf, "bars": [[b, *a] for b, a in sorted(by.items()) if b > bf]} if bf is not None else {"from": None, "bars": []}}


def liq_tree() -> tuple[dict[str, bytes], dict]:
    rnd = random.Random(11)
    now = int(NOW)
    nb = now // BAR * BAR

    def evs(t0, t1, k, px):
        out = []
        for _ in range(k):
            t = rnd.randint(t0 * 1000, t1 * 1000)
            q = rnd.choice([0.01, 0.5, 1.234, 7.0, 33.3]) * rnd.random()
            p = px * (1 + rnd.gauss(0, 0.004))
            out.append({"ts_ms": t, "side": rnd.choice(["long", "short"]), "qty": q, "price": p, "usd": q * p})
        return sorted(out, key=lambda e: e["ts_ms"])
    mem = {}
    okx_rows, hl_eth, hl_sx = [], [], []
    for a, px, k in (("eth", 3123.45, 900), ("sol", 141.7, 400), ("xrp", 2.3456, 150)):
        store = {}
        from_s = now - 15 * 3600 + 137.25 if a != "xrp" else now - 3600 * 30
        for e in evs(now - 30 * 3600, now, k, px):
            if e["ts_ms"] / 1000 >= from_s:
                liq_5m_add(store, e)
        if a == "eth":   # 가격 0(= 극값 안 건드림) 한 건
            liq_5m_add(store, {"ts_ms": (nb - 3 * BAR + 5) * 1000, "side": "long", "usd": 12.5, "price": 0})
        okx_ev = [{"ts_ms": e["ts_ms"], "side": e["side"], "usd": e["usd"]} for e in evs(now - 5 * 3600, now, 200 if a == "eth" else 0, px)]
        bys = {"eth": now - 6 * 3600 + 123, "xrp": now - 2 * 3600 - 7}.get(a)
        by_ev = [{"ts_ms": e["ts_ms"], "side": e["side"], "usd": e["usd"]} for e in evs(now - 7 * 3600, now, 150, px)] if bys else []
        mem[a] = {"store": store, "from_s": from_s, "okx_first_bar": nb - 40 * BAR if a == "eth" else 0, "okx_events": okx_ev,
                  "bybit_events": by_ev, "bybit_start": bys * 1000 if bys else None}
        if a in ("eth", "sol"):
            inst = f"{a.upper()}-USDT-SWAP"
            t0 = now - (10 if a == "eth" else 3) * 3600 - 61
            okx_rows.append((inst, t0 * 1000 + 17, "long", 0.0, px))   # 시작 행(sz 0 = 합에서 빠짐 · min 은 잡힘)
            for _ in range(300 if a == "eth" else 80):
                okx_rows.append((inst, rnd.randint(t0 * 1000, now * 1000), rnd.choice(["long", "short", "net"]), rnd.random() * 30,
                                 None if rnd.random() < 0.05 else px * (1 + rnd.gauss(0, 0.003))))
        for _ in range(120 if a == "eth" else 60):
            t = rnd.randint((now - 26 * 3600) * 1000, now * 1000)
            row = (a.upper(), t, px * (1 + rnd.gauss(0, 0.003)), rnd.random() * 50, rnd.choice(["Close Long", "Close Short", None, "", "Long > Short"]),
                   rnd.choice(["A", "B"]), rnd.choice([f"0xwhale{i:02d}" for i in range(9)] + [None]))
            (hl_eth if a == "eth" else hl_sx).append(row)
    for i in range(7):   # 한 봉 주소 7개 → users 앞 5개
        hl_eth.append(("ETH", (nb - 2 * BAR + 10 + i) * 1000, 3100.0, 1.0 + i, "Close Long", "A", f"0xburst{i}"))
    okx_db = g1.sqlite_bytes([("CREATE TABLE okx_liquidations (inst_id TEXT, ts_ms INTEGER, pos_side TEXT, sz_base REAL, bk_px REAL)", []),
                              ("INSERT INTO okx_liquidations VALUES (?,?,?,?,?)", okx_rows)])
    hl_ddl = "CREATE TABLE hl_liquidations (coin TEXT, fill_ms INTEGER, px REAL, sz REAL, dir TEXT, side TEXT, user TEXT)"
    files = {"data/hot/okx_ctx.sqlite": okx_db,
             "data/hot/hl_positions.sqlite": g1.sqlite_bytes([(hl_ddl, []), ("INSERT INTO hl_liquidations VALUES (?,?,?,?,?,?,?)", hl_eth)]),
             "data/hot/hl_positions_sol_xrp.sqlite": g1.sqlite_bytes([(hl_ddl, []), ("INSERT INTO hl_liquidations VALUES (?,?,?,?,?,?,?)", hl_sx)])}
    return files, mem


LIQ_REQ = [("GET", "/api/liquidation-5m-history" + q, {}) for q in (
    "", "?bars=144", "?bars=288", "?bars=999", "?bars=abc", "?bars=0", "?bars=-3", "?asset=SOL&bars=200", "?asset=xrp&bars=100",
    "?asset=xrp&bars=3", "?asset=doge")] + [("GET", "/api/liquidation-5m-history?bars=48", {"Accept-Encoding": "gzip"})]

REQ = [("GET", p, {}) for p in ("/api/oi-5m", "/api/oi-5m?bars=12", "/api/oi-5m?bars=abc", "/api/oi-5m?bars=999", "/api/oi-5m?asset=SOL&bars=24", "/api/oi-5m?asset=sol",
                                "/api/oi-5m?asset=xrp", "/api/oi-5m?asset=btc")]


async def run_scenario(files, app_fn, reqs):
    with tempfile.TemporaryDirectory() as td:
        R = Path(td)
        (R / "dashboard" / "live").mkdir(parents=True)
        for rel, data in files.items():
            p = R / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            os.utime(p, ns=(MTIME_NS, MTIME_NS))
        app = app_fn(R)
        app.freeze()
        return [{"method": m, "path": p, "req_headers": h, **await g1.run_one(app, m, p, h)} for m, p, h in reqs]


async def main():
    lfiles, lmem = liq_tree()
    lcases = await run_scenario(lfiles, lambda R: make_liq_app(R, lmem), LIQ_REQ)
    files, mem = tree()
    with tempfile.TemporaryDirectory() as td:
        R = Path(td)
        (R / "dashboard" / "live").mkdir(parents=True)
        for rel, data in files.items():
            p = R / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            os.utime(p, ns=(MTIME_NS, MTIME_NS))
        app = make_app(R, mem)
        app.freeze()
        cases = []
        for method, path, headers in REQ:
            out = await g1.run_one(app, method, path, headers)
            cases.append({"method": method, "path": path, "req_headers": headers, **out})
    okx_from = {"ethusdt": mem["eth"]["okx_fp"]["first_bar"]}
    out = {"now": NOW, "mtime_ns": MTIME_NS, "dashboard_assets": ",".join(g1.DASHBOARD_ASSETS),
           "micro": {s: {"okx_oi5m": v, "okx_from": okx_from.get(s)} for s, v in mem["_okx_d"].items() if v},
           "liq5m": {f"{a}usdt": core_liq5m(m) for a, m in lmem.items()},
           "scenarios": [{"name": "oi5m", "files": {k: base64.b64encode(v).decode() for k, v in files.items()}, "cases": cases},
                         {"name": "liqhist", "files": {k: base64.b64encode(v).decode() for k, v in lfiles.items()}, "cases": lcases}]}
    OUT.write_text(json.dumps(out, ensure_ascii=False) + "\n")
    for c in cases + lcases:
        print(f"  {c['path']:36} {c['status']} {base64.b64decode(c['body_b64'])[:160]}")


if __name__ == "__main__":
    asyncio.run(main())
