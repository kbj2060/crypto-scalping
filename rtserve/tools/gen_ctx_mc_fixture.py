#!/usr/bin/env python3
"""rtserve 시장 맥락(src/ctx_mc.rs) 일치 시험의 입력·정답지 (2026-11-22).

정답 = 원본. import 해서 그대로 부르는 것: dashboard/market_ctx.py(pct_rank·quad_1h·oi_stats·oi_series_live·oi_sum_series·ring_at·lev_state·
hl_liq_levels·liq_profile·value_area·profile_nodes) · dashboard/micro_ref.py deriv_from_ring · dashboard/situation.py BASIS_PCT.
server.py 의 것은 **옮겨 적었다**(Read 로 본 줄 그대로 · 들여쓰기만):
  · market_context_payload 2632~2708(swr_cached 자리는 미리 센 값을 인자로 · time.time() 은 고정 시계)
  · _mc_collectors 2534~2556 · _mc_history 2558~2583 · _mc_liq_profile 2585~2595 · tape_levels 931~969 · own_db_rows 1022~1030
  · LIQ_BURST_60S_USD 369 · MC_LIQ_PROFILE_S 2525 · api_market_context 의 묵은 값 갈래 2727~2731
합성 파일: 수집기 hot SQLite 넷(OKX·Bybit·HL 맥락·HL 포지션) · 바이낸스 맥락(oi_1s·mark_price_1s·liquidations) · 체결 테이프 · 롱숏비 duckdb.
🔴롱숏비 SQL 은 duckdb now() 를 쓴다(원본 그대로) -- 행을 «만든 때 + 30일»까지 두어 시험을 언제 돌려도 같은 행이 25시간 창에 든다.
    source <(bash rtmodel/tools/server_env.sh <venv> --env) && $PY rtserve/tools/gen_ctx_mc_fixture.py
출력: rtserve/tests/fixtures/gen/ctx_mc/ (커밋 안 함)
"""
from __future__ import annotations

import json
import math
import random
import shutil
import sqlite3
import sys
import time as _time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "ctx_mc"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
sys.path.insert(0, str(ROOT))
import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import dashboard.market_ctx as mctx  # noqa: E402
import dashboard.micro_ref as mref  # noqa: E402
import dashboard.situation as sit  # noqa: E402

R = random.Random(20261126)
NOW = 1_763_740_123.25
LIQ_BURST_60S_USD = (round(math.expm1(12.732)), round(math.expm1(13.21)))   # server.py 369
MC_LIQ_PROFILE_S = 12 * 3600   # 2525


class FakeTime:
    @staticmethod
    def time():
        return NOW


time = FakeTime()


def _read_only_rows(db: Path, sql: str, params: list) -> list:   # 대역: 읽기 전용(sqlite · duckdb)
    if str(db).endswith(".duckdb"):
        con = duckdb.connect(str(db), read_only=True)
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


read_rows = _read_only_rows


def own_db_rows(path: Path, sql: str, params: list) -> list[tuple]:   # 1022~1030 옮겨 적음
    if not path.exists():
        return []
    try:
        return read_rows(path, sql, params)
    except Exception as exc:  # noqa: BLE001
        print(f"own db read failed ({path.name}): {exc}", flush=True)
        return []


def tape_levels(tape_db: Path, sym: str, bk: float, now: int) -> dict[str, Any]:   # 931~969 옮겨 적음
    d0 = now // 86400 * 86400
    w0 = d0 - ((d0 // 86400 + 3) % 7) * 86400                 # 1970-01-01 = 목요일 → 이번 주 월요일 00:00 UTC
    vw = _read_only_rows(tape_db, """
        WITH t AS (SELECT ts_sec, price_bin, buy_qty + sell_qty AS q FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ?),
             pd AS (SELECT ts_sec, price_bin FROM t WHERE ts_sec >= ? AND ts_sec < ?),
             ah AS (SELECT ts_sec FROM pd ORDER BY price_bin DESC, ts_sec LIMIT 1),
             al AS (SELECT ts_sec FROM pd ORDER BY price_bin, ts_sec LIMIT 1)
        SELECT (SELECT sum(price_bin * q) / nullif(sum(q), 0) FROM t WHERE ts_sec >= ?),
               (SELECT sum(price_bin * q) / nullif(sum(q), 0) FROM t WHERE ts_sec >= (SELECT ts_sec FROM ah)),
               (SELECT sum(price_bin * q) / nullif(sum(q), 0) FROM t WHERE ts_sec >= (SELECT ts_sec FROM al)),
               (SELECT ts_sec FROM ah), (SELECT ts_sec FROM al),
               (SELECT price_bin FROM t ORDER BY ts_sec DESC, price_bin DESC LIMIT 1),
               (SELECT price_bin FROM pd ORDER BY ts_sec DESC, price_bin DESC LIMIT 1)""",
        [sym, min(w0, d0 - 86400), d0 - 86400, d0, w0])[0]
    px = lambda b: None if b is None else (float(b) + 0.5) * bk   # noqa: E731 -- 테이프 칸 가운데
    last = px(vw[5])
    if not last:
        return {"available": False}
    bw = max(bk, round((px(vw[6]) or last) * mctx.PROFILE_BIN_FRAC / bk) * bk)
    rows = _read_only_rows(tape_db, """
        SELECT 0, CAST(floor((price_bin + 0.5) * ? / ?) AS BIGINT), sum(buy_qty + sell_qty) FROM trade_tape_1s
        WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY 2
        UNION ALL
        SELECT 1, CAST(floor((price_bin + 0.5) * ? / ?) AS BIGINT), sum(buy_qty + sell_qty) FROM trade_tape_1s
        WHERE symbol = ? AND ts_sec >= ? GROUP BY 2""", [bk, bw, sym, d0 - 86400, d0, bk, bw, sym, now - 86400])
    prev = [(int(k), float(v)) for g, k, v in rows if g == 0]
    h24 = [(int(k), float(v)) for g, k, v in rows if g == 1]
    return {"available": True, "last": last, "bin": bw,
            "prev_day": mctx.value_area([k for k, _ in prev], [v for _, v in prev], bw),
            **mctx.profile_nodes([k for k, _ in h24], [v for _, v in h24], bw),
            "vwap_week": px(vw[0]), "week_start": w0,
            "avwap_hi": px(vw[1]), "avwap_hi_ts": vw[3], "avwap_lo": px(vw[2]), "avwap_lo_ts": vw[4]}


D = OUT
P = {"okx": D / "okx_ctx.sqlite", "bybit": D / "bybit_ctx.sqlite", "hl": D / "hl_ctx.sqlite", "hl_pos": D / "hl_pos.sqlite",
     "ls": D / "oi_lsratio.duckdb", "bn": D / "binance_ctx.sqlite", "tape": D / "tape.sqlite"}
OKX_CTX_DB_PATH, BYBIT_CTX_DB, HL_CTX_DB_PATH, OI_LSRATIO_DB_PATH = P["okx"], P["bybit"], P["hl"], P["ls"]
OI_1S_DB_PATH, MARK_PRICE_DB_PATH, BINANCE_CTX_DB = P["bn"], P["bn"], P["bn"]
OI_1S_TABLE, MARK_PRICE_TABLE = "oi_1s", "mark_price_1s"
HL_LIQ_BY_ASSET = {"eth": P["hl_pos"], "xrp": P["hl_pos"]}
SPEC = {"eth": {"okx_inst": "ETH-USDT-SWAP", "symbol": "ETHUSDT", "bucket": 0.5}, "xrp": {"okx_inst": "XRP-USDT-SWAP", "symbol": "XRPUSDT", "bucket": 0.0003}}
mc_last_good_by: dict = {}   # 원본 cctx[asset].mc_last_good(코인마다)


def _px_scale(asset):   # 2532 옮겨 적음(flows[asset].spec.bucket / flows["eth"].spec.bucket)
    return SPEC[asset]["bucket"] / SPEC["eth"]["bucket"]


def _mc_collectors(asset: str = "eth") -> dict[str, Any]:   # 2534~2556 옮겨 적음
    mc_last_good, OKX_INST, coin = mc_last_good_by.setdefault(asset, {}), SPEC[asset]["okx_inst"], asset.upper()   # noqa: N806
    ls_table = "oi_lsratio_5m" if asset == "eth" else f"oi_lsratio_5m_{asset}"
    out: dict[str, Any] = {"errors": {}}
    def q(key: str, path: Path, sql: str, params: list) -> None:
        try:
            out[key] = _read_only_rows(path, sql, params) if path.exists() else None
            mc_last_good[key] = out[key]
        except Exception as exc:  # noqa: BLE001
            out[key] = mc_last_good.get(key)
            out["errors"][key] = repr(exc)[:100]
    q("okx_fund", OKX_CTX_DB_PATH, "SELECT funding_rate, funding_time FROM okx_funding WHERE inst = ? ORDER BY ts_ms DESC LIMIT 1", [OKX_INST])
    q("okx_mark", OKX_CTX_DB_PATH, "SELECT mark_px, ts_ms FROM okx_mark WHERE inst = ? ORDER BY ts_ms DESC LIMIT 1", [OKX_INST])
    q("okx_oi", OKX_CTX_DB_PATH, "SELECT oi_base FROM okx_oi WHERE inst = ? ORDER BY ts_ms DESC LIMIT 1", [OKX_INST])
    q("by_oi", BYBIT_CTX_DB, "SELECT oi_base FROM bybit_oi WHERE inst = ? ORDER BY ts_ms DESC LIMIT 1", [SPEC[asset]["symbol"]])
    q("hl_ctx", HL_CTX_DB_PATH, "SELECT funding, open_interest, premium, mark_px, mid_px, recv_ms FROM hl_asset_ctx WHERE coin = ? ORDER BY recv_ms DESC LIMIT 1", [coin])
    q("hl_pos", HL_LIQ_BY_ASSET[asset], "SELECT szi, liq_px FROM hl_positions WHERE coin = ? AND ts_ms >= (SELECT max(ts_ms) FROM hl_cycles)", [coin])
    q("ls", OI_LSRATIO_DB_PATH, f"""SELECT epoch(ts), global_ls_ratio, top_pos_ls_ratio, taker_ls_ratio FROM {ls_table}
                                   WHERE ts >= now() - INTERVAL 25 HOUR ORDER BY ts""", [])
    return out


def _mc_history(asset: str = "eth") -> dict[str, Any]:   # 2558~2583 옮겨 적음
    since, sym = int(time.time() * 1000) - 7 * 86400 * 1000, SPEC[asset]["symbol"].lower()
    oi = [(int(b), float(v)) for b, v, _ in own_db_rows(
        OI_1S_DB_PATH, f"""SELECT ts_ms / 300000 * 300 AS b, open_interest, max(ts_ms) FROM {OI_1S_TABLE}
                           WHERE symbol = ? AND ts_ms >= ? GROUP BY b ORDER BY b""", [sym, since])]
    per = {"binance": oi}
    for k, path, tbl, ts, val, kc, key in (("okx", OKX_CTX_DB_PATH, "okx_oi", "ts_ms", "oi_base", "inst", SPEC[asset]["okx_inst"]),
                                           ("hl", HL_CTX_DB_PATH, "hl_asset_ctx", "recv_ms", "open_interest", "coin", asset.upper()),
                                           ("bybit", BYBIT_CTX_DB, "bybit_oi", "ts_ms", "oi_base", "inst", SPEC[asset]["symbol"])):
        try:
            rows = _read_only_rows(path, f"""SELECT {ts} / 300000 * 300 AS b, {val}, max({ts}) FROM {tbl}
                                            WHERE {kc} = ? AND {ts} >= ? AND {val} IS NOT NULL GROUP BY b ORDER BY b""",
                                   [key, since]) if path.exists() else []
        except (sqlite3.Error, duckdb.Error, OSError) as exc:
            print(f"mc_history oi {k}: {exc!r}", flush=True)
            rows = []
        per[k] = [(int(b), float(v)) for b, v, _ in rows]
    basis = own_db_rows(MARK_PRICE_DB_PATH, f"""SELECT (mark - index_px) / index_px * 1e4 FROM {MARK_PRICE_TABLE}
                                              WHERE symbol = ? AND ts_ms >= ? AND index_px > 0 AND ts_ms % 60000 < 1000""", [sym, since])
    return {"oi": [(int(t), float(v)) for t, v in oi], "oi_per": per, "basis": [float(b[0]) for b in basis]}


def _mc_liq_profile(asset: str = "eth") -> list[list[float]]:   # 2585~2595 옮겨 적음
    cut = int((time.time() - MC_LIQ_PROFILE_S) * 1000)
    try:
        rows = read_rows(BINANCE_CTX_DB, "SELECT price, usd, side FROM liquidations WHERE symbol = ? AND ts_ms >= ?",
                         [SPEC[asset]["symbol"], cut])
    except Exception:  # noqa: BLE001
        return []
    return mctx.liq_profile([(float(p or 0), float(u or 0), side == "long") for p, u, side in rows],
                            mctx.LIQ_PROFILE_BIN_USD * _px_scale(asset))


def payload(asset, col, hist, prof, fh, dvy, vprof, micro_payload, x, ev, mark_ring, mid_ring, depth25_ring, oi_1s, mp_state, sym):   # 2632~2708 옮겨 적음
    one = lambda k: (col.get(k) or [None])[0]   # noqa: E731 -- 없으면 None 한 줄
    mp = micro_payload if micro_payload.get("available") else {}
    dv = mref.deriv_from_ring(mark_ring, 1800, sit.BASIS_PCT)
    bn_mark = mark_ring[max(mark_ring)][0] if mark_ring else None
    mid = mp.get("mid") or bn_mark
    oi = mctx.oi_stats(mctx.oi_series_live(hist["oi"], max(oi_1s), oi_1s[max(oi_1s)]) if oi_1s else hist["oi"])
    okx_oi, hl, by_oi = one("okx_oi"), one("hl_ctx"), (one("by_oi") or [None])[0]
    now_s = int(time.time())
    oi_ser, oi_ven = mctx.oi_sum_series(hist.get("oi_per") or {}, since=(now_s - 7 * 86400) // 300 * 300)
    latest = {"binance": oi_1s[max(oi_1s)] if oi_1s else None, "okx": okx_oi[0] if okx_oi else None,
              "hl": hl[1] if hl else None, "bybit": by_oi}
    oi_all = None
    if oi_ser and all(latest.get(k) for k in oi_ven):
        oi_all = {**mctx.oi_stats(mctx.oi_series_live(oi_ser, now_s, sum(float(latest[k]) for k in oi_ven))), "venues": oi_ven,
                  "by_venue": {k: (oi["d1h_pct"] if k == "binance" else mctx.oi_stats((hist.get("oi_per") or {}).get(k) or [])["d1h_pct"])
                               for k in oi_ven}}
        oi_all["d1h"] = oi_all["oi"] * (1 - 1 / (1 + oi_all["d1h_pct"] / 100)) if oi_all.get("d1h_pct") is not None else None
    okx_mark, okx_fund = one("okx_mark"), one("okx_fund")
    bp = lambda a, b: (a / b - 1) * 1e4 if a and b else None   # noqa: E731
    ls = col.get("ls") or []
    lsv = lambda j, rows: next((r[j] for r in rows if r[j] is not None), None)   # noqa: E731
    ls24 = lambda j: (lambda d, v: v if d <= 7200 else None)(*min(((abs(r[0] - (time.time() - 86400)), r[j]) for r in ls if r[j] is not None), default=(1e9, None)))   # noqa: E731
    d25 = np.array(depth25_ring, dtype=float) if len(depth25_ring) >= 600 else None
    sw = mp.get("sweep")
    basis_pct = mctx.pct_rank(dv["basis_bp"], hist["basis"], min_n=600)
    liq60 = mp.get("liq60") or {}
    return {
        "available": True, "asset": asset, "ts": time.time(), "mid": mid,
        "bands_usd": list((10_000.0, 100_000.0)),   # 원본 list(size_bands(sym)) 자리 -- 시험은 같은 값을 인자로
        "funding": {"bn": dv["funding"], "bn_pct180": mctx.pct_rank(dv["funding"], fh), "next_ms": mp_state.get("next_funding_ms"),
                    "bn_at_base": dv["funding"] is not None and abs(dv["funding"] - 0.0001) < 1e-9,
                    "okx": okx_fund[0] if okx_fund else None, "okx_next_ms": okx_fund[1] if okx_fund else None,
                    "hl_8h": hl[0] * 8 if hl and hl[0] is not None else None},
        "basis": {"bp": dv["basis_bp"], "d30_bp": dv["basis_d_bp"], "pct7d": basis_pct,
                  "hl_premium_bp": hl[2] * 1e4 if hl and hl[2] is not None else None},
        "lev": mctx.lev_state(basis_pct, oi_all["z1h"] if oi_all and oi_all.get("z1h") is not None else oi["z1h"]),
        "oi": {**oi, "okx": okx_oi[0] if okx_oi else None, "hl": hl[1] if hl else None, "bybit": by_oi, "all": oi_all},
        "ls": ({"global": lsv(1, ls[::-1]), "top_pos": lsv(2, ls[::-1]), "taker": lsv(3, ls[::-1]),
                "global_24h": ls24(1), "top_pos_24h": ls24(2), "age_s": time.time() - ls[-1][0]} if ls else None),
        "quad": mctx.quad_1h(x.get("move60"), oi_all["d1h"] if oi_all and oi_all.get("d1h") is not None else x.get("oi60"), x.get("move60_p75")),
        "move60": x.get("move60"), "oi60": x.get("oi60"),
        "flow": {"z60": x.get("z60"), "z60h": x.get("z60h"), "net60": x.get("net60"), "cvd30_z": x.get("cvd30_z"),
                 "venues": x.get("venues"), "z60h_all": x.get("z60h_all"), "net60_all": x.get("net60_all"),
                 "z30h": x.get("z30h"), "z30h_all": x.get("z30h_all"), "net30_all": x.get("net30_all"),
                 "cvd30_all_z": x.get("cvd30_all_z"), "taker_all": x.get("taker_all"),
                 "bn30": ev.get("cvd"), "okx30": x.get("okx30")},
        "btc": {"move_bp": ev.get("btc_move_bp"), "rel": ev.get("btc_rel")},
        "venues": {"okx_bp": bp(okx_mark[0] if okx_mark else None,
                                mctx.ring_at(mark_ring, int(okx_mark[1]) // 1000, (bn_mark,))[0] if okx_mark else None),
                   "hl_bp": bp(hl[4] if hl else None, mctx.ring_at(mid_ring, int(hl[5]) // 1000, mp.get("mid")) if hl else None)},
        "book": {"spread": mp.get("spread"), "sweep": sw, "bps": list(mctx.SWEEP_BPS),
                 "bid25_pct": float(np.mean(d25[:, 0] <= sw["bid"][0])) if d25 is not None and sw else None,
                 "ask25_pct": float(np.mean(d25[:, 1] <= sw["ask"][0])) if d25 is not None and sw else None},
        "hl_liq": mctx.hl_liq_levels(col.get("hl_pos") or [], (hl[3] if hl and hl[3] else mid), mctx.HL_LIQ_BIN_USD * _px_scale(asset))
                  if (mid or (hl and hl[3])) else {"below": [], "above": []},
        "liq_profile": prof,
        "profile": vprof,
        "iv_rank": ({"dvol": dvy[-1], "rank365": mctx.pct_rank(dvy[-1], dvy[:-1], min_n=200), "n": len(dvy) - 1} if dvy else None),
        "burst": ({"long_usd_60s": liq60.get("long"), "short_usd_60s": liq60.get("short"), "thr": LIQ_BURST_60S_USD, "ts": mp.get("ts")}
                  if asset == "eth" and liq60 else None),
        "errors": col.get("errors") or {},
    }


def stale(result, last, now):   # 2722~2731 옮겨 적음(성공·실패 갈래)
    if not isinstance(result, Exception):
        last.update(payload=result, at=now)
        return result
    err = f"{type(result).__name__}: {result}"[:200]
    age = now - last.get("at", 0.0)
    return ({**last["payload"], "stale": True, "stale_age_s": age, "error": err} if age < 120.0 else {"available": False, "error": err})


# ── 합성 파일 ──
now_ms = int(NOW * 1000)
since7 = now_ms - 7 * 86400 * 1000


def mk(path, ddl, rows_by_table):
    con = sqlite3.connect(path)
    for d in ddl:
        con.execute(d)
    for t, rows in rows_by_table.items():
        if rows:
            con.executemany(f"INSERT INTO {t} VALUES ({','.join('?' * len(rows[0]))})", rows)
    con.commit(); con.close()


oi_rows_okx = [(now_ms - k * 61_000, inst, 1000.0 + R.gauss(0, 30)) for inst in ("ETH-USDT-SWAP", "XRP-USDT-SWAP") for k in range(0, 7 * 1440, 1)]
mk(P["okx"], ["CREATE TABLE okx_funding (ts_ms INTEGER, inst TEXT, funding_rate REAL, funding_time INTEGER)",
              "CREATE TABLE okx_mark (ts_ms INTEGER, inst TEXT, mark_px REAL)", "CREATE TABLE okx_oi (ts_ms INTEGER, inst TEXT, oi_base REAL)"],
   {"okx_funding": [(now_ms - 5000, "ETH-USDT-SWAP", 0.000123, now_ms + 3_600_000)],
    "okx_mark": [(now_ms - 2400, "ETH-USDT-SWAP", 3001.25), (now_ms - 9000, "ETH-USDT-SWAP", 2999.0)], "okx_oi": [(t, i, v) for t, i, v in oi_rows_okx]})
mk(P["bybit"], ["CREATE TABLE bybit_oi (ts_ms INTEGER, inst TEXT, oi_base REAL)"],
   {"bybit_oi": [(now_ms - k * 300_000 * R.choice([1, 1, 3]), "ETHUSDT", 500.0 + R.gauss(0, 10)) for k in range(7 * 288 - 30)]})
mk(P["hl"], ["CREATE TABLE hl_asset_ctx (recv_ms INTEGER, coin TEXT, funding REAL, open_interest REAL, premium REAL, mark_px REAL, mid_px REAL)"],
   {"hl_asset_ctx": [(now_ms - k * 30_000, "ETH", 1.25e-5, 800.0 + R.gauss(0, 20), 0.0002, 3000.5, 3000.75) for k in range(0, 7 * 2880, 3)]})
mk(P["hl_pos"], ["CREATE TABLE hl_positions (ts_ms INTEGER, coin TEXT, szi REAL, liq_px REAL)", "CREATE TABLE hl_cycles (ts_ms INTEGER)"],
   {"hl_cycles": [(now_ms - 60_000,), (now_ms - 20_000,)],
    "hl_positions": [(now_ms - 10_000, "ETH", R.choice([-1, 1]) * R.uniform(0.1, 300), R.choice([None, 0.0, R.uniform(2700, 3300), 2900.0, 3100.0])) for _ in range(300)]
                    + [(now_ms - 50_000, "ETH", 5.0, 2950.0)]})
oi1 = [(now_ms - k * 1000 * R.choice([1, 7]), "ethusdt", 2_000_000.0 + R.gauss(0, 5000)) for k in range(0, 7 * 86400, 61)]
mk(P["bn"], ["CREATE TABLE oi_1s (ts_ms INTEGER, symbol TEXT, open_interest REAL)", "CREATE TABLE mark_price_1s (ts_ms INTEGER, symbol TEXT, mark REAL, index_px REAL)",
             "CREATE TABLE liquidations (ts_ms INTEGER, symbol TEXT, price REAL, usd REAL, side TEXT)"],
   {"oi_1s": oi1, "mark_price_1s": [(now_ms - k * 60_000 + R.choice([0, 500, 1500]), "ethusdt", 3000 + R.gauss(0, 1), 3000.0) for k in range(7 * 1440)],
    "liquidations": [(now_ms - R.randrange(0, 13 * 3600_000), "ETHUSDT", R.choice([0.0, R.uniform(2900, 3100)]), R.choice([0.0, R.uniform(100, 1e5)]), R.choice(["long", "short"])) for _ in range(400)]})
d0 = int(NOW) // 86400 * 86400
mk(P["tape"], ["CREATE TABLE trade_tape_1s (ts_sec INTEGER, symbol TEXT, price_bin INTEGER, buy_qty REAL, sell_qty REAL)"],
   {"trade_tape_1s": [(t, "ethusdt", 30000 + int(R.gauss(0, 120)), R.uniform(0, 5), R.uniform(0, 5)) for t in range(d0 - 9 * 86400, int(NOW), 37)]})
con = duckdb.connect(str(P["ls"]))
con.execute("CREATE TABLE oi_lsratio_5m (ts TIMESTAMPTZ, global_ls_ratio DOUBLE, top_pos_ls_ratio DOUBLE, taker_ls_ratio DOUBLE)")
base = _time.time() + 30 * 86400   # 원본 SQL 의 now() 창에 늘 든다(머리 주석)
con.executemany("INSERT INTO oi_lsratio_5m VALUES (to_timestamp(?), ?, ?, ?)",
                [(base - k * 300, R.choice([None, R.uniform(0.5, 3)]), R.uniform(0.5, 3), R.choice([None, R.uniform(0.8, 1.2)])) for k in range(300)])
con.close()

col = _mc_collectors("eth")
col_x = _mc_collectors("xrp")
hist = _mc_history("eth")
prof = _mc_liq_profile("eth")
vprof = tape_levels(P["tape"], "ethusdt", 0.1, int(time.time()))
cases = []
for i in range(80):
    mark_ring = {int(NOW) - s: (3000 + R.gauss(0, 2), 3000 + R.gauss(0, 0.1), R.choice([0.0001, 0.00012])) for s in range(0, R.choice([0, 2000, 4000]), R.choice([1, 2]))}
    mid_ring = {int(NOW) - s: 3000 + R.gauss(0, 1) for s in range(0, 600, R.choice([1, 3]))}
    sweep = R.choice([None, {"bid": [R.uniform(0, 500), 1.0, 2.0], "ask": [R.uniform(0, 500), 1.0, 2.0]}])
    mp = {"available": R.random() < 0.9, "ts": int(NOW), "mid": R.choice([None, 3000.5]), "spread": 0.01, "sweep": sweep,
          "liq60": R.choice([{}, {"long": 1.5e5, "short": 0.0, "n": 2}])}
    x = {"move60": R.choice([None, R.gauss(0, 30)]), "oi60": R.choice([None, R.gauss(0, 900)]), "move60_p75": R.choice([None, 15.0]),
         "z60": {"whale": 1.0, "mid": 0.0, "retail": -1.0}, "venues": R.choice([None, ["binance", "okx"]]), "okx30": R.choice([None, 12.5])}
    ev = {"cvd": R.choice([None, 120]), "btc_move_bp": R.choice([None, 3.5]), "btc_rel": R.choice([None, "동행"])}
    d25 = [(R.uniform(0, 500), R.uniform(0, 500)) for _ in range(R.choice([0, 599, 700]))]
    oi_1s = {int(NOW) - s: 2_000_000.0 + R.gauss(0, 100) for s in range(0, R.choice([0, 30]))}
    fh = [R.gauss(1e-4, 5e-5) for _ in range(R.choice([0, 10, 540]))]
    dvy = [R.uniform(40, 90) for _ in range(R.choice([0, 150, 366]))]
    asset = R.choice(["eth", "eth", "xrp"])
    c = col if asset == "eth" else col_x
    p = payload(asset, c, hist, prof, fh, dvy, vprof, mp, x, ev, mark_ring, mid_ring, d25, oi_1s, {"next_funding_ms": R.choice([None, now_ms + 1000])}, "ETHUSDT")
    cases.append({"asset": asset, "micro": mp, "x": x, "ev": ev, "mark_ring": [[k, list(v)] for k, v in mark_ring.items()], "mid_ring": [[k, v] for k, v in mid_ring.items()],
                  "depth25": [list(d) for d in d25], "oi_1s": [[k, v] for k, v in oi_1s.items()], "fh": fh, "dvy": dvy,
                  "next_ms": p["funding"]["next_ms"], "payload": json.dumps(p)})
# 함수 직접(여러 모양) -- oi_stats·oi_sum_series·hl_liq·liq_profile·value_area·profile_nodes·pct_rank
fn = {"oi_stats": [], "sum": [], "hl": [], "lp": [], "va": [], "pn": []}
for _ in range(60):
    t0 = 1_763_000_000 // 300 * 300
    n = R.choice([5, 13, 300, 2016])
    stall = R.choice([0.1, 0.7])   # 0.7 = OI 피드가 자주 멈춘 꼴(변화 0 이 쌓여 MAD 하한이 걸린다)
    ser = [(t0 + 300 * k, 1000 + R.gauss(0, 5) if R.random() > stall else 1000.0) for k in range(n) if R.random() > 0.03]
    fn["oi_stats"].append({"s": ser, "out": json.dumps(mctx.oi_stats(ser))})
    per = {"binance": ser, "okx": [(t, v / 2) for t, v in ser[R.randrange(0, 3):] if R.random() > 0.2], "bybit": [(t + 3600 * R.choice([0, 1, 3]), v) for t, v in ser[::3]]}
    s2, u2 = mctx.oi_sum_series(per, since=t0)
    fn["sum"].append({"per": [[k, v] for k, v in per.items()], "since": t0, "out": json.dumps([s2, u2])})
for _ in range(60):
    mid = R.choice([3000.0, 2.15])
    rows = [(R.choice([None, 0.0, R.uniform(-50, 50)]), R.choice([None, 0.0, mid * R.uniform(0.85, 1.15)])) for _ in range(R.choice([0, 30, 300]))]
    fn["hl"].append({"rows": rows, "mid": mid, "bin": 5.0 if mid > 100 else 0.003, "out": json.dumps(mctx.hl_liq_levels(rows, mid, 5.0 if mid > 100 else 0.003))})
    ev_ = [(R.choice([0.0, R.uniform(2.0, 2.3), R.uniform(2900, 3100)]), R.choice([0.0, R.uniform(1, 1e5)]), R.random() < 0.5) for _ in range(200)]
    b_ = R.choice([2.5, 0.0015])
    fn["lp"].append({"ev": ev_, "bin": b_, "out": json.dumps(mctx.liq_profile(ev_, b_))})
for _ in range(80):
    k = [R.randrange(0, R.choice([5, 40, 400])) for _ in range(R.choice([0, 3, 11, 12, 40, 400]))]
    v = [R.choice([0.0, R.uniform(0, 10)]) for _ in k]
    bw = R.choice([1.5, 0.0003])
    fn["va"].append({"k": k, "v": v, "bw": bw, "out": json.dumps(mctx.value_area(k, v, bw))})
    fn["pn"].append({"k": k, "v": v, "bw": bw, "out": json.dumps(mctx.profile_nodes(k, v, bw))})
last = {}
st = [stale({"available": True, "a": 1}, last, 100.0), stale(TimeoutError(), last, 150.0), stale(TimeoutError(), last, 190.0), stale(RuntimeError("x" * 300), last, 230.0)]
(OUT / "mc.json").write_text(json.dumps({"now": NOW, "col": col, "col_x": col_x, "hist": hist, "prof": prof, "vprof": vprof, "cases": cases, "fn": fn,
                                         "stale": [json.dumps(s) for s in st], "burst": LIQ_BURST_60S_USD}, default=list))
print(f"사례 {len(cases)} · oi_stats {len(fn['oi_stats'])} · hl {len(fn['hl'])} · va {len(fn['va'])} → {OUT}")
