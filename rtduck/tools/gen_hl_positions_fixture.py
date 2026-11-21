#!/usr/bin/env python3
"""HL 고래 포지션(live_hyperliquid_positions_collector_20260924.py) 일치 시험 (2026-11-21).

원본 `run()` 을 그대로: 입력 체결 = 합성 날짜별 DuckDB(HL_ROOT/<코인>/<날짜>.duckdb -- 체결 수집기와 같은 표) · HTTP = 가짜 aiohttp
(요청 본문 글자 기록 · 응답은 «그 주소·코인을 몇 번째 묻는가» 로 정한다) · 시계 = asyncio.sleep 만큼 간다(요청은 0초). 응답 열을 기록해 Rust 가 같은 열을 먹는다.
코인 BTC·SOL(다코인 합집합) · UNIVERSE_N 4 · 다시 뽑기 5초(셋째 바퀴부터 다시 뽑음). 합성 자료만 · 네트워크 없음.

    python3 rtduck/tools/gen_hl_positions_fixture.py           # tests/fixtures/hl_positions_{input,golden}.json
    python3 rtduck/tools/gen_hl_positions_fixture.py --check   # Rust 가 쓴 SQLite 에 서버 SQL 둘(고래 청산 원 · 시장 맥락 hl_pos) 대조
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "rtduck" / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rtcollect" / "tools"))
import gen_venue_fixture as gv  # noqa: E402

import duckdb  # noqa: E402

COINS = ["BTC", "SOL"]
T0 = 1_790_300_000.0                       # 2026-09-25 언저리(합성)
PX = {"BTC": 90_000.0, "SOL": 140.0, "ETH": 2500.0}
USERS = [f"0x{k:02x}" + "ab" * 19 for k in range(10)]


def trade_rows(rng) -> dict:
    """{"<코인>/<날짜>": [행…]} -- 코인마다 날짜 4개(3개만 읽힘) · 48시간 밖 체결 · 다른 코인 행. 파일은 시험마다 만든다(DuckDB 파일은 개당 수백 KB)."""
    out = {}
    for coin in COINS:
        for back in range(4):
            day = time.strftime("%Y-%m-%d", time.gmtime(T0 - back * 86400))
            rows = []
            for k in range(40):
                ts = int((T0 - back * 86400 - rng.uniform(0, 80000)) * 1000)
                b, s = rng.sample(USERS[:7] if coin == "BTC" else USERS[3:], 2)
                rows.append([coin if k % 9 else "ETH", ts, k, "0x", "B", PX[coin] * rng.uniform(0.99, 1.01), rng.uniform(0.1, 3) * (1 + USERS.index(b)), b, s])
            out[f"{coin}/{day}"] = rows
    return out


def make_trades(rows: dict, root: Path) -> None:
    """체결 수집기와 같은 표로 날짜 파일을 만든다(+ 확장자 다른 파일 하나 -- 읽으면 안 된다)."""
    for key, rs in rows.items():
        p = root / f"{key}.duckdb"
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.parent / "notes.txt").write_text("x")
        con = duckdb.connect(str(p))
        con.execute("""CREATE TABLE hl_trades (
            coin VARCHAR, ts_ms BIGINT, tid BIGINT, hash VARCHAR, side VARCHAR,
            px DOUBLE, sz DOUBLE, buyer VARCHAR, seller VARCHAR)""")
        con.executemany("INSERT INTO hl_trades VALUES (?,?,?,?,?,?,?,?,?)", rs)
        con.close()


def state_for(rng, u: str, n: int) -> dict | tuple:
    """주소 u 를 n 번째 물을 때의 응답. tuple = (상태 코드,) 또는 ("err", 이름)."""
    k = USERS.index(u)
    if n == 0 and k == 3:
        return (429,)                                                     # 30초 쉼 · 간격 잠 없이 다음 주소
    if n == 1 and k == 5:
        return (500,)
    if n == 2 and k == 6:
        return ("err", "ClientConnectorError")
    aps = []
    for coin in ("BTC", "SOL", "ETH"):
        sign = 1 if (k + n) % 3 else -1
        szi = sign * (k + 1) * (1.0 if coin != "SOL" else 30.0)
        if n >= 1 and k in (1, 4) and coin != "ETH":
            szi = 0.0 if k == 1 else szi * 0.5                          # 사라짐 · 줄어듦 → 청산 후보(청산가가 범위 안이면)
        liq = PX[coin] * (0.995 if sign > 0 else 1.004) if k % 4 != 3 else None   # None = 교차 증거금에 청산가 없음
        p = {"coin": coin, "szi": f"{szi:.4f}", "entryPx": f"{PX[coin] * 1.001:.2f}", "liquidationPx": None if liq is None else f"{liq:.4f}",
             "leverage": {"type": "cross" if k % 2 else "isolated", "value": 5 + k}, "marginUsed": f"{rng.uniform(1e4, 1e6):.2f}",
             "positionValue": f"{abs(szi) * PX[coin]:.2f}", "unrealizedPnl": f"{rng.gauss(0, 1e4):.2f}", "cumFunding": {"allTime": "1", "sinceOpen": f"{rng.gauss(0, 50):.3f}"}}
        if k == 7:
            p.pop("leverage")
            p.pop("cumFunding")
        aps.append({"type": "oneWay", "position": p})
    st = {"marginSummary": {"accountValue": f"{rng.uniform(1e5, 1e7):.2f}"}, "assetPositions": aps,
          "time": [1790300000123 + n, str(1790300000456 + n), 1790300000.5, "x"][k % 4]}
    if k == 8 and n == 1:
        st["assetPositions"] = [{"type": "oneWay", "position": [1]}]       # 참인 비사전 → AttributeError(응답 수에는 든다)
    return st


def candles_for(coin: str, n: int):
    if coin == "SOL" and n == 1:
        return []
    lo, hi = PX[coin] * 0.990, PX[coin] * 1.0005
    return [{"t": 1, "o": str(PX[coin]), "h": f"{hi:.4f}", "l": f"{lo:.4f}", "c": str(PX[coin])}, {"t": 2, "h": f"{hi * 0.999:.4f}", "l": f"{lo * 1.002:.4f}"}]


def fills_for(u: str, n: int):
    k = USERS.index(u)
    if n == 1 and k == 4:
        return (500,)
    out = []
    for coin in COINS:
        out.append({"coin": coin, "tid": 1000 * k + n, "time": 1790300100000 + n, "px": str(PX[coin] * 0.995), "sz": "1.5", "side": "A", "dir": "Close Long",
                    "startPosition": "2.0", "closedPnl": "-900.5", "liquidation": {"liquidatedUser": u.upper(), "markPx": str(PX[coin] * 0.9951), "method": "market"}})
        out.append({"coin": coin, "tid": 1000 * k + 500, "time": 1790300100001, "px": "1", "sz": "1", "side": "B", "liquidation": {"liquidatedUser": "0xother"}})
        out.append({"coin": coin, "tid": 1000 * k + 7, "time": "1790300100002", "px": "2", "sz": 3, "side": None, "liquidation": {"liquidatedUser": u, "method": None}})  # 바퀴마다 같은 tid = 한 번만
    out.append({"coin": "ETH", "tid": 5, "liquidation": {"liquidatedUser": u}})
    return out


def python_reference(d: Path, rng, tr: Path) -> tuple[list, list]:
    m = __import__("scripts.live_hyperliquid_positions_collector_20260924", fromlist=["x"])
    clock = [T0]

    async def fake_sleep(s, *a, **k):
        clock[0] += s
        await asyncio.sleep(0)

    m.time = gv._ns(time, time=lambda: clock[0])
    m.asyncio = gv._ns(asyncio, sleep=fake_sleep)
    m.DB, m.TRADES_ROOT, m.COINS, m.UNIVERSE_N, m.UNIVERSE_REFRESH_S = d / "hl_positions_btc_sol.sqlite", tr, list(COINS), 4, 5.0
    m.log = lambda msg: None
    calls: dict = {}
    steps, sent = [], []

    def respond(body):
        key = (body["type"], body.get("user") or body.get("req", {}).get("coin"))
        n = calls.get(key, 0)
        calls[key] = n + 1
        if body["type"] == "clearinghouseState":
            if n == 5:
                raise asyncio.CancelledError()                               # 바퀴 5번 뒤 각본 끝
            return state_for(rng, body["user"], n)
        if body["type"] == "candleSnapshot":
            return candles_for(body["req"]["coin"], n)
        return fills_for(body["user"], n)

    class Resp:
        def __init__(self, status, js):
            self.status, self.js = status, js

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def raise_for_status(self):
            if self.status >= 400:
                raise RuntimeError(f"ClientResponseError {self.status}")

        async def json(self):
            return self.js

    class Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def post(self, url, json=None):
            assert url == m.INFO_URL
            text = __import__("json").dumps(json)
            r = respond(json)
            sent.append(text)
            if isinstance(r, tuple) and r[0] == "err":
                steps.append({"err": r[1]})

                class Boom:
                    async def __aenter__(self):
                        raise gv._exc(r[1] + ": 가짜")

                    async def __aexit__(self, *a):
                        return False
                return Boom()
            st, js = (r[0], None) if isinstance(r, tuple) else (200, r)
            steps.append({"status": st, "json": js})
            return Resp(st, js)

    fake = types.SimpleNamespace(ClientSession=Session, ClientTimeout=lambda **k: None)
    with gv._fake_modules(aiohttp=fake):
        try:
            asyncio.run(m.run())
        except asyncio.CancelledError:
            pass
    return steps, sent


def dump(db: Path) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {"schema": [r[0] for r in con.execute("SELECT sql FROM sqlite_master ORDER BY rowid")]}
    for t in ("hl_positions", "hl_universe", "hl_liquidations", "hl_cycles"):
        out[t] = [[repr(x) if isinstance(x, float) else x for x in r] for r in con.execute(f"SELECT * FROM {t} ORDER BY rowid")]
    out["pragma"] = [con.execute("PRAGMA journal_mode").fetchone()[0], con.execute("PRAGMA auto_vacuum").fetchone()[0]]
    con.close()
    return out


def server_sql(db: Path) -> dict:
    """서버 hl_whale_liq_events(server.py:508) · 시장 맥락 hl_pos(server.py:2534) 와 같은 SQL."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {}
    for coin in COINS:
        out[f"liq:{coin}"] = con.execute("""SELECT fill_ms, px * sz, dir, side, user FROM hl_liquidations
                                      WHERE coin = ? AND fill_ms >= ?""", [coin, 0]).fetchall()
        out[f"pos:{coin}"] = con.execute("SELECT szi, liq_px FROM hl_positions WHERE coin = ? AND ts_ms >= (SELECT max(ts_ms) FROM hl_cycles)", [coin]).fetchall()
    con.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        inp = json.loads((FIX / "hl_positions_input.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            make_trades(inp["trades"], td / "tr")
            python_reference(td, random.Random(20261130), td / "tr")
            env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
            r = subprocess.run(["bash", "-c", 'ulimit -v 4000000; exec timeout 300 "$@"', "_", "cargo", "run", "-q", "-j", "2", "--release", "--example", "replay_hl_positions",
                                str(FIX / "hl_positions_input.json"), str(td / "rs.sqlite"), str(td / "tr_rs")], cwd=ROOT / "rtduck", env=env, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            x, y = server_sql(td / "hl_positions_btc_sol.sqlite"), server_sql(td / "rs.sqlite")
            for k in x:
                print(f"  {'같음' if x[k] == y[k] else '다름'}  {k}: {len(x[k])}행")
            assert x == y and x["liq:BTC"] and x["pos:SOL"]      # SOL 청산 0 = 합성 각본(그 주소의 체결 조회가 500)
        print("check OK")
        return
    rng = random.Random(20261130)
    trades = trade_rows(random.Random(7))
    with tempfile.TemporaryDirectory() as td:
        make_trades(trades, Path(td) / "tr")
        steps, sent = python_reference(Path(td), rng, Path(td) / "tr")
        golden = {"sent": sent, "db": dump(Path(td) / "hl_positions_btc_sol.sqlite")}
    inp = {"coins": COINS, "universe_n": 4, "refresh_s": 5.0, "t0": T0, "steps": steps, "trades": trades}
    (FIX / "hl_positions_input.json").write_text(json.dumps(inp, separators=(",", ":")))
    (FIX / "hl_positions_golden.json").write_text(json.dumps(golden, separators=(",", ":")))
    g = golden["db"]
    print({t: len(g[t]) for t in ("hl_positions", "hl_universe", "hl_liquidations", "hl_cycles")}, f"요청 {len(sent)}", g["hl_cycles"],
          f"{(FIX / 'hl_positions_input.json').stat().st_size // 1024}+{(FIX / 'hl_positions_golden.json').stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
