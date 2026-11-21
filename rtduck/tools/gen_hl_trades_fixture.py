#!/usr/bin/env python3
"""하이퍼리퀴드 주소단위 체결(live_hyperliquid_trade_collector_20260916.py) 일치 시험 (2026-11-21).

원본 `run()` 을 가짜 websockets(사건 열 · 보낸 글자 기록)·가짜 시계(time.time)·빠른 asyncio.sleep 로 그대로 돌린다. 합성 자료만 · 네트워크 없음.
자정을 넘기고(날짜 파일 둘 · 3일 전 파일 압축) · 재구독 스냅숏 중복 · users 없는 체결 · 500행 플러시 · 20초 플러시 · 깨진 JSON·비사전·못 읽는 칸(연결 접음)
· 서버 닫음·접속 실패(공백 = 처음 끊긴 시각 → 다시 붙은 시각) · 30초 무음(ping).

    python3 rtduck/tools/gen_hl_trades_fixture.py            # tests/fixtures/hl_trades_{input,golden}.json
    python3 rtduck/tools/gen_hl_trades_fixture.py --check    # Rust 가 쓴 파일을 파이썬 duckdb 로 · 포지션 수집기 universe 읽기 SQL 대조
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import random
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_venue_fixture as gv  # noqa: E402 -- 소켓 막기 · 가짜 모듈·시계 도구
from duckdump import dump  # noqa: E402

COINS = ["ETH", "BTC"]
W0 = gv.W0                 # 2025-11-21 00:00 UTC 40초 앞
OLD = "2025-11-18"         # 날짜가 넘어가면 압축되는 3일 전 파일(미리 둔다)


def gen(rng) -> dict:
    e = gv.Ev(rng, W0 - 30)
    tid = [137_993_694_000_000]
    addrs = [f"0x{rng.getrandbits(160):040X}" for _ in range(12)]

    def tr(coin="ETH", **over):
        tid[0] += rng.randint(1, 50)
        t = {"coin": coin, "side": rng.choice("AB"), "px": f"{(2747.4 if coin == 'ETH' else 90123.0) + rng.gauss(0, 2):.1f}", "sz": f"{rng.uniform(0.001, 20):.4f}",
             "time": e.ms(0.02), "tid": tid[0], "hash": f"0x{rng.getrandbits(256):064x}", "users": rng.sample(addrs, 2)}
        t.update(over)
        return {k: v for k, v in t.items() if v is not gv._DROP}

    def msg(dt, rows):
        e.msg(dt, {"channel": "trades", "data": rows})
        return rows

    e.add(0.2, "o")
    e.msg(0.05, {"channel": "subscriptionResponse", "data": {"method": "subscribe", "subscription": {"type": "trades", "coin": "ETH"}}})
    snap = msg(0.05, [tr(time=e.ms(5)) for _ in range(3)])                       # 구독 직후 스냅숏
    for k in range(25):
        msg(rng.uniform(0.5, 1.5), [tr(rng.choice(COINS)) for _ in range(rng.randint(1, 4))])
    msg(0.3, [tr(users=["0xAA"]), tr(users=gv._DROP), tr(users=None), tr(hash=gv._DROP), tr(hash=None), tr(px=2747, sz=1, time=str(e.ms())), tr(users=["0xAbC", "0xDeF"])])
    msg(0.2, [tr() for _ in range(520)])                                         # 500 넘음 = 바로 플러시
    for k in range(10):
        msg(3.0, [tr()])                                                         # 20초 플러시(자정을 넘긴다)
    e.add(31.0, "w")                                                             # 30초 무음 = ping(끊지 않음)
    last = msg(0.5, [tr(), tr()])
    e.add(0.3, "c", "ConnectionClosedError: received 1011")
    e.add(5.0, "f", "OSError: 가짜 접속 실패")
    e.add(5.0, "o")
    msg(0.05, snap[-1:] + last + [tr()])                                         # 재구독 스냅숏: 이미 적은 것(cut 이하)은 버린다
    for k in range(8):
        msg(2.6, [tr()])
    for bad in ('{"channel": "trades", "data": [', "[1, 2]", json.dumps({"channel": "trades", "data": {"a": 1}}),
                json.dumps({"channel": "trades", "data": [tr(px="x")]}), json.dumps({"channel": "trades", "data": [tr(tid=gv._DROP)]})):
        msg(0.5, [tr()])
        e.msg(0.1, bad)                                                          # 예외 = 연결 접음(버퍼는 쓰고 공백)
        e.add(5.0, "o")
    for k in range(12):
        msg(2.0, [tr(rng.choice(COINS))])
    e.add(0.3, "c", "ConnectionClosedOK: received 1000 (OK)")
    return {"coins": COINS, "t0": e.ev[0][0] - 0.2, "events": e.ev}


def python_reference(inp, d: Path) -> list:
    m = __import__("scripts.live_hyperliquid_trade_collector_20260916", fromlist=["x"])
    clock = [0.0]
    m.time = gv._ns(time, time=lambda: clock[0])
    m.asyncio = gv._ns(asyncio, sleep=gv._fast_sleep)
    m.DB_ROOT, m.COINS, m._last_db = d, list(inp["coins"]), None
    ev, st, sent = inp["events"], {"i": 0}, []

    def nxt(ks):
        if st["i"] >= len(ev):
            raise asyncio.CancelledError()
        t, k, body = ev[st["i"]]
        gv._want(k in ks, (st["i"], k, ks))
        st["i"] += 1
        clock[0] = t
        return k, body

    class WS:
        async def send(self, s):
            sent.append(s)

        async def recv(self):
            k, body = nxt("twc")
            if k == "t":
                return body
            if k == "w":
                raise asyncio.TimeoutError()
            raise gv._exc(body)

    class Conn:
        async def __aenter__(self):
            k, body = nxt("of")
            if k == "f":
                raise gv._exc(body)
            return WS()

        async def __aexit__(self, *a):
            return False

    clock[0] = inp["t0"]
    with gv._fake_modules(websockets=types.SimpleNamespace(connect=lambda *a, **k: Conn())), gv._quiet():
        try:
            asyncio.run(m.run())
        except asyncio.CancelledError:
            pass
    assert st["i"] == len(ev), "사건을 다 먹지 못했다"
    return sent


def seed_old(d: Path) -> None:
    p = d / "_".join(COINS) / f"{OLD}.duckdb"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"3-days-old placeholder\n" * 100)


def dump_dir(d: Path) -> dict:
    out = {}
    for p in sorted((d / "_".join(COINS)).iterdir()):
        out[p.name] = gzip.decompress(p.read_bytes()).hex() if p.suffix == ".gz" else dump(p) if p.suffix == ".duckdb" else p.read_bytes().hex()
    return out


def universe_sql(d: Path) -> dict:
    """HL 포지션 수집기가 체결 파일을 읽는 SQL(live_hyperliquid_positions_collector_20260924.py universe) -- 파이썬 duckdb 로 두 쪽 파일에."""
    import duckdb
    out = {}
    for p in sorted((d / "_".join(COINS)).glob("*.duckdb")):
        con = duckdb.connect(str(p), read_only=True)
        out[p.name] = con.execute("""SELECT coin, addr, sum(n) n, sum(usd) usd FROM (
              SELECT coin, buyer AS addr, count(*) n, sum(px * sz) usd FROM hl_trades GROUP BY 1, 2
              UNION ALL SELECT coin, seller, count(*), sum(px * sz) FROM hl_trades GROUP BY 1, 2) GROUP BY 1, 2 ORDER BY 1, 4 DESC, 2""").fetchall()
        out[p.name] = [[repr(x) if isinstance(x, float) else x for x in r] for r in out[p.name]]
        con.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        inp = json.loads((FIX / "hl_trades_input.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            seed_old(td / "py")
            seed_old(td / "rs")
            python_reference(inp, td / "py")
            env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
            r = subprocess.run(["bash", "-c", 'ulimit -v 4000000; exec timeout 300 "$@"', "_", "cargo", "run", "-q", "-j", "2", "--release", "--example", "replay_hl_trades",
                                str(FIX / "hl_trades_input.json"), str(td / "rs")], cwd=ROOT / "rtduck", env=env, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            x, y = dump_dir(td / "py"), dump_dir(td / "rs")
            print(f"  파이썬 duckdb 가 Rust 파일을 읽음: {'같음' if x == y else '다름'} (파일 {sorted(y)})")
            assert x == y
            u1, u2 = universe_sql(td / "py"), universe_sql(td / "rs")
            print(f"  주소별 합계 SQL: {'같음' if u1 == u2 else '다름'} ({ {k: len(v) for k, v in u1.items()} })")
            assert u1 == u2
        print("check OK")
        return
    inp = gen(random.Random(20261129))
    with tempfile.TemporaryDirectory() as td:
        seed_old(Path(td))
        sent = python_reference(inp, Path(td))
        golden = {"sent": sent, "files": dump_dir(Path(td))}
    (FIX / "hl_trades_input.json").write_text(json.dumps(inp, separators=(",", ":")))
    (FIX / "hl_trades_golden.json").write_text(json.dumps(golden, separators=(",", ":")))
    print({k: ({t: len(x["rows"]) for t, x in v.items()} if isinstance(v, dict) else len(v) // 2) for k, v in golden["files"].items()}, f"보낸 글자 {len(sent)}",
          f"{(FIX / 'hl_trades_input.json').stat().st_size // 1024}+{(FIX / 'hl_trades_golden.json').stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
