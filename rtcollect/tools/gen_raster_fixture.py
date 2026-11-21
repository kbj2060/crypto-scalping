#!/usr/bin/env python3
"""호가 래스터(.f32) 일치 시험의 입력·정답지 (2026-11-21). 통과 기준 = 시각 파일 **바이트까지 같다**(정답지는 행마다 해시 + 파일 해시).

원본 scripts/live_orderflow_raster_collector_20260914.py 의 `OrderflowRasterCollector._depth_loop`·`_merge_snapshot`·`_apply`·`_prune`·`_row`·
`_RasterWriter` 를 그대로 돌린다. 바뀌는 것은 바깥뿐 -- websockets = 가짜(사건 열 · recv 가 사건을 내주기 전에 그 시각까지의 틱을 먼저 돈다) ·
스냅숏 REST(`_fetch_snapshot`) = 정해 둔 본문(None = 실패) · 시계 = 모듈의 time.time 을 사건·틱 시각으로 · asyncio.sleep = 0.
틱 몸통은 `_raster_loop`(:665-683)를 글자 그대로 옮긴 `tick()` -- 잠·지각 셈만 뺐다(가짜 시계라 지각이 없다). 합성 자료만 · 네트워크 없음.

    python3 rtcollect/tools/gen_raster_fixture.py                    # tests/fixtures/raster_{input,golden}.json
    python3 rtcollect/tools/gen_raster_fixture.py --check-readers    # Rust 가 쓴 파일을 원본 read_window·row_stats·vol_scores 로 교차 확인
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_venue_fixture as gv  # noqa: E402

FIX, ROOT = gv.FIX, gv.ROOT
SEED = 20261126
W0 = gv.W0                    # 2025-11-21 00:00 UTC 40초 앞 -- 시각 경계를 건넌다
SYM = "ethusdt"


class Feed:
    def __init__(self, rng, t0):
        self.rng, self.t, self.ev, self.snaps = rng, t0, [], []
        self.mid, self.u = 2500.0, 5_000_000
        self.bid_px: set[str] = set()
        self.ask_px: set[str] = set()

    def add(self, dt, kind, body=""):
        self.t = round(self.t + dt, 6)
        self.ev.append([self.t, kind, body])

    def snapshot(self, last=None, fail=False, width=300):
        if fail:
            self.snaps.append(None)
            return
        bids = [[f"{self.mid - 0.01 * (k + 1):.2f}", f"{self.rng.uniform(0.001, 40):.3f}"] for k in range(width)]
        asks = [[f"{self.mid + 0.01 * k:.2f}", f"{self.rng.uniform(0.001, 40):.3f}"] for k in range(width)]
        bids.append([bids[3][0], "0.000"])                       # 0 은 거른다
        bids.append([bids[5][0], "7.777"])                       # 같은 가격 두 번 = 뒤 값(자리는 처음)
        self.bid_px |= {p for p, q in bids if q != "0.000"}
        self.ask_px |= {p for p, q in asks}
        self.snaps.append(json.dumps({"lastUpdateId": self.u if last is None else last, "E": int(self.t * 1000), "T": int(self.t * 1000),
                                      "bids": bids, "asks": asks}, separators=(",", ":")))

    def depth(self, dt, U=None, pu=None, n=6, far=False, extra_a=()):
        self.t = round(self.t + dt, 6)
        u0 = self.u + 1
        self.u += self.rng.randint(1, 20)
        self.last_U = u0
        b, a = [], []
        for _ in range(n):
            off = self.rng.randint(1, 900 if far else 120) * 0.01
            q = "0" if self.rng.random() < 0.25 else f"{self.rng.lognormvariate(0, 1.5):.3f}"
            if self.rng.random() < 0.5:
                b.append([f"{self.mid - off:.2f}", q])
            else:
                a.append([f"{self.mid + off - 0.01:.2f}", q])
        a += [list(x) for x in extra_a]
        # 교차를 막는다: 지금 mid 쪽으로 넘어온 옛 레벨은 지운다(실제 거래소도 그렇다) -- 안 그러면 북이 교차해 모든 행이 무효
        b += [[p, "0"] for p in sorted(self.bid_px) if float(p) >= self.mid]
        a += [[p, "0"] for p in sorted(self.ask_px) if float(p) < self.mid]
        for side, rows in ((self.bid_px, b), (self.ask_px, a)):
            for p, q in rows:
                (side.discard if q == "0" else side.add)(p)
        d = {"e": "depthUpdate", "E": int(self.t * 1000) + 5, "T": int(self.t * 1000), "s": "ETHUSDT", "U": u0 if U is None else U,
             "u": self.u, "pu": u0 - 1 if pu is None else pu, "b": b, "a": a}
        self.ev.append([self.t, "t", json.dumps(d, separators=(",", ":"))])

    def walk(self, secs, step=0.5, drift=0.0, **kw):
        for _ in range(int(secs / step)):
            self.mid = round(self.mid + drift + self.rng.gauss(0, 0.02), 2)
            self.depth(step, **kw)


def gen(rng) -> dict:
    """스냅숏 lastUpdateId = 그 스냅숏을 부른 이벤트 «다음» 이벤트의 U(= 동기 첫 이벤트가 U <= L <= u) -- 스냅숏을 부른 이벤트 자체는 버려진다."""
    runs = []
    f = Feed(rng, W0)
    f.add(0.3, "o")

    def sync(dt=0.2):
        f.depth(dt)                               # 이 이벤트가 스냅숏을 부른다(버려진다)
        f.snapshot(last=f.u + 1)

    sync()
    f.walk(25)
    f.walk(15, far=True)                          # 먼 레벨(±9달러 -- 가지치기 ±4% 안)
    # 양성 대조용: +3.5% 매도 레벨(가지치기 ±4% 안 · ±3% 밖) -- 뒤의 +2% 드리프트 뒤 래스터 폭(±2.4%) 안으로 들어와 보인다
    f.depth(0.2, extra_a=[[f"{f.mid * 1.035:.2f}", "55.555"]])
    f.ev.append([round(f.t + 0.1, 6), "t", '{"e":"depthUpdate","U":1'])   # 깨진 JSON → 재접속
    f.t = round(f.t + 0.1, 6)
    f.add(3.0, "o")
    sync()
    f.walk(20)
    f.add(0.2, "t", json.dumps({"e": "kline"}))   # depth 아닌 이벤트는 건너뜀
    f.depth(0.3, pu=1)                            # 동기 뒤 pu 단절 → 다음 이벤트에서 다시 받는다
    sync()
    f.walk(10)
    f.add(8.0, "w")                               # 8초 조용(STALE 5초 → 무효행) 뒤 수신 시한 → 재접속
    f.add(3.0, "o")
    f.depth(0.2)
    f.snapshot(last=f.u - 1000)                   # 스트림보다 낡은 스냅숏 → U > L → 동기 전 단절 = 재접속(물러남)
    f.depth(0.3)
    f.add(4.5, "o")
    f.depth(0.2)
    f.snapshot(fail=True)                         # 스냅숏 실패 → 재접속
    f.add(6.75, "o")
    sync()
    # 1% 넘게 밀린다 → 틱이 재동기를 건다 → 그 뒤 첫 이벤트가 스냅숏을 부른다. L 은 그다음 이벤트의 U 여야 해서, 자리가 미리 정해지지 않는다 --
    #   «그 뒤 이벤트는 u < L 이면 건너뛴다» 규칙을 쓴다: L = 밀린 구간 마지막 이벤트의 U(그 이벤트에서 다시 동기된다)
    f.walk(30, drift=0.9)
    f.snapshot(last=f.last_U)
    f.walk(30)
    f.add(0.3, "c", "ConnectionClosedError: x")
    runs.append({"t0": W0, "events": f.ev, "snaps": f.snaps})
    # 재시작(같은 시 파일 -- 사이 초는 무효행으로 메운다)
    g = Feed(rng, f.t + 20.0)
    g.mid, g.u = f.mid, f.u + 500
    g.add(0.4, "o")
    g.depth(0.2)
    g.snapshot(last=g.u + 1)
    g.walk(40)
    g.add(0.3, "c", "x")
    runs.append({"t0": g.ev[0][0] - 0.4, "events": g.ev, "snaps": g.snaps})
    return {"symbol": SYM, "runs": runs}


def python_reference(inp, d: Path) -> None:
    m = __import__("scripts.live_orderflow_raster_collector_20260914", fromlist=["x"])
    import logging
    logging.disable(logging.CRITICAL)
    clock = [0.0]
    m.time = gv._ns(time, time=lambda: clock[0])
    m.asyncio = gv._ns(asyncio, sleep=gv._fast_sleep)
    m.OF_ROOT = d
    for run in inp["runs"]:
        c = m.OrderflowRasterCollector(inp["symbol"])
        c._running = True
        ev, st, snaps = run["events"], {"i": 0, "target": int(run["t0"]) + 1, "tick": 0}, list(run["snaps"])

        def tick(target):
            """`_raster_loop` 몸통(:665-683) 그대로(잠·지각 셈 뺌)."""
            clock[0] = float(target)
            sec_ms = target * 1000
            row = c._row()
            if row is None:
                c._writer.write(sec_ms, 0, math.nan, None)
                c.rows_invalid += 1
            else:
                c._writer.write(sec_ms, *row)
                c.rows_written += 1
                mid = row[1]
                if (abs(mid - c._snap_mid) / mid * 100 > m.RESYNC_DRIFT_PCT
                        or m.time.time() - c._snap_ts > m.RESYNC_INTERVAL_SEC):
                    c._want_resync = True
                if st["tick"] % 60 == 0:
                    c._prune(mid)
            st["tick"] += 1

        def nxt(ks):
            if st["i"] >= len(ev):
                raise asyncio.CancelledError()
            t, k, body = ev[st["i"]]
            while st["target"] <= t:
                tick(st["target"])
                st["target"] += 1
            gv._want(k in ks, (st["i"], k, ks))
            st["i"] += 1
            clock[0] = t
            return k, body

        async def fetch():
            b = snaps.pop(0)
            if b is None:
                raise RuntimeError("가짜 REST 실패")
            return json.loads(b)

        c._fetch_snapshot = fetch

        class WS:
            async def recv(self):
                k, body = nxt("twc")
                if k == "w":
                    raise asyncio.TimeoutError()
                if k == "c":
                    raise gv._exc(body or "ConnectionClosedError: x")
                return body

        class Conn:
            async def __aenter__(self):
                k, body = nxt("of")
                if k == "f":
                    raise gv._exc(body)
                return WS()

            async def __aexit__(self, *a):
                return False

        with gv._fake_modules(websockets=types.SimpleNamespace(connect=lambda *a, **k: Conn())), gv._quiet():
            try:
                asyncio.run(c._depth_loop())
            except asyncio.CancelledError:
                pass
        assert st["i"] == len(ev) and not snaps, ("사건·스냅숏을 다 먹지 못했다", st["i"], len(ev), snaps)
        c._writer.close()


def dump(d: Path) -> dict:
    """파일마다 크기 · crc32(zlib -- Rust 시험이 같은 식을 직접 돈다) · 헤더 · 쓴 행(메움행 ts 0 제외)의 [행 번호, crc32]."""
    import zlib
    out = {}
    for p in sorted((d / SYM.upper()).glob("*.f32")):
        b = p.read_bytes()
        rows = [[k, zlib.crc32(b[32 + k * 976:32 + (k + 1) * 976])] for k in range((len(b) - 32) // 976) if b[32 + k * 976:32 + k * 976 + 8] != b"\0" * 8]
        out[p.name] = {"size": len(b), "crc32": zlib.crc32(b), "sha256": hashlib.sha256(b).hexdigest(), "header": b[:32].hex(), "rows": rows}
    return out


def cargo_replay(inp: Path, out: Path) -> None:
    env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
    r = subprocess.run(["bash", "-c", 'ulimit -v 4000000; exec timeout 120 "$@"', "_", "cargo", "run", "-q", "-j", "2", "--release",
                        "--example", "replay_raster", "--", str(inp), str(out)], cwd=ROOT / "rtcollect", env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"cargo replay_raster 실패:\n{r.stderr}")


def readers(d: Path, inp) -> dict:
    """원본 읽는 쪽(서버 히트맵·시장 맥락이 import 하는 것): read_window(agg 1·5) · row_stats · vol_scores."""
    m = __import__("scripts.live_orderflow_raster_collector_20260914", fromlist=["x"])
    import numpy as np
    out = {}
    ends = [int(r["events"][-1][0]) for r in inp["runs"]] + [int(W0) + 30, int(W0) + 45]
    for to in ends:
        for cols, agg in ((120, 1), (60, 5)):
            w = m.read_window(SYM, to * 1000, cols, agg, root=d)
            out[f"read_window:{to}:{cols}x{agg}"] = {k: (v.tobytes().hex() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
            a = np.abs(w["qty"][np.isfinite(w["mid"])])
            if a.size:
                out[f"row_stats:{to}:{cols}x{agg}"] = {k: (np.asarray(v).tobytes().hex() if hasattr(v, "__len__") else repr(v))
                                                       for k, v in m.row_stats(a, agg).items()}
            try:
                vs = m.vol_scores(w)
                out[f"vol_scores:{to}:{cols}x{agg}"] = repr(vs)
            except Exception as e:  # noqa: BLE001 -- 자료가 짧으면 원본도 못 낸다 -- 두 쪽이 같은 예외면 같다
                out[f"vol_scores:{to}:{cols}x{agg}"] = type(e).__name__
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-readers", action="store_true")
    a = ap.parse_args()
    if a.check_readers:
        inp = json.loads((FIX / "raster_input.json").read_text())["input"]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            python_reference(inp, td / "py")
            cargo_replay(FIX / "raster_input.json", td / "rs")
            x, y = readers(td / "py", inp), readers(td / "rs", inp)
            for k in x:
                print(f"  {'같음' if x[k] == y.get(k) else '다름'}  {k}")
            assert x == y
            assert dump(td / "py") == dump(td / "rs")
        print("check-readers OK")
        return
    inp = gen(random.Random(SEED))
    with tempfile.TemporaryDirectory() as td:
        python_reference(inp, Path(td))
        golden = dump(Path(td))
    (FIX / "raster_input.json").write_text(json.dumps({"seed": SEED, "input": inp}, separators=(",", ":"), ensure_ascii=False))
    (FIX / "raster_golden.json").write_text(json.dumps(golden, separators=(",", ":")))
    print("raster: " + " · ".join(f"{n} {g['size']:,}B 유효·무효 행 {len(g['rows'])}" for n, g in golden.items())
          + f" · {(FIX / 'raster_input.json').stat().st_size // 1024}+{(FIX / 'raster_golden.json').stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
