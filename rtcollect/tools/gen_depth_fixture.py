#!/usr/bin/env python3
"""depth_diff 수집기 일치 시험의 입력·정답지 (2026-11-21). 통과 기준 = 시각 파일들의 **압축을 푼 내용이 바이트까지 같다**(파일 이름·.gz 여부 포함).

원본 scripts/live_depth_diff_collector_20260915.py 의 `run()`·`HourFile`·`_gzip_done` 을 그대로 돌린다. 바뀌는 것은 바깥뿐 --
websockets = 가짜 모듈(사건 열을 차례로) · 스냅숏 REST(urllib.urlopen) = 정해 둔 본문을 차례로(None = 실패) · 시계 = 모듈의 time.time 을 사건 시각으로 ·
asyncio.sleep = 0 · 압축 스레드는 끝에 기다린다 · 입력이 끝나면 열린 파일을 닫는다(Rust replay 도 같게). 합성 자료만 · 네트워크 없음.

    python3 rtcollect/tools/gen_depth_fixture.py                    # tests/fixtures/depth_{input,golden}.json
    python3 rtcollect/tools/gen_depth_fixture.py --check-readers    # Rust 가 쓴 파일을 live_ob5m.replay · bookdepth_feat.cum 으로 교차 확인
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
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_venue_fixture as gv  # noqa: E402

FIX, ROOT = gv.FIX, gv.ROOT
SEED = 20261125
W0 = gv.W0                      # 2025-11-21 00:00 UTC 40초 앞 -- 시각 경계를 건넌다(압축·새 파일 스냅숏)
SYM = "ethusdt"


class Book:
    def __init__(self, rng, t0):
        self.rng, self.t, self.u, self.ev, self.snaps = rng, t0, 9_000_000_000, [], []
        self.bid = {round(2500.0 - 0.01 * k, 2): round(rng.uniform(0.1, 20), 3) for k in range(1, 40)}
        self.ask = {round(2500.0 + 0.01 * k, 2): round(rng.uniform(0.1, 20), 3) for k in range(0, 39)}

    def add(self, dt, kind, body=""):
        self.t = round(self.t + dt, 6)
        self.ev.append([self.t, kind, body])

    def snapshot(self, fail=False):
        """REST 본문(열쇠 순서 = 거래소 그대로) -- 새 시 파일마다 하나 소비된다."""
        if fail:
            self.snaps.append(None)
            return
        ms = int(self.t * 1000)
        self.snaps.append(json.dumps({"lastUpdateId": self.u + 1, "E": ms + 3, "T": ms, "bids": [[f"{p:.2f}", f"{q:.3f}"] for p, q in sorted(self.bid.items(), reverse=True)],
                                      "asks": [[f"{p:.2f}", f"{q:.3f}"] for p, q in sorted(self.ask.items())]}, separators=(",", ":")))

    def frame(self, dt, **over):
        """depthUpdate 원문(바이낸스 모양 -- 공백 없음). 글자 그대로 파일에 간다."""
        self.t = round(self.t + dt, 6)
        u0, self.u = self.u + 1, self.u + self.rng.randint(1, 30)
        b, a = [], []
        for _ in range(self.rng.randint(0, 4)):
            p = round(2500.0 - 0.01 * self.rng.randint(1, 45), 2)
            q = 0.0 if self.rng.random() < 0.2 else round(self.rng.uniform(0.1, 30), 3)
            self.bid[p] = q
            b.append([f"{p:.2f}", f"{q:.3f}"])
        for _ in range(self.rng.randint(0, 4)):
            p = round(2500.0 + 0.01 * self.rng.randint(0, 45), 2)
            q = 0.0 if self.rng.random() < 0.2 else round(self.rng.uniform(0.1, 30), 3)
            self.ask[p] = q
            a.append([f"{p:.2f}", f"{q:.3f}"])
        ms = int(self.t * 1000)
        d = {"e": "depthUpdate", "E": ms + 2, "T": ms, "s": "ETHUSDT", "U": u0, "u": self.u, "pu": u0 - 1, "b": b, "a": a}
        d.update(over)
        self.ev.append([self.t, "t", json.dumps({k: v for k, v in d.items() if v is not gv._DROP}, separators=(",", ":"))])


def gen(rng) -> dict:
    D = gv._DROP
    bk = Book(rng, W0)
    bk.add(0.0, "o")
    bk.snapshot()                     # 첫 파일(23시)
    for _ in range(60):
        bk.frame(0.1)
    # 구멍(pu 어긋남 · pu 없음 · u 없음/null 다음은 대조 안 함) · 시각 칸 변형
    bk.frame(0.1, pu=123)
    bk.frame(0.1, pu=D)
    bk.frame(0.1, u=None)
    bk.frame(0.1, pu=5)               # 앞 u 가 null 이라 구멍 아님
    bk.frame(0.1, u=D)
    bk.frame(0.1)
    bk.frame(0.1, T=D)                # E 로
    bk.frame(0.1, T=D, E=D)           # 받은 시각
    bk.frame(0.1, T=0)                # 0 은 거짓 → E
    bk.frame(0.1, T=str(int(bk.t * 1000)))
    bk.frame(0.1, pu=float(bk.u))     # 1.0 == 1 (구멍 아님)
    bk.frame(0.1, extra="é가")        # 비 ASCII 원문 그대로
    for _ in range(40):
        bk.frame(0.1)
    # 예외로 끊김(같은 시 → 이어 쓰기, 스냅숏 없음) · 깨끗한 닫힘(곧바로 다시) · 접속 실패
    bk.add(0.2, "c", "ConnectionClosedError: x")
    bk.add(3.0, "o")
    for _ in range(10):
        bk.frame(0.1)
    for bad in ('{"e":"depthUpdate","T":1', "[1,2]", '"x"', json.dumps({"e": "depthUpdate", "T": "abc"})):
        bk.ev.append([round(bk.t + 0.1, 6), "t", bad])
        bk.t = round(bk.t + 0.1, 6)
        bk.add(3.0, "o")
        for _ in range(5):
            bk.frame(0.1)
    bk.add(0.1, "e")
    bk.add(0.0, "o")
    for _ in range(5):
        bk.frame(0.1)
    bk.add(0.1, "c", "x")
    bk.add(3.0, "f", "OSError: 가짜")
    bk.add(3.0, "o")
    # 시각 경계를 건넌다(00시): 23시 파일 압축 · 00시 새 파일 = 스냅숏 실패(null)
    while bk.t < W0 + 40.5:
        bk.frame(0.3)
    bk.snapshot(fail=True)
    for _ in range(30):
        bk.frame(0.2)
    # 🔴늦게 온 지난 시 프레임은 넣지 않는다: 원본은 압축 스레드가 도는 파일을 다시 열어 이어 쓰고, 그 스레드가 끝나며 원본을 지워
    #   이어 쓴 줄이 사라질 수 있다(경쟁 -- 결과가 돌릴 때마다 다르다). REMAINING/보고의 «이상한 점».
    for _ in range(20):
        bk.frame(0.2)
    bk.add(0.1, "c", "x")
    return {"symbol": SYM, "events": bk.ev, "snaps": bk.snaps}


def python_reference(inp, d: Path) -> list:
    m = __import__("scripts.live_depth_diff_collector_20260915", fromlist=["x"])
    import logging
    logging.disable(logging.CRITICAL)
    clock = [inp["events"][0][0]]
    m.time = gv._ns(time, time=lambda: clock[0])
    m.asyncio = gv._ns(asyncio, sleep=gv._fast_sleep)
    snaps, urls = list(inp["snaps"]), []

    def urlopen(url, timeout=None):
        urls.append(url)
        b = snaps.pop(0)
        if b is None:
            raise OSError("가짜 REST 실패")
        import io
        return io.BytesIO(b.encode())

    files = []
    real_hf = m.HourFile

    class HF(real_hf):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            files.append(self)

    m.HourFile = HF
    real_open = m.urllib.request.urlopen
    m.urllib.request.urlopen = urlopen
    ev, st = inp["events"], {"i": 0}

    def nxt(ks):
        if st["i"] >= len(ev):
            raise asyncio.CancelledError()
        t, k, body = ev[st["i"]]
        gv._want(k in ks, (st["i"], k, ks))
        st["i"] += 1
        clock[0] = t
        return k, body

    class WS:
        def __aiter__(self):
            return self

        async def __anext__(self):
            k, body = nxt("tce")
            if k == "t":
                return body
            if k == "e":
                raise StopAsyncIteration
            raise gv._exc(body or "ConnectionClosedError: x")

    class Conn:
        async def __aenter__(self):
            k, body = nxt("of")
            if k == "f":
                raise gv._exc(body)
            return WS()

        async def __aexit__(self, *a):
            return False

    try:
        with gv._fake_modules(websockets=types.SimpleNamespace(connect=lambda *a, **k: Conn())), gv._quiet():
            try:
                asyncio.run(m.run(inp["symbol"], d))
            except asyncio.CancelledError:
                pass
        assert st["i"] == len(ev), "사건을 다 먹지 못했다"
        for f in files:
            f.close()                          # 입력 끝 = 열린 파일을 닫는다(압축 없음)
        for t in threading.enumerate():
            if t is not threading.current_thread() and t.daemon:
                t.join()
    finally:
        m.HourFile, m.urllib.request.urlopen = real_hf, real_open
    return urls


def dump(d: Path) -> list:
    """[(파일 이름, 압축을 푼 내용 글자)] -- 이름 순."""
    out = []
    for p in sorted((d / SYM.upper()).iterdir()):
        raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
        out.append([p.name, raw.decode("utf-8")])
    return out


def cargo_replay(inp: Path, out: Path) -> None:
    env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
    r = subprocess.run(["bash", "-c", 'ulimit -v 4000000; exec timeout 120 "$@"', "_", "cargo", "run", "-q", "-j", "2", "--release",
                        "--example", "replay_depth", "--", str(inp), str(out)], cwd=ROOT / "rtcollect", env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"cargo replay_depth 실패:\n{r.stderr}")


def readers(d: Path) -> dict:
    """원본 읽는 쪽: live_ob5m.replay(북 재구성 · 매 초 콜백 · 사건) + bookdepth_feat.cum(누적 수량) -- 시 파일마다."""
    import numpy as np
    ob = __import__("scripts.live_ob5m_20261026", fromlist=["x"])
    bf = __import__("scripts.live_bookdepth_feat_20261028", fromlist=["x"])
    out = {}
    for p in sorted((d / SYM.upper()).iterdir()):
        raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
        secs, evs = [], []

        def on_sec(t, bid, ask, bb, ba):
            secs.append([t, bb, ba, float(bid.sum()), float(ask.sum()), repr(bf.cum(bid, ask, bb, ba))])

        st = ob.replay(iter([ln for ln in raw.split(b"\n") if ln]), on_sec, lambda e: evs.append(sorted(e.items())))
        out[p.name] = [st, secs, evs]
        assert np is not None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-readers", action="store_true")
    a = ap.parse_args()
    if a.check_readers:
        inp = json.loads((FIX / "depth_input.json").read_text())["input"]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            python_reference(inp, td / "py")
            cargo_replay(FIX / "depth_input.json", td / "rs")
            x, y = readers(td / "py"), readers(td / "rs")
            assert x.keys() == y.keys(), (x.keys(), y.keys())
            for k in x:
                print(f"  {'같음' if x[k] == y[k] else '다름'}  {k}: 상태 {x[k][0]} · 초 {len(x[k][1])} · 사건 {len(x[k][2])}")
            assert x == y
            assert dump(td / "py") == dump(td / "rs")
        print("check-readers OK")
        return
    inp = gen(random.Random(SEED))
    with tempfile.TemporaryDirectory() as td:
        urls = python_reference(inp, Path(td))
        golden = {"files": dump(Path(td)), "urls": urls}
    (FIX / "depth_input.json").write_text(json.dumps({"seed": SEED, "input": inp}, separators=(",", ":"), ensure_ascii=False))
    (FIX / "depth_golden.json").write_text(json.dumps(golden, separators=(",", ":"), ensure_ascii=False))
    print(f"depth: 사건 {len(inp['events'])} · 스냅숏 {len(inp['snaps'])} · 파일 " + " · ".join(f"{n} {c.count(chr(10))}줄" for n, c in golden["files"])
          + f" · REST {len(urls)} · {(FIX / 'depth_input.json').stat().st_size // 1024}+{(FIX / 'depth_golden.json').stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
