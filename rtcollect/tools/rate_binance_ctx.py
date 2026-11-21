#!/usr/bin/env python3
"""바이낸스 맥락 OI REST 분당 요청 수: 원본 파이썬 vs rtcollect binance_ctx 를 **같은 가짜 서버 조건**에서 나란히 잰다(2026-11-22).
넘기기 전에 «Rust 가 맡으면 분당 요청·가중치가 원본과 같다»를 그림자가 아닌 길로 보이려고. 개발기 전용 · 바깥 접속 없음(127.0.0.1 만).

  - 가짜 서버 둘(127.0.0.1 · 응답 지연 --latency 초 · {"symbol","openInterest","time"}) -- 파이썬·Rust 가 각자 하나.
  - 파이썬: 원본 모듈을 불러 OI_URL·WS_BASE·밴 파일·DB 만 바꿔 run() (WS 는 닫힌 포트 -- 3초마다 다시 붙기, OI 와 무관).
  - Rust: RTCOLLECT_KIND=binance_ctx RTCOLLECT_REST=1 + 같은 가짜 주소 · DB·밴 파일은 임시 폴더.
  - 종목 목록은 둘 다 원본 기본값(빠른 4 · 느린 5). openInterest 가중치 = 1(바이낸스 문서) → 가중치 합 = 요청 수.

    python3 rtcollect/tools/rate_binance_ctx.py [--seconds 300] [--latency 0.03] [--bin rtcollect/target/release/rtcollect]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]


def server(latency: float):
    hits: Counter = Counter()   # (종목, 초) → 수

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            sym = parse_qs(urlparse(self.path).query).get("symbol", ["?"])[0]
            hits[(sym, int(time.time()))] += 1
            time.sleep(latency)
            body = json.dumps({"symbol": sym, "openInterest": "1.0", "time": int(time.time() * 1000)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, hits


PY_RUNNER = r'''
import asyncio, sys, pathlib
sys.path.insert(0, sys.argv[1])
import scripts.binance_ban_guard as bg
bg.BAN_FILE = pathlib.Path(sys.argv[3]) / "ban"
import scripts.live_binance_ctx_collector_20261001 as m
m.OI_URL = sys.argv[2]
m.WS_BASE = "ws://127.0.0.1:9/"
asyncio.run(m.run(pathlib.Path(sys.argv[3]) / "binance_ctx.sqlite"))
'''


def per_min(hits: Counter, t0: float, t1: float) -> tuple[float, dict]:
    """첫 10초(기동)·끝 5초를 빼고 분당으로."""
    a, b = int(t0) + 10, int(t1) - 5
    tot = sum(n for (_, sec), n in hits.items() if a <= sec < b)
    by = Counter()
    for (sym, sec), n in hits.items():
        if a <= sec < b:
            by[sym] += n
    span = (b - a) / 60
    return tot / span, {k: v / span for k, v in sorted(by.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--latency", type=float, default=0.03, help="가짜 서버 응답 지연(초) -- 실서버 왕복을 흉내")
    ap.add_argument("--bin", type=Path, default=ROOT / "rtcollect/target/release/rtcollect")
    a = ap.parse_args()
    assert a.bin.exists(), f"실행 파일 없음: {a.bin}"
    (sp, hp), (sr, hr) = server(a.latency), server(a.latency)
    up, ur = (f"http://127.0.0.1:{s.server_address[1]}/fapi/v1/openInterest" for s in (sp, sr))
    assert up.startswith("http://127.0.0.1:") and ur.startswith("http://127.0.0.1:")
    with tempfile.TemporaryDirectory() as dp, tempfile.TemporaryDirectory() as dr:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("BN_CTX_", "RTCOLLECT_"))}   # 종목 목록 = 둘 다 원본 기본값
        t0 = time.time()
        py = subprocess.Popen([sys.executable, "-c", PY_RUNNER, str(ROOT), up, dp], env=env, cwd=dp,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rs = subprocess.Popen([str(a.bin)], cwd=dr, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              env={**env, "RTCOLLECT_KIND": "binance_ctx", "RTCOLLECT_REST": "1", "RTCOLLECT_TEST": "1", "RTCOLLECT_OI_URL": ur,
                                   "RTCOLLECT_WS_URL": "ws://127.0.0.1:9/", "BN_CTX_DB": f"{dr}/binance_ctx.sqlite", "RTCOLLECT_BAN_FILE": f"{dr}/ban"})
        try:
            time.sleep(a.seconds)
        finally:
            t1 = time.time()
            for p in (py, rs):
                p.terminate()
            for p in (py, rs):
                p.wait(30)
    (tp, bp), (tr, br) = per_min(hp, t0, t1), per_min(hr, t0, t1)
    print(f"지연 {a.latency * 1000:.0f}ms · {a.seconds}초(앞 10초·끝 5초 뺌) -- 분당 요청(= 가중치, openInterest 1)")
    print(f"{'종목':<14}{'파이썬':>10}{'Rust':>10}{'차':>9}")
    for s in sorted(set(bp) | set(br)):
        x, y = bp.get(s, 0.0), br.get(s, 0.0)
        print(f"{s:<14}{x:>10.1f}{y:>10.1f}{(y - x) / x * 100 if x else 0:>8.2f}%")
    print(f"{'합':<14}{tp:>10.1f}{tr:>10.1f}{(tr - tp) / tp * 100 if tp else 0:>8.2f}%")
    ok = tp > 0 and abs(tr - tp) / tp <= 0.02 and set(bp) == set(br)
    print("판정:", "같다(±2% 안 · 종목 같음)" if ok else "🔴다르다")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
