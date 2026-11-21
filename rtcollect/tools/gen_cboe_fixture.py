#!/usr/bin/env python3
"""CBOE 옵션 요약기(cboe_option_summary_20261101.write_state → cboe_option_state.json) 일치 시험 (2026-11-21).

원본 write_state·summarize·Deribit 계산 도구를 그대로 부른다 -- 바뀌는 것은 바깥뿐: 시계(time.time · datetime.now) · 실현변동성 klines REST(가짜 응답 ·
urlopen 막음) · 경로(상태·캐시·스냅숏 폴더). 스냅숏 = 합성 체인(행사가 60~150 · 만기 4 + 지난 만기 · 스마일 · IV 0 · 거래량·델타·감마)
을 종목 폴더 <SYM>/<UTC날짜>/<HHMMSS>.json.gz 에 차례로 두고 «그 시각까지 있던 파일» 로 평소 대비 거래량을 잰다.
🔴NPY_DISABLE_CPU_FEATURES=X86_V4(np.log·exp 를 libm 과 같게 -- 서버와 같다). 합성 자료만 · 네트워크 없음.

    NPY_DISABLE_CPU_FEATURES=X86_V4 python3 rtcollect/tools/gen_cboe_fixture.py
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import math
import os
import random
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_venue_fixture as gv  # noqa: E402

FIX, ROOT = gv.FIX, gv.ROOT
sys.path.insert(0, str(ROOT / "scripts"))
ET = __import__("zoneinfo").ZoneInfo("America/New_York")


def chain(rng, sym, S, now_utc: dt.datetime, vol_scale=1.0):
    today = now_utc.astimezone(ET).date()
    exps = [today + dt.timedelta(days=d) for d in (-1, 2, 9, 30, 64)]   # -1 = 지난 만기(빠진다)
    opts = []
    for e in exps:
        for k in [60 + 2.5 * j for j in range(37)]:
            for cp in "CP":
                t = max((dt.datetime.combine(e, dt.time(16), ET) - now_utc).total_seconds() / (365 * 86400), 1e-4)
                iv = 0.0 if ((cp == "C" and k < S * 0.75) or (cp == "P" and k > S * 1.3)) else round(0.55 + 0.9 * (math.log(k / S)) ** 2 - 0.08 * math.log(k / S) + rng.uniform(-0.01, 0.01), 4)
                mid = max(0.05, (S - k if cp == "C" else k - S) if iv == 0 else abs(S - k) * 0.3 + S * iv * math.sqrt(t) * 0.4)
                vol = 0 if rng.random() < 0.4 else rng.randint(1, 900) * vol_scale
                o = {"option": f"{sym}{e:%y%m%d}{cp}{int(k * 1000):08d}", "bid": round(mid * 0.97, 2), "ask": round(mid * 1.03, 2), "iv": iv,
                     "open_interest": float(rng.randint(0, 5000)), "volume": float(vol), "delta": round(rng.uniform(0, 1) * (1 if cp == "C" else -1), 4),
                     "gamma": round(rng.uniform(0, 0.05), 4), "last_trade_price": round(mid, 2) if rng.random() < 0.8 else 0.0,
                     "last_trade_time": (now_utc - dt.timedelta(minutes=rng.randint(1, 300))).astimezone(ET).strftime("%Y-%m-%dT%H:%M:%S") if vol else None}
                if rng.random() < 0.03:
                    o["delta"] = float("nan")                  # NaN 델타(원본이 0 으로)
                opts.append(o)
    opts.append({"option": "BADNAME", "iv": 0.5, "bid": 1, "ask": 1, "open_interest": 1})
    return {"symbol": sym, "current_price": S, "iv30": 62.5, "seqno": rng.randint(1, 10**6),
            "last_trade_time": (now_utc - dt.timedelta(minutes=3)).astimezone(ET).strftime("%Y-%m-%dT%H:%M:%S"), "options": opts}


def gen(rng) -> dict:
    D1, D2 = dt.datetime(2026, 11, 19, tzinfo=dt.timezone.utc), dt.datetime(2026, 11, 20, tzinfo=dt.timezone.utc)
    steps = []
    for when, sym, S, write in ((D1 + dt.timedelta(hours=14, minutes=50), "SNDK", 101.13, False), (D1 + dt.timedelta(hours=19, minutes=30), "SNDK", 99.81, False),
                                (D2 + dt.timedelta(hours=15, minutes=30), "SNDK", 100.37, True), (D2 + dt.timedelta(hours=15, minutes=40), "SNDK", 100.52, True),
                                (D2 + dt.timedelta(hours=15, minutes=41), "SOXL", 37.29, True), (D2 + dt.timedelta(hours=22, minutes=5), "SNDK", 100.61, True),
                                # 서머타임 끝나는 날 01:3x 동부(fold=0 = EDT) -- et_to_utc 끝 시각 경계
                                (dt.datetime(2026, 11, 1, 5, 40, tzinfo=dt.timezone.utc), "MU", 88.4, True)):
        steps.append({"rel": f"{sym}/{when:%Y%m%d}/{when:%H%M%S}.json.gz", "now": when.timestamp() + 7.25, "write": write,
                      "raw": json.dumps({"data": chain(rng, sym, S, when, 1.0 if write else 1.3)})})   # 원본 응답 글자(NaN 낱말 포함 -- 정답지 JSON 안에는 글자로)
    closes = [round(100 + 3 * math.sin(i / 7) + rng.gauss(0, 0.4), 2) for i in range(169)]
    kl = [[1, "0", "0", "0", f"{c:.2f}", "0"] for c in closes]
    return {"steps": steps, "klines": {"SNDK": json.dumps(kl), "SOXL": None, "MU": None}}


def python_reference(inp, d: Path) -> dict:
    summ = __import__("cboe_option_summary_20261101")
    clock = [0.0]

    class FakeDT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime.fromtimestamp(clock[0], tz=tz)

    summ.datetime = FakeDT
    summ.time = gv._ns(time, time=lambda: clock[0])
    reqs = []

    def urlopen(req, timeout=None):
        reqs.append(req.full_url)
        sym = req.full_url.split("symbol=")[1].split("USDT")[0]
        body = inp["klines"][sym]
        if body is None:
            raise urllib.error.HTTPError(req.full_url, 500, "x", {}, None)
        return io.BytesIO(body.encode())

    real = urllib.request.urlopen
    urllib.request.urlopen = urlopen
    states = []
    try:
        for st in inp["steps"]:
            p = d / "cboe" / st["rel"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(gzip.compress(st["raw"].encode(), 6))
            if st["write"]:
                clock[0] = st["now"]
                summ.RV_CACHE = d / "rv.json"
                summ.realized_vol_7d.__defaults__ = (d / "rv.json", None)
                summ.write_state(p, d / "state.json")
                states.append((d / "state.json").read_text(encoding="utf-8"))
    finally:
        urllib.request.urlopen = real
    return {"states": states, "requests": reqs, "rv_cache": (d / "rv.json").read_text() if (d / "rv.json").exists() else None}


def main() -> None:
    assert "X86_V4" in os.environ.get("NPY_DISABLE_CPU_FEATURES", "")
    inp = gen(random.Random(20261135))
    with tempfile.TemporaryDirectory() as td:
        ref = python_reference(inp, Path(td))
    (FIX / "cboe_input.json").write_text(json.dumps(inp, separators=(",", ":")))
    (FIX / "cboe_golden.json").write_text(json.dumps(ref, separators=(",", ":"), ensure_ascii=False))
    print(f"state {len(ref['states'])} · 요청 {len(ref['requests'])} · 마지막 {len(ref['states'][-1])}B", ref["rv_cache"],
          f"{(FIX / 'cboe_input.json').stat().st_size // 1024}+{(FIX / 'cboe_golden.json').stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
