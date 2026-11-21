#!/usr/bin/env python3
"""예약 수집기 셋(treasury_futures · macro_5m · cboe 옵션 체인 원본) 일치 시험 (2026-11-21).

원본 main() 을 그대로 부른다 -- 바뀌는 것은 바깥뿐: urllib.request.urlopen = 가짜(요청 URL·User-Agent 기록 · 정해 둔 응답 · 4xx/5xx 는 HTTPError)
· 시계(time.time · sleep = 시계만 민다) · 경로. cboe 는 저장 직후 부르는 요약기(cboe_option_summary_20261101 -- 아직 안 옮김)를 빈 함수로 막는다.
합성 응답만 · 네트워크 없음.   python3 rtcollect/tools/gen_cron_fixture.py
"""
from __future__ import annotations

import gzip
import io
import json
import random
import sqlite3
import sys
import tempfile
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_venue_fixture as gv  # noqa: E402

FIX, ROOT = gv.FIX, gv.ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
T0 = 1_792_500_000.0


def chart(rng, t_end, step, n, short=None, drop_vol=False, partial=False):
    ts = [t_end - step * (n - 1 - k) for k in range(n)]
    if partial:
        ts[-1] += 23                                      # 진행 중 봉(경계 아님)
    q = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    for k in range(n):
        px = round(100 + rng.gauss(0, 0.5), 4)
        none = k % 11 == 5
        for c in ("open", "high", "low", "close"):
            q[c].append(None if none else (int(px) if k % 13 == 7 else px))
        q["volume"].append(None if k % 7 == 3 else rng.randint(0, 900))
    if drop_vol:
        q.pop("volume")
    r = {"meta": {"symbol": "X"} | ({"shortName": short} if short else {}), "timestamp": ts, "indicators": {"quote": [q]}}
    return json.dumps({"chart": {"result": [r], "error": None}})


def gen(rng) -> dict:
    resp = {}
    t1 = int(T0) // 60 * 60
    for sym in ("ZT", "ZF", "ZN", "ZB"):
        for rg, n in (("5d", 60), ("1d", 20)):
            u = f"/v8/finance/chart/{sym}%3DF?range={rg}&interval=1m"
            resp[u] = {"status": 200, "body": chart(rng, t1, 60, n, short=f"{sym} Futures,Dec-2026" if sym != "ZB" else None, partial=True)}
    resp["/v8/finance/chart/ZF%3DF?range=5d&interval=1m"] = {"status": 429, "reason": "Too Many Requests", "body": ""}
    resp["/v8/finance/chart/ZN%3DF?range=1d&interval=1m"] = {"status": 503, "reason": "Service Unavailable", "body": ""}
    t5 = int(T0) // 300 * 300
    import urllib.parse
    keys = __import__("live_macro_5m_collector_20261028").SYMS
    for key, tk in keys.items():
        for rg, n in (("5d", 30), ("60d", 50)):
            resp[f"/v8/finance/chart/{urllib.parse.quote(tk)}?range={rg}&interval=5m"] = {"status": 200, "body": chart(rng, t5, 300, n, drop_vol=(key == "VIX"))}
    resp["/v8/finance/chart/HYG?range=60d&interval=5m"] = {"status": 500, "reason": "Internal Server Error", "body": ""}
    resp["/v8/finance/chart/TIP?range=60d&interval=5m"] = {"status": 200, "body": json.dumps({"chart": {"result": [{"indicators": {"quote": [{"open": [], "high": [], "low": [], "close": [], "volume": []}]}}]}})}  # 휴장
    for sym, seq in (("SNDK", 1234), ("SOXL", 777), ("MU", 55)):
        opts = [{"option": f"{sym}261120C00{100 + k:03d}000", "iv": rng.random(), "bid": 1.0, "ask": 1.2, "open_interest": k} for k in range(5)]
        resp[f"/api/global/delayed_quotes/options/{sym}.json"] = {"status": 200, "body": json.dumps({"timestamp": "2026-11-21 10:00:00", "data": {"symbol": sym, "seqno": seq, "options": opts}})}
    resp["/api/global/delayed_quotes/options/MU.json"] = {"status": 500, "reason": "Internal Server Error", "body": ""}
    seed_macro = [[t5 - 3600, "USDJPY", 1.0, 1.0, 1.0, 1.0, 5], [t5 - 6 * 86400, "EURUSD", 2.0, 2.0, 2.0, 2.0, 0],
                  [t5 - int(4.5 * 86400), "USDKRW", 3.0, 3.0, 3.0, 3.0, 1]]   # 1시간 전(5d) · 6일 전(60d) · 4.5일 전(60d -- 4일 문턱 바로 밖)
    return {"t0": T0, "resp": resp, "seed_macro": seed_macro, "cboe_args": ["sndk", "../x"], "cboe_seed": {"SOXL": "777"}}


def dump(db: Path) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {"schema": [r[0] for r in con.execute("SELECT sql FROM sqlite_master ORDER BY rowid")]}
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
        out[t] = [[repr(x) if isinstance(x, float) else x for x in r] for r in con.execute(f"SELECT * FROM {t} ORDER BY rowid")]
    con.close()
    return out


def python_reference(inp, d: Path) -> dict:
    clock, reqs = [inp["t0"]], []

    def urlopen(req, timeout=None):
        u = req.full_url.split("://", 1)[1].split("/", 1)[1]
        u = "/" + u
        reqs.append([u, req.get_header("User-agent")])
        clock[0] += 0.25                                  # 응답에 걸린 시간
        r = inp["resp"][u]
        if r["status"] >= 400:
            raise urllib.error.HTTPError(req.full_url, r["status"], r["reason"], {}, None)
        return io.BytesIO(r["body"].encode())

    tns = gv._ns(time, time=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    real = urllib.request.urlopen
    urllib.request.urlopen = urlopen
    rc = {}
    try:
        with gv._quiet():
            tf = __import__("scripts.live_treasury_futures_collector_20261006", fromlist=["x"])
            tf.time, tf.DB = tns, d / "treasury_futures.sqlite"
            rc["treasury_5d"] = tf.main("5d")
            rc["treasury_1d"] = tf.main("1d")
            mm = __import__("scripts.live_macro_5m_collector_20261028", fromlist=["x"])
            mm.time, mm.DB = tns, d / "macro_5m.sqlite"
            con = sqlite3.connect(mm.DB)
            con.execute(mm.DDL[0])
            con.executemany("INSERT INTO yahoo_5m VALUES(?,?,?,?,?,?,?)", inp["seed_macro"])
            con.commit()
            con.close()
            rc["macro"] = mm.main()
            sys.modules["cboe_option_summary_20261101"] = types.SimpleNamespace(write_state=lambda p: None)   # 요약기(안 옮김)
            cb = __import__("scripts.live_cboe_option_chain_collector_20261101", fromlist=["x"])
            cb.time, cb.OUT = tns, d / "cboe"
            cb.collect.__defaults__ = (d / "cboe", cb.fetch, None)
            for s, v in inp["cboe_seed"].items():
                (d / "cboe" / s).mkdir(parents=True, exist_ok=True)
                (d / "cboe" / s / "last_seqno").write_text(v)
            rc["cboe"] = cb.main(inp["cboe_args"])
    finally:
        urllib.request.urlopen = real
    files = {}
    for p in sorted((d / "cboe").rglob("*")):
        if p.is_file():
            files[str(p.relative_to(d / "cboe"))] = gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()
    return {"rc": rc, "requests": reqs, "treasury": dump(d / "treasury_futures.sqlite"), "macro": dump(d / "macro_5m.sqlite"), "cboe_files": files}


def main() -> None:
    inp = gen(random.Random(20261134))
    with tempfile.TemporaryDirectory() as td:
        ref = python_reference(inp, Path(td))
    (FIX / "cron_input.json").write_text(json.dumps(inp, separators=(",", ":")))
    (FIX / "cron_golden.json").write_text(json.dumps(ref, separators=(",", ":"), ensure_ascii=False))
    print(ref["rc"], f"요청 {len(ref['requests'])}", {k: len(ref['treasury'][k]) for k in ("treasury_fut_1m", "polls")}, {k: len(ref['macro'][k]) for k in ("yahoo_5m", "polls")},
          list(ref["cboe_files"]), [p[2] for p in ref["treasury"]["polls"]][:4], f"{(FIX / 'cron_input.json').stat().st_size // 1024}+{(FIX / 'cron_golden.json').stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
