#!/usr/bin/env python3
"""market_data_worker 의 화면·감시 몫(청산 요격기 tail_risk_1m · liq_burst_state.json · dashboard_state.json) 일치 시험 (2026-11-21).

원본 `TailRiskInterceptor` 의 `_db_init`·`_write_liq_burst_state`·`_update_hawkes_state`·`_compute_shadow_aftershock` 등은 그대로 부르고,
루프 몸통(_ws_loop 의 메시지 처리 :393-417 · _agg_loop 한 바퀴 :437-462)만 글자 그대로 옮긴 함수로 돈다(잠·태스크 대신 사건 열 · 가짜 시계).
10초 틱 = 경계 + 0.05초(원본 sleep(10 - now%10 + 0.05) 의 이상적 시각). dashboard_state = 원본 `write_dashboard_state()` 그대로(pd.Timestamp.now 만 가짜 시계).
🔴NPY_DISABLE_CPU_FEATURES=X86_V4 로 돌린다(서버와 같은 np.log). 합성 자료만 · 네트워크 없음.

    NPY_DISABLE_CPU_FEATURES=X86_V4 python3 rtduck/tools/gen_market_fixture.py
    NPY_DISABLE_CPU_FEATURES=X86_V4 python3 rtduck/tools/gen_market_fixture.py --check   # 감시 check_data_sources·check_dashboard 를 두 쪽 파일에
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "rtduck" / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rtcollect" / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
assert "X86_V4" in os.environ.get("NPY_DISABLE_CPU_FEATURES", ""), "NPY_DISABLE_CPU_FEATURES=X86_V4 로 돌린다"
import gen_venue_fixture as gv  # noqa: E402 -- 소켓 막기
from duckdump import dump  # noqa: E402

T0 = 1_795_312_200.0 - 600.0     # 2026-11-21 UTC 부근(합성) -- 아래 세션 시각들은 따로 고른다


def fo(rng, t, side, usd, **over):
    p = 2500.0 + rng.gauss(0, 5)
    o = {"s": "ETHUSDT", "S": side, "o": "LIMIT", "f": "IOC", "q": "1", "p": f"{p:.2f}", "ap": f"{p:.2f}", "X": "FILLED", "l": "0.1",
         "z": f"{usd / p:.3f}", "T": int(t * 1000)}
    o.update(over)
    return json.dumps({"e": "forceOrder", "E": int(t * 1000) + 3, "o": o})


def gen(rng) -> dict:
    """청산이 드문드문(조용한 분 = 0 이 쌓여 σ 붕괴) → 큰 롱 청산 폭발(Hawkes 켜짐·$1만 하한) → 감쇠 → 숏 폭발 → 깨진 메시지·0 · 끊김."""
    ev, t = [], T0
    ev.append([t, "o", ""])
    for k in range(40):                                  # 40분: 가끔 작은 청산
        t += 60
        if k % 3 == 0:
            te = round(t + rng.uniform(1, 50), 3)                      # 거래소 시각 E = 받은 시각 근처(그 1분 창에 든다)
            ev.append([te, "t", fo(rng, te - 0.2, rng.choice(("BUY", "SELL")), rng.uniform(500, 30000))])
        if k % 5 == 1:
            te = round(t + 58 + rng.uniform(0, 1), 3)                   # 분 끝 직전 = 다음 분 저장(정각 틱)에 든다
            ev.append([te, "t", fo(rng, te - 0.1, "SELL", rng.uniform(1000, 9000))])
    t += 60
    for j in range(6):                                   # 롱 청산 폭발
        ev.append([round(t + 2 * j + 0.3, 3), "t", fo(rng, t + 2 * j, "SELL", rng.uniform(2e5, 9e5))])
    ev.append([round(t + 13, 3), "t", fo(rng, t + 12.9, "SELL", 5000)])
    for bad in ('{"o": [1]}', "[1]", '{"E": "x", "o": {"S": "SELL", "z": "1", "ap": "2500"}}', '{"o": {"S": "SELL", "z": "abc", "ap": "1"}}',
                '{"o": {"S": "HOLD", "z": "1", "ap": "2500"}}', '{"E": 5, "o": {"S": "BUY", "z": 0, "ap": "2500"}}', "not json"):
        ev.append([round(t + 20 + rng.uniform(0, 5), 3), "t", bad])
    t += 600                                             # 10분 감쇠
    for j in range(4):                                   # 숏 폭발
        ev.append([round(t + 3 * j + 0.7, 3), "t", fo(rng, t + 3 * j + 0.5, "BUY", rng.uniform(3e5, 1e6))])
    ev.append([round(t + 30, 3), "e", ""])               # 예외로 끊김
    ev.append([round(t + 33, 3), "o", ""])
    ev.append([round(t + 40, 3), "n", ""])               # 정상 종료(1000) = 곧바로 다시 -- 상태 그대로
    t += 400
    ev.append([round(t, 3), "t", fo(rng, t - 0.1, "SELL", 1234.5)])
    ev.sort(key=lambda e: e[0])
    # dashboard_state: 10초마다 몇 번 + 세션 경계 시각(JPX 개장·점심 시작·점심 끝·마감 · LSE · NYSE · 주말 · 휴장 · 범위 밖)
    for k in range(5):
        ev.append([round(T0 + 15 + 10 * k + 0.123456789, 9), "d", ""])
    ev.sort(key=lambda e: e[0])
    probes = []
    for iso in ("2026-11-20T00:00:00", "2026-11-20T02:29:59", "2026-11-20T02:30:00", "2026-11-20T03:30:00", "2026-11-20T06:30:00.5", "2026-11-20T06:31:00",
                "2026-11-20T08:00:00", "2026-11-20T16:30:59", "2026-11-20T16:31:00", "2026-11-20T14:30:00", "2026-11-20T21:00:30", "2026-11-20T21:01:00",
                "2026-11-21T03:00:00", "2026-11-23T01:00:00", "2026-11-26T15:00:00", "2026-11-27T17:59:00", "2026-11-27T18:00:00", "2026-12-24T12:29:00",
                "2026-12-24T12:31:00", "2031-01-06T15:00:00", "2024-12-31T05:59:00"):
        probes.append([dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc).timestamp(), "d", ""])
    return {"symbol": "ethusdt", "t0": T0, "events": ev, "probes": probes}


def python_reference(inp, d: Path) -> dict:
    import pandas as pd
    tr = __import__("tail_risk_interceptor")
    pw = __import__("scripts.duckdb_persist_worker", fromlist=["x"])
    clock = [inp["t0"]]

    class FakeDT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime.fromtimestamp(clock[0], tz=tz)

        @classmethod
        def utcnow(cls):
            return dt.datetime.fromtimestamp(clock[0], tz=dt.timezone.utc).replace(tzinfo=None)

    tr.time = gv._ns(time, time=lambda: clock[0])
    tr.datetime = FakeDT
    tr._DB_PATH = str(d / "tail_risk.duckdb")
    tr._LIQ_BURST_STATE_PATH = d / "liq_burst_state.json"
    pw._STATE_PATH = d / "dashboard_state.json"

    class FakeTs(pd.Timestamp):
        @classmethod
        def now(cls, tz=None):
            return pd.Timestamp(dt.datetime.fromtimestamp(clock[0], tz=dt.timezone.utc)).tz_convert(tz)

    pw.pd = gv._ns(pd, Timestamp=FakeTs)
    c = tr.TailRiskInterceptor("ethusdt")
    burst = []
    orig = c._write_liq_burst_state

    def w(l, s):
        orig(l, s)
        burst.append(c._liq_burst_state_path.read_text())

    c._write_liq_burst_state = w
    c._db_init()
    st = {"last_saved_minute": -1}

    def ws_msg(raw):                                    # _ws_loop 메시지 처리 :393-417 글자 그대로
        self = c
        try:
            msg = json.loads(raw)
            o = msg.get("o", {})
            side = str(o.get("S", ""))
            qty = float(o.get("z", 0.0))
            price = float(o.get("ap", 0.0))
            ts_ms = int(msg.get("E", 0))
            qty_usd = qty * price
            self._ws_last_msg_ts = tr.time.time()
            if qty_usd > 0 and side in ("BUY", "SELL"):
                self._liq_events.append((ts_ms, side, qty_usd, price))
                long_1m, short_1m = self._aggregate_1m()
                self._update_hawkes_state(long_1m, short_1m)
                self._write_liq_burst_state(long_1m, short_1m)
        except Exception:
            pass

    def tick():                                         # _agg_loop 한 바퀴 :437-462 글자 그대로(잠 뺌)
        self = c
        long_1m, short_1m = self._aggregate_1m()
        liq_event_count_1m = self._count_liq_events_1m()
        self._update_hawkes_state(long_1m, short_1m)
        self._write_liq_burst_state(long_1m, short_1m)
        self._shadow_state = self._compute_shadow_aftershock(long_1m, short_1m)
        dt_now = tr.datetime.now()
        if dt_now.minute != st["last_saved_minute"] and dt_now.second < 15:
            st["last_saved_minute"] = dt_now.minute
            self._history_long.append(long_1m)
            self._history_short.append(short_1m)
            if len(self._history_long) >= 15:
                self.is_warmed_up = True
            self._recalculate_stats()
            bucket_ts = (dt_now - dt.timedelta(minutes=1)).replace(second=0, microsecond=0)
            self._db_insert(bucket_ts, long_1m, short_1m, liq_event_count_1m)

    dash = []
    nxt = (inp["t0"] // 10) * 10 + 10 + 0.05
    import math
    for t, kind, body in inp["events"]:
        while nxt <= t:
            clock[0] = nxt
            tick()
            nxt = (math.floor(nxt / 10) + 1) * 10 + 0.05
        clock[0] = t
        if kind == "o":
            c._ws_connected = True
            c._ws_connected_since = tr.time.time()
        elif kind == "t":
            ws_msg(body)
        elif kind == "e":
            c._ws_connected = False
        elif kind == "d":
            pw.write_dashboard_state()
            dash.append(pw._STATE_PATH.read_text())
    for t, _, _ in inp["probes"]:                       # 세션 경계 시각만(틱 없이)
        clock[0] = t
        pw.write_dashboard_state()
        dash.append(pw._STATE_PATH.read_text())
    return {"burst": burst, "dash": dash}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    inp = gen(random.Random(20261131))
    with tempfile.TemporaryDirectory() as td:
        ref = python_reference(inp, Path(td))
        golden = dict(ref, tables=dump(Path(td) / "tail_risk.duckdb"))
    (FIX / "market_input.json").write_text(json.dumps(inp, separators=(",", ":")))
    (FIX / "market_golden.json").write_text(json.dumps(golden, separators=(",", ":"), ensure_ascii=False))
    rows = golden["tables"]["tail_risk_1m"]["rows"]
    on = sum('"hawkes_active": true' in b for b in ref["burst"])
    print(f"burst {len(ref['burst'])} · dash {len(ref['dash'])} · tail_risk_1m {len(rows)}행 · hawkes 켜진 글자 {on}",
          f"{(FIX / 'market_input.json').stat().st_size // 1024}+{(FIX / 'market_golden.json').stat().st_size // 1024}KB")
    print(ref["dash"][0])
    print(ref["burst"][-1])


def check() -> None:
    """감시 ops_watchdog.check_dashboard 의 시각 판정 · check_data_sources 가 읽는 liq_burst 칸을 두 쪽 파일로."""
    inp = json.loads((FIX / "market_input.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        python_reference(inp, td / "py")
        env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
        r = subprocess.run(["bash", "-c", 'ulimit -v 4000000; exec timeout 300 "$@"', "_", "cargo", "run", "-q", "-j", "2", "--release", "--example", "replay_market",
                            str(FIX / "market_input.json"), str(td / "rs")], cwd=ROOT / "rtduck", env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        for f in ("dashboard_state.json", "liq_burst_state.json"):
            a, b = (td / "py" / f).read_text(), (td / "rs" / f).read_text()
            print(f"  {'같음' if a == b else '다름'}  마지막 {f} 글자")
            assert a == b
        import scripts.ops_watchdog as wd
        out = {}
        for side in ("py", "rs"):
            s = json.loads((td / side / "dashboard_state.json").read_text())
            out[side] = (wd.age_minutes(s.get("updated_at")) is not None, json.loads((td / side / "liq_burst_state.json").read_text()).keys())
        print("  감시가 읽는 칸(updated_at 파싱 · liq 칸 이름):", "같음" if out["py"][0] == out["rs"][0] and list(out["py"][1]) == list(out["rs"][1]) else "다름")
        assert dump(td / "py" / "tail_risk.duckdb") == dump(td / "rs" / "tail_risk.duckdb")
        print("  tail_risk_1m: 파이썬 duckdb 가 Rust 파일을 읽어 같음")
    print("check OK")


if __name__ == "__main__":
    main()
