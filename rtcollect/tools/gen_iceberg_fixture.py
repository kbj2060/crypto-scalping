#!/usr/bin/env python3
"""칸별 아이스버그(live_eth_iceberg_20261016.py) 일치 시험 (2026-11-21).

원본 `main()` 을 **그대로** 돌린다 -- 바뀌는 것은 바깥뿐: DD·TAPE·OUT 경로 · 시계(time.time·gmtime = 가짜 · time.sleep(2) = 2초 전진, 끝 시각을 넘으면 멈춤).
입력 = 합성 depthdiff 시 파일 둘(.jsonl -- 원본은 지금 시 .jsonl 만 연다) + hot 테이프(직전 24h 창별 체결 = 기준선 · 그 두 시간의 배경 체결 + 아이스버그 체결)
+ 이미 있는 사건 한 줄(다시 안 쓴다). 출력 = 사건 jsonl 글자 · 바퀴마다 state 글자(crc32 · 길이 + 앞·뒤 몇 개 글자).
🔴NPY_DISABLE_CPU_FEATURES=X86_V4(np.quantile · 짝 더하기는 SIMD 와 무관하지만 다른 정답지와 같게). 합성 자료만 · 네트워크 없음.

    NPY_DISABLE_CPU_FEATURES=X86_V4 python3 rtcollect/tools/gen_iceberg_fixture.py
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
import tempfile
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_venue_fixture as gv  # noqa: E402 -- 소켓 막기

FIX, ROOT = gv.FIX, gv.ROOT
sys.path.insert(0, str(ROOT))
H0 = 497_900                     # 시 번호(합성 · 전진 시작 FWD0 뒤라 n_forward 가 센다)
T0, T_END = H0 * 3600 + 7.0, (H0 + 2) * 3600 - 30.0
TAPE_DDL = ("CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL, buy_n INTEGER, sell_n INTEGER, "
            "buy_max REAL, sell_max REAL, retail_buy_qty REAL, retail_sell_qty REAL, whale_buy_qty REAL, whale_sell_qty REAL, whale_buy_n INTEGER, "
            "whale_sell_n INTEGER, retail_buy_n INTEGER, retail_sell_n INTEGER, order_buy_n INTEGER, order_sell_n INTEGER, PRIMARY KEY (symbol, ts_sec, price_bin)) WITHOUT ROWID")


def gen(rng) -> dict:
    mid = 2500.0
    files, tape, u = {}, {}, 9_000_000
    # 기준선: 직전 24h 창마다 한 칸 체결(매도·매수 0~30)
    for w in range(H0 * 120 - 2900, H0 * 120):
        tape[(w * 30 + 3, 25000 + rng.randint(-30, 30))] = [round(rng.uniform(0, 200), 3), round(rng.uniform(0, 200), 3)]   # 90분위 ≈ 180 · 95분위 ≈ 190
    ice = {}                                             # 창 → (쪽, 표시 잔량, 체결) : 이 창에 아이스버그(또는 경계 사례)를 만든다
    for k in range(6):
        ice[H0 * 120 + 7 + 7 * k] = (1 if k % 2 == 0 else -1, 20.0, 250.0)
        ice[(H0 + 1) * 120 + 3 + 9 * k] = (1 if k % 2 else -1, 20.0, 250.0)
    for k in range(6):                                   # 경계: 체결 ≈ 최대 표시 × 1.6~1.8(표시 112) · 체결이 90~95분위 사이(185)
        ice[H0 * 120 + 5 + 7 * k] = (1 if k % 2 else -1, 112.0, 210.0)
        ice[(H0 + 1) * 120 + 6 + 9 * k] = (1 if k % 2 == 0 else -1, 20.0, 185.0)
    for h in (H0, H0 + 1):
        lines = []
        bids = {round(mid - 0.01 * (j + 1), 2): round(rng.uniform(1, 8), 3) for j in range(150)}
        asks = {round(mid + 0.01 * j, 2): round(rng.uniform(1, 8), 3) for j in range(150)}
        lines.append({"_snapshot": {"lastUpdateId": u, "bids": [[f"{p:.2f}", f"{q:.3f}"] for p, q in sorted(bids.items(), reverse=True)],
                                    "asks": [[f"{p:.2f}", f"{q:.3f}"] for p, q in sorted(asks.items())]}, "at_ms": h * 3_600_000})
        t = h * 3600 + 0.2
        first = True
        while t < h * 3600 + (1500 if h == H0 else 1200):   # 시 파일 = 앞 25분 · 20분만(크기) -- 시계는 두 시간을 다 돈다
            w = int(t) // 30
            b, a = [], []
            for _ in range(rng.randint(1, 3)):
                p = round(mid + rng.choice((-1, 1)) * rng.randint(1, 140) * 0.01, 2)
                q = round(rng.uniform(0.5, 8), 3)
                (b if p < mid else a).append([f"{p:.2f}", f"{q:.3f}"])
            if w in ice:                                 # 최우선 근처 칸에 표시 잔량을 계속 다시 채운다 · 그 칸에 체결
                side, disp, _ = ice[w]
                p = round(mid - 0.01, 2) if side == 1 else round(mid, 2)
                (b if side == 1 else a).append([f"{p:.2f}", f"{disp:.3f}"])
            u0 = u + 1
            u += rng.randint(1, 20)
            pu = u0 - 1
            if h == H0 and 1000 < t - h * 3600 < 1001.7:
                pu -= 7                                  # 끊김 → 그 창 valid 0(계속 읽는다)
            msg = {"e": "depthUpdate", "E": int(t * 1000) + 5, "T": int(t * 1000), "s": "ETHUSDT", "U": u0 if not first else lines[0]["_snapshot"]["lastUpdateId"] - 2,
                   "u": u, "pu": pu, "b": b, "a": a}
            first = False
            lines.append(msg)
            if h == H0 + 1 and abs(t - (h * 3600 + 900)) < 0.9:
                lines.append('{"e":"depthUpdate","U":')     # 깨진 줄 → valid 0
            t = round(t + rng.uniform(0.7, 1.6), 3)
        files[f"{time.strftime('%Y-%m-%dT%H', time.gmtime(h * 3600))}.jsonl"] = "".join((x if isinstance(x, str) else json.dumps(x)) + "\n" for x in lines)
        u += 50
    for s in range(H0 * 3600, (H0 + 2) * 3600, 2):        # 배경 체결
        tape[(s, 25000 + rng.randint(-5, 5))] = [round(rng.uniform(0, 3), 3), round(rng.uniform(0, 3), 3)]
    for w, (side, _, vol) in ice.items():                 # 그 칸 · 창 안 체결
        cell = 24999 if side == 1 else 25000
        tape[(w * 30 + 11, cell)] = [0.0 if side == 1 else vol, vol if side == 1 else 0.0]
    prior = json.dumps({"t_end": (H0 * 120 + 7 + 7 * 2 + 1) * 30, "side": 1, "px": 2499.9, "vol": 1.0, "max": 1.0, "start": 1.0, "end": 1.0})
    return {"symbol": "ETHUSDT", "t0": T0, "t_end": T_END, "files": files, "tape": [[s, b, q[0], q[1]] for (s, b), q in sorted(tape.items())], "prior_events": prior + "\n"}


def materialize(inp, d: Path) -> None:
    dd = d / "dd" / inp["symbol"]
    dd.mkdir(parents=True)
    for name, text in inp["files"].items():
        (dd / name).write_text(text)
    con = sqlite3.connect(d / "tape.sqlite")
    con.execute(TAPE_DDL)
    con.executemany("INSERT INTO trade_tape_1s(symbol, ts_sec, price_bin, buy_qty, sell_qty) VALUES ('ethusdt', ?, ?, ?, ?)", inp["tape"])
    con.commit()
    con.close()
    (d / "out").mkdir()
    (d / "out" / "eth_events.jsonl").write_text(inp["prior_events"])


class _End(BaseException):
    pass


def python_reference(inp, d: Path) -> dict:
    m = __import__("scripts.live_eth_iceberg_20261016", fromlist=["x"])
    m.DD, m.TAPE, m.OUT = d / "dd" / inp["symbol"], d / "tape.sqlite", d / "out"
    clock, states = [inp["t0"]], []

    def sleep(s):
        states.append((d / "out" / "eth_state.json").read_text())
        clock[0] += s
        if clock[0] > inp["t_end"]:
            raise _End()

    m.time = gv._ns(time, time=lambda: clock[0], gmtime=lambda t=None: time.gmtime(clock[0] if t is None else t), sleep=sleep)
    with gv._quiet():
        try:
            m.main()
        except _End:
            pass
    return {"events": (d / "out" / "eth_events.jsonl").read_text(), "states": states}


def main() -> None:
    assert "X86_V4" in os.environ.get("NPY_DISABLE_CPU_FEATURES", "")
    inp = gen(random.Random(20261133))
    with tempfile.TemporaryDirectory() as td:
        materialize(inp, Path(td))
        ref = python_reference(inp, Path(td))
    st = ref["states"]
    golden = {"events": ref["events"], "n_states": len(st), "states_crc": [[zlib.crc32(s.encode()), len(s)] for s in st],
              "states_text": {str(k): st[k] for k in list(range(5)) + list(range(len(st) - 5, len(st)))}}
    (FIX / "iceberg_input.json").write_text(json.dumps(inp, separators=(",", ":")))
    (FIX / "iceberg_golden.json").write_text(json.dumps(golden, separators=(",", ":")))
    print(f"사건 {ref['events'].count(chr(10))}줄(이미 있던 1 포함) · state {len(st)}", st[-1][:200],
          f"{(FIX / 'iceberg_input.json').stat().st_size // 1024}+{(FIX / 'iceberg_golden.json').stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
