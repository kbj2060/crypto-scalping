"""liq_tab 정답지: 원본 live_liq_tab_20261107.main() 을 합성 klines(가짜 urlopen) · 합성 lake parquet · 합성 hot SQLite · 가짜 시계로 두 번.

  1회: 1h 봉 캐시 없음(70일 = 두 쪽 · 진행 중 봉 버림) · ETH(계산) · BTC(봉 230 · lake 청산 없음 -- 띠·실측 없이) · 묶음 파일에 옛 SOL 몫(남김)
  2회: 한 시간 뒤(캐시 + 뒤 한 쪽 + 앞쪽 메움 한 쪽)
출력 = tests/fixtures/liq_tab_{input,golden}.json. 바깥 접속 없음.
  NPY_DISABLE_CPU_FEATURES=X86_V4 python3 rtduck/tools/gen_liq_tab_fixture.py
"""
from __future__ import annotations

import io
import json
import os
import random
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import scripts.live_liq_tab_20261107 as L  # noqa: E402

FIX = ROOT / "rtduck/tests/fixtures"
rnd = random.Random(20261121)
HR, D = 3_600_000, 86_400_000
NOW0 = 1_795_231_200_000 + 20 * 60_000          # 2026-11-21T03:20Z
H_LAST = NOW0 // HR * HR                         # 진행 중인 봉의 시작


def klines(sym: str, n_hours: int, px: float) -> list:
    rows = []
    for j in range(n_hours + 2):                 # 끝 둘 = 1회엔 진행 중 하나 + 미래 하나(2회의 마지막 닫힌 봉)
        t = H_LAST - (n_hours - 1 - j) * HR
        o = px
        c = round(o * (1 + rnd.gauss(0, 0.007)), 2)
        hi, lo = round(max(o, c) * (1 + abs(rnd.gauss(0, 0.003))), 2), round(min(o, c) * (1 - abs(rnd.gauss(0, 0.003))), 2)
        qv = round(rnd.uniform(1e7, 9e7), 2)
        rows.append([t, f"{o:.2f}", f"{hi:.2f}", f"{lo:.2f}", f"{c:.2f}", "1", t + HR - 1, f"{qv:.2f}", 1, "1", "1", "0"])
        px = c
    return rows


K = {"ETHUSDT": klines("ETHUSDT", 70 * 24, 2500.0), "BTCUSDT": klines("BTCUSDT", 230, 60000.0)}   # BTC = 봉 230(계산 · lake 청산 없음)


def respond(url: str, now_ms: int) -> bytes:
    q = dict(x.split("=") for x in url.split("?")[1].split("&"))
    st = int(q["startTime"])
    rows = [r for r in K[q["symbol"]] if r[0] >= st and r[0] <= now_ms][: int(q["limit"])]   # 거래소는 아직 안 연 봉은 안 준다
    return json.dumps(rows).encode()


def lake_and_hot(lake: Path, hot: Path) -> dict:
    eth = K["ETHUSDT"]
    lo_t, hi_t = eth[24 * 30][0], H_LAST - 2 * D        # 40일 전 ~ 이틀 전 = lake · 그 뒤 = hot
    bn, okx = [], []
    used = set()

    def ts(a, b):
        while True:
            t = rnd.randrange(a, b)
            if t not in used:                           # 같은 ms 는 만들지 않는다(원본 sort_values 가 안정 정렬이 아니다)
                used.add(t)
                return t
    for _ in range(20000):            # 시간당 ~20건 -- 시간 합이 Kahan 과 그냥 합에서 갈릴 만큼
        t = ts(lo_t, hi_t)
        px = float(eth[(t - eth[0][0]) // HR][4])
        bn.append((t, rnd.choice(["long", "short"]), round(rnd.uniform(1e3, 5e5), 2), round(px * rnd.uniform(0.995, 1.005), 2)))
    for _ in range(2500):
        t = ts(lo_t, H_LAST)
        px = float(eth[(t - eth[0][0]) // HR][4])
        inst = "ETH-USDT-SWAP" if rnd.random() < 0.8 else "BTC-USDT-SWAP"
        okx.append((inst, t, rnd.choice(["long", "short"]), round(px * rnd.uniform(0.99, 1.01), 2), round(rnd.uniform(0.1, 80), 3)))
    bn.append((ts(lo_t, hi_t), "long", None, 2500.0))   # usd NULL = 버림
    by_day: dict = {}
    for r in bn:
        by_day.setdefault(r[0] // D, []).append(r)
    for day, rs in by_day.items():
        p = lake / f"binance/liquidations/coin=ETH/date={day}/part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"ts_ms": [r[0] for r in rs], "side": [r[1] for r in rs], "usd": pa.array([r[2] for r in rs], pa.float64()),
                                 "price": [r[3] for r in rs], "symbol": ["ETHUSDT"] * len(rs)}), p)
    by_day = {}
    for r in okx:
        by_day.setdefault(r[1] // D, []).append(r)
    for day, rs in by_day.items():
        p = lake / f"okx/liquidations/coin=ALL/date={day}/part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"inst_id": [r[0] for r in rs], "ts_ms": [r[1] for r in rs], "pos_side": [r[2] for r in rs], "bk_px": [r[3] for r in rs],
                                 "sz_base": [r[4] for r in rs]}), p)
    (lake / "binance/liquidations/coin=BTC/date=1").mkdir(parents=True)   # BTC = 폴더만(코인 목록에 듦 · 봉 모자라 건너뜀)
    hot_bn = [(ts(hi_t - HR, H_LAST + 2 * HR), rnd.choice(["long", "short"]), 1.0, round(2500 * rnd.uniform(0.98, 1.02), 2), round(rnd.uniform(1e3, 3e5), 2), "ETHUSDT")
              for _ in range(600)]
    hot_bn += [(ts(hi_t, H_LAST), "long", 1.0, 2500.0, 9e9, "BTCUSDT")]
    c = sqlite3.connect(hot / "binance_ctx.sqlite")
    c.execute("CREATE TABLE liquidations(ts_ms INTEGER, side TEXT, qty REAL, price REAL, usd REAL, symbol TEXT)")
    c.execute("CREATE TABLE oi_1s(ts_ms INTEGER, symbol TEXT, open_interest REAL, PRIMARY KEY (ts_ms, symbol))")
    c.executemany("INSERT INTO liquidations VALUES (?,?,?,?,?,?)", hot_bn)
    oi = [(NOW0 - 50_000, "ethusdt", 4012345.5), (NOW0 - 40_000, "ethusdt", 4012400.25), (NOW0 - 30_000, "btcusdt", 91234.5)]   # 실제 규모(ETH ~1e10$) -- 봉우리 합 6자리에서 1ulp 가 보인다
    c.executemany("INSERT INTO oi_1s VALUES (?,?,?)", oi)
    c.commit(); c.close()
    c = sqlite3.connect(hot / "okx_ctx.sqlite")
    c.execute("CREATE TABLE okx_oi(ts_ms INTEGER, inst TEXT, oi_base REAL)")
    okx_oi = [(NOW0 - 1_000_000, "ETH-USDT-SWAP", 1.0), (NOW0 - 100_000, "ETH-USDT-SWAP", 333333.0)]
    c.executemany("INSERT INTO okx_oi VALUES (?,?,?)", okx_oi); c.commit(); c.close()
    c = sqlite3.connect(hot / "bybit_ctx.sqlite")
    c.execute("CREATE TABLE bybit_oi(ts_ms INTEGER, inst TEXT, oi_base REAL)")
    by_oi = [(NOW0 - 700_000, "ETHUSDT", 55555.0)]       # 10분 넘음 → 안 셈(2회는 더 오래됨)
    c.executemany("INSERT INTO bybit_oi VALUES (?,?,?)", by_oi); c.commit(); c.close()
    return {"bn": bn, "okx": okx, "hot_bn": hot_bn, "oi": oi, "okx_oi": okx_oi, "bybit_oi": by_oi}


def describe_parquet(p: Path) -> dict:
    t = pq.read_table(p)
    return {"cols": [[f.name, str(f.type)] for f in t.schema if not f.name.startswith("__")], "rows": [list(r.values()) for r in t.to_pylist()]}


def main() -> None:
    reqs: list = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for d in ("lake", "live", "hot"):
            (td / d).mkdir()
        L.LAKE, L.LIVE, L.HOT = td / "lake", td / "live", td / "hot"
        hot = lake_and_hot(L.LAKE, L.HOT)
        all_pre = json.dumps({"ok": True, "coins": {"sol": {"generated_ms": 1, "price": 150.0}}}, separators=(",", ":"))
        (L.LIVE / "liq_tab_all.json").write_text(all_pre)
        runs, gold = [], []
        for now in (NOW0, NOW0 + HR):
            reqs.clear()

            def urlopen(req, timeout=None, now=now):
                reqs.append([req.full_url, req.get_header("User-agent"), timeout])
                return io.BytesIO(respond(req.full_url, now))
            L.urllib.request.urlopen = urlopen
            L.time = types.SimpleNamespace(time=lambda now=now: now / 1000, sleep=lambda s: None)
            sys.argv = ["x"]
            L.main()
            g = {"reqs": list(reqs), "files": {}}
            for p in sorted(L.LIVE.iterdir()):
                g["files"][p.name] = describe_parquet(p) if p.suffix == ".parquet" else p.read_text()
            runs.append({"now_ms": now})
            gold.append(g)
    lake_rows = {"bn": hot["bn"], "okx": hot["okx"]}
    (FIX / "liq_tab_input.json").write_text(json.dumps({"runs": runs, "klines": K, "hot": hot, "all_pre": all_pre, "lake": lake_rows}, separators=(",", ":")))
    (FIX / "liq_tab_golden.json").write_text(json.dumps({"runs": gold}, separators=(",", ":"), ensure_ascii=False))
    for g in gold:
        print("요청", len(g["reqs"]), {k: (len(v) if isinstance(v, str) else len(v["rows"])) for k, v in g["files"].items()})


if __name__ == "__main__":
    os.environ.pop("LIQ_TAB_NO_REST", None)
    main()
