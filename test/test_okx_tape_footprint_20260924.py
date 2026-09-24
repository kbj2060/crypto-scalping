"""OKX 테이프 → 풋프린트 봉 복원(A안, 2026-09-24) 규칙.   python -m pytest -q test/test_okx_tape_footprint_20260924.py"""
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.server import okx_tape_footprint  # noqa: E402
from scripts.live_trade_tape_collector_20260916 import TapeStore  # noqa: E402

INST = "ETH-USDT-SWAP"


def test_bars_minutes_and_oi(tmp_path):
    tape = TapeStore(tmp_path / "t.duckdb", INST, 0.1)
    now = int(time.time()) // 300 * 300
    b0, b1 = now - 600, now - 300                   # 두 봉(최근). t0 는 b1 안
    t0_ms = (b1 + 130) * 1000
    row = lambda sec, pbin, buy, sell: (sec, pbin, buy, sell, 1, 1, buy, sell,  # noqa: E731
                                        0.0, 0.0, buy, 0.0, 1, 0, 0, 0, 1, 1)
    tape.write([row(b0 + 1, 26442, 2.0, 0.0),        # 2644.2 -> 0.5칸 5288
                row(b0 + 2, 26443, 1.0, 3.0),        # 2644.3 -> 5289 (칸 경계를 넘는다)
                row(b1 + 10, 26450, 4.0, 0.0),       # t0 앞 -> 들어간다
                row(b1 + 130, 26450, 9.0, 0.0),      # t0 의 초 -> 라이브 몫이라 뺀다
                row(b1 + 200, 26450, 9.0, 0.0)])     # t0 뒤 -> 뺀다
    with tape._connect() as con:
        con.execute("INSERT INTO verify_1m VALUES (?, ?, 1, 1, 0.0, now())", [INST, b0])
        con.execute("INSERT INTO verify_1m VALUES (?, ?, 1, 1, -0.2, now())", [INST, b0 + 60])
    tape.record_gap((b1 + 60) * 1000, (b1 + 70) * 1000, "t")   # b1+60 분은 공백과 겹친다
    ctx = tmp_path / "c.duckdb"
    c = duckdb.connect(str(ctx))
    c.execute("CREATE TABLE okx_oi(inst VARCHAR, ts_ms BIGINT, oi_contracts DOUBLE, oi_base DOUBLE, oi_usd DOUBLE)")
    c.executemany("INSERT INTO okx_oi VALUES (?,?,0,?,0)",
                  [(INST, b0 * 1000 + 5, 100.0), (INST, b0 * 1000 + 200_000, 104.0), (INST, t0_ms + 1, 999.0)])
    c.execute("CREATE TABLE okx_liquidations(inst_id VARCHAR, inst_family VARCHAR, ts_ms BIGINT, side VARCHAR, "
              "pos_side VARCHAR, bk_px DOUBLE, sz_contracts DOUBLE, sz_base DOUBLE, bk_loss DOUBLE)")
    c.executemany("INSERT INTO okx_liquidations VALUES (?,'',?,?,?,?,0,?,0)",
                  [(INST, b0 * 1000 + 9, "sell", "long", 2600.0, 2.0),
                   ("BTC-USDT-SWAP", b0 * 1000 + 10, "buy", "short", 1.0, 5.0),   # 다른 종목
                   (INST, t0_ms + 5, "buy", "short", 2600.0, 1.0)])               # t0 뒤 -- 라이브 몫
    c.close()

    got = okx_tape_footprint(tape.db_path, ctx, INST, b0, t0_ms, 300, 0.5)
    assert got["bars"][b0] == {5288: [2.0, 0.0, 2.0, 0.0, 0.0, 0.0], 5289: [1.0, 3.0, 1.0, 0.0, 0.0, 0.0]}, got["bars"]
    assert got["bars"][b1] == {5290: [4.0, 0.0, 4.0, 0.0, 0.0, 0.0]}, "t0 의 초와 그 뒤는 라이브 몫"
    ok = got["ok_min"]
    assert b0 in ok, "1분봉과 맞은 분"
    assert b0 + 60 not in ok, "1분봉과 안 맞은 분"
    assert b1 in ok, "검사 전인 최근 분 · 공백 없음"
    assert b1 + 60 not in ok, "검사 전이지만 기록된 공백과 겹친다"
    assert got["oi"] == {b0: (100.0, 104.0)}, ("봉의 첫/끝 OI · t0 뒤는 뺀다", got["oi"])
    assert got["tape_max"] == b1 + 200
    assert got["liq"] == [{"ts_ms": b0 * 1000 + 9, "side": "long", "qty": 2.0, "price": 2600.0,
                           "usd": 5200.0, "symbol": INST}], ("ETH·t0 앞·라이브와 같은 모양", got["liq"])
