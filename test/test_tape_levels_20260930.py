"""체결 테이프 → 가격 지도 레벨 (2026-09-30).   python -m pytest -q test/test_tape_levels_20260930.py

전일 가치영역은 **전일(UTC)만** · 앵커드 VWAP 은 전일 고가/저가를 **처음** 찍은 초부터 · 주간 VWAP 은 월요일 00:00 UTC부터 ·
마지막 가격은 가장 늦은 초 · 다른 심볼은 안 섞는다.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard.server as srv  # noqa: E402

BK = 0.1
ts = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())   # noqa: E731
px = lambda b: (b + 0.5) * BK   # noqa: E731


def test_tape_levels(tmp_path):
    rows = [  # (심볼, 초, 칸, 매수, 매도)
        ("ethusdt", ts("2026-09-28 01:00"), 30000, 1000, 0),   # 월요일 -- 주간 VWAP 에만
        ("ethusdt", ts("2026-09-29 03:00"), 21000, 5, 0),      # 전일 고가 처음
        ("ethusdt", ts("2026-09-29 05:00"), 21000, 1, 0),      # 전일 고가 두 번째(앵커 아님)
        ("ethusdt", ts("2026-09-29 10:00"), 19000, 0, 5),      # 전일 저가
        ("ethusdt", ts("2026-09-29 12:00"), 20000, 60, 40),    # 전일 POC
        ("ethusdt", ts("2026-09-30 11:59"), 20500, 500, 0),    # 오늘 -- 전일 프로파일에 안 들어간다 · 마지막 가격
        ("btcusdt", ts("2026-09-29 12:00"), 99999, 1e6, 0),    # 다른 심볼
    ]
    db = tmp_path / "tape.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE trade_tape_1s (symbol VARCHAR, ts_sec BIGINT, price_bin INTEGER, buy_qty DOUBLE, sell_qty DOUBLE)")
    con.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?)", rows)
    con.close()
    now = ts("2026-09-30 12:00")                                  # 수요일
    r = srv.tape_levels(db, "ethusdt", BK, now)
    assert r["available"] and abs(r["last"] - px(20500)) < 1e-9
    assert r["week_start"] == ts("2026-09-28 00:00")
    assert r["avwap_hi_ts"] == ts("2026-09-29 03:00") and r["avwap_lo_ts"] == ts("2026-09-29 10:00")
    wavg = lambda xs: px(sum(b * q for b, q in xs) / sum(q for _, q in xs))   # noqa: E731
    assert abs(r["avwap_hi"] - wavg([(21000, 6), (19000, 5), (20000, 100), (20500, 500)])) < 1e-9
    assert abs(r["avwap_lo"] - wavg([(19000, 5), (20000, 100), (20500, 500)])) < 1e-9
    assert abs(r["vwap_week"] - wavg([(30000, 1000), (21000, 6), (19000, 5), (20000, 100), (20500, 500)])) < 1e-9
    va = r["prev_day"]                                            # 전일 111 중 POC 칸 100 → 가치영역 = POC 칸 하나
    assert va["val"] <= px(20000) < va["vah"] and abs(va["vah"] - va["val"] - r["bin"]) < 1e-9, va
    assert abs(r["bin"] - 1.0) < 1e-12                                # 전일 마지막 가격 2000.05 × 0.05% → 테이프 칸(0.1) 격자 1.0 -- 지금 가격(2050)과 무관
    assert {"hvn", "lvn"} <= set(r)


def test_tape_levels_empty(tmp_path):
    db = tmp_path / "tape.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE trade_tape_1s (symbol VARCHAR, ts_sec BIGINT, price_bin INTEGER, buy_qty DOUBLE, sell_qty DOUBLE)")
    con.close()
    assert srv.tape_levels(db, "ethusdt", BK, ts("2026-09-30 12:00")) == {"available": False}
