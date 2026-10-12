"""체결 버블 API 원천(2026-10-13): 창 안 1초 행만 · 고래 합 · 빈 값 0.   python -m pytest -q test/test_bubble_tape_20261013.py"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.server import bubble_tape_payload  # noqa: E402


def test_window_and_whale_sum(tmp_path):
    db = tmp_path / "tape.sqlite"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INT, price_bin INT, buy_qty REAL, sell_qty REAL, "
              "whale_buy_qty REAL, whale_sell_qty REAL)")
    c.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?,?,?)", [
        ("ethusdt", 1000 - 200, 40000, 1, 0, 0, 0),       # 창(180+5초) 밖
        ("ethusdt", 1000 - 10, 40001, 2, 1, 1.5, None),   # 고래 None → 0
        ("btcusdt", 1000 - 5, 1, 9, 9, 0, 0),             # 다른 종목
    ])
    c.commit(); c.close()
    p = bubble_tape_payload("ethusdt", 180, db=db, now=1000)
    assert p["bucket"] == 0.1 and p["end"] == 1000 and p["lo"] == 1000 - 185
    assert p["rows"] == [[990, 40001, 2.0, 1.0, 1.5]]
