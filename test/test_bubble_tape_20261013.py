"""체결 버블 API 원천(2026-10-13): 창 안 1초 행만 · 고래 합 · 빈 값 0 · since 덧붙이기 · 바이낸스+OKX+Bybit 합산.
python -m pytest -q test/test_bubble_tape_20261013.py"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.server import bubble_tape_payload  # noqa: E402

DDL = ("CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INT, price_bin INT, buy_qty REAL, sell_qty REAL, "
       "whale_buy_qty REAL, whale_sell_qty REAL)")


def _db(path, rows):
    c = sqlite3.connect(path)
    c.execute(DDL)
    c.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?,?,?)", rows)
    c.commit(); c.close()
    return path


def test_window_and_whale_sum(tmp_path):
    db = _db(tmp_path / "tape.sqlite", [
        ("ethusdt", 1000 - 200, 40000, 1, 0, 0, 0),       # 창(180+5초) 밖
        ("ethusdt", 1000 - 10, 40001, 2, 1, 1.5, None),   # 고래 None → 0
        ("btcusdt", 1000 - 5, 1, 9, 9, 0, 0),             # 다른 종목
    ])
    none = tmp_path / "missing.sqlite"
    p = bubble_tape_payload("ethusdt", 180, db=db, now=1000, okx_db=none, bybit_db=none)
    assert p["bucket"] == 0.1 and p["end"] == 1000 and p["lo"] == 1000 - 185 and p["venues"] == ["binance"]
    assert p["rows"] == [[990, 40001, 2.0, 1.0, 1.5]]
    assert bubble_tape_payload("ethusdt", 180, db=db, now=1000, since=990, okx_db=none, bybit_db=none)["rows"] == []
    assert len(bubble_tape_payload("ethusdt", 180, db=db, now=1000, since=989, okx_db=none, bybit_db=none)["rows"]) == 1


def test_three_venue_sum(tmp_path):
    bn = _db(tmp_path / "bn.sqlite", [("ethusdt", 990, 40001, 2, 1, 0, 0)])
    okx = _db(tmp_path / "okx.sqlite", [("ETH-USDT-SWAP", 990, 40001, 0.5, 0, 0, 0),    # 같은 초·칸 → 더함
                                        ("ETH-USDT-SWAP", 991, 40002, 0, 3, 3, 0)])
    by = _db(tmp_path / "by.sqlite", [("ETHUSDT", 990, 40001, 0.25, 0.5, 0, 0), ("SOLUSDT", 990, 1, 9, 9, 0, 0)])
    p = bubble_tape_payload("ethusdt", 180, db=bn, now=1000, okx_db=okx, bybit_db=by)
    assert p["venues"] == ["binance", "okx", "bybit"]
    assert p["rows"] == [[990, 40001, 2.75, 1.5, 0.0], [991, 40002, 0.0, 3.0, 3.0]]
