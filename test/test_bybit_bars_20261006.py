"""Bybit 5분봉 읽기(2026-10-06, 사용자 «바로 합산») -- 끊김이 걸친 봉은 «모름»으로 뺀다.
  python3 test/test_bybit_bars_20261006.py"""
from __future__ import annotations

import importlib.util
import pathlib
import sqlite3
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dash_srv_bybit", ROOT / "dashboard" / "server.py")
srv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = srv
spec.loader.exec_module(srv)

db = pathlib.Path(tempfile.mkdtemp()) / "bybit_tape.sqlite"
con = sqlite3.connect(db)
con.execute("""CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL,
               whale_buy_qty REAL, whale_sell_qty REAL, retail_buy_qty REAL, retail_sell_qty REAL)""")
con.execute("CREATE TABLE gaps(symbol TEXT, from_ms INTEGER, to_ms INTEGER, reason TEXT)")
t0 = 1_791_214_500
# 12봉 각각 한 초: 매수 10(고래 4·리테일 1) · 매도 3(고래 0·리테일 2)
con.executemany("INSERT INTO trade_tape_1s VALUES ('ETHUSDT', ?, 1, 10, 3, 4, 0, 1, 2)", [(t0 + 300 * i + 7,) for i in range(12)])
con.commit()
got = srv.bybit_bars("ETHUSDT", t0, t0 + 3600, db)
assert len(got) == 12 and got[t0] == (10.0, 3.0, 4.0, -1.0), got[t0]          # (매수, 매도, 고래 순, 리테일 순)
con.execute("INSERT INTO gaps VALUES ('ETHUSDT', ?, ?, 'ws_reconnect')", ((t0 + 1000) * 1000, (t0 + 1005) * 1000))
con.commit()
got = srv.bybit_bars("ETHUSDT", t0, t0 + 3600, db)
assert len(got) == 11 and t0 + 900 not in got, sorted(got)                      # 끊김이 걸친 봉(900~1200)만 «모름»
assert srv.bybit_bars("BTCUSDT", t0, t0 + 3600, db) == {}
assert srv.bybit_bars("ETHUSDT", t0, t0 + 3600, db.with_name("none.sqlite")) == {}
print("ok")
