"""풋프린트 머리 띠 국채선물 읽기(2026-10-06) -- 마지막 봉 기준 12시간 · 5분 경계 봉 + 마지막 봉 · 없으면 ok False.
  python3 test/test_treasury_payload_20261006.py"""
from __future__ import annotations

import importlib.util
import pathlib
import sqlite3
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dash_srv_treasury", ROOT / "dashboard" / "server.py")
srv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = srv
spec.loader.exec_module(srv)

db = pathlib.Path(tempfile.mkdtemp()) / "treasury_futures.sqlite"
con = sqlite3.connect(db)
con.execute("CREATE TABLE treasury_fut_1m(ts_sec INTEGER, symbol TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, contract TEXT)")
last = 1_791_000_000 // 300 * 300 + 120                      # 마지막 봉은 5분 경계가 아니다(그래도 들어와야 한다)
rows = [(t, "ZN", 100 + t % 7, "10Y") for t in range(last - 20 * 3600, last + 1, 60)]   # 20시간치 → 12시간만
rows += [(last - 3600, "ZT", None, "2Y")]                    # 종가 없는 분은 버린다
con.executemany("INSERT INTO treasury_fut_1m(ts_sec, symbol, close, contract) VALUES (?, ?, ?, ?)", rows)
con.commit(); con.close()

p = srv.treasury_payload(db)
zn = p["symbols"]["ZN"]["bars"]
assert p["ok"] and "ZT" not in p["symbols"], p["symbols"].keys()
assert zn[0][0] >= last - 12 * 3600 and zn[-1][0] == last, (zn[0], zn[-1])
assert all(t % 300 == 0 for t, _ in zn[:-1]) and len(zn) == 12 * 12 + 1, len(zn)
assert srv.treasury_payload(db.with_name("none.sqlite"))["ok"] is False
print("ok")
