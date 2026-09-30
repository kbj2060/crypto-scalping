"""seal_to_lake + data_store -- 합성 원천으로: 끝난 UTC 날짜만 · 코인 파티션 · 멱등 · 찢긴 사본 거부 · jsonl · .bt 형식."""
import datetime as dt
import gzip
import json
import sys
import tempfile
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts import data_store as ds  # noqa: E402
from scripts.live_trade_tape_collector_20260916 import HotMirror  # noqa: E402
from scripts.seal_to_lake import prune_hot, seal  # noqa: E402

D0 = int(dt.datetime(2026, 9, 28, tzinfo=dt.timezone.utc).timestamp())   # 09-28 00:00 UTC


def test_seal_and_read():
    root = Path(tempfile.mkdtemp())
    lake = root / "lake"
    (root / "live").mkdir()
    con = duckdb.connect(str(root / "live" / "tape.duckdb"))
    con.execute("create table trade_tape_1s (symbol varchar, ts_sec bigint, q double)")
    # 09-28 23:59:59 · 09-29 00:00:00 ×2코인 · 09-30 12:00 (지금 = 안 끝난 날)
    con.executemany("insert into trade_tape_1s values (?, ?, ?)",
                    [("ethusdt", D0 + 86399, 1.0), ("btcusdt", D0 + 86400, 2.0), ("ethusdt", D0 + 86400, 3.0),
                     ("ethusdt", D0 + 2 * 86400 + 43200, 4.0)])
    con.close()
    (root / "live" / "liq.jsonl").write_text(
        json.dumps({"ts_ms": (D0 + 10) * 1000, "usd": 5.0, "symbol": "SOLUSDT"}) + "\n" + '{"ts_ms": 12')   # 잘린 끝줄
    (root / "live" / "torn.duckdb").write_bytes(b"not a database")
    sym = "upper(regexp_replace(symbol, '(?i)usdt$', ''))"
    specs = [("x", "tape", [("live/tape.duckdb", "trade_tape_1s", sym)], "ts_sec * 1000"),
             ("x", "liq", [("live/liq.jsonl", None, sym)], "ts_ms"),
             ("x", "torn", [("live/torn.duckdb", "t", "'ETH'")], "ts_ms")]
    now = dt.datetime(2026, 9, 30, 12, tzinfo=dt.timezone.utc)
    s = seal(specs, root=root, lake=lake, now=now)
    assert s == {"written": 4, "rows": 4, "skipped_existing": 0, "failed": 1, "missing_source": 0}, s
    again = seal(specs[:2], root=root, lake=lake, now=now)
    assert again["written"] == 0 and again["skipped_existing"] == 4, again
    eth = ds.read("x", "tape", "ETH", lake=lake)
    assert sorted(eth["q"]) == [1.0, 3.0] and "coin" in eth and "__day" not in eth      # 오늘 행(4.0)은 아직 안 봉인
    assert sorted(ds.read("x", "tape", "*", "2026-09-29", "2026-09-30", lake=lake)["q"]) == [2.0, 3.0]
    assert ds.read("x", "liq", "SOL", lake=lake)["usd"].tolist() == [5.0]
    assert not list(lake.glob(".snap_*")) and not list(lake.rglob("*.part"))


def test_bt_format_matches_collector():
    import live_book_ticker_collector_20260914 as bt
    assert bt.ROW.size == ds.BT_DTYPE.itemsize and bt.HDR.size == ds.BT_HEADER_BYTES
    p = Path(tempfile.mkdtemp()) / "2026-09-30T00.bt.gz"
    rows = [(1, 100.5, 2.0, 100.6, 3.0), (2, 101.0, 1.5, 101.1, 0.5)]
    with gzip.open(p, "wb") as fh:
        fh.write(bt.HDR.pack(bt.MAGIC, bt.VER, bt.ROW.size, 0, 0, 0, 0))
        fh.write(b"".join(bt.ROW.pack(*r) for r in rows) + b"\x00" * 7)                  # 잘린 끝 행
    a = ds.read_bt(p)
    assert a["ts_ms"].tolist() == [1, 2] and a["ask_px"].tolist() == [100.6, 101.1] and a["bid_qty"].tolist() == [2.0, 1.5]


def test_prune_hot():
    import sqlite3
    hot = Path(tempfile.mkdtemp()) / "h.sqlite"
    m = HotMirror(hot)
    now = dt.datetime(2026, 10, 1, 5, tzinfo=dt.timezone.utc)
    cut = int(now.timestamp()) // 86400 * 86400 - 16 * 86400
    row = lambda ts: ("ethusdt", ts, 1) + (1.0,) * 16                                    # noqa: E731
    m.run([(HotMirror.INSERT, [row(cut - 1), row(cut), row(cut + 3600)] + [row(cut - 10 - i) for i in range(3000)], True),
           ("INSERT INTO verify_1m VALUES (?,?,?,?,?,?)", ("ethusdt", cut - 60, 1, 1, 0, "x"), False),
           ("INSERT INTO gaps VALUES (?,?,?,?)", ("ethusdt", (cut - 9) * 1000, (cut - 5) * 1000, "t"), False)])
    size0 = hot.stat().st_size
    assert prune_hot(hot, now) == 3001                                                     # 경계 초(cut)는 남는다
    c = sqlite3.connect(hot)
    assert [r[0] for r in c.execute("SELECT ts_sec FROM trade_tape_1s ORDER BY 1")] == [cut, cut + 3600]
    assert c.execute("SELECT count(*) FROM verify_1m").fetchone()[0] == 0 == c.execute("SELECT count(*) FROM gaps").fetchone()[0]
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert hot.stat().st_size < size0, "공간을 안 돌려줬다(auto_vacuum)"


if __name__ == "__main__":
    test_prune_hot()
    test_seal_and_read()
    test_bt_format_matches_collector()
    print("ok")
