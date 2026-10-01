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
from scripts.live_trade_tape_collector_20260916 import TapeStore  # noqa: E402
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
    assert s == {"written": 4, "rows": 4, "skipped_existing": 0, "resealed": 0, "failed": 1, "missing_source": 0}, s
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


def _row(sym, ts, q=1.0):
    return (ts, 1) + (q,) * 16                                               # TapeStore.write 행 = 심볼 뺀 18칸


def test_hot_source_reseal_and_guarded_prune():
    import sqlite3
    root = Path(tempfile.mkdtemp())
    hot = root / "data" / "hot" / "binance_tape.sqlite"
    lake = root / "lake"
    eth, xrp = TapeStore(hot, "ethusdt", 0.1), TapeStore(hot, "xrpusdt", 0.0001)
    eth.write([_row("ethusdt", D0 + 10), _row("ethusdt", D0 + 86400 + 10)])
    xrp.write([_row("xrpusdt", D0 + 20, 2.0)])
    spec = [("binance", "tape", [("data/hot/binance_tape.sqlite", "trade_tape_1s",
                                  "upper(regexp_replace(symbol, '(?i)usdt$', ''))")], "ts_sec * 1000")]
    now = dt.datetime(2026, 9, 30, 12, tzinfo=dt.timezone.utc)
    s1 = seal(spec, root=root, lake=lake, now=now)
    assert s1["written"] == 3 and s1["failed"] == 0, s1                       # ETH 09-28·09-29, XRP 09-28 -- 복사 없이 읽음
    assert sorted(ds.read("binance", "tape", "ETH", lake=lake)["ts_sec"]) == [D0 + 10, D0 + 86400 + 10]
    # 백필이 봉인된 날짜를 고쳐 쓴다(분 통째 교체) -> 지문이 달라져 그 날짜 하나만 다시 쓴다
    eth.replace_minute((D0 + 86400) // 60 * 60, [_row("ethusdt", D0 + 86400 + 10, 5.0)[0:]], 5.0, "test", "t")
    s2 = seal(spec, root=root, lake=lake, now=now)
    assert (s2["resealed"], s2["written"], s2["skipped_existing"]) == (1, 1, 2), s2
    assert ds.read("binance", "tape", "ETH", "2026-09-29", "2026-09-30", lake=lake)["buy_qty"].tolist() == [5.0]
    # 정리: 지울 행이 있는 (코인, 날짜)가 전부 lake 에 있어야 지운다
    later = now + dt.timedelta(days=17)                                      # cut = 10-01 00:00 -> 09-28·09-29 행이 대상
    c = sqlite3.connect(hot)
    gone = ds.lake_path("binance", "tape", "XRP", "2026-09-28", lake)
    gone.rename(gone.with_suffix(".bak"))                                    # 봉인이 빠진 상황
    args = (hot, "trade_tape_1s", "ts_sec", "symbol", "binance", "tape", lambda s: s.upper().removesuffix("USDT"), 16,
            (("verify_1m", "ts_min", 1), ("gaps", "to_ms", 1000)))
    assert prune_hot(*args, now=later, lake=lake) == 0, "XRP 09-28 lake 가 없는데 지웠다"
    assert c.execute("SELECT count(*) FROM trade_tape_1s").fetchone()[0] == 3
    gone.with_suffix(".bak").rename(gone)
    assert prune_hot(*args, now=later, lake=lake) == 3                       # 행 없는 날(XRP 09-29)은 파일이 없어도 된다
    assert c.execute("SELECT count(*) FROM trade_tape_1s").fetchone()[0] == 0
    c.close()


if __name__ == "__main__":
    test_hot_source_reseal_and_guarded_prune()
    test_seal_and_read()
    test_bt_format_matches_collector()
    print("ok")
