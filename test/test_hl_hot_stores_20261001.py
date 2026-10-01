"""HL 맥락·포지션 저장 (2026-10-01 저장 재설계 4c) -- 같은 코드가 DuckDB·SQLite(hot) 둘 다에 쓰고 읽힌다.
SQLite 에는 ADD COLUMN IF NOT EXISTS 가 없다(hl_universe.coin) · WAL·auto_vacuum · 인덱스."""
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import data_store as ds  # noqa: E402
import scripts.live_hyperliquid_book_ticker_collector_20260923 as bt  # noqa: E402
import scripts.live_hyperliquid_positions_collector_20260924 as pos  # noqa: E402


def test_ctx_and_positions_both_backends():
    d = Path(tempfile.mkdtemp())
    for suffix in (".duckdb", ".sqlite"):
        st = bt.CtxStore(d / f"ctx{suffix}")
        st.write([("ETH", 1000, 1e-5, 1.0, 0.0, 2500.0, 2500.1, 2500.05, 1.0, 1.0)])
        st.write([])
        assert not st.pending
        assert ds.read_rows(d / f"ctx{suffix}", "SELECT coin, recv_ms FROM hl_asset_ctx") == [("ETH", 1000)], suffix

        db = d / f"pos{suffix}"
        pos.init_db(db)
        pos.init_db(db)                                     # 두 번 불러도 된다(ALTER 는 이미 있음)
        pos.DB = db
        pos.write({"hl_cycles": [(2000, 3, 3, 1, 0.5)],
                   "hl_universe": [(2000, "0xabc", 1, 1e6, "ETH")],
                   "hl_positions": [(2000, 2000, "0xabc", "ETH", 1.0, 2500.0, 2000.0, 5.0, "cross", 1.0, 2500.0,
                                     0.0, 0.0, 10_000.0)]})
        got = ds.read_rows(db, "SELECT szi, liq_px FROM hl_positions WHERE coin = 'ETH' "
                               "AND ts_ms >= (SELECT max(ts_ms) FROM hl_cycles)")
        assert got == [(1.0, 2000.0)], (suffix, got)
        assert ds.read_rows(db, "SELECT coin FROM hl_universe") == [("ETH",)], suffix
    c = sqlite3.connect(d / "pos.sqlite")
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal" and c.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
    assert {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'index'")} >= {"hl_positions_i", "hl_cycles_i"}


if __name__ == "__main__":
    test_ctx_and_positions_both_backends()
    print("ok")
