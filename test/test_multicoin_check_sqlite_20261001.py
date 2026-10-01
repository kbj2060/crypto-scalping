"""multicoin_collectors check 가 hot(.sqlite) 신선도를 읽는다 (2026-10-01).
매니페스트의 hot 검사식은 감시기와 같은 SQLite 식인데 check 가 DuckDB 로 열어 `datetime` 함수가 없다며 죽었다."""
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ops.multicoin_collectors_20260926 import _duckdb_latest  # noqa: E402

TS = "datetime(max(ts_ms) / 1000, 'unixepoch', 'localtime')"


def test_sqlite_freshness():
    d = Path(tempfile.mkdtemp()) / "okx_ctx.sqlite"
    c = sqlite3.connect(d)
    c.execute("CREATE TABLE okx_mark(inst TEXT, ts_ms INTEGER)")
    c.execute("INSERT INTO okx_mark VALUES ('SOL-USDT-SWAP', ?)", [int(time.time() * 1000) - 60_000])
    c.commit()
    c.close()
    v = _duckdb_latest(d, "okx_mark WHERE inst = 'SOL-USDT-SWAP'", TS)
    assert isinstance(v, datetime) and timedelta(seconds=50) < datetime.now() - v < timedelta(seconds=75), v
    assert _duckdb_latest(d, "okx_mark WHERE inst = 'XRP-USDT-SWAP'", TS) is None


if __name__ == "__main__":
    test_sqlite_freshness()
    print("ok")
