"""seed_hot_from_duckdb -- PK 표는 INSERT OR IGNORE, PK 없는 표는 --key 로 중복 건너뜀, 두 번 돌려도 같음."""
import sqlite3
import sys
import tempfile
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.live_trade_tape_collector_20260916 import TapeStore  # noqa: E402
from scripts.ops.seed_hot_from_duckdb import seed  # noqa: E402


def test_seed_idempotent_with_and_without_pk():
    d = Path(tempfile.mkdtemp())
    src, dst = d / "src.duckdb", d / "hot.sqlite"
    TapeStore(src, "ETH-USDT-SWAP", 0.1).write([(100, 1) + (1.0,) * 16, (101, 1) + (2.0,) * 16])
    hot = TapeStore(dst, "ETH-USDT-SWAP", 0.1)
    hot.write([(101, 1) + (9.0,) * 16])                                   # 이미 있는 초·칸 -> 원천으로 덮지 않는다
    c = duckdb.connect(str(src))
    c.execute("INSERT INTO verify_1m VALUES ('ETH-USDT-SWAP', 60, 1, 1, 0, now()), ('ETH-USDT-SWAP', 120, 1, 1, 0, now())")
    c.close()
    sqlite3.connect(dst).execute("INSERT INTO verify_1m VALUES ('ETH-USDT-SWAP', 60, 1, 1, 0, 'x')").connection.commit()
    assert seed(src, dst, "trade_tape_1s") == 1
    assert seed(src, dst, "verify_1m", key=["symbol", "ts_min"]) == 1
    assert seed(src, dst, "trade_tape_1s") == 0 and seed(src, dst, "verify_1m", key=["symbol", "ts_min"]) == 0
    h = sqlite3.connect(dst)
    assert h.execute("SELECT ts_sec, buy_qty FROM trade_tape_1s ORDER BY 1").fetchall() == [(100, 1.0), (101, 9.0)]
    assert [r[0] for r in h.execute("SELECT ts_min FROM verify_1m ORDER BY 1")] == [60, 120]
    assert seed(src, dst, "trade_tape_1s", where="ts_sec >= 200") == 0


if __name__ == "__main__":
    test_seed_idempotent_with_and_without_pk()
    print("ok")
