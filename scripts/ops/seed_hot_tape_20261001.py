"""1회용: 체결 테이프 DuckDB 이력 -> hot SQLite(data/hot/binance_tape.sqlite) 채우기 (2026-10-01 저장 재설계 3단계).

수집기의 HotMirror 는 켜진 뒤의 행만 쓴다. 대시보드는 7일(micro_ref)·14일(flow_hour_scales)을 읽으므로 지난 16일을
한 번 채운다. 라이브 DuckDB 는 안 연다(seal_to_lake.snapshot = 복사 후 검증). 미러가 이미 쓴 행은 기본키로 건너뛰고
(INSERT OR IGNORE -- 같은 초·칸이면 값도 같다), gaps·verify_1m 은 미러의 첫 행보다 앞선 것만 넣는다.
두 번 돌려도 같다.   python scripts/ops/seed_hot_tape_20261001.py [--days 16]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.live_trade_tape_collector_20260916 import HOT_TAPE_DB, HotMirror, _TAPE_COLS, default_db  # noqa: E402
from scripts.seal_to_lake import snapshot  # noqa: E402

SYMBOLS = ("ethusdt", "btcusdt", "solusdt", "xrpusdt", "hypeusdt")


def seed(days: int, hot: Path = HOT_TAPE_DB, dbs: dict[str, Path] | None = None) -> dict[str, int]:
    HotMirror(hot)                                   # 표·WAL 보장
    since = int(time.time()) // 86400 * 86400 - days * 86400
    out: dict[str, int] = {}
    with tempfile.TemporaryDirectory(dir=hot.parent) as td:
        for sym in SYMBOLS:
            src = (dbs or {}).get(sym, default_db(sym))
            if not src.exists():
                continue
            snap = snapshot(src, Path(td))
            if snap is None:
                raise RuntimeError(f"torn snapshot x3: {src}")
            d = duckdb.connect(str(snap), read_only=True)
            h = sqlite3.connect(hot, timeout=30)
            try:
                first_ver = h.execute("SELECT min(ts_min) FROM verify_1m WHERE symbol = ?", [sym]).fetchone()[0]
                first_gap = h.execute("SELECT min(from_ms) FROM gaps WHERE symbol = ?", [sym]).fetchone()[0]
                before = h.total_changes
                cur = d.execute(f"SELECT {', '.join(_TAPE_COLS)} FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ?",
                                [sym, since])
                while rows := cur.fetchmany(100_000):
                    with h:
                        h.executemany(HotMirror.INSERT.replace("OR REPLACE", "OR IGNORE"), rows)
                with h:
                    h.executemany("INSERT INTO verify_1m VALUES (?,?,?,?,?,?)",
                                  [(*r[:5], str(r[5])) for r in d.execute(
                                      "SELECT * FROM verify_1m WHERE symbol = ? AND ts_min >= ? AND ts_min < ?",
                                      [sym, since, first_ver or 1 << 62]).fetchall()])
                    h.executemany("INSERT INTO gaps VALUES (?,?,?,?)", d.execute(
                        "SELECT * FROM gaps WHERE symbol = ? AND to_ms >= ? AND from_ms < ?",
                        [sym, since * 1000, first_gap or 1 << 62]).fetchall())
                out[sym] = h.total_changes - before
            finally:
                d.close()
                h.close()
            snap.unlink()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=16)
    t0 = time.time()
    print(f"[seed] {seed(ap.parse_args().days)} {time.time() - t0:.0f}s", flush=True)
