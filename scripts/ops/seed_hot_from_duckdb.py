"""라이브 DuckDB 이력 -> hot SQLite 채우기 (저장 재설계 4단계 -- 스트림을 hot 으로 옮길 때마다 쓴다).

라이브 DuckDB 는 복사 스냅샷(seal_to_lake.snapshot = 복사 후 표마다 count 검증)으로만 연다. 대상 표는 수집기가 먼저
만들어 둔다(타입·PK). 열은 두 표에 다 있는 이름만. 이미 있는 행은 건너뛴다: --key 가 없으면 INSERT OR IGNORE(PK),
있으면 그 열들이 같은 행이 이미 있을 때 건너뛴다(PK 없는 gaps·verify_1m 등). 두 번 돌려도 같다.

  python scripts/ops/seed_hot_from_duckdb.py SRC.duckdb DST.sqlite TABLE [--src-table T] [--where "ts_sec >= 1790000000"] [--key symbol,ts_min]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.seal_to_lake import snapshot  # noqa: E402


def seed(src: Path, dst: Path, table: str, where: str = "true", key: list[str] | None = None,
         src_table: str | None = None) -> int:
    h = sqlite3.connect(dst, timeout=30)
    try:
        dst_cols = [r[1] for r in h.execute(f"PRAGMA table_info({table})")]
        if not dst_cols:
            raise SystemExit(f"{dst}:{table} 가 없다 -- 수집기가 표를 먼저 만들게 하라")
        with tempfile.TemporaryDirectory(dir=dst.parent) as td:
            snap = snapshot(src, Path(td))
            if snap is None:
                raise SystemExit(f"torn snapshot x3: {src}")
            d = duckdb.connect(str(snap), read_only=True)
            try:
                src_cols = [r[0] for r in d.execute(f'describe "{src_table or table}"').fetchall()]
                cols = [c for c in dst_cols if c in src_cols]
                rows = d.execute(f'SELECT {", ".join(cols)} FROM "{src_table or table}" WHERE {where}').fetchall()
            finally:
                d.close()
        rows = [tuple(str(v) if hasattr(v, "isoformat") else v for v in r) for r in rows]   # 시각은 텍스트로
        ph = ", ".join("?" * len(cols))
        if key:
            idx = [cols.index(k) for k in key]
            sql = (f"INSERT INTO {table}({', '.join(cols)}) SELECT {ph} WHERE NOT EXISTS "
                   f"(SELECT 1 FROM {table} WHERE {' AND '.join(f'{k} = ?' for k in key)})")
            rows = [r + tuple(r[i] for i in idx) for r in rows]
        else:
            sql = f"INSERT OR IGNORE INTO {table}({', '.join(cols)}) VALUES ({ph})"
        before = h.total_changes
        with h:
            h.executemany(sql, rows)
        return h.total_changes - before
    finally:
        h.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("table")
    ap.add_argument("--src-table")
    ap.add_argument("--where", default="true")
    ap.add_argument("--key", type=lambda s: s.split(","))
    a = ap.parse_args()
    print(f"[seed] {a.src.name}:{a.src_table or a.table} -> {a.dst.name}:{a.table} "
          f"{seed(a.src, a.dst, a.table, a.where, a.key, a.src_table)} rows", flush=True)
