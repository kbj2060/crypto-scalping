"""DuckDB 파일 덤프(Rust src/duckdump.rs 와 같은 꼴) -- 표마다 열·형 · (ts 열이 있으면) 물리 순서 ts · 행(ts 순 또는 물리 순서 · 실수 = repr · 시각 = epoch µs)."""
from pathlib import Path

import duckdb


def dump(db: Path) -> dict:
    con = duckdb.connect(str(db), read_only=True)
    out = {}
    for (t,) in con.execute("SELECT table_name FROM duckdb_tables() ORDER BY table_name").fetchall():
        cols = con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position", [t]).fetchall()
        has_ts = any(c == "ts" for c, _ in cols)
        sel = ", ".join(f"epoch_us({c})" if ty.startswith("TIMESTAMP") else c for c, ty in cols)
        rows = con.execute(f"SELECT {sel} FROM {t}" + (" ORDER BY ts" if has_ts else "")).fetchall()
        phys = [r[0] for r in con.execute(f"SELECT epoch_us(ts) FROM {t}").fetchall()] if has_ts else []
        out[t] = {"cols": [list(c) for c in cols], "phys_ts": phys,
                  "rows": [[repr(v) if isinstance(v, float) else v for v in r] for r in rows]}
    con.close()
    return out
