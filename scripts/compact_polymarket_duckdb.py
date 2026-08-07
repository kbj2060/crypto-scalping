#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "live" / "polymarket.duckdb"


def _backup_db(path: Path) -> Path:
    backup_dir = path.parent / "backups" / f"polymarket_raw_only_{datetime.now():%Y%m%d_%H%M%S}_kst"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / path.name
    shutil.copy2(path, backup_path)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only the raw Polymarket snapshot table in live DuckDB.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    db_path = args.db
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    backup_path = None if args.no_backup else _backup_db(db_path)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("DROP VIEW IF EXISTS polymarket_model_ready_v1")
        con.execute("DROP VIEW IF EXISTS polymarket_features_v1")
        con.execute("DROP TABLE IF EXISTS polymarket_features_10s")
        con.execute("DROP TABLE IF EXISTS polymarket_market_probs_10s")

        cols = {str(r[1]) for r in con.execute("PRAGMA table_info('polymarket_markets_10s_json')").fetchall()}
        if not cols:
            con.execute(
                """
                CREATE TABLE polymarket_markets_10s_json (
                    ts TIMESTAMP WITH TIME ZONE,
                    markets_json VARCHAR,
                    snapshot_json VARCHAR,
                    current_price DOUBLE,
                    schema_version INTEGER
                )
                """
            )
        else:
            if "snapshot_json" not in cols:
                con.execute("ALTER TABLE polymarket_markets_10s_json ADD COLUMN snapshot_json VARCHAR")
            if "current_price" not in cols:
                con.execute("ALTER TABLE polymarket_markets_10s_json ADD COLUMN current_price DOUBLE")
            if "schema_version" not in cols:
                con.execute("ALTER TABLE polymarket_markets_10s_json ADD COLUMN schema_version INTEGER")

        tables = [str(r[0]) for r in con.execute("SHOW TABLES").fetchall()]
        rows = con.execute("SELECT count(*) FROM polymarket_markets_10s_json").fetchone()[0]
    finally:
        con.close()

    if backup_path is not None:
        print(f"backup={backup_path}")
    print(f"tables={tables}")
    print(f"polymarket_markets_10s_json_rows={rows}")


if __name__ == "__main__":
    main()
