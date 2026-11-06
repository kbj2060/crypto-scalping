#!/usr/bin/env python3
"""백업 복원 시험 (2026-11-07). 사본이 «있다»가 아니라 «열린다»를 본다. 월 1회 cron.

사본에서 종류별로 무작위 표본을 골라 실제로 연다: sqlite 표 count · duckdb 표 count · parquet 행 수 · gzip 전체 ·
crontab 비어 있지 않음. 표본이 0개인 종류도 실패다(사본이 비었는데 통과하면 안 된다).
결과 = data/live/ops_watchdog/backup_restore_check.json (감시기 check_backup 이 읽는다).

ponytail: 표본 시험이라 전수 보증은 아니다 · 서버 전체를 새로 세우는 훈련도 아니다. 그 훈련은 WSL 이전 때 롤백 절차로 한다.

  python scripts/ops/backup_restore_check_20261107.py [--root /mnt/data2/crypto-scalping-backups/data]
  python scripts/ops/backup_restore_check_20261107.py --selftest
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("/mnt/data2/crypto-scalping-backups/data")
OUT = ROOT / "data" / "live" / "ops_watchdog" / "backup_restore_check.json"
N = 3
MAX_DB_BYTES = 600_000_000   # 큰 파일 count 는 USB 를 몇 분 붙든다 -- 표본은 이 아래에서 고른다


def _sqlite(p: Path) -> int:
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=30)
    try:
        tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        if not tables:
            raise ValueError("no tables")
        return sum(con.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tables)
    finally:
        con.close()


def _duckdb(p: Path) -> int:
    con = duckdb.connect(str(p), read_only=True)
    try:
        return sum(con.execute(f'select count(*) from "{s}"."{t}"').fetchone()[0]
                   for s, t in con.execute("select schema_name, table_name from duckdb_tables()").fetchall())
    finally:
        con.close()


def _parquet(p: Path) -> int:
    return duckdb.sql(f"select count(*) from read_parquet('{p}', hive_partitioning=false)").fetchone()[0]


def _gzip(p: Path) -> int:
    n = 0
    with gzip.open(p, "rb") as fh:
        while chunk := fh.read(1 << 20):
            n += len(chunk)
    if not n:
        raise ValueError("empty gzip")
    return n


def _text(p: Path) -> int:
    n = len(p.read_text(errors="ignore").splitlines())
    if not n:
        raise ValueError("empty")
    return n


KINDS = (("sqlite", "hot_backup/*.sqlite", _sqlite),
         ("duckdb", "live/*.duckdb", _duckdb),
         ("parquet", "lake/**/part.parquet", _parquet),
         ("gzip", "live/orderflow/*/*/*.gz", _gzip),
         ("crontab", "live/backups/crontab_latest.txt", _text))


def run(root: Path, n: int = N, seed: int | None = None) -> dict:
    rng = random.Random(seed)
    res: dict = {"root": str(root), "finished_at": time.time(), "ok": True, "kinds": {}}
    for kind, pattern, opener in KINDS:
        files = [p for p in root.glob(pattern) if p.is_file() and p.stat().st_size <= MAX_DB_BYTES]
        sample = rng.sample(files, min(n, len(files)))
        rows, errors = [], []
        for p in sample:
            try:
                rows.append({"file": str(p.relative_to(root)), "count": int(opener(p))})
            except Exception as exc:  # noqa: BLE001 -- 어떤 오류든 «못 열었다»가 결과다
                errors.append({"file": str(p.relative_to(root)), "error": f"{type(exc).__name__}: {exc}"[:200]})
        ok = bool(sample) and not errors
        res["kinds"][kind] = {"candidates": len(files), "opened": rows, "errors": errors, "ok": ok}
        res["ok"] &= ok
    return res


def selftest() -> None:
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        (r / "hot_backup").mkdir()
        (r / "live/orderflow/depthdiff/ETHUSDT").mkdir(parents=True)
        (r / "live/backups").mkdir(parents=True)
        (r / "lake/x/coin=ETH/date=2026-01-01").mkdir(parents=True)
        con = sqlite3.connect(r / "hot_backup/a.sqlite")
        con.execute("create table t(a)"); con.execute("insert into t values (1),(2)"); con.commit(); con.close()
        c = duckdb.connect(str(r / "live/a.duckdb")); c.execute("create table t as select 1 a"); c.close()
        duckdb.sql(f"copy (select 1 a) to '{r}/lake/x/coin=ETH/date=2026-01-01/part.parquet' (format parquet)")
        gz = r / "live/orderflow/depthdiff/ETHUSDT/h.jsonl.gz"
        with gzip.open(gz, "wb") as fh:
            fh.write(b"x\n" * 10)
        (r / "live/backups/crontab_latest.txt").write_text("* * * * * true\n")
        good = run(r, seed=1)
        assert good["ok"], good
        assert good["kinds"]["sqlite"]["opened"][0]["count"] == 2
        # 음성 대조: 손상된 파일·빈 종류는 반드시 실패해야 한다
        gz.write_bytes(gz.read_bytes()[:12])
        bad = run(r, seed=1)
        assert not bad["ok"] and not bad["kinds"]["gzip"]["ok"], bad
        gz.unlink()
        empty = run(r, seed=1)
        assert not empty["ok"] and empty["kinds"]["gzip"]["candidates"] == 0, empty
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    res = run(a.root)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    tmp.replace(OUT)
    print(json.dumps({k: {"ok": v["ok"], "candidates": v["candidates"], "errors": v["errors"]}
                      for k, v in res["kinds"].items()}, ensure_ascii=False))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
