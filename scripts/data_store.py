"""데이터 경로와 읽기를 한 곳에 (저장 재설계 2단계, 2026-10-01).

lake = 봉인된 이력. 일별 Parquet, 한 번 쓰면 안 바뀐다(scripts/seal_to_lake.py 가 쓴다):
    data/lake/{venue}/{stream}/coin={COIN}/date={YYYY-MM-DD}/part.parquet
dev 에는 서버 lake 가 없고 매일 04시 백업 사본(~/backups/crypto-scalping-server/data/lake)이 있다 -- 저장소에
data/lake 가 없으면 그쪽을 읽는다. DATA_LAKE 환경변수가 둘 다 이긴다.

호가 원시 시각 파일(bookticker .bt · depthdiff .jsonl · raster .f32 …)은 아직 data/live/orderflow 에 그대로 있다.

    from scripts.data_store import read
    df = read("binance", "tape", "ETH", "2026-09-20", "2026-09-27")
"""
from __future__ import annotations

import gzip
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
ORDERFLOW = LIVE / "orderflow"
_BACKUP_LAKE = Path.home() / "backups" / "crypto-scalping-server" / "data" / "lake"
LAKE = Path(os.getenv("DATA_LAKE") or (_BACKUP_LAKE if not (ROOT / "data" / "lake").exists() and _BACKUP_LAKE.exists()
                                        else ROOT / "data" / "lake"))

# scripts/live_book_ticker_collector_20260914.py 의 HDR(32B)·ROW("<qdfdf", 32B)와 같아야 한다 -- 테스트가 대조한다.
BT_HEADER_BYTES = 32
BT_DTYPE = np.dtype([("ts_ms", "<i8"), ("bid_px", "<f8"), ("bid_qty", "<f4"), ("ask_px", "<f8"), ("ask_qty", "<f4")])


def sqlite_init(path: Path, indexes: tuple[str, ...] = ()) -> None:
    """hot SQLite 새 파일 설정 + 인덱스. auto_vacuum 은 새 파일에서만, WAL 전환보다 먼저(뒤면 조용히 0)."""
    import sqlite3
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    try:
        con.execute("PRAGMA auto_vacuum = INCREMENTAL")
        con.execute("PRAGMA journal_mode = WAL")
        for ddl in indexes:                       # 표는 호출 쪽이 먼저 만든다
            con.execute(ddl)
        con.commit()
    finally:
        con.close()


@contextmanager
def rw_connect(path: Path):
    """수집기 쓰기 연결 하나 -- .sqlite(hot) = 자동커밋(isolation_level=None)이라 트랜잭션은 BEGIN/COMMIT 문장으로
    두 엔진 똑같이 연다. .duckdb = 읽는 쪽과 잠금 충돌이면 잠깐 기다렸다 연다(duckdb_connect_retry)."""
    if path.suffix == ".sqlite":
        import sqlite3
        con = sqlite3.connect(path, timeout=30, isolation_level=None)
    else:
        from scripts.live_trade_tape_collector_20260916 import duckdb_connect_retry
        con = duckdb_connect_retry(path)
    try:
        yield con
    finally:
        con.close()


def read_rows(path: Path, sql: str, params: list | tuple = ()) -> list[tuple]:
    """라이브 저장소 읽기 한 곳. .sqlite(hot, 저장 재설계 3단계) = WAL 이라 쓰는 쪽과 서로 안 막는다 -- 그냥 연다.
    .duckdb = read_only 로 열고 쓰는 쪽 락과 겹치면 0.2초씩 최대 25번 다시(읽는 쪽이 쓰는 쪽을 막을 수도 있다).
    🔴SQL 은 두 엔진에서 같은 결과가 나오게 쓴다: `a // b` 대신 `a - a % b`, arg_max 대신 ORDER BY … LIMIT 1,
    quantile_cont 는 파이썬에서. (DuckDB 의 sqlite ATTACH 는 조건을 못 내려보내 670만 행을 매번 다 읽는다 --
    쿼리당 0.8~1.7초 vs SQLite 인덱스로 수 ms~수백 ms, 10-01 서버 실측.)"""
    if path.suffix == ".sqlite":
        import sqlite3
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()
    import duckdb
    for i in range(25):
        try:
            con = duckdb.connect(str(path), read_only=True)
            try:
                return con.execute(sql, params).fetchall()
            finally:
                con.close()
        except duckdb.IOException as exc:
            if "lock" not in str(exc).lower() or i == 24:
                raise
            time.sleep(0.2)
    return []


def lake_path(venue: str, stream: str, coin: str, day: str, lake: Path | None = None) -> Path:
    return (lake or LAKE) / venue / stream / f"coin={coin}" / f"date={day}" / "part.parquet"


def read(venue: str, stream: str, coin: str = "*", start: str | None = None, end: str | None = None,
         columns: str = "*", con=None, lake: Path | None = None):
    """lake 의 [start, end) UTC 날짜('YYYY-MM-DD') 구간을 DataFrame 으로. coin·date 컬럼이 붙는다."""
    import duckdb
    con = con or duckdb.connect()
    pat = lake_path(venue, stream, coin, "*", lake).as_posix()
    where = [f"date >= DATE '{start}'"] if start else []
    where += [f"date < DATE '{end}'"] if end else []
    q = f"select {columns} from read_parquet('{pat}', hive_partitioning = true, union_by_name = true)"
    return con.sql(q + (" where " + " and ".join(where) if where else "")).df()


def orderflow_files(stream: str, symbol: str, start: str | None = None, end: str | None = None) -> list[Path]:
    """호가 원시 시각 파일. stream = bookticker·depthdiff·raster·okx_bookticker·hyperliquid_bookticker,
    symbol = 디렉터리 이름(ETHUSDT · ETH-USDT-SWAP · ETH). start/end = 'YYYY-MM-DDTHH' 접두 비교(UTC)."""
    d = ORDERFLOW / stream / symbol
    return sorted(p for p in d.iterdir()
                  if p.is_file() and (start is None or p.name[:13] >= start) and (end is None or p.name[:13] < end))


def read_bt(path: Path) -> np.ndarray:
    """BTKR bookTicker 시각 파일(.bt 또는 .bt.gz) -> ts_ms·bid_px·bid_qty·ask_px·ask_qty 배열.
    쓰는 중인 파일의 잘린 마지막 행은 버린다."""
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as fh:
        raw = fh.read()
    if raw[:4] != b"BTKR":
        raise ValueError(f"not a BTKR file: {path}")
    body = raw[BT_HEADER_BYTES:]
    return np.frombuffer(body[: len(body) - len(body) % BT_DTYPE.itemsize], dtype=BT_DTYPE)
