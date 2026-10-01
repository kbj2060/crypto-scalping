"""라이브 원천(DuckDB·jsonl) -> data/lake 일별 Parquet 봉인 (저장 재설계 2단계, 2026-10-01).

- 🔴라이브 파일을 직접 열지 않는다. 읽기 전용으로 열어도 쓰는 프로세스가 «Conflicting lock» 으로 죽는다
  (10-01 실측). 파일을 복사해 스냅샷을 뜨고, 그 사본을 read_only 로 열어 표마다 count 가 되면 쓴다(찢긴
  사본이면 최대 3번 다시 복사). 수집기는 쓸 때마다 열고 닫아 닫을 때 체크포인트하므로 본체만으로 충분하다.
- 끝난 UTC 날짜만 쓴다(자정 + LAG_HOURS 뒤). 파일이 있으면 건너뛰므로 여러 번 돌려도 같다 -- 첫 실행이 곧
  기존 이력 전체 내보내기다.
- 쓰기는 .part -> 행 수 대조 -> rename. 원천 행 수와 다르면 버리고 실패로 센다.
- hot SQLite(바이낸스 테이프, 3b)는 복사하지 않고 그대로 읽는다 -- WAL 이라 쓰는 쪽과 안 막힌다.
- 🔴봉인 뒤에도 원천이 그 날짜를 고쳐 쓴다(테이프 백필: 바이낸스 47시간·OKX 7일 전까지). 그래서 최근 RESEAL_DAYS 일은
  이미 있는 파일도 «행 수 + 행 해시 XOR» 지문을 원천과 대조해 다르면 다시 쓴다(.part -> rename, 한 날짜 파일 하나).

cron(서버, 매일 11:05 KST = 02:05 UTC):  python scripts/seal_to_lake.py
수동: python scripts/seal_to_lake.py --dry-run [--only binance/tape]
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_store import LAKE, ROOT, lake_path  # noqa: E402
from scripts.live_okx_trade_tape_collector_20260923 import OKX_HOT_CTX_DB, OKX_HOT_TAPE_DB  # noqa: E402
from scripts.live_trade_tape_collector_20260916 import HOT_TAPE_DB  # noqa: E402

LAG_HOURS = 2
HOT_KEEP_DAYS = 16     # hot 테이프: micro_ref 기준선 7일 · flow_hour_scales 14일(오늘 0시 전) + 여유. 맥락은 8일
RESEAL_DAYS = 8        # 테이프 백필이 고쳐 쓰는 기간(OKX 7일·바이낸스 47시간)보다 길게
HOT_REL = "data/hot/binance_tape.sqlite"
OKX = "split_part(symbol, '-', 1)"                                     # ETH-USDT-SWAP -> ETH
OKX_CTX_REL = "data/hot/okx_ctx.sqlite"
BN_CTX_REL = "data/hot/binance_ctx.sqlite"
_bn_coin, _okx_coin = (lambda s: s.upper().removesuffix("USDT")), (lambda s: s.split("-")[0])
_TAPE_SIDE = (("verify_1m", "ts_min", 1), ("gaps", "to_ms", 1000))
# hot 정리 대상: (파일, 표, 초 단위 시각 SQL, 코인 열, venue, stream, 코인 변환, 보존일, 딸린 표[(표, 시각 열, 배수)])
HOT_PRUNE = (
    (HOT_TAPE_DB, "trade_tape_1s", "ts_sec", "symbol", "binance", "tape", _bn_coin, HOT_KEEP_DAYS, _TAPE_SIDE),
    (OKX_HOT_TAPE_DB, "trade_tape_1s", "ts_sec", "symbol", "okx", "tape", _okx_coin, HOT_KEEP_DAYS, _TAPE_SIDE),
    *[(OKX_HOT_CTX_DB, f"okx_{t}", "ts_ms / 1000", "inst", "okx", t, _okx_coin, 8, ()) for t in ("oi", "mark", "funding")],
    (OKX_HOT_CTX_DB, "okx_liquidations", "ts_ms / 1000", "inst_id", "okx", "liquidations", lambda s: "ALL", 8,
     (("gaps", "to_ms", 1000),)),
    (ROOT / "data/hot/hl_ctx.sqlite", "hl_asset_ctx", "recv_ms / 1000", "coin", "hl", "asset_ctx", str, 8, ()),
    (ROOT / "data/hot/hl_positions.sqlite", "hl_positions", "ts_ms / 1000", "coin", "hl", "positions", str, 8,
     (("hl_cycles", "ts_ms", 1000), ("hl_universe", "ts_ms", 1000))),
    (ROOT / "data/hot/hl_positions.sqlite", "hl_liquidations", "detected_ms / 1000", "coin", "hl", "liquidations", str, 8, ()),
    (ROOT / "data/hot/binance_spot_tape.sqlite", "trade_tape_1s", "ts_sec", "symbol", "binance", "spot_tape", _bn_coin, 8,
     _TAPE_SIDE),
    *[(ROOT / BN_CTX_REL, t, "ts_ms / 1000", "symbol", "binance", st, _bn_coin, 8, ())
      for t, st in (("oi_1s", "oi_1s"), ("mark_price_1s", "mark_1s"), ("liquidations", "liquidations"))],
)
L, A = "data/live", "data/archive/live_retired_20261001"     # A = 1단계에서 보관한 09-19 정지 코인별 DB
SYM = "upper(regexp_replace(symbol, '(?i)usdt$', ''))"         # ethusdt / ETHUSDT -> ETH
DERIBIT = "split_part(split_part(instrument_name, '-', 1), '_', 1)"   # SOL_USDC-... -> SOL


def sfx(c: str) -> str:            # ETH 는 접미사 없는 옛 이름
    return "" if c == "eth" else f"_{c}"


def lit(c: str) -> str:
    return f"'{c.upper()}'"


# (venue, stream, [(원천 파일, 표 이름 | None=jsonl, coin SQL)], ts_ms SQL)
SPECS = [
    ("binance", "tape", [(HOT_REL, "trade_tape_1s", SYM)], "ts_sec * 1000"),     # 3b: 5코인 hot 한 파일(09-30 까지는 옛 DuckDB 에서 봉인됨)
    ("binance", "spot_tape", [("data/hot/binance_spot_tape.sqlite", "trade_tape_1s", SYM)], "ts_sec * 1000"),   # 4d hot(10-01 까지는 대시보드 DuckDB 에서)
    # 4d(2026-10-01): 바이낸스 맥락 수집기 hot 한 파일(09-30 까지는 옛 oi_1s/mark_price_1s.duckdb · liq_events*.jsonl 에서 봉인됨)
    ("binance", "oi_1s", [(BN_CTX_REL, "oi_1s", SYM)], "ts_ms"),
    ("binance", "mark_1s", [(BN_CTX_REL, "mark_price_1s", SYM)], "ts_ms"),
    ("binance", "liquidations", [(BN_CTX_REL, "liquidations", SYM)], "ts_ms"),
    ("binance", "oi_lsratio_5m", [(f"{L}/oi_lsratio.duckdb", f"oi_lsratio_5m{sfx(c)}", lit(c)) for c in ("eth", "btc", "sol")]
     + [(f"{A}/oi_lsratio_{c}.duckdb", f"oi_lsratio_5m_{c}", lit(c)) for c in ("xrp", "hype")], "epoch_ms(ts)"),
    ("binance", "micro_1m", [(f"{L}/microstructure.duckdb", f"microstructure_1m{sfx(c)}", lit(c)) for c in ("eth", "btc", "sol")]
     + [(f"{A}/microstructure_{c}.duckdb", f"microstructure_1m_{c}", lit(c)) for c in ("xrp", "hype")], "epoch_ms(ts)"),
    ("binance", "tail_risk_1m", [(f"{L}/tail_risk.duckdb", "tail_risk_1m", lit("eth"))]
     + [(f"{A}/tail_risk_btc_sol.duckdb", f"tail_risk_1m_{c}", lit(c)) for c in ("btc", "sol")]
     + [(f"{A}/tail_risk_{c}.duckdb", f"tail_risk_1m_{c}", lit(c)) for c in ("xrp", "hype")], "epoch_ms(ts)"),
    ("okx", "tape", [("data/hot/okx_tape.sqlite", "trade_tape_1s", OKX)], "ts_sec * 1000"),   # 4a hot(10-01 까지는 옛 DuckDB 에서)
    # 4b(2026-10-01): OKX 맥락 = hot 한 파일(종목 = inst). 청산은 전 종목(~300)이라 날짜당 파일 하나(coin=ALL)
    *[("okx", s, [(OKX_CTX_REL, f"okx_{s}", "split_part(inst, '-', 1)")], "ts_ms") for s in ("oi", "mark", "funding")],
    ("okx", "liquidations", [(OKX_CTX_REL, "okx_liquidations", "'ALL'")], "ts_ms"),
    ("hl", "asset_ctx", [("data/hot/hl_ctx.sqlite", "hl_asset_ctx", "coin")], "recv_ms"),                    # 4c hot
    ("hl", "positions", [("data/hot/hl_positions.sqlite", "hl_positions", "coin"),
                         (f"{L}/hyperliquid_positions_btc_sol_xrp_hype.from_pi.duckdb", "hl_positions", "coin")], "ts_ms"),
    ("hl", "liquidations", [("data/hot/hl_positions.sqlite", "hl_liquidations", "coin"),
                            (f"{L}/hyperliquid_positions_btc_sol_xrp_hype.from_pi.duckdb", "hl_liquidations", "coin")],
     "detected_ms"),
    ("deribit", "option_trades", [(f"{L}/deribit_options.duckdb", "option_trades", DERIBIT)], "ts_ms"),
    ("deribit", "chain", [(f"{L}/deribit_options.duckdb", "option_chain_snapshot", "currency")], "epoch_ms(recorded_at_utc)"),
    ("deribit", "gex_summary", [(f"{L}/deribit_options.duckdb", "gex_summary", "currency")], "epoch_ms(recorded_at_utc)"),
    ("deribit", "option_summary", [(f"{L}/deribit_options.duckdb", "option_summary", "currency")], "epoch_ms(recorded_at_utc)"),
    ("deribit", "block_rfq", [(f"{L}/deribit_options.duckdb", "block_rfq_trades", "'ALL'")], "ts_ms"),
    ("deribit", "block_future_legs", [(f"{L}/deribit_options.duckdb", "block_future_legs", DERIBIT)], "ts_ms"),
]


def snapshot(src: Path, tmp: Path) -> Path | None:
    """라이브 파일 사본. duckdb 는 열어 표마다 count 가 되는지 확인한다(찢긴 사본이면 다시). .sqlite 는 원본 그대로(WAL)."""
    if src.suffix == ".sqlite":
        return src
    dst = tmp / f"{len(list(tmp.iterdir()))}_{src.name}"
    for _ in range(3):
        shutil.copyfile(src, dst)
        if src.suffix != ".duckdb":
            return dst
        try:
            con = duckdb.connect(str(dst), read_only=True)
            for (t,) in con.execute("select table_name from duckdb_tables()").fetchall():
                con.execute(f'select count(*) from "{t}"').fetchone()
            con.close()
            return dst
        except duckdb.Error:
            time.sleep(2)
    dst.unlink(missing_ok=True)
    return None


def seal(specs=SPECS, root: Path = ROOT, lake: Path = LAKE, now: dt.datetime | None = None,
         dry_run: bool = False, only: str | None = None) -> dict[str, int]:
    now = now or dt.datetime.now(dt.timezone.utc)
    last_day = (now - dt.timedelta(hours=LAG_HOURS)).date() - dt.timedelta(days=1)   # 이 날짜까지 끝났다
    stats = {"written": 0, "rows": 0, "skipped_existing": 0, "resealed": 0, "failed": 0, "missing_source": 0}
    lake.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".snap_", dir=lake))
    snaps: dict[Path, str | None] = {}           # 원천 -> attach 별칭 (jsonl 은 사본 경로)
    try:
        con = duckdb.connect()
        con.execute("set threads = 2; set memory_limit = '1GB'")
        for venue, stream, sources, ts_sql in specs:
            if only and only != f"{venue}/{stream}":
                continue
            parts = []
            for rel, table, coin_sql in sources:
                src = root / rel
                if not src.exists():
                    stats["missing_source"] += 1
                    continue
                if src not in snaps:
                    snap = snapshot(src, tmp)
                    if snap is not None and table is not None:
                        alias = f"s{len(snaps)}"
                        kind = ", TYPE sqlite" if snap.suffix == ".sqlite" else ""
                        con.execute(f"attach '{snap.as_posix()}' as {alias} (read_only{kind})")
                        snaps[src] = alias
                    else:
                        snaps[src] = None if snap is None else snap.as_posix()
                handle = snaps[src]
                if handle is None:
                    print(f"[seal] 🔴 torn snapshot x3, skip: {rel}", flush=True)
                    stats["failed"] += 1
                    continue
                if table is None:
                    rel_sql = f"read_json_auto('{handle}', format = 'newline_delimited', ignore_errors = true)"
                else:
                    has = con.execute("select count(*) from duckdb_tables() where database_name = ? and table_name = ?",
                                      [handle, table]).fetchone()[0]
                    if not has:
                        stats["missing_source"] += 1
                        continue
                    rel_sql = f'{handle}."{table}"'
                parts.append(f"select *, ({coin_sql}) as __coin, "
                             f"cast(timezone('UTC', to_timestamp(({ts_sql}) / 1000.0)) as date) as __day from {rel_sql}")
            if not parts:
                continue
            con.execute(f"create or replace temp view v as {' union all by name '.join(parts)}")
            # 원천 지문(행 수 + 행 해시 XOR)을 날짜 고르는 쿼리에서 한 번에 -- 날짜마다 원천을 다시 훑으면 hot(SQLite)은
            #   조건을 못 내려 매번 전체를 읽는다(10-01 서버: 바이낸스 테이프 31날짜 135초).
            cols = [r[0] for r in con.execute("describe select * exclude (__coin, __day) from v").fetchall()]
            row = "hash(row(" + ", ".join(f'"{c}"' for c in cols) + "))"
            todo = con.execute(f"select __coin, __day, count(*), bit_xor({row}) from v "
                               "where __coin is not null and __day <= ? group by all order by all", [last_day]).fetchall()
            for coin, day, n, src_hash in todo:
                out = lake_path(venue, stream, coin, day.isoformat(), lake)
                if out.exists():
                    if day <= last_day - dt.timedelta(days=RESEAL_DAYS) or (n, src_hash) == con.execute(
                            f"select count(*), bit_xor({row}) from read_parquet('{out.as_posix()}', hive_partitioning = false)"
                            ).fetchone():   # 🔴hive_partitioning=false -- 경로의 coin=/date= 가 열로 붙으면 해시가 달라진다
                        stats["skipped_existing"] += 1
                        continue
                    stats["resealed"] += 1           # 원천이 그 날짜를 고쳐 썼다(백필) -- 아래에서 통째로 다시 쓴다
                if dry_run:
                    stats["written"] += 1
                    stats["rows"] += n
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                part = out.with_suffix(".parquet.part")
                con.execute(f"copy (select * exclude (__coin, __day) from v where __coin = ? and __day = ?) "
                            f"to '{part.as_posix()}' (format parquet, compression zstd)", [coin, day])
                got = con.execute(f"select count(*) from read_parquet('{part.as_posix()}')").fetchone()[0]
                if got != n:
                    part.unlink()
                    print(f"[seal] 🔴 row mismatch {venue}/{stream} {coin} {day}: {got} != {n}", flush=True)
                    stats["failed"] += 1
                    continue
                part.rename(out)
                stats["written"] += 1
                stats["rows"] += n
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return stats


def prune_hot(hot: Path, table: str, ts_sql: str, coin_col: str, venue: str, stream: str, coin_of,
              keep_days: int, side=(), now: dt.datetime | None = None, lake: Path = LAKE) -> int:
    """hot(SQLite) 한 표에서 keep_days 넘은 행을 지우고 공간을 돌려준다(auto_vacuum=INCREMENTAL).
    🔴3b 부터 hot 이 원본이다 -- 지울 행이 있는 (코인, 날짜)가 **전부** lake 에 봉인돼 있을 때만 지운다(아니면 0, 로그).
    side = 같은 파일의 딸린 표(검증·공백 기록 등) -- 같은 경계로 지운다."""
    if not hot.exists():
        return 0
    now = now or dt.datetime.now(dt.timezone.utc)
    cut = int(now.timestamp()) // 86400 * 86400 - keep_days * 86400
    con = sqlite3.connect(f"file:{hot}?mode=ro", uri=True, timeout=30)
    try:
        held = con.execute(f"SELECT DISTINCT {coin_col}, CAST({ts_sql} AS INTEGER) / 86400 FROM {table} "
                           f"WHERE {ts_sql} < ?", [cut]).fetchall()
    finally:
        con.close()
    day = lambda d: dt.datetime.fromtimestamp(d * 86400, dt.timezone.utc).date().isoformat()   # noqa: E731
    unsealed = sorted({(coin_of(c), day(d)) for c, d in held
                       if not lake_path(venue, stream, coin_of(c), day(d), lake).exists()})
    if unsealed:
        print(f"[seal] 🔴 hot 정리 건너뜀 {venue}/{stream}: lake 에 없는 날짜 {unsealed[:5]}", flush=True)
        return 0
    con = sqlite3.connect(hot, timeout=30)
    try:
        with con:
            n = con.execute(f"DELETE FROM {table} WHERE {ts_sql} < ?", [cut]).rowcount
            for t, col, mult in side:
                con.execute(f"DELETE FROM {t} WHERE {col} < ?", [cut * mult])
        con.execute("PRAGMA incremental_vacuum")
    finally:
        con.close()
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="venue/stream 하나만 (예: binance/tape)")
    a = ap.parse_args()
    t0 = time.time()
    s = seal(dry_run=a.dry_run, only=a.only)
    if not a.dry_run and not a.only and not s["failed"]:
        s["hot_pruned"] = {f"{v}/{st}": prune_hot(h, t, ts, cc, v, st, fn, k, side)
                           for h, t, ts, cc, v, st, fn, k, side in HOT_PRUNE}
    print(f"[seal] {'DRY-RUN ' if a.dry_run else ''}{dt.datetime.now().isoformat(timespec='seconds')} "
          f"lake={LAKE} {s} {time.time() - t0:.0f}s", flush=True)
    sys.exit(1 if s["failed"] else 0)
