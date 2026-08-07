#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_DIR = ROOT / "data" / "live"


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return any(str(r[0]) == table for r in con.execute("SHOW TABLES").fetchall())


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {str(r[1]) for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}


def _print_query(con: duckdb.DuckDBPyConnection, title: str, sql: str) -> None:
    print(f"\n[{title}]")
    try:
        print(con.execute(sql).fetchdf().to_string(index=False))
    except Exception as exc:
        print(f"ERROR: {exc}")


def audit_micro(path: Path, hours: int) -> None:
    print(f"\n=== microstructure: {path} ===")
    if not path.exists():
        print("missing")
        return
    try:
        con = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        print(f"locked/unavailable: {exc}")
        return
    if not _table_exists(con, "microstructure_1m"):
        print("missing table microstructure_1m")
        return
    _print_query(
        con,
        "freshness",
        """
        SELECT count(*) AS rows, min(ts) AS min_ts, max(ts) AS max_ts,
               date_diff('second', max(ts), now()) AS max_age_sec
        FROM microstructure_1m
        """,
    )
    cols = _columns(con, "microstructure_1m")
    if {"data_stale", "valid_taker_flow", "valid_nif", "recent_trade_count_5m"}.issubset(cols):
        _print_query(
            con,
            f"quality_last_{hours}h",
            f"""
            SELECT
                count(*) AS rows,
                sum(CASE WHEN data_stale THEN 1 ELSE 0 END) AS stale_rows,
                sum(CASE WHEN valid_taker_flow THEN 1 ELSE 0 END) AS valid_taker_rows,
                sum(CASE WHEN valid_nif THEN 1 ELSE 0 END) AS valid_nif_rows,
                sum(CASE WHEN taker_buy_ratio IS NULL THEN 1 ELSE 0 END) AS taker_null_rows,
                sum(CASE WHEN nif_whale IS NULL THEN 1 ELSE 0 END) AS nif_null_rows,
                min(recent_trade_count_5m) AS min_trade_count_5m,
                max(recent_trade_count_5m) AS max_trade_count_5m,
                avg(recent_trade_count_5m) AS avg_trade_count_5m
            FROM microstructure_1m
            WHERE ts >= now() - INTERVAL '{int(hours)} hours'
            """,
        )
        latest_sql = """
            SELECT ts, obi, taker_buy_ratio, nif_whale, eai, data_stale,
                   depth_connected, trade_connected, poll_connected,
                   depth_age_sec, trade_age_sec, poll_age_sec,
                   recent_trade_count_5m, recent_trade_notional_5m, recent_whale_count_5m,
                   valid_taker_flow, valid_nif
            FROM microstructure_1m
            ORDER BY ts DESC
            LIMIT 5
        """
    else:
        _print_query(
            con,
            f"legacy_quality_last_{hours}h",
            f"""
            SELECT
                count(*) AS rows,
                sum(CASE WHEN taker_buy_ratio = 0 THEN 1 ELSE 0 END) AS taker_zero_rows,
                sum(CASE WHEN nif_whale = 0 AND nif_retail = 0 THEN 1 ELSE 0 END) AS nif_both_zero_rows,
                sum(CASE WHEN abs(obi) > 0 THEN 1 ELSE 0 END) AS obi_nonzero_rows,
                sum(CASE WHEN abs(eai) > 0 THEN 1 ELSE 0 END) AS eai_nonzero_rows,
                sum(CASE WHEN abs(shadow_toxicity_score) > 0 THEN 1 ELSE 0 END) AS toxicity_nonzero_rows
            FROM microstructure_1m
            WHERE ts >= now() - INTERVAL '{int(hours)} hours'
            """,
        )
        latest_sql = """
            SELECT ts, obi, taker_buy_ratio, nif_whale, nif_retail, eai,
                   shadow_toxicity_score, shadow_absorption_score
            FROM microstructure_1m
            ORDER BY ts DESC
            LIMIT 5
        """
    _print_query(con, "latest", latest_sql)
    con.close()


def audit_tail(path: Path, hours: int) -> None:
    print(f"\n=== tail_risk: {path} ===")
    if not path.exists():
        print("missing")
        return
    try:
        con = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        print(f"locked/unavailable: {exc}")
        return
    if not _table_exists(con, "tail_risk_1m"):
        print("missing table tail_risk_1m")
        return
    _print_query(
        con,
        "freshness",
        """
        SELECT count(*) AS rows, min(ts) AS min_ts, max(ts) AS max_ts,
               date_diff('second', max(ts), now()) AS max_age_sec
        FROM tail_risk_1m
        """,
    )
    cols = _columns(con, "tail_risk_1m")
    if {"valid_liq_stream", "ws_connected", "liq_event_count_1m"}.issubset(cols):
        _print_query(
            con,
            f"quality_last_{hours}h",
            f"""
            SELECT
                count(*) AS rows,
                sum(CASE WHEN long_usd_1m = 0 AND short_usd_1m = 0 THEN 1 ELSE 0 END) AS zero_liq_rows,
                sum(CASE WHEN valid_liq_stream THEN 1 ELSE 0 END) AS valid_stream_rows,
                sum(CASE WHEN ws_connected THEN 1 ELSE 0 END) AS ws_connected_rows,
                sum(liq_event_count_1m) AS liq_event_count_sum,
                max(liq_event_count_1m) AS max_liq_event_count_1m
            FROM tail_risk_1m
            WHERE ts >= now() - INTERVAL '{int(hours)} hours'
            """,
        )
        latest_sql = """
            SELECT ts, long_usd_1m, short_usd_1m, shadow_aftershock_prob,
                   ws_connected, ws_stale, ws_age_sec, liq_event_count_1m, valid_liq_stream
            FROM tail_risk_1m
            ORDER BY ts DESC
            LIMIT 5
        """
    else:
        _print_query(
            con,
            f"legacy_quality_last_{hours}h",
            f"""
            SELECT
                count(*) AS rows,
                sum(CASE WHEN long_usd_1m = 0 AND short_usd_1m = 0 THEN 1 ELSE 0 END) AS zero_liq_rows,
                sum(CASE WHEN shadow_aftershock_prob = 0 THEN 1 ELSE 0 END) AS zero_aftershock_rows
            FROM tail_risk_1m
            WHERE ts >= now() - INTERVAL '{int(hours)} hours'
            """,
        )
        latest_sql = """
            SELECT ts, long_usd_1m, short_usd_1m, shadow_aftershock_prob,
                   shadow_decay_half_life, shadow_risk_bucket
            FROM tail_risk_1m
            ORDER BY ts DESC
            LIMIT 5
        """
    _print_query(con, "latest", latest_sql)
    con.close()


def audit_polymarket(path: Path, hours: int) -> None:
    print(f"\n=== polymarket: {path} ===")
    if not path.exists():
        print("missing")
        return
    try:
        con = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        print(f"locked/unavailable: {exc}")
        return
    table = "polymarket_markets_10s_json"
    if not _table_exists(con, table):
        print(f"missing table {table}")
        con.close()
        return
    _print_query(
        con,
        f"{table}_freshness",
        f"""
        SELECT count(*) AS rows, min(ts) AS min_ts, max(ts) AS max_ts,
               date_diff('second', max(ts), now()) AS max_age_sec
        FROM {table}
        """,
    )
    cols = _columns(con, table)
    if {"snapshot_json", "current_price", "schema_version"}.issubset(cols):
        _print_query(
            con,
            f"{table}_latest",
            f"""
            SELECT ts, current_price, schema_version, markets_json,
                   length(snapshot_json) AS snapshot_json_bytes
            FROM {table}
            ORDER BY ts DESC
            LIMIT 3
            """,
        )
    else:
        _print_query(
            con,
            f"{table}_legacy_latest",
            f"SELECT ts, markets_json FROM {table} ORDER BY ts DESC LIMIT 3",
        )
    deprecated = [
        name
        for name in ("polymarket_features_10s", "polymarket_market_probs_10s")
        if _table_exists(con, name)
    ]
    if deprecated:
        print(f"deprecated tables still present: {', '.join(deprecated)}")
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit live DuckDB storage freshness and quality flags.")
    parser.add_argument("--live-dir", type=Path, default=DEFAULT_LIVE_DIR)
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    live_dir = args.live_dir
    audit_micro(live_dir / "microstructure.duckdb", args.hours)
    audit_tail(live_dir / "tail_risk.duckdb", args.hours)
    audit_polymarket(live_dir / "polymarket.duckdb", args.hours)


if __name__ == "__main__":
    main()
