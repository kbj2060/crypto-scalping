#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_DIR = ROOT / "data" / "live"


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return any(str(r[0]) == table for r in con.execute("SHOW TABLES").fetchall())


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {str(r[1]) for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}


def _expr(col: str, existing: set[str], default_sql: str) -> str:
    if col in existing:
        return _quote_ident(col)
    return default_sql


def _select_exprs(cols: Iterable[tuple[str, str]], existing: set[str]) -> str:
    return ",\n       ".join(
        f"{_expr(name, existing, default_sql)} AS {_quote_ident(name)}"
        for name, default_sql in cols
    )


def _create_view(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    view_name: str,
    cols: Iterable[tuple[str, str]],
) -> bool:
    if not _table_exists(con, source_table):
        print(f"skip {view_name}: missing table {source_table}")
        return False
    existing = _columns(con, source_table)
    if "ts" not in existing:
        print(f"skip {view_name}: missing required column ts")
        return False
    select_sql = _select_exprs(cols, existing)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {_quote_ident(view_name)} AS
        SELECT
           {select_sql}
        FROM {_quote_ident(source_table)}
        """
    )
    print(f"created view {view_name}")
    return True


def create_micro_views(path: Path) -> None:
    if not path.exists():
        print(f"skip microstructure: missing db {path}")
        return
    con = duckdb.connect(str(path))
    try:
        cols = [
            ("ts", "CAST(NULL AS TIMESTAMP)"),
            ("obi", "CAST(NULL AS DOUBLE)"),
            ("taker_buy_ratio", "CAST(NULL AS DOUBLE)"),
            ("spoofing_score", "CAST(NULL AS DOUBLE)"),
            ("nif_whale", "CAST(NULL AS DOUBLE)"),
            ("nif_retail", "CAST(NULL AS DOUBLE)"),
            ("eai", "CAST(NULL AS DOUBLE)"),
            ("oi_delta_pct", "CAST(NULL AS DOUBLE)"),
            ("funding_rate", "CAST(NULL AS DOUBLE)"),
            ("signal_bias", "CAST(NULL AS DOUBLE)"),
            ("shadow_toxicity_score", "CAST(NULL AS DOUBLE)"),
            ("shadow_queue_collapse", "CAST(NULL AS DOUBLE)"),
            ("shadow_absorption_score", "CAST(NULL AS DOUBLE)"),
            ("shadow_queue_bias", "CAST(NULL AS DOUBLE)"),
            ("shadow_regime_tag", "CAST(NULL AS VARCHAR)"),
            ("shadow_regime_conf", "CAST(NULL AS DOUBLE)"),
            ("recent_trade_count_5m", "CAST(NULL AS BIGINT)"),
            ("recent_trade_notional_5m", "CAST(NULL AS DOUBLE)"),
            ("recent_whale_count_5m", "CAST(NULL AS BIGINT)"),
            ("data_stale", "FALSE"),
            ("valid_taker_flow", "TRUE"),
            ("valid_nif", "TRUE"),
            ("schema_version", "CAST(NULL AS INTEGER)"),
        ]
        created = _create_view(con, "microstructure_1m", "microstructure_features_v1", cols)
        if created:
            con.execute(
                """
                CREATE OR REPLACE VIEW microstructure_model_ready_v1 AS
                SELECT *
                FROM microstructure_features_v1
                WHERE COALESCE(data_stale, FALSE) = FALSE
                  AND COALESCE(valid_taker_flow, TRUE) = TRUE
                  AND COALESCE(valid_nif, TRUE) = TRUE
                """
            )
            print("created view microstructure_model_ready_v1")
    finally:
        con.close()


def create_tail_views(path: Path) -> None:
    if not path.exists():
        print(f"skip tail_risk: missing db {path}")
        return
    con = duckdb.connect(str(path))
    try:
        cols = [
            ("ts", "CAST(NULL AS TIMESTAMP)"),
            ("long_usd_1m", "CAST(NULL AS DOUBLE)"),
            ("short_usd_1m", "CAST(NULL AS DOUBLE)"),
            ("mu_long", "CAST(NULL AS DOUBLE)"),
            ("sigma_long", "CAST(NULL AS DOUBLE)"),
            ("mu_short", "CAST(NULL AS DOUBLE)"),
            ("sigma_short", "CAST(NULL AS DOUBLE)"),
            ("shadow_aftershock_prob", "CAST(NULL AS DOUBLE)"),
            ("shadow_decay_half_life", "CAST(NULL AS DOUBLE)"),
            ("shadow_risk_bucket", "CAST(NULL AS VARCHAR)"),
            ("ws_connected", "TRUE"),
            ("ws_stale", "FALSE"),
            ("ws_age_sec", "CAST(NULL AS DOUBLE)"),
            ("liq_event_count_1m", "CAST(NULL AS BIGINT)"),
            ("valid_liq_stream", "TRUE"),
            ("schema_version", "CAST(NULL AS INTEGER)"),
        ]
        created = _create_view(con, "tail_risk_1m", "tail_risk_features_v1", cols)
        if created:
            con.execute(
                """
                CREATE OR REPLACE VIEW tail_risk_model_ready_v1 AS
                SELECT *
                FROM tail_risk_features_v1
                WHERE COALESCE(ws_stale, FALSE) = FALSE
                  AND COALESCE(valid_liq_stream, TRUE) = TRUE
                """
            )
            print("created view tail_risk_model_ready_v1")
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create curated live DuckDB feature views.")
    parser.add_argument("--live-dir", type=Path, default=DEFAULT_LIVE_DIR)
    args = parser.parse_args()

    live_dir = args.live_dir
    create_micro_views(live_dir / "microstructure.duckdb")
    create_tail_views(live_dir / "tail_risk.duckdb")
    print("polymarket: keep polymarket_markets_10s_json raw table only; derive features at preprocessing time")


if __name__ == "__main__":
    main()
