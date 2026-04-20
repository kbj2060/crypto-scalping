import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def analyze(db_path: str, table: str, out_dir: str, top_n: int) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = {x[0] for x in con.execute("SHOW TABLES").fetchall()}
        if table not in tables:
            print(f"[ERROR] table not found: {table}")
            print(f"[INFO] available tables: {sorted(tables)}")
            return 1

        df = con.execute(f"SELECT * FROM {table} ORDER BY ts").df()
    finally:
        con.close()

    if df.empty:
        print("[ERROR] no rows in table")
        return 1

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    for c in [
        "mode_prob",
        "weighted_target",
        "prob_momentum_1m",
        "shock_delta_1m",
        "shock_delta_3m",
        "shock_z_1m",
        "book_imbalance",
        "event_volatility",
        "current_price",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Update interval diagnostics
    diffs = df["ts"].diff().dt.total_seconds().dropna()
    interval_summary = {
        "median_sec": _safe_float(diffs.median(), 0.0) if len(diffs) else 0.0,
        "mean_sec": _safe_float(diffs.mean(), 0.0) if len(diffs) else 0.0,
        "p90_sec": _safe_float(diffs.quantile(0.9), 0.0) if len(diffs) else 0.0,
        "max_sec": _safe_float(diffs.max(), 0.0) if len(diffs) else 0.0,
    }

    status_counts = (
        df["status"].fillna("NULL").value_counts().to_dict()
        if "status" in df.columns
        else {}
    )

    shocks = df[df.get("shock_trigger", False) == True].copy()  # noqa: E712
    if "prob_momentum_1m" in df.columns:
        top_momentum = (
            df.assign(abs_mom=df["prob_momentum_1m"].abs())
            .sort_values("abs_mom", ascending=False)
            .head(max(1, int(top_n)))
        )
    else:
        top_momentum = df.head(0).copy()

    summary = {
        "db_path": str(Path(db_path).resolve()),
        "table": table,
        "rows": int(len(df)),
        "ts_start_utc": str(df["ts"].min()),
        "ts_end_utc": str(df["ts"].max()),
        "status_counts": status_counts,
        "shock_trigger_count": int(len(shocks)),
        "shock_trigger_ratio_pct": _safe_float(len(shocks) * 100.0 / max(len(df), 1), 0.0),
        "interval_seconds": interval_summary,
        "avg_mode_prob": _safe_float(df.get("mode_prob", pd.Series(dtype=float)).mean(), 0.0),
        "avg_event_volatility": _safe_float(df.get("event_volatility", pd.Series(dtype=float)).mean(), 0.0),
        "avg_book_imbalance": _safe_float(df.get("book_imbalance", pd.Series(dtype=float)).mean(), 0.0),
    }

    summary_path = out / "polymarket_analysis_summary.json"
    top_path = out / "polymarket_top_momentum.csv"
    shocks_path = out / "polymarket_shock_triggers.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    keep_cols = [c for c in [
        "ts",
        "status",
        "slug",
        "mode_label",
        "mode_prob",
        "weighted_target",
        "current_price",
        "prob_momentum_1m",
        "shock_delta_1m",
        "shock_delta_3m",
        "shock_z_1m",
        "shock_trigger",
        "book_imbalance",
        "event_volatility",
        "signal",
        "risk_state",
    ] if c in df.columns]

    if len(top_momentum):
        top_momentum.loc[:, keep_cols + (["abs_mom"] if "abs_mom" in top_momentum.columns else [])].to_csv(top_path, index=False)
    else:
        pd.DataFrame(columns=keep_cols + ["abs_mom"]).to_csv(top_path, index=False)

    if len(shocks):
        shocks.loc[:, keep_cols].to_csv(shocks_path, index=False)
    else:
        pd.DataFrame(columns=keep_cols).to_csv(shocks_path, index=False)

    print("[OK] analysis complete")
    print(f"- rows: {summary['rows']}")
    print(f"- range(UTC): {summary['ts_start_utc']} -> {summary['ts_end_utc']}")
    print(f"- shock_trigger_count: {summary['shock_trigger_count']} ({summary['shock_trigger_ratio_pct']:.2f}%)")
    print(f"- interval median/mean sec: {summary['interval_seconds']['median_sec']:.2f} / {summary['interval_seconds']['mean_sec']:.2f}")
    print(f"- saved: {summary_path}")
    print(f"- saved: {top_path}")
    print(f"- saved: {shocks_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Analyze polymarket DuckDB snapshots.")
    parser.add_argument("--db", default="data/live/polymarket.duckdb", help="DuckDB path")
    parser.add_argument("--table", default="polymarket_10s", help="Table name")
    parser.add_argument("--out", default="data/live", help="Output directory")
    parser.add_argument("--top-n", type=int, default=30, help="Top momentum rows to export")
    args = parser.parse_args()
    raise SystemExit(analyze(args.db, args.table, args.out, args.top_n))


if __name__ == "__main__":
    main()
