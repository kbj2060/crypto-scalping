"""Test the frozen ETH micro-scalp v4 research policy on BTC and SOL.

This is a no-training, no-selection, counterfactual transfer diagnostic.  Each
asset uses its own public market data, microstructure table, and order-book
table.  Decisions are made on completed one-minute bars and settled only from
the following completed close.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from features.engineering import FeatureEngineer  # noqa: E402
import build_eth_micro_scalp_v3_feature_stream_20260718 as base  # noqa: E402
import run_eth_micro_scalp_v4_fresh_forward_observer_20260718 as binding  # noqa: E402
import train_eval_deepscalp_pnl_20260717 as deep  # noqa: E402
import train_eval_eth_micro_scalp_source_stable_v4_20260718 as v4  # noqa: E402


MODEL_PATH = v4.MODEL_PATH
MICRO_DB = ROOT / "data/live/microstructure.duckdb"
REPORT_PATH = (
    ROOT
    / "data/ensemble/reports/eth_micro_scalp_v4_cross_asset_transfer_test_20260718.json"
)
ASSETS = {
    "btc": {
        "symbol": "BTCUSDT",
        "micro_table": "microstructure_1m_btc",
        "book_table": "orderbook_decision_snapshots_btc",
    },
    "sol": {
        "symbol": "SOLUSDT",
        "micro_table": "microstructure_1m_sol",
        "book_table": "orderbook_decision_snapshots_sol",
    },
}
FEE_SCENARIOS_BP = (2.0, 4.5, 5.5, 9.0)
MODEL_WINDOW_WARMUP_MINUTES = 60
FEATURE_CONTEXT_HOURS = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_naive(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def snapshot_database(source: Path, destination: Path, retries: int = 5) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    for attempt in range(retries):
        before = source.stat()
        shutil.copy2(source, destination)
        after = source.stat()
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            return {
                "source_size": after.st_size,
                "source_mtime_ns": after.st_mtime_ns,
                "snapshot_sha256": _sha256(destination),
                "copy_attempts": attempt + 1,
            }
        time.sleep(0.2)
    raise RuntimeError("microstructure database changed during every snapshot attempt")


def table_coverage(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    timestamp_column: str,
) -> dict[str, Any]:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    if table not in tables:
        raise RuntimeError(f"required asset table is missing: {table}")
    count, start, end = connection.execute(
        f"""
        SELECT count(*), min(timezone('UTC', \"{timestamp_column}\")),
               max(timezone('UTC', \"{timestamp_column}\"))
        FROM \"{table}\"
        """
    ).fetchone()
    if not count or start is None or end is None:
        raise RuntimeError(f"required asset table is empty: {table}")
    return {
        "rows": int(count),
        "start_utc": _utc_naive(start),
        "end_utc": _utc_naive(end),
    }


def read_asset_micro_frames(
    database: Path,
    config: dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    connection = duckdb.connect(str(database), read_only=True)
    try:
        micro_select = ", ".join(f'"{column}"' for column in deep.MICRO_COLUMNS)
        micro = connection.execute(
            f"""
            SELECT timezone('UTC', ts) AS timestamp,
                   timezone('UTC', ts) AS micro_source_ts,
                   {micro_select}
            FROM \"{config['micro_table']}\"
            WHERE timezone('UTC', ts) BETWEEN ? AND ?
            ORDER BY timestamp
            """,
            [start.to_pydatetime(), end.to_pydatetime()],
        ).fetchdf()
        book_select = ", ".join(f'"{column}"' for column in deep.BOOK_COLUMNS)
        book = connection.execute(
            f"""
            SELECT timezone('UTC', recorded_at_kst) AS timestamp,
                   timezone('UTC', recorded_at_kst) AS book_source_ts,
                   {book_select}
            FROM \"{config['book_table']}\"
            WHERE timezone('UTC', recorded_at_kst) BETWEEN ? AND ?
            ORDER BY timestamp
            """,
            [start.to_pydatetime(), end.to_pydatetime()],
        ).fetchdf()
    finally:
        connection.close()
    for frame in (micro, book):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    micro["micro_source_ts"] = pd.to_datetime(micro["micro_source_ts"])
    book["book_source_ts"] = pd.to_datetime(book["book_source_ts"])
    return micro, book


def _fetch_metric_rows(
    session: requests.Session,
    endpoint: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    final_end_ms = int(end.tz_localize("UTC").timestamp() * 1000)
    cursor_end = final_end_ms
    rows: list[dict[str, Any]] = []
    while cursor_end >= start_ms:
        batch = base._request_json(
            session,
            endpoint,
            {
                "symbol": symbol,
                "period": "5m",
                "startTime": start_ms,
                "endTime": cursor_end,
                "limit": 500,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        first_timestamp = min(int(row["timestamp"]) for row in batch)
        if first_timestamp <= start_ms:
            break
        next_end = first_timestamp - 1
        if next_end >= cursor_end:
            raise RuntimeError(f"metric pagination did not retreat: {endpoint}")
        cursor_end = next_end
        time.sleep(0.05)
    unique = {
        int(row["timestamp"]): row
        for row in rows
        if start_ms <= int(row["timestamp"]) <= final_end_ms
    }
    return [unique[key] for key in sorted(unique)]


def fetch_asset_metrics(
    session: requests.Session,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    series = {
        name: _fetch_metric_rows(
            session, base.PUBLIC_ENDPOINTS[name], symbol, start, end
        )
        for name in (
            "open_interest",
            "top_position",
            "top_account",
            "global_account",
            "taker",
        )
    }
    open_interest = pd.DataFrame(series["open_interest"])
    if open_interest.empty:
        raise RuntimeError(f"{symbol} open-interest history is empty")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(open_interest["timestamp"], unit="ms"),
            "sum_open_interest": pd.to_numeric(open_interest["sumOpenInterest"]),
            "sum_open_interest_value": pd.to_numeric(
                open_interest["sumOpenInterestValue"]
            ),
        }
    ).sort_values("timestamp")
    mappings = (
        ("top_position", "sum_toptrader_long_short_ratio", "longShortRatio"),
        ("top_account", "count_toptrader_long_short_ratio", "longShortRatio"),
        ("global_account", "count_long_short_ratio", "longShortRatio"),
        ("taker", "sum_taker_long_short_vol_ratio", "buySellRatio"),
    )
    for source, target, value_name in mappings:
        raw = pd.DataFrame(series[source])
        if raw.empty:
            raise RuntimeError(f"{symbol} metric history is empty: {source}")
        aligned = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(raw["timestamp"], unit="ms"),
                target: pd.to_numeric(raw[value_name]),
            }
        ).sort_values("timestamp")
        frame = pd.merge_asof(
            frame,
            aligned,
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta(minutes=5),
        )
    required = [target for _, target, _ in mappings]
    if frame[required].isna().any().any():
        raise RuntimeError(f"{symbol} metric series do not align causally")
    return frame


def fetch_asset_funding(
    session: requests.Session,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    end_ms = int(end.tz_localize("UTC").timestamp() * 1000)
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        batch = base._request_json(
            session,
            base.PUBLIC_ENDPOINTS["funding"],
            {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise RuntimeError(f"{symbol} funding pagination did not advance")
        cursor = next_cursor
        if len(batch) < 1000:
            break
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError(f"{symbol} funding history is empty")
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["fundingTime"], unit="ms"),
            "last_funding_rate": pd.to_numeric(raw["fundingRate"]),
        }
    ).drop_duplicates("timestamp", keep="last").sort_values("timestamp")


def build_asset_stream(
    database: Path,
    asset: str,
    config: dict[str, str],
    source_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "crypto-scalping-v4-cross-asset-transfer-test/1.0"}
    )
    symbol = config["symbol"]
    primary = base.fetch_klines(
        session, symbol, "1m", source_start, evaluation_end
    )
    btc_context = base.fetch_klines(
        session,
        "BTCUSDT",
        "5m",
        source_start.floor("5min"),
        evaluation_end.floor("5min"),
    )
    context_start = source_start - pd.Timedelta(hours=FEATURE_CONTEXT_HOURS)
    metrics = fetch_asset_metrics(
        session, symbol, context_start, evaluation_end
    )
    funding = fetch_asset_funding(
        session, symbol, context_start, evaluation_end
    )
    primary = pd.merge_asof(
        primary,
        metrics,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(hours=9),
    )
    primary = pd.merge_asof(
        primary, funding, on="timestamp", direction="backward"
    )
    required_market = [
        "sum_open_interest_value",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "last_funding_rate",
    ]
    if primary[required_market].isna().any().any():
        raise RuntimeError(f"{asset} causal market-data join has missing values")
    engineered = FeatureEngineer(
        candle_minutes=1, keep_only_active=True, include_entry_price=False
    ).process(
        primary,
        btc_context[["timestamp", "close", "volume", "quote_volume"]],
    )
    base_values, base_names = deep.make_base_features(engineered)
    base_indices = [base_names.index(name) for name in v4.SOURCE_STABLE_FEATURES]
    micro, book = read_asset_micro_frames(
        database,
        config,
        pd.Timestamp(engineered["timestamp"].min()),
        pd.Timestamp(engineered["timestamp"].max()),
    )
    micro_values, micro_names, coverage = deep.make_micro_features(
        engineered["timestamp"], micro, book
    )
    micro_indices = [
        micro_names.index(name) for name in v4.v3.core.MICRO_FEATURES
    ]
    frame = pd.DataFrame(
        base_values[:, base_indices], columns=v4.SOURCE_STABLE_FEATURES
    )
    for index, name in enumerate(v4.v3.core.MICRO_FEATURES):
        frame[name] = micro_values[:, micro_indices[index]]
    frame.insert(
        0,
        "close",
        pd.to_numeric(engineered["close"], errors="coerce").to_numpy(),
    )
    frame.insert(0, "timestamp", pd.to_datetime(engineered["timestamp"]).to_numpy())
    differences = frame["timestamp"].diff().dropna()
    if len(differences) and (differences != pd.Timedelta(minutes=1)).any():
        raise RuntimeError(f"{asset} feature stream is not one-minute causal data")
    required = ["close", *v4.SOURCE_STABLE_FEATURES, "micro_available", "book_available"]
    if not np.isfinite(frame[required].to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"{asset} feature stream has non-finite required inputs")
    return frame, {
        "symbol": symbol,
        "primary_kline_rows": len(primary),
        "btc_context_rows": len(btc_context),
        "metric_rows": len(metrics),
        "funding_rows": len(funding),
        **coverage,
    }


def holding_summary(positions: np.ndarray) -> dict[str, Any]:
    completed: list[int] = []
    current_length = 0
    current_position = 0
    for position in positions.astype(int):
        if position == current_position and position != 0:
            current_length += 1
            continue
        if current_position != 0 and current_length:
            completed.append(current_length)
        current_position = position
        current_length = 1 if position != 0 else 0
    open_bars = current_length if current_position != 0 else 0
    values = np.asarray(completed, dtype=float)
    return {
        "completed_count": len(completed),
        "min_minutes": int(values.min()) if len(values) else None,
        "median_minutes": float(np.median(values)) if len(values) else None,
        "p95_minutes": float(np.quantile(values, 0.95)) if len(values) else None,
        "max_minutes": int(values.max()) if len(values) else None,
        "open_position_minutes_at_end": int(open_bars),
    }


def replay_decisions(
    decisions: list[dict[str, Any]], fee_bp: float
) -> dict[str, Any]:
    settled = []
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for current, following in zip(decisions, decisions[1:]):
        timestamp = _utc_naive(current["timestamp"])
        settlement = _utc_naive(following["timestamp"])
        if settlement - timestamp != pd.Timedelta(minutes=1):
            raise RuntimeError("decision settlement cadence is not one minute")
        start_close = float(current["close"])
        end_close = float(following["close"])
        previous = int(current["previous_position"])
        position = int(current["target_position"])
        turnover = abs(position - previous)
        gross = position * (end_close / start_close - 1.0)
        cost = fee_bp / 10_000.0 * turnover
        net = gross - cost
        equity *= 1.0 + net
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 1.0 - equity / peak)
        settled.append((timestamp, position, turnover, gross, cost, net, equity))
    positions = np.asarray([row[1] for row in settled], dtype=np.int8)
    turnovers = np.asarray([row[2] for row in settled], dtype=float)
    gross = np.asarray([row[3] for row in settled], dtype=float)
    costs = np.asarray([row[4] for row in settled], dtype=float)
    entries_or_reversals = int(
        sum(
            row[2] > 0 and int(decisions[index]["target_position"]) != 0
            for index, row in enumerate(settled)
        )
    )
    exits_or_reversals = int(
        sum(
            row[2] > 0 and int(decisions[index]["previous_position"]) != 0
            for index, row in enumerate(settled)
        )
    )
    return {
        "fee_bp_per_notional_change": fee_bp,
        "settled_intervals": len(settled),
        "compounded_return_pct": (equity - 1.0) * 100.0,
        "additive_gross_return_pct": float(gross.sum() * 100.0),
        "additive_cost_pct": float(costs.sum() * 100.0),
        "max_drawdown_pct": max_drawdown * 100.0,
        "turnover": float(turnovers.sum()),
        "entries_or_reversals": entries_or_reversals,
        "exits_or_reversals": exits_or_reversals,
        "exposure_fraction": float(np.mean(positions != 0)) if len(positions) else 0.0,
        "long_fraction": float(np.mean(positions > 0)) if len(positions) else 0.0,
        "short_fraction": float(np.mean(positions < 0)) if len(positions) else 0.0,
        "holding": holding_summary(positions),
    }


def drift_summary(frame: pd.DataFrame, runtime: Any) -> dict[str, Any]:
    scalers = runtime.checkpoint["scalers"]
    groups = {
        "base": (
            list(v4.SOURCE_STABLE_FEATURES),
            np.asarray(scalers["base_center"]),
            np.asarray(scalers["base_scale"]),
        ),
        "micro": (
            list(v4.v3.core.MICRO_FEATURES),
            np.asarray(scalers["micro_center"]),
            np.asarray(scalers["micro_scale"]),
        ),
    }
    result: dict[str, Any] = {}
    for group, (names, center, scale) in groups.items():
        raw = frame[names].to_numpy(dtype=np.float64)
        z = np.abs((raw - center) / scale)
        finite = np.isfinite(z)
        per_feature_p99 = np.nanquantile(z, 0.99, axis=0)
        worst = np.argsort(np.nan_to_num(per_feature_p99, nan=-1.0))[-5:][::-1]
        result[group] = {
            "finite_fraction": float(finite.mean()),
            "fraction_abs_z_gt_5": float(np.mean(z[finite] > 5.0)) if finite.any() else None,
            "fraction_abs_z_gt_10": float(np.mean(z[finite] > 10.0)) if finite.any() else None,
            "worst_feature_p99_abs_z": [
                {"name": names[index], "p99_abs_z": float(per_feature_p99[index])}
                for index in worst
            ],
        }
    return result


def run(
    micro_db: Path = MICRO_DB,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    runtime = binding.observer.load_runtime(device_name="cpu")
    if runtime.model_sha256 != _sha256(MODEL_PATH):
        raise RuntimeError("frozen ETH v4 model hash changed during test")
    with tempfile.TemporaryDirectory(prefix="eth-v4-cross-asset-") as directory:
        snapshot_path = Path(directory) / "microstructure.duckdb"
        snapshot = snapshot_database(micro_db, snapshot_path)
        connection = duckdb.connect(str(snapshot_path), read_only=True)
        try:
            coverage = {
                asset: {
                    "micro": table_coverage(
                        connection, config["micro_table"], "ts"
                    ),
                    "book": table_coverage(
                        connection, config["book_table"], "recorded_at_kst"
                    ),
                }
                for asset, config in ASSETS.items()
            }
        finally:
            connection.close()
        common_source_start = max(
            max(values["micro"]["start_utc"], values["book"]["start_utc"])
            for values in coverage.values()
        ).ceil("min")
        evaluation_start = common_source_start + pd.Timedelta(
            minutes=MODEL_WINDOW_WARMUP_MINUTES
        )
        evaluation_end = min(
            min(values["micro"]["end_utc"], values["book"]["end_utc"])
            for values in coverage.values()
        ).floor("min")
        if evaluation_end <= evaluation_start:
            raise RuntimeError("BTC/SOL common evaluation interval is empty")
        source_start = common_source_start - pd.Timedelta(
            hours=FEATURE_CONTEXT_HOURS
        )
        results: dict[str, Any] = {}
        for asset, config in ASSETS.items():
            print(f"building {asset.upper()} causal stream", flush=True)
            stream, source = build_asset_stream(
                snapshot_path, asset, config, source_start, evaluation_end
            )
            if len(stream) < runtime.config.window:
                raise RuntimeError(f"{asset} stream is shorter than the model window")
            decisions = binding.observer.build_decisions(
                stream,
                runtime,
                previous_position=0,
                after_timestamp=None,
                fresh_start=evaluation_start,
            )
            if len(decisions) < 2:
                raise RuntimeError(f"{asset} produced fewer than two decisions")
            evaluated_frame = stream[
                (stream["timestamp"] >= evaluation_start)
                & (stream["timestamp"] <= evaluation_end)
            ]
            available = binding.observer._available(evaluated_frame)
            results[asset] = {
                "symbol": config["symbol"],
                "evaluation_start_utc": str(evaluation_start),
                "evaluation_end_utc": str(evaluation_end),
                "calendar_days": int(
                    pd.Series(pd.to_datetime(evaluated_frame["timestamp"]).dt.date).nunique()
                ),
                "feature_rows": len(evaluated_frame),
                "decision_count": len(decisions),
                "usable_decision_fraction": float(np.mean(available)),
                "final_position": int(decisions[-1]["target_position"]),
                "position_changes_all_decisions": int(
                    sum(int(row["position_change"]) != 0 for row in decisions)
                ),
                "source": source,
                "input_drift_against_eth_scaler": drift_summary(
                    evaluated_frame, runtime
                ),
                "fee_scenarios": {
                    f"{fee:.2f}bp_per_notional_change": replay_decisions(
                        decisions, fee
                    )
                    for fee in FEE_SCENARIOS_BP
                },
            }
    report = {
        "schema_version": "eth_micro_scalp_v4.cross_asset_transfer_test.v1",
        "created_at_utc": str(pd.Timestamp.utcnow()),
        "model_id": v4.MODEL_ID,
        "model_sha256": runtime.model_sha256,
        "selected_research_policy": asdict(runtime.policy),
        "assets": results,
        "data_coverage": coverage,
        "microstructure_snapshot": snapshot,
        "evidence_class": (
            "short-window BTC/SOL cross-asset counterfactual transfer diagnostic; "
            "not training, selection, promotion, or actual execution evidence"
        ),
        "limitations": {
            "approximately_four_days_of_asset_microstructure": True,
            "eth_frozen_cache_parity_applicable": False,
            "cross_asset_distribution_shift_expected": True,
            "sample_sufficient_for_training": False,
            "sample_sufficient_for_promotion": False,
        },
        "compliance": {
            "training_performed": False,
            "parameter_updates": 0,
            "policy_selection_performed": False,
            "asset_specific_threshold_tuning_performed": False,
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "fixed_holding_period_used": False,
            "settlement_rule": "decision close t to following completed close t+1",
            "unit_notional": 1.0,
        },
    }
    base._write_json_atomic(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-db", type=Path, default=MICRO_DB)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.micro_db, args.report)
    print(json.dumps(result, indent=2, default=base._json_default))
