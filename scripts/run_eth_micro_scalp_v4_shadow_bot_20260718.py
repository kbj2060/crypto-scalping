"""Dedicated non-executing one-minute shadow bot for ETH micro-scalp v4.

The bot rebuilds the exact public-data feature stream, runs the frozen v4
research policy bar by bar, and records unit-notional counterfactual PnL only
after the following completed minute is available.  It has no order or account
execution capability.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_eth_micro_scalp_v4_feature_stream_20260718 as builder  # noqa: E402
import run_eth_micro_scalp_v4_fresh_forward_observer_20260718 as binding  # noqa: E402


MODEL_ID = binding.v4.MODEL_ID
MODEL_PATH = binding.v4.MODEL_PATH
FRESH_START_UTC = binding.FRESH_START_UTC
FEATURE_STREAM_PATH = binding.DEFAULT_FEATURE_STREAM
FEATURE_BUILD_REPORT = binding.OBSERVER_DIR / "feature_stream_build.json"
SHADOW_DB_PATH = ROOT / "data/live/eth_micro_scalp_v4_shadow.duckdb"
SHADOW_STATE_PATH = ROOT / "data/live/eth_micro_scalp_v4_shadow_state.json"
FEE_SCENARIOS_BP = (2.0, 4.5, 5.5, 9.0)
DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_MAX_STREAM_AGE_MINUTES = 5.0


def _normalize_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _validate_build_report(
    report: dict[str, Any],
    runtime: binding.observer.Runtime,
    feature_stream: Path = FEATURE_STREAM_PATH,
) -> None:
    failures: list[str] = []
    if report.get("model_id") != MODEL_ID:
        failures.append("model_id")
    if report.get("model_sha256") != runtime.model_sha256:
        failures.append("model_sha256")
    if _normalize_timestamp(report.get("fresh_start_utc")) != FRESH_START_UTC:
        failures.append("fresh_start_utc")
    if report.get("published") is not True:
        failures.append("published")
    if not bool((report.get("parity") or {}).get("pass")):
        failures.append("parity")
    if not bool((report.get("stream_contract") or {}).get("pass")):
        failures.append("stream_contract")
    if report.get("order_endpoints_used") is not False:
        failures.append("order_endpoints_used")
    if Path(str(report.get("output", ""))).resolve() != feature_stream.resolve():
        failures.append("output")
    if failures:
        raise RuntimeError(f"v4 shadow feature-build contract mismatch: {failures}")


def _load_validated_stream(
    runtime: binding.observer.Runtime,
    max_stream_age_minutes: float,
    feature_stream: Path = FEATURE_STREAM_PATH,
    report_path: Path = FEATURE_BUILD_REPORT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if max_stream_age_minutes <= 0.0:
        raise ValueError("max_stream_age_minutes must be positive")
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text())
    _validate_build_report(report, runtime, feature_stream)
    frame = binding.observer.load_feature_stream(feature_stream)
    latest = _normalize_timestamp(frame["timestamp"].iloc[-1])
    completed_at = latest + pd.Timedelta(minutes=1)
    now = pd.Timestamp.utcnow().tz_localize(None)
    age_minutes = max(0.0, float((now - completed_at).total_seconds()) / 60.0)
    if age_minutes > max_stream_age_minutes:
        raise RuntimeError(
            f"v4 shadow feature stream is stale: age_minutes={age_minutes:.3f}"
        )
    return frame, {
        "latest_feature_timestamp_utc": str(latest),
        "latest_feature_completed_at_utc": str(completed_at),
        "stream_age_minutes": age_minutes,
        "rows": len(frame),
        "build_report": str(report_path),
    }


def _initialize_shadow_pnl(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS shadow_pnl (
            decision_timestamp TIMESTAMP NOT NULL,
            settlement_timestamp TIMESTAMP NOT NULL,
            fee_bp DOUBLE NOT NULL,
            previous_position INTEGER NOT NULL,
            position INTEGER NOT NULL,
            start_close DOUBLE NOT NULL,
            end_close DOUBLE NOT NULL,
            turnover DOUBLE NOT NULL,
            price_return DOUBLE NOT NULL,
            gross_return DOUBLE NOT NULL,
            cost_return DOUBLE NOT NULL,
            net_return DOUBLE NOT NULL,
            equity DOUBLE NOT NULL,
            causal_settlement BOOLEAN NOT NULL,
            PRIMARY KEY (decision_timestamp, fee_bp)
        )
        """
    )


def settle_shadow_pnl(
    database: Path = SHADOW_DB_PATH,
    fee_scenarios_bp: tuple[float, ...] = FEE_SCENARIOS_BP,
) -> int:
    """Settle t only from t+1 close; the latest decision always remains open."""
    if not database.exists():
        raise FileNotFoundError(database)
    fees = tuple(float(value) for value in fee_scenarios_bp)
    if not fees or len(set(fees)) != len(fees) or any(value < 0.0 for value in fees):
        raise ValueError("fee scenarios must be unique non-negative basis-point values")
    connection = duckdb.connect(str(database))
    inserted_intervals = 0
    try:
        connection.execute("BEGIN TRANSACTION")
        _initialize_shadow_pnl(connection)
        decisions = connection.execute(
            """
            SELECT timestamp, close, previous_position, target_position
            FROM decisions
            ORDER BY timestamp
            """
        ).fetchall()
        last_equity = {fee: 1.0 for fee in fees}
        for fee in fees:
            row = connection.execute(
                """
                SELECT equity FROM shadow_pnl
                WHERE fee_bp = ? ORDER BY decision_timestamp DESC LIMIT 1
                """,
                [fee],
            ).fetchone()
            if row is not None:
                last_equity[fee] = float(row[0])
        maximum_settled = connection.execute(
            "SELECT max(decision_timestamp) FROM shadow_pnl"
        ).fetchone()[0]
        maximum_settled = (
            _normalize_timestamp(maximum_settled)
            if maximum_settled is not None
            else None
        )
        for current, following in zip(decisions, decisions[1:]):
            timestamp = _normalize_timestamp(current[0])
            settlement_timestamp = _normalize_timestamp(following[0])
            if settlement_timestamp - timestamp != pd.Timedelta(minutes=1):
                continue
            existing = int(
                connection.execute(
                    "SELECT count(*) FROM shadow_pnl WHERE decision_timestamp = ?",
                    [timestamp],
                ).fetchone()[0]
            )
            if existing == len(fees):
                continue
            if existing != 0:
                raise RuntimeError(
                    f"partial shadow PnL fee set at {timestamp}: {existing}/{len(fees)}"
                )
            if maximum_settled is not None and timestamp <= maximum_settled:
                raise RuntimeError("non-monotonic shadow PnL backfill is forbidden")
            start_close = float(current[1])
            end_close = float(following[1])
            previous_position = int(current[2])
            position = int(current[3])
            if previous_position not in (-1, 0, 1) or position not in (-1, 0, 1):
                raise RuntimeError("invalid shadow position")
            if not (
                math.isfinite(start_close)
                and math.isfinite(end_close)
                and start_close > 0.0
                and end_close > 0.0
            ):
                raise RuntimeError("invalid shadow settlement price")
            turnover = abs(float(position - previous_position))
            price_return = end_close / start_close - 1.0
            gross_return = float(position) * price_return
            for fee in fees:
                cost_return = fee / 10_000.0 * turnover
                net_return = gross_return - cost_return
                equity = last_equity[fee] * (1.0 + net_return)
                if not math.isfinite(equity) or equity <= 0.0:
                    raise RuntimeError("invalid shadow equity")
                connection.execute(
                    """
                    INSERT INTO shadow_pnl VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, true
                    )
                    """,
                    [
                        timestamp,
                        settlement_timestamp,
                        fee,
                        previous_position,
                        position,
                        start_close,
                        end_close,
                        turnover,
                        price_return,
                        gross_return,
                        cost_return,
                        net_return,
                        equity,
                    ],
                )
                last_equity[fee] = equity
            maximum_settled = timestamp
            inserted_intervals += 1
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return inserted_intervals


def shadow_summary(database: Path = SHADOW_DB_PATH) -> dict[str, Any]:
    if not database.exists():
        return {
            "schema_version": "eth_micro_scalp_v4.shadow_bot.v1",
            "model_id": MODEL_ID,
            "database": str(database),
            "decision_count": 0,
            "settled_intervals": 0,
            "performance_eligible": False,
            "reason": "shadow database does not exist",
            "order_submission_supported": False,
        }
    connection = duckdb.connect(str(database), read_only=True)
    try:
        decisions = int(connection.execute("SELECT count(*) FROM decisions").fetchone()[0])
        latest = connection.execute(
            """
            SELECT timestamp, target_position, close
            FROM decisions ORDER BY timestamp DESC LIMIT 1
            """
        ).fetchone()
        pnl_table = bool(
            connection.execute(
                """
                SELECT count(*) > 0 FROM information_schema.tables
                WHERE table_name = 'shadow_pnl'
                """
            ).fetchone()[0]
        )
        settled = 0
        scenarios: dict[str, Any] = {}
        if pnl_table:
            settled = int(
                connection.execute(
                    "SELECT count(DISTINCT decision_timestamp) FROM shadow_pnl"
                ).fetchone()[0]
            )
            for fee in FEE_SCENARIOS_BP:
                rows = connection.execute(
                    """
                    SELECT decision_timestamp, gross_return, cost_return, net_return, equity
                    FROM shadow_pnl WHERE fee_bp = ? ORDER BY decision_timestamp
                    """,
                    [fee],
                ).fetchall()
                equities = np.asarray([float(row[4]) for row in rows], dtype=np.float64)
                curve = np.r_[1.0, equities]
                peak = np.maximum.accumulate(curve)
                drawdown = 1.0 - curve / peak
                scenarios[f"{fee:.2f}bp_per_notional_change"] = {
                    "compounded_return_pct": (
                        float((equities[-1] - 1.0) * 100.0) if len(equities) else 0.0
                    ),
                    "additive_gross_return_pct": float(
                        sum(float(row[1]) for row in rows) * 100.0
                    ),
                    "additive_cost_pct": float(
                        sum(float(row[2]) for row in rows) * 100.0
                    ),
                    "max_drawdown_pct": (
                        float(drawdown.max() * 100.0) if len(drawdown) else 0.0
                    ),
                }
    finally:
        connection.close()
    return {
        "schema_version": "eth_micro_scalp_v4.shadow_bot.v1",
        "model_id": MODEL_ID,
        "database": str(database),
        "fresh_start_utc": str(FRESH_START_UTC),
        "decision_count": decisions,
        "settled_intervals": settled,
        "unsettled_decisions": max(0, decisions - settled),
        "latest_decision_utc": str(latest[0]) if latest is not None else None,
        "current_position": int(latest[1]) if latest is not None else 0,
        "latest_close": float(latest[2]) if latest is not None else None,
        "unit_notional": 1.0,
        "fee_scenarios": scenarios,
        "evidence_class": "counterfactual completed-close-to-next-completed-close",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "fixed_holding_period_used": False,
        "performance_eligible": False,
        "reason": "counterfactual shadow PnL is diagnostic and is not actual execution evidence",
        "order_submission_supported": False,
    }


def run_step(
    runtime: binding.observer.Runtime,
    *,
    build_features: bool,
    lookback_days: int,
    max_stream_age_minutes: float,
    database: Path = SHADOW_DB_PATH,
    state_path: Path = SHADOW_STATE_PATH,
) -> dict[str, Any]:
    if build_features:
        build_report = builder.build(lookback_days=lookback_days)
        if not build_report["published"]:
            raise RuntimeError("v4 exact feature build was not published")
    frame, stream_status = _load_validated_stream(
        runtime, max_stream_age_minutes
    )
    last_timestamp, previous_position = binding.observer.observer_state(database)
    decisions = binding.observer.build_decisions(
        frame,
        runtime,
        previous_position=previous_position,
        after_timestamp=last_timestamp,
        fresh_start=FRESH_START_UTC,
    )
    inserted_decisions = binding.observer.commit_decisions(
        database, runtime, decisions
    )
    inserted_settlements = settle_shadow_pnl(database)
    summary = shadow_summary(database)
    payload = {
        "schema_version": "eth_micro_scalp_v4.shadow_bot_step.v1",
        "model_id": MODEL_ID,
        "model_sha256": runtime.model_sha256,
        "computed_decisions": len(decisions),
        "inserted_decisions": inserted_decisions,
        "inserted_settlements": inserted_settlements,
        "stream": stream_status,
        "summary": summary,
        "activation_allowed": False,
        "order_submission_supported": False,
    }
    binding.observer._write_json_atomic(state_path, payload)
    return payload


def _write_failure_state(state_path: Path, error: Exception) -> None:
    binding.observer._write_json_atomic(
        state_path,
        {
            "schema_version": "eth_micro_scalp_v4.shadow_bot_step.v1",
            "model_id": MODEL_ID,
            "status": "failed_closed",
            "error_type": type(error).__name__,
            "error": str(error),
            "activation_allowed": False,
            "order_submission_supported": False,
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    step = subparsers.add_parser("step")
    step.add_argument("--skip-build", action="store_true")
    step.add_argument("--lookback-days", type=int, default=21)
    step.add_argument(
        "--max-stream-age-minutes",
        type=float,
        default=DEFAULT_MAX_STREAM_AGE_MINUTES,
    )
    step.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    serve = subparsers.add_parser("serve")
    serve.add_argument("--skip-build", action="store_true")
    serve.add_argument("--lookback-days", type=int, default=21)
    serve.add_argument(
        "--max-stream-age-minutes",
        type=float,
        default=DEFAULT_MAX_STREAM_AGE_MINUTES,
    )
    serve.add_argument(
        "--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS
    )
    serve.add_argument("--max-cycles", type=int, default=0)
    serve.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    subparsers.add_parser("summary")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "summary":
        print(json.dumps(shadow_summary(), indent=2))
        return 0
    runtime = binding.observer.load_runtime(device_name=args.device)
    if args.command == "step":
        try:
            payload = run_step(
                runtime,
                build_features=not args.skip_build,
                lookback_days=args.lookback_days,
                max_stream_age_minutes=args.max_stream_age_minutes,
            )
        except Exception as error:
            _write_failure_state(SHADOW_STATE_PATH, error)
            raise
        print(json.dumps(payload, indent=2, default=binding.observer._json_default))
        return 0
    if args.interval_seconds < 60:
        raise ValueError("interval_seconds must be at least 60")
    if args.max_cycles < 0:
        raise ValueError("max_cycles cannot be negative")
    cycle = 0
    while args.max_cycles == 0 or cycle < args.max_cycles:
        started = time.monotonic()
        try:
            payload = run_step(
                runtime,
                build_features=not args.skip_build,
                lookback_days=args.lookback_days,
                max_stream_age_minutes=args.max_stream_age_minutes,
            )
            print(json.dumps(payload, default=binding.observer._json_default), flush=True)
        except Exception as error:
            _write_failure_state(SHADOW_STATE_PATH, error)
            print(
                json.dumps(
                    {
                        "status": "failed_closed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "order_submission_supported": False,
                    }
                ),
                flush=True,
            )
        cycle += 1
        if args.max_cycles and cycle >= args.max_cycles:
            break
        remaining = max(0.0, float(args.interval_seconds) - (time.monotonic() - started))
        while remaining > 0.0:
            pause = min(30.0, remaining)
            time.sleep(pause)
            remaining -= pause
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
