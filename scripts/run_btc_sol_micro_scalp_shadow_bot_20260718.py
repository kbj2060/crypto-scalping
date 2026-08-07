"""Non-executing BTC/SOL shadow bot for the frozen ETH-v4 transfer adapters."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_eth_micro_scalp_v4_fresh_forward_observer_20260718 as binding  # noqa: E402
import run_eth_micro_scalp_v4_shadow_bot_20260718 as eth_shadow  # noqa: E402
import train_eval_eth_micro_scalp_source_stable_v4_20260718 as v4  # noqa: E402
import tune_btc_sol_micro_scalp_transfer_adapters_20260718 as tuner  # noqa: E402


transfer = tuner.transfer


ASSETS = {
    asset: {
        **config,
        "artifact": ROOT / f"data/ensemble/{tuner.MODEL_IDS[asset]}/adapter.json",
        "database": ROOT / f"data/live/{asset}_micro_scalp_shadow.duckdb",
        "state": ROOT / f"data/live/{asset}_micro_scalp_shadow_state.json",
    }
    for asset, config in transfer.ASSETS.items()
}
STATE_SCHEMA = "cross_asset_micro_scalp.shadow_bot_step.v1"
SUMMARY_SCHEMA = "cross_asset_micro_scalp.shadow_bot.v1"
OBSERVER_SCHEMA = "cross_asset_micro_scalp.shadow_observer.v1"
FEE_SCENARIOS_BP = transfer.FEE_SCENARIOS_BP
DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_MAX_STREAM_AGE_MINUTES = 10.0


def _utc_naive(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def load_asset_artifact(asset: str, runtime: Any) -> dict[str, Any]:
    path = ASSETS[asset]["artifact"]
    if not path.exists():
        raise FileNotFoundError(path)
    artifact = json.loads(path.read_text())
    required = {
        "schema_version": "cross_asset_micro_scalp.transfer_adapter.v1",
        "asset": asset,
        "model_id": tuner.MODEL_IDS[asset],
        "parent_model_id": v4.MODEL_ID,
        "parent_model_sha256": runtime.model_sha256,
        "parent_weights_frozen": True,
        "training_performed": False,
        "parameter_updates": 0,
        "activation_allowed": False,
        "order_submission_supported": False,
        "fixed_holding_period_used": False,
    }
    mismatches = [name for name, expected in required.items() if artifact.get(name) != expected]
    if mismatches:
        raise RuntimeError(f"{asset} adapter contract mismatch: {mismatches}")
    if bool((artifact.get("artifact_execution_policy") or {}).get("enabled")):
        raise RuntimeError(f"{asset} artifact execution policy must remain disabled")
    fresh_start = _utc_naive(artifact["fresh_forward_start_utc"])
    if fresh_start <= _utc_naive(artifact["split_times"]["development_end"]):
        raise RuntimeError(f"{asset} fresh-forward boundary overlaps development")
    artifact["artifact_sha256"] = transfer._sha256(path)
    artifact["fresh_start"] = fresh_start
    return artifact


def _shadow_runtime(runtime: Any, artifact: dict[str, Any]) -> Any:
    policy = binding.v4.v3.OpportunityPolicy(**artifact["selected_research_policy"])
    if not policy.enabled:
        expert_count = len(runtime.models) * runtime.config.experts
        policy = binding.v4.v3.OpportunityPolicy(
            True, 1_000_000_000.0, expert_count, False, 0.0, expert_count, 0.0
        )
    return replace(runtime, policy=policy)


def _initialize_database(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS observer_metadata (
            singleton BOOLEAN PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            asset VARCHAR NOT NULL,
            model_id VARCHAR NOT NULL,
            model_sha256 VARCHAR NOT NULL,
            parent_model_id VARCHAR NOT NULL,
            parent_model_sha256 VARCHAR NOT NULL,
            fresh_start_utc TIMESTAMP NOT NULL,
            research_policy_enabled BOOLEAN NOT NULL,
            order_submission_supported BOOLEAN NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            timestamp TIMESTAMP PRIMARY KEY,
            model_id VARCHAR NOT NULL,
            model_sha256 VARCHAR NOT NULL,
            feature_hash_sha256 VARCHAR NOT NULL,
            close DOUBLE NOT NULL,
            available BOOLEAN NOT NULL,
            previous_position INTEGER NOT NULL,
            target_position INTEGER NOT NULL,
            position_change INTEGER NOT NULL,
            intent_id VARCHAR,
            intent_side VARCHAR,
            notional_change DOUBLE NOT NULL,
            diagnostics_json VARCHAR NOT NULL,
            execution_evidence_status VARCHAR NOT NULL
        )
        """
    )


def _expected_metadata(asset: str, artifact: dict[str, Any]) -> tuple[Any, ...]:
    return (
        True,
        OBSERVER_SCHEMA,
        asset,
        artifact["model_id"],
        artifact["artifact_sha256"],
        artifact["parent_model_id"],
        artifact["parent_model_sha256"],
        artifact["fresh_start"].to_pydatetime(),
        bool(artifact["selected_research_policy"]["enabled"]),
        False,
    )


def commit_decisions(
    database: Path,
    asset: str,
    artifact: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> int:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    inserted = 0
    try:
        connection.execute("BEGIN TRANSACTION")
        _initialize_database(connection)
        metadata = connection.execute(
            "SELECT * FROM observer_metadata WHERE singleton = true"
        ).fetchone()
        expected = _expected_metadata(asset, artifact)
        if metadata is None:
            connection.execute(
                "INSERT INTO observer_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                expected,
            )
        elif metadata != expected:
            raise RuntimeError(f"{asset} shadow database artifact contract mismatch")
        latest = connection.execute("SELECT max(timestamp) FROM decisions").fetchone()[0]
        latest = _utc_naive(latest) if latest is not None else None
        for row in decisions:
            timestamp = _utc_naive(row["timestamp"])
            if latest is not None and timestamp <= latest:
                continue
            connection.execute(
                """
                INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    timestamp,
                    artifact["model_id"],
                    artifact["artifact_sha256"],
                    row["feature_hash_sha256"],
                    row["close"],
                    row["available"],
                    row["previous_position"],
                    row["target_position"],
                    row["position_change"],
                    row["intent_id"],
                    row["intent_side"],
                    row["notional_change"],
                    json.dumps(row["diagnostics"], sort_keys=True, default=str),
                    "unobserved" if row["intent_id"] else "not_applicable",
                ],
            )
            inserted += 1
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return inserted


def shadow_summary(asset: str, artifact: dict[str, Any]) -> dict[str, Any]:
    database = ASSETS[asset]["database"]
    connection = duckdb.connect(str(database), read_only=True)
    try:
        decisions = int(connection.execute("SELECT count(*) FROM decisions").fetchone()[0])
        latest = connection.execute(
            "SELECT timestamp, target_position, close FROM decisions ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        settled = int(
            connection.execute(
                "SELECT count(DISTINCT decision_timestamp) FROM shadow_pnl"
            ).fetchone()[0]
        )
        scenarios: dict[str, Any] = {}
        for fee in FEE_SCENARIOS_BP:
            rows = connection.execute(
                """
                SELECT gross_return, cost_return, equity
                FROM shadow_pnl WHERE fee_bp = ? ORDER BY decision_timestamp
                """,
                [fee],
            ).fetchall()
            equities = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
            curve = np.r_[1.0, equities]
            peak = np.maximum.accumulate(curve)
            scenarios[f"{fee:.2f}bp_per_notional_change"] = {
                "compounded_return_pct": float((equities[-1] - 1.0) * 100.0) if len(equities) else 0.0,
                "additive_gross_return_pct": float(sum(float(row[0]) for row in rows) * 100.0),
                "additive_cost_pct": float(sum(float(row[1]) for row in rows) * 100.0),
                "max_drawdown_pct": float((1.0 - curve / peak).max() * 100.0),
            }
    finally:
        connection.close()
    return {
        "schema_version": SUMMARY_SCHEMA,
        "asset": asset,
        "symbol": artifact["symbol"],
        "model_id": artifact["model_id"],
        "parent_model_id": artifact["parent_model_id"],
        "decision_count": decisions,
        "settled_intervals": settled,
        "unsettled_decisions": max(0, decisions - settled),
        "latest_decision_utc": str(latest[0]) if latest else None,
        "current_position": int(latest[1]) if latest else 0,
        "latest_close": float(latest[2]) if latest else None,
        "research_policy_enabled": bool(artifact["selected_research_policy"]["enabled"]),
        "unit_notional": 1.0,
        "fee_scenarios": scenarios,
        "evidence_class": "counterfactual completed-close-to-next-completed-close",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "fixed_holding_period_used": False,
        "performance_eligible": False,
        "reason": "fresh shadow diagnostics are not actual execution evidence",
        "order_submission_supported": False,
    }


def _observer_state(database: Path) -> tuple[pd.Timestamp | None, int]:
    if not database.exists():
        return None, 0
    connection = duckdb.connect(str(database), read_only=True)
    try:
        row = connection.execute(
            "SELECT timestamp, target_position FROM decisions ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    return (_utc_naive(row[0]), int(row[1])) if row else (None, 0)


def run_asset(
    asset: str,
    snapshot_path: Path,
    coverage: dict[str, dict[str, Any]],
    runtime: Any,
    artifact: dict[str, Any],
    max_stream_age_minutes: float,
) -> dict[str, Any]:
    database = ASSETS[asset]["database"]
    last_timestamp, previous_position = _observer_state(database)
    if not artifact["selected_research_policy"]["enabled"] and previous_position != 0:
        raise RuntimeError(f"{asset} disabled research policy must remain cash")
    evaluation_end = min(
        coverage["micro"]["end_utc"], coverage["book"]["end_utc"]
    ).floor("min")
    completed_at = evaluation_end + pd.Timedelta(minutes=1)
    age_minutes = max(
        0.0,
        float((pd.Timestamp.utcnow().tz_localize(None) - completed_at).total_seconds()) / 60.0,
    )
    if age_minutes > max_stream_age_minutes:
        raise RuntimeError(f"{asset} microstructure stream is stale: {age_minutes:.3f} minutes")
    decision_start = last_timestamp or artifact["fresh_start"]
    if evaluation_end < decision_start:
        raise RuntimeError(f"{asset} feature stream has not reached its fresh-forward start")
    source_start = decision_start - pd.Timedelta(
        hours=transfer.FEATURE_CONTEXT_HOURS + 1
    )
    frame, source = transfer.build_asset_stream(
        snapshot_path, asset, ASSETS[asset], source_start, evaluation_end
    )
    adapted = tuner.apply_adapter(frame, artifact["adapter"], runtime)
    shadow_runtime = _shadow_runtime(runtime, artifact)
    decisions = binding.observer.build_decisions(
        adapted,
        shadow_runtime,
        previous_position=previous_position,
        after_timestamp=last_timestamp,
        fresh_start=artifact["fresh_start"],
    )
    for row in decisions:
        row["model_id"] = artifact["model_id"]
        row["model_sha256"] = artifact["artifact_sha256"]
    inserted_decisions = commit_decisions(database, asset, artifact, decisions)
    inserted_settlements = eth_shadow.settle_shadow_pnl(database, FEE_SCENARIOS_BP)
    summary = shadow_summary(asset, artifact)
    payload = {
        "schema_version": STATE_SCHEMA,
        "asset": asset,
        "model_id": artifact["model_id"],
        "model_sha256": artifact["artifact_sha256"],
        "parent_model_id": artifact["parent_model_id"],
        "parent_model_sha256": artifact["parent_model_sha256"],
        "research_policy_enabled": bool(artifact["selected_research_policy"]["enabled"]),
        "computed_decisions": len(decisions),
        "inserted_decisions": inserted_decisions,
        "inserted_settlements": inserted_settlements,
        "stream": {
            "latest_feature_timestamp_utc": str(evaluation_end),
            "latest_feature_completed_at_utc": str(completed_at),
            "stream_age_minutes": age_minutes,
            "rows": len(frame),
            "source": source,
        },
        "summary": summary,
        "activation_allowed": False,
        "order_submission_supported": False,
    }
    binding.observer._write_json_atomic(ASSETS[asset]["state"], payload)
    return payload


def run_cycle(runtime: Any, max_stream_age_minutes: float) -> dict[str, Any]:
    artifacts = {asset: load_asset_artifact(asset, runtime) for asset in ASSETS}
    with tempfile.TemporaryDirectory(prefix="btc-sol-scalp-shadow-") as directory:
        snapshot_path = Path(directory) / "microstructure.duckdb"
        snapshot = transfer.snapshot_database(transfer.MICRO_DB, snapshot_path)
        connection = duckdb.connect(str(snapshot_path), read_only=True)
        try:
            coverage = {
                asset: {
                    "micro": transfer.table_coverage(connection, config["micro_table"], "ts"),
                    "book": transfer.table_coverage(
                        connection, config["book_table"], "recorded_at_kst"
                    ),
                }
                for asset, config in ASSETS.items()
            }
        finally:
            connection.close()
        results: dict[str, Any] = {}
        for asset in ASSETS:
            results[asset] = run_asset(
                asset,
                snapshot_path,
                coverage[asset],
                runtime,
                artifacts[asset],
                max_stream_age_minutes,
            )
    return {
        "schema_version": "cross_asset_micro_scalp.shadow_cycle.v1",
        "created_at_utc": str(pd.Timestamp.utcnow()),
        "microstructure_snapshot": snapshot,
        "assets": results,
        "activation_allowed": False,
        "order_submission_supported": False,
    }


def _write_failure_states(error: Exception) -> None:
    for asset, config in ASSETS.items():
        binding.observer._write_json_atomic(
            config["state"],
            {
                "schema_version": STATE_SCHEMA,
                "asset": asset,
                "model_id": tuner.MODEL_IDS[asset],
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
    for command in ("step", "serve"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        subparser.add_argument(
            "--max-stream-age-minutes",
            type=float,
            default=DEFAULT_MAX_STREAM_AGE_MINUTES,
        )
        if command == "serve":
            subparser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
            subparser.add_argument("--max-cycles", type=int, default=0)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--asset", choices=tuple(ASSETS), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = binding.observer.load_runtime(device_name=getattr(args, "device", "cpu"))
    if args.command == "summary":
        print(json.dumps(shadow_summary(args.asset, load_asset_artifact(args.asset, runtime)), indent=2))
        return 0
    if args.max_stream_age_minutes <= 0.0:
        raise ValueError("max_stream_age_minutes must be positive")
    if args.command == "step":
        try:
            payload = run_cycle(runtime, args.max_stream_age_minutes)
        except Exception as error:
            _write_failure_states(error)
            raise
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.interval_seconds < 60 or args.max_cycles < 0:
        raise ValueError("invalid serve cadence")
    cycle = 0
    while args.max_cycles == 0 or cycle < args.max_cycles:
        started = time.monotonic()
        try:
            payload = run_cycle(runtime, args.max_stream_age_minutes)
            print(json.dumps(payload, default=str), flush=True)
        except Exception as error:
            _write_failure_states(error)
            print(json.dumps({"status": "failed_closed", "error": str(error)}), flush=True)
        cycle += 1
        if args.max_cycles and cycle >= args.max_cycles:
            break
        remaining = max(0.0, args.interval_seconds - (time.monotonic() - started))
        while remaining:
            pause = min(30.0, remaining)
            time.sleep(pause)
            remaining -= pause
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
