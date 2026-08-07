"""Non-executing shadow service for selected reusable micro-scalp layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
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

import evaluate_micro_scalp_reuse_layers_20260718 as reuse  # noqa: E402
import run_eth_micro_scalp_v4_fresh_forward_observer_20260718 as binding  # noqa: E402
import run_eth_micro_scalp_v4_shadow_bot_20260718 as eth_shadow  # noqa: E402
import train_eval_eth_micro_scalp_source_stable_v4_20260718 as v4  # noqa: E402


REPORT_PATH = reuse.REPORT_PATH
STATE_SCHEMA = "micro_scalp_reuse.shadow_bot_step.v1"
SUMMARY_SCHEMA = "micro_scalp_reuse.shadow_bot.v1"
OBSERVER_SCHEMA = "micro_scalp_reuse.shadow_observer.v1"
FEE_SCENARIOS_BP = reuse.FEE_SCENARIOS_BP
DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_MAX_STREAM_AGE_MINUTES = 15.0
MODES = {
    "eth_lifecycle": {
        "asset": "eth",
        "dynamic_exit": True,
        "model_id": "eth_micro_scalp_dynamic_lifecycle_shadow_v1_20260718",
        "database": ROOT / "data/live/eth_micro_scalp_lifecycle_shadow.duckdb",
        "state": ROOT / "data/live/eth_micro_scalp_lifecycle_shadow_state.json",
    },
    "sol_entry": {
        "asset": "sol",
        "dynamic_exit": False,
        "model_id": "sol_micro_scalp_entry_only_shadow_v1_20260718",
        "database": ROOT / "data/live/sol_micro_scalp_entry_shadow.duckdb",
        "state": ROOT / "data/live/sol_micro_scalp_entry_shadow_state.json",
    },
}


def _utc_naive(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(runtime: Any) -> dict[str, Any]:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(REPORT_PATH)
    report = json.loads(REPORT_PATH.read_text())
    required = {
        "schema_version": "micro_scalp.reuse_layers_test.v1",
        "parent_model_id": v4.MODEL_ID,
        "parent_model_sha256": runtime.model_sha256,
        "selection_uses_only_tune_split": True,
        "validation_used_for_selection": False,
        "development_used_for_selection": False,
        "fresh_shadow_used_for_selection": False,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "fixed_holding_period_used": False,
        "activation_allowed": False,
        "order_submission_supported": False,
    }
    mismatches = [name for name, expected in required.items() if report.get(name) != expected]
    if mismatches:
        raise RuntimeError(f"reuse report contract mismatch: {mismatches}")
    report_hash = _sha256(REPORT_PATH)
    fresh_start = _utc_naive(report["common_interval"]["end_utc"]) + pd.Timedelta(minutes=1)
    modes: dict[str, Any] = {}
    for mode, config in MODES.items():
        asset = config["asset"]
        policy = reuse.LifecyclePolicy(**report["assets"][asset]["selected_policy"])
        if policy.entry_margin_bp >= 1e8:
            raise RuntimeError(f"selected {mode} policy is cash-only")
        adapter = None
        adapter_hash = None
        if asset in reuse.ADAPTER_PATHS:
            path = reuse.ADAPTER_PATHS[asset]
            adapter = json.loads(path.read_text())
            adapter_hash = _sha256(path)
            if adapter["asset"] != asset or adapter["parent_model_sha256"] != runtime.model_sha256:
                raise RuntimeError(f"{mode} adapter contract mismatch")
        identity = hashlib.sha256(
            f"{mode}|{report_hash}|{adapter_hash or 'native'}".encode()
        ).hexdigest()
        modes[mode] = {
            **config,
            "policy": policy,
            "detector": report["assets"][asset]["risk_detector"],
            "adapter": adapter,
            "adapter_sha256": adapter_hash,
            "model_sha256": identity,
            "fresh_start": fresh_start,
        }
    return {"report": report, "report_sha256": report_hash, "modes": modes}


def _initialize_database(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS observer_metadata (
            singleton BOOLEAN PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            mode VARCHAR NOT NULL,
            asset VARCHAR NOT NULL,
            model_id VARCHAR NOT NULL,
            model_sha256 VARCHAR NOT NULL,
            parent_model_id VARCHAR NOT NULL,
            parent_model_sha256 VARCHAR NOT NULL,
            policy_report_sha256 VARCHAR NOT NULL,
            fresh_start_utc TIMESTAMP NOT NULL,
            dynamic_exit BOOLEAN NOT NULL,
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
            execution_evidence_status VARCHAR NOT NULL,
            risk_score DOUBLE NOT NULL,
            high_risk BOOLEAN NOT NULL,
            dynamic_exit BOOLEAN NOT NULL
        )
        """
    )


def _expected_metadata(
    mode: str, config: dict[str, Any], contract: dict[str, Any], runtime: Any
) -> tuple[Any, ...]:
    return (
        True,
        OBSERVER_SCHEMA,
        mode,
        config["asset"],
        config["model_id"],
        config["model_sha256"],
        v4.MODEL_ID,
        runtime.model_sha256,
        contract["report_sha256"],
        config["fresh_start"].to_pydatetime(),
        config["dynamic_exit"],
        False,
    )


def observer_state(database: Path) -> tuple[pd.Timestamp | None, int]:
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


def commit_decisions(
    database: Path,
    mode: str,
    config: dict[str, Any],
    contract: dict[str, Any],
    runtime: Any,
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
        expected = _expected_metadata(mode, config, contract, runtime)
        if metadata is None:
            connection.execute(
                "INSERT INTO observer_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                expected,
            )
        elif metadata != expected:
            raise RuntimeError(f"{mode} shadow database contract mismatch")
        latest = connection.execute("SELECT max(timestamp) FROM decisions").fetchone()[0]
        latest = _utc_naive(latest) if latest is not None else None
        for row in decisions:
            timestamp = _utc_naive(row["timestamp"])
            if latest is not None and timestamp <= latest:
                continue
            connection.execute(
                """
                INSERT INTO decisions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    timestamp,
                    config["model_id"],
                    config["model_sha256"],
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
                    row["risk_score"],
                    row["high_risk"],
                    config["dynamic_exit"],
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


def shadow_summary(mode: str, config: dict[str, Any]) -> dict[str, Any]:
    database = config["database"]
    connection = duckdb.connect(str(database), read_only=True)
    try:
        decisions = int(connection.execute("SELECT count(*) FROM decisions").fetchone()[0])
        latest = connection.execute(
            "SELECT timestamp, target_position, close FROM decisions ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        high_risk_bars = int(
            connection.execute("SELECT count(*) FROM decisions WHERE high_risk").fetchone()[0]
        )
        settled = int(
            connection.execute("SELECT count(DISTINCT decision_timestamp) FROM shadow_pnl").fetchone()[0]
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
        "mode": mode,
        "asset": config["asset"],
        "symbol": reuse.ASSET_CONFIG[config["asset"]]["symbol"],
        "model_id": config["model_id"],
        "parent_model_id": v4.MODEL_ID,
        "decision_count": decisions,
        "settled_intervals": settled,
        "unsettled_decisions": max(0, decisions - settled),
        "latest_decision_utc": str(latest[0]) if latest else None,
        "current_position": int(latest[1]) if latest else 0,
        "latest_close": float(latest[2]) if latest else None,
        "high_risk_bars": high_risk_bars,
        "dynamic_exit_enabled": config["dynamic_exit"],
        "unit_notional": 1.0,
        "fee_scenarios": scenarios,
        "evidence_class": "counterfactual completed-close-to-next-completed-close",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "fixed_holding_period_used": False,
        "performance_eligible": False,
        "order_submission_supported": False,
    }


def run_mode(
    mode: str,
    config: dict[str, Any],
    contract: dict[str, Any],
    runtime: Any,
    snapshot_path: Path,
    coverage: dict[str, Any],
    max_stream_age_minutes: float,
) -> dict[str, Any]:
    last_timestamp, previous_position = observer_state(config["database"])
    evaluation_end = min(
        coverage["micro"]["end_utc"], coverage["book"]["end_utc"]
    ).floor("min")
    completed_at = evaluation_end + pd.Timedelta(minutes=1)
    age_minutes = max(
        0.0,
        float((pd.Timestamp.utcnow().tz_localize(None) - completed_at).total_seconds()) / 60.0,
    )
    if age_minutes > max_stream_age_minutes:
        raise RuntimeError(f"{mode} stream is stale: {age_minutes:.3f} minutes")
    decision_start = last_timestamp or config["fresh_start"]
    if evaluation_end < decision_start:
        raise RuntimeError(f"{mode} stream has not reached fresh start")
    source_start = decision_start - pd.Timedelta(
        hours=reuse.transfer.FEATURE_CONTEXT_HOURS + 1
    )
    asset = config["asset"]
    frame, source = reuse.transfer.build_asset_stream(
        snapshot_path,
        asset,
        reuse.ASSET_CONFIG[asset],
        source_start,
        evaluation_end,
    )
    data = reuse.prepare_asset(
        asset,
        frame,
        runtime,
        config["adapter"],
        require_next_return=False,
    )
    reuse.apply_risk_detector(data, config["detector"])
    mask = data["timestamps"] >= config["fresh_start"]
    if last_timestamp is not None:
        mask &= data["timestamps"] > last_timestamp
    new_data = reuse.slice_data(data, mask)
    positions, counters, _ = reuse.lifecycle_positions(
        new_data,
        config["policy"],
        dynamic_exit=config["dynamic_exit"],
        initial_position=previous_position,
    )
    decisions: list[dict[str, Any]] = []
    current = previous_position
    for index, target_value in enumerate(positions):
        target = int(target_value)
        change = target - current
        timestamp = _utc_naive(new_data["timestamps"][index])
        intent_id = (
            hashlib.sha256(
                f"{config['model_sha256']}|{timestamp}|{current}|{target}".encode()
            ).hexdigest()
            if change
            else None
        )
        decisions.append(
            {
                "timestamp": timestamp,
                "feature_hash_sha256": str(new_data["feature_hash"][index]),
                "close": float(new_data["close"][index]),
                "available": bool(new_data["available"][index]),
                "previous_position": current,
                "target_position": target,
                "position_change": change,
                "intent_id": intent_id,
                "intent_side": "BUY" if change > 0 else "SELL" if change < 0 else None,
                "notional_change": abs(float(change)),
                "risk_score": float(new_data["risk_score"][index]),
                "high_risk": bool(new_data["high_risk"][index]),
                "diagnostics": {
                    "mode": mode,
                    "dynamic_exit": config["dynamic_exit"],
                    "parent_desired": int(new_data["desired"][index]),
                    "risk_score": float(new_data["risk_score"][index]),
                    "high_risk": bool(new_data["high_risk"][index]),
                },
            }
        )
        current = target
    inserted_decisions = commit_decisions(
        config["database"], mode, config, contract, runtime, decisions
    )
    inserted_settlements = eth_shadow.settle_shadow_pnl(
        config["database"], FEE_SCENARIOS_BP
    )
    summary = shadow_summary(mode, config)
    payload = {
        "schema_version": STATE_SCHEMA,
        "mode": mode,
        "asset": asset,
        "model_id": config["model_id"],
        "model_sha256": config["model_sha256"],
        "parent_model_id": v4.MODEL_ID,
        "parent_model_sha256": runtime.model_sha256,
        "policy_report_sha256": contract["report_sha256"],
        "fresh_start_utc": str(config["fresh_start"]),
        "selected_policy": reuse.asdict(config["policy"]),
        "dynamic_exit_enabled": config["dynamic_exit"],
        "computed_decisions": len(decisions),
        "inserted_decisions": inserted_decisions,
        "inserted_settlements": inserted_settlements,
        "decision_counters": counters,
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
    binding.observer._write_json_atomic(config["state"], payload)
    return payload


def run_cycle(runtime: Any, max_stream_age_minutes: float) -> dict[str, Any]:
    contract = load_contract(runtime)
    with tempfile.TemporaryDirectory(prefix="micro-scalp-reuse-shadow-") as directory:
        snapshot_path = Path(directory) / "microstructure.duckdb"
        snapshot = reuse.transfer.snapshot_database(reuse.transfer.MICRO_DB, snapshot_path)
        connection = duckdb.connect(str(snapshot_path), read_only=True)
        try:
            coverage = {
                mode: {
                    "micro": reuse.transfer.table_coverage(
                        connection,
                        reuse.ASSET_CONFIG[config["asset"]]["micro_table"],
                        "ts",
                    ),
                    "book": reuse.transfer.table_coverage(
                        connection,
                        reuse.ASSET_CONFIG[config["asset"]]["book_table"],
                        "recorded_at_kst",
                    ),
                }
                for mode, config in contract["modes"].items()
            }
        finally:
            connection.close()
        results = {
            mode: run_mode(
                mode,
                config,
                contract,
                runtime,
                snapshot_path,
                coverage[mode],
                max_stream_age_minutes,
            )
            for mode, config in contract["modes"].items()
        }
    return {
        "schema_version": "micro_scalp_reuse.shadow_cycle.v1",
        "created_at_utc": str(pd.Timestamp.utcnow()),
        "microstructure_snapshot": snapshot,
        "modes": results,
        "activation_allowed": False,
        "order_submission_supported": False,
    }


def _write_failure_states(error: Exception) -> None:
    for mode, config in MODES.items():
        binding.observer._write_json_atomic(
            config["state"],
            {
                "schema_version": STATE_SCHEMA,
                "mode": mode,
                "asset": config["asset"],
                "model_id": config["model_id"],
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
        child = subparsers.add_parser(command)
        child.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        child.add_argument(
            "--max-stream-age-minutes",
            type=float,
            default=DEFAULT_MAX_STREAM_AGE_MINUTES,
        )
        if command == "serve":
            child.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
            child.add_argument("--max-cycles", type=int, default=0)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--mode", choices=tuple(MODES), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = binding.observer.load_runtime(device_name=getattr(args, "device", "cpu"))
    contract = load_contract(runtime)
    if args.command == "summary":
        print(json.dumps(shadow_summary(args.mode, contract["modes"][args.mode]), indent=2))
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
            print(json.dumps(run_cycle(runtime, args.max_stream_age_minutes), default=str), flush=True)
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
