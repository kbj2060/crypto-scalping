"""Non-executing fresh-forward observer for ETH Opportunity-MoE v3.

This process can read an exact one-minute feature stream, compute the frozen
research policy, and store decisions in its own DuckDB.  It cannot submit or
simulate orders.  Execution observations are separate, provenance-tagged input;
counterfactual order-book observations never become performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_eval_eth_micro_scalp_opportunity_moe_20260718 as v3  # noqa: E402


MODEL_PATH = v3.MODEL_PATH
MODEL_ID = v3.MODEL_ID
FRESH_START_UTC = pd.Timestamp("2026-07-17 16:35:00")
DEFAULT_FEATURE_STREAM = ROOT / "data/live/eth_micro_scalp_v3_features_1m.csv"
OBSERVER_DIR = v3.ARTIFACT_DIR / "fresh_forward_observer"
DEFAULT_OBSERVER_DB = OBSERVER_DIR / "observer.duckdb"
DEFAULT_READINESS_REPORT = OBSERVER_DIR / "readiness.json"
FEATURE_BUILD_REPORT = OBSERVER_DIR / "feature_stream_build.json"
TRAINING_FEATURES = ROOT / "data/training_features_1m.csv"
LIVE_FRAME_SNAPSHOT = ROOT / "data/live/decision_feature_frame_snapshot.pkl.gz"
MICRO_DB = ROOT / "data/live/microstructure.duckdb"

POSITION_INDEX = {-1: 0, 0: 1, 1: 2}
OBSERVATION_TYPES = {"actual_exchange_fill", "orderbook_counterfactual"}
EXECUTION_STATUSES = {"not_submitted", "submitted", "partial", "filled", "canceled", "rejected"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _normalize_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _last_nonempty_lines(path: Path, count: int = 2) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    block_size = 8192
    with path.open("rb") as handle:
        position = handle.seek(0, os.SEEK_END)
        data = b""
        while position > 0 and data.count(b"\n") <= count:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            data = handle.read(read_size) + data
    lines = [line.decode("utf-8") for line in data.splitlines() if line.strip()]
    return lines[-count:]


def _csv_tail_timestamps(path: Path) -> list[pd.Timestamp]:
    timestamps: list[pd.Timestamp] = []
    for line in _last_nonempty_lines(path, 2):
        first = line.split(",", 1)[0]
        try:
            timestamps.append(_normalize_timestamp(first))
        except Exception:
            continue
    return timestamps


def audit_sources(
    feature_stream: Path = DEFAULT_FEATURE_STREAM,
    training_features: Path = TRAINING_FEATURES,
    live_snapshot: Path = LIVE_FRAME_SNAPSHOT,
    micro_db: Path = MICRO_DB,
) -> dict[str, Any]:
    required_columns = ["timestamp", "close", *v3.core.BASE_FEATURES, *v3.core.MICRO_FEATURES]
    training_tail = _csv_tail_timestamps(training_features)
    training_latest = training_tail[-1] if training_tail else None
    training_cadence = training_tail[-1] - training_tail[-2] if len(training_tail) == 2 else None

    snapshot_latest: pd.Timestamp | None = None
    snapshot_cadence: pd.Timedelta | None = None
    snapshot_rows = 0
    if live_snapshot.exists():
        payload = pd.read_pickle(live_snapshot)
        frame = payload.get("frame") if isinstance(payload, dict) else None
        if isinstance(frame, pd.DataFrame) and "timestamp" in frame and len(frame):
            timestamps = pd.to_datetime(frame["timestamp"])
            snapshot_rows = len(timestamps)
            snapshot_latest = _normalize_timestamp(timestamps.iloc[-1])
            differences = timestamps.diff().dropna()
            if len(differences):
                modes = differences.mode()
                snapshot_cadence = modes.iloc[0] if len(modes) else differences.iloc[-1]

    micro_latest: pd.Timestamp | None = None
    book_latest: pd.Timestamp | None = None
    if micro_db.exists():
        connection = duckdb.connect(str(micro_db), read_only=True)
        try:
            micro_value = connection.execute(
                "SELECT max(timezone('UTC', ts)) FROM microstructure_1m"
            ).fetchone()[0]
            book_value = connection.execute(
                "SELECT max(timezone('UTC', recorded_at_kst)) FROM orderbook_decision_snapshots"
            ).fetchone()[0]
        finally:
            connection.close()
        if micro_value is not None:
            micro_latest = _normalize_timestamp(micro_value)
        if book_value is not None:
            book_latest = _normalize_timestamp(book_value)

    stream_status: dict[str, Any] = {"exists": feature_stream.exists()}
    if feature_stream.exists():
        try:
            stream = load_feature_stream(feature_stream)
            stream_status.update(
                {
                    "valid": True,
                    "rows": len(stream),
                    "latest_utc": str(stream["timestamp"].iloc[-1]),
                    "covers_fresh_start": bool(stream["timestamp"].iloc[-1] >= FRESH_START_UTC),
                }
            )
        except Exception as error:
            stream_status.update({"valid": False, "error": str(error)})
    feature_build: dict[str, Any] | None = None
    if FEATURE_BUILD_REPORT.exists():
        feature_build = json.loads(FEATURE_BUILD_REPORT.read_text())

    ready = bool(
        stream_status.get("valid")
        and stream_status.get("covers_fresh_start")
        and micro_latest is not None
        and micro_latest >= FRESH_START_UTC
        and book_latest is not None
        and book_latest >= FRESH_START_UTC
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not stream_status.get("exists"):
        blockers.append("exact one-minute v3 feature stream is missing")
    elif not stream_status.get("valid"):
        blockers.append("v3 feature stream violates its strict schema/cadence contract")
    elif not stream_status.get("covers_fresh_start"):
        blockers.append("v3 feature stream does not reach the post-freeze interval")
    if feature_build is not None and not bool((feature_build.get("parity") or {}).get("pass")):
        blockers.append("public-source feature reconstruction failed frozen-cache parity")
    if snapshot_cadence is not None and snapshot_cadence != pd.Timedelta(minutes=1):
        warnings.append(f"legacy live decision frame cadence is {snapshot_cadence}, not one minute")
    if training_latest is None or training_latest < FRESH_START_UTC:
        warnings.append("frozen one-minute training feature file ends before the fresh interval")
    if micro_latest is None or micro_latest < FRESH_START_UTC:
        blockers.append("microstructure stream does not reach the fresh interval")
    if book_latest is None or book_latest < FRESH_START_UTC:
        blockers.append("order-book stream does not reach the fresh interval")
    return {
        "schema_version": "eth_micro_scalp_v3.observer_readiness.v1",
        "model_id": MODEL_ID,
        "model_sha256": _sha256(MODEL_PATH),
        "fresh_start_utc": str(FRESH_START_UTC),
        "ready": ready,
        "blockers": blockers,
        "warnings": warnings,
        "required_feature_columns": required_columns,
        "sources": {
            "feature_stream": {"path": str(feature_stream), **stream_status},
            "feature_stream_build": {
                "path": str(FEATURE_BUILD_REPORT),
                "published": feature_build.get("published") if feature_build else None,
                "parity": feature_build.get("parity") if feature_build else None,
                "stream_contract_pass": (
                    feature_build.get("stream_contract_pass")
                    if feature_build and "stream_contract_pass" in feature_build
                    else (feature_build.get("stream_contract") or {}).get("pass")
                    if feature_build else None
                ),
            },
            "training_features": {
                "path": str(training_features),
                "latest_utc": str(training_latest) if training_latest is not None else None,
                "tail_cadence": str(training_cadence) if training_cadence is not None else None,
            },
            "live_decision_frame_snapshot": {
                "path": str(live_snapshot),
                "rows": snapshot_rows,
                "latest_utc": str(snapshot_latest) if snapshot_latest is not None else None,
                "modal_cadence": str(snapshot_cadence) if snapshot_cadence is not None else None,
                "eligible_as_v3_input": snapshot_cadence == pd.Timedelta(minutes=1),
            },
            "microstructure": {
                "path": str(micro_db),
                "latest_utc": str(micro_latest) if micro_latest is not None else None,
            },
            "order_book": {
                "path": str(micro_db),
                "latest_utc": str(book_latest) if book_latest is not None else None,
            },
        },
        "execution_capability": {
            "order_submission_supported": False,
            "simulated_fill_supported": False,
            "external_observation_ingest_only": True,
        },
    }


def load_feature_stream(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"feature stream does not exist: {path}")
    frame = pd.read_csv(path)
    required = ["timestamp", "close", *v3.core.BASE_FEATURES, *v3.core.MICRO_FEATURES]
    missing = [name for name in required if name not in frame.columns]
    unexpected_model_aliases = [name for name in frame.columns if name.startswith("legacy_")]
    if missing or unexpected_model_aliases:
        raise RuntimeError(
            f"feature stream contract mismatch: missing={missing}, forbidden={unexpected_model_aliases}"
        )
    frame = frame[required].copy()
    frame["timestamp"] = frame["timestamp"].map(_normalize_timestamp)
    if frame["timestamp"].duplicated().any():
        raise RuntimeError("feature stream contains duplicate timestamps")
    if not frame["timestamp"].is_monotonic_increasing:
        raise RuntimeError("feature stream timestamps are not increasing")
    differences = frame["timestamp"].diff().dropna()
    invalid = differences != pd.Timedelta(minutes=1)
    if invalid.any():
        first = int(np.flatnonzero(invalid.to_numpy())[0]) + 1
        raise RuntimeError(
            f"feature stream cadence violation at row {first}: delta={differences.iloc[first - 1]}"
        )
    numeric = frame.drop(columns="timestamp").apply(pd.to_numeric, errors="coerce")
    numeric_values = numeric.to_numpy(dtype=np.float64)
    if np.isinf(numeric_values).any():
        raise RuntimeError("feature stream contains infinite model inputs")
    always_required = ["close", *v3.core.BASE_FEATURES, "micro_available", "book_available"]
    if not np.isfinite(numeric[always_required].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("feature stream contains non-finite required inputs")
    micro_payload = [name for name in v3.core.MICRO_FEATURES if name.startswith("micro_")]
    book_payload = [name for name in v3.core.MICRO_FEATURES if name.startswith("book_")]
    micro_present = numeric["micro_available"] > 0.5
    book_present = numeric["book_available"] > 0.5
    if micro_present.any() and not np.isfinite(
        numeric.loc[micro_present, micro_payload].to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("available microstructure rows contain non-finite inputs")
    if book_present.any() and not np.isfinite(
        numeric.loc[book_present, book_payload].to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("available order-book rows contain non-finite inputs")
    frame[numeric.columns] = numeric
    return frame


@dataclass
class Runtime:
    checkpoint: dict[str, Any]
    config: v3.OpportunityConfig
    policy: v3.OpportunityPolicy
    models: list[v3.OpportunityCostMoE]
    device: torch.device
    model_sha256: str


def load_runtime(model_path: Path | None = None, device_name: str | None = None) -> Runtime:
    model_path = MODEL_PATH if model_path is None else model_path
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_id") != MODEL_ID:
        raise RuntimeError(f"model id mismatch: {checkpoint.get('model_id')}")
    if checkpoint.get("activation_allowed") is not False:
        raise RuntimeError("observer requires a research-only, activation-blocked artifact")
    if bool((checkpoint.get("policy") or {}).get("enabled")):
        raise RuntimeError("artifact execution policy must remain disabled")
    if checkpoint.get("trainer_script_sha256") != _sha256(Path(v3.__file__)):
        raise RuntimeError("v3 trainer script hash mismatch")
    if tuple(checkpoint.get("base_feature_names", ())) != tuple(v3.core.BASE_FEATURES):
        raise RuntimeError("base feature contract mismatch")
    if tuple(checkpoint.get("micro_feature_names", ())) != tuple(v3.core.MICRO_FEATURES):
        raise RuntimeError("micro feature contract mismatch")
    policy = v3.OpportunityPolicy(**checkpoint["selected_research_policy"])
    if not policy.enabled:
        raise RuntimeError("selected research policy is disabled")
    config = v3.OpportunityConfig(**checkpoint["config"])
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    scalers = checkpoint.get("scalers") or {}
    for name, width in (
        ("base_center", len(v3.core.BASE_FEATURES)),
        ("base_scale", len(v3.core.BASE_FEATURES)),
        ("micro_center", len(v3.core.MICRO_FEATURES)),
        ("micro_scale", len(v3.core.MICRO_FEATURES)),
    ):
        if np.asarray(scalers.get(name)).shape != (width,):
            raise RuntimeError(f"scaler contract mismatch: {name}")
    models: list[v3.OpportunityCostMoE] = []
    runtime_seeds = checkpoint.get("selected_ensemble_seeds", checkpoint["seeds"])
    for seed in runtime_seeds:
        state = checkpoint["seed_model_states"][str(seed)]
        n_auxiliary_outputs = int(state["auxiliary_head.2.weight"].shape[0])
        model = v3.OpportunityCostMoE(
            len(v3.core.BASE_FEATURES), len(v3.core.MICRO_FEATURES),
            n_auxiliary_outputs, config,
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        models.append(model)
    return Runtime(checkpoint, config, policy, models, device, _sha256(model_path))


def _available(frame: pd.DataFrame) -> np.ndarray:
    return (
        (frame["micro_available"].to_numpy(dtype=float) > 0.5)
        & (frame["micro_data_stale"].to_numpy(dtype=float) < 0.5)
        & (frame["micro_depth_connected"].to_numpy(dtype=float) > 0.5)
        & (frame["micro_warmup_30m_ready"].to_numpy(dtype=float) > 0.5)
        & (frame["micro_age_min"].to_numpy(dtype=float) >= 0.0)
        & (frame["micro_age_min"].to_numpy(dtype=float) <= 2.0)
    )


def infer_stream(frame: pd.DataFrame, runtime: Runtime) -> tuple[dict[str, np.ndarray], np.ndarray]:
    scalers = runtime.checkpoint["scalers"]
    base = v3.core.apply_scaler(
        frame[list(v3.core.BASE_FEATURES)].to_numpy(dtype=np.float32),
        np.asarray(scalers["base_center"]), np.asarray(scalers["base_scale"]),
    )
    micro = v3.core.apply_scaler(
        frame[list(v3.core.MICRO_FEATURES)].to_numpy(dtype=np.float32),
        np.asarray(scalers["micro_center"]), np.asarray(scalers["micro_scale"]),
    )
    if len(frame) < runtime.config.window:
        raise RuntimeError(
            f"feature stream needs at least {runtime.config.window} consecutive rows, got {len(frame)}"
        )
    end_indices = np.arange(runtime.config.window - 1, len(frame), dtype=np.int64)
    rows = [
        v3.infer(model, base, micro, end_indices, runtime.config, runtime.device)
        for model in runtime.models
    ]
    return v3.aggregate_seed_predictions(rows), end_indices


def decide_next(
    prediction: dict[str, np.ndarray],
    row_index: int,
    usable: bool,
    previous_position: int,
    policy: v3.OpportunityPolicy,
) -> tuple[int, dict[str, Any]]:
    if previous_position not in POSITION_INDEX:
        raise ValueError(f"invalid previous position: {previous_position}")
    previous_idx = POSITION_INDEX[previous_position]
    q_values = np.asarray(prediction["q"][row_index], dtype=np.float64)
    expert_q = np.asarray(prediction["expert_q"][row_index], dtype=np.float64)
    continuation = np.asarray(prediction["continuation"][row_index], dtype=np.float64)
    expert_continuation = np.asarray(
        prediction["expert_continuation"][row_index], dtype=np.float64
    )
    opportunity_exit = False
    switch_agreement = 0
    exit_agreement = 0
    if not usable or not np.isfinite(q_values).all():
        action_idx = 1
        state_q = np.full(3, np.nan)
    else:
        expert_state_q = expert_q[:, previous_idx]
        state_q = (
            q_values[previous_idx]
            - policy.uncertainty_penalty * np.std(expert_state_q, axis=0)
        )
        action_idx = int(np.argmax(state_q))
        improvement = float(state_q[action_idx] - state_q[previous_idx])
        if action_idx != previous_idx and improvement < policy.switch_margin_bp:
            action_idx = previous_idx
        if action_idx != previous_idx:
            votes = np.argmax(expert_state_q, axis=1)
            switch_agreement = int(np.sum(votes == action_idx))
            if switch_agreement < policy.min_switch_agreement:
                action_idx = previous_idx
        if (
            policy.exit_overlay_enabled
            and previous_idx != 1
            and action_idx == previous_idx
            and np.isfinite(continuation[previous_idx])
            and continuation[previous_idx] < policy.continuation_floor_bp
        ):
            exit_agreement = int(
                np.sum(expert_continuation[:, previous_idx] < policy.continuation_floor_bp)
            )
            if exit_agreement >= policy.min_exit_agreement:
                alternatives = [candidate for candidate in range(3) if candidate != previous_idx]
                action_idx = alternatives[int(np.argmax(state_q[alternatives]))]
                opportunity_exit = True
    target = int(v3.core.ACTIONS[action_idx])
    diagnostics = {
        "usable": bool(usable),
        "previous_position": previous_position,
        "target_position": target,
        "state_q": state_q,
        "switch_agreement": switch_agreement,
        "opportunity_exit": opportunity_exit,
        "exit_agreement": exit_agreement,
        "continuation_advantage_bp": float(continuation[previous_idx]),
    }
    return target, diagnostics


def build_decisions(
    frame: pd.DataFrame,
    runtime: Runtime,
    previous_position: int = 0,
    after_timestamp: pd.Timestamp | None = None,
    fresh_start: pd.Timestamp = FRESH_START_UTC,
) -> list[dict[str, Any]]:
    prediction, end_indices = infer_stream(frame, runtime)
    usable = _available(frame)
    decisions: list[dict[str, Any]] = []
    current = int(previous_position)
    after = _normalize_timestamp(after_timestamp) if after_timestamp is not None else None
    for prediction_index, frame_index in enumerate(end_indices):
        timestamp = _normalize_timestamp(frame["timestamp"].iloc[frame_index])
        if timestamp < fresh_start or (after is not None and timestamp <= after):
            continue
        target, diagnostics = decide_next(
            prediction, prediction_index, bool(usable[frame_index]), current, runtime.policy
        )
        feature_values = np.r_[
            frame[list(v3.core.BASE_FEATURES)].iloc[frame_index].to_numpy(dtype=np.float64),
            frame[list(v3.core.MICRO_FEATURES)].iloc[frame_index].to_numpy(dtype=np.float64),
        ]
        feature_hash = hashlib.sha256(feature_values.tobytes()).hexdigest()
        change = target - current
        intent_id = None
        if change:
            intent_payload = f"{runtime.model_sha256}|{timestamp}|{current}|{target}"
            intent_id = hashlib.sha256(intent_payload.encode()).hexdigest()
        decisions.append(
            {
                "timestamp": timestamp,
                "model_id": MODEL_ID,
                "model_sha256": runtime.model_sha256,
                "feature_hash_sha256": feature_hash,
                "close": float(frame["close"].iloc[frame_index]),
                "available": bool(usable[frame_index]),
                "previous_position": current,
                "target_position": target,
                "position_change": change,
                "intent_id": intent_id,
                "intent_side": "BUY" if change > 0 else "SELL" if change < 0 else None,
                "notional_change": abs(float(change)),
                "diagnostics": diagnostics,
            }
        )
        current = target
    return decisions


def _initialize_db(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS observer_metadata (
            singleton BOOLEAN PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            model_id VARCHAR NOT NULL,
            model_sha256 VARCHAR NOT NULL,
            fresh_start_utc TIMESTAMP NOT NULL,
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
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_observations (
            decision_timestamp TIMESTAMP NOT NULL,
            observation_type VARCHAR NOT NULL,
            observation_id VARCHAR NOT NULL,
            observed_at_utc TIMESTAMP NOT NULL,
            execution_status VARCHAR NOT NULL,
            order_id VARCHAR,
            requested_quantity DOUBLE,
            filled_quantity DOUBLE,
            average_fill_price DOUBLE,
            fee_paid DOUBLE,
            queue_ahead_quantity DOUBLE,
            source VARCHAR NOT NULL,
            payload_json VARCHAR NOT NULL,
            performance_eligible BOOLEAN NOT NULL,
            PRIMARY KEY (decision_timestamp, observation_type, observation_id)
        )
        """
    )


def _check_metadata(connection: duckdb.DuckDBPyConnection, runtime: Runtime) -> None:
    row = connection.execute("SELECT * FROM observer_metadata WHERE singleton = true").fetchone()
    expected = (
        True, "eth_micro_scalp_v3.fresh_forward_observer.v1", MODEL_ID,
        runtime.model_sha256, FRESH_START_UTC.to_pydatetime(), False,
    )
    if row is None:
        connection.execute("INSERT INTO observer_metadata VALUES (?, ?, ?, ?, ?, ?)", expected)
        return
    if row[1] != expected[1] or row[2] != expected[2] or row[3] != expected[3]:
        raise RuntimeError("observer database artifact contract mismatch")
    if _normalize_timestamp(row[4]) != FRESH_START_UTC or bool(row[5]):
        raise RuntimeError("observer database safety contract mismatch")


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
    if row is None:
        return None, 0
    return _normalize_timestamp(row[0]), int(row[1])


def commit_decisions(database: Path, runtime: Runtime, decisions: list[dict[str, Any]]) -> int:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    inserted = 0
    try:
        connection.execute("BEGIN TRANSACTION")
        _initialize_db(connection)
        _check_metadata(connection, runtime)
        latest = connection.execute("SELECT max(timestamp) FROM decisions").fetchone()[0]
        latest_timestamp = _normalize_timestamp(latest) if latest is not None else None
        new_rows = [
            row for row in decisions
            if latest_timestamp is None or _normalize_timestamp(row["timestamp"]) > latest_timestamp
        ]
        for row in new_rows:
            connection.execute(
                """
                INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    row["timestamp"], row["model_id"], row["model_sha256"],
                    row["feature_hash_sha256"], row["close"], row["available"],
                    row["previous_position"], row["target_position"], row["position_change"],
                    row["intent_id"], row["intent_side"], row["notional_change"],
                    json.dumps(row["diagnostics"], sort_keys=True, default=_json_default),
                    "unobserved" if row["intent_id"] else "not_applicable",
                ],
            )
        inserted = len(new_rows)
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return inserted


def validate_execution_observation(payload: dict[str, Any]) -> dict[str, Any]:
    required = ("decision_timestamp", "observation_type", "observation_id", "observed_at_utc", "execution_status", "source")
    missing = [name for name in required if payload.get(name) in (None, "")]
    if missing:
        raise ValueError(f"execution observation missing fields: {missing}")
    observation_type = str(payload["observation_type"])
    status = str(payload["execution_status"])
    if observation_type not in OBSERVATION_TYPES:
        raise ValueError(f"invalid observation_type: {observation_type}")
    if status not in EXECUTION_STATUSES:
        raise ValueError(f"invalid execution_status: {status}")
    normalized = dict(payload)
    normalized["decision_timestamp"] = _normalize_timestamp(payload["decision_timestamp"])
    normalized["observed_at_utc"] = _normalize_timestamp(payload["observed_at_utc"])
    for name in ("requested_quantity", "filled_quantity", "average_fill_price", "fee_paid", "queue_ahead_quantity"):
        value = payload.get(name)
        normalized[name] = None if value is None else float(value)
        if normalized[name] is not None and not math.isfinite(normalized[name]):
            raise ValueError(f"non-finite execution field: {name}")
    if observation_type == "actual_exchange_fill":
        if not payload.get("order_id"):
            raise ValueError("actual exchange observation requires order_id")
        requested = normalized["requested_quantity"]
        filled = normalized["filled_quantity"]
        if requested is None or requested <= 0.0 or filled is None or not (0.0 <= filled <= requested):
            raise ValueError("invalid actual requested/filled quantity")
        if filled > 0.0 and (normalized["average_fill_price"] is None or normalized["average_fill_price"] <= 0.0):
            raise ValueError("positive actual fill requires average_fill_price")
        if status == "filled" and not math.isclose(filled, requested, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("filled status requires full requested quantity")
        if status == "partial" and not (0.0 < filled < requested):
            raise ValueError("partial status requires a non-zero incomplete fill")
        if status in {"not_submitted", "submitted", "rejected"} and filled != 0.0:
            raise ValueError(f"{status} status requires zero filled quantity")
        if status == "canceled" and math.isclose(filled, requested, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("canceled status cannot contain a full fill")
        normalized["performance_eligible"] = status == "filled"
    else:
        if payload.get("order_id"):
            raise ValueError("counterfactual observation must not contain order_id")
        normalized["performance_eligible"] = False
    return normalized


def record_execution_observations(database: Path, payloads: list[dict[str, Any]]) -> int:
    if not database.exists():
        raise FileNotFoundError(database)
    observations = [validate_execution_observation(payload) for payload in payloads]
    connection = duckdb.connect(str(database))
    inserted = 0
    try:
        connection.execute("BEGIN TRANSACTION")
        for row in observations:
            decision = connection.execute(
                "SELECT intent_id FROM decisions WHERE timestamp = ?", [row["decision_timestamp"]]
            ).fetchone()
            if decision is None or decision[0] is None:
                raise RuntimeError("execution observation has no matching position-change intent")
            exists = connection.execute(
                """
                SELECT count(*) FROM execution_observations
                WHERE decision_timestamp = ? AND observation_type = ? AND observation_id = ?
                """,
                [row["decision_timestamp"], row["observation_type"], row["observation_id"]],
            ).fetchone()[0]
            if exists:
                continue
            connection.execute(
                """
                INSERT INTO execution_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    row["decision_timestamp"], row["observation_type"], row["observation_id"],
                    row["observed_at_utc"], row["execution_status"], row.get("order_id"),
                    row.get("requested_quantity"), row.get("filled_quantity"),
                    row.get("average_fill_price"), row.get("fee_paid"),
                    row.get("queue_ahead_quantity"), row["source"],
                    json.dumps(row, sort_keys=True, default=_json_default), row["performance_eligible"],
                ],
            )
            inserted += 1
        connection.execute(
            """
            UPDATE decisions AS d
            SET execution_evidence_status = CASE
                WHEN EXISTS (
                    SELECT 1 FROM execution_observations AS e
                    WHERE e.decision_timestamp = d.timestamp AND e.performance_eligible
                ) THEN 'actual_filled'
                WHEN EXISTS (
                    SELECT 1 FROM execution_observations AS e
                    WHERE e.decision_timestamp = d.timestamp
                      AND e.observation_type = 'actual_exchange_fill'
                ) THEN 'actual_observed_incomplete'
                WHEN EXISTS (
                    SELECT 1 FROM execution_observations AS e
                    WHERE e.decision_timestamp = d.timestamp
                ) THEN 'counterfactual_only'
                ELSE d.execution_evidence_status
            END
            """
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return inserted


def observer_summary(database: Path) -> dict[str, Any]:
    if not database.exists():
        return {
            "schema_version": "eth_micro_scalp_v3.observer_summary.v1",
            "database": str(database),
            "decision_count": 0,
            "position_change_intents": 0,
            "performance_eligible": False,
            "reason": "observer database does not exist",
        }
    connection = duckdb.connect(str(database), read_only=True)
    try:
        decisions = connection.execute("SELECT count(*) FROM decisions").fetchone()[0]
        intents = connection.execute("SELECT count(*) FROM decisions WHERE intent_id IS NOT NULL").fetchone()[0]
        actual = connection.execute(
            "SELECT count(DISTINCT decision_timestamp) FROM execution_observations WHERE performance_eligible"
        ).fetchone()[0]
        actual_observed = connection.execute(
            """
            SELECT count(DISTINCT decision_timestamp)
            FROM execution_observations
            WHERE observation_type = 'actual_exchange_fill'
            """
        ).fetchone()[0]
        counterfactual = connection.execute(
            "SELECT count(DISTINCT decision_timestamp) FROM execution_observations WHERE NOT performance_eligible"
        ).fetchone()[0]
        time_range = connection.execute("SELECT min(timestamp), max(timestamp) FROM decisions").fetchone()
    finally:
        connection.close()
    eligible = bool(intents > 0 and actual == intents)
    reason = (
        "all position-change intents have actual exchange execution observations"
        if eligible else "actual execution evidence is incomplete; PnL must not be reported"
    )
    return {
        "schema_version": "eth_micro_scalp_v3.observer_summary.v1",
        "database": str(database),
        "decision_count": int(decisions),
        "position_change_intents": int(intents),
        "actual_execution_observed_intents": int(actual_observed),
        "fully_filled_actual_intents": int(actual),
        "counterfactual_observed_intents": int(counterfactual),
        "first_decision_utc": str(time_range[0]) if time_range[0] is not None else None,
        "last_decision_utc": str(time_range[1]) if time_range[1] is not None else None,
        "performance_eligible": eligible,
        "reason": reason,
        "order_submission_supported": False,
    }


def _read_observation_payloads(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, list) else [payload]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--feature-stream", type=Path, default=DEFAULT_FEATURE_STREAM)
    audit.add_argument("--report", type=Path, default=DEFAULT_READINESS_REPORT)
    run = subparsers.add_parser("run")
    run.add_argument("--feature-stream", type=Path, default=DEFAULT_FEATURE_STREAM)
    run.add_argument("--observer-db", type=Path, default=DEFAULT_OBSERVER_DB)
    run.add_argument("--device", choices=("cpu", "cuda"), default=None)
    run.add_argument("--dry-run", action="store_true")
    record = subparsers.add_parser("record-observation")
    record.add_argument("--observer-db", type=Path, default=DEFAULT_OBSERVER_DB)
    record.add_argument("--input", type=Path, required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--observer-db", type=Path, default=DEFAULT_OBSERVER_DB)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "audit":
        report = audit_sources(args.feature_stream)
        _write_json_atomic(args.report, report)
        print(json.dumps(report, indent=2, default=_json_default))
        return 0 if report["ready"] else 2
    if args.command == "run":
        frame = load_feature_stream(args.feature_stream)
        runtime = load_runtime(device_name=args.device)
        last_timestamp, previous_position = observer_state(args.observer_db)
        decisions = build_decisions(frame, runtime, previous_position, last_timestamp)
        inserted = 0 if args.dry_run else commit_decisions(args.observer_db, runtime, decisions)
        payload = {
            "dry_run": bool(args.dry_run),
            "computed_decisions": len(decisions),
            "inserted_decisions": inserted,
            "order_submission_supported": False,
            "summary": observer_summary(args.observer_db),
        }
        print(json.dumps(payload, indent=2, default=_json_default))
        return 0
    if args.command == "record-observation":
        inserted = record_execution_observations(
            args.observer_db, _read_observation_payloads(args.input)
        )
        print(json.dumps({"inserted_observations": inserted, "summary": observer_summary(args.observer_db)}, indent=2))
        return 0
    if args.command == "summary":
        print(json.dumps(observer_summary(args.observer_db), indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
