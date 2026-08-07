#!/usr/bin/env python3
"""Run HexaPulse-R v1 as a non-executing six-input forward shadow."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading_bot_modules.hexa_pulse_formula import (  # noqa: E402
    FORMULA_ID,
    HexaPulseConfig,
    HexaPulseState,
    prepare_live_formula_frame,
    reconstruct_whale_position_score,
    step_formula,
)
from trading_bot_modules.hexa_pulse_overlay import (  # noqa: E402
    OVERLAY_ID,
    HexaPulseOverlayConfig,
    decide_overlay,
)


MICRO_DB = ROOT / "data/live/microstructure.duckdb"
TAIL_DB = ROOT / "data/live/tail_risk.duckdb"
STATE_PATH = ROOT / "data/live/hexa_pulse_formula_shadow_state.json"
PARENT_DB = ROOT / "data/live/eth_micro_scalp_v4_shadow.duckdb"
OVERLAY_DECISION_LOG = ROOT / "data/live/hexa_pulse_v4_overlay_decisions.jsonl"
DIAGNOSTIC_REPORT = ROOT / "data/ensemble/reports/hexa_pulse_r_v1_diagnostic_20260718.json"
FEE_PER_NOTIONAL_CHANGE = 0.00045
OVERLAY_FRESH_START_UTC = pd.Timestamp("2026-07-18 15:00:00")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _snapshot_database(source: Path, destination: Path, retries: int = 5) -> None:
    for _attempt in range(retries):
        before = source.stat()
        shutil.copy2(source, destination)
        after = source.stat()
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            return
        time.sleep(0.2)
    raise RuntimeError(f"database changed during every snapshot attempt: {source}")


def _load_frame() -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="hexa-pulse-shadow-") as directory:
        snapshot_dir = Path(directory)
        micro_path = snapshot_dir / "microstructure.duckdb"
        tail_path = snapshot_dir / "tail_risk.duckdb"
        _snapshot_database(MICRO_DB, micro_path)
        _snapshot_database(TAIL_DB, tail_path)
        return _load_snapshot_frame(micro_path, tail_path)


def _load_snapshot_frame(micro_path: Path, tail_path: Path) -> pd.DataFrame:
    micro_con = duckdb.connect(str(micro_path), read_only=True)
    micro = micro_con.execute(
        """
        SELECT ts, nif_whale, obi, eai, oi_delta_pct, shadow_toxicity_score,
               whale_position_score, mark_price, data_stale, valid_nif,
               warmup_30m_ready, schema_version AS micro_schema_version
        FROM microstructure_1m
        ORDER BY ts DESC
        LIMIT 240
        """
    ).fetchdf()
    micro_con.close()
    tail_con = duckdb.connect(str(tail_path), read_only=True)
    tail = tail_con.execute(
        """
        SELECT ts, shadow_aftershock_prob, valid_liq_stream,
               schema_version AS tail_schema_version
        FROM tail_risk_1m
        ORDER BY ts DESC
        LIMIT 240
        """
    ).fetchdf()
    tail_con.close()
    if micro.empty or tail.empty:
        raise RuntimeError("microstructure or tail-risk history is empty")

    for frame in (micro, tail):
        frame["ts"] = pd.to_datetime(frame["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)
        frame.sort_values("ts", inplace=True)
        frame.drop_duplicates("ts", keep="last", inplace=True)
    reconstructed = reconstruct_whale_position_score(micro["nif_whale"], micro["oi_delta_pct"])
    micro["whale_position_score"] = pd.to_numeric(
        micro["whale_position_score"], errors="coerce"
    ).fillna(reconstructed)
    merged = pd.merge_asof(micro, tail, on="ts", direction="backward", tolerance=pd.Timedelta("2min"))
    return merged.set_index("ts").sort_index()


def _load_parent_decisions() -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="hexa-parent-shadow-") as directory:
        snapshot = Path(directory) / "eth_micro_scalp_v4_shadow.duckdb"
        _snapshot_database(PARENT_DB, snapshot)
        connection = duckdb.connect(str(snapshot), read_only=True)
        try:
            frame = connection.execute(
                """
                SELECT timestamp, model_id, close, available AS parent_available,
                       previous_position, target_position
                FROM decisions
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 240
                """,
                [OVERLAY_FRESH_START_UTC.to_pydatetime()],
            ).fetchdf()
        finally:
            connection.close()
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()


def _append_overlay_records(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    OVERLAY_DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OVERLAY_DECISION_LOG.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _process_parent_overlay(
    previous: dict[str, Any],
    prepared: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = dict(previous)
    runtime.setdefault("last_timestamp_utc", None)
    runtime.setdefault("last_close", 0.0)
    runtime.setdefault("parent_position", 0)
    runtime.setdefault("overlay_position", 0)
    runtime.setdefault("parent_equity", 1.0)
    runtime.setdefault("overlay_equity", 1.0)
    runtime.setdefault("processed_decisions", 0)
    runtime.setdefault("entry_delays", 0)
    runtime.setdefault("risk_blocks", 0)
    runtime.setdefault("exit_assists", 0)
    config = HexaPulseOverlayConfig()
    parent = _load_parent_decisions()
    if parent.empty:
        return runtime, []
    columns = ["score", "toxicity", "tail_risk", "available"]
    formula = prepared[columns].rename(columns={"available": "formula_available"})
    joined = parent.join(formula, how="left")
    if runtime["last_timestamp_utc"]:
        joined = joined.loc[joined.index > pd.Timestamp(runtime["last_timestamp_utc"])]
    records: list[dict[str, Any]] = []
    for timestamp, row in joined.iterrows():
        close = float(row["close"])
        previous_close = float(runtime["last_close"] or 0.0)
        parent_equity = float(runtime["parent_equity"])
        overlay_equity = float(runtime["overlay_equity"])
        if previous_close > 0.0 and close > 0.0:
            price_return = close / previous_close - 1.0
            parent_equity *= 1.0 + int(runtime["parent_position"]) * price_return
            overlay_equity *= 1.0 + int(runtime["overlay_position"]) * price_return
        else:
            price_return = 0.0

        parent_position = int(row["target_position"])
        score = float(row["score"]) if pd.notna(row["score"]) else float("nan")
        toxicity = float(row["toxicity"]) if pd.notna(row["toxicity"]) else 1.0
        tail_risk = float(row["tail_risk"]) if pd.notna(row["tail_risk"]) else 1.0
        parent_available = (
            bool(row["parent_available"]) if pd.notna(row["parent_available"]) else False
        )
        formula_available = (
            bool(row["formula_available"]) if pd.notna(row["formula_available"]) else False
        )
        available = parent_available and formula_available
        decision = decide_overlay(
            parent_position=parent_position,
            overlay_position=int(runtime["overlay_position"]),
            score=score,
            toxicity=toxicity,
            tail_risk=tail_risk,
            available=available,
            config=config,
        )
        parent_turnover = abs(parent_position - int(runtime["parent_position"]))
        overlay_turnover = abs(int(decision.position) - int(runtime["overlay_position"]))
        parent_equity *= max(0.0, 1.0 - FEE_PER_NOTIONAL_CHANGE * parent_turnover)
        overlay_equity *= max(0.0, 1.0 - FEE_PER_NOTIONAL_CHANGE * overlay_turnover)

        if decision.action == "DELAY":
            runtime["entry_delays"] = int(runtime["entry_delays"]) + 1
        if decision.reason in {"TOXICITY_BLOCK", "TAIL_RISK_BLOCK"}:
            runtime["risk_blocks"] = int(runtime["risk_blocks"]) + 1
        if decision.reason == "HEXA_OPPOSITION_EXIT":
            runtime["exit_assists"] = int(runtime["exit_assists"]) + 1
        runtime.update(
            {
                "last_timestamp_utc": timestamp.isoformat(),
                "last_close": close,
                "parent_model_id": str(row["model_id"]),
                "parent_position": parent_position,
                "overlay_position": int(decision.position),
                "parent_equity": parent_equity,
                "overlay_equity": overlay_equity,
                "processed_decisions": int(runtime["processed_decisions"]) + 1,
                "last_action": decision.action,
                "last_reason": decision.reason,
                "last_score": score if np.isfinite(score) else None,
            }
        )
        records.append(
            {
                "schema_version": "live.hexa_pulse_v4_overlay_decision.v1",
                "timestamp_utc": timestamp.isoformat(),
                "parent_model_id": str(row["model_id"]),
                "formula_id": FORMULA_ID,
                "overlay_id": OVERLAY_ID,
                "parent_position": parent_position,
                "overlay_position": int(decision.position),
                "action": decision.action,
                "reason": decision.reason,
                "score": score if np.isfinite(score) else None,
                "toxicity": toxicity,
                "tail_risk": tail_risk,
                "available": available,
                "parent_available": parent_available,
                "formula_available": formula_available,
                "close": close,
                "interval_price_return": price_return,
                "parent_equity": parent_equity,
                "overlay_equity": overlay_equity,
                "order_submission_supported": False,
            }
        )
    return runtime, records


def _close_shadow_position(runtime: dict[str, Any], mark_price: float) -> float:
    position = int(runtime.get("position", 0))
    entry_price = float(runtime.get("entry_price", 0.0) or 0.0)
    closed_equity = float(runtime.get("closed_equity", 1.0) or 1.0)
    if position and entry_price > 0.0 and mark_price > 0.0:
        trade_move = position * (mark_price / entry_price - 1.0)
        before_exit_fee = closed_equity * (1.0 + trade_move)
        after_exit_fee = before_exit_fee * (1.0 - FEE_PER_NOTIONAL_CHANGE)
        runtime["closed_equity"] = after_exit_fee
        runtime["trade_count"] = int(runtime.get("trade_count", 0)) + 1
        runtime["win_count"] = int(runtime.get("win_count", 0)) + int(after_exit_fee > closed_equity)
        runtime["last_trade_return_pct"] = (after_exit_fee / closed_equity - 1.0) * 100.0
    runtime["position"] = 0
    runtime["entry_price"] = 0.0
    return float(runtime.get("closed_equity", closed_equity))


def _apply_shadow_action(runtime: dict[str, Any], action: str, position: int, mark_price: float) -> None:
    previous = int(runtime.get("position", 0))
    if previous and position == 0:
        _close_shadow_position(runtime, mark_price)
    elif previous == 0 and position and mark_price > 0.0:
        runtime["closed_equity"] = float(runtime.get("closed_equity", 1.0)) * (
            1.0 - FEE_PER_NOTIONAL_CHANGE
        )
        runtime["position"] = int(position)
        runtime["entry_price"] = float(mark_price)
        runtime["entry_action"] = action


def run_once(max_stream_age_minutes: float = 5.0) -> dict[str, Any]:
    previous_state = _load_json(STATE_PATH)
    previous_overlay_runtime = dict(previous_state.get("parent_overlay_runtime") or {})
    overlay_records: list[dict[str, Any]] = []
    runtime = dict(previous_state.get("runtime") or {})
    runtime.setdefault("closed_equity", 1.0)
    runtime.setdefault("position", 0)
    runtime.setdefault("entry_price", 0.0)
    runtime.setdefault("trade_count", 0)
    runtime.setdefault("win_count", 0)
    saved_machine = previous_state.get("machine") or {}
    machine = HexaPulseState(
        position=int(runtime.get("position", 0)),
        long_streak=int(saved_machine.get("long_streak", 0)),
        short_streak=int(saved_machine.get("short_streak", 0)),
    )
    config = HexaPulseConfig()
    blockers: list[str] = []

    try:
        raw = _load_frame()
        prepared = prepare_live_formula_frame(raw)
        overlay_runtime, overlay_records = _process_parent_overlay(
            previous_overlay_runtime,
            prepared,
        )
        now_minute = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
        causal = prepared.loc[prepared.index <= now_minute]
        if causal.empty:
            raise RuntimeError("no causally available HexaPulse decision row")
        row = causal.iloc[-1]
        decision_ts = causal.index[-1]
        source_ts = decision_ts - pd.Timedelta(minutes=2)
        source = raw.loc[source_ts]
        age_minutes = (now_minute - decision_ts).total_seconds() / 60.0
        available = bool(row["available"]) and age_minutes <= max_stream_age_minutes
        if not bool(row["available"]):
            blockers.append("six-input schema or stream quality contract is not ready")
        if age_minutes > max_stream_age_minutes:
            blockers.append(f"formula stream stale: {age_minutes:.2f} minutes")
        score = float(row["score"]) if np.isfinite(row["score"]) else float("nan")
        toxicity = float(row["toxicity"])
        tail_risk = float(row["tail_risk"])
        mark_price = float(source.get("mark_price", 0.0) or 0.0)
        decision = step_formula(
            machine,
            score=score,
            toxicity=toxicity,
            tail_risk=tail_risk,
            available=available,
            config=config,
        )
        _apply_shadow_action(runtime, decision.action, decision.position, mark_price)
        open_move = 0.0
        if int(runtime["position"]) and float(runtime["entry_price"]) > 0.0 and mark_price > 0.0:
            open_move = int(runtime["position"]) * (mark_price / float(runtime["entry_price"]) - 1.0)
        marked_equity = float(runtime["closed_equity"]) * (1.0 + open_move)
        win_rate = (
            float(runtime["win_count"]) / int(runtime["trade_count"])
            if int(runtime["trade_count"])
            else 0.0
        )
        state = {
            "schema_version": "live.hexa_pulse_formula_shadow.v1",
            "formula_id": FORMULA_ID,
            "status": "ok" if available else "unavailable",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_timestamp_utc": decision_ts.isoformat(),
            "available": available,
            "blockers": blockers,
            "score": score if np.isfinite(score) else None,
            "components": {
                "flow_z": float(row["flow_z"]),
                "obi_z": float(row["obi_z"]),
                "whale_position": float(row["whale_position"]),
                "energy_percentile": float(row["energy_percentile"]),
                "toxicity": toxicity,
                "tail_risk": tail_risk,
                "risk_multiplier": float(row["risk_multiplier"]),
                "pressure": float(row["pressure"]),
            },
            "decision": {
                "action": decision.action,
                "reason": decision.reason,
                "position": int(decision.position),
            },
            "machine": {
                "long_streak": int(decision.long_streak),
                "short_streak": int(decision.short_streak),
            },
            "runtime": runtime,
            "performance": {
                "marked_return_pct": (marked_equity - 1.0) * 100.0,
                "closed_return_pct": (float(runtime["closed_equity"]) - 1.0) * 100.0,
                "trade_count": int(runtime["trade_count"]),
                "win_rate": win_rate,
                "fee_bps_per_notional_change": FEE_PER_NOTIONAL_CHANGE * 10_000.0,
            },
            "parent_overlay": {
                "overlay_id": OVERLAY_ID,
                "parent_model_id": overlay_runtime.get("parent_model_id"),
                "fresh_start_utc": OVERLAY_FRESH_START_UTC.isoformat(),
                "last_timestamp_utc": overlay_runtime.get("last_timestamp_utc"),
                "parent_position": int(overlay_runtime.get("parent_position", 0)),
                "overlay_position": int(overlay_runtime.get("overlay_position", 0)),
                "last_action": overlay_runtime.get("last_action", "CASH"),
                "last_reason": overlay_runtime.get("last_reason", "NO_PARENT_DECISION"),
                "processed_decisions": int(overlay_runtime.get("processed_decisions", 0)),
                "entry_delays": int(overlay_runtime.get("entry_delays", 0)),
                "risk_blocks": int(overlay_runtime.get("risk_blocks", 0)),
                "exit_assists": int(overlay_runtime.get("exit_assists", 0)),
                "parent_return_pct": (float(overlay_runtime.get("parent_equity", 1.0)) - 1.0) * 100.0,
                "overlay_return_pct": (float(overlay_runtime.get("overlay_equity", 1.0)) - 1.0) * 100.0,
                "incremental_return_pct": (
                    float(overlay_runtime.get("overlay_equity", 1.0))
                    - float(overlay_runtime.get("parent_equity", 1.0))
                ) * 100.0,
                "fee_bps_per_notional_change": FEE_PER_NOTIONAL_CHANGE * 10_000.0,
                "decision_log": str(OVERLAY_DECISION_LOG),
                "fresh_forward_bar_by_bar": True,
                "trade_ledgers_used_as_input": False,
                "saved_parent_exit_timestamps_used": False,
                "future_rows_used_for_entry": False,
                "order_submission_supported": False,
                "activation_allowed": False,
            },
            "parent_overlay_runtime": overlay_runtime,
            "formula_config": asdict(config),
            "historical_diagnostic_report": str(DIAGNOSTIC_REPORT),
            "evaluation_class": "forward_shadow_only",
            "promotion_pass": False,
            "order_submission_supported": False,
            "activation_allowed": False,
        }
    except Exception as exc:
        state = {
            "schema_version": "live.hexa_pulse_formula_shadow.v1",
            "formula_id": FORMULA_ID,
            "status": "unavailable",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "available": False,
            "blockers": [str(exc)],
            "machine": asdict(machine),
            "runtime": runtime,
            "parent_overlay_runtime": previous_overlay_runtime,
            "formula_config": asdict(config),
            "evaluation_class": "forward_shadow_only",
            "promotion_pass": False,
            "order_submission_supported": False,
            "activation_allowed": False,
        }
    _write_json(STATE_PATH, state)
    _append_overlay_records(overlay_records)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("once")
    serve = sub.add_parser("serve")
    serve.add_argument("--interval-seconds", type=float, default=60.0)
    serve.add_argument("--max-stream-age-minutes", type=float, default=5.0)
    args = parser.parse_args()
    if args.command == "once":
        print(json.dumps(run_once(), ensure_ascii=False, indent=2))
        return
    while True:
        state = run_once(args.max_stream_age_minutes)
        print(json.dumps({"status": state.get("status"), "score": state.get("score")}), flush=True)
        time.sleep(max(5.0, args.interval_seconds))


if __name__ == "__main__":
    main()
