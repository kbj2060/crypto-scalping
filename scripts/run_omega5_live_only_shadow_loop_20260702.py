#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb


KST = ZoneInfo("Asia/Seoul")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data/live/microstructure.duckdb"
DEFAULT_TABLE = "decision_feature_frame_live_only_shadow_20260702"
DEFAULT_SOURCE_JSONL = ROOT / "data/live/decision_feature_snapshot.jsonl"
DEFAULT_OUT = ROOT / "data/live/omega5_live_only_upgrade_loop_20260702_v3_ml"
ROUNDTRIP_COST_PER_NOTIONAL = 0.0006


@dataclass(frozen=True)
class Candidate:
    name: str
    max_hold_minutes: int
    take_profit: float
    stop_loss: float
    notional: float


CANDIDATES = (
    Candidate("cash_control", 30, 0.0, 0.0, 0.0),
    Candidate("omega5_live_micro_balanced_v1", 30, 0.006, 0.004, 1.0),
    Candidate("omega5_live_short_guard_v1", 30, 0.007, 0.0045, 1.0),
    Candidate("omega5_live_shock_fade_v1", 20, 0.004, 0.0035, 0.6),
    Candidate("omega5_live_trend_follow_v1", 45, 0.009, 0.005, 1.2),
    Candidate("omega5_live_micro_flow_v2", 20, 0.0035, 0.0025, 0.8),
    Candidate("omega5_live_short_momentum_v2", 25, 0.0045, 0.0030, 1.0),
    Candidate("omega5_live_long_reversal_v2", 25, 0.0040, 0.0030, 0.8),
    Candidate("omega5_live_online_logit_v3", 25, 0.0040, 0.0030, 0.8),
    Candidate("omega5_live_online_bandit_v3", 20, 0.0035, 0.0025, 0.7),
    Candidate("omega5_live_online_fast_logit_v4", 10, 0.0025, 0.0018, 0.6),
    Candidate("omega5_live_online_fast_bandit_v4", 10, 0.0022, 0.0016, 0.5),
    Candidate("omega5_live_online_short_bandit_v5", 15, 0.0030, 0.0020, 0.8),
    Candidate("omega5_live_short_momentum_online_v6", 15, 0.0035, 0.0022, 1.0),
    Candidate("omega5_live_online_short_guarded_v7", 12, 0.0028, 0.0018, 0.7),
    Candidate("omega5_live_rule_plus_guarded_ml_v8", 15, 0.0035, 0.0022, 1.0),
)

ONLINE_CANDIDATES = {
    "omega5_live_online_logit_v3",
    "omega5_live_online_bandit_v3",
    "omega5_live_online_fast_logit_v4",
    "omega5_live_online_fast_bandit_v4",
    "omega5_live_online_short_bandit_v5",
    "omega5_live_short_momentum_online_v6",
    "omega5_live_online_short_guarded_v7",
    "omega5_live_rule_plus_guarded_ml_v8",
}
ONLINE_FEATURE_NAMES = (
    "bias",
    "net_taker_ratio",
    "taker_acceleration",
    "smart_money_flow",
    "cvd_slope_12",
    "log_return",
    "btc_ret_1",
    "rsi_z",
    "mtf_mix",
    "jump_z",
    "bb_width_rank_z",
    "compression_up",
    "compression_down",
)


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=KST)
    text = str(value).strip()
    if not text:
        return None
    try:
        text = text.replace("Z", "+00:00")
        out = datetime.fromisoformat(text)
        return out if out.tzinfo else out.replace(tzinfo=KST)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            out = datetime.strptime(text, fmt)
            return out if out.tzinfo else out.replace(tzinfo=KST)
        except ValueError:
            continue
    return None


def parse_exchange_bar_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            out = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return parse_dt(value)
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(KST)


def finite_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def clip_unit(value: float) -> float:
    return float(max(-1.0, min(1.0, value)))


def sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def stable_bucket(text: str, modulo: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(modulo)


def online_feature_vector(row: dict[str, Any]) -> dict[str, float]:
    mtf_mix = finite_float(row, "mtf_trend_1h") + 0.5 * finite_float(row, "mtf_trend_4h")
    return {
        "bias": 1.0,
        "net_taker_ratio": clip_unit(finite_float(row, "net_taker_ratio")),
        "taker_acceleration": clip_unit(finite_float(row, "taker_acceleration")),
        "smart_money_flow": clip_unit(finite_float(row, "smart_money_flow") * 1000.0),
        "cvd_slope_12": clip_unit(finite_float(row, "cvd_slope_12") * 10.0),
        "log_return": clip_unit(finite_float(row, "log_return") * 500.0),
        "btc_ret_1": clip_unit(finite_float(row, "btc_ret_1") * 10.0),
        "rsi_z": clip_unit((finite_float(row, "rsi", 50.0) - 50.0) / 25.0),
        "mtf_mix": clip_unit(mtf_mix * 1000.0),
        "jump_z": clip_unit(finite_float(row, "jump_z") / 3.0),
        "bb_width_rank_z": clip_unit((finite_float(row, "bb_width_pct_rank_288", 0.5) - 0.5) * 2.0),
        "compression_up": clip_unit(finite_float(row, "compression_release_up") * 25.0),
        "compression_down": clip_unit(finite_float(row, "compression_release_down") * 25.0),
    }


def online_initial_state() -> dict[str, Any]:
    models: dict[str, Any] = {}
    for candidate in ONLINE_CANDIDATES:
        models[candidate] = {
            "long": {"weights": {}, "updates": 0},
            "short": {"weights": {}, "updates": 0},
        }
    return {
        "schema_version": "omega5.live_only_online_models.v1",
        "models": models,
        "updated_closed_keys": [],
        "learning_rate": 0.08,
        "l2": 0.001,
    }


def online_score(model_state: dict[str, Any], candidate: str, side_name: str, features: dict[str, float]) -> tuple[float, float]:
    model = ((model_state.get("models") or {}).get(candidate) or {}).get(side_name) or {}
    weights = dict(model.get("weights") or {})
    raw = 0.0
    for name in ONLINE_FEATURE_NAMES:
        raw += float(weights.get(name, 0.0)) * float(features.get(name, 0.0))
    return raw, sigmoid(raw)


def update_online_models(model_state: dict[str, Any], closed: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(model_state, dict) or not model_state:
        model_state = online_initial_state()
    model_state.setdefault("models", online_initial_state()["models"])
    updated_keys = set(str(x) for x in model_state.get("updated_closed_keys", []))
    lr = float(model_state.get("learning_rate", 0.08))
    l2 = float(model_state.get("l2", 0.001))
    for row in closed:
        candidate = str(row.get("candidate", ""))
        if candidate not in ONLINE_CANDIDATES:
            continue
        key = str(row.get("signal_key", ""))
        if not key or key in updated_keys:
            continue
        side = int(row.get("side", 0))
        if side == 0:
            continue
        features = row.get("online_features")
        if not isinstance(features, dict):
            continue
        side_name = "long" if side > 0 else "short"
        models = model_state.setdefault("models", {})
        candidate_state = models.setdefault(candidate, {"long": {"weights": {}, "updates": 0}, "short": {"weights": {}, "updates": 0}})
        side_state = candidate_state.setdefault(side_name, {"weights": {}, "updates": 0})
        weights = dict(side_state.get("weights") or {})
        _, prob = online_score(model_state, candidate, side_name, features)
        target = 1.0 if float(row.get("net_account_pnl", 0.0)) > 0.0 else 0.0
        reward_scale = min(2.0, 1.0 + abs(float(row.get("net_account_pnl", 0.0))) * 100.0)
        for name in ONLINE_FEATURE_NAMES:
            x = float(features.get(name, 0.0))
            w = float(weights.get(name, 0.0))
            weights[name] = w + lr * reward_scale * ((target - prob) * x - l2 * w)
        side_state["weights"] = {k: float(v) for k, v in weights.items()}
        side_state["updates"] = int(side_state.get("updates", 0)) + 1
        updated_keys.add(key)
    model_state["updated_closed_keys"] = sorted(updated_keys)
    return model_state


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
    )


def latest_row(db_path: Path, table: str) -> dict[str, Any] | None:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if not table_exists(con, table):
            return None
        rel = con.execute(
            f"SELECT * FROM {table} ORDER BY live_recorded_at_kst DESC LIMIT 1"
        )
        cols = [d[0] for d in rel.description]
        row = rel.fetchone()
        if row is None:
            return None
        return dict(zip(cols, row))
    finally:
        con.close()


def rows_after(db_path: Path, table: str, after_ts: datetime) -> list[dict[str, Any]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if not table_exists(con, table):
            return []
        rel = con.execute(
            f"""
            SELECT *
            FROM {table}
            WHERE live_recorded_at_kst > ?
            ORDER BY live_recorded_at_kst ASC
            """,
            [after_ts.isoformat()],
        )
        cols = [d[0] for d in rel.description]
        return [dict(zip(cols, row)) for row in rel.fetchall()]
    finally:
        con.close()


def snapshot_payload_to_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    values = payload.get("values")
    if not isinstance(values, dict):
        return None
    row = dict(values)
    raw_ts = row.get("timestamp")
    bar_dt = parse_exchange_bar_dt(raw_ts)
    if bar_dt is not None:
        row["source_timestamp_raw"] = raw_ts
        row["timestamp"] = bar_dt.isoformat()
    row["live_recorded_at_kst"] = payload.get("created_at")
    row["live_row_count"] = payload.get("row_count")
    row["live_pipeline_stage"] = (payload.get("health_summary") or {}).get("status", "")
    return row


def read_new_snapshot_rows(path: Path, offset: int) -> tuple[int, list[dict[str, Any]]]:
    if not path.exists():
        return offset, []
    size = path.stat().st_size
    if offset < 0 or offset > size:
        offset = size
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        fh.seek(offset)
        while True:
            pos_before = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.endswith("\n"):
                return pos_before, rows
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = snapshot_payload_to_row(payload)
            if row is not None:
                rows.append(row)
        return fh.tell(), rows


def side_for_candidate(candidate: Candidate, row: dict[str, Any], model_state: dict[str, Any] | None = None) -> int:
    if candidate.name == "cash_control":
        return 0

    m7_up = finite_float(row, "m7_prob_up")
    m7_dn = finite_float(row, "m7_prob_dn")
    m7_conf = finite_float(row, "m7_confidence")
    rsi = finite_float(row, "rsi", 50.0)
    jump_z = finite_float(row, "jump_z")
    ret = finite_float(row, "log_return")
    mtf_1h = finite_float(row, "mtf_trend_1h")
    mtf_4h = finite_float(row, "mtf_trend_4h")
    hma = finite_float(row, "hma_slope")
    taker = finite_float(row, "net_taker_ratio")
    taker_accel = finite_float(row, "taker_acceleration")
    bb_width = finite_float(row, "bb_width")
    smf = finite_float(row, "smart_money_flow")
    oi_change = finite_float(row, "oi_change_rate")
    cvd_slope = finite_float(row, "cvd_slope_12")
    btc_ret_1 = finite_float(row, "btc_ret_1")
    compression_up = finite_float(row, "compression_release_up")
    compression_down = finite_float(row, "compression_release_down")
    lower_wick_z = finite_float(row, "lower_wick_z")
    upper_wick_z = finite_float(row, "upper_wick_z")

    if candidate.name == "omega5_live_micro_balanced_v1":
        if m7_conf >= 0.45 and m7_up - m7_dn >= 0.12 and taker > -0.15:
            return 1
        if m7_conf >= 0.45 and m7_dn - m7_up >= 0.10 and taker < 0.20:
            return -1
        return 0

    if candidate.name == "omega5_live_short_guard_v1":
        if m7_dn >= 0.48 and m7_up <= 0.42 and rsi <= 62.0 and jump_z > -3.0:
            return -1
        return 0

    if candidate.name == "omega5_live_shock_fade_v1":
        if jump_z >= 2.0 or ret >= 0.004:
            return -1
        if jump_z <= -2.0 or ret <= -0.004:
            return 1
        return 0

    if candidate.name == "omega5_live_trend_follow_v1":
        trend = mtf_1h + 0.5 * mtf_4h + (1.0 if hma > 0 else -1.0 if hma < 0 else 0.0)
        if bb_width > 0.0 and trend >= 1.0 and m7_up >= 0.42:
            return 1
        if bb_width > 0.0 and trend <= -1.0 and m7_dn >= 0.42:
            return -1
        return 0

    if candidate.name == "omega5_live_micro_flow_v2":
        flow_score = (
            0.45 * max(-1.0, min(1.0, taker))
            + 0.20 * max(-1.0, min(1.0, taker_accel))
            + 0.20 * (1.0 if smf > 0.0 else -1.0 if smf < 0.0 else 0.0)
            + 0.15 * (1.0 if cvd_slope > 0.0 else -1.0 if cvd_slope < 0.0 else 0.0)
        )
        if flow_score >= 0.22 and rsi <= 74.0:
            return 1
        if flow_score <= -0.22 and rsi >= 26.0:
            return -1
        return 0

    if candidate.name == "omega5_live_short_momentum_v2":
        short_pressure = (
            (taker <= -0.18)
            + (taker_accel < -0.05)
            + (btc_ret_1 < 0.0)
            + (oi_change >= 0.0 and ret <= 0.0)
            + (compression_down > 0.0)
        )
        if short_pressure >= 3 and rsi >= 35.0 and upper_wick_z < 2.5:
            return -1
        return 0

    if candidate.name == "omega5_live_long_reversal_v2":
        long_reversal = (
            (taker >= 0.18)
            + (taker_accel > 0.05)
            + (btc_ret_1 >= 0.0)
            + (compression_up > 0.0)
            + (lower_wick_z > -2.5)
        )
        if long_reversal >= 3 and rsi <= 65.0:
            return 1
        return 0

    if candidate.name in ONLINE_CANDIDATES:
        state = model_state or online_initial_state()
        features = online_feature_vector(row)
        long_raw, long_prob = online_score(state, candidate.name, "long", features)
        short_raw, short_prob = online_score(state, candidate.name, "short", features)
        models = ((state.get("models") or {}).get(candidate.name) or {})
        updates = int((models.get("long") or {}).get("updates", 0)) + int((models.get("short") or {}).get("updates", 0))
        flow_prior = (
            0.45 * max(-1.0, min(1.0, taker))
            + 0.20 * max(-1.0, min(1.0, taker_accel))
            + 0.20 * (1.0 if smf > 0.0 else -1.0 if smf < 0.0 else 0.0)
            + 0.15 * (1.0 if cvd_slope > 0.0 else -1.0 if cvd_slope < 0.0 else 0.0)
        )
        short_pressure = (
            (taker <= -0.18)
            + (taker_accel < -0.05)
            + (btc_ret_1 < 0.0)
            + (oi_change >= 0.0 and ret <= 0.0)
            + (compression_down > 0.0)
        )
        upward_shock_block = bool(jump_z > 1.2 and ret > 0.0)
        if candidate.name == "omega5_live_online_short_guarded_v7":
            if upward_shock_block:
                return 0
            if updates > 0 and short_prob >= 0.52 and short_prob >= long_prob:
                return -1 if rsi >= 24.0 else 0
            if short_pressure >= 3 and rsi >= 32.0 and upper_wick_z < 2.5:
                return -1
            if flow_prior <= -0.18 and rsi >= 28.0:
                return -1
            return 0
        if candidate.name == "omega5_live_rule_plus_guarded_ml_v8":
            if short_pressure >= 3 and rsi >= 32.0 and upper_wick_z < 2.5:
                return -1
            if upward_shock_block:
                return 0
            if updates > 0 and short_prob >= 0.54 and short_prob >= long_prob:
                return -1 if rsi >= 26.0 else 0
            return 0
        if candidate.name == "omega5_live_short_momentum_online_v6":
            if updates > 0 and short_prob >= 0.52 and short_prob >= long_prob:
                return -1 if rsi >= 24.0 else 0
            if short_pressure >= 3 and rsi >= 32.0 and upper_wick_z < 2.5:
                return -1
            return 0
        if candidate.name == "omega5_live_online_short_bandit_v5":
            if updates > 0 and short_prob >= 0.52 and short_prob >= long_prob:
                return -1 if rsi >= 24.0 else 0
            if flow_prior <= -0.12 and rsi >= 26.0:
                return -1
            bucket = stable_bucket(f"{candidate.name}|{row.get('timestamp', '')}", 5)
            if updates < 3 and bucket in {0, 1} and rsi >= 28.0:
                return -1
            return 0
        if updates > 0 and max(long_prob, short_prob) >= 0.52 and abs(long_raw - short_raw) >= 0.03:
            if long_prob > short_prob and rsi <= 76.0:
                return 1
            if short_prob > long_prob and rsi >= 24.0:
                return -1
        if updates < 4:
            if flow_prior >= 0.20 and rsi <= 72.0:
                return 1
            if flow_prior <= -0.20 and rsi >= 28.0:
                return -1
            if candidate.name in {"omega5_live_online_fast_logit_v4", "omega5_live_online_fast_bandit_v4"}:
                bucket = stable_bucket(f"{candidate.name}|{row.get('timestamp', '')}", 4)
                if bucket in {0, 1} and rsi <= 74.0:
                    return 1
                if bucket in {2, 3} and rsi >= 26.0:
                    return -1
            bucket = stable_bucket(f"{candidate.name}|{row.get('timestamp', '')}", 10)
            if candidate.name == "omega5_live_online_bandit_v3":
                if bucket == 0 and rsi <= 70.0:
                    return 1
                if bucket == 1 and rsi >= 30.0:
                    return -1
            return 0
        if max(long_prob, short_prob) < 0.53 or abs(long_raw - short_raw) < 0.04:
            return 0
        if long_prob > short_prob and rsi <= 76.0:
            return 1
        if short_prob > long_prob and rsi >= 24.0:
            return -1
        return 0

    return 0


def signal_key(candidate: str, signal_ts: str) -> str:
    return f"{candidate}|{signal_ts}"


def build_signals(
    row: dict[str, Any],
    seen: set[str],
    started_at: datetime,
    model_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    signal_dt = parse_dt(row.get("timestamp"))
    recorded_at = parse_dt(row.get("live_recorded_at_kst"))
    if signal_dt is None or recorded_at is None or recorded_at < started_at:
        return []
    close = finite_float(row, "close")
    if close <= 0.0:
        return []

    signals: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        side = side_for_candidate(candidate, row, model_state)
        key = signal_key(candidate.name, signal_dt.isoformat())
        if key in seen:
            continue
        seen.add(key)
        online_features = online_feature_vector(row) if candidate.name in ONLINE_CANDIDATES else {}
        payload = {
            "schema_version": "omega5.live_only_shadow_signal.v1",
            "candidate": candidate.name,
            "signal_key": key,
            "signal_timestamp": signal_dt.isoformat(),
            "recorded_at_kst": recorded_at.isoformat(),
            "entry_price": close,
            "side": side,
            "notional": candidate.notional if side else 0.0,
            "max_hold_minutes": candidate.max_hold_minutes,
            "take_profit_price_move": candidate.take_profit,
            "stop_loss_price_move": candidate.stop_loss,
            "live_forward_only": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "historical_replay_used_for_selection": False,
            "features": {
                "m7_prob_up": finite_float(row, "m7_prob_up"),
                "m7_prob_dn": finite_float(row, "m7_prob_dn"),
                "m7_confidence": finite_float(row, "m7_confidence"),
                "rsi": finite_float(row, "rsi", 50.0),
                "jump_z": finite_float(row, "jump_z"),
                "log_return": finite_float(row, "log_return"),
                "mtf_trend_1h": finite_float(row, "mtf_trend_1h"),
                "mtf_trend_4h": finite_float(row, "mtf_trend_4h"),
                "net_taker_ratio": finite_float(row, "net_taker_ratio"),
            },
            "online_features": online_features,
        }
        if candidate.name in ONLINE_CANDIDATES:
            _, payload["online_long_prob"] = online_score(model_state or online_initial_state(), candidate.name, "long", online_features)
            _, payload["online_short_prob"] = online_score(model_state or online_initial_state(), candidate.name, "short", online_features)
        signals.append(payload)
    return signals


def close_payload(signal: dict[str, Any], row: dict[str, Any], reason: str, exit_price: float) -> dict[str, Any]:
    entry = float(signal["entry_price"])
    side = int(signal["side"])
    notional = float(signal["notional"])
    raw_move = 0.0 if side == 0 else side * (exit_price / entry - 1.0)
    net = raw_move * notional - ROUNDTRIP_COST_PER_NOTIONAL * notional
    exit_dt = parse_dt(row.get("timestamp")) or now_kst()
    signal_dt = parse_dt(signal.get("signal_timestamp")) or exit_dt
    return {
        "schema_version": "omega5.live_only_shadow_close.v1",
        "candidate": signal["candidate"],
        "signal_key": signal["signal_key"],
        "signal_timestamp": signal["signal_timestamp"],
        "exit_timestamp": exit_dt.isoformat(),
        "reason": reason,
        "entry_price": entry,
        "exit_price": exit_price,
        "side": side,
        "notional": notional,
        "raw_price_move": raw_move,
        "net_account_pnl": net,
        "hold_minutes": max(0.0, (exit_dt - signal_dt).total_seconds() / 60.0),
        "features": dict(signal.get("features") or {}),
        "online_features": dict(signal.get("online_features") or {}),
        "online_long_prob": signal.get("online_long_prob"),
        "online_short_prob": signal.get("online_short_prob"),
        "closed_at_kst": now_kst().isoformat(),
        "live_forward_only": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "historical_replay_used_for_selection": False,
    }


def resolve_pending(
    pending: list[dict[str, Any]],
    db_path: Path,
    table: str,
    closed_keys: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    still_pending: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    for signal in pending:
        if signal["signal_key"] in closed_keys:
            continue
        side = int(signal["side"])
        signal_dt = parse_dt(signal.get("signal_timestamp"))
        if signal_dt is None:
            continue
        if side == 0:
            closed_payload = dict(signal)
            closed_payload.update(
                {
                    "schema_version": "omega5.live_only_shadow_close.v1",
                    "exit_timestamp": signal["signal_timestamp"],
                    "reason": "cash_no_trade",
                    "exit_price": signal["entry_price"],
                    "raw_price_move": 0.0,
                    "net_account_pnl": 0.0,
                    "hold_minutes": 0.0,
                    "closed_at_kst": now_kst().isoformat(),
                }
            )
            closed.append(closed_payload)
            closed_keys.add(signal["signal_key"])
            continue

        future_rows = rows_after(db_path, table, signal_dt)
        if not future_rows:
            still_pending.append(signal)
            continue

        entry = float(signal["entry_price"])
        tp = float(signal["take_profit_price_move"])
        sl = float(signal["stop_loss_price_move"])
        deadline = signal_dt + timedelta(minutes=int(signal["max_hold_minutes"]))
        resolved = None
        for row in future_rows:
            bar_dt = parse_dt(row.get("timestamp"))
            if bar_dt is None or bar_dt <= signal_dt:
                continue
            high = finite_float(row, "high", finite_float(row, "close"))
            low = finite_float(row, "low", finite_float(row, "close"))
            close = finite_float(row, "close")
            if close <= 0.0:
                continue
            favorable = (high / entry - 1.0) if side > 0 else (entry / max(low, 1.0e-12) - 1.0)
            adverse = (low / entry - 1.0) if side > 0 else (entry / max(high, 1.0e-12) - 1.0)
            if favorable >= tp > 0.0:
                resolved = close_payload(signal, row, "take_profit", entry * (1.0 + side * tp))
                break
            if adverse <= -sl < 0.0:
                resolved = close_payload(signal, row, "stop_loss", entry * (1.0 - side * sl))
                break
            if bar_dt >= deadline:
                resolved = close_payload(signal, row, "time_exit", close)
                break
        if resolved is None:
            still_pending.append(signal)
        else:
            closed.append(resolved)
            closed_keys.add(signal["signal_key"])
    return still_pending, closed


def resolve_pending_with_rows(
    pending: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    closed_keys: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    still_pending: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    for signal in pending:
        if signal["signal_key"] in closed_keys:
            continue
        side = int(signal["side"])
        signal_dt = parse_dt(signal.get("signal_timestamp"))
        if signal_dt is None:
            continue
        if side == 0:
            closed_payload = dict(signal)
            closed_payload.update(
                {
                    "schema_version": "omega5.live_only_shadow_close.v1",
                    "exit_timestamp": signal["signal_timestamp"],
                    "reason": "cash_no_trade",
                    "exit_price": signal["entry_price"],
                    "raw_price_move": 0.0,
                    "net_account_pnl": 0.0,
                    "hold_minutes": 0.0,
                    "closed_at_kst": now_kst().isoformat(),
                }
            )
            closed.append(closed_payload)
            closed_keys.add(signal["signal_key"])
            continue

        entry = float(signal["entry_price"])
        tp = float(signal["take_profit_price_move"])
        sl = float(signal["stop_loss_price_move"])
        deadline = signal_dt + timedelta(minutes=int(signal["max_hold_minutes"]))
        resolved = None
        for row in rows:
            bar_dt = parse_dt(row.get("timestamp"))
            if bar_dt is None or bar_dt <= signal_dt:
                continue
            high = finite_float(row, "high", finite_float(row, "close"))
            low = finite_float(row, "low", finite_float(row, "close"))
            close = finite_float(row, "close")
            if close <= 0.0:
                continue
            favorable = (high / entry - 1.0) if side > 0 else (entry / max(low, 1.0e-12) - 1.0)
            adverse = (low / entry - 1.0) if side > 0 else (entry / max(high, 1.0e-12) - 1.0)
            if favorable >= tp > 0.0:
                resolved = close_payload(signal, row, "take_profit", entry * (1.0 + side * tp))
                break
            if adverse <= -sl < 0.0:
                resolved = close_payload(signal, row, "stop_loss", entry * (1.0 - side * sl))
                break
            if bar_dt >= deadline:
                resolved = close_payload(signal, row, "time_exit", close)
                break
        if resolved is None:
            still_pending.append(signal)
        else:
            closed.append(resolved)
            closed_keys.add(signal["signal_key"])
    return still_pending, closed


def summarize(closed: list[dict[str, Any]], pending: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        by_candidate[candidate.name] = {
            "closed": 0,
            "active_signals": 0,
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
            "mdd": 0.0,
            "wr": None,
        }
    for signal in pending:
        by_candidate.setdefault(signal["candidate"], {})["active_signals"] = (
            by_candidate.setdefault(signal["candidate"], {}).get("active_signals", 0) + 1
        )
    equity_by_candidate: dict[str, list[float]] = {}
    for row in closed:
        cand = row["candidate"]
        rec = by_candidate.setdefault(cand, {"closed": 0, "active_signals": 0, "trades": 0, "wins": 0, "pnl": 0.0, "mdd": 0.0, "wr": None})
        pnl = float(row.get("net_account_pnl", 0.0))
        rec["closed"] += 1
        if int(row.get("side", 0)) != 0:
            rec["trades"] += 1
            rec["wins"] += 1 if pnl > 0.0 else 0
            rec["pnl"] += pnl
            equity_by_candidate.setdefault(cand, []).append(rec["pnl"])
    for cand, rec in by_candidate.items():
        trades = int(rec.get("trades", 0))
        rec["wr"] = (float(rec.get("wins", 0)) / trades) if trades else None
        curve = equity_by_candidate.get(cand, [])
        peak = 0.0
        mdd = 0.0
        for value in curve:
            peak = max(peak, value)
            mdd = min(mdd, value - peak)
        rec["mdd"] = mdd
    frontier = sorted(
        by_candidate.items(),
        key=lambda kv: (float(kv[1].get("pnl", 0.0)), float(kv[1].get("mdd", 0.0)), int(kv[1].get("trades", 0))),
        reverse=True,
    )
    return {
        "schema_version": "omega5.live_only_shadow_report.v1",
        "updated_at_kst": now_kst().isoformat(),
        "live_forward_only": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "historical_replay_used_for_selection": False,
        "closed_total": len(closed),
        "pending_total": len(pending),
        "candidates": by_candidate,
        "frontier_candidate": frontier[0][0] if frontier else None,
        "frontier_note": "Only live-forward closed signals after loop start are counted.",
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    state_path = out_dir / "state.json"
    signals_path = out_dir / "shadow_signals.jsonl"
    closed_path = out_dir / "shadow_closed.jsonl"
    report_path = out_dir / "report.json"

    state = load_json(state_path, {})
    started_at = parse_dt(state.get("started_at_kst")) or now_kst()
    end_at = parse_dt(args.end_at_kst)
    if end_at is None:
        raise SystemExit(f"invalid --end-at-kst: {args.end_at_kst}")

    pending = list(state.get("pending", []))
    seen = set(state.get("seen_signal_keys", []))
    model_state = state.get("online_model_state")
    if not isinstance(model_state, dict) or not model_state:
        model_state = online_initial_state()
    source_jsonl = Path(args.source_jsonl) if args.source_jsonl else None
    jsonl_offset = int(state.get("source_jsonl_offset", -1))
    if source_jsonl is not None and jsonl_offset < 0:
        jsonl_offset = source_jsonl.stat().st_size if source_jsonl.exists() else 0
    closed_rows = load_jsonl(closed_path)
    closed_keys = {str(r.get("signal_key")) for r in closed_rows if r.get("signal_key")}

    while now_kst() < end_at:
        if source_jsonl is not None:
            jsonl_offset, new_rows = read_new_snapshot_rows(source_jsonl, jsonl_offset)
            for row in new_rows:
                pending, newly_closed = resolve_pending_with_rows(pending, [row], closed_keys)
                for closed in newly_closed:
                    append_jsonl(closed_path, closed)
                    closed_rows.append(closed)
                model_state = update_online_models(model_state, newly_closed)
                new_signals = build_signals(row, seen, started_at, model_state)
                for signal in new_signals:
                    append_jsonl(signals_path, signal)
                    pending.append(signal)
        else:
            row = latest_row(Path(args.db), str(args.table))
            if row is not None:
                new_signals = build_signals(row, seen, started_at, model_state)
                for signal in new_signals:
                    append_jsonl(signals_path, signal)
                    pending.append(signal)
                pending, newly_closed = resolve_pending(pending, Path(args.db), str(args.table), closed_keys)
                for closed in newly_closed:
                    append_jsonl(closed_path, closed)
                    closed_rows.append(closed)
                model_state = update_online_models(model_state, newly_closed)

        report = summarize(closed_rows, pending)
        report.update(
            {
                "started_at_kst": started_at.isoformat(),
                "end_at_kst": end_at.isoformat(),
                "db": str(args.db),
                "table": str(args.table),
                "source_jsonl": str(source_jsonl) if source_jsonl is not None else "",
                "source_jsonl_offset": int(jsonl_offset),
                "poll_seconds": float(args.poll_seconds),
            }
        )
        write_json(report_path, report)
        write_json(
            state_path,
            {
                "schema_version": "omega5.live_only_shadow_state.v1",
                "started_at_kst": started_at.isoformat(),
                "updated_at_kst": now_kst().isoformat(),
                "end_at_kst": end_at.isoformat(),
                "pending": pending,
                "seen_signal_keys": sorted(seen),
                "source_jsonl_offset": int(jsonl_offset),
                "online_model_state": model_state,
            },
        )
        time.sleep(max(5.0, float(args.poll_seconds)))

    report = summarize(closed_rows, pending)
    report.update(
        {
            "started_at_kst": started_at.isoformat(),
            "end_at_kst": end_at.isoformat(),
            "loop_complete": True,
            "db": str(args.db),
            "table": str(args.table),
            "source_jsonl": str(source_jsonl) if source_jsonl is not None else "",
        }
    )
    write_json(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--source-jsonl", default=str(DEFAULT_SOURCE_JSONL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--end-at-kst", default="2026-07-02T18:00:00+09:00")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
