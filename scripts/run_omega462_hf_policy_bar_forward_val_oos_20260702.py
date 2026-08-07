#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmp.causal_regen_20260516.extended_oos_20260702.run_omega5_additional_oos_replay import (  # noqa: E402
    ROUNDTRIP_COST_DEFAULT,
    atr_pct_series,
    parent_decision_at,
)
from trading_bot_modules.omega4_6_2_source_parent_live import EPS, Omega462SourceParentLiveAdapter  # noqa: E402


MODEL_ID = "omega4_6_2_source_parent_fresh_forward_with_hf_policy_overlay_20260702"
BASE = ROOT / "tmp/causal_regen_20260516/extended_oos_20260702"
DEFAULT_VALIDATION_FEATURES = ROOT / "data/splits/year_oos/training_features_2025.csv"
DEFAULT_OOS_FEATURES = BASE / "training_features_2026_0101_0702_m7_ai_for_omega5_parity.csv"
DEFAULT_POLICY_CONFIG = BASE / "omega4_6_2_cached_parent_policy_upgrade_hf_papers_fast_20260702/report.json"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/omega462_hf_policy_bar_forward_val_oos_20260702"
BARS_PER_HOUR = 12.0


class Omega462Cap220NoV5LiveAdapter(Omega462SourceParentLiveAdapter):
    def _apply_v5_exposure(self, candidate: dict[str, Any], now: pd.Timestamp) -> dict[str, Any]:
        out = dict(candidate)
        side = int(out["side"])
        notional = float(out.get("notional", 0.0) or 0.0)
        if side == 0 or notional <= EPS:
            out.update({"loss_governor_scale": 0.0, "source_parent_side_factor": 0.0, "pre_governor_notional": 0.0})
            return out
        out.update(
            {
                "source_parent_side_factor": 1.0,
                "pre_governor_notional": float(notional),
                "loss_governor_scale": 1.0,
                "notional": float(notional),
            }
        )
        return out


def make_parent(variant: str) -> Omega462SourceParentLiveAdapter:
    if variant == "source_v5":
        return Omega462SourceParentLiveAdapter()
    if variant == "cap220_no_v5":
        return Omega462Cap220NoV5LiveAdapter()
    raise RuntimeError(f"unknown parent runtime variant: {variant}")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def load_policy_config(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    cfg = dict(report["selected_by_calibration"]["config"])
    required = {"tp", "sl", "cap", "loss1", "loss2"}
    missing = sorted(required - set(cfg))
    if missing:
        raise RuntimeError(f"policy config missing keys: {missing}")
    return cfg


def parse_side_router_specs(values: list[str]) -> set[tuple[int, str]]:
    out: set[tuple[int, str]] = set()
    for raw in values:
        side_raw, router = raw.split("|", 1)
        side = int(side_raw)
        if side not in (-1, 1):
            raise RuntimeError(f"invalid side router spec side: {raw}")
        if not router:
            raise RuntimeError(f"invalid side router spec router: {raw}")
        out.add((side, router))
    return out


def parse_side_router_scales(values: list[str]) -> dict[tuple[int, str], float]:
    out: dict[tuple[int, str], float] = {}
    for raw in values:
        side_raw, router, scale_raw = raw.split("|", 2)
        side = int(side_raw)
        scale = float(scale_raw)
        if side not in (-1, 1):
            raise RuntimeError(f"invalid side router scale side: {raw}")
        if not router:
            raise RuntimeError(f"invalid side router scale router: {raw}")
        if not np.isfinite(scale) or scale < 0.0:
            raise RuntimeError(f"invalid side router scale value: {raw}")
        out[(side, router)] = scale
    return out


def parse_feature_veto_specs(values: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    valid_ops = {"<", "<=", ">", ">=", "==", "!="}
    for raw in values:
        side_raw, feature, op, threshold_raw = raw.split("|", 3)
        side = int(side_raw)
        if side not in (-1, 1):
            raise RuntimeError(f"invalid feature veto side: {raw}")
        if not feature:
            raise RuntimeError(f"invalid feature veto feature: {raw}")
        if op not in valid_ops:
            raise RuntimeError(f"invalid feature veto op: {raw}")
        threshold = float(threshold_raw)
        if not np.isfinite(threshold):
            raise RuntimeError(f"invalid feature veto threshold: {raw}")
        out.append({"side": side, "feature": feature, "op": op, "threshold": threshold, "raw": raw})
    return out


def feature_veto_hit(row: pd.Series, side: int, specs: list[dict[str, Any]]) -> tuple[bool, str]:
    for spec in specs:
        if int(spec["side"]) != int(side):
            continue
        feature = str(spec["feature"])
        if feature not in row.index:
            raise RuntimeError(f"feature veto column missing from live feature row: {feature}")
        value = float(row[feature])
        if not np.isfinite(value):
            raise RuntimeError(f"feature veto column non-finite in live feature row: {feature}={value}")
        threshold = float(spec["threshold"])
        op = str(spec["op"])
        hit = (
            (op == "<" and value < threshold)
            or (op == "<=" and value <= threshold)
            or (op == ">" and value > threshold)
            or (op == ">=" and value >= threshold)
            or (op == "==" and value == threshold)
            or (op == "!=" and value != threshold)
        )
        if hit:
            return True, f"paper_feature_veto:{feature}{op}{threshold:g}"
    return False, ""


def build_policy_controls(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "min_quality": float(args.min_quality),
        "min_confidence": float(args.min_confidence),
        "long_scale": float(args.long_scale),
        "short_scale": float(args.short_scale),
        "veto_side_routers": parse_side_router_specs(args.veto_side_router),
        "scale_side_routers": parse_side_router_scales(args.scale_side_router),
        "feature_vetoes": parse_feature_veto_specs(args.feature_veto),
        "max_hold_hours": None if args.max_hold_hours is None else float(args.max_hold_hours),
        "atr_tp_mult": None if args.atr_tp_mult is None else float(args.atr_tp_mult),
        "atr_sl_mult": None if args.atr_sl_mult is None else float(args.atr_sl_mult),
        "hard_stop_hours": None if args.hard_stop_hours is None else float(args.hard_stop_hours),
        "loss_after_hours": None if args.loss_after_hours is None else float(args.loss_after_hours),
        "loss_stop_move": None if args.loss_stop_move is None else float(args.loss_stop_move),
        "trail_after_hours": None if args.trail_after_hours is None else float(args.trail_after_hours),
        "trail_arm_move": None if args.trail_arm_move is None else float(args.trail_arm_move),
        "trail_giveback_move": None if args.trail_giveback_move is None else float(args.trail_giveback_move),
        "trail_floor_move": None if args.trail_floor_move is None else float(args.trail_floor_move),
        "stall_after_hours": None if args.stall_after_hours is None else float(args.stall_after_hours),
        "stall_lookback_hours": None if args.stall_lookback_hours is None else float(args.stall_lookback_hours),
        "stall_min_profit_move": None if args.stall_min_profit_move is None else float(args.stall_min_profit_move),
        "stall_slope_max": None if args.stall_slope_max is None else float(args.stall_slope_max),
        "paper_overlay_name": str(args.paper_overlay_name),
        "paper_basis": [
            "Conformal abstention: gate uncertain predictions instead of forcing a point action.",
            "CVaR / risk-sensitive control: reduce exposure after adverse tail outcomes.",
            "MoE time-series routing: specialize side/expert clusters instead of using a single dense rule.",
            "Distributional/risk-aware execution: use price-move brackets and optional time exit from live-available state.",
        ],
    }


def required_columns(parent: Omega462SourceParentLiveAdapter) -> set[str]:
    raw_required = {"timestamp", "open", "high", "low", "close", "rsi", "bb_width"}
    runtime_prefixes = ("regime3_current_sensitive_wide24_",)
    for component in parent.components.values():
        pos_cols = set(component.bundle["pos_cols"])
        for _, _, input_cols in component.models.values():
            for col in input_cols:
                if col in pos_cols or any(str(col).startswith(prefix) for prefix in runtime_prefixes):
                    continue
                raw_required.add(str(col))
    return raw_required


def load_frame(path: Path, start: str, end: str, raw_required: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = list(pd.read_csv(path, nrows=0).columns)
    missing = sorted(raw_required - set(cols))
    if missing:
        raise RuntimeError(f"{path} missing required model input columns: {missing[:80]}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    active = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)]
    if active.empty:
        raise RuntimeError(f"{path} produced empty split {start}..{end}")
    nan_required = active[list(sorted(raw_required - {"timestamp"}))].isna().sum()
    nan_required = nan_required[nan_required > 0].sort_values(ascending=False)
    if len(nan_required):
        raise RuntimeError(f"{path} active split required columns contain NaN: {nan_required.head(80).to_dict()}")
    return df, {
        "path": str(path),
        "source_rows": int(len(df)),
        "source_start": str(df["timestamp"].min()),
        "source_end": str(df["timestamp"].max()),
        "decision_start": start,
        "decision_end_exclusive": end,
        "decision_rows": int(len(active)),
        "required_column_count": int(len(raw_required)),
        "missing_required_columns": [],
        "required_nan_counts": {},
    }


def close_position(frame: pd.DataFrame, position: dict[str, Any], exit_i: int, reason: str, raw_move: float) -> dict[str, Any]:
    entry_i = int(position["entry_i"])
    entry_price = float(position["entry_price"])
    side = int(position["side"])
    window = frame.iloc[entry_i : exit_i + 1]
    if side > 0:
        mfe = float(window["high"].astype(float).max() / entry_price - 1.0)
        mae = float(window["low"].astype(float).min() / entry_price - 1.0)
    else:
        mfe = float(entry_price / window["low"].astype(float).min() - 1.0)
        mae = float(entry_price / window["high"].astype(float).max() - 1.0)
    net_per_notional = float(raw_move) - float(position["roundtrip_cost"])
    trade_return = net_per_notional * float(position["notional"])
    return {
        "entry_i": entry_i,
        "exit_i": int(exit_i),
        "entry_timestamp": pd.Timestamp(frame.iloc[entry_i]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_timestamp": pd.Timestamp(frame.iloc[exit_i]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
        "side": int(side),
        "reason": reason,
        "raw_exit_price_move": float(raw_move),
        "mfe_price_move": float(mfe),
        "mae_price_move": float(mae),
        "net_per_notional": float(net_per_notional),
        "trade_return": float(trade_return),
        "win": int(trade_return > 0.0),
        "hold_hours": float((exit_i - entry_i) / BARS_PER_HOUR),
        "notional": float(position["notional"]),
        "base_parent_notional": float(position["base_parent_notional"]),
        "margin_fraction": float(position["margin_fraction"]),
        "leverage": float(position["leverage"]),
        "entry_price": float(entry_price),
        "exit_price": float(frame.iloc[exit_i]["close"]),
        "tp_price_move": float(position["tp_price_move"]),
        "sl_price_move": float(position["sl_price_move"]),
        "roundtrip_cost": float(position["roundtrip_cost"]),
        "router_expert": str(position["router_expert"]),
        "parent_quality_score": float(position["parent_quality_score"]),
        "parent_confidence": float(position["parent_confidence"]),
        "overlay_loss_scale": float(position["overlay_loss_scale"]),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }


def current_raw_move(row: pd.Series, position: dict[str, Any]) -> float:
    entry_price = float(position["entry_price"])
    close_price = float(row["close"])
    side = int(position["side"])
    return close_price / entry_price - 1.0 if side > 0 else entry_price / close_price - 1.0


def position_mfe(frame: pd.DataFrame, position: dict[str, Any], exit_i: int) -> float:
    entry_i = int(position["entry_i"])
    entry_price = float(position["entry_price"])
    side = int(position["side"])
    window = frame.iloc[entry_i : exit_i + 1]
    if side > 0:
        return float(window["high"].astype(float).max() / entry_price - 1.0)
    return float(entry_price / window["low"].astype(float).min() - 1.0)


def maybe_optstop_exit(frame: pd.DataFrame, position: dict[str, Any], i: int, controls: dict[str, Any]) -> tuple[str, float] | None:
    hold_hours = float((i - int(position["entry_i"])) / BARS_PER_HOUR)
    row = frame.iloc[i]
    raw_move = current_raw_move(row, position)

    loss_after = controls["loss_after_hours"]
    if loss_after is not None and hold_hours >= float(loss_after):
        loss_stop = controls["loss_stop_move"]
        if loss_stop is not None and raw_move <= float(loss_stop):
            return "paper_optstop_loss_exit", raw_move

    trail_after = controls["trail_after_hours"]
    if trail_after is not None and hold_hours >= float(trail_after):
        mfe = position_mfe(frame, position, i)
        trail_arm = controls["trail_arm_move"]
        trail_giveback = controls["trail_giveback_move"]
        trail_floor = controls["trail_floor_move"]
        if (
            trail_arm is not None
            and trail_giveback is not None
            and trail_floor is not None
            and mfe >= float(trail_arm)
            and raw_move >= float(trail_floor)
            and raw_move <= mfe - float(trail_giveback)
        ):
            return "paper_optstop_trail_exit", raw_move

    stall_after = controls["stall_after_hours"]
    if stall_after is not None and hold_hours >= float(stall_after):
        stall_lookback = controls["stall_lookback_hours"]
        stall_min_profit = controls["stall_min_profit_move"]
        stall_slope_max = controls["stall_slope_max"]
        if stall_lookback is not None and stall_min_profit is not None and stall_slope_max is not None:
            lookback_bars = int(round(float(stall_lookback) * BARS_PER_HOUR))
            lb_i = max(int(position["entry_i"]), int(i) - lookback_bars)
            raw_move_lb = current_raw_move(frame.iloc[lb_i], position)
            if raw_move >= float(stall_min_profit) and raw_move - raw_move_lb <= float(stall_slope_max):
                return "paper_optstop_stall_exit", raw_move

    hard_stop = controls["hard_stop_hours"]
    if hard_stop is not None and hold_hours >= float(hard_stop):
        return "paper_optstop_time_exit", raw_move

    return None


def summarize_ledger(ledger: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, Any]:
    returns = ledger["trade_return"].astype(float).to_numpy(dtype=np.float64) if len(ledger) else np.array([], dtype=np.float64)
    additive_curve = np.cumsum(returns) if len(returns) else np.array([], dtype=np.float64)
    additive_peak = np.maximum.accumulate(np.concatenate([[0.0], additive_curve])) if len(returns) else np.array([0.0])
    additive_dd = np.concatenate([[0.0], additive_curve]) - additive_peak if len(returns) else np.array([0.0])
    compound_curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    compound_peak = np.maximum.accumulate(compound_curve)
    compound_dd = compound_curve / np.maximum(compound_peak, 1.0e-12) - 1.0
    worst = np.sort(returns)[: max(1, int(np.ceil(0.2 * len(returns))))] if len(returns) else np.array([])
    return {
        "additive_pnl": float(returns.sum()) if len(returns) else 0.0,
        "additive_pnl_pct": float(returns.sum() * 100.0) if len(returns) else 0.0,
        "additive_mdd": float(additive_dd.min()) if len(additive_dd) else 0.0,
        "additive_mdd_pct": float(additive_dd.min() * 100.0) if len(additive_dd) else 0.0,
        "compound_pnl": float(compound_curve[-1] - 1.0),
        "compound_pnl_pct": float((compound_curve[-1] - 1.0) * 100.0),
        "compound_mdd": float(compound_dd.min()),
        "compound_mdd_pct": float(compound_dd.min() * 100.0),
        "trades": int(len(ledger)),
        "wr": float((returns > 0.0).mean()) if len(returns) else None,
        "avg_hold_hours": float(ledger["hold_hours"].astype(float).mean()) if len(ledger) else 0.0,
        "max_hold_hours": float(ledger["hold_hours"].astype(float).max()) if len(ledger) else 0.0,
        "cvar20": float(worst.mean()) if len(worst) else 0.0,
        "long_trades": int((ledger["side"].astype(int) > 0).sum()) if len(ledger) else 0,
        "short_trades": int((ledger["side"].astype(int) < 0).sum()) if len(ledger) else 0,
        "avg_notional": float(ledger["notional"].astype(float).mean()) if len(ledger) else 0.0,
        "max_notional": float(ledger["notional"].astype(float).max()) if len(ledger) else 0.0,
        "max_leverage": float(ledger["leverage"].astype(float).max()) if len(ledger) else 0.0,
        "reason_counts": dict(Counter(ledger["reason"].astype(str))) if len(ledger) else {},
        "cash_decisions": int((decisions["action"].astype(int) == 0).sum()) if len(decisions) else 0,
        "entry_decisions": int((decisions["action"].astype(int) != 0).sum()) if len(decisions) else 0,
        "decision_rows": int(len(decisions)),
        "ledger_replay_trace_count": int(decisions["ledger_replay_used"].astype(bool).sum()) if len(decisions) else 0,
        "non_live_native_trace_count": int((~decisions["source_parent_live_native_adapter"].astype(bool)).sum()) if len(decisions) else 0,
        "non_minus_one_policy_row_count": int((decisions["source_parent_policy_row"].astype(int) != -1).sum()) if len(decisions) else 0,
        "paper_gate_counts": dict(Counter(decisions["paper_gate_reason"].astype(str))) if "paper_gate_reason" in decisions else {},
    }


def run_split(
    *,
    split: str,
    feature_path: Path,
    start: str,
    end: str,
    policy: dict[str, Any],
    controls: dict[str, Any],
    raw_required: set[str],
    out_dir: Path,
    parent_runtime_variant: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = make_parent(parent_runtime_variant)
    frame, source_audit = load_frame(feature_path, start, end, raw_required)
    work = parent.regime3._append_current(frame.copy())
    atr = atr_pct_series(work)
    start_i = int(np.flatnonzero(work["timestamp"].to_numpy() >= pd.Timestamp(start).to_datetime64())[0])
    end_idx = np.flatnonzero(work["timestamp"].to_numpy() < pd.Timestamp(end).to_datetime64())
    end_i = int(end_idx[-1]) if len(end_idx) else len(work) - 1

    decisions: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    overlay_loss_streak = 0

    for i in range(start_i, end_i + 1):
        row = work.iloc[i]
        now = pd.Timestamp(row["timestamp"])
        if (i - start_i) % 5000 == 0:
            print(
                json.dumps(
                    {
                        "split": split,
                        "done": int(i - start_i),
                        "total": int(end_i - start_i + 1),
                        "timestamp": str(now),
                        "closed": int(len(ledger)),
                        "position": None if position is None else int(position["side"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if position is not None and i > int(position["entry_i"]):
            side = int(position["side"])
            entry_price = float(position["entry_price"])
            high = float(row["high"])
            low = float(row["low"])
            tp_move = float(position["tp_price_move"])
            sl_move = float(position["sl_price_move"])
            if side > 0:
                hit_sl = (low / entry_price - 1.0) <= -sl_move
                hit_tp = (high / entry_price - 1.0) >= tp_move
            else:
                hit_sl = (entry_price / high - 1.0) <= -sl_move
                hit_tp = (entry_price / low - 1.0) >= tp_move
            if hit_sl or hit_tp:
                reason = "fresh_policy_sl" if hit_sl else "fresh_policy_tp"
                raw_move = -sl_move if hit_sl else tp_move
                closed = close_position(work, position, i, reason, raw_move)
                ledger.append(closed)
                parent.record_closed_trade(
                    exit_timestamp=closed["exit_timestamp"],
                    net_per_notional=float(closed["net_per_notional"]),
                )
                overlay_loss_streak = overlay_loss_streak + 1 if float(closed["trade_return"]) <= 0.0 else 0
                position = None
                continue
            optstop = maybe_optstop_exit(work, position, i, controls)
            if optstop is not None:
                reason, raw_move = optstop
                closed = close_position(work, position, i, reason, raw_move)
                ledger.append(closed)
                parent.record_closed_trade(
                    exit_timestamp=closed["exit_timestamp"],
                    net_per_notional=float(closed["net_per_notional"]),
                )
                overlay_loss_streak = overlay_loss_streak + 1 if float(closed["trade_return"]) <= 0.0 else 0
                position = None
                continue
            max_hold_hours = controls["max_hold_hours"]
            if max_hold_hours is not None and (i - int(position["entry_i"])) / BARS_PER_HOUR >= float(max_hold_hours):
                close_price = float(row["close"])
                raw_move = close_price / entry_price - 1.0 if side > 0 else entry_price / close_price - 1.0
                closed = close_position(work, position, i, "fresh_policy_time_exit", raw_move)
                ledger.append(closed)
                parent.record_closed_trade(
                    exit_timestamp=closed["exit_timestamp"],
                    net_per_notional=float(closed["net_per_notional"]),
                )
                overlay_loss_streak = overlay_loss_streak + 1 if float(closed["trade_return"]) <= 0.0 else 0
                position = None
                continue

        parent_dec = parent_decision_at(parent, work.iloc[i : i + 1], float(atr[i]), now)
        parent_trace = dict(parent_dec.trace or {})
        in_position = position is not None
        gate_reason = ""
        router = str(parent_dec.router_expert)
        if (
            not in_position
            and int(parent_dec.action) != 0
            and int(parent_dec.side) != 0
            and float(parent_dec.notional_exposure) > EPS
        ):
            if float(parent_dec.quality_score) < float(controls["min_quality"]):
                gate_reason = "paper_conformal_quality_abstain"
            elif float(parent_dec.confidence) < float(controls["min_confidence"]):
                gate_reason = "paper_conformal_confidence_abstain"
            elif (int(parent_dec.side), router) in controls["veto_side_routers"]:
                gate_reason = "paper_moe_side_router_veto"
            else:
                feature_hit, feature_reason = feature_veto_hit(row, int(parent_dec.side), controls["feature_vetoes"])
                if feature_hit:
                    gate_reason = feature_reason
        decisions.append(
            {
                "split": split,
                "row": int(i),
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "action": int(parent_dec.action),
                "side": int(parent_dec.side),
                "notional": float(parent_dec.notional_exposure),
                "margin_fraction": float(parent_dec.position_fraction),
                "leverage": float(parent_dec.leverage),
                "quality_score": float(parent_dec.quality_score),
                "confidence": float(parent_dec.confidence),
                "router_expert": str(parent_dec.router_expert),
                "reason": str(parent_trace.get("omega462_reason", "")),
                "paper_gate_reason": gate_reason,
                "ignored_because_in_position": bool(in_position),
                "ledger_replay_used": bool(parent_trace.get("ledger_replay_used", True)),
                "source_parent_live_native_adapter": bool(parent_trace.get("source_parent_live_native_adapter", False)),
                "source_parent_policy_row": int(parent_trace.get("source_parent_policy_row", -999)),
                "fresh_forward_bar_by_bar": True,
                "future_rows_used_for_entry": False,
            }
        )
        if in_position:
            continue
        if int(parent_dec.action) == 0 or int(parent_dec.side) == 0 or float(parent_dec.notional_exposure) <= EPS:
            continue
        if gate_reason:
            continue

        loss_scale = 1.0
        if overlay_loss_streak >= 2:
            loss_scale = float(policy["loss2"])
        elif overlay_loss_streak == 1:
            loss_scale = float(policy["loss1"])
        leverage = float(parent_dec.leverage)
        side_scale = float(controls["long_scale"] if int(parent_dec.side) > 0 else controls["short_scale"])
        router_scale = float(controls["scale_side_routers"].get((int(parent_dec.side), router), 1.0))
        notional = min(float(parent_dec.notional_exposure), float(policy["cap"])) * loss_scale * side_scale * router_scale
        if notional <= EPS:
            continue
        margin = notional / max(leverage, EPS)
        if abs(margin * leverage - notional) > 1.0e-8:
            raise RuntimeError("overlay violates notional=margin_fraction*leverage")
        tp_move = float(policy["tp"])
        sl_move = float(policy["sl"])
        if controls["atr_tp_mult"] is not None:
            tp_move = max(tp_move, float(atr[i]) * float(controls["atr_tp_mult"]))
        if controls["atr_sl_mult"] is not None:
            sl_move = max(sl_move, float(atr[i]) * float(controls["atr_sl_mult"]))
        position = {
            "entry_i": int(i),
            "side": int(parent_dec.side),
            "entry_price": float(row["close"]),
            "notional": float(notional),
            "base_parent_notional": float(parent_dec.notional_exposure),
            "margin_fraction": float(margin),
            "leverage": float(leverage),
            "tp_price_move": float(tp_move),
            "sl_price_move": float(sl_move),
            "roundtrip_cost": float(ROUNDTRIP_COST_DEFAULT),
            "router_expert": str(parent_dec.router_expert),
            "parent_quality_score": float(parent_dec.quality_score),
            "parent_confidence": float(parent_dec.confidence),
            "overlay_loss_scale": float(loss_scale),
        }

    open_position = None
    if position is not None:
        open_position = {
            "entry_i": int(position["entry_i"]),
            "entry_timestamp": pd.Timestamp(work.iloc[int(position["entry_i"])]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "last_timestamp": pd.Timestamp(work.iloc[end_i]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "side": int(position["side"]),
            "entry_price": float(position["entry_price"]),
            "last_close": float(work.iloc[end_i]["close"]),
            "notional": float(position["notional"]),
            "margin_fraction": float(position["margin_fraction"]),
            "leverage": float(position["leverage"]),
            "note": "open position not force-closed at split_end",
        }

    decisions_df = pd.DataFrame(decisions)
    ledger_df = pd.DataFrame(ledger)
    decisions_path = out_dir / f"{split}_bar_forward_decisions.csv"
    ledger_path = out_dir / f"{split}_bar_forward_ledger.csv"
    decisions_df.to_csv(decisions_path, index=False)
    ledger_df.to_csv(ledger_path, index=False)

    metrics = summarize_ledger(ledger_df, decisions_df)
    metrics["open_position"] = open_position
    return metrics, {
        "source": source_audit,
        "ledger": str(ledger_path),
        "decisions": str(decisions_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = load_policy_config(Path(args.policy_config))
    if args.tp_override is not None:
        policy["tp"] = float(args.tp_override)
    if args.sl_override is not None:
        policy["sl"] = float(args.sl_override)
    if args.cap_override is not None:
        policy["cap"] = float(args.cap_override)
    if args.loss1_override is not None:
        policy["loss1"] = float(args.loss1_override)
    if args.loss2_override is not None:
        policy["loss2"] = float(args.loss2_override)
    controls = build_policy_controls(args)
    parent_for_contract = make_parent(str(args.parent_runtime_variant))
    raw_required = required_columns(parent_for_contract)
    del parent_for_contract

    validation_metrics, validation_artifacts = run_split(
        split="validation",
        feature_path=Path(args.validation_features),
        start=args.validation_start,
        end=args.validation_end,
        policy=policy,
        controls=controls,
        raw_required=raw_required,
        out_dir=out_dir,
        parent_runtime_variant=str(args.parent_runtime_variant),
    )
    oos_metrics, oos_artifacts = run_split(
        split="oos",
        feature_path=Path(args.oos_features),
        start=args.oos_start,
        end=args.oos_end,
        policy=policy,
        controls=controls,
        raw_required=raw_required,
        out_dir=out_dir,
        parent_runtime_variant=str(args.parent_runtime_variant),
    )

    report = {
        "schema_version": "omega462.hf_policy_overlay.bar_forward_val_oos_report.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_under_test": str(args.model_under_test),
        "parent_runtime_variant": str(args.parent_runtime_variant),
        "fresh_forward_definition": "fixed historical validation/OOS split, causal 5m bar-by-bar walk-forward",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_entry_timestamps_used": False,
        "saved_parent_exit_timestamps_used": False,
        "parent_decision_cache_used": False,
        "future_rows_used_for_entry": False,
        "feature_frame_replay_only": True,
        "promotion_evidence_allowed": True,
        "policy_config_source": str(Path(args.policy_config)),
        "policy": {
            "tp_price_move": float(policy["tp"]),
            "sl_price_move": float(policy["sl"]),
            "notional_cap": float(policy["cap"]),
            "loss1_scale": float(policy["loss1"]),
            "loss2_scale": float(policy["loss2"]),
            "same_bar_bracket_ambiguity": "stop_loss_first",
            "dataset_end_force_close": False,
            "roundtrip_cost": float(ROUNDTRIP_COST_DEFAULT),
            "parent_runtime_variant": str(args.parent_runtime_variant),
        },
        "paper_overlay_controls": {
            "paper_overlay_name": str(controls["paper_overlay_name"]),
            "paper_basis": list(controls["paper_basis"]),
            "min_quality": float(controls["min_quality"]),
            "min_confidence": float(controls["min_confidence"]),
            "long_scale": float(controls["long_scale"]),
            "short_scale": float(controls["short_scale"]),
            "veto_side_routers": [
                {"side": side, "router_expert": router}
                for side, router in sorted(controls["veto_side_routers"])
            ],
            "feature_vetoes": [
                {
                    "side": int(spec["side"]),
                    "feature": str(spec["feature"]),
                    "op": str(spec["op"]),
                    "threshold": float(spec["threshold"]),
                }
                for spec in controls["feature_vetoes"]
            ],
            "scale_side_routers": [
                {"side": side, "router_expert": router, "scale": scale}
                for (side, router), scale in sorted(controls["scale_side_routers"].items())
            ],
            "max_hold_hours": controls["max_hold_hours"],
            "atr_tp_mult": controls["atr_tp_mult"],
            "atr_sl_mult": controls["atr_sl_mult"],
            "hard_stop_hours": controls["hard_stop_hours"],
            "loss_after_hours": controls["loss_after_hours"],
            "loss_stop_move": controls["loss_stop_move"],
            "trail_after_hours": controls["trail_after_hours"],
            "trail_arm_move": controls["trail_arm_move"],
            "trail_giveback_move": controls["trail_giveback_move"],
            "trail_floor_move": controls["trail_floor_move"],
            "stall_after_hours": controls["stall_after_hours"],
            "stall_lookback_hours": controls["stall_lookback_hours"],
            "stall_min_profit_move": controls["stall_min_profit_move"],
            "stall_slope_max": controls["stall_slope_max"],
            "calibration_scope": "no validation/OOS trade ledger is used as model input; optional thresholds are CLI policy parameters",
        },
        "splits": {
            "validation": {
                "start": args.validation_start,
                "end_exclusive": args.validation_end,
                **validation_artifacts,
            },
            "oos": {
                "start": args.oos_start,
                "end_exclusive": args.oos_end,
                **oos_artifacts,
            },
        },
        "metrics": {
            "validation": validation_metrics,
            "oos": oos_metrics,
        },
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
        },
    }
    write_json(out_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-features", default=str(DEFAULT_VALIDATION_FEATURES))
    parser.add_argument("--oos-features", default=str(DEFAULT_OOS_FEATURES))
    parser.add_argument("--policy-config", default=str(DEFAULT_POLICY_CONFIG))
    parser.add_argument("--validation-start", default="2025-09-01 00:00:00")
    parser.add_argument("--validation-end", default="2026-01-01 00:00:00")
    parser.add_argument("--oos-start", default="2026-01-01 00:00:00")
    parser.add_argument("--oos-end", default="2026-04-01 00:00:00")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model-under-test", default=MODEL_ID)
    parser.add_argument("--parent-runtime-variant", choices=["source_v5", "cap220_no_v5"], default="source_v5")
    parser.add_argument("--paper-overlay-name", default="baseline")
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--long-scale", type=float, default=1.0)
    parser.add_argument("--short-scale", type=float, default=1.0)
    parser.add_argument("--veto-side-router", action="append", default=[])
    parser.add_argument("--scale-side-router", action="append", default=[])
    parser.add_argument("--feature-veto", action="append", default=[])
    parser.add_argument("--tp-override", type=float, default=None)
    parser.add_argument("--sl-override", type=float, default=None)
    parser.add_argument("--cap-override", type=float, default=None)
    parser.add_argument("--loss1-override", type=float, default=None)
    parser.add_argument("--loss2-override", type=float, default=None)
    parser.add_argument("--max-hold-hours", type=float, default=None)
    parser.add_argument("--atr-tp-mult", type=float, default=None)
    parser.add_argument("--atr-sl-mult", type=float, default=None)
    parser.add_argument("--hard-stop-hours", type=float, default=None)
    parser.add_argument("--loss-after-hours", type=float, default=None)
    parser.add_argument("--loss-stop-move", type=float, default=None)
    parser.add_argument("--trail-after-hours", type=float, default=None)
    parser.add_argument("--trail-arm-move", type=float, default=None)
    parser.add_argument("--trail-giveback-move", type=float, default=None)
    parser.add_argument("--trail-floor-move", type=float, default=None)
    parser.add_argument("--stall-after-hours", type=float, default=None)
    parser.add_argument("--stall-lookback-hours", type=float, default=None)
    parser.add_argument("--stall-min-profit-move", type=float, default=None)
    parser.add_argument("--stall-slope-max", type=float, default=None)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2, default=json_default), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
