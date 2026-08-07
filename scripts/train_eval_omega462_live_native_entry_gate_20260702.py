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

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_omega462_hf_policy_bar_forward_val_oos_20260702 import (  # noqa: E402
    BARS_PER_HOUR,
    close_position,
    current_raw_move,
    json_default,
    load_frame,
    make_parent,
    required_columns,
    summarize_ledger,
    write_json,
)
from tmp.causal_regen_20260516.extended_oos_20260702.run_omega5_additional_oos_replay import (  # noqa: E402
    ROUNDTRIP_COST_DEFAULT,
    atr_pct_series,
    parent_decision_at,
)
from trading_bot_modules.omega4_6_2_source_parent_live import EPS, Omega462SourceParentLiveAdapter  # noqa: E402


MODEL_ID = "omega462_live_native_entry_gate_20260702"
DEFAULT_FEATURES_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
DEFAULT_TRAIN_FEATURES = ROOT / "tmp/causal_regen_20260516/live_native_inputs/training_features_2024_2025_base_20260702.csv"
DEFAULT_OOS_FEATURES = (
    ROOT
    / "tmp/causal_regen_20260516/extended_oos_20260702/"
    / "training_features_2026_0101_0702_m7_ai_for_omega5_parity.csv"
)
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/omega462_live_native_entry_gate_20260702"
CURRENT_PREFIX = "regime3_current_sensitive_wide24_"


ROUTER_VALUES = (
    "h48qual:bull",
    "h48qual:bear",
    "h48qual:chop",
    "zig075:bull",
    "zig075:bear",
    "zig075:chop",
)
EXPERT_VALUES = ("bull", "bear", "chop")
ALIASES = ("h48qual", "zig075")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def load_policy(args: argparse.Namespace) -> dict[str, float]:
    return {
        "tp": float(args.tp),
        "sl": float(args.sl),
        "cap": float(args.cap),
        "max_hold_hours": float(args.max_hold_hours),
    }


def frame_with_runtime_features(
    *,
    feature_path: Path,
    start: str,
    end: str,
    parent_variant: str,
) -> tuple[Omega462SourceParentLiveAdapter, pd.DataFrame, np.ndarray, int, int, dict[str, Any]]:
    parent_for_contract = make_parent(parent_variant)
    raw_required = required_columns(parent_for_contract)
    del parent_for_contract

    parent = make_parent(parent_variant)
    frame, source_audit = load_frame(feature_path, start, end, raw_required)
    work = parent.regime3._append_current(frame.copy())
    atr = atr_pct_series(work)
    ts = work["timestamp"].to_numpy()
    start_i = int(np.flatnonzero(ts >= pd.Timestamp(start).to_datetime64())[0])
    end_idx = np.flatnonzero(ts < pd.Timestamp(end).to_datetime64())
    end_i = int(end_idx[-1]) if len(end_idx) else len(work) - 1
    return parent, work, atr, start_i, end_i, source_audit


def feature_row(
    *,
    work: pd.DataFrame,
    atr: np.ndarray,
    i: int,
    parent_dec: Any,
    parent_trace: dict[str, Any],
    overlay_loss_streak: int,
) -> dict[str, float]:
    row = work.iloc[i]
    now = pd.Timestamp(row["timestamp"])
    close = safe_float(row["close"], 1.0)

    out: dict[str, float] = {
        "side": float(int(parent_dec.side)),
        "notional": safe_float(parent_dec.notional_exposure),
        "margin_fraction": safe_float(parent_dec.position_fraction),
        "leverage": safe_float(parent_dec.leverage),
        "quality_score": safe_float(parent_dec.quality_score),
        "confidence": safe_float(parent_dec.confidence),
        "atr_pct": safe_float(atr[i]),
        "rsi": safe_float(row.get("rsi")),
        "bb_width": safe_float(row.get("bb_width")),
        "overlay_loss_streak": float(min(int(overlay_loss_streak), 3)),
        "hour_sin": float(np.sin(2.0 * np.pi * now.hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * now.hour / 24.0)),
        "dow_sin": float(np.sin(2.0 * np.pi * now.dayofweek / 7.0)),
        "dow_cos": float(np.cos(2.0 * np.pi * now.dayofweek / 7.0)),
    }
    for bars, name in ((12, "1h"), (72, "6h"), (288, "24h")):
        if i >= bars:
            prev = safe_float(work.iloc[i - bars]["close"], close)
            out[f"ret_{name}"] = close / max(prev, EPS) - 1.0
        else:
            out[f"ret_{name}"] = 0.0
    if i >= 288:
        rets = pd.to_numeric(work.iloc[i - 288 : i + 1]["close"], errors="raise").pct_change().dropna()
        out["rv_24h"] = float(rets.std()) if len(rets) else 0.0
    else:
        out["rv_24h"] = 0.0

    for col in (
        f"{CURRENT_PREFIX}bull_prob",
        f"{CURRENT_PREFIX}bear_prob",
        f"{CURRENT_PREFIX}chop_prob",
        f"{CURRENT_PREFIX}confidence",
        f"{CURRENT_PREFIX}margin",
        f"{CURRENT_PREFIX}entropy",
    ):
        out[col] = safe_float(row.get(col))

    router = str(parent_dec.router_expert)
    for value in ROUTER_VALUES:
        out[f"router_{value}"] = 1.0 if router == value else 0.0
    for alias in ALIASES:
        for expert in EXPERT_VALUES:
            out[f"router_alias_{alias}_{expert}"] = 1.0 if router == f"{alias}:{expert}" else 0.0

    comps = {str(c.get("alias")): c for c in parent_trace.get("component_predictions", [])}
    for alias in ALIASES:
        comp = comps.get(alias, {})
        out[f"{alias}_final_action"] = safe_float(comp.get("final_action"))
        out[f"{alias}_side"] = safe_float(comp.get("side"))
        out[f"{alias}_quality"] = safe_float(comp.get("quality_for_action"))
        out[f"{alias}_confidence"] = safe_float(comp.get("confidence"))
        out[f"{alias}_base_notional"] = safe_float(comp.get("base_notional"))
        out[f"{alias}_notional"] = safe_float(comp.get("notional"))
        out[f"{alias}_margin_fraction"] = safe_float(comp.get("margin_fraction"))
        out[f"{alias}_leverage"] = safe_float(comp.get("leverage"))
        out[f"{alias}_sidecar_score"] = safe_float(comp.get("sidecar_score"))
        out[f"{alias}_loss_governor_scale"] = safe_float(comp.get("loss_governor_scale"))
        out[f"{alias}_cap220_skipped"] = 1.0 if bool(comp.get("cap220_skipped", False)) else 0.0
        expert = str(comp.get("expert", ""))
        for value in EXPERT_VALUES:
            out[f"{alias}_expert_{value}"] = 1.0 if expert == value else 0.0
    return out


def close_if_needed(
    *,
    work: pd.DataFrame,
    position: dict[str, Any],
    i: int,
    policy: dict[str, float],
) -> tuple[dict[str, Any] | None, str, float]:
    if i <= int(position["entry_i"]):
        return None, "", 0.0
    row = work.iloc[i]
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
        return close_position(work, position, i, reason, raw_move), reason, raw_move

    hold_hours = (i - int(position["entry_i"])) / BARS_PER_HOUR
    if hold_hours >= float(policy["max_hold_hours"]):
        raw_move = current_raw_move(row, position)
        return close_position(work, position, i, "fresh_policy_time_exit", raw_move), "fresh_policy_time_exit", raw_move
    return None, "", 0.0


def counterfactual_entry_label(
    *,
    work: pd.DataFrame,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    ts_arr: np.ndarray,
    i: int,
    end_i: int,
    parent_dec: Any,
    policy: dict[str, float],
) -> dict[str, Any] | None:
    leverage = safe_float(parent_dec.leverage)
    notional = min(safe_float(parent_dec.notional_exposure), float(policy["cap"]))
    if leverage <= EPS or notional <= EPS or i >= end_i:
        return None
    margin = notional / max(leverage, EPS)
    if abs(margin * leverage - notional) > 1.0e-8:
        raise RuntimeError("counterfactual label violates notional=margin_fraction*leverage")

    max_hold_bars = int(round(float(policy["max_hold_hours"]) * BARS_PER_HOUR))
    forced_exit_i = i + max_hold_bars
    if max_hold_bars <= 0 or forced_exit_i > end_i:
        return None

    entry_price = float(close_arr[i])
    side = int(parent_dec.side)
    tp_move = float(policy["tp"])
    sl_move = float(policy["sl"])
    highs = high_arr[i + 1 : forced_exit_i + 1]
    lows = low_arr[i + 1 : forced_exit_i + 1]
    if side > 0:
        sl_hits = (lows / entry_price - 1.0) <= -sl_move
        tp_hits = (highs / entry_price - 1.0) >= tp_move
    else:
        sl_hits = (entry_price / highs - 1.0) <= -sl_move
        tp_hits = (entry_price / lows - 1.0) >= tp_move

    hit_any = sl_hits | tp_hits
    if bool(hit_any.any()):
        offset = int(np.argmax(hit_any))
        exit_i = i + 1 + offset
        if bool(sl_hits[offset]):
            reason = "counterfactual_fresh_policy_sl"
            raw_move = -sl_move
        else:
            reason = "counterfactual_fresh_policy_tp"
            raw_move = tp_move
    else:
        exit_i = forced_exit_i
        reason = "counterfactual_fresh_policy_time_exit"
        raw_move = close_arr[exit_i] / entry_price - 1.0 if side > 0 else entry_price / close_arr[exit_i] - 1.0

    high_window = high_arr[i : exit_i + 1]
    low_window = low_arr[i : exit_i + 1]
    if side > 0:
        mfe = float(high_window.max() / entry_price - 1.0)
        mae = float(low_window.min() / entry_price - 1.0)
    else:
        mfe = float(entry_price / low_window.min() - 1.0)
        mae = float(entry_price / high_window.max() - 1.0)
    net_per_notional = float(raw_move) - float(ROUNDTRIP_COST_DEFAULT)
    trade_return = net_per_notional * float(notional)
    return {
        "entry_i": int(i),
        "exit_i": int(exit_i),
        "entry_timestamp": pd.Timestamp(ts_arr[i]).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_timestamp": pd.Timestamp(ts_arr[exit_i]).strftime("%Y-%m-%d %H:%M:%S"),
        "side": int(side),
        "reason": reason,
        "raw_exit_price_move": float(raw_move),
        "mfe_price_move": float(mfe),
        "mae_price_move": float(mae),
        "net_per_notional": float(net_per_notional),
        "trade_return": float(trade_return),
        "win": int(trade_return > 0.0),
        "hold_hours": float((exit_i - i) / BARS_PER_HOUR),
        "notional": float(notional),
        "base_parent_notional": safe_float(parent_dec.notional_exposure),
        "margin_fraction": float(margin),
        "leverage": float(leverage),
        "entry_price": float(entry_price),
        "exit_price": float(close_arr[exit_i]),
        "tp_price_move": float(tp_move),
        "sl_price_move": float(sl_move),
        "roundtrip_cost": float(ROUNDTRIP_COST_DEFAULT),
        "router_expert": str(parent_dec.router_expert),
        "parent_quality_score": safe_float(parent_dec.quality_score),
        "parent_confidence": safe_float(parent_dec.confidence),
        "overlay_loss_scale": 1.0,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }


def summarize_entries(entries: pd.DataFrame) -> dict[str, Any]:
    if entries.empty:
        return {"rows": 0, "pnl": 0.0, "wr": None}
    out = {
        "rows": int(len(entries)),
        "pnl": float(entries["trade_return"].sum()),
        "pnl_pct": float(entries["trade_return"].sum() * 100.0),
        "wr": float((entries["trade_return"].astype(float) > 0.0).mean()),
        "long_rows": int((entries["side"].astype(int) > 0).sum()),
        "short_rows": int((entries["side"].astype(int) < 0).sum()),
        "avg_hold_hours": float(entries["hold_hours"].astype(float).mean()),
        "max_hold_hours": float(entries["hold_hours"].astype(float).max()),
    }
    if "candidate_was_in_position" in entries.columns:
        was_in_position = entries["candidate_was_in_position"].astype(int)
        out["flat_candidate_rows"] = int((was_in_position == 0).sum())
        out["in_position_counterfactual_rows"] = int((was_in_position == 1).sum())
    return out


def simulate(
    *,
    split: str,
    feature_path: Path,
    start: str,
    end: str,
    parent_variant: str,
    policy: dict[str, float],
    out_dir: Path,
    gate_model: HistGradientBoostingRegressor | None = None,
    feature_cols: list[str] | None = None,
    gate_threshold: float | None = None,
    collect_entry_labels: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent, work, atr, start_i, end_i, source_audit = frame_with_runtime_features(
        feature_path=feature_path,
        start=start,
        end=end,
        parent_variant=parent_variant,
    )
    high_arr = pd.to_numeric(work["high"], errors="raise").to_numpy(dtype=np.float64)
    low_arr = pd.to_numeric(work["low"], errors="raise").to_numpy(dtype=np.float64)
    close_arr = pd.to_numeric(work["close"], errors="raise").to_numpy(dtype=np.float64)
    ts_arr = work["timestamp"].to_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    entry_rows: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    overlay_loss_streak = 0
    gate_counts: Counter[str] = Counter()

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

        if position is not None:
            closed, _, _ = close_if_needed(work=work, position=position, i=i, policy=policy)
            if closed is not None:
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
        signal = int(parent_dec.action) != 0 and int(parent_dec.side) != 0 and float(parent_dec.notional_exposure) > EPS
        candidate = not in_position and signal
        gate_score = np.nan
        gate_reason = ""
        features: dict[str, float] | None = None
        if signal:
            features = feature_row(
                work=work,
                atr=atr,
                i=i,
                parent_dec=parent_dec,
                parent_trace=parent_trace,
                overlay_loss_streak=overlay_loss_streak,
            )
            if collect_entry_labels:
                label = counterfactual_entry_label(
                    work=work,
                    high_arr=high_arr,
                    low_arr=low_arr,
                    close_arr=close_arr,
                    ts_arr=ts_arr,
                    i=i,
                    end_i=end_i,
                    parent_dec=parent_dec,
                    policy=policy,
                )
                if label is not None:
                    labeled = {
                        **features,
                        "split": split,
                        "entry_i": int(i),
                        "entry_timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": float(row["close"]),
                        "router_expert": str(parent_dec.router_expert),
                        "candidate_was_in_position": int(in_position),
                    }
                    labeled.update(
                        {
                            "exit_i": int(label["exit_i"]),
                            "exit_timestamp": str(label["exit_timestamp"]),
                            "reason": str(label["reason"]),
                            "raw_exit_price_move": float(label["raw_exit_price_move"]),
                            "mfe_price_move": float(label["mfe_price_move"]),
                            "mae_price_move": float(label["mae_price_move"]),
                            "net_per_notional": float(label["net_per_notional"]),
                            "trade_return": float(label["trade_return"]),
                            "win": int(label["win"]),
                            "hold_hours": float(label["hold_hours"]),
                        }
                    )
                    entry_rows.append(labeled)

        if candidate:
            if gate_model is not None:
                if feature_cols is None or gate_threshold is None:
                    raise RuntimeError("gate_model requires feature_cols and gate_threshold")
                if features is None:
                    raise RuntimeError("entry features missing for gate candidate")
                x = pd.DataFrame([{col: features.get(col, 0.0) for col in feature_cols}], columns=feature_cols)
                gate_score = float(gate_model.predict(x)[0])
                if gate_score < float(gate_threshold):
                    gate_reason = "live_native_entry_gate_veto"
                    gate_counts[gate_reason] += 1
                else:
                    gate_counts["live_native_entry_gate_allow"] += 1

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
                "ignored_because_in_position": bool(in_position),
                "entry_gate_score": gate_score,
                "entry_gate_reason": gate_reason,
                "ledger_replay_used": bool(parent_trace.get("ledger_replay_used", True)),
                "source_parent_live_native_adapter": bool(parent_trace.get("source_parent_live_native_adapter", False)),
                "source_parent_policy_row": int(parent_trace.get("source_parent_policy_row", -999)),
                "fresh_forward_bar_by_bar": True,
                "future_rows_used_for_entry": False,
            }
        )
        if not candidate or gate_reason:
            continue

        leverage = float(parent_dec.leverage)
        notional = min(float(parent_dec.notional_exposure), float(policy["cap"]))
        if notional <= EPS:
            continue
        margin = notional / max(leverage, EPS)
        if abs(margin * leverage - notional) > 1.0e-8:
            raise RuntimeError("entry gate replay violates notional=margin_fraction*leverage")
        position = {
            "entry_i": int(i),
            "side": int(parent_dec.side),
            "entry_price": float(row["close"]),
            "notional": float(notional),
            "base_parent_notional": float(parent_dec.notional_exposure),
            "margin_fraction": float(margin),
            "leverage": float(leverage),
            "tp_price_move": float(policy["tp"]),
            "sl_price_move": float(policy["sl"]),
            "roundtrip_cost": float(ROUNDTRIP_COST_DEFAULT),
            "router_expert": str(parent_dec.router_expert),
            "parent_quality_score": float(parent_dec.quality_score),
            "parent_confidence": float(parent_dec.confidence),
            "overlay_loss_scale": 1.0,
        }
    decisions_df = pd.DataFrame(decisions)
    ledger_df = pd.DataFrame(ledger)
    entries_df = pd.DataFrame(entry_rows)
    decisions_path = out_dir / f"{split}_decisions.csv"
    ledger_path = out_dir / f"{split}_ledger.csv"
    entries_path = out_dir / f"{split}_entry_labels.csv"
    decisions_df.to_csv(decisions_path, index=False)
    ledger_df.to_csv(ledger_path, index=False)
    if collect_entry_labels:
        entries_df.to_csv(entries_path, index=False)
    metrics = summarize_ledger(ledger_df, decisions_df)
    metrics["entry_gate_counts"] = dict(gate_counts)
    return metrics, {
        "source": source_audit,
        "decisions": str(decisions_path),
        "ledger": str(ledger_path),
        "entry_labels": str(entries_path) if collect_entry_labels else "",
        "entry_label_summary": summarize_entries(entries_df) if collect_entry_labels else {},
    }


def select_threshold(calib_entries: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    if calib_entries.empty:
        raise RuntimeError("cannot select gate threshold from empty calibration entries")
    candidates = sorted(set(float(x) for x in np.quantile(scores, [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9])))
    rows: list[dict[str, Any]] = []
    min_trades = max(10, int(np.floor(len(calib_entries) * 0.25)))
    for threshold in candidates:
        keep = scores >= threshold
        picked = calib_entries.loc[keep]
        if picked.empty:
            pnl = 0.0
            wr = 0.0
        else:
            pnl = float(picked["trade_return"].sum())
            wr = float((picked["trade_return"].astype(float) > 0.0).mean())
        rows.append(
            {
                "threshold": float(threshold),
                "trades": int(len(picked)),
                "pnl": pnl,
                "wr": wr,
                "avg_score": float(scores[keep].mean()) if keep.any() else 0.0,
                "eligible": int(len(picked) >= min_trades),
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        eligible = rows
    selected = max(eligible, key=lambda row: (row["pnl"], row["wr"], row["trades"]))
    return {
        "selected": selected,
        "grid": rows,
        "selection_method": "train-only chronological calibration; filter counterfactual parent signal rows by predicted return",
        "min_trades": int(min_trades),
    }


def train_gate(train_entries: pd.DataFrame, calib_entries: pd.DataFrame, out_dir: Path) -> tuple[Any, list[str], dict[str, Any]]:
    if train_entries.empty:
        raise RuntimeError("empty train entry labels")
    drop_cols = {
        "split",
        "entry_timestamp",
        "exit_timestamp",
        "router_expert",
        "reason",
        "entry_i",
        "exit_i",
        "entry_price",
        "raw_exit_price_move",
        "mfe_price_move",
        "mae_price_move",
        "net_per_notional",
        "trade_return",
        "win",
        "hold_hours",
        "candidate_was_in_position",
    }
    feature_cols = [c for c in train_entries.columns if c not in drop_cols]
    if not feature_cols:
        raise RuntimeError("no feature columns for entry gate")
    x_train = train_entries[feature_cols].apply(pd.to_numeric, errors="raise").astype(np.float32)
    y_train = train_entries["trade_return"].astype(float).to_numpy(dtype=np.float64)
    model = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.045,
        max_leaf_nodes=15,
        l2_regularization=0.02,
        random_state=260702,
    )
    model.fit(x_train, y_train)
    x_calib = calib_entries[feature_cols].apply(pd.to_numeric, errors="raise").astype(np.float32)
    calib_scores = model.predict(x_calib)
    threshold_payload = select_threshold(calib_entries, calib_scores)
    model_path = out_dir / "entry_gate_hgb_regressor.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols, "threshold": threshold_payload}, model_path)
    report = {
        "model_path": str(model_path),
        "feature_cols": feature_cols,
        "feature_count": int(len(feature_cols)),
        "train_summary": summarize_entries(train_entries),
        "calibration_summary": summarize_entries(calib_entries),
        "threshold": threshold_payload,
    }
    return model, feature_cols, report


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args)

    train_metrics, train_artifacts = simulate(
        split="train_live_native",
        feature_path=Path(args.train_features),
        start=args.train_start,
        end=args.train_end,
        parent_variant=args.parent_runtime_variant,
        policy=policy,
        out_dir=out_dir,
        collect_entry_labels=True,
    )
    train_entries_all = pd.read_csv(train_artifacts["entry_labels"])
    train_entries_all["entry_timestamp"] = pd.to_datetime(train_entries_all["entry_timestamp"], errors="raise")
    train_cut = pd.Timestamp(args.gate_train_end)
    train_entries = train_entries_all[train_entries_all["entry_timestamp"] < train_cut].reset_index(drop=True)
    calib_entries = train_entries_all[train_entries_all["entry_timestamp"] >= train_cut].reset_index(drop=True)
    if train_entries.empty or calib_entries.empty:
        raise RuntimeError(
            f"entry gate chronological split empty: train={len(train_entries)} calibration={len(calib_entries)}"
        )
    train_entries.to_csv(out_dir / "entry_gate_train_rows.csv", index=False)
    calib_entries.to_csv(out_dir / "entry_gate_calibration_rows.csv", index=False)
    gate_model, feature_cols, gate_report = train_gate(train_entries, calib_entries, out_dir)
    threshold = float(gate_report["threshold"]["selected"]["threshold"])

    validation_metrics, validation_artifacts = simulate(
        split="validation",
        feature_path=Path(args.features_2025),
        start=args.validation_start,
        end=args.validation_end,
        parent_variant=args.parent_runtime_variant,
        policy=policy,
        out_dir=out_dir,
        gate_model=gate_model,
        feature_cols=feature_cols,
        gate_threshold=threshold,
    )
    oos_metrics, oos_artifacts = simulate(
        split="oos",
        feature_path=Path(args.oos_features),
        start=args.oos_start,
        end=args.oos_end,
        parent_variant=args.parent_runtime_variant,
        policy=policy,
        out_dir=out_dir,
        gate_model=gate_model,
        feature_cols=feature_cols,
        gate_threshold=threshold,
    )

    report = {
        "schema_version": "omega462.live_native_entry_gate.train_eval.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "parent_runtime_variant": str(args.parent_runtime_variant),
        "policy": policy,
        "training_contract": {
            "train_tape_start": args.train_start,
            "train_tape_end_exclusive": args.train_end,
            "gate_fit_rows": str(out_dir / "entry_gate_train_rows.csv"),
            "gate_calibration_rows": str(out_dir / "entry_gate_calibration_rows.csv"),
            "validation_rows_used_for_training": False,
            "oos_rows_used_for_training": False,
            "trade_ledgers_used_as_model_input": False,
            "labels_use_future_prices_only_inside_train_split": True,
        },
        "fresh_forward_definition": "fixed split, causal 5m bar-by-bar replay; learned gate sees only current-row live-native features",
        "gate": gate_report,
        "splits": {
            "train_live_native": {
                "start": args.train_start,
                "end_exclusive": args.train_end,
                **train_artifacts,
            },
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
            "train_live_native": train_metrics,
            "validation": validation_metrics,
            "oos": oos_metrics,
        },
        "integrity": {
            "validation_ledger_replay_trace_count": int(validation_metrics["ledger_replay_trace_count"]),
            "validation_non_live_native_trace_count": int(validation_metrics["non_live_native_trace_count"]),
            "validation_non_minus_one_policy_row_count": int(validation_metrics["non_minus_one_policy_row_count"]),
            "oos_ledger_replay_trace_count": int(oos_metrics["ledger_replay_trace_count"]),
            "oos_non_live_native_trace_count": int(oos_metrics["non_live_native_trace_count"]),
            "oos_non_minus_one_policy_row_count": int(oos_metrics["non_minus_one_policy_row_count"]),
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
    parser.add_argument("--train-features", default=str(DEFAULT_TRAIN_FEATURES))
    parser.add_argument("--features-2025", default=str(DEFAULT_FEATURES_2025))
    parser.add_argument("--oos-features", default=str(DEFAULT_OOS_FEATURES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--parent-runtime-variant", choices=["source_v5", "cap220_no_v5"], default="source_v5")
    parser.add_argument("--train-start", default="2024-01-01 00:00:00")
    parser.add_argument("--train-end", default="2025-09-01 00:00:00")
    parser.add_argument("--gate-train-end", default="2025-05-01 00:00:00")
    parser.add_argument("--validation-start", default="2025-09-01 00:00:00")
    parser.add_argument("--validation-end", default="2026-01-01 00:00:00")
    parser.add_argument("--oos-start", default="2026-01-01 00:00:00")
    parser.add_argument("--oos-end", default="2026-04-01 00:00:00")
    parser.add_argument("--tp", type=float, default=0.026)
    parser.add_argument("--sl", type=float, default=0.014)
    parser.add_argument("--cap", type=float, default=4.106)
    parser.add_argument("--max-hold-hours", type=float, default=90.0)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2, default=json_default), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
