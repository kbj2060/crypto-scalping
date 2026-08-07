#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT  # noqa: E402
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_EVAL,
    DEFAULT_PREPROCESS_MANIFEST,
    DEFAULT_TRAIN,
    POSITION_CONTEXT_COLS,
    POSITION_CONTEXT_NORM_BARS,
    REGIMES,
    ROUTER_COLS,
    _router_matrix,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _json_default, _read  # noqa: E402


DEFAULT_SPECIALISTS = ROOT / "tmp/causal_regen_20260516/alpha5_3_state24_sticky090_hmm_dqn_router_parent_soft0_800_20260518/specialists"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/alpha5_3_state24_sticky090_hmm_dqn_router_parent_soft0_800_20260518/fast_backtest_summary.json"


def _softmax(q: np.ndarray, temp: float) -> np.ndarray:
    z = q / max(float(temp), 1e-4)
    z = z - np.max(z)
    p = np.exp(z)
    return p / max(float(p.sum()), 1e-12)


def _load_specialists(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for regime in REGIMES:
        p = path / f"{regime}_dqn_parent.pkl"
        if not p.exists():
            raise FileNotFoundError(p)
        out[regime] = joblib.load(p)
    return out


def _prepare_runtime_models(specialists: dict[str, dict[str, Any]], frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for regime, parent in specialists.items():
        action_model = parent["action_model"]
        feature_cols = list(parent["feature_cols"])
        arr = frame.reindex(columns=feature_cols).to_numpy(dtype=np.float32, copy=True)
        med = np.asarray(action_model.medians, dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, med)
        pos_idx = [feature_cols.index(c) for c in POSITION_CONTEXT_COLS]
        prepared[regime] = {
            "base": arr,
            "pos_idx": pos_idx,
            "mean": np.asarray(action_model.mean, dtype=np.float32),
            "std": np.maximum(np.asarray(action_model.std, dtype=np.float32), 1e-6),
            "model": action_model._model(),
            "temperature": float(action_model.config.get("temperature", 0.18)),
        }
    return prepared


def _predict_one(prepared: dict[str, Any], i: int, ctx: np.ndarray) -> np.ndarray:
    row = prepared["base"][i].copy()
    row[np.asarray(prepared["pos_idx"], dtype=np.int64)] = ctx.astype(np.float32)
    z = (row - prepared["mean"]) / prepared["std"]
    with torch.no_grad():
        q = prepared["model"](torch.from_numpy(z[None, :])).cpu().numpy()[0]
    # DQN classes are [cash, long, short].
    p = _softmax(q, float(prepared["temperature"]))
    return np.asarray([p[ACTION_LONG], p[ACTION_SHORT], p[ACTION_CASH]], dtype=np.float64)


def _desired_side(
    prepared: dict[str, dict[str, Any]],
    weights: np.ndarray,
    i: int,
    ctx: np.ndarray,
    *,
    mode: str,
    threshold: float,
) -> int:
    if mode == "hard_current":
        regime = REGIMES[int(np.argmax(weights[i]))]
        probs = _predict_one(prepared[regime], i, ctx)
    elif mode == "soft_current":
        probs = np.zeros(3, dtype=np.float64)
        for j, regime in enumerate(REGIMES):
            probs += float(weights[i, j]) * _predict_one(prepared[regime], i, ctx)
        probs = probs / max(float(probs.sum()), 1e-12)
    else:
        raise ValueError(mode)
    if float(threshold) > 0.0:
        edge = float(probs[0] - probs[1])
        if edge > float(threshold):
            return 1
        if edge < -float(threshold):
            return -1
        return 0
    action = int(np.argmax(np.asarray([probs[2], probs[0], probs[1]], dtype=np.float64)))
    if action == 1:
        return 1
    if action == 2:
        return -1
    return 0


def _run(
    frame: pd.DataFrame,
    specialists: dict[str, dict[str, Any]],
    *,
    mode: str,
    threshold: float,
    fee: float,
    slip: float,
    unit_exposure: float,
    log_name: str,
    log_every: int,
) -> dict[str, Any]:
    close = _close(frame)
    weights = _router_matrix(frame)
    prepared = _prepare_runtime_models(specialists, frame)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    action_counts: dict[str, int] = {"cash": 0, "long": 0, "short": 0}
    exits: dict[str, int] = {}
    exposure = float(unit_exposure)

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * exposure
        return cash * (1.0 + unreal), unreal

    def enter(i: int, side: int) -> None:
        nonlocal pos, entry_price, entry_equity, entry_idx, cash, long_entries, short_entries, notional_sum, leverage_sum
        fill_i = min(i + 1, len(frame) - 1)
        pos = int(side)
        entry_price = _fill_price(frame, fill_i, pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        cash -= cash * fee * exposure
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += exposure
        leverage_sum += 1.0

    def exit_position(i: int, reason: str) -> None:
        nonlocal pos, entry_price, cash, trades, wins
        fill_i = min(i + 1, len(frame) - 1)
        exit_px = _fill_price(frame, fill_i, pos, slip, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        pos = 0

    for i in range(0, len(frame) - 2):
        eq, _ = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos == 0:
            ctx = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            bars = float(max(0, i - entry_idx))
            entry_dist = float(close[i]) / max(float(entry_price), 1e-12) - 1.0
            raw = entry_dist if pos > 0 else -entry_dist
            ctx = np.asarray(
                [
                    float(pos),
                    float(np.clip(bars / POSITION_CONTEXT_NORM_BARS, 0.0, 1.0)),
                    raw * exposure,
                    entry_dist,
                    bars,
                ],
                dtype=np.float32,
            )
        desired = _desired_side(prepared, weights, i, ctx, mode=mode, threshold=float(threshold))
        if desired > 0:
            action_counts["long"] += 1
        elif desired < 0:
            action_counts["short"] += 1
        else:
            action_counts["cash"] += 1
        if pos != 0 and desired != pos:
            exit_position(i, "action_cash" if desired == 0 else "action_flip")
        if pos == 0 and desired != 0:
            enter(i, desired)
        if int(log_every) > 0 and (i + 1) % int(log_every) == 0:
            print(json.dumps({"stage": "backtest_progress", "name": log_name, "bar": i + 1, "bars": len(frame), "cash": cash, "trades": trades}, ensure_ascii=False), flush=True)

    if pos != 0:
        exit_position(len(frame) - 2, "end_of_data")
        eq = cash
    else:
        eq, _ = mark(len(frame) - 1)
    peak = max(peak, eq)
    mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
    n = max(len(frame), 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / max(long_entries + short_entries, 1)),
        "action_counts": action_counts,
        "exits": exits,
    }


def _metrics(frame: pd.DataFrame, specialists: dict[str, dict[str, Any]], *, mode: str, threshold: float, fee: float, slip: float, unit_exposure: float, name: str, log_every: int) -> dict[str, Any]:
    return {
        f"cost{mult}": _run(
            frame,
            specialists,
            mode=mode,
            threshold=float(threshold),
            fee=fee * float(mult),
            slip=slip * float(mult),
            unit_exposure=float(unit_exposure),
            log_name=f"{name}_cost{mult}",
            log_every=int(log_every),
        )
        for mult in (1, 2, 3)
    }


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.30 * c3["pnl"] - 0.35 * abs(c1["mdd"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--specialists-dir", type=Path, default=DEFAULT_SPECIALISTS)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mode-filter", choices=["all", "hard", "soft0", "soft005", "soft010"], default="all")
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=2000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    audit = _verify_state24_sticky090_inputs(train_all, eval_df, DEFAULT_PREPROCESS_MANIFEST, DEFAULT_CLEAN4_REPORT)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    specialists = _load_specialists(args.specialists_dir)
    parent0 = next(iter(specialists.values()))
    print(json.dumps({"stage": "loaded_specialists", "dir": str(args.specialists_dir), "feature_count": len(parent0["feature_cols"]), "position_context": [c for c in parent0["feature_cols"] if c.startswith("position_")]}, ensure_ascii=False), flush=True)
    mode_specs = [("hard_current", 0.0), ("soft_current", 0.0), ("soft_current", 0.05), ("soft_current", 0.10)]
    if args.mode_filter == "hard":
        mode_specs = [("hard_current", 0.0)]
    elif args.mode_filter == "soft0":
        mode_specs = [("soft_current", 0.0)]
    elif args.mode_filter == "soft005":
        mode_specs = [("soft_current", 0.05)]
    elif args.mode_filter == "soft010":
        mode_specs = [("soft_current", 0.10)]
    rows: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    fee = 0.0005
    slip = 0.0002
    for mode, th in mode_specs:
        name = mode if mode == "hard_current" else f"{mode}_th{th:.2f}"
        print(json.dumps({"stage": "mode_start", "name": name}, ensure_ascii=False), flush=True)
        val = _metrics(val_df, specialists, mode=mode, threshold=th, fee=fee, slip=slip, unit_exposure=float(args.unit_exposure), name=f"val_{name}", log_every=int(args.log_every))
        ev = _metrics(eval_df, specialists, mode=mode, threshold=th, fee=fee, slip=slip, unit_exposure=float(args.unit_exposure), name=f"eval_{name}", log_every=int(args.log_every))
        score = _score(val["cost1"], val["cost2"], val["cost3"])
        row = {
            "name": name,
            "selection_score": score,
            "validation_metrics": val,
            "metrics": ev,
            "selected_metrics_compact": {
                "cost1": {k: ev["cost1"][k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional")},
                "cost2": {k: ev["cost2"][k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional")},
                "cost3": {k: ev["cost3"][k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional")},
            },
        }
        experiments.append(row)
        rows.append({"name": name, "selection_score": score, "val_cost1_pnl": val["cost1"]["pnl"], "val_cost1_mdd": val["cost1"]["mdd"], "val_trades": val["cost1"]["trades"], "eval_cost1_pnl": ev["cost1"]["pnl"], "eval_cost1_mdd": ev["cost1"]["mdd"], "eval_trades": ev["cost1"]["trades"]})
        print(json.dumps({"stage": "mode_done", "name": name, "metrics": row["selected_metrics_compact"]}, ensure_ascii=False, default=_json_default), flush=True)
    best = max(experiments, key=lambda r: float(r["selection_score"]))
    report = {"model_id": "alpha5_3_state24_sticky090_fast_backtest_20260518", "feature_audit": audit, "specialists_dir": str(args.specialists_dir), "experiments": experiments, "best_by_selection": best["name"], "selected_metrics": best["selected_metrics_compact"], "out": str(args.out)}
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({"stage": "backtest_complete", "report": str(args.out), "best": best["name"], "metrics": report["selected_metrics"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
