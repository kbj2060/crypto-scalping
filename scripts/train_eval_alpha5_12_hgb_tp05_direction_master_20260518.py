#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_alpha5_10_hgb_legacy_v4_vs_regime4_20260518 import (  # noqa: E402
    DEFAULT_LEGACY_EVAL,
    DEFAULT_LEGACY_TRAIN,
    DEFAULT_REGIME4_EVAL,
    DEFAULT_REGIME4_TRAIN,
    _assert_same_clock,
)
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    CLEAN4_PREFIX,
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    REGIMES,
    ROUTER_COLS,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_alpha5_8_hgb_action_feature_contract_compare_20260518 import _alpha4_mapped_features  # noqa: E402
from scripts.train_eval_alpha5_11_hgb_direction_master_20260518 import (  # noqa: E402
    LEGACY_CLUSTER_COLS,
    _apply_regime_thresholds,
    _base_features,
    _class_balanced_weight,
    _eval_actions,
    _gate_future,
    _merge_legacy_cluster,
    _regime_ids,
    _sample_indices,
    _score_candidate,
    _split,
    _x,
)
from scripts.train_eval_alpha5_5_lgbm_supervised_parent_20260518 import _decide_actions, _predict_proba_3  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _days, _fill_price, _json_default, _read  # noqa: E402
from scripts.tune_alpha5_9_hgb_action_master_20260518 import _fit_hgb, _hgb_specs  # noqa: E402


MODEL_ID = "alpha5_12_hgb_tp05_direction_master_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_12_hgb_tp05_direction_master_20260518"


def _horizons(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _first_touch_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    take_profit: float,
    min_agree: int,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    n = len(frame)
    votes = np.zeros((n, len(horizons)), dtype=np.int8)
    first_bars = np.full((n, len(horizons)), np.nan, dtype=np.float64)
    tp = float(take_profit)
    for i in range(n):
        entry = close[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        long_px = entry * (1.0 + tp)
        short_px = entry * (1.0 - tp)
        for j, h in enumerate(horizons):
            end = min(n, i + int(h) + 1)
            if end <= i + 1:
                continue
            fh = high[i + 1 : end]
            fl = low[i + 1 : end]
            long_hits = np.flatnonzero(fh >= long_px)
            short_hits = np.flatnonzero(fl <= short_px)
            if len(long_hits) == 0 and len(short_hits) == 0:
                continue
            lb = int(long_hits[0]) + 1 if len(long_hits) else 10**9
            sb = int(short_hits[0]) + 1 if len(short_hits) else 10**9
            if lb < sb:
                votes[i, j] = 1
                first_bars[i, j] = lb
            elif sb < lb:
                votes[i, j] = 2
                first_bars[i, j] = sb
            else:
                votes[i, j] = 0
    long_count = np.sum(votes == 1, axis=1)
    short_count = np.sum(votes == 2, axis=1)
    action = np.zeros(n, dtype=np.int64)
    action[(long_count >= int(min_agree)) & (long_count > short_count)] = 1
    action[(short_count >= int(min_agree)) & (short_count > long_count)] = 2
    max_h = max(horizons)
    valid = np.arange(0, max(0, n - max_h - 1), dtype=np.int64)
    confidence = np.abs(long_count - short_count).astype(np.float64) / max(float(len(horizons)), 1.0)
    report = {
        "rows": int(len(valid)),
        "horizons": list(horizons),
        "take_profit": float(tp),
        "min_agree": int(min_agree),
        "action_counts": {
            "cash": int(np.sum(action[valid] == 0)),
            "long": int(np.sum(action[valid] == 1)),
            "short": int(np.sum(action[valid] == 2)),
        },
        "trade_ratio": float(np.mean(action[valid] != 0)) if len(valid) else 0.0,
        "confidence_mean": float(np.mean(confidence[valid])) if len(valid) else 0.0,
        "first_touch_bar_mean": float(np.nanmean(first_bars[valid])) if np.any(np.isfinite(first_bars[valid])) else None,
    }
    return {"action": action, "valid_idx": valid, "confidence": confidence, "report": report}


def _sample_weight(y: np.ndarray, confidence: np.ndarray, mode: str) -> np.ndarray:
    w = np.ones(len(y), dtype=np.float64)
    if mode in {"confidence", "balanced_confidence"}:
        w *= np.clip(1.0 + confidence, 0.5, 2.0)
    if mode in {"balanced", "balanced_confidence"}:
        w *= _class_balanced_weight(y)
    return w


def _direction_metrics(actions: np.ndarray, label_payload: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(label_payload["action"], dtype=np.int64)
    valid = np.asarray(label_payload["valid_idx"], dtype=np.int64)
    mask = np.zeros(len(actions), dtype=bool)
    mask[valid] = True
    trade = (actions != 0) & mask
    n_trade = int(np.sum(trade))
    out: dict[str, Any] = {"coverage": float(n_trade / max(int(np.sum(mask)), 1)), "trades_pred": n_trade}
    if n_trade == 0:
        out.update({"tp05_precision": 0.0, "long_tp05_precision": 0.0, "short_tp05_precision": 0.0, "balanced_tp05_precision": 0.0})
        return out
    out["tp05_precision"] = float(np.mean(actions[trade] == y[trade]))
    parts = []
    for cls, name in [(1, "long"), (2, "short")]:
        m = trade & (actions == cls)
        if np.any(m):
            p = float(np.mean(y[m] == cls))
            parts.append(p)
            out[f"{name}_tp05_precision"] = p
            out[f"{name}_pred"] = int(np.sum(m))
        else:
            out[f"{name}_tp05_precision"] = 0.0
            out[f"{name}_pred"] = 0
    out["balanced_tp05_precision"] = float(np.mean(parts)) if parts else 0.0
    return out


def _backtest_tp05(
    frame: pd.DataFrame,
    actions: np.ndarray,
    *,
    fee: float,
    slip: float,
    unit_exposure: float,
    max_hold_bars: int,
    take_profit: float,
    exit_policy: str,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_equity = 1.0
    hold = 0
    trades = wins = long_entries = short_entries = 0
    exits: dict[str, int] = {}
    action_counts = {"flat": 0, "long": 0, "short": 0}
    exposure = float(unit_exposure)
    tp = float(take_profit)

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int) -> None:
        nonlocal side, entry, entry_equity, cash, hold, long_entries, short_entries
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        entry = _fill_price(frame, fill_i, side, float(slip), entry=True)
        entry_equity = cash
        cash -= cash * float(fee) * exposure
        hold = 0
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str, fill_px: float | None = None) -> None:
        nonlocal side, entry, cash, hold, trades, wins
        if fill_px is None:
            fill_i = min(i + 1, len(frame) - 1)
            fill_px = _fill_price(frame, fill_i, side, float(slip), entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * float(fee) * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        hold = 0

    for i in range(len(frame) - 2):
        exited_this_bar = False
        desired = int(actions[i])
        action_counts["flat" if desired == 0 else "long" if desired == 1 else "short"] += 1
        if side != 0:
            hold += 1
            if side > 0 and high[i] >= entry * (1.0 + tp):
                exit_pos(i, "tp05", entry * (1.0 + tp) * (1.0 - float(slip)))
                exited_this_bar = True
            elif side > 0 and exit_policy == "barrier" and low[i] <= entry * (1.0 - tp):
                exit_pos(i, "adverse05", entry * (1.0 - tp) * (1.0 - float(slip)))
                exited_this_bar = True
            elif side < 0 and low[i] <= entry * (1.0 - tp):
                exit_pos(i, "tp05", entry * (1.0 - tp) * (1.0 + float(slip)))
                exited_this_bar = True
            elif side < 0 and exit_policy == "barrier" and high[i] >= entry * (1.0 + tp):
                exit_pos(i, "adverse05", entry * (1.0 + tp) * (1.0 + float(slip)))
                exited_this_bar = True
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        desired_side = 0 if desired == 0 else 1 if desired == 1 else -1
        if side != 0 and int(max_hold_bars) > 0 and hold >= int(max_hold_bars):
            exit_pos(i, "max_hold")
        elif side != 0 and exit_policy == "model_flip" and (desired_side == 0 or desired_side == -side):
            exit_pos(i, "model_flat_or_flip")
        if side == 0 and desired_side != 0 and not exited_this_bar:
            enter(i, desired_side)
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    pnl = (cash - 1.0) * 100.0
    return {
        "pnl": float(pnl),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(trades * exposure / max(len(frame), 1)),
        "action_counts": action_counts,
        "exits": exits,
    }


def _eval_tp05(frame: pd.DataFrame, actions: np.ndarray, labels: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    bt = {
        f"cost{m}": _backtest_tp05(
            frame,
            actions,
            fee=float(args.fee) * float(m),
            slip=float(args.slip) * float(m),
            unit_exposure=float(args.unit_exposure),
            max_hold_bars=int(args.max_hold_bars),
            take_profit=float(args.take_profit),
            exit_policy=str(args.exit_policy),
        )
        for m in (1, 2, 3)
    }
    dm = _direction_metrics(actions, labels)
    c1, c2, c3 = bt["cost1"], bt["cost2"], bt["cost3"]
    if int(c1["trades"]) < 15:
        score = -1e6 + float(c1["pnl"])
    else:
        score = (
            20.0 * float(dm["balanced_tp05_precision"])
            + 12.0 * float(dm["tp05_precision"])
            + float(c1["pnl"])
            + 0.45 * float(c2["pnl"])
            + 0.20 * float(c3["pnl"])
            - 0.25 * abs(float(c1["mdd"]))
            - max(0.0, 0.015 - float(dm["coverage"])) * 100.0
        )
    return {"backtest": bt, "direction": dm, "score": float(score)}


def _tune_global(frame: pd.DataFrame, proba: np.ndarray, labels: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for prob in [float(x) for x in args.prob_thresholds.split(",") if x]:
        for margin in [float(x) for x in args.margin_thresholds.split(",") if x]:
            actions = _decide_actions(proba, prob, margin)
            ev = _eval_tp05(frame, actions, labels, args)
            row = {"kind": "global", "prob": prob, "margin": margin, "actions": actions, **ev}
            if best is None or float(row["score"]) > float(best["score"]):
                best = row
    assert best is not None
    return best


def _tune_regime(frame: pd.DataFrame, proba: np.ndarray, labels: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rid = _regime_ids(frame)
    pair_grid = [(float(p), float(m)) for p in args.prob_thresholds.split(",") if p for m in args.margin_thresholds.split(",") if m]
    thresholds: dict[int, tuple[float, float]] = {}
    for r in range(len(REGIMES)):
        best_pair = (0.90, 0.10)
        best_score = -1e18
        mask = rid == r
        for prob, margin in pair_grid:
            actions = np.zeros(len(proba), dtype=np.int64)
            actions[mask] = _decide_actions(proba[mask], prob, margin)
            dm = _direction_metrics(actions, labels)
            score = 30.0 * dm["balanced_tp05_precision"] + 15.0 * dm["tp05_precision"] + min(dm["trades_pred"], 120) * 0.02
            if dm["trades_pred"] < 5:
                score -= 10.0
            if score > best_score:
                best_score = score
                best_pair = (prob, margin)
        thresholds[r] = best_pair
    actions = _apply_regime_thresholds(proba, rid, thresholds)
    ev = _eval_tp05(frame, actions, labels, args)
    return {"kind": "regime_threshold", "thresholds": {REGIMES[k]: v for k, v in thresholds.items()}, "actions": actions, **ev}


def main() -> None:
    p = argparse.ArgumentParser(description="Alpha5.12 HGB direction parent with TP 0.5 first-touch labels and TP 0.5 backtest.")
    p.add_argument("--regime4-train-csv", type=Path, default=DEFAULT_REGIME4_TRAIN)
    p.add_argument("--regime4-eval-csv", type=Path, default=DEFAULT_REGIME4_EVAL)
    p.add_argument("--legacy-train-csv", type=Path, default=DEFAULT_LEGACY_TRAIN)
    p.add_argument("--legacy-eval-csv", type=Path, default=DEFAULT_LEGACY_EVAL)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train-end", default="2025-10-01")
    p.add_argument("--val-start", default="2025-10-01")
    p.add_argument("--val-end", default="2026-01-01")
    p.add_argument("--horizons", default="12,24,48,96")
    p.add_argument("--min-agree-grid", default="1,2")
    p.add_argument("--tracks", default="regime4_core,regime4_core_future,regime4_core_legacy_cluster")
    p.add_argument("--weight-modes", default="balanced,balanced_confidence")
    p.add_argument("--prob-thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.93,0.95")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12,0.16,0.20,0.25")
    p.add_argument("--take-profit", type=float, default=0.005)
    p.add_argument("--exit-policy", choices=["model_flip", "barrier", "tp_timeout"], default="barrier")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--seed", type=int, default=51201)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_train = _read(args.regime4_train_csv)
    raw_eval = _read(args.regime4_eval_csv)
    legacy_train = _read(args.legacy_train_csv)
    legacy_eval = _read(args.legacy_eval_csv)
    _assert_same_clock(raw_train, legacy_train, "train")
    _assert_same_clock(raw_eval, legacy_eval, "eval")
    train_cluster_all = _merge_legacy_cluster(raw_train, legacy_train)
    eval_cluster_all = _merge_legacy_cluster(raw_eval, legacy_eval)
    train_df = _split(raw_train, None, args.train_end)
    val_df = _split(raw_train, args.val_start, args.val_end)
    eval_df = raw_eval.reset_index(drop=True)
    train_cluster = _split(train_cluster_all, None, args.train_end)
    val_cluster = _split(train_cluster_all, args.val_start, args.val_end)
    eval_cluster = eval_cluster_all.reset_index(drop=True)
    audit = _verify_state24_sticky090_inputs(raw_train, raw_eval, args.manifest, args.clean4_report)
    horizons = _horizons(args.horizons)
    min_agrees = [int(x) for x in args.min_agree_grid.split(",") if x]

    print(json.dumps({"stage": "start", "model_id": MODEL_ID, "take_profit": args.take_profit, "exit_policy": args.exit_policy, "horizons": horizons, "min_agrees": min_agrees, "audit": {"expected_model_found_in_manifest": audit.get("expected_model_found_in_manifest")}}, ensure_ascii=False), flush=True)

    base_cols = _alpha4_mapped_features(raw_train, raw_eval, include_future=False)
    future_cols = _alpha4_mapped_features(raw_train, raw_eval, include_future=True)
    cluster_cols = base_cols + [c for c in LEGACY_CLUSTER_COLS if c in train_cluster_all.columns and c not in base_cols]
    payloads: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]] = {}
    for track in [x.strip() for x in args.tracks.split(",") if x.strip()]:
        if track == "regime4_core":
            cols, tr, va, oo = base_cols, train_df, val_df, eval_df
        elif track == "regime4_core_future":
            cols, tr, va, oo = future_cols, train_df, val_df, eval_df
        elif track == "regime4_core_legacy_cluster":
            cols, tr, va, oo = cluster_cols, train_cluster, val_cluster, eval_cluster
        else:
            raise ValueError(track)
        payloads[track] = (_x(tr, cols), _x(va, cols), _x(oo, cols), tr, va, oo, cols)
        print(json.dumps({"stage": "features_ready", "track": track, "feature_count": len(cols), "regime4_count": sum(c.startswith(CLEAN4_PREFIX) for c in cols), "future_pred_count": sum(c.startswith("regime4_pred_") for c in cols), "legacy_cluster_count": sum(c in LEGACY_CLUSTER_COLS for c in cols)}, ensure_ascii=False), flush=True)

    labels: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for agree in min_agrees:
        name = f"tp{int(args.take_profit * 10000):04d}_a{agree}"
        labels[name] = (
            _first_touch_labels(train_df, horizons=horizons, take_profit=args.take_profit, min_agree=agree),
            _first_touch_labels(val_df, horizons=horizons, take_profit=args.take_profit, min_agree=agree),
            _first_touch_labels(eval_df, horizons=horizons, take_profit=args.take_profit, min_agree=agree),
        )
        print(json.dumps({"stage": "label_built", "label_cfg": name, "train": labels[name][0]["report"], "validation": labels[name][1]["report"], "oos": labels[name][2]["report"]}, ensure_ascii=False, default=_json_default), flush=True)

    rows: list[dict[str, Any]] = []
    hgb_specs = _hgb_specs()
    total = len(labels) * len(payloads) * len([x for x in args.weight_modes.split(",") if x]) * len(hgb_specs)
    done = 0
    for li, (lname, (tr_lab, va_lab, oo_lab)) in enumerate(labels.items()):
        idx = _sample_indices(tr_lab["valid_idx"], args.stride)
        y = tr_lab["action"][idx].astype(np.int64)
        conf = tr_lab["confidence"][idx].astype(np.float64)
        for ti, (track, (x_train_full, x_val, x_eval, tr_frame, va_frame, oo_frame, cols)) in enumerate(payloads.items()):
            x_train = x_train_full.iloc[idx].reset_index(drop=True)
            for wi, wmode in enumerate([x.strip() for x in args.weight_modes.split(",") if x.strip()]):
                sw = _sample_weight(y, conf, wmode)
                for si, spec in enumerate(hgb_specs):
                    done += 1
                    print(json.dumps({"stage": "fit", "done": done, "total": total, "label_cfg": lname, "track": track, "weight_mode": wmode, "hgb": spec.name}, ensure_ascii=False), flush=True)
                    model = _fit_hgb(x_train, y, sw, spec, args.seed + li * 1000 + ti * 200 + wi * 50 + si)
                    vp = _predict_proba_3(model, x_val)
                    ep = _predict_proba_3(model, x_eval)
                    global_best = _tune_global(va_frame, vp, va_lab, args)
                    regime_best = _tune_regime(va_frame, vp, va_lab, args)
                    selected_name, selected_val = max([("global", global_best), ("regime_threshold", regime_best)], key=lambda kv: float(kv[1]["score"]))
                    if selected_name == "global":
                        eact = _decide_actions(ep, selected_val["prob"], selected_val["margin"])
                    else:
                        eact = _apply_regime_thresholds(ep, _regime_ids(oo_frame), {i: tuple(selected_val["thresholds"][REGIMES[i]]) for i in range(len(REGIMES))})
                    eres = _eval_tp05(oo_frame, eact, oo_lab, args)
                    artifact = args.out_dir / f"{lname}_{track}_{wmode}_{spec.name}_tp05_hgb.joblib"
                    joblib.dump({"model_id": MODEL_ID, "model": model, "feature_cols": cols, "label_cfg": {"name": lname, "take_profit": args.take_profit, "horizons": list(horizons)}, "track": track, "weight_mode": wmode, "hgb": asdict(spec), "decision": {k: v for k, v in selected_val.items() if k not in {"actions", "backtest", "direction"}}}, artifact)
                    row = {"label_cfg": lname, "track": track, "weight_mode": wmode, "hgb": asdict(spec), "decision": selected_name, "validation": {k: v for k, v in selected_val.items() if k != "actions"}, "oos": eres, "artifact": str(artifact)}
                    rows.append(row)
                    print(json.dumps({"stage": "candidate", "label_cfg": lname, "track": track, "weight_mode": wmode, "hgb": spec.name, "decision": selected_name, "val_score": selected_val["score"], "val_dir": selected_val["direction"], "val_cost1": selected_val["backtest"]["cost1"], "oos_score": eres["score"], "oos_dir": eres["direction"], "oos_cost1": eres["backtest"]["cost1"]}, ensure_ascii=False, default=_json_default), flush=True)

    best = max(rows, key=lambda r: float(r["validation"]["score"]))
    summary = {"model_id": MODEL_ID, "design": "TP 0.5 first-touch direction labels and TP 0.5 barrier exit backtest.", "exit_policy": args.exit_policy, "experiments": rows, "best": best, "top20": sorted(rows, key=lambda r: float(r["validation"]["score"]), reverse=True)[:20]}
    summary_path = args.out_dir / "alpha5_12_hgb_tp05_direction_master_summary.json"
    grid_path = args.out_dir / "alpha5_12_hgb_tp05_direction_master_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "label_cfg": r["label_cfg"],
                "track": r["track"],
                "weight_mode": r["weight_mode"],
                "hgb_name": r["hgb"]["name"],
                "decision": r["decision"],
                "val_score": r["validation"]["score"],
                "val_tp05_precision": r["validation"]["direction"]["tp05_precision"],
                "val_bal_tp05_precision": r["validation"]["direction"]["balanced_tp05_precision"],
                "val_coverage": r["validation"]["direction"]["coverage"],
                "val_cost1_pnl": r["validation"]["backtest"]["cost1"]["pnl"],
                "val_cost1_mdd": r["validation"]["backtest"]["cost1"]["mdd"],
                "val_cost1_trades": r["validation"]["backtest"]["cost1"]["trades"],
                "oos_score": r["oos"]["score"],
                "oos_tp05_precision": r["oos"]["direction"]["tp05_precision"],
                "oos_bal_tp05_precision": r["oos"]["direction"]["balanced_tp05_precision"],
                "oos_coverage": r["oos"]["direction"]["coverage"],
                "oos_cost1_pnl": r["oos"]["backtest"]["cost1"]["pnl"],
                "oos_cost1_mdd": r["oos"]["backtest"]["cost1"]["mdd"],
                "oos_cost1_trades": r["oos"]["backtest"]["cost1"]["trades"],
                "oos_cost2_pnl": r["oos"]["backtest"]["cost2"]["pnl"],
                "oos_cost3_pnl": r["oos"]["backtest"]["cost3"]["pnl"],
                "artifact": r["artifact"],
            }
            for r in rows
        ]
    ).sort_values("val_score", ascending=False).to_csv(grid_path, index=False)
    print(json.dumps({"stage": "complete", "summary": str(summary_path), "grid": str(grid_path), "best": {"label_cfg": best["label_cfg"], "track": best["track"], "weight_mode": best["weight_mode"], "hgb": best["hgb"]["name"], "decision": best["decision"], "val_score": best["validation"]["score"], "oos_score": best["oos"]["score"], "oos_cost1": best["oos"]["backtest"]["cost1"], "oos_direction": best["oos"]["direction"]}}, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
