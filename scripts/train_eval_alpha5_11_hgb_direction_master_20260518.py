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
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402
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
from scripts.train_eval_alpha5_5_lgbm_supervised_parent_20260518 import (  # noqa: E402
    _backtest_actions,
    _decide_actions,
    _predict_proba_3,
)
from scripts.train_eval_alpha5_8_hgb_action_feature_contract_compare_20260518 import _alpha4_mapped_features  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.tune_alpha5_9_hgb_action_master_20260518 import HGBSpec, _fit_hgb, _hgb_specs  # noqa: E402


MODEL_ID = "alpha5_11_hgb_direction_master_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_11_hgb_direction_master_20260518"
LEGACY_CLUSTER_COLS = [
    "clean_regime_2024_unsup_v4_cluster",
    "clean_regime_2024_unsup_v4_cluster_confidence",
    "clean_regime_2024_unsup_v4_cluster_prob_0",
    "clean_regime_2024_unsup_v4_cluster_prob_1",
    "clean_regime_2024_unsup_v4_cluster_prob_2",
    "clean_regime_2024_unsup_v4_cluster_prob_3",
    "clean_regime_2024_unsup_v4_cluster_prob_4",
    "clean_regime_2024_unsup_v4_state_code",
    "clean_regime_2024_unsup_v4_normal_prob",
]


def _split(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = frame.copy()
    if start:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end:
        out = out[out["timestamp"] < pd.Timestamp(end)]
    return out.reset_index(drop=True)


def _horizons(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _base_features(train: pd.DataFrame, eval_df: pd.DataFrame, *, include_future: bool) -> list[str]:
    return _alpha4_mapped_features(train, eval_df, include_future=include_future)


def _merge_legacy_cluster(regime4: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    _assert_same_clock(regime4, legacy, "legacy_cluster_merge")
    out = regime4.copy()
    for col in LEGACY_CLUSTER_COLS:
        if col in legacy.columns:
            out[col] = legacy[col].to_numpy()
    return out


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _future_returns(close: np.ndarray, horizons: tuple[int, ...]) -> np.ndarray:
    out = np.full((len(close), len(horizons)), np.nan, dtype=np.float64)
    for j, h in enumerate(horizons):
        if h <= 0 or h >= len(close):
            continue
        out[:-h, j] = close[h:] / np.clip(close[:-h], 1e-12, None) - 1.0
    return out


def _direction_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    fee: float,
    slip: float,
    min_edge: float,
    min_agree: int,
    margin_edge: float,
) -> dict[str, Any]:
    close = frame["close"].to_numpy(dtype=np.float64)
    rets = _future_returns(close, horizons)
    cost = 2.0 * (float(fee) + float(slip))
    long_edge = rets - cost
    short_edge = -rets - cost
    long_agree = np.sum(long_edge > float(min_edge), axis=1)
    short_agree = np.sum(short_edge > float(min_edge), axis=1)
    long_score = np.nanmean(long_edge, axis=1)
    short_score = np.nanmean(short_edge, axis=1)
    action = np.zeros(len(frame), dtype=np.int64)
    long_ok = (long_agree >= int(min_agree)) & (long_score > float(min_edge)) & ((long_score - short_score) > float(margin_edge))
    short_ok = (short_agree >= int(min_agree)) & (short_score > float(min_edge)) & ((short_score - long_score) > float(margin_edge))
    action[long_ok] = 1
    action[short_ok] = 2
    valid = np.arange(0, max(0, len(frame) - max(horizons) - 1), dtype=np.int64)
    confidence = np.nan_to_num(np.abs(long_score - short_score), nan=0.0)
    report = {
        "rows": int(len(valid)),
        "horizons": list(horizons),
        "min_edge": float(min_edge),
        "min_agree": int(min_agree),
        "margin_edge": float(margin_edge),
        "action_counts": {
            "cash": int(np.sum(action[valid] == 0)),
            "long": int(np.sum(action[valid] == 1)),
            "short": int(np.sum(action[valid] == 2)),
        },
        "trade_ratio": float(np.mean(action[valid] != 0)) if len(valid) else 0.0,
        "confidence_mean": float(np.mean(confidence[valid])) if len(valid) else 0.0,
    }
    return {"action": action, "valid_idx": valid, "confidence": confidence, "report": report}


def _sample_indices(valid_idx: np.ndarray, stride: int) -> np.ndarray:
    return np.asarray(valid_idx[:: max(1, int(stride))], dtype=np.int64)


def _class_balanced_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.ones(len(y), dtype=np.float64)
    classes, counts = np.unique(y, return_counts=True)
    total = float(max(len(y), 1))
    for cls, count in zip(classes, counts):
        out[y == int(cls)] = total / (float(len(classes)) * max(float(count), 1.0))
    return out


def _sample_weight(y: np.ndarray, confidence: np.ndarray, mode: str) -> np.ndarray:
    w = np.ones(len(y), dtype=np.float64)
    if mode in {"confidence", "balanced_confidence"}:
        w *= np.clip(confidence / (np.nanmedian(confidence) + 1e-8), 0.25, 4.0)
    if mode in {"balanced", "balanced_confidence"}:
        w *= _class_balanced_weight(y)
    return w


def _direction_metrics(frame: pd.DataFrame, actions: np.ndarray, label_payload: dict[str, Any], valid_mask_only: bool = True) -> dict[str, Any]:
    y = np.asarray(label_payload["action"], dtype=np.int64)
    valid = np.asarray(label_payload["valid_idx"], dtype=np.int64)
    mask = np.zeros(len(actions), dtype=bool)
    mask[valid] = True
    trade = (actions != 0) & mask if valid_mask_only else actions != 0
    n_trade = int(np.sum(trade))
    out: dict[str, Any] = {
        "coverage": float(n_trade / max(int(np.sum(mask)), 1)),
        "trades_pred": n_trade,
    }
    if n_trade == 0:
        out.update({"trade_precision": 0.0, "long_precision": 0.0, "short_precision": 0.0, "balanced_trade_precision": 0.0})
        return out
    correct = actions[trade] == y[trade]
    out["trade_precision"] = float(np.mean(correct))
    parts = []
    for cls, name in [(1, "long"), (2, "short")]:
        m = trade & (actions == cls)
        if np.any(m):
            p = float(np.mean(y[m] == cls))
            parts.append(p)
            out[f"{name}_precision"] = p
            out[f"{name}_pred"] = int(np.sum(m))
        else:
            out[f"{name}_precision"] = 0.0
            out[f"{name}_pred"] = 0
    out["balanced_trade_precision"] = float(np.mean(parts)) if parts else 0.0
    close = frame["close"].to_numpy(dtype=np.float64)
    h = 48 if len(close) > 49 else 1
    ret = np.full(len(close), np.nan, dtype=np.float64)
    ret[:-h] = close[h:] / np.clip(close[:-h], 1e-12, None) - 1.0
    signed = np.where(actions == 1, ret, np.where(actions == 2, -ret, np.nan))
    out["mean_signed_ret_h48"] = float(np.nanmean(signed[trade])) if n_trade else 0.0
    return out


def _regime_ids(frame: pd.DataFrame) -> np.ndarray:
    probs = frame[ROUTER_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return np.argmax(probs, axis=1)


def _apply_regime_thresholds(proba: np.ndarray, regime_ids: np.ndarray, thresholds: dict[int, tuple[float, float]]) -> np.ndarray:
    actions = np.zeros(len(proba), dtype=np.int64)
    for ridx, (prob, margin) in thresholds.items():
        m = regime_ids == int(ridx)
        if np.any(m):
            actions[m] = _decide_actions(proba[m], float(prob), float(margin))
    return actions


def _gate_future(frame: pd.DataFrame, actions: np.ndarray, *, confidence_min: float, whipsaw_max: float, entropy_max: float, require_trend_agree: bool) -> np.ndarray:
    out = actions.copy()
    conf = pd.to_numeric(frame.get("regime4_pred_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    whip = pd.to_numeric(frame.get("regime4_pred_whipsaw_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ent = pd.to_numeric(frame.get("regime4_pred_entropy", 0.0), errors="coerce").fillna(9.0).to_numpy(dtype=np.float64)
    veto = (conf < float(confidence_min)) | (whip > float(whipsaw_max)) | (ent > float(entropy_max))
    if require_trend_agree:
        pbull = pd.to_numeric(frame.get("regime4_pred_bull_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        pbear = pd.to_numeric(frame.get("regime4_pred_bear_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        veto |= ((out == 1) & (pbull < pbear)) | ((out == 2) & (pbear < pbull))
    out[veto] = 0
    return out


def _score_candidate(bt: dict[str, Any], dm: dict[str, Any]) -> float:
    c1, c2, c3 = bt["cost1"], bt["cost2"], bt["cost3"]
    trades = int(c1.get("trades", 0))
    if trades < 15:
        return -1e6 + float(c1.get("pnl", 0.0))
    return (
        18.0 * float(dm.get("balanced_trade_precision", 0.0))
        + 10.0 * float(dm.get("trade_precision", 0.0))
        + float(c1["pnl"])
        + 0.35 * float(c2["pnl"])
        + 0.15 * float(c3["pnl"])
        - 0.25 * abs(float(c1["mdd"]))
        - max(0.0, 0.20 - float(dm.get("coverage", 0.0))) * 8.0
        - max(0.0, float(c1["trades_per_day"]) - 5.0) * 1.5
    )


def _eval_actions(frame: pd.DataFrame, actions: np.ndarray, labels: dict[str, Any], fee: float, slip: float, exposure: float, max_hold: int) -> dict[str, Any]:
    bt = {
        f"cost{m}": _backtest_actions(
            frame,
            actions,
            fee=float(fee) * float(m),
            slip=float(slip) * float(m),
            unit_exposure=float(exposure),
            max_hold_bars=int(max_hold),
        )
        for m in (1, 2, 3)
    }
    dm = _direction_metrics(frame, actions, labels)
    return {"backtest": bt, "direction": dm, "score": _score_candidate(bt, dm)}


def _tune_global(frame: pd.DataFrame, proba: np.ndarray, labels: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for prob in [float(x) for x in args.prob_thresholds.split(",") if x]:
        for margin in [float(x) for x in args.margin_thresholds.split(",") if x]:
            actions = _decide_actions(proba, prob, margin)
            ev = _eval_actions(frame, actions, labels, args.fee, args.slip, args.unit_exposure, args.max_hold_bars)
            row = {"kind": "global", "prob": prob, "margin": margin, "actions": actions, **ev}
            if best is None or float(row["score"]) > float(best["score"]):
                best = row
    assert best is not None
    return best


def _tune_regime(frame: pd.DataFrame, proba: np.ndarray, labels: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    regime_ids = _regime_ids(frame)
    pair_grid = [(float(p), float(m)) for p in args.prob_thresholds.split(",") if p for m in args.margin_thresholds.split(",") if m]
    thresholds: dict[int, tuple[float, float]] = {}
    for ridx in range(len(REGIMES)):
        best_pair = (0.90, 0.10)
        best_score = -1e18
        m = regime_ids == ridx
        if not np.any(m):
            thresholds[ridx] = best_pair
            continue
        for prob, margin in pair_grid:
            partial = np.zeros(len(proba), dtype=np.int64)
            partial[m] = _decide_actions(proba[m], prob, margin)
            dm = _direction_metrics(frame, partial, labels)
            # Regime-local tuning should prefer precision first and avoid empty solutions.
            score = 25.0 * dm["balanced_trade_precision"] + 10.0 * dm["trade_precision"] + min(dm["trades_pred"], 60) * 0.03
            if dm["trades_pred"] < 5:
                score -= 10.0
            if score > best_score:
                best_score = score
                best_pair = (prob, margin)
        thresholds[ridx] = best_pair
    actions = _apply_regime_thresholds(proba, regime_ids, thresholds)
    ev = _eval_actions(frame, actions, labels, args.fee, args.slip, args.unit_exposure, args.max_hold_bars)
    return {"kind": "regime_threshold", "thresholds": {REGIMES[k]: v for k, v in thresholds.items()}, "actions": actions, **ev}


def _tune_future_gate(frame: pd.DataFrame, base_actions: np.ndarray, labels: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for conf in [0.0, 0.35, 0.45, 0.55, 0.65]:
        for whip in [0.35, 0.45, 0.55, 0.70, 1.01]:
            for ent in [0.95, 1.05, 1.20, 1.40, 9.0]:
                for agree in [False, True]:
                    actions = _gate_future(frame, base_actions, confidence_min=conf, whipsaw_max=whip, entropy_max=ent, require_trend_agree=agree)
                    ev = _eval_actions(frame, actions, labels, args.fee, args.slip, args.unit_exposure, args.max_hold_bars)
                    row = {"kind": "future_gate", "confidence_min": conf, "whipsaw_max": whip, "entropy_max": ent, "require_trend_agree": agree, "actions": actions, **ev}
                    if best is None or float(row["score"]) > float(best["score"]):
                        best = row
    assert best is not None
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Train/tune Alpha5.11 HGB direction master with direction labels, regime thresholds, future gates, and legacy cluster ablation.")
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
    p.add_argument("--label-configs", default="edge004_a2_m002,edge006_a2_m003,edge008_a2_m004")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--tracks", default="regime4_core,regime4_core_future,regime4_core_legacy_cluster")
    p.add_argument("--weight-modes", default="balanced,balanced_confidence")
    p.add_argument("--prob-thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.93,0.95")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12,0.16,0.20,0.25")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=51101)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_train = _read(args.regime4_train_csv)
    raw_eval = _read(args.regime4_eval_csv)
    legacy_train = _read(args.legacy_train_csv)
    legacy_eval = _read(args.legacy_eval_csv)
    _assert_same_clock(raw_train, legacy_train, "train")
    _assert_same_clock(raw_eval, legacy_eval, "eval")
    train_all_with_cluster = _merge_legacy_cluster(raw_train, legacy_train)
    eval_with_cluster = _merge_legacy_cluster(raw_eval, legacy_eval)
    audit = _verify_state24_sticky090_inputs(raw_train, raw_eval, args.manifest, args.clean4_report)
    train_df = _split(raw_train, None, args.train_end)
    val_df = _split(raw_train, args.val_start, args.val_end)
    eval_df = raw_eval.reset_index(drop=True)
    train_cluster = _split(train_all_with_cluster, None, args.train_end)
    val_cluster = _split(train_all_with_cluster, args.val_start, args.val_end)
    eval_cluster = eval_with_cluster.reset_index(drop=True)

    horizons = _horizons(args.horizons)
    label_cfgs = {}
    for raw in [x.strip() for x in args.label_configs.split(",") if x.strip()]:
        # Format: edge004_a2_m002 means 0.004 edge, 2 agreeing horizons, 0.002 margin.
        parts = raw.split("_")
        edge = float(parts[0].replace("edge", "0."))
        agree = int(parts[1].replace("a", ""))
        margin = float(parts[2].replace("m", "0."))
        label_cfgs[raw] = {"min_edge": edge, "min_agree": agree, "margin_edge": margin}

    print(
        json.dumps(
            {
                "stage": "start",
                "model_id": MODEL_ID,
                "rows": {"train": len(train_df), "validation": len(val_df), "oos": len(eval_df)},
                "horizons": horizons,
                "label_cfgs": label_cfgs,
                "tracks": args.tracks.split(","),
                "audit": {"expected_model_found_in_manifest": audit.get("expected_model_found_in_manifest"), "legacy_v4_count": audit.get("legacy_v4_count")},
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )

    feature_payloads: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]] = {}
    base_cols = _base_features(raw_train, raw_eval, include_future=False)
    future_cols = _base_features(raw_train, raw_eval, include_future=True)
    cluster_cols = base_cols + [c for c in LEGACY_CLUSTER_COLS if c in train_all_with_cluster.columns and c not in base_cols]
    for track in [x.strip() for x in args.tracks.split(",") if x.strip()]:
        if track == "regime4_core":
            cols, tr, va, oo = base_cols, train_df, val_df, eval_df
        elif track == "regime4_core_future":
            cols, tr, va, oo = future_cols, train_df, val_df, eval_df
        elif track == "regime4_core_legacy_cluster":
            cols, tr, va, oo = cluster_cols, train_cluster, val_cluster, eval_cluster
        else:
            raise ValueError(track)
        feature_payloads[track] = (_x(tr, cols), _x(va, cols), _x(oo, cols), cols)
        print(
            json.dumps(
                {
                    "stage": "features_ready",
                    "track": track,
                    "feature_count": len(cols),
                    "regime4_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in cols)),
                    "future_pred_count": int(sum(c.startswith("regime4_pred_") for c in cols)),
                    "legacy_cluster_count": int(sum(c in LEGACY_CLUSTER_COLS for c in cols)),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    val_labels_by_cfg = {}
    eval_labels_by_cfg = {}
    train_labels_by_cfg = {}
    for name, cfg in label_cfgs.items():
        train_labels_by_cfg[name] = _direction_labels(train_df, horizons=horizons, fee=args.fee, slip=args.slip, **cfg)
        val_labels_by_cfg[name] = _direction_labels(val_df, horizons=horizons, fee=args.fee, slip=args.slip, **cfg)
        eval_labels_by_cfg[name] = _direction_labels(eval_df, horizons=horizons, fee=args.fee, slip=args.slip, **cfg)
        print(json.dumps({"stage": "label_built", "label_cfg": name, "train": train_labels_by_cfg[name]["report"], "validation": val_labels_by_cfg[name]["report"], "oos": eval_labels_by_cfg[name]["report"]}, ensure_ascii=False, default=_json_default), flush=True)

    rows: list[dict[str, Any]] = []
    hgb_specs = _hgb_specs()
    weight_modes = [x.strip() for x in args.weight_modes.split(",") if x.strip()]
    total = len(label_cfgs) * len(feature_payloads) * len(weight_modes) * len(hgb_specs)
    done = 0
    for cfg_i, (label_name, label_payload) in enumerate(train_labels_by_cfg.items()):
        sample_idx = _sample_indices(label_payload["valid_idx"], args.stride)
        y_train = label_payload["action"][sample_idx].astype(np.int64)
        conf_train = label_payload["confidence"][sample_idx].astype(np.float64)
        for track_i, (track, (x_train_full, x_val, x_eval, cols)) in enumerate(feature_payloads.items()):
            x_train = x_train_full.iloc[sample_idx].reset_index(drop=True)
            for weight_i, weight_mode in enumerate(weight_modes):
                sw = _sample_weight(y_train, conf_train, weight_mode)
                for spec_i, spec in enumerate(hgb_specs):
                    done += 1
                    print(json.dumps({"stage": "fit", "done": done, "total": total, "label_cfg": label_name, "track": track, "weight_mode": weight_mode, "hgb": spec.name}, ensure_ascii=False), flush=True)
                    model = _fit_hgb(x_train, y_train, sw, spec, args.seed + cfg_i * 1000 + track_i * 200 + weight_i * 50 + spec_i)
                    val_proba = _predict_proba_3(model, x_val)
                    eval_proba = _predict_proba_3(model, x_eval)
                    val_label = val_labels_by_cfg[label_name]
                    eval_label = eval_labels_by_cfg[label_name]
                    global_best = _tune_global(val_df, val_proba, val_label, args)
                    regime_best = _tune_regime(val_df, val_proba, val_label, args)
                    future_best = _tune_future_gate(val_df, regime_best["actions"], val_label, args) if track != "regime4_core_legacy_cluster" else None

                    candidates = [("global", global_best), ("regime_threshold", regime_best)]
                    if future_best is not None:
                        candidates.append(("regime_threshold_future_gate", future_best))
                    selected_name, selected_val = max(candidates, key=lambda kv: float(kv[1]["score"]))

                    if selected_name == "global":
                        eval_actions = _decide_actions(eval_proba, selected_val["prob"], selected_val["margin"])
                    elif selected_name == "regime_threshold":
                        eval_actions = _apply_regime_thresholds(eval_proba, _regime_ids(eval_df), {i: tuple(selected_val["thresholds"][REGIMES[i]]) for i in range(len(REGIMES))})
                    else:
                        base_eval_actions = _apply_regime_thresholds(eval_proba, _regime_ids(eval_df), {i: tuple(regime_best["thresholds"][REGIMES[i]]) for i in range(len(REGIMES))})
                        eval_actions = _gate_future(
                            eval_df,
                            base_eval_actions,
                            confidence_min=selected_val["confidence_min"],
                            whipsaw_max=selected_val["whipsaw_max"],
                            entropy_max=selected_val["entropy_max"],
                            require_trend_agree=selected_val["require_trend_agree"],
                        )
                    eval_result = _eval_actions(eval_df, eval_actions, eval_label, args.fee, args.slip, args.unit_exposure, args.max_hold_bars)
                    artifact = args.out_dir / f"{label_name}_{track}_{weight_mode}_{spec.name}_direction_hgb.joblib"
                    joblib.dump(
                        {
                            "model_id": MODEL_ID,
                            "model": model,
                            "feature_cols": cols,
                            "label_cfg": {"name": label_name, **label_cfgs[label_name], "horizons": list(horizons)},
                            "track": track,
                            "weight_mode": weight_mode,
                            "hgb": asdict(spec),
                            "selected_decision": {k: v for k, v in selected_val.items() if k not in {"actions", "backtest", "direction"}},
                        },
                        artifact,
                    )
                    row = {
                        "label_cfg": label_name,
                        "track": track,
                        "weight_mode": weight_mode,
                        "hgb": asdict(spec),
                        "decision": selected_name,
                        "validation": {k: v for k, v in selected_val.items() if k != "actions"},
                        "oos": eval_result,
                        "feature_count": len(cols),
                        "artifact": str(artifact),
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "stage": "candidate",
                                "label_cfg": label_name,
                                "track": track,
                                "weight_mode": weight_mode,
                                "hgb": spec.name,
                                "decision": selected_name,
                                "val_score": selected_val["score"],
                                "val_dir": selected_val["direction"],
                                "val_cost1": selected_val["backtest"]["cost1"],
                                "oos_score": eval_result["score"],
                                "oos_dir": eval_result["direction"],
                                "oos_cost1": eval_result["backtest"]["cost1"],
                            },
                            ensure_ascii=False,
                            default=_json_default,
                        ),
                        flush=True,
                    )

    best = max(rows, key=lambda r: float(r["validation"]["score"]))
    summary = {
        "model_id": MODEL_ID,
        "design": "HGB direction-only parent. Multi-horizon direction labels, precision/coverage selection, regime-conditioned thresholds, optional future-regime gate, and legacy v4 cluster/state ablation.",
        "state24_sticky090_audit": audit,
        "experiments": rows,
        "best": best,
        "top20": sorted(rows, key=lambda r: float(r["validation"]["score"]), reverse=True)[:20],
    }
    summary_path = args.out_dir / "alpha5_11_hgb_direction_master_summary.json"
    grid_path = args.out_dir / "alpha5_11_hgb_direction_master_grid.csv"
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
                "val_precision": r["validation"]["direction"]["trade_precision"],
                "val_bal_precision": r["validation"]["direction"]["balanced_trade_precision"],
                "val_coverage": r["validation"]["direction"]["coverage"],
                "val_trades_pred": r["validation"]["direction"]["trades_pred"],
                "val_cost1_pnl": r["validation"]["backtest"]["cost1"]["pnl"],
                "val_cost1_mdd": r["validation"]["backtest"]["cost1"]["mdd"],
                "val_cost1_trades": r["validation"]["backtest"]["cost1"]["trades"],
                "oos_score": r["oos"]["score"],
                "oos_precision": r["oos"]["direction"]["trade_precision"],
                "oos_bal_precision": r["oos"]["direction"]["balanced_trade_precision"],
                "oos_coverage": r["oos"]["direction"]["coverage"],
                "oos_trades_pred": r["oos"]["direction"]["trades_pred"],
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
    print(
        json.dumps(
            {
                "stage": "complete",
                "summary": str(summary_path),
                "grid": str(grid_path),
                "best": {
                    "label_cfg": best["label_cfg"],
                    "track": best["track"],
                    "weight_mode": best["weight_mode"],
                    "hgb": best["hgb"]["name"],
                    "decision": best["decision"],
                    "val_score": best["validation"]["score"],
                    "oos_score": best["oos"]["score"],
                    "oos_cost1": best["oos"]["backtest"]["cost1"],
                    "oos_direction": best["oos"]["direction"],
                },
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
