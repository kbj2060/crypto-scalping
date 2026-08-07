#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha5_governor_v1_infer_20260519 import load_governor, predict_heads  # noqa: E402
from scripts.train_eval_alpha5_23_hgb_direction_refined_20260519 import _eval  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha5_governor_v1_meta_veto_20260519"
DEFAULT_MODEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_governor_v1_contracts_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable005_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_governor_v1_meta_veto_20260519"

META_FEATURES = [
    "p_clean_wait",
    "p_ambiguous",
    "p_amb_trade_like",
    "p_amb_structural",
    "p_trade",
    "p_long",
    "p_short",
    "quality_pred",
    "trade_score_loose",
    "best_side",
    "side_margin",
    "score_abs",
    "volatility_z",
    "garch_vol_z",
    "funding_pressure",
    "funding_abs",
    "smart_money_flow",
    "net_taker_ratio",
    "whale_retail_ratio",
    "breakout_strength",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "clean_regime4_2024_unsup_v1_bull_prob",
    "clean_regime4_2024_unsup_v1_bear_prob",
    "clean_regime4_2024_unsup_v1_chop_prob",
    "clean_regime4_2024_unsup_v1_whipsaw_prob",
    "clean_regime4_2024_unsup_v1_confidence",
    "clean_regime4_2024_unsup_v1_directional_bias",
    "clean_regime4_2024_unsup_v1_transition_risk",
    "session_us",
    "session_europe",
]


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame.get(col, default), errors="coerce").fillna(default)


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.ones(len(y), dtype=np.float64)
    cls, cnt = np.unique(y, return_counts=True)
    total = float(len(y))
    for c, n in zip(cls, cnt):
        out[y == int(c)] = total / (len(cls) * max(float(n), 1.0))
    return out


def _candidate_frame(frame: pd.DataFrame, head_pred: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    pred = head_pred.reset_index(drop=True).copy()
    pred["trade_score_loose"] = (
        pred["p_trade"]
        - 0.50 * pred["p_clean_wait"]
        - 0.35 * pred["p_amb_structural"] * pred["p_ambiguous"]
        + 0.20 * pred["quality_pred"]
    )
    pred["best_side"] = np.maximum(pred["p_long"], pred["p_short"])
    pred["side_margin"] = np.abs(pred["p_long"] - pred["p_short"])
    pred["score_abs"] = np.abs(pred["current_direction_score"])
    pred["candidate_action"] = np.where(pred["p_long"] >= pred["p_short"], 1, 2).astype(np.int64)
    full = pd.concat([out, pred], axis=1)
    keep = (
        (full["regime4_state"].astype(str) != "whipsaw")
        & (full["trade_score_loose"] >= -0.05)
        & (full["best_side"] >= 0.48)
        & (full["score_abs"] >= 0.05)
    )
    return full.loc[keep].reset_index(drop=True)


def _execute_target(frame: pd.DataFrame) -> np.ndarray:
    label_action = _num(frame, "label_action", 0.0).astype(np.int64).to_numpy()
    candidate_action = _num(frame, "candidate_action", 0.0).astype(np.int64).to_numpy()
    profitable = _num(frame, "meta_is_profitable", 0.0).astype(np.int8).to_numpy() == 1
    tp_first = _num(frame, "meta_tp_first", 0.0).astype(np.int8).to_numpy() == 1
    event_ret = _num(frame, "meta_event_return", 0.0).to_numpy(np.float64)
    return ((candidate_action == label_action) & profitable & tp_first & (event_ret >= 0.004)).astype(np.int64)


def _meta_x(frame: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=frame.index)
    for col in META_FEATURES:
        x[col] = _num(frame, col, 0.0)
    return x.astype(np.float32)


def _report_binary(y_true: np.ndarray, y_pred: np.ndarray, p1: np.ndarray) -> dict[str, Any]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "positive_rate": float(np.mean(y_pred == 1)),
        "prob_mean": float(np.mean(p1)),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0),
    }


def _fit_meta(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int,
    devices: str,
) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=900,
        depth=7,
        learning_rate=0.03,
        l2_leaf_reg=4.0,
        random_strength=0.6,
        bagging_temperature=0.15,
        random_seed=seed,
        task_type="GPU",
        devices=devices,
        allow_writing_files=False,
        verbose=100,
        use_best_model=True,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=(x_val, y_val),
        early_stopping_rounds=100,
        verbose=100,
    )
    return model


def _compose_with_meta(
    candidate_frame: pd.DataFrame,
    meta_prob: np.ndarray,
    *,
    meta_threshold: float,
    side_threshold: float,
    side_margin_min: float,
    quality_min: float,
) -> np.ndarray:
    actions = _num(candidate_frame, "candidate_action", 0.0).astype(np.int64).to_numpy()
    best_side = _num(candidate_frame, "best_side", 0.0).to_numpy(np.float64)
    side_margin = _num(candidate_frame, "side_margin", 0.0).to_numpy(np.float64)
    quality = _num(candidate_frame, "quality_pred", 0.0).to_numpy(np.float64)
    keep = (
        (meta_prob >= float(meta_threshold))
        & (best_side >= float(side_threshold))
        & (side_margin >= float(side_margin_min))
        & (quality >= float(quality_min))
    )
    return np.where(keep, actions, 0).astype(np.int64)


def _lift_to_full(frame: pd.DataFrame, candidate_actions: np.ndarray) -> np.ndarray:
    full = np.zeros(len(frame), dtype=np.int64)
    full[np.asarray(frame.index)] = candidate_actions
    return full


def main() -> None:
    p = argparse.ArgumentParser(description="Train alpha5 governor v1 execute/veto meta gate and evaluate looser candidate trading.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--devices", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--max-hold-bars", type=int, default=96)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    governor = load_governor(args.model_dir)

    train_df = pd.read_parquet(args.data_dir / "alpha5_30_direction_learnable_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_30_direction_learnable_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_30_direction_learnable_oos.parquet")

    train_c = _candidate_frame(train_df, predict_heads(governor, train_df))
    val_c = _candidate_frame(val_df, predict_heads(governor, val_df))
    oos_c = _candidate_frame(oos_df, predict_heads(governor, oos_df))

    y_train = _execute_target(train_c)
    y_val = _execute_target(val_c)
    y_oos = _execute_target(oos_c)
    x_train = _meta_x(train_c)
    x_val = _meta_x(val_c)
    x_oos = _meta_x(oos_c)

    base_weight = np.clip(_num(train_c, "quality_pred", 0.0).abs().to_numpy(np.float64) + 0.25, 1e-4, None)
    sample_weight = base_weight * _balanced_weights(y_train)

    t0 = time.perf_counter()
    model = _fit_meta(x_train, y_train, sample_weight, x_val, y_val, seed=args.seed, devices=args.devices)
    fit_seconds = float(time.perf_counter() - t0)

    p_val = np.asarray(model.predict_proba(x_val), dtype=np.float64)[:, 1]
    p_oos = np.asarray(model.predict_proba(x_oos), dtype=np.float64)[:, 1]
    pred_val = (p_val >= 0.5).astype(np.int64)
    pred_oos = (p_oos >= 0.5).astype(np.int64)

    labels_val_full = _num(val_df, "label_action", 0.0).astype(np.int64).to_numpy()
    labels_oos_full = _num(oos_df, "label_action", 0.0).astype(np.int64).to_numpy()

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for meta_threshold, side_threshold, side_margin_min, quality_min in product(
        [0.35, 0.45, 0.55, 0.65],
        [0.48, 0.50, 0.55],
        [0.00, 0.03, 0.05],
        [-0.10, -0.05, 0.00],
    ):
        val_actions_candidate = _compose_with_meta(
            val_c,
            p_val,
            meta_threshold=meta_threshold,
            side_threshold=side_threshold,
            side_margin_min=side_margin_min,
            quality_min=quality_min,
        )
        val_actions_full = np.zeros(len(val_df), dtype=np.int64)
        val_actions_full[val_c.index.to_numpy()] = val_actions_candidate
        val_eval = _eval(val_df, val_actions_full, labels_val_full, fee=args.fee, slip=args.slip, exposure=args.exposure, max_hold=args.max_hold_bars)
        trades = int(val_eval["backtest"]["cost1"]["trades"])
        selection_score = float(val_eval["score"]) + min(trades, 120) * 0.20 - max(0, trades - 180) * 0.08
        row = {
            "meta_threshold": float(meta_threshold),
            "side_threshold": float(side_threshold),
            "side_margin_min": float(side_margin_min),
            "quality_min": float(quality_min),
            "selection_score": selection_score,
            "val_score": float(val_eval["score"]),
            "val_cost1_pnl": float(val_eval["backtest"]["cost1"]["pnl"]),
            "val_cost1_mdd": float(val_eval["backtest"]["cost1"]["mdd"]),
            "val_trades": trades,
            "val_trade_precision": float(val_eval["direction"]["trade_precision"]),
            "val_balanced_trade_precision": float(val_eval["direction"]["balanced_trade_precision"]),
        }
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row

    assert best is not None
    oos_actions_candidate = _compose_with_meta(
        oos_c,
        p_oos,
        meta_threshold=float(best["meta_threshold"]),
        side_threshold=float(best["side_threshold"]),
        side_margin_min=float(best["side_margin_min"]),
        quality_min=float(best["quality_min"]),
    )
    oos_actions_full = np.zeros(len(oos_df), dtype=np.int64)
    oos_actions_full[oos_c.index.to_numpy()] = oos_actions_candidate
    oos_eval = _eval(oos_df, oos_actions_full, labels_oos_full, fee=args.fee, slip=args.slip, exposure=args.exposure, max_hold=args.max_hold_bars)

    model_path = args.out_dir / "execute_veto_catboost_gpu.cbm"
    meta_path = args.out_dir / "execute_veto_meta.joblib"
    model.save_model(str(model_path))
    joblib.dump({"feature_cols": META_FEATURES, "candidate_rules": "loose_v1"}, meta_path)

    summary = {
        "model_id": MODEL_ID,
        "fit_seconds": fit_seconds,
        "candidate_rows": {
            "train": int(len(train_c)),
            "val": int(len(val_c)),
            "oos": int(len(oos_c)),
        },
        "target_rate": {
            "train": float(np.mean(y_train)),
            "val": float(np.mean(y_val)),
            "oos": float(np.mean(y_oos)),
        },
        "validation_meta": _report_binary(y_val, pred_val, p_val),
        "oos_meta": _report_binary(y_oos, pred_oos, p_oos),
        "selection": best,
        "oos": {
            "cost1": oos_eval["backtest"]["cost1"],
            "cost2": oos_eval["backtest"]["cost2"],
            "cost3": oos_eval["backtest"]["cost3"],
            "direction": oos_eval["direction"],
            "score": float(oos_eval["score"]),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(args.out_dir / "grid.csv", index=False)
    print(json.dumps({"stage": "alpha5_governor_v1_meta_veto_done", "summary_path": str(args.out_dir / "summary.json"), "oos_pnl": float(oos_eval["backtest"]["cost1"]["pnl"]), "oos_trades": int(oos_eval["backtest"]["cost1"]["trades"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
