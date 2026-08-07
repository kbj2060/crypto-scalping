#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha5_router_v2_train_20260519 import ROUTER_FEATURE_COLS, _ensure_momentum  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha5_router_v3_ovr_train_20260520"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_29_hier_label_factory_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_router_v3_ovr_contracts_20260520"


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(frame.get(col, default), errors="coerce").fillna(default).to_numpy(np.float64)


def _router_label(frame: pd.DataFrame) -> np.ndarray:
    split_keep = _num(frame, "split_keep", 0.0).astype(np.int8) == 1
    entry_state = _num(frame, "entry_state", 1.0).astype(np.int8)
    direction_label = _num(frame, "direction_label", 0.0).astype(np.int8)
    tp_first = _num(frame, "meta_tp_first", 0.0).astype(np.int8) == 1
    profitable = _num(frame, "meta_is_profitable", 0.0).astype(np.int8) == 1
    regime = frame.get("regime4_state", pd.Series(["unknown"] * len(frame))).astype(str).to_numpy()
    non_whipsaw = regime != "whipsaw"
    long_mask = split_keep & (entry_state == 2) & (direction_label == 1) & tp_first & profitable & non_whipsaw
    short_mask = split_keep & (entry_state == 2) & (direction_label == 2) & tp_first & profitable & non_whipsaw
    y = np.zeros(len(frame), dtype=np.int64)
    y[long_mask] = 1
    y[short_mask] = 2
    return y


def _base_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    work = _ensure_momentum(frame.copy())
    for col in ROUTER_FEATURE_COLS:
        if col not in work.columns:
            work[col] = np.nan
    X = work[ROUTER_FEATURE_COLS].copy()
    for col in ROUTER_FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = _router_label(work)
    keep = _num(work, "split_keep", 0.0).astype(np.int8) == 1
    return work.loc[keep].reset_index(drop=True), y[keep]


def _binary_weights(frame: pd.DataFrame, y3: np.ndarray, positive_class: int) -> np.ndarray:
    entry_state = _num(frame, "entry_state", 1.0).astype(np.int8)
    subtype = _num(frame, "ambiguous_subtype", 0.0).astype(np.int8)
    label_conf = np.clip(_num(frame, "label_confidence", 0.0), 0.0, 1.0)
    uniq = np.clip(_num(frame, "sample_uniqueness_weight", 0.0), 0.0, 1.0)
    quality = np.abs(_num(frame, "quality_score", 0.0))
    regime = frame.get("regime4_state", pd.Series(["unknown"] * len(frame))).astype(str).to_numpy()

    yb = (y3 == positive_class).astype(np.int8)
    opposite = 2 if positive_class == 1 else 1

    w = np.ones(len(frame), dtype=np.float64)
    w *= 0.70 + 0.30 * label_conf
    w *= 0.75 + 0.25 * uniq
    w *= np.where(y3 == positive_class, 1.25 + np.clip(quality, 0.0, 1.0), 1.0)
    w *= np.where(y3 == opposite, 2.10, 1.0)
    w *= np.where((entry_state == 1) & (subtype == 2), 2.30, 1.0)  # trade-like ambiguous hard negatives
    w *= np.where((entry_state == 1) & (subtype == 1), 1.35, 1.0)  # structural ambiguous
    w *= np.where((entry_state == 0) & (regime == "whipsaw"), 1.15, 1.0)
    if positive_class == 1:
        w *= np.where(regime == "bull", 1.15, 1.0)
        w *= np.where(regime == "bear", 0.95, 1.0)
    else:
        w *= np.where(regime == "bear", 1.15, 1.0)
        w *= np.where(regime == "bull", 0.95, 1.0)

    pos = max(float(yb.sum()), 1.0)
    neg = max(float(len(yb) - yb.sum()), 1.0)
    pos_w = len(yb) / (2.0 * pos)
    neg_w = len(yb) / (2.0 * neg)
    w *= np.where(yb == 1, pos_w, neg_w)
    return np.clip(w, 1e-4, None)


def _fit_binary(
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
        iterations=1400,
        depth=7,
        learning_rate=0.025,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        bagging_temperature=0.4,
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
        early_stopping_rounds=140,
        verbose=100,
    )
    return model


def _compose_pred(p_long: np.ndarray, p_short: np.ndarray, long_threshold: float, short_threshold: float, min_margin: float) -> np.ndarray:
    pred = np.zeros(len(p_long), dtype=np.int64)
    long_ok = (p_long >= long_threshold) & ((p_long - p_short) >= min_margin)
    short_ok = (p_short >= short_threshold) & ((p_short - p_long) >= min_margin)
    pred[long_ok & ~short_ok] = 1
    pred[short_ok & ~long_ok] = 2
    ties = long_ok & short_ok
    pred[ties & ((p_long - p_short) > 0.0)] = 1
    pred[ties & ((p_short - p_long) > 0.0)] = 2
    return pred


def _search_thresholds(y_true: np.ndarray, p_long: np.ndarray, p_short: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    best_score = -1.0
    for t_long in (0.35, 0.40, 0.45, 0.50, 0.55):
        for t_short in (0.35, 0.40, 0.45, 0.50, 0.55):
            for min_margin in (0.00, 0.02, 0.04, 0.06, 0.08):
                pred = _compose_pred(p_long, p_short, t_long, t_short, min_margin)
                bal = float(balanced_accuracy_score(y_true, pred))
                macro = float(f1_score(y_true, pred, average="macro"))
                none_rate = float(np.mean(pred == 0))
                score = bal + 0.30 * macro - 0.04 * abs(none_rate - 0.70)
                if score > best_score:
                    best_score = score
                    best = {
                        "long_threshold": t_long,
                        "short_threshold": t_short,
                        "min_margin": min_margin,
                        "val_balanced_accuracy": bal,
                        "val_macro_f1": macro,
                        "val_none_rate": none_rate,
                    }
    assert best is not None
    return best


def _report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "class_counts": {str(int(k)): int(v) for k, v in pd.Series(y_true).value_counts().sort_index().to_dict().items()},
        "pred_counts": {str(int(k)): int(v) for k, v in pd.Series(y_pred).value_counts().sort_index().to_dict().items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train alpha5 router v3 as long/short OVR pair with abstain composition.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--devices", default="0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_oos.parquet")

    train_work, y_train3 = _base_frame(train_df)
    val_work, y_val3 = _base_frame(val_df)
    oos_work, y_oos3 = _base_frame(oos_df)
    x_train = train_work[ROUTER_FEATURE_COLS]
    x_val = val_work[ROUTER_FEATURE_COLS]
    x_oos = oos_work[ROUTER_FEATURE_COLS]

    y_train_long = (y_train3 == 1).astype(np.int8)
    y_val_long = (y_val3 == 1).astype(np.int8)
    y_train_short = (y_train3 == 2).astype(np.int8)
    y_val_short = (y_val3 == 2).astype(np.int8)
    w_long = _binary_weights(train_work, y_train3, positive_class=1)
    w_short = _binary_weights(train_work, y_train3, positive_class=2)

    long_model = _fit_binary(x_train, y_train_long, w_long, x_val, y_val_long, seed=args.seed, devices=args.devices)
    short_model = _fit_binary(x_train, y_train_short, w_short, x_val, y_val_short, seed=args.seed + 17, devices=args.devices)

    p_long_val = np.asarray(long_model.predict_proba(x_val), dtype=np.float32)[:, 1]
    p_short_val = np.asarray(short_model.predict_proba(x_val), dtype=np.float32)[:, 1]
    best = _search_thresholds(y_val3, p_long_val, p_short_val)

    p_long_oos = np.asarray(long_model.predict_proba(x_oos), dtype=np.float32)[:, 1]
    p_short_oos = np.asarray(short_model.predict_proba(x_oos), dtype=np.float32)[:, 1]
    pred_val = _compose_pred(p_long_val, p_short_val, best["long_threshold"], best["short_threshold"], best["min_margin"])
    pred_oos = _compose_pred(p_long_oos, p_short_oos, best["long_threshold"], best["short_threshold"], best["min_margin"])

    long_model_path = args.out_dir / "router_long_catboost_gpu.cbm"
    short_model_path = args.out_dir / "router_short_catboost_gpu.cbm"
    meta_path = args.out_dir / "router_ovr_meta.joblib"
    report_path = args.out_dir / "router_ovr_summary.json"
    long_model.save_model(str(long_model_path))
    short_model.save_model(str(short_model_path))
    joblib.dump(
        {
            "feature_cols": list(ROUTER_FEATURE_COLS),
            "type": "router_ovr_pair",
            "long_model_path": str(long_model_path),
            "short_model_path": str(short_model_path),
            "classes": [0, 1, 2],
            "label_meaning": {"0": "none", "1": "long", "2": "short"},
            "long_threshold": float(best["long_threshold"]),
            "short_threshold": float(best["short_threshold"]),
            "min_margin": float(best["min_margin"]),
        },
        meta_path,
    )
    payload = {
        "model_id": MODEL_ID,
        "long_model_path": str(long_model_path),
        "short_model_path": str(short_model_path),
        "meta_path": str(meta_path),
        "feature_cols": list(ROUTER_FEATURE_COLS),
        "train_rows": int(len(y_train3)),
        "val_rows": int(len(y_val3)),
        "oos_rows": int(len(y_oos3)),
        "thresholds": best,
        "val": _report(y_val3, pred_val),
        "oos": _report(y_oos3, pred_oos),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"stage": "router_ovr_fit_done", "report_path": str(report_path), "val_bal_acc": payload["val"]["balanced_accuracy"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
