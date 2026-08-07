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

from scripts.alpha5_router_v2_train_20260519 import ROUTER_FEATURE_COLS  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha5_router_v4_4class_train_20260520"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_29_hier_label_factory_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_router_v4_4class_contracts_20260520"


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(frame.get(col, default), errors="coerce").fillna(default).to_numpy(np.float64)


def _router4_label(frame: pd.DataFrame) -> np.ndarray:
    split_keep = _num(frame, "split_keep", 0.0).astype(np.int8) == 1
    entry_state = _num(frame, "entry_state", 1.0).astype(np.int8)
    direction_label = _num(frame, "direction_label", 0.0).astype(np.int8)
    tp_first = _num(frame, "meta_tp_first", 0.0).astype(np.int8) == 1
    profitable = _num(frame, "meta_is_profitable", 0.0).astype(np.int8) == 1
    regime = frame.get("regime4_state", pd.Series(["unknown"] * len(frame))).astype(str).to_numpy()
    y = np.zeros(len(frame), dtype=np.int64)
    ambiguous = split_keep & (entry_state == 1)
    y[ambiguous] = 1
    long_mask = split_keep & (entry_state == 2) & (direction_label == 1) & tp_first & profitable & (regime != "whipsaw")
    short_mask = split_keep & (entry_state == 2) & (direction_label == 2) & tp_first & profitable & (regime != "whipsaw")
    y[long_mask] = 2
    y[short_mask] = 3
    return y


def _collapse4(y4: np.ndarray) -> np.ndarray:
    y3 = np.zeros(len(y4), dtype=np.int64)
    y3[y4 == 2] = 1
    y3[y4 == 3] = 2
    return y3


def _weights(frame: pd.DataFrame, y4: np.ndarray) -> np.ndarray:
    subtype = _num(frame, "ambiguous_subtype", 0.0).astype(np.int8)
    label_conf = np.clip(_num(frame, "label_confidence", 0.0), 0.0, 1.0)
    uniq = np.clip(_num(frame, "sample_uniqueness_weight", 0.0), 0.0, 1.0)
    quality = np.abs(_num(frame, "quality_score", 0.0))
    regime = frame.get("regime4_state", pd.Series(["unknown"] * len(frame))).astype(str).to_numpy()

    w = np.ones(len(frame), dtype=np.float64)
    w *= 0.70 + 0.30 * label_conf
    w *= 0.75 + 0.25 * uniq
    w *= np.where(y4 == 1, 1.40, 1.0)
    w *= np.where((y4 == 1) & (subtype == 2), 1.60, 1.0)
    w *= np.where(y4 >= 2, 1.20 + np.clip(quality, 0.0, 1.0), 1.0)
    w *= np.where((y4 == 2) & (regime == "bull"), 1.10, 1.0)
    w *= np.where((y4 == 3) & (regime == "bear"), 1.10, 1.0)
    classes, counts = np.unique(y4, return_counts=True)
    total = float(len(y4))
    cw = {int(cls): float(total / (len(classes) * max(float(cnt), 1.0))) for cls, cnt in zip(classes, counts)}
    w *= np.asarray([cw[int(v)] for v in y4], dtype=np.float64)
    return np.clip(w, 1e-4, None)


def _prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = frame.copy()
    for col in ROUTER_FEATURE_COLS:
        if col not in work.columns:
            work[col] = np.nan
    X = work[ROUTER_FEATURE_COLS].copy()
    for col in ROUTER_FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y4 = _router4_label(work)
    keep = _num(work, "split_keep", 0.0).astype(np.int8) == 1
    work = work.loc[keep].reset_index(drop=True)
    return X.loc[keep].reset_index(drop=True), y4[keep], _weights(work, y4[keep])


def _fit(
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
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=1200,
        depth=7,
        learning_rate=0.025,
        l2_leaf_reg=5.0,
        random_strength=0.8,
        bagging_temperature=0.2,
        random_seed=seed,
        task_type="GPU",
        devices=devices,
        allow_writing_files=False,
        verbose=100,
        use_best_model=True,
    )
    model.fit(x_train, y_train, sample_weight=w_train, eval_set=(x_val, y_val), early_stopping_rounds=120, verbose=100)
    return model


def _report(y_true3: np.ndarray, y_pred3: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true3, y_pred3)),
        "macro_f1": float(f1_score(y_true3, y_pred3, average="macro")),
        "confusion_matrix": confusion_matrix(y_true3, y_pred3, labels=labels).tolist(),
        "classification_report": classification_report(y_true3, y_pred3, labels=labels, output_dict=True, zero_division=0),
        "class_counts": {str(int(k)): int(v) for k, v in pd.Series(y_true3).value_counts().sort_index().to_dict().items()},
        "pred_counts": {str(int(k)): int(v) for k, v in pd.Series(y_pred3).value_counts().sort_index().to_dict().items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train alpha5 router v4 as clean_none/ambiguous/long/short router.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--devices", default="0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_oos.parquet")

    x_train, y_train4, w_train = _prepare(train_df)
    x_val, y_val4, _ = _prepare(val_df)
    x_oos, y_oos4, _ = _prepare(oos_df)
    y_val3 = _collapse4(y_val4)
    y_oos3 = _collapse4(y_oos4)

    model = _fit(x_train, y_train4, w_train, x_val, y_val4, seed=args.seed, devices=args.devices)
    pred_val4 = np.asarray(model.predict(x_val)).reshape(-1).astype(np.int64)
    pred_oos4 = np.asarray(model.predict(x_oos)).reshape(-1).astype(np.int64)
    pred_val3 = _collapse4(pred_val4)
    pred_oos3 = _collapse4(pred_oos4)

    model_path = args.out_dir / "router4_catboost_gpu.cbm"
    meta_path = args.out_dir / "router4_meta.joblib"
    report_path = args.out_dir / "router4_summary.json"
    model.save_model(str(model_path))
    joblib.dump(
        {
            "feature_cols": list(ROUTER_FEATURE_COLS),
            "type": "router4_collapse",
            "classes": [0, 1, 2, 3],
            "label_meaning": {"0": "clean_none", "1": "ambiguous_none", "2": "long", "3": "short"},
        },
        meta_path,
    )
    payload = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "feature_cols": list(ROUTER_FEATURE_COLS),
        "train_rows": int(len(y_train4)),
        "val_rows": int(len(y_val4)),
        "oos_rows": int(len(y_oos4)),
        "val": _report(y_val3, pred_val3),
        "oos": _report(y_oos3, pred_oos3),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"stage": "router4_fit_done", "report_path": str(report_path), "val_bal_acc": payload["val"]["balanced_accuracy"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
