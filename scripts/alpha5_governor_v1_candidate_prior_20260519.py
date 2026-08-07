#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_alpha5_27_label_factory_contracts_20260519 import _feature_importance, _num  # noqa: E402
from scripts.train_eval_alpha5_23_hgb_direction_refined_20260519 import (  # noqa: E402
    _direction_feature_cols,
    _entry_feature_cols,
    _feature_cols,
    _x,
)
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


MODEL_ID = "alpha5_governor_v1_candidate_prior_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable005_20260519"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_governor_v1_candidate_prior_20260519"


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.ones(len(y), dtype=np.float64)
    cls, cnt = np.unique(y, return_counts=True)
    total = float(len(y))
    for c, n in zip(cls, cnt):
        out[y == int(c)] = total / (len(cls) * max(float(n), 1.0))
    return out


def _candidate_target(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_action = np.asarray(_num(frame, "label_action", 0.0), dtype=np.int64)
    learnable = np.asarray(_num(frame, "direction_train_keep30", 0.0), dtype=np.int8) == 1
    tp_first = np.asarray(_num(frame, "meta_tp_first", 0.0), dtype=np.int8) == 1
    profitable = np.asarray(_num(frame, "meta_is_profitable", 0.0), dtype=np.int8) == 1
    event_ret = np.asarray(_num(frame, "meta_event_return", 0.0), dtype=np.float64)
    entry_state = np.asarray(_num(frame, "entry_state", 1.0), dtype=np.int8)
    ambiguous_subtype = np.asarray(_num(frame, "ambiguous_subtype", 0.0), dtype=np.int8)
    regime = frame.get("regime4_state", "unknown").astype(str).to_numpy()

    positive = (
        (label_action != 0)
        & learnable
        & tp_first
        & profitable
        & (event_ret >= 0.004)
        & (regime != "whipsaw")
    )
    negative = (
        ((entry_state == 0) | (ambiguous_subtype == 1) | (regime == "whipsaw"))
        & ~positive
    )
    keep = positive | negative
    y = positive.astype(np.int64)
    return y, keep.astype(bool), positive.astype(bool)


def _fit_model(
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
        iterations=1000,
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


def _report(y_true: np.ndarray, p1: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (p1 >= float(threshold)).astype(np.int64)
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "positive_rate": float(np.mean(pred == 1)),
        "prob_mean": float(np.mean(p1)),
        "classification_report": classification_report(y_true, pred, labels=[0, 1], output_dict=True, zero_division=0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train candidate-prior head for alpha5 governor v1.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--devices", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)

    train_df = pd.read_parquet(args.data_dir / "alpha5_30_direction_learnable_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_30_direction_learnable_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_30_direction_learnable_oos.parquet")

    cols = _feature_cols(raw_2025, raw_2026, set(train_df.columns))
    feature_cols = list(dict.fromkeys(_entry_feature_cols(cols) + _direction_feature_cols(cols)))
    x_train = _x(train_df, feature_cols)
    x_val = _x(val_df, feature_cols)
    x_oos = _x(oos_df, feature_cols)

    y_train, keep_train, pos_train = _candidate_target(train_df)
    y_val, keep_val, pos_val = _candidate_target(val_df)
    y_oos, keep_oos, pos_oos = _candidate_target(oos_df)

    base_weight = np.clip(np.asarray(_num(train_df, "entry_sample_weight", 1.0), dtype=np.float64), 1e-4, None)
    dir_weight = np.clip(np.asarray(_num(train_df, "direction_sample_weight30", 1.0), dtype=np.float64), 0.0, None)
    w_train = np.where(pos_train, np.maximum(base_weight, dir_weight), base_weight)
    w_train = w_train[keep_train] * _balanced_weights(y_train[keep_train])

    print(
        json.dumps(
            {
                "stage": "candidate_prior_fit",
                "train_rows": int(np.sum(keep_train)),
                "val_rows": int(np.sum(keep_val)),
                "oos_rows": int(np.sum(keep_oos)),
                "train_positive_rate": float(np.mean(y_train[keep_train])),
                "val_positive_rate": float(np.mean(y_val[keep_val])),
                "oos_positive_rate": float(np.mean(y_oos[keep_oos])),
                "feature_count": len(feature_cols),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    t0 = time.perf_counter()
    model = _fit_model(
        x_train.loc[keep_train].reset_index(drop=True),
        y_train[keep_train],
        w_train,
        x_val.loc[keep_val].reset_index(drop=True),
        y_val[keep_val],
        seed=args.seed,
        devices=args.devices,
    )
    fit_seconds = float(time.perf_counter() - t0)

    p_val = np.asarray(model.predict_proba(x_val.loc[keep_val].reset_index(drop=True)), dtype=np.float64)[:, 1]
    p_oos = np.asarray(model.predict_proba(x_oos.loc[keep_oos].reset_index(drop=True)), dtype=np.float64)[:, 1]

    thresholds = [0.35, 0.45, 0.55, 0.65]
    val_reports = []
    best = None
    for th in thresholds:
        rep = _report(y_val[keep_val], p_val, th)
        rep["selection_score"] = rep["precision"] * 2.0 + rep["balanced_accuracy"] + rep["recall"] * 0.25
        val_reports.append(rep)
        if best is None or rep["selection_score"] > best["selection_score"]:
            best = rep
    assert best is not None
    oos_report = _report(y_oos[keep_oos], p_oos, float(best["threshold"]))

    model_path = args.out_dir / "candidate_prior_catboost_gpu.cbm"
    meta_path = args.out_dir / "candidate_prior_meta.joblib"
    model.save_model(str(model_path))
    joblib.dump({"feature_cols": feature_cols}, meta_path)

    summary = {
        "model_id": MODEL_ID,
        "fit_seconds": fit_seconds,
        "feature_count": len(feature_cols),
        "rows": {
            "train": int(np.sum(keep_train)),
            "val": int(np.sum(keep_val)),
            "oos": int(np.sum(keep_oos)),
        },
        "positive_rate": {
            "train": float(np.mean(y_train[keep_train])),
            "val": float(np.mean(y_val[keep_val])),
            "oos": float(np.mean(y_oos[keep_oos])),
        },
        "validation_reports": val_reports,
        "selected_threshold": float(best["threshold"]),
        "oos_report": oos_report,
        "feature_importance": _feature_importance(model, feature_cols),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"stage": "candidate_prior_done", "summary_path": str(args.out_dir / "summary.json"), "selected_threshold": float(best["threshold"]), "oos_precision": oos_report["precision"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
