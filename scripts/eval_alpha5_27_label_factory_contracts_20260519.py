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
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


MODEL_ID = "alpha5_27_label_factory_contracts_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_27_label_factory_20260519"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_27_label_factory_contracts_20260519"


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(frame.get(col, default), errors="coerce").fillna(default).to_numpy(np.float64)


def _class_weights(y: np.ndarray) -> dict[int, float]:
    y = np.asarray(y, dtype=np.int64)
    classes, counts = np.unique(y, return_counts=True)
    total = float(len(y))
    return {int(cls): float(total / (len(classes) * max(float(cnt), 1.0))) for cls, cnt in zip(classes, counts)}


def _sample_class_weight(y: np.ndarray) -> np.ndarray:
    cw = _class_weights(y)
    return np.asarray([cw[int(v)] for v in np.asarray(y, dtype=np.int64)], dtype=np.float64)


def _fit_entry(
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
        iterations=900,
        depth=7,
        learning_rate=0.03,
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
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=(x_val, y_val),
        early_stopping_rounds=100,
        verbose=100,
    )
    return model


def _fit_direction(
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
        learning_rate=0.028,
        l2_leaf_reg=4.0,
        random_strength=0.6,
        bagging_temperature=0.1,
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


def _fit_quality(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int,
    devices: str,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=900,
        depth=7,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        bagging_temperature=0.2,
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


def _report_entry(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = sorted({int(v) for v in np.unique(np.r_[y_true, y_pred])})
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def _report_direction(y_true: np.ndarray, y_pred: np.ndarray, p_long: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    margin = np.abs(p_long - (1.0 - p_long))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": report,
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "long_rate": float(np.mean(y_pred == 1)),
    }


def _quality_deciles(y_true: np.ndarray, pred: np.ndarray, event_ret: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(pred)
    if len(order) == 0:
        return []
    buckets = np.array_split(order, 10)
    out: list[dict[str, Any]] = []
    for i, idx in enumerate(buckets, start=1):
        if len(idx) == 0:
            continue
        out.append(
            {
                "decile": i,
                "rows": int(len(idx)),
                "pred_mean": float(np.mean(pred[idx])),
                "true_quality_mean": float(np.mean(y_true[idx])),
                "event_return_mean": float(np.mean(event_ret[idx])),
            }
        )
    return out


def _report_quality(y_true: np.ndarray, pred: np.ndarray, event_ret: np.ndarray) -> dict[str, Any]:
    spearman = float(pd.Series(pred).corr(pd.Series(y_true), method="spearman"))
    event_spearman = float(pd.Series(pred).corr(pd.Series(event_ret), method="spearman"))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "spearman_quality": spearman if np.isfinite(spearman) else 0.0,
        "spearman_event_return": event_spearman if np.isfinite(event_spearman) else 0.0,
        "deciles": _quality_deciles(y_true, pred, event_ret),
    }


def _feature_importance(model: Any, cols: list[str], topn: int = 20) -> list[dict[str, Any]]:
    imp = np.asarray(model.get_feature_importance(), dtype=np.float64)
    pairs = sorted(zip(cols, imp.tolist()), key=lambda x: x[1], reverse=True)
    return [{"feature": c, "importance": float(v)} for c, v in pairs[:topn]]


def main() -> None:
    p = argparse.ArgumentParser(description="Validate alpha5_27 label separability with entry_state, direction, and quality CatBoost GPU models.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--devices", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--train-file", default="alpha5_27_label_factory_train.parquet")
    p.add_argument("--val-file", default="alpha5_27_label_factory_val.parquet")
    p.add_argument("--oos-file", default="alpha5_27_label_factory_oos.parquet")
    p.add_argument("--entry-col", default="entry_state")
    p.add_argument("--entry-keep-col", default="entry_train_keep")
    p.add_argument("--entry-weight-col", default="entry_sample_weight")
    p.add_argument("--entry-label-name", default="entry_state")
    p.add_argument("--direction-col", default="direction_label")
    p.add_argument("--direction-keep-col", default="direction_train_keep")
    p.add_argument("--direction-weight-col", default="direction_sample_weight")
    args = p.parse_args()

    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.data_dir / str(args.train_file))
    val_df = pd.read_parquet(args.data_dir / str(args.val_file))
    oos_df = pd.read_parquet(args.data_dir / str(args.oos_file))

    cols = _feature_cols(raw_2025, raw_2026, set(train_df.columns))
    entry_cols = _entry_feature_cols(cols)
    direction_cols = _direction_feature_cols(cols)
    quality_cols = list(dict.fromkeys(entry_cols + direction_cols))
    if not entry_cols or not direction_cols or not quality_cols:
        raise ValueError("failed to select alpha5_27 feature columns")

    x_train_entry = _x(train_df, entry_cols)
    x_val_entry = _x(val_df, entry_cols)
    x_oos_entry = _x(oos_df, entry_cols)
    x_train_dir = _x(train_df, direction_cols)
    x_val_dir = _x(val_df, direction_cols)
    x_oos_dir = _x(oos_df, direction_cols)
    x_train_q = _x(train_df, quality_cols)
    x_val_q = _x(val_df, quality_cols)
    x_oos_q = _x(oos_df, quality_cols)

    entry_keep_tr = _num(train_df, str(args.entry_keep_col), 0.0).astype(np.int8) == 1
    entry_keep_va = _num(val_df, str(args.entry_keep_col), 0.0).astype(np.int8) == 1
    entry_keep_oo = _num(oos_df, str(args.entry_keep_col), 0.0).astype(np.int8) == 1
    dir_keep_tr = _num(train_df, str(args.direction_keep_col), 0.0).astype(np.int8) == 1
    dir_keep_va = _num(val_df, str(args.direction_keep_col), 0.0).astype(np.int8) == 1
    dir_keep_oo = _num(oos_df, str(args.direction_keep_col), 0.0).astype(np.int8) == 1

    y_entry_tr = _num(train_df, str(args.entry_col), 1.0).astype(np.int64)
    y_entry_va = _num(val_df, str(args.entry_col), 1.0).astype(np.int64)
    y_entry_oo = _num(oos_df, str(args.entry_col), 1.0).astype(np.int64)

    y_dir_tr = (_num(train_df, str(args.direction_col), 0.0).astype(np.int64)[dir_keep_tr] == 1).astype(np.int64)
    y_dir_va = (_num(val_df, str(args.direction_col), 0.0).astype(np.int64)[dir_keep_va] == 1).astype(np.int64)
    y_dir_oo = (_num(oos_df, str(args.direction_col), 0.0).astype(np.int64)[dir_keep_oo] == 1).astype(np.int64)

    y_q_tr = _num(train_df, "quality_score", 0.0).astype(np.float64)
    y_q_va = _num(val_df, "quality_score", 0.0).astype(np.float64)
    y_q_oo = _num(oos_df, "quality_score", 0.0).astype(np.float64)
    ev_q_va = _num(val_df, "meta_event_return", 0.0).astype(np.float64)
    ev_q_oo = _num(oos_df, "meta_event_return", 0.0).astype(np.float64)

    w_entry_tr = _num(train_df, str(args.entry_weight_col), 1.0)[entry_keep_tr] * _sample_class_weight(y_entry_tr[entry_keep_tr])
    w_dir_tr = _num(train_df, str(args.direction_weight_col), 1.0)[dir_keep_tr] * _sample_class_weight(y_dir_tr)
    w_q_tr = np.clip(np.abs(y_q_tr[entry_keep_tr]) + 0.25, 1e-4, None)

    print(json.dumps({
        "stage": "alpha5_27_contract_fit",
        "entry_rows": int(np.sum(entry_keep_tr)),
        "direction_rows": int(np.sum(dir_keep_tr)),
        "quality_rows": int(np.sum(entry_keep_tr)),
        "entry_target": str(args.entry_col),
        "direction_target": str(args.direction_col),
        "entry_features": len(entry_cols),
        "direction_features": len(direction_cols),
        "quality_features": len(quality_cols),
    }, ensure_ascii=False), flush=True)

    t0 = time.perf_counter()
    entry_model = _fit_entry(
        x_train_entry.loc[entry_keep_tr].reset_index(drop=True),
        y_entry_tr[entry_keep_tr],
        w_entry_tr,
        x_val_entry.loc[entry_keep_va].reset_index(drop=True),
        y_entry_va[entry_keep_va],
        seed=args.seed + 1,
        devices=args.devices,
    )
    t1 = time.perf_counter()
    direction_model = _fit_direction(
        x_train_dir.loc[dir_keep_tr].reset_index(drop=True),
        y_dir_tr,
        w_dir_tr,
        x_val_dir.loc[dir_keep_va].reset_index(drop=True),
        y_dir_va,
        seed=args.seed + 2,
        devices=args.devices,
    )
    t2 = time.perf_counter()
    quality_model = _fit_quality(
        x_train_q.loc[entry_keep_tr].reset_index(drop=True),
        y_q_tr[entry_keep_tr],
        w_q_tr,
        x_val_q.loc[entry_keep_va].reset_index(drop=True),
        y_q_va[entry_keep_va],
        seed=args.seed + 3,
        devices=args.devices,
    )
    t3 = time.perf_counter()

    entry_pred_val = np.asarray(entry_model.predict(x_val_entry.loc[entry_keep_va].reset_index(drop=True))).reshape(-1).astype(np.int64)
    entry_pred_oos = np.asarray(entry_model.predict(x_oos_entry.loc[entry_keep_oo].reset_index(drop=True))).reshape(-1).astype(np.int64)
    entry_report_val = _report_entry(y_entry_va[entry_keep_va], entry_pred_val)
    entry_report_oos = _report_entry(y_entry_oo[entry_keep_oo], entry_pred_oos)

    p_long_val = np.asarray(direction_model.predict_proba(x_val_dir.loc[dir_keep_va].reset_index(drop=True)), dtype=np.float64)[:, 1]
    p_long_oos = np.asarray(direction_model.predict_proba(x_oos_dir.loc[dir_keep_oo].reset_index(drop=True)), dtype=np.float64)[:, 1]
    dir_pred_val = (p_long_val >= 0.5).astype(np.int64)
    dir_pred_oos = (p_long_oos >= 0.5).astype(np.int64)
    direction_report_val = _report_direction(y_dir_va, dir_pred_val, p_long_val)
    direction_report_oos = _report_direction(y_dir_oo, dir_pred_oos, p_long_oos)

    q_pred_val = np.asarray(quality_model.predict(x_val_q.loc[entry_keep_va].reset_index(drop=True)))
    q_pred_oos = np.asarray(quality_model.predict(x_oos_q.loc[entry_keep_oo].reset_index(drop=True)))
    q_pred_val = np.asarray(q_pred_val, dtype=np.float64).reshape(-1)
    q_pred_oos = np.asarray(q_pred_oos, dtype=np.float64).reshape(-1)
    quality_report_val = _report_quality(y_q_va[entry_keep_va], q_pred_val, ev_q_va[entry_keep_va])
    quality_report_oos = _report_quality(y_q_oo[entry_keep_oo], q_pred_oos, ev_q_oo[entry_keep_oo])

    entry_art = args.out_dir / "entry_state_catboost_gpu.cbm"
    direction_art = args.out_dir / "direction_catboost_gpu.cbm"
    quality_art = args.out_dir / "quality_catboost_gpu.cbm"
    entry_model.save_model(str(entry_art))
    direction_model.save_model(str(direction_art))
    quality_model.save_model(str(quality_art))
    joblib.dump({"feature_cols": entry_cols, "type": "entry_state"}, args.out_dir / "entry_state_meta.joblib")
    joblib.dump({"feature_cols": direction_cols, "type": "direction"}, args.out_dir / "direction_meta.joblib")
    joblib.dump({"feature_cols": quality_cols, "type": "quality"}, args.out_dir / "quality_meta.joblib")

    summary = {
        "model_id": MODEL_ID,
        "devices": args.devices,
        "data_dir": str(args.data_dir),
        "entry_target": str(args.entry_col),
        "entry_label_name": str(args.entry_label_name),
        "direction_target": str(args.direction_col),
        "feature_counts": {
            "entry": len(entry_cols),
            "direction": len(direction_cols),
            "quality": len(quality_cols),
        },
        "train_rows": {
            "entry": int(np.sum(entry_keep_tr)),
            "direction": int(np.sum(dir_keep_tr)),
            "quality": int(np.sum(entry_keep_tr)),
        },
        "fit_seconds": {
            "entry": float(t1 - t0),
            "direction": float(t2 - t1),
            "quality": float(t3 - t2),
            "total": float(t3 - t0),
        },
        "validation": {
            "entry_state": entry_report_val,
            "direction": direction_report_val,
            "quality": quality_report_val,
        },
        "oos": {
            "entry_state": entry_report_oos,
            "direction": direction_report_oos,
            "quality": quality_report_oos,
        },
        "feature_importance": {
            "entry_state": _feature_importance(entry_model, entry_cols),
            "direction": _feature_importance(direction_model, direction_cols),
            "quality": _feature_importance(quality_model, quality_cols),
        },
    }
    (args.out_dir / "alpha5_27_label_contract_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "alpha5_27_contract_done",
        "summary_path": str(args.out_dir / "alpha5_27_label_contract_summary.json"),
        "entry_bal_acc_oos": summary["oos"]["entry_state"]["balanced_accuracy"],
        "direction_bal_acc_oos": summary["oos"]["direction"]["balanced_accuracy"],
        "quality_spearman_oos": summary["oos"]["quality"]["spearman_quality"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
