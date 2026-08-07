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

from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


MODEL_ID = "alpha5_router_v2_train_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_29_hier_label_factory_20260519"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_router_v2_contracts_20260519"

ROUTER_FEATURE_COLS = [
    "log_return",
    "funding_abs",
    "funding_pressure",
    "funding_price_divergence",
    "crowding_pressure",
    "smart_money_flow",
    "net_taker_ratio",
    "taker_acceleration",
    "ofi_acceleration",
    "whale_conviction",
    "m7_expected_ret",
    "m7_composite_score",
    "m7_confidence",
    "ai_dir_edge",
    "ai_dir_p_up",
    "ai_dir_p_down",
    "ai_flow_pressure",
    "ai_flow_exhaustion",
    "ai_flow_flip_prob",
    "ai_flow_slope",
    "dlinear_smf_ema",
    "dlinear_smf_slope",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "rsi",
    "big_trade_ratio",
    "whale_retail_ratio",
    "breakout_strength",
    "execution_quality",
    "clean_regime4_2024_unsup_v1_bear_prob",
    "clean_regime4_2024_unsup_v1_bull_prob",
    "clean_regime4_2024_unsup_v1_factor_flow",
    "clean_regime4_2024_unsup_v1_factor_trend",
    "clean_regime4_2024_unsup_v1_trend_bias",
    "clean_regime4_2024_unsup_v1_trend_prob",
    "clean_regime4_2024_unsup_v1_directional_bias",
    "clean_regime4_2024_unsup_v1_margin",
    "clean_regime4_2024_unsup_v1_whipsaw_prob",
]


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


def _router_weight(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    entry_state = _num(frame, "entry_state", 1.0).astype(np.int8)
    subtype = _num(frame, "ambiguous_subtype", 0.0).astype(np.int8)
    label_conf = np.clip(_num(frame, "label_confidence", 0.0), 0.0, 1.0)
    uniq = np.clip(_num(frame, "sample_uniqueness_weight", 0.0), 0.0, 1.0)
    quality = _num(frame, "quality_score", 0.0)
    regime = frame.get("regime4_state", pd.Series(["unknown"] * len(frame))).astype(str).to_numpy()

    w = np.ones(len(frame), dtype=np.float64)
    w *= 0.75 + 0.25 * label_conf
    w *= 0.80 + 0.20 * uniq
    w *= np.where(entry_state == 0, 1.00, 1.0)
    w *= np.where((entry_state == 1) & (subtype == 1), 0.85, 1.0)  # structural ambiguous -> abstain
    w *= np.where((entry_state == 1) & (subtype == 2), 0.55, 1.0)  # trade-like ambiguous -> noisy abstain
    w *= np.where(regime == "whipsaw", 1.15, 1.0)
    trade_boost = 1.0 + np.clip(np.abs(quality), 0.0, 1.0)
    w *= np.where(y > 0, trade_boost, 1.0)

    classes, counts = np.unique(y, return_counts=True)
    total = float(len(y))
    cw = {int(cls): float(total / (len(classes) * max(float(cnt), 1.0))) for cls, cnt in zip(classes, counts)}
    w *= np.asarray([cw[int(v)] for v in y], dtype=np.float64)
    return np.clip(w, 1e-4, None)


def _prepare_xy(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = frame.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = np.nan
    X = work[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = _router_label(work)
    w = _router_weight(work, y)
    keep = _num(work, "split_keep", 0.0).astype(np.int8) == 1
    return X.loc[keep].reset_index(drop=True), y[keep], w[keep]


def _fit_router(
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
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=(x_val, y_val),
        early_stopping_rounds=120,
        verbose=100,
    )
    return model


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
    p = argparse.ArgumentParser(description="Train causal alpha5 router v2 with explicit none/long/short classes.")
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

    train_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_oos.parquet")

    x_train, y_train, w_train = _prepare_xy(train_df, ROUTER_FEATURE_COLS)
    x_val, y_val, _ = _prepare_xy(val_df, ROUTER_FEATURE_COLS)
    x_oos, y_oos, _ = _prepare_xy(oos_df, ROUTER_FEATURE_COLS)

    model = _fit_router(x_train, y_train, w_train, x_val, y_val, seed=args.seed, devices=args.devices)
    pred_val = np.asarray(model.predict(x_val)).reshape(-1).astype(np.int64)
    pred_oos = np.asarray(model.predict(x_oos)).reshape(-1).astype(np.int64)

    model_path = args.out_dir / "router3_catboost_gpu.cbm"
    meta_path = args.out_dir / "router3_meta.joblib"
    report_path = args.out_dir / "router3_summary.json"
    model.save_model(str(model_path))
    joblib.dump(
        {
            "feature_cols": list(ROUTER_FEATURE_COLS),
            "classes": [0, 1, 2],
            "type": "router3",
            "label_meaning": {"0": "none", "1": "long", "2": "short"},
        },
        meta_path,
    )
    payload = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "feature_cols": list(ROUTER_FEATURE_COLS),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "oos_rows": int(len(y_oos)),
        "val": _report(y_val, pred_val),
        "oos": _report(y_oos, pred_oos),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"stage": "router3_fit_done", "report_path": str(report_path), "val_bal_acc": payload["val"]["balanced_accuracy"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
