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


MODEL_ID = "alpha5_router_v5_train_20260520"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_29_hier_label_factory_20260519"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_router_v5_train_20260520"
DEFAULT_FEATURE_LIST_JSON = ROOT / "tmp/causal_regen_20260516/alpha5_router5_full_candidate_search_20260521/rank_pruned_stable_top48_feature_list.json"

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


def _resolve_feature_cols(feature_list_json: Path | None) -> list[str]:
    if feature_list_json is None:
        return list(ROUTER_FEATURE_COLS)
    payload = json.loads(Path(feature_list_json).read_text(encoding="utf-8"))
    cols = payload.get("features", payload)
    if not isinstance(cols, list) or not cols:
        raise ValueError(f"invalid feature list json: {feature_list_json}")
    out = []
    seen = set()
    for col in cols:
        name = str(col).strip()
        if not name or name in seen:
            continue
        out.append(name)
        seen.add(name)
    if not out:
        raise ValueError(f"empty feature list json: {feature_list_json}")
    return out


def _router3_label(frame: pd.DataFrame) -> np.ndarray:
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


def _router3_weight(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
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
    w *= np.where((entry_state == 1) & (subtype == 1), 0.85, 1.0)
    w *= np.where((entry_state == 1) & (subtype == 2), 0.55, 1.0)
    w *= np.where(regime == "whipsaw", 1.15, 1.0)
    trade_boost = 1.0 + np.clip(np.abs(quality), 0.0, 1.0)
    w *= np.where(y > 0, trade_boost, 1.0)
    classes, counts = np.unique(y, return_counts=True)
    total = float(len(y))
    cw = {int(cls): float(total / (len(classes) * max(float(cnt), 1.0))) for cls, cnt in zip(classes, counts)}
    w *= np.asarray([cw[int(v)] for v in y], dtype=np.float64)
    return np.clip(w, 1e-4, None)


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


def _router4_weight(frame: pd.DataFrame, y4: np.ndarray) -> np.ndarray:
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


def _collapse4(y4: np.ndarray) -> np.ndarray:
    y3 = np.zeros(len(y4), dtype=np.int64)
    y3[y4 == 2] = 1
    y3[y4 == 3] = 2
    return y3


def _prepare_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    work = frame.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = np.nan
    X = work[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = _num(work, "split_keep", 0.0).astype(np.int8) == 1
    X = X.loc[keep].reset_index(drop=True)
    W = work.loc[keep].reset_index(drop=True)
    return X.join(W.drop(columns=[c for c in feature_cols if c in W.columns], errors="ignore"))


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
    p = argparse.ArgumentParser(description="Train router5 in a single script (router3 + router4 + weighted ensemble).")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--devices", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-router3", type=float, default=0.8)
    p.add_argument("--weight-router4", type=float, default=0.2)
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--feature-list-json", type=Path, default=DEFAULT_FEATURE_LIST_JSON)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)
    feature_cols = _resolve_feature_cols(args.feature_list_json)

    train_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_29_hier_label_factory_oos.parquet")

    train_work = _prepare_frame(train_df, feature_cols)
    val_work = _prepare_frame(val_df, feature_cols)
    oos_work = _prepare_frame(oos_df, feature_cols)

    x_train = train_work[feature_cols].copy()
    x_val = val_work[feature_cols].copy()
    x_oos = oos_work[feature_cols].copy()

    y3_train = _router3_label(train_df)[_num(train_df, "split_keep", 0.0).astype(np.int8) == 1]
    y3_val = _router3_label(val_df)[_num(val_df, "split_keep", 0.0).astype(np.int8) == 1]
    y3_oos = _router3_label(oos_df)[_num(oos_df, "split_keep", 0.0).astype(np.int8) == 1]
    w3_train = _router3_weight(train_work, y3_train)

    y4_train = _router4_label(train_df)[_num(train_df, "split_keep", 0.0).astype(np.int8) == 1]
    y4_val = _router4_label(val_df)[_num(val_df, "split_keep", 0.0).astype(np.int8) == 1]
    y4_oos = _router4_label(oos_df)[_num(oos_df, "split_keep", 0.0).astype(np.int8) == 1]
    w4_train = _router4_weight(train_work, y4_train)

    model3 = _fit_router(x_train, y3_train, w3_train, x_val, y3_val, seed=args.seed, devices=args.devices)
    model4 = _fit_router(x_train, y4_train, w4_train, x_val, y4_val, seed=args.seed, devices=args.devices)

    proba3_val = np.asarray(model3.predict_proba(x_val), dtype=np.float32)
    proba3_oos = np.asarray(model3.predict_proba(x_oos), dtype=np.float32)

    proba4_val_raw = np.asarray(model4.predict_proba(x_val), dtype=np.float32)
    proba4_oos_raw = np.asarray(model4.predict_proba(x_oos), dtype=np.float32)

    classes3 = [int(c) for c in getattr(model3, "classes_", [0, 1, 2])]
    classes4 = [int(c) for c in getattr(model4, "classes_", [0, 1, 2, 3])]
    idx3 = {c: i for i, c in enumerate(classes3)}
    idx4 = {c: i for i, c in enumerate(classes4)}

    p3_val = np.stack([proba3_val[:, idx3[0]], proba3_val[:, idx3[1]], proba3_val[:, idx3[2]]], axis=1)
    p3_oos = np.stack([proba3_oos[:, idx3[0]], proba3_oos[:, idx3[1]], proba3_oos[:, idx3[2]]], axis=1)
    p4_val = np.stack(
        [
            proba4_val_raw[:, idx4[0]] + proba4_val_raw[:, idx4[1]],
            proba4_val_raw[:, idx4[2]],
            proba4_val_raw[:, idx4[3]],
        ],
        axis=1,
    )
    p4_oos = np.stack(
        [
            proba4_oos_raw[:, idx4[0]] + proba4_oos_raw[:, idx4[1]],
            proba4_oos_raw[:, idx4[2]],
            proba4_oos_raw[:, idx4[3]],
        ],
        axis=1,
    )

    w3 = float(args.weight_router3)
    w4 = float(args.weight_router4)
    ens_val = (w3 * p3_val + w4 * p4_val) / max(w3 + w4, 1e-9)
    ens_oos = (w3 * p3_oos + w4 * p4_oos) / max(w3 + w4, 1e-9)

    pred3_val = p3_val.argmax(axis=1).astype(np.int64)
    pred3_oos = p3_oos.argmax(axis=1).astype(np.int64)
    pred4_val = p4_val.argmax(axis=1).astype(np.int64)
    pred4_oos = p4_oos.argmax(axis=1).astype(np.int64)
    pred5_val = ens_val.argmax(axis=1).astype(np.int64)
    pred5_oos = ens_oos.argmax(axis=1).astype(np.int64)

    model3_path = args.out_dir / "router3_catboost_gpu.cbm"
    model4_path = args.out_dir / "router4_catboost_gpu.cbm"
    meta3_path = args.out_dir / "router3_meta.joblib"
    meta4_path = args.out_dir / "router4_meta.joblib"
    meta5_path = args.out_dir / "router_ensemble_meta.joblib"
    summary_path = args.out_dir / "router5_summary.json"

    model3.save_model(str(model3_path))
    model4.save_model(str(model4_path))
    joblib.dump(
        {
            "feature_cols": list(feature_cols),
            "classes": [0, 1, 2],
            "type": "router3",
            "label_meaning": {"0": "none", "1": "long", "2": "short"},
        },
        meta3_path,
    )
    joblib.dump(
        {
            "feature_cols": list(feature_cols),
            "type": "router4_collapse",
            "classes": [0, 1, 2, 3],
            "label_meaning": {"0": "clean_none", "1": "ambiguous_none", "2": "long", "3": "short"},
        },
        meta4_path,
    )
    joblib.dump(
        {
            "feature_cols": list(feature_cols),
            "type": "router_ensemble",
            "classes": [0, 1, 2],
            "label_meaning": {"0": "none", "1": "long", "2": "short"},
            "components": [
                {"model_path": str(model3_path), "meta_path": str(meta3_path), "weight": w3},
                {"model_path": str(model4_path), "meta_path": str(meta4_path), "weight": w4},
            ],
        },
        meta5_path,
    )

    payload = {
        "model_id": MODEL_ID,
        "meta_path": str(meta5_path),
        "feature_cols": list(feature_cols),
        "feature_list_json": str(args.feature_list_json) if args.feature_list_json is not None else None,
        "weights": {"router3": w3, "router4": w4},
        "val": _report(y3_val, pred5_val),
        "oos": _report(y3_oos, pred5_oos),
        "router3": {"val": _report(y3_val, pred3_val), "oos": _report(y3_oos, pred3_oos)},
        "router4": {"val": _report(y3_val, pred4_val), "oos": _report(y3_oos, pred4_oos)},
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"stage": "router5_fit_done", "summary_path": str(summary_path), "val_bal_acc": payload["val"]["balanced_accuracy"], "oos_bal_acc": payload["oos"]["balanced_accuracy"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
