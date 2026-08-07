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
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_alpha5_27_label_factory_contracts_20260519 import (  # noqa: E402
    _class_weights,
    _feature_importance,
    _fit_direction,
    _fit_entry,
    _fit_quality,
    _num,
    _report_direction,
    _report_entry,
    _report_quality,
)
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


MODEL_ID = "alpha5_governor_v1_train_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable005_20260519"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_governor_v1_contracts_20260519"


def _sample_class_weight(y: np.ndarray) -> np.ndarray:
    cw = _class_weights(np.asarray(y, dtype=np.int64))
    return np.asarray([cw[int(v)] for v in np.asarray(y, dtype=np.int64)], dtype=np.float64)


def _fit_ambiguous_subtype(
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
        iterations=700,
        depth=6,
        learning_rate=0.03,
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
        early_stopping_rounds=80,
        verbose=100,
    )
    return model


def _report_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train alpha5 governor v1 heads on alpha5_29/30 contracts.")
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
    entry_cols = _entry_feature_cols(cols)
    direction_cols = _direction_feature_cols(cols)
    quality_cols = list(dict.fromkeys(entry_cols + direction_cols))

    x_train_entry = _x(train_df, entry_cols)
    x_val_entry = _x(val_df, entry_cols)
    x_oos_entry = _x(oos_df, entry_cols)
    x_train_dir = _x(train_df, direction_cols)
    x_val_dir = _x(val_df, direction_cols)
    x_oos_dir = _x(oos_df, direction_cols)
    x_train_q = _x(train_df, quality_cols)
    x_val_q = _x(val_df, quality_cols)
    x_oos_q = _x(oos_df, quality_cols)

    entry_keep_tr = _num(train_df, "entry_train_keep", 0.0).astype(np.int8) == 1
    entry_keep_va = _num(val_df, "entry_train_keep", 0.0).astype(np.int8) == 1
    entry_keep_oo = _num(oos_df, "entry_train_keep", 0.0).astype(np.int8) == 1
    dir_keep_tr = _num(train_df, "direction_train_keep30", 0.0).astype(np.int8) == 1
    dir_keep_va = _num(val_df, "direction_train_keep30", 0.0).astype(np.int8) == 1
    dir_keep_oo = _num(oos_df, "direction_train_keep30", 0.0).astype(np.int8) == 1
    amb_keep_tr = _num(train_df, "ambiguous_subtype_train_keep", 0.0).astype(np.int8) == 1
    amb_keep_va = _num(val_df, "ambiguous_subtype_train_keep", 0.0).astype(np.int8) == 1
    amb_keep_oo = _num(oos_df, "ambiguous_subtype_train_keep", 0.0).astype(np.int8) == 1

    y_entry_tr = _num(train_df, "entry_state", 1.0).astype(np.int64)
    y_entry_va = _num(val_df, "entry_state", 1.0).astype(np.int64)
    y_entry_oo = _num(oos_df, "entry_state", 1.0).astype(np.int64)
    y_dir_tr = (_num(train_df, "direction_label", 0.0).astype(np.int64)[dir_keep_tr] == 1).astype(np.int64)
    y_dir_va = (_num(val_df, "direction_label", 0.0).astype(np.int64)[dir_keep_va] == 1).astype(np.int64)
    y_dir_oo = (_num(oos_df, "direction_label", 0.0).astype(np.int64)[dir_keep_oo] == 1).astype(np.int64)
    y_q_tr = _num(train_df, "quality_score", 0.0).astype(np.float64)
    y_q_va = _num(val_df, "quality_score", 0.0).astype(np.float64)
    y_q_oo = _num(oos_df, "quality_score", 0.0).astype(np.float64)
    ev_q_va = _num(val_df, "meta_event_return", 0.0).astype(np.float64)
    ev_q_oo = _num(oos_df, "meta_event_return", 0.0).astype(np.float64)
    y_amb_tr = (_num(train_df, "ambiguous_subtype", 0.0).astype(np.int64)[amb_keep_tr] == 2).astype(np.int64)
    y_amb_va = (_num(val_df, "ambiguous_subtype", 0.0).astype(np.int64)[amb_keep_va] == 2).astype(np.int64)
    y_amb_oo = (_num(oos_df, "ambiguous_subtype", 0.0).astype(np.int64)[amb_keep_oo] == 2).astype(np.int64)

    w_entry_tr = _num(train_df, "entry_sample_weight", 1.0)[entry_keep_tr] * _sample_class_weight(y_entry_tr[entry_keep_tr])
    w_dir_tr = _num(train_df, "direction_sample_weight30", 1.0)[dir_keep_tr] * _sample_class_weight(y_dir_tr)
    w_q_tr = np.clip(np.abs(y_q_tr[entry_keep_tr]) + 0.25, 1e-4, None)
    w_amb_tr = _num(train_df, "ambiguous_subtype_sample_weight", 1.0)[amb_keep_tr] * _sample_class_weight(y_amb_tr)

    print(
        json.dumps(
            {
                "stage": "alpha5_governor_v1_fit",
                "entry_rows": int(np.sum(entry_keep_tr)),
                "direction_rows": int(np.sum(dir_keep_tr)),
                "quality_rows": int(np.sum(entry_keep_tr)),
                "ambiguous_subtype_rows": int(np.sum(amb_keep_tr)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

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
    ambiguous_model = _fit_ambiguous_subtype(
        x_train_entry.loc[amb_keep_tr].reset_index(drop=True),
        y_amb_tr,
        w_amb_tr,
        x_val_entry.loc[amb_keep_va].reset_index(drop=True),
        y_amb_va,
        seed=args.seed + 4,
        devices=args.devices,
    )
    t4 = time.perf_counter()

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

    q_pred_val = np.asarray(quality_model.predict(x_val_q.loc[entry_keep_va].reset_index(drop=True)), dtype=np.float64).reshape(-1)
    q_pred_oos = np.asarray(quality_model.predict(x_oos_q.loc[entry_keep_oo].reset_index(drop=True)), dtype=np.float64).reshape(-1)
    quality_report_val = _report_quality(y_q_va[entry_keep_va], q_pred_val, ev_q_va[entry_keep_va])
    quality_report_oos = _report_quality(y_q_oo[entry_keep_oo], q_pred_oos, ev_q_oo[entry_keep_oo])

    amb_val_p = np.asarray(ambiguous_model.predict_proba(x_val_entry.loc[amb_keep_va].reset_index(drop=True)), dtype=np.float64)[:, 1]
    amb_oos_p = np.asarray(ambiguous_model.predict_proba(x_oos_entry.loc[amb_keep_oo].reset_index(drop=True)), dtype=np.float64)[:, 1]
    ambiguous_report_val = _report_binary(y_amb_va, (amb_val_p >= 0.5).astype(np.int64))
    ambiguous_report_oos = _report_binary(y_amb_oo, (amb_oos_p >= 0.5).astype(np.int64))

    entry_art = args.out_dir / "entry_state_catboost_gpu.cbm"
    direction_art = args.out_dir / "direction_catboost_gpu.cbm"
    quality_art = args.out_dir / "quality_catboost_gpu.cbm"
    ambiguous_art = args.out_dir / "ambiguous_subtype_catboost_gpu.cbm"
    entry_model.save_model(str(entry_art))
    direction_model.save_model(str(direction_art))
    quality_model.save_model(str(quality_art))
    ambiguous_model.save_model(str(ambiguous_art))
    joblib.dump({"feature_cols": entry_cols, "type": "entry_state"}, args.out_dir / "entry_state_meta.joblib")
    joblib.dump({"feature_cols": direction_cols, "type": "direction"}, args.out_dir / "direction_meta.joblib")
    joblib.dump({"feature_cols": quality_cols, "type": "quality"}, args.out_dir / "quality_meta.joblib")
    joblib.dump({"feature_cols": entry_cols, "type": "ambiguous_subtype"}, args.out_dir / "ambiguous_subtype_meta.joblib")

    summary = {
        "model_id": MODEL_ID,
        "devices": args.devices,
        "data_dir": str(args.data_dir),
        "feature_counts": {
            "entry": len(entry_cols),
            "direction": len(direction_cols),
            "quality": len(quality_cols),
            "ambiguous_subtype": len(entry_cols),
        },
        "train_rows": {
            "entry": int(np.sum(entry_keep_tr)),
            "direction": int(np.sum(dir_keep_tr)),
            "quality": int(np.sum(entry_keep_tr)),
            "ambiguous_subtype": int(np.sum(amb_keep_tr)),
        },
        "fit_seconds": {
            "entry": float(t1 - t0),
            "direction": float(t2 - t1),
            "quality": float(t3 - t2),
            "ambiguous_subtype": float(t4 - t3),
            "total": float(t4 - t0),
        },
        "validation": {
            "entry_state": entry_report_val,
            "direction": direction_report_val,
            "quality": quality_report_val,
            "ambiguous_subtype": ambiguous_report_val,
        },
        "oos": {
            "entry_state": entry_report_oos,
            "direction": direction_report_oos,
            "quality": quality_report_oos,
            "ambiguous_subtype": ambiguous_report_oos,
        },
        "feature_importance": {
            "entry_state": _feature_importance(entry_model, entry_cols),
            "direction": _feature_importance(direction_model, direction_cols),
            "quality": _feature_importance(quality_model, quality_cols),
            "ambiguous_subtype": _feature_importance(ambiguous_model, entry_cols),
        },
    }
    (args.out_dir / "alpha5_27_label_contract_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": "alpha5_governor_v1_done",
                "summary_path": str(args.out_dir / "alpha5_27_label_contract_summary.json"),
                "entry_bal_acc_oos": summary["oos"]["entry_state"]["balanced_accuracy"],
                "direction_bal_acc_oos": summary["oos"]["direction"]["balanced_accuracy"],
                "ambiguous_bal_acc_oos": summary["oos"]["ambiguous_subtype"]["balanced_accuracy"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
