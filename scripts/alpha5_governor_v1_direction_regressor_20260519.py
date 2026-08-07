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
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha5_governor_v1_infer_20260519 import load_governor, predict_heads  # noqa: E402
from scripts.train_eval_alpha5_23_hgb_direction_refined_20260519 import (  # noqa: E402
    _direction_feature_cols,
    _entry_feature_cols,
    _eval,
    _feature_cols,
    _x,
)
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


MODEL_ID = "alpha5_governor_v1_direction_regressor_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable005_20260519"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_GOVERNOR_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_governor_v1_contracts_20260519"
DEFAULT_PRIOR_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_governor_v1_candidate_prior_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_governor_v1_direction_regressor_20260519"


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(frame.get(col, default), errors="coerce").fillna(default).to_numpy(np.float64)


def _signed_target(frame: pd.DataFrame, scale: float) -> tuple[np.ndarray, np.ndarray]:
    raw_edge = _num(frame, "meta_long_score", 0.0) - _num(frame, "meta_short_score", 0.0)
    target = np.tanh(raw_edge / float(scale))

    entry_state = _num(frame, "entry_state", 1.0).astype(np.int8)
    ambiguous_subtype = _num(frame, "ambiguous_subtype", 0.0).astype(np.int8)
    direction_keep = _num(frame, "direction_train_keep30", 0.0).astype(np.int8) == 1

    # Clean wait and structural ambiguous collapse to 0.
    target = np.where(entry_state == 0, 0.0, target)
    target = np.where(ambiguous_subtype == 1, 0.0, target)
    # Trade-like ambiguous keeps weak signed value only.
    target = np.where(ambiguous_subtype == 2, 0.35 * target, target)
    # Non-learnable direction trades get softened but not zeroed.
    target = np.where((entry_state == 2) & (~direction_keep), 0.50 * target, target)

    keep = (
        (_num(frame, "split_keep", 0.0).astype(np.int8) == 1)
        & (frame.get("regime4_state", "unknown").astype(str).to_numpy() != "whipsaw")
        & ((entry_state == 2) | (entry_state == 0) | (ambiguous_subtype == 2))
    )
    return target.astype(np.float32), keep.astype(bool)


def _sample_weight(frame: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    entry_state = _num(frame, "entry_state", 1.0).astype(np.int8)
    ambiguous_subtype = _num(frame, "ambiguous_subtype", 0.0).astype(np.int8)
    direction_keep = _num(frame, "direction_train_keep30", 0.0).astype(np.int8) == 1
    base = np.clip(np.abs(target).astype(np.float64) + 0.15, 1e-4, None)
    base *= np.clip(_num(frame, "sample_uniqueness_weight", 1.0), 0.05, 1.0)
    base *= np.where(entry_state == 2, 3.5, 1.0)
    base *= np.where(direction_keep, 2.5, 1.0)
    base *= np.where(ambiguous_subtype == 2, 1.25, 1.0)
    base *= np.where(ambiguous_subtype == 1, 0.0, 1.0)
    base *= np.where(entry_state == 0, 0.30, 1.0)
    return base.astype(np.float32)


def _fit_regressor(
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
        iterations=1200,
        depth=7,
        learning_rate=0.03,
        l2_leaf_reg=4.0,
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


def _report_regression(y_true: np.ndarray, pred: np.ndarray, signed_ret: np.ndarray) -> dict[str, Any]:
    spearman_target = float(pd.Series(pred).corr(pd.Series(y_true), method="spearman"))
    spearman_signed_ret = float(pd.Series(pred).corr(pd.Series(signed_ret), method="spearman"))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "spearman_target": spearman_target if np.isfinite(spearman_target) else 0.0,
        "spearman_signed_return": spearman_signed_ret if np.isfinite(spearman_signed_ret) else 0.0,
    }


def _load_candidate_prior(prior_dir: Path) -> tuple[CatBoostClassifier, list[str], float]:
    model = CatBoostClassifier()
    model.load_model(str(prior_dir / "candidate_prior_catboost_gpu.cbm"))
    meta = joblib.load(prior_dir / "candidate_prior_meta.joblib")
    summary = json.loads((prior_dir / "summary.json").read_text(encoding="utf-8"))
    return model, meta["feature_cols"], float(summary["selected_threshold"])


def main() -> None:
    p = argparse.ArgumentParser(description="Train direction regressor and evaluate candidate_prior + quality + signed direction gate.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--governor-dir", type=Path, default=DEFAULT_GOVERNOR_DIR)
    p.add_argument("--prior-dir", type=Path, default=DEFAULT_PRIOR_DIR)
    p.add_argument("--devices", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-scale", type=float, default=3.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--max-hold-bars", type=int, default=96)
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
    y_train, keep_train = _signed_target(train_df, args.target_scale)
    y_val, keep_val = _signed_target(val_df, args.target_scale)
    y_oos, keep_oos = _signed_target(oos_df, args.target_scale)
    w_train = _sample_weight(train_df, y_train)[keep_train]

    signed_ret_val = _num(val_df, "meta_raw_terminal_return", 0.0)[keep_val]
    signed_ret_oos = _num(oos_df, "meta_raw_terminal_return", 0.0)[keep_oos]

    print(
        json.dumps(
            {
                "stage": "direction_regressor_fit",
                "train_rows": int(np.sum(keep_train)),
                "val_rows": int(np.sum(keep_val)),
                "oos_rows": int(np.sum(keep_oos)),
                "feature_count": len(feature_cols),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    t0 = time.perf_counter()
    model = _fit_regressor(
        x_train.loc[keep_train].reset_index(drop=True),
        y_train[keep_train],
        w_train,
        x_val.loc[keep_val].reset_index(drop=True),
        y_val[keep_val],
        seed=args.seed,
        devices=args.devices,
    )
    fit_seconds = float(time.perf_counter() - t0)

    pred_val = np.asarray(model.predict(x_val.loc[keep_val].reset_index(drop=True)), dtype=np.float64).reshape(-1)
    pred_oos = np.asarray(model.predict(x_oos.loc[keep_oos].reset_index(drop=True)), dtype=np.float64).reshape(-1)
    reg_report_val = _report_regression(y_val[keep_val], pred_val, signed_ret_val)
    reg_report_oos = _report_regression(y_oos[keep_oos], pred_oos, signed_ret_oos)

    model_path = args.out_dir / "direction_regressor_catboost_gpu.cbm"
    meta_path = args.out_dir / "direction_regressor_meta.joblib"
    model.save_model(str(model_path))
    joblib.dump({"feature_cols": feature_cols, "target_scale": float(args.target_scale)}, meta_path)

    governor = load_governor(args.governor_dir)
    prior_model, prior_cols, prior_threshold_default = _load_candidate_prior(args.prior_dir)
    head_val = predict_heads(governor, val_df)
    head_oos = predict_heads(governor, oos_df)
    q_val = head_val["quality_pred"].to_numpy(np.float64)
    q_oos = head_oos["quality_pred"].to_numpy(np.float64)
    prior_p_val = np.asarray(prior_model.predict_proba(_x(val_df, prior_cols)), dtype=np.float64)[:, 1]
    prior_p_oos = np.asarray(prior_model.predict_proba(_x(oos_df, prior_cols)), dtype=np.float64)[:, 1]
    dir_score_val = np.asarray(model.predict(_x(val_df, feature_cols)), dtype=np.float64).reshape(-1)
    dir_score_oos = np.asarray(model.predict(_x(oos_df, feature_cols)), dtype=np.float64).reshape(-1)
    labels_val = _num(val_df, "label_action", 0.0).astype(np.int64)
    labels_oos = _num(oos_df, "label_action", 0.0).astype(np.int64)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for prior_threshold, dir_abs_min, quality_min in product(
        [max(0.45, prior_threshold_default - 0.10), prior_threshold_default, min(0.85, prior_threshold_default + 0.10)],
        [0.10, 0.15, 0.20, 0.25, 0.30],
        [-0.05, 0.00, 0.05],
    ):
        actions_val = np.where(dir_score_val > 0.0, 1, 2).astype(np.int64)
        actions_val = np.where(prior_p_val >= float(prior_threshold), actions_val, 0)
        actions_val = np.where(np.abs(dir_score_val) >= float(dir_abs_min), actions_val, 0)
        actions_val = np.where(q_val >= float(quality_min), actions_val, 0)
        actions_val = np.where(val_df.get("regime4_state", "unknown").astype(str).to_numpy() == "whipsaw", 0, actions_val)
        val_eval = _eval(val_df, actions_val, labels_val, fee=args.fee, slip=args.slip, exposure=args.exposure, max_hold=args.max_hold_bars)
        trades = int(val_eval["backtest"]["cost1"]["trades"])
        selection_score = float(val_eval["score"]) + min(trades, 140) * 0.10 - max(0, trades - 220) * 0.10
        row = {
            "prior_threshold": float(prior_threshold),
            "dir_abs_min": float(dir_abs_min),
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

    actions_oos = np.where(dir_score_oos > 0.0, 1, 2).astype(np.int64)
    actions_oos = np.where(prior_p_oos >= float(best["prior_threshold"]), actions_oos, 0)
    actions_oos = np.where(np.abs(dir_score_oos) >= float(best["dir_abs_min"]), actions_oos, 0)
    actions_oos = np.where(q_oos >= float(best["quality_min"]), actions_oos, 0)
    actions_oos = np.where(oos_df.get("regime4_state", "unknown").astype(str).to_numpy() == "whipsaw", 0, actions_oos)
    oos_eval = _eval(oos_df, actions_oos, labels_oos, fee=args.fee, slip=args.slip, exposure=args.exposure, max_hold=args.max_hold_bars)

    summary = {
        "model_id": MODEL_ID,
        "fit_seconds": fit_seconds,
        "regression_validation": reg_report_val,
        "regression_oos": reg_report_oos,
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
    print(json.dumps({"stage": "direction_regressor_done", "summary_path": str(args.out_dir / "summary.json"), "oos_pnl": float(oos_eval["backtest"]["cost1"]["pnl"]), "oos_trades": int(oos_eval["backtest"]["cost1"]["trades"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
