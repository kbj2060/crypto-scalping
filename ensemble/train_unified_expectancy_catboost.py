from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_ROOT_DIR, _THIS_DIR, os.path.join(_ROOT_DIR, "ensemble")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.supervised.common import make_future_return, median_fill_by_train, time_split_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CSV = "data/rl_training_2025_unified_supdir_cat.csv"
DEFAULT_SAVE = "data/ensemble/supervised/unified_expectancy_catboost.json"

FEATURE_COLS = [
    "ud_sup_long_prob",
    "ud_sup_flat_prob",
    "ud_sup_short_prob",
    "ud_sup_edge",
    "smart_money_flow",
    "taker_acceleration",
    "trade_intensity",
    "garch_vol_z",
    "rogers_satchell_vol",
    "amihud_illiquidity_z",
    "m7_expected_ret",
    "m7_tail_risk",
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
]


def _resolve_meta_paths(save_path: str) -> tuple[str, str]:
    meta_path = save_path
    if meta_path.endswith(".pkl"):
        meta_path = meta_path[:-4] + ".json"
    model_path = meta_path[:-5] + ".pkl" if meta_path.endswith(".json") else meta_path + ".pkl"
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    return model_path, meta_path


def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(args.csv_path)
    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    df = _safe_fill(df)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required features: {missing}")

    lp = pd.to_numeric(df["ud_sup_long_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    sp = pd.to_numeric(df["ud_sup_short_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    fp = pd.to_numeric(df["ud_sup_flat_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    direction = np.where((lp >= sp) & (lp >= fp), 1, np.where((sp > lp) & (sp >= fp), -1, 0)).astype(np.int8)
    fwd = make_future_return(df, horizon=args.horizon)
    y = np.clip(fwd * direction, -args.clip_abs, args.clip_abs)
    valid = np.isfinite(y) & (direction != 0)
    df = df.loc[valid].reset_index(drop=True)
    y = y[valid]

    tr_idx, va_idx, te_idx = time_split_indices(len(df), args.train_ratio, args.val_ratio)
    x = df[FEATURE_COLS].copy()
    x_train = x.iloc[tr_idx].copy()
    x_val = x.iloc[va_idx].copy()
    x_test = x.iloc[te_idx].copy()
    x_train, x_val = median_fill_by_train(x_train, x_val)
    x_train, x_test = median_fill_by_train(x_train, x_test)
    y_train, y_val, y_test = y[tr_idx], y[va_idx], y[te_idx]

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.seed,
        od_type="Iter",
        od_wait=args.od_wait,
        verbose=False,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    val_mae = float(mean_absolute_error(y_val, val_pred))
    test_mae = float(mean_absolute_error(y_test, test_pred))
    val_ic = float(np.corrcoef(y_val, val_pred)[0, 1]) if len(y_val) > 1 else 0.0
    test_ic = float(np.corrcoef(y_test, test_pred)[0, 1]) if len(y_test) > 1 else 0.0

    model_path, meta_path = _resolve_meta_paths(args.save_path)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)
    artifact = {
        "feature_cols": FEATURE_COLS,
        "model_path": os.path.basename(model_path),
        "meta": {
            "algorithm": "unified_expectancy_catboost",
            "csv_path": args.csv_path,
            "horizon": args.horizon,
            "clip_abs": args.clip_abs,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
            "val_mae": val_mae,
            "test_mae": test_mae,
            "val_ic": val_ic,
            "test_ic": test_ic,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    logger.info("saved model: %s", model_path)
    logger.info("saved meta: %s", meta_path)
    logger.info("val_rmse=%.5f test_rmse=%.5f val_ic=%.4f test_ic=%.4f", val_rmse, test_rmse, val_ic, test_ic)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train unified expectancy regressor")
    p.add_argument("--csv-path", default=DEFAULT_CSV)
    p.add_argument("--save-path", default=DEFAULT_SAVE)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--clip-abs", type=float, default=0.05)
    p.add_argument("--iterations", type=int, default=700)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--l2-leaf-reg", type=float, default=8.0)
    p.add_argument("--od-wait", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
