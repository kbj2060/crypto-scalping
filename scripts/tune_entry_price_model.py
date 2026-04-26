#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.supervised.common import (
    DEFAULT_DATA_PATH,
    DEFAULT_RL_DATA_PATH,
    load_feature_frame,
    median_fill_by_train,
    select_feature_columns,
    time_split_indices,
)
from ensemble.supervised.train_entry_price_model import (
    SAVE_PATH,
    _add_trend_structure_features,
    _future_extrema_offsets,
    _require_lightgbm,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-search tuner for entry price LightGBM models")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--rl-path", default=DEFAULT_RL_DATA_PATH)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--clip-pct", type=float, default=0.02)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--n-trials", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="data/ensemble/supervised/tuning_entry_price")
    p.add_argument("--save-best-path", default="data/ensemble/supervised/entry_price_model_tuned.json")
    p.add_argument("--separate-sides", action="store_true")
    return p.parse_args()


def _sample_space(rng: random.Random) -> dict:
    return {
        "max_features": rng.choice([72, 84, 96, 108, 120]),
        "n_estimators": rng.choice([250, 350, 450, 600, 800]),
        "learning_rate": rng.choice([0.015, 0.02, 0.025, 0.03, 0.04, 0.05]),
        "num_leaves": rng.choice([31, 47, 63, 79, 95, 127]),
        "subsample": rng.choice([0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": rng.choice([0.7, 0.8, 0.9, 1.0]),
        "min_child_samples": rng.choice([20, 30, 40, 60, 80, 100]),
        "reg_alpha": rng.choice([0.0, 0.01, 0.03, 0.1, 0.2, 0.5]),
        "reg_lambda": rng.choice([0.5, 1.0, 2.0, 3.0, 5.0]),
        "objective": rng.choice(["quantile", "l1"]),
    }


def _fit_and_score(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_long: np.ndarray,
    y_short: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    params: dict,
) -> dict:
    LGBMRegressor = _require_lightgbm()
    model_params = dict(
        objective=params["objective"],
        alpha=0.5,
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_samples=params["min_child_samples"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    long_model = LGBMRegressor(**model_params)
    short_model = LGBMRegressor(**model_params)
    long_model.fit(x_train, y_long[tr_idx])
    short_model.fit(x_train, y_short[tr_idx])

    val_long_pred = np.asarray(long_model.predict(x_val), dtype=np.float64)
    val_short_pred = np.asarray(short_model.predict(x_val), dtype=np.float64)
    test_long_pred = np.asarray(long_model.predict(x_test), dtype=np.float64)
    test_short_pred = np.asarray(short_model.predict(x_test), dtype=np.float64)

    val_long_mae = float(np.mean(np.abs(val_long_pred - y_long[va_idx])))
    val_short_mae = float(np.mean(np.abs(val_short_pred - y_short[va_idx])))
    test_long_mae = float(np.mean(np.abs(test_long_pred - y_long[te_idx])))
    test_short_mae = float(np.mean(np.abs(test_short_pred - y_short[te_idx])))
    score = val_long_mae + val_short_mae
    return {
        "score": score,
        "val_long_mae": val_long_mae,
        "val_short_mae": val_short_mae,
        "test_long_mae": test_long_mae,
        "test_short_mae": test_short_mae,
        "long_model": long_model,
        "short_model": short_model,
    }


def _fit_side_and_score(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    params: dict,
) -> dict:
    LGBMRegressor = _require_lightgbm()
    model_params = dict(
        objective=params["objective"],
        alpha=0.5,
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_samples=params["min_child_samples"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model = LGBMRegressor(**model_params)
    model.fit(x_train, y[tr_idx])
    val_pred = np.asarray(model.predict(x_val), dtype=np.float64)
    test_pred = np.asarray(model.predict(x_test), dtype=np.float64)
    val_mae = float(np.mean(np.abs(val_pred - y[va_idx])))
    test_mae = float(np.mean(np.abs(test_pred - y[te_idx])))
    return {"score": val_mae, "val_mae": val_mae, "test_mae": test_mae, "model": model}


def _search_side(
    side_name: str,
    y: np.ndarray,
    feature_cols: list[str],
    df: pd.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    n_trials: int,
    rng: random.Random,
) -> tuple[dict, list[dict]]:
    best: dict | None = None
    trials: list[dict] = []
    for trial_idx in tqdm(range(1, n_trials + 1), desc=f"entry-price-{side_name}"):
        trial_params = _sample_space(rng)
        trial_features = feature_cols[: trial_params["max_features"]]
        x_all = df[trial_features].replace([np.inf, -np.inf], np.nan)
        x_train = x_all.iloc[tr_idx].copy()
        x_val = x_all.iloc[va_idx].copy()
        x_test = x_all.iloc[te_idx].copy()
        x_train, x_val = median_fill_by_train(x_train, x_val)
        x_train, x_test = median_fill_by_train(x_train, x_test)
        x_val, x_test = median_fill_by_train(x_val, x_test)
        result = _fit_side_and_score(x_train, x_val, x_test, y, tr_idx, va_idx, te_idx, trial_params)
        row = {
            "trial": trial_idx,
            "side": side_name,
            **trial_params,
            "score": result["score"],
            "val_mae": result["val_mae"],
            "test_mae": result["test_mae"],
        }
        trials.append(row)
        if best is None or row["score"] < best["score"]:
            best = deepcopy(row)
            best["feature_cols"] = trial_features
            best["model"] = result["model"]
    if best is None:
        raise RuntimeError(f"No trials completed for {side_name}")
    return best, trials


def main() -> int:
    args = parse_args()
    output_dir = os.path.normpath(args.output_dir)
    save_meta_path = os.path.normpath(args.save_best_path)
    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(args.seed)
    df = load_feature_frame(args.data_path, args.rl_path)
    df = _add_trend_structure_features(df)
    long_target, short_target = _future_extrema_offsets(df, horizon=args.horizon, clip_pct=args.clip_pct)
    valid = np.isfinite(long_target) & np.isfinite(short_target)
    df = df.loc[valid].reset_index(drop=True)
    y_long = long_target[valid]
    y_short = short_target[valid]

    feature_cols = select_feature_columns(df)
    tr_idx, va_idx, te_idx = time_split_indices(len(df), args.train_ratio, args.val_ratio)

    if args.separate_sides:
        best_long, long_trials = _search_side("long", y_long, feature_cols, df, tr_idx, va_idx, te_idx, args.n_trials, rng)
        best_short, short_trials = _search_side("short", y_short, feature_cols, df, tr_idx, va_idx, te_idx, args.n_trials, rng)
        trials = long_trials + short_trials
        best = {
            "trial": "separate-sides",
            "score": best_long["score"] + best_short["score"],
            "val_long_mae": best_long["val_mae"],
            "val_short_mae": best_short["val_mae"],
            "test_long_mae": best_long["test_mae"],
            "test_short_mae": best_short["test_mae"],
            "long_params": {k: v for k, v in best_long.items() if k not in {"feature_cols", "model"}},
            "short_params": {k: v for k, v in best_short.items() if k not in {"feature_cols", "model"}},
            "long_feature_cols": best_long["feature_cols"],
            "short_feature_cols": best_short["feature_cols"],
            "long_model": best_long["model"],
            "short_model": best_short["model"],
        }
    else:
        trials = []
        best: dict | None = None
        for trial_idx in tqdm(range(1, args.n_trials + 1), desc="entry-price-tuning"):
            trial_params = _sample_space(rng)
            trial_features = feature_cols[: trial_params["max_features"]]
            x_all = df[trial_features].replace([np.inf, -np.inf], np.nan)
            x_train = x_all.iloc[tr_idx].copy()
            x_val = x_all.iloc[va_idx].copy()
            x_test = x_all.iloc[te_idx].copy()
            x_train, x_val = median_fill_by_train(x_train, x_val)
            x_train, x_test = median_fill_by_train(x_train, x_test)
            x_val, x_test = median_fill_by_train(x_val, x_test)

            result = _fit_and_score(x_train, x_val, x_test, y_long, y_short, tr_idx, va_idx, te_idx, trial_params)
            row = {
                "trial": trial_idx,
                **trial_params,
                "score": result["score"],
                "val_long_mae": result["val_long_mae"],
                "val_short_mae": result["val_short_mae"],
                "test_long_mae": result["test_long_mae"],
                "test_short_mae": result["test_short_mae"],
            }
            trials.append(row)
            if best is None or row["score"] < best["score"]:
                best = deepcopy(row)
                best["feature_cols"] = trial_features
                best["long_model"] = result["long_model"]
                best["short_model"] = result["short_model"]

        if best is None:
            raise RuntimeError("No trials completed")

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    trial_report_path = os.path.normpath(os.path.join(output_dir, f"entry_price_tuning_trials_{stamp}.json"))
    os.makedirs(os.path.dirname(trial_report_path), exist_ok=True)
    with open(trial_report_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.utcnow().isoformat(), "trials": trials}, f, ensure_ascii=False, indent=2)

    model_path = os.path.splitext(save_meta_path)[0] + ".pkl"
    payload = {
        "long_model": best.pop("long_model"),
        "short_model": best.pop("short_model"),
        "feature_cols": best.get("feature_cols", best.get("long_feature_cols", [])),
        "long_feature_cols": best.get("long_feature_cols", best.get("feature_cols", [])),
        "short_feature_cols": best.get("short_feature_cols", best.get("feature_cols", [])),
        "horizon": int(args.horizon),
        "long_clip": float(args.clip_pct),
        "short_clip": float(args.clip_pct),
        "metrics": {
            "long_mae": best["test_long_mae"],
            "short_mae": best["test_short_mae"],
        },
        "val_metrics": {
            "long_mae": best["val_long_mae"],
            "short_mae": best["val_short_mae"],
        },
    }
    os.makedirs(os.path.dirname(save_meta_path), exist_ok=True)
    import pickle

    with open(model_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "model_path": os.path.basename(model_path),
        "feature_cols": payload["feature_cols"],
        "long_feature_cols": payload["long_feature_cols"],
        "short_feature_cols": payload["short_feature_cols"],
        "horizon": int(args.horizon),
        "long_clip": float(args.clip_pct),
        "short_clip": float(args.clip_pct),
        "metrics": payload["metrics"],
        "val_metrics": payload["val_metrics"],
        "best_trial": {k: v for k, v in best.items() if k != "feature_cols"},
        "trial_report_path": trial_report_path,
    }
    with open(save_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta["best_trial"], ensure_ascii=False, indent=2))
    print(f"saved_best={save_meta_path}")
    print(f"saved_trials={trial_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
