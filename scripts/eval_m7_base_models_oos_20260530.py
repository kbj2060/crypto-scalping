from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.supervised.common import (  # noqa: E402
    is_forbidden_feature,
    load_feature_frame,
    make_future_return,
    make_triple_barrier_targets,
)
from ensemble.supervised.train_entry_price_model import (  # noqa: E402
    EntryPriceBrain,
    _future_extrema_offsets,
)
from ensemble.supervised.train_multitarget_lgbm import _build_quality_and_hold_targets  # noqa: E402
from ensemble.supervised.train_quantile_forest import _predict_quantiles, _score_predictions  # noqa: E402
from ensemble.supervised.train_trend_xgb import XGBTrendBrain  # noqa: E402
from ensemble.seven_model_ensemble import _add_trend_structure_features, _to_numeric_frame  # noqa: E402


def _read_meta(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _model_path(meta_path: str, meta: dict) -> str:
    ref = meta.get("model_path", "")
    return ref if os.path.isabs(ref) else os.path.join(os.path.dirname(meta_path), ref)


def _check_features(name: str, cols: list[str]) -> dict:
    bad = [c for c in cols if is_forbidden_feature(c)]
    return {
        f"{name}_feature_count": len(cols),
        f"{name}_forbidden_feature_count": len(bad),
        f"{name}_forbidden_features": bad[:20],
    }


def eval_trend(meta_path: str, df: pd.DataFrame) -> dict:
    brain = XGBTrendBrain.load(meta_path)
    y = make_triple_barrier_targets(df, atr_mult=0.8, max_hold=12, atr_window=14)
    valid = (y == 0) | (y == 2)
    x_df = _add_trend_structure_features(df)
    x = _to_numeric_frame(x_df, brain.feature_cols).iloc[valid].copy()
    yv = (y[valid] == 2).astype(np.int64)
    pred = brain.model.predict(x)
    return {
        "trend_bacc": float(balanced_accuracy_score(yv, pred)),
        "trend_f1_weighted": float(f1_score(yv, pred, average="weighted")),
        **_check_features("trend", list(brain.feature_cols)),
    }


def eval_entry(meta_path: str, df: pd.DataFrame) -> dict:
    brain = EntryPriceBrain.load(meta_path)
    long_t, short_t = _future_extrema_offsets(df, horizon=brain.horizon, clip_pct=max(brain.long_clip, brain.short_clip))
    pred = brain.predict_batch_from_df(df)
    valid_l = np.isfinite(long_t)
    valid_s = np.isfinite(short_t)
    long_mae = mean_absolute_error(long_t[valid_l], pred.loc[valid_l, "entry_long_offset"])
    short_mae = mean_absolute_error(short_t[valid_s], pred.loc[valid_s, "entry_short_offset"])
    cols = sorted(set(brain.feature_cols) | set(brain.long_feature_cols) | set(brain.short_feature_cols))
    return {
        "entry_long_mae": float(long_mae),
        "entry_short_mae": float(short_mae),
        "entry_avg_mae": float((long_mae + short_mae) / 2.0),
        **_check_features("entry", cols),
    }


def eval_multitarget(meta_path: str, df: pd.DataFrame) -> dict:
    meta = _read_meta(meta_path)
    with open(_model_path(meta_path, meta), "rb") as f:
        payload = pickle.load(f)
    cols = list(meta.get("feature_cols", payload.get("feature_cols", [])))
    y_dir = make_triple_barrier_targets(df, atr_mult=0.8, max_hold=12, atr_window=14)
    y_q, y_h = _build_quality_and_hold_targets(df, y_dir, int(payload.get("horizon", meta.get("horizon", 12))))
    valid = (y_dir >= 0) & np.isfinite(y_q) & np.isfinite(y_h)
    x = _to_numeric_frame(df, cols).iloc[valid]
    models = payload
    dir_valid = ((y_dir[valid] == 0) | (y_dir[valid] == 2))
    yv = (y_dir[valid][dir_valid] == 2).astype(np.int64)
    pred = models["direction_model"].predict(x.loc[dir_valid])
    q_pred = models["quality_model"].predict(x)
    h_pred = models["hold_model"].predict(x)
    return {
        "mtl_dir_bacc": float(balanced_accuracy_score(yv, pred)),
        "mtl_quality_mae": float(np.mean(np.abs(q_pred - y_q[valid]))),
        "mtl_hold_mae": float(np.mean(np.abs(h_pred - y_h[valid]))),
        **_check_features("mtl", cols),
    }


def eval_quantile(meta_path: str, df: pd.DataFrame) -> dict:
    meta = _read_meta(meta_path)
    with open(_model_path(meta_path, meta), "rb") as f:
        payload = pickle.load(f)
    cols = list(meta.get("feature_cols", payload.get("feature_cols", [])))
    horizon = int(payload.get("horizon", meta.get("horizon", 12)))
    flat = float(payload.get("flat_threshold", meta.get("flat_threshold", 0.0005)))
    y = make_future_return(df, horizon=horizon)
    valid = np.isfinite(y)
    x = _to_numeric_frame(df, cols).iloc[valid]
    q10, q50, q90 = _predict_quantiles(payload["models"], x)
    mae, dir_acc, interval_width = _score_predictions(y[valid], q10, q50, q90, flat)
    return {
        "quant_mae": float(mae),
        "quant_dir_acc": float(dir_acc),
        "quant_interval_width": float(interval_width),
        **_check_features("quant", cols),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--rl-path", default="")
    p.add_argument("--baseline-dir", default="data/ensemble/supervised")
    p.add_argument("--candidate-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = load_feature_frame(args.data_path, args.rl_path)
    specs = {
        "trend_xgb": ("trend_xgb.json", eval_trend),
        "entry_price_model": ("entry_price_model.json", eval_entry),
        "multi_target_lgbm": ("multi_target_lgbm.json", eval_multitarget),
        "quantile_forest": ("quantile_forest.json", eval_quantile),
    }
    rows = []
    for model_name, (file_name, fn) in specs.items():
        for tag, directory in [("baseline", args.baseline_dir), ("candidate", args.candidate_dir)]:
            path = os.path.join(directory, file_name)
            row = {"model": model_name, "variant": tag, "path": path}
            row.update(fn(path, df))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
