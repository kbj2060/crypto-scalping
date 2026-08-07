from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENSEMBLE_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_ENSEMBLE_DIR)
for _p in (_ROOT_DIR, _ENSEMBLE_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.artifact_utils import resolve_model_meta_paths, save_pickle
from ensemble.supervised.common import (
    DEFAULT_DATA_PATH,
    DEFAULT_RL_DATA_PATH,
    load_feature_frame,
    median_fill_by_train,
    select_feature_columns,
    time_split_indices,
)
from features.high_order_state import HIGH_ORDER_STATE_COLS
from features.selection import auto_select_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FORBIDDEN_FEATURE_FRAGMENTS = (
    "future",
    "target",
    "label",
    "realized",
    "cash_after",
    "trade_pnl",
)
FORBIDDEN_FEATURE_PREFIXES = (
    "m7_",
    "clean_regime_2024_unsup_v4_",
    "clean_regime4_2024_unsup_v1_",
)
FORBIDDEN_FEATURE_NAMES = {
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
}


@dataclass(frozen=True)
class PathLabelConfig:
    horizon: int = 12
    tp_ref: float = 0.0020
    sl_ref: float = 0.0015
    cost: float = 0.00055
    mae_penalty: float = 0.65


def _require_lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        raise ImportError("lightgbm is required for train_lightgbm_ensemble.py") from exc
    return LGBMClassifier, LGBMRegressor


def _is_allowed_feature(col: str) -> bool:
    c = str(col)
    lo = c.lower()
    if lo in FORBIDDEN_FEATURE_NAMES:
        return False
    if any(lo.startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES):
        return False
    if any(fragment in lo for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
        return False
    return True


def _path_labels(df: pd.DataFrame, cfg: PathLabelConfig) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(df.get("high", df["close"]), errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(df.get("low", df["close"]), errors="coerce").to_numpy(dtype=np.float64)
    n = len(df)
    out = {
        "long_mfe": np.full(n, np.nan, dtype=np.float64),
        "long_mae": np.full(n, np.nan, dtype=np.float64),
        "short_mfe": np.full(n, np.nan, dtype=np.float64),
        "short_mae": np.full(n, np.nan, dtype=np.float64),
        "long_sl_first": np.full(n, np.nan, dtype=np.float64),
        "short_sl_first": np.full(n, np.nan, dtype=np.float64),
    }
    for i in range(0, n - cfg.horizon):
        cur = max(float(close[i]), 1e-12)
        h = high[i + 1 : i + cfg.horizon + 1]
        l = low[i + 1 : i + cfg.horizon + 1]
        if len(h) != cfg.horizon or not np.isfinite(cur):
            continue
        long_fav = h / cur - 1.0
        long_adv = 1.0 - l / cur
        short_fav = cur / np.maximum(l, 1e-12) - 1.0
        short_adv = h / cur - 1.0
        out["long_mfe"][i] = float(np.nanmax(long_fav))
        out["long_mae"][i] = float(np.nanmax(long_adv))
        out["short_mfe"][i] = float(np.nanmax(short_fav))
        out["short_mae"][i] = float(np.nanmax(short_adv))
        long_tp_hit = np.flatnonzero(long_fav >= cfg.tp_ref)
        long_sl_hit = np.flatnonzero(long_adv >= cfg.sl_ref)
        short_tp_hit = np.flatnonzero(short_fav >= cfg.tp_ref)
        short_sl_hit = np.flatnonzero(short_adv >= cfg.sl_ref)
        long_tp_i = int(long_tp_hit[0]) if len(long_tp_hit) else cfg.horizon + 1
        long_sl_i = int(long_sl_hit[0]) if len(long_sl_hit) else cfg.horizon + 1
        short_tp_i = int(short_tp_hit[0]) if len(short_tp_hit) else cfg.horizon + 1
        short_sl_i = int(short_sl_hit[0]) if len(short_sl_hit) else cfg.horizon + 1
        out["long_sl_first"][i] = float(long_sl_i < long_tp_i)
        out["short_sl_first"][i] = float(short_sl_i < short_tp_i)
    labels = pd.DataFrame(out)
    labels["long_edge"] = labels["long_mfe"] - cfg.mae_penalty * labels["long_mae"] - cfg.cost
    labels["short_edge"] = labels["short_mfe"] - cfg.mae_penalty * labels["short_mae"] - cfg.cost
    labels["tradeability"] = labels[["long_edge", "short_edge"]].max(axis=1)
    labels["best_side"] = np.where(labels["long_edge"] >= labels["short_edge"], 1.0, -1.0)
    return labels


def _select_features(df: pd.DataFrame, labels: pd.DataFrame, tr_idx: np.ndarray, max_features: int) -> list[str]:
    must_include = [c for c in HIGH_ORDER_STATE_COLS if c in df.columns and _is_allowed_feature(c)]
    candidates = [c for c in select_feature_columns(df, must_include=must_include) if _is_allowed_feature(c)]
    tmp = df.iloc[tr_idx].copy()
    tmp.index = range(len(tmp))
    tmp["_target"] = (labels["tradeability"].iloc[tr_idx].to_numpy() > 0.0).astype(np.int64)
    selected = auto_select_features(
        tmp,
        candidates,
        target_col="_target",
        max_features=max_features,
        corr_threshold=0.85,
        must_include=must_include,
    )
    return [c for c in selected if _is_allowed_feature(c)]


def _reg_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_samples": args.min_child_samples,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "verbose": -1,
    }


def _clf_params(args: argparse.Namespace) -> dict[str, Any]:
    p = _reg_params(args)
    p["objective"] = "binary"
    return p


def _fit_models(args: argparse.Namespace, x_train: pd.DataFrame, y_train: pd.DataFrame) -> dict[str, Any]:
    LGBMClassifier, LGBMRegressor = _require_lightgbm()
    rp = _reg_params(args)
    cp = _clf_params(args)
    models = {
        "long_edge": LGBMRegressor(objective="regression_l1", **rp),
        "short_edge": LGBMRegressor(objective="regression_l1", **rp),
        "tradeability": LGBMRegressor(objective="regression_l1", **rp),
        "long_mae_q90": LGBMRegressor(objective="quantile", alpha=0.90, **rp),
        "short_mae_q90": LGBMRegressor(objective="quantile", alpha=0.90, **rp),
        "long_adverse": LGBMClassifier(**cp),
        "short_adverse": LGBMClassifier(**cp),
    }
    models["long_edge"].fit(x_train, y_train["long_edge"])
    models["short_edge"].fit(x_train, y_train["short_edge"])
    models["tradeability"].fit(x_train, y_train["tradeability"])
    models["long_mae_q90"].fit(x_train, y_train["long_mae"])
    models["short_mae_q90"].fit(x_train, y_train["short_mae"])
    models["long_adverse"].fit(x_train, y_train["long_sl_first"].astype(np.int64))
    models["short_adverse"].fit(x_train, y_train["short_sl_first"].astype(np.int64))
    return models


def predict_lightgbm_ensemble(models: dict[str, Any], x: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=x.index)
    out["m7_long_edge"] = models["long_edge"].predict(x)
    out["m7_short_edge"] = models["short_edge"].predict(x)
    out["m7_tradeability_score"] = models["tradeability"].predict(x)
    out["m7_long_mae_q90"] = np.maximum(0.0, models["long_mae_q90"].predict(x))
    out["m7_short_mae_q90"] = np.maximum(0.0, models["short_mae_q90"].predict(x))
    out["m7_long_adverse_prob"] = models["long_adverse"].predict_proba(x)[:, 1]
    out["m7_short_adverse_prob"] = models["short_adverse"].predict_proba(x)[:, 1]
    out["m7_path_best_side"] = np.where(out["m7_long_edge"] >= out["m7_short_edge"], 1.0, -1.0)
    return out.astype(np.float32)


def _score_predictions(pred: pd.DataFrame, y: pd.DataFrame) -> dict[str, float]:
    y_trade = y["tradeability"].to_numpy(dtype=np.float64)
    y_pos = (y_trade > 0.0).astype(np.int64)
    score = pred["m7_tradeability_score"].to_numpy(dtype=np.float64)
    q = np.quantile(score, 0.90)
    selected = score >= q
    long_adv = y["long_sl_first"].astype(np.int64).to_numpy()
    short_adv = y["short_sl_first"].astype(np.int64).to_numpy()
    out = {
        "tradeability_mae": float(np.mean(np.abs(score - y_trade))),
        "top10_rate": float(np.mean(selected)),
        "top10_positive_rate": float(np.mean(y_pos[selected])) if selected.any() else 0.0,
        "top10_mean_tradeability": float(np.mean(y_trade[selected])) if selected.any() else 0.0,
        "long_edge_corr": float(np.corrcoef(pred["m7_long_edge"], y["long_edge"])[0, 1]),
        "short_edge_corr": float(np.corrcoef(pred["m7_short_edge"], y["short_edge"])[0, 1]),
        "long_adverse_rate": float(np.mean(long_adv)),
        "short_adverse_rate": float(np.mean(short_adv)),
    }
    for side, target in (("long", long_adv), ("short", short_adv)):
        prob = pred[f"m7_{side}_adverse_prob"].to_numpy(dtype=np.float64)
        if len(np.unique(target)) > 1:
            out[f"{side}_adverse_auc"] = float(roc_auc_score(target, prob))
            out[f"{side}_adverse_ap"] = float(average_precision_score(target, prob))
        else:
            out[f"{side}_adverse_auc"] = 0.5
            out[f"{side}_adverse_ap"] = float(np.mean(target))
    return out


def _prepare_xy(data_path: str, rl_path: str, cfg: PathLabelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_feature_frame(data_path, rl_path)
    labels = _path_labels(df, cfg)
    valid = labels.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].reset_index(drop=True)
    return df, labels


def train(args: argparse.Namespace) -> dict[str, Any]:
    cfg = PathLabelConfig(horizon=args.horizon, tp_ref=args.tp_ref, sl_ref=args.sl_ref, cost=args.cost, mae_penalty=args.mae_penalty)
    df, labels = _prepare_xy(args.data_path, args.rl_path, cfg)
    tr_idx, va_idx, te_idx = time_split_indices(len(df), args.train_ratio, args.val_ratio)
    selected = _select_features(df, labels, tr_idx, args.max_features)
    x_all = df[selected].replace([np.inf, -np.inf], np.nan)

    x_train = x_all.iloc[tr_idx].copy()
    x_val = x_all.iloc[va_idx].copy()
    x_test = x_all.iloc[te_idx].copy()
    x_train, x_val = median_fill_by_train(x_train, x_val)
    x_train, x_test = median_fill_by_train(x_train, x_test)
    x_trainval = x_all.iloc[np.concatenate([tr_idx, va_idx])].copy()
    x_trainval, x_test = median_fill_by_train(x_trainval, x_test)

    y_trainval = labels.iloc[np.concatenate([tr_idx, va_idx])].reset_index(drop=True)
    models = _fit_models(args, x_trainval[selected], y_trainval)
    test_pred = predict_lightgbm_ensemble(models, x_test[selected])
    test_metrics = _score_predictions(test_pred, labels.iloc[te_idx].reset_index(drop=True))

    oos_metrics: dict[str, float] | None = None
    if args.oos_data_path:
        oos_df, oos_labels = _prepare_xy(args.oos_data_path, args.oos_rl_path, cfg)
        oos_x = oos_df[selected].replace([np.inf, -np.inf], np.nan)
        _, oos_x = median_fill_by_train(x_trainval[selected], oos_x)
        oos_pred = predict_lightgbm_ensemble(models, oos_x[selected])
        oos_metrics = _score_predictions(oos_pred, oos_labels)

    model_path, meta_path = resolve_model_meta_paths(args.save_path)
    save_pickle(
        {
            "models": models,
            "feature_cols": selected,
            "label_config": asdict(cfg),
            "model_family": "lightgbm_ensemble",
        },
        model_path,
    )
    artifact = {
        "feature_cols": selected,
        "model_path": os.path.basename(model_path),
        "label_config": asdict(cfg),
        "meta": {
            "algorithm": "lightgbm_ensemble",
            "heads": [
                "long_edge",
                "short_edge",
                "tradeability",
                "long_mae_q90",
                "short_mae_q90",
                "long_adverse",
                "short_adverse",
            ],
            "train_rows": int(len(tr_idx)),
            "val_rows": int(len(va_idx)),
            "test_rows": int(len(te_idx)),
            "test_metrics": test_metrics,
            "oos_metrics": oos_metrics,
            "params": {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "min_child_samples": args.min_child_samples,
                "reg_alpha": args.reg_alpha,
                "reg_lambda": args.reg_lambda,
                "seed": args.seed,
            },
        },
    }
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=True)
    results_path = os.path.join(os.path.dirname(meta_path), "lightgbm_ensemble_training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"test_metrics": test_metrics, "oos_metrics": oos_metrics, "artifact": meta_path}, f, indent=2)
    logger.info("saved model: %s", model_path)
    logger.info("saved meta: %s", meta_path)
    logger.info("test_metrics=%s", test_metrics)
    logger.info("oos_metrics=%s", oos_metrics)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train M7 LightGBM ensemble heads.")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--rl-path", default=DEFAULT_RL_DATA_PATH)
    p.add_argument("--oos-data-path", default="")
    p.add_argument("--oos-rl-path", default="")
    p.add_argument("--save-path", default="data/ensemble/supervised/lightgbm_ensemble.json")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--tp-ref", type=float, default=0.0020)
    p.add_argument("--sl-ref", type=float, default=0.0015)
    p.add_argument("--cost", type=float, default=0.00055)
    p.add_argument("--mae-penalty", type=float, default=0.65)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--max-features", type=int, default=72)
    p.add_argument("--n-estimators", type=int, default=550)
    p.add_argument("--learning-rate", type=float, default=0.035)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--subsample", type=float, default=0.82)
    p.add_argument("--colsample-bytree", type=float, default=0.82)
    p.add_argument("--min-child-samples", type=int, default=30)
    p.add_argument("--reg-alpha", type=float, default=0.03)
    p.add_argument("--reg-lambda", type=float, default=0.15)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--startup-check-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: train_lightgbm_ensemble")
        raise SystemExit(0)
    train(args)
