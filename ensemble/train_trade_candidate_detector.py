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
try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    CatBoostClassifier = None
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_ROOT_DIR, _THIS_DIR, os.path.join(_ROOT_DIR, "ensemble")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.supervised.common import median_fill_by_train, time_split_indices


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/ensemble/event_driven/trade_candidates_v1.csv"
DEFAULT_SAVE = "/home/llewyn/crypto-scalping/data/ensemble/event_driven/trade_candidate_detector_catboost.json"
DEFAULT_SCORED = "/home/llewyn/crypto-scalping/data/ensemble/event_driven/trade_candidates_v1_scored.csv"

FEATURE_COLS = [
    "m7_trend_xgb_up",
    "m7_trend_xgb_dn",
    "m7_mtl_up",
    "m7_mtl_dn",
    "m7_quant_up",
    "m7_quant_dn",
    "m7_confidence",
    "m7_action",
    "m7_size",
    "m7_q50",
    "m7_qwidth",
    "m7_quality_pred",
    "m7_hold_pred",
    "m7_gmm_vol_rank",
    "m7_iso_score",
    "m7_composite_score",
    "m7_expected_ret",
    "m7_tail_risk",
    "smart_money_flow",
    "taker_acceleration",
    "trade_intensity",
    "garch_vol_z",
    "rogers_satchell_vol",
    "amihud_illiquidity_z",
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
    "evt_atr_ret",
    "evt_candidate_raw_edge",
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
    if CatBoostClassifier is None:
        raise ModuleNotFoundError("catboost is required to train the trade candidate detector")
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
    if "evt_candidate_label" not in df.columns:
        raise ValueError("missing evt_candidate_label")

    y = pd.to_numeric(df["evt_candidate_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    valid = np.isin(y, [0, 1, 2])
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

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.seed,
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=args.od_wait,
        verbose=False,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

    val_pred = model.predict(x_val).reshape(-1).astype(int)
    test_pred = model.predict(x_test).reshape(-1).astype(int)
    val_bal_acc = float(balanced_accuracy_score(y_val, val_pred))
    test_bal_acc = float(balanced_accuracy_score(y_test, test_pred))
    cand_mask = np.isin(y_test, [1, 2])
    cand_dir_f1 = float(f1_score(y_test[cand_mask], test_pred[cand_mask], average="macro")) if np.any(cand_mask) else 0.0
    report = classification_report(y_test, test_pred, output_dict=True)

    model_path, meta_path = _resolve_meta_paths(args.save_path)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)

    full_x = x.fillna(x_train.median(numeric_only=True))
    probs = model.predict_proba(full_x)
    scored = df.copy()
    scored["evt_det_none_prob"] = probs[:, 0]
    scored["evt_det_long_prob"] = probs[:, 1]
    scored["evt_det_short_prob"] = probs[:, 2]
    scored["evt_det_prob_max"] = probs.max(axis=1)
    scored["evt_det_edge"] = scored["evt_det_long_prob"] - scored["evt_det_short_prob"]
    if args.scored_csv:
        os.makedirs(os.path.dirname(args.scored_csv), exist_ok=True)
        scored.to_csv(args.scored_csv, index=False)

    artifact = {
        "feature_cols": FEATURE_COLS,
        "model_path": os.path.basename(model_path),
        "meta": {
            "algorithm": "trade_candidate_detector_catboost",
            "csv_path": args.csv_path,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "val_balanced_acc": val_bal_acc,
            "test_balanced_acc": test_bal_acc,
            "test_candidate_dir_f1": cand_dir_f1,
            "classification_report": report,
            "scored_csv": args.scored_csv,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    logger.info("saved model: %s", model_path)
    logger.info("saved meta: %s", meta_path)
    logger.info("val_balanced_acc=%.4f test_balanced_acc=%.4f test_candidate_dir_f1=%.4f", val_bal_acc, test_bal_acc, cand_dir_f1)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CatBoost event candidate detector")
    p.add_argument("--csv-path", default=DEFAULT_CSV)
    p.add_argument("--save-path", default=DEFAULT_SAVE)
    p.add_argument("--scored-csv", default=DEFAULT_SCORED)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--l2-leaf-reg", type=float, default=8.0)
    p.add_argument("--od-wait", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
