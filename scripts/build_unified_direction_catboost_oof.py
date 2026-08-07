from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ensemble.supervised.common import make_triple_barrier_targets, median_fill_by_train
from ensemble.train_unified_direction_catboost import FEATURE_COLS


DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified.csv"
DEFAULT_OUT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_dircat_oof.csv"
DEFAULT_OUT_JSON = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_direction_catboost_oof_report.json"


def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


@dataclass
class FoldSpec:
    train_start: int
    train_end: int
    val_start: int
    val_end: int


def build_folds(
    n: int,
    train_end: int,
    min_train_frac: float,
    val_frac: float,
    step_frac: float,
) -> list[FoldSpec]:
    min_train = max(200, int(train_end * min_train_frac))
    val_size = max(100, int(train_end * val_frac))
    step = max(100, int(train_end * step_frac))
    folds: list[FoldSpec] = []
    cursor = min_train
    while cursor + val_size <= train_end:
        folds.append(FoldSpec(0, cursor, cursor, cursor + val_size))
        cursor += step
    if not folds or folds[-1].val_end < train_end:
        last_train_end = max(min_train, train_end - val_size)
        folds.append(FoldSpec(0, last_train_end, last_train_end, train_end))
    dedup: list[FoldSpec] = []
    seen: set[tuple[int, int, int, int]] = set()
    for f in folds:
        key = (f.train_start, f.train_end, f.val_start, f.val_end)
        if key not in seen and f.train_end > f.train_start and f.val_end > f.val_start:
            seen.add(key)
            dedup.append(f)
    return dedup


def _fit_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame | None,
    y_val: np.ndarray | None,
    args: argparse.Namespace,
) -> CatBoostClassifier:
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
    if x_val is not None and y_val is not None and len(x_val) > 0:
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
    else:
        model.fit(x_train, y_train, verbose=False)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OOF CatBoost direction probabilities with walk-forward folds")
    ap.add_argument("--csv-path", default=DEFAULT_CSV)
    ap.add_argument("--output-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--output-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--atr-mult", type=float, default=0.8)
    ap.add_argument("--max-hold", type=int, default=8)
    ap.add_argument("--atr-window", type=int, default=14)
    ap.add_argument("--min-train-frac", type=float, default=0.35)
    ap.add_argument("--oof-val-frac", type=float, default=0.10)
    ap.add_argument("--oof-step-frac", type=float, default=0.10)
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--l2-leaf-reg", type=float, default=8.0)
    ap.add_argument("--od-wait", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    df = _safe_fill(df)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required features: {missing}")

    y = make_triple_barrier_targets(df, atr_mult=args.atr_mult, max_hold=args.max_hold, atr_window=args.atr_window)
    valid = y >= 0
    work = df.loc[valid].reset_index(drop=True).copy()
    y = y[valid]

    n = len(work)
    train_end = int(n * (args.train_ratio + args.val_ratio))
    test_start = train_end
    folds = build_folds(n=n, train_end=train_end, min_train_frac=args.min_train_frac, val_frac=args.oof_val_frac, step_frac=args.oof_step_frac)

    x_all = work[FEATURE_COLS].copy()
    oof_probs = np.full((n, 3), np.nan, dtype=np.float64)
    oof_fold = np.full(n, -1, dtype=np.int32)
    fold_reports: list[dict[str, Any]] = []

    for fold_idx, fold in enumerate(folds):
        x_train = x_all.iloc[fold.train_start:fold.train_end].copy()
        y_train = y[fold.train_start:fold.train_end]
        x_val = x_all.iloc[fold.val_start:fold.val_end].copy()
        y_val = y[fold.val_start:fold.val_end]
        x_train, x_val = median_fill_by_train(x_train, x_val)
        model = _fit_model(x_train, y_train, x_val, y_val, args)
        probs = model.predict_proba(x_val)
        preds = np.argmax(probs, axis=1)
        bal_acc = float(balanced_accuracy_score(y_val, preds))
        updown_mask = np.isin(y_val, [0, 2])
        dir_f1 = float(f1_score(y_val[updown_mask], preds[updown_mask], average="macro")) if np.any(updown_mask) else 0.0
        oof_probs[fold.val_start:fold.val_end] = probs
        oof_fold[fold.val_start:fold.val_end] = fold_idx
        fold_reports.append(
            {
                "fold": fold_idx,
                "train_rows": int(fold.train_end - fold.train_start),
                "val_rows": int(fold.val_end - fold.val_start),
                "val_balanced_acc": bal_acc,
                "val_dir_f1": dir_f1,
                "val_start_idx": int(fold.val_start),
                "val_end_idx": int(fold.val_end),
            }
        )

    covered = np.isfinite(oof_probs).all(axis=1)
    if covered[:train_end].sum() == 0:
        raise RuntimeError("no OOF coverage generated for calibration region")

    # Final model for holdout test predictions.
    x_pretest = x_all.iloc[:train_end].copy()
    y_pretest = y[:train_end]
    x_holdout = x_all.iloc[test_start:].copy()
    if len(x_holdout) > 0:
        x_pretest, x_holdout = median_fill_by_train(x_pretest, x_holdout)
    final_model = _fit_model(x_pretest, y_pretest, None, None, args)
    if len(x_holdout) > 0:
        holdout_probs = final_model.predict_proba(x_holdout)
        oof_probs[test_start:] = holdout_probs
        oof_fold[test_start:] = 9999

    out = work.copy()
    out["ud_cat_short_prob"] = oof_probs[:, 0]
    out["ud_cat_flat_prob"] = oof_probs[:, 1]
    out["ud_cat_long_prob"] = oof_probs[:, 2]
    out["ud_cat_edge"] = out["ud_cat_long_prob"] - out["ud_cat_short_prob"]
    prob_max = np.full(n, np.nan, dtype=np.float64)
    pred_class = np.full(n, -1, dtype=np.int32)
    finite_rows = np.isfinite(oof_probs).all(axis=1)
    if np.any(finite_rows):
        prob_max[finite_rows] = np.max(oof_probs[finite_rows], axis=1)
        pred_class[finite_rows] = np.argmax(oof_probs[finite_rows], axis=1)
    out["ud_cat_prob_max"] = prob_max
    out["ud_cat_pred_class"] = pred_class
    out["ud_cat_oof_fold"] = oof_fold
    out["ud_cat_is_holdout"] = (np.arange(len(out)) >= test_start).astype(np.int8)
    out["ud_cat_tb_label"] = y.astype(np.int64)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    calib_mask = (np.arange(n) < train_end) & covered
    holdout_mask = np.arange(n) >= test_start
    calib_pred = np.argmax(oof_probs[calib_mask], axis=1)
    holdout_pred = np.argmax(oof_probs[holdout_mask], axis=1) if np.any(holdout_mask) else np.array([], dtype=np.int64)
    report = {
        "csv_path": args.csv_path,
        "output_csv": args.output_csv,
        "atr_mult": args.atr_mult,
        "max_hold": args.max_hold,
        "atr_window": args.atr_window,
        "feature_cols": FEATURE_COLS,
        "folds": fold_reports,
        "calibration_rows": int(calib_mask.sum()),
        "holdout_rows": int(holdout_mask.sum()),
        "calibration_balanced_acc": float(balanced_accuracy_score(y[calib_mask], calib_pred)),
        "holdout_balanced_acc": float(balanced_accuracy_score(y[holdout_mask], holdout_pred)) if np.any(holdout_mask) else None,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
