from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ensemble.train_trade_candidate_detector import FEATURE_COLS
from scripts.build_trade_candidates_v1 import build_candidate_labels


DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/splits/year_oos/rl_training_2025_m7.csv"
DEFAULT_OUT_CSV = "/home/llewyn/crypto-scalping/data/ensemble/event_driven/trade_candidates_v1_oof.csv"
DEFAULT_OUT_JSON = "/home/llewyn/crypto-scalping/data/ensemble/event_driven/trade_candidates_v1_oof.json"
DEFAULT_MODEL_JSON = "/home/llewyn/crypto-scalping/data/ensemble/event_driven/trade_candidate_detector_catboost_oof.json"


def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _resolve_meta_paths(save_path: str) -> tuple[str, str]:
    meta_path = save_path
    if meta_path.endswith(".pkl"):
        meta_path = meta_path[:-4] + ".json"
    model_path = meta_path[:-5] + ".pkl" if meta_path.endswith(".json") else meta_path + ".pkl"
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    return model_path, meta_path


def _fit_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
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
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
    return model


def _build_forward_folds(n: int, warmup_frac: float, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    warm_end = max(512, int(n * float(warmup_frac)))
    warm_end = min(max(warm_end, 1), max(n - 2, 1))
    rem = max(n - warm_end, 0)
    if rem <= 0:
        return []
    fold_size = max(256, rem // max(int(n_folds), 1))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    start = warm_end
    while start < n:
        end = min(n, start + fold_size)
        tr_idx = np.arange(0, start, dtype=np.int64)
        va_idx = np.arange(start, end, dtype=np.int64)
        if len(tr_idx) >= 512 and len(va_idx) > 0:
            folds.append((tr_idx, va_idx))
        start = end
    return folds


def main() -> None:
    ap = argparse.ArgumentParser(description="Build full-timeline OOF candidate detector scores")
    ap.add_argument("--csv-path", default=DEFAULT_CSV)
    ap.add_argument("--output-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--output-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--save-path", default=DEFAULT_MODEL_JSON)
    ap.add_argument("--horizons", default="4,8,12")
    ap.add_argument("--atr-window", type=int, default=14)
    ap.add_argument("--tp-mult", type=float, default=0.90)
    ap.add_argument("--sl-mult", type=float, default=0.60)
    ap.add_argument("--min-move", type=float, default=0.0018)
    ap.add_argument("--side-margin-min", type=float, default=0.00025)
    ap.add_argument("--warmup-frac", type=float, default=0.35)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--l2-leaf-reg", type=float, default=8.0)
    ap.add_argument("--od-wait", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = pd.read_csv(args.csv_path)
    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    df = build_candidate_labels(
        raw,
        horizons=horizons,
        atr_window=int(args.atr_window),
        tp_mult=float(args.tp_mult),
        sl_mult=float(args.sl_mult),
        min_move=float(args.min_move),
        side_margin_min=float(args.side_margin_min),
    )
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    df = _safe_fill(df)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required features: {missing}")
    y = pd.to_numeric(df["evt_candidate_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    valid = np.isin(y, [0, 1, 2])
    df = df.loc[valid].reset_index(drop=True)
    y = y[valid]

    x = df[FEATURE_COLS].copy()
    n = len(df)
    folds = _build_forward_folds(n, warmup_frac=float(args.warmup_frac), n_folds=int(args.n_folds))

    oof_probs = np.zeros((n, 3), dtype=np.float64)
    oof_available = np.zeros(n, dtype=np.int8)
    fold_reports: list[dict[str, Any]] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        x_train = x.iloc[tr_idx].copy()
        x_val = x.iloc[va_idx].copy()
        med = x_train.median(numeric_only=True)
        x_train = x_train.fillna(med)
        x_val = x_val.fillna(med)
        y_train = y[tr_idx]
        y_val = y[va_idx]
        model = _fit_model(x_train, y_train, x_val, y_val, args)
        probs = model.predict_proba(x_val)
        oof_probs[va_idx, :] = probs
        oof_available[va_idx] = 1
        pred = probs.argmax(axis=1).astype(int)
        cand_mask = np.isin(y_val, [1, 2])
        fold_reports.append(
            {
                "fold": int(fold_idx),
                "train_rows": int(len(tr_idx)),
                "val_rows": int(len(va_idx)),
                "val_balanced_acc": float(balanced_accuracy_score(y_val, pred)),
                "val_candidate_dir_f1": float(f1_score(y_val[cand_mask], pred[cand_mask], average="macro")) if np.any(cand_mask) else 0.0,
            }
        )

    avail_mask = oof_available == 1
    if np.any(avail_mask):
        y_av = y[avail_mask]
        probs_av = oof_probs[avail_mask]
        pred_av = probs_av.argmax(axis=1).astype(int)
        oof_bal_acc = float(balanced_accuracy_score(y_av, pred_av))
        cand_mask = np.isin(y_av, [1, 2])
        oof_dir_f1 = float(f1_score(y_av[cand_mask], pred_av[cand_mask], average="macro")) if np.any(cand_mask) else 0.0
        oof_report = classification_report(y_av, pred_av, output_dict=True)
    else:
        oof_bal_acc = 0.0
        oof_dir_f1 = 0.0
        oof_report = {}

    model_path, meta_path = _resolve_meta_paths(args.save_path)
    x_full = x.copy()
    med_full = x_full.median(numeric_only=True)
    x_full = x_full.fillna(med_full)
    final_model = _fit_model(x_full, y, x_full.iloc[-max(256, n // 10):].copy(), y[-max(256, n // 10):], args)
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": FEATURE_COLS}, f)

    out = df.copy()
    out["evt_oof_available"] = oof_available
    out["evt_oof_none_prob"] = oof_probs[:, 0]
    out["evt_oof_long_prob"] = oof_probs[:, 1]
    out["evt_oof_short_prob"] = oof_probs[:, 2]
    out["evt_oof_prob_max"] = oof_probs.max(axis=1)
    out["evt_oof_edge"] = out["evt_oof_long_prob"] - out["evt_oof_short_prob"]

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    artifact = {
        "csv_path": args.csv_path,
        "output_csv": args.output_csv,
        "model_path": model_path,
        "feature_cols": FEATURE_COLS,
        "rows": int(n),
        "oof_available_rows": int(avail_mask.sum()),
        "oof_balanced_acc": float(oof_bal_acc),
        "oof_candidate_dir_f1": float(oof_dir_f1),
        "oof_report": oof_report,
        "fold_reports": fold_reports,
        "params": {
            "horizons": horizons,
            "atr_window": int(args.atr_window),
            "tp_mult": float(args.tp_mult),
            "sl_mult": float(args.sl_mult),
            "min_move": float(args.min_move),
            "side_margin_min": float(args.side_margin_min),
            "warmup_frac": float(args.warmup_frac),
            "n_folds": int(args.n_folds),
            "iterations": int(args.iterations),
            "depth": int(args.depth),
            "learning_rate": float(args.learning_rate),
            "l2_leaf_reg": float(args.l2_leaf_reg),
            "od_wait": int(args.od_wait),
            "seed": int(args.seed),
        },
    }
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"feature_cols": FEATURE_COLS, "meta": artifact}, f, ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
