from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_sparse_candidates.csv"
DEFAULT_OUT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_sparse_gate_oof.csv"
DEFAULT_OUT_JSON = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_gate_oof_report.json"
DEFAULT_OUT_DIR = "/home/llewyn/crypto-scalping/data/ensemble/supervised"

FEATURE_COLS = [
    "ud_cand_side",
    "ud_cand_hold",
    "ud_cand_quality",
    "ud_cand_raw_edge",
    "ud_cand_sup_prob_max",
    "ud_cand_agree",
    "ud_cat_long_prob",
    "ud_cat_flat_prob",
    "ud_cat_short_prob",
    "ud_cat_edge",
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
]


def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _compute_candidate_pnl(df: pd.DataFrame) -> np.ndarray:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    side = pd.to_numeric(df["ud_cand_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    hold = pd.to_numeric(df["ud_cand_hold"], errors="coerce").fillna(6).clip(4, 8).astype(np.int32).to_numpy()
    pnl = np.zeros(len(df), dtype=np.float64)
    for i in range(len(df)):
        if side[i] == 0:
            continue
        j = min(len(df) - 1, i + int(hold[i]))
        entry = close[i] * (1.0 + 0.0002) if side[i] == 1 else close[i] * (1.0 - 0.0002)
        exitp = close[j] * (1.0 - 0.0002) if side[i] == 1 else close[j] * (1.0 + 0.0002)
        pnl[i] = ((exitp - entry) / entry) if side[i] == 1 else ((entry - exitp) / entry)
    return pnl


def _make_binary_target(pnl: np.ndarray, cost_buffer: float) -> np.ndarray:
    return (pnl > cost_buffer).astype(np.int64)


def _make_expectancy_bins(pnl: np.ndarray, cost_buffer: float, hi_quantile: float) -> tuple[np.ndarray, float]:
    positive = pnl[pnl > cost_buffer]
    hi_thr = float(np.quantile(positive, hi_quantile)) if len(positive) > 10 else float(cost_buffer * 3.0)
    y = np.zeros(len(pnl), dtype=np.int64)
    y[(pnl > cost_buffer) & (pnl <= hi_thr)] = 1
    y[pnl > hi_thr] = 2
    return y, hi_thr


def _fit_catboost(x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray, multiclass: bool, seed: int) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function="MultiClass" if multiclass else "Logloss",
        eval_metric="MultiClass" if multiclass else "AUC",
        iterations=500,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=8.0,
        random_seed=seed,
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=40,
        verbose=False,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
    return model


def _predict_positive_prob(model: CatBoostClassifier, x: pd.DataFrame, mode: str) -> np.ndarray:
    probs = model.predict_proba(x)
    if mode == "binary":
        return probs[:, 1]
    return probs[:, 2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OOF sparse candidate gate probabilities")
    ap.add_argument("--csv-path", default=DEFAULT_CSV)
    ap.add_argument("--output-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--output-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--cost-buffer", type=float, default=0.0015)
    ap.add_argument("--hi-quantile", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    df = _safe_fill(df)
    cand = df[df["ud_cand_flag"] == 1].copy().reset_index(drop=True)
    pnl = _compute_candidate_pnl(cand)
    y_binary = _make_binary_target(pnl, args.cost_buffer)
    y_exp, hi_thr = _make_expectancy_bins(pnl, args.cost_buffer, args.hi_quantile)

    pre_mask = pd.to_numeric(cand["ud_cat_is_holdout"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 0
    hold_mask = ~pre_mask
    fold_col = pd.to_numeric(cand["ud_cat_oof_fold"], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
    fold_ids = sorted(int(x) for x in np.unique(fold_col[pre_mask]) if x >= 0)
    x_all = cand[FEATURE_COLS].copy()

    results: dict[str, Any] = {"csv_path": args.csv_path, "feature_cols": FEATURE_COLS, "cost_buffer": args.cost_buffer, "hi_quantile": args.hi_quantile, "hi_threshold": hi_thr, "modes": {}}

    for mode, y in [("binary", y_binary), ("expbin", y_exp)]:
        oof_prob = np.full(len(cand), np.nan, dtype=np.float64)
        fold_reports: list[dict[str, Any]] = []
        multiclass = mode == "expbin"
        for fid in fold_ids:
            train_mask = pre_mask & (fold_col >= 0) & (fold_col < fid)
            val_mask = pre_mask & (fold_col == fid)
            if train_mask.sum() < 20 or val_mask.sum() == 0:
                continue
            x_train = x_all.loc[train_mask].copy()
            x_val = x_all.loc[val_mask].copy()
            med = x_train.median(numeric_only=True)
            x_train = x_train.fillna(med)
            x_val = x_val.fillna(med)
            y_train = y[train_mask]
            y_val = y[val_mask]
            model = _fit_catboost(x_train, y_train, x_val, y_val, multiclass, args.seed)
            prob = _predict_positive_prob(model, x_val, mode)
            oof_prob[val_mask] = prob
            if mode == "binary":
                fold_metric = float(roc_auc_score(y_val, prob)) if len(np.unique(y_val)) > 1 else None
                fold_ap = float(average_precision_score(y_val, prob)) if len(np.unique(y_val)) > 1 else None
            else:
                top = (y_val == 2).astype(np.int64)
                fold_metric = float(roc_auc_score(top, prob)) if len(np.unique(top)) > 1 else None
                fold_ap = float(average_precision_score(top, prob)) if len(np.unique(top)) > 1 else None
            fold_reports.append({"fold": fid, "train_rows": int(train_mask.sum()), "val_rows": int(val_mask.sum()), "auc_or_top_auc": fold_metric, "ap_or_top_ap": fold_ap})

        # Final model on all preholdout, predict holdout.
        if pre_mask.sum() > 20 and hold_mask.sum() > 0:
            x_train = x_all.loc[pre_mask].copy()
            y_train = y[pre_mask]
            x_hold = x_all.loc[hold_mask].copy()
            med = x_train.median(numeric_only=True)
            x_train = x_train.fillna(med)
            x_hold = x_hold.fillna(med)
            # carve small tail validation from preholdout
            pre_idx = np.flatnonzero(pre_mask)
            split = max(20, int(len(pre_idx) * 0.85))
            train_idx = pre_idx[:split]
            val_idx = pre_idx[split:]
            x_fit = x_all.iloc[train_idx].copy().fillna(med)
            y_fit = y[train_idx]
            x_val = x_all.iloc[val_idx].copy().fillna(med) if len(val_idx) else x_fit.iloc[:1]
            y_val = y[val_idx] if len(val_idx) else y_fit[:1]
            model = _fit_catboost(x_fit, y_fit, x_val, y_val, multiclass, args.seed)
            hold_prob = _predict_positive_prob(model, x_hold, mode)
            oof_prob[hold_mask] = hold_prob

            model_path = os.path.join(args.out_dir, f"unified_sparse_gate_{mode}.pkl")
            meta_path = os.path.join(args.out_dir, f"unified_sparse_gate_{mode}.json")
            with open(model_path, "wb") as f:
                pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)
            hold_y = y[hold_mask]
            if mode == "binary":
                test_auc = float(roc_auc_score(hold_y, hold_prob)) if len(np.unique(hold_y)) > 1 else None
                test_ap = float(average_precision_score(hold_y, hold_prob)) if len(np.unique(hold_y)) > 1 else None
            else:
                top = (hold_y == 2).astype(np.int64)
                test_auc = float(roc_auc_score(top, hold_prob)) if len(np.unique(top)) > 1 else None
                test_ap = float(average_precision_score(top, hold_prob)) if len(np.unique(top)) > 1 else None
            meta = {
                "feature_cols": FEATURE_COLS,
                "model_path": os.path.basename(model_path),
                "meta": {
                    "algorithm": f"unified_sparse_gate_{mode}",
                    "csv_path": args.csv_path,
                    "test_auc": test_auc,
                    "test_ap": test_ap,
                    "hi_threshold": hi_thr if mode == "expbin" else None,
                    "fold_reports": fold_reports,
                },
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            results["modes"][mode] = {"meta_path": meta_path, "model_path": model_path, "fold_reports": fold_reports, "test_auc": test_auc, "test_ap": test_ap}

        cand[f"ud_gate_oof_{mode}_prob"] = oof_prob

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    cand.to_csv(args.output_csv, index=False)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
