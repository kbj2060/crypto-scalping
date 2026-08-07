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
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_ROOT_DIR, _THIS_DIR, os.path.join(_ROOT_DIR, "ensemble")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.supervised.common import median_fill_by_train, time_split_indices
from ensemble.supervised.unified_stack_utils import apply_regime_veto, build_hazard_rows, build_sparse_candidates, simulate_trades

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CSV = "data/rl_training_2025_unified_dircat_tsfm_oof.csv"
DEFAULT_RULE = "data/ensemble/supervised/unified_sparse_shadow_rule.json"
DEFAULT_GATE = "data/ensemble/supervised/unified_tsfm_conformal_gate.json"
DEFAULT_SAVE = "data/ensemble/supervised/unified_exit_hazard_catboost.json"

FEATURE_COLS = [
    "haz_side",
    "haz_bars_held_norm",
    "haz_remaining_norm",
    "haz_current_pnl",
    "haz_mfe_sofar",
    "haz_mae_sofar",
    "haz_long_prob",
    "haz_flat_prob",
    "haz_short_prob",
    "haz_prob_max",
    "m7_target_quality",
    "smart_money_flow",
    "taker_acceleration",
    "trade_intensity",
    "garch_vol_z",
    "rogers_satchell_vol",
    "amihud_illiquidity_z",
    "patchtst_regime_sim",
    "timesnet_cycle_delta",
    "dlinear_smf_slope",
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


def _accept_mask(df: pd.DataFrame, gate_cfg: dict[str, Any], veto_mask: np.ndarray) -> np.ndarray:
    mode = str(gate_cfg["best"]["mode"])
    thresholds = dict(gate_cfg["best"]["thresholds"])
    pred_prob = (
        df[["ud_tsfm_long_prob", "ud_tsfm_flat_prob", "ud_tsfm_short_prob"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .max(axis=1)
        .to_numpy(np.float64)
    )
    regimes = df["ud_stack_regime"].astype(str).to_numpy()
    accept_conf = np.zeros(len(df), dtype=bool)
    if mode == "global":
        thr = float(thresholds["global"])
        accept_conf = pred_prob >= thr
    else:
        fallback = float(thresholds.get("_fallback", 1.0))
        for i, rg in enumerate(regimes):
            accept_conf[i] = pred_prob[i] >= float(thresholds.get(rg, fallback))
    return veto_mask & accept_conf & (pd.to_numeric(df["ud_stack_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1)


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(args.csv_path)
    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    df = _safe_fill(df)
    with open(args.rule_path, "r", encoding="utf-8") as f:
        rule_cfg = json.load(f)
    with open(args.gate_path, "r", encoding="utf-8") as f:
        gate_cfg = json.load(f)

    full = build_sparse_candidates(
        df,
        long_col="ud_tsfm_long_prob",
        flat_col="ud_tsfm_flat_prob",
        short_col="ud_tsfm_short_prob",
        params=rule_cfg["candidate_params"],
        prefix="ud_stack",
    )
    calib = full[full["ud_tsfm_is_holdout"] == 0].copy().reset_index(drop=True)
    veto_mask = apply_regime_veto(calib, rule_cfg["veto_rule"], prefix="ud_stack")
    take_mask = _accept_mask(calib, gate_cfg, veto_mask)
    trades = simulate_trades(
        calib,
        take_mask=take_mask,
        prefix="ud_stack",
        hold_scale=float(rule_cfg["hold_scale"]),
        close_on_opp=bool(rule_cfg["close_on_opp"]),
    )
    haz = build_hazard_rows(
        calib,
        trades,
        prob_cols={"long": "ud_tsfm_long_prob", "flat": "ud_tsfm_flat_prob", "short": "ud_tsfm_short_prob"},
        min_hold_bars=args.min_hold_bars,
        improve_margin=args.improve_margin,
        adverse_gap=args.adverse_gap,
    )
    haz = _safe_fill(haz)
    if haz.empty:
        raise RuntimeError("no hazard rows generated")

    tr_idx, va_idx, te_idx = time_split_indices(len(haz), args.train_ratio, args.val_ratio)
    x = haz[FEATURE_COLS].copy()
    y = pd.to_numeric(haz["haz_exit_label"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    x_train = x.iloc[tr_idx].copy()
    x_val = x.iloc[va_idx].copy()
    x_test = x.iloc[te_idx].copy()
    x_train, x_val = median_fill_by_train(x_train, x_val)
    x_train, x_test = median_fill_by_train(x_train, x_test)
    y_train, y_val, y_test = y[tr_idx], y[va_idx], y[te_idx]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
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
    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_prob))
    test_auc = float(roc_auc_score(y_test, test_prob))
    val_ap = float(average_precision_score(y_val, val_prob))
    test_ap = float(average_precision_score(y_test, test_prob))
    threshold = float(np.quantile(val_prob, args.exit_quantile))
    test_pred = (test_prob >= threshold).astype(np.int64)
    report = classification_report(y_test, test_pred, output_dict=True)

    model_path, meta_path = _resolve_meta_paths(args.save_path)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS, "threshold": threshold}, f)
    artifact = {
        "feature_cols": FEATURE_COLS,
        "model_path": os.path.basename(model_path),
        "threshold": threshold,
        "meta": {
            "algorithm": "unified_exit_hazard_catboost",
            "csv_path": args.csv_path,
            "rule_path": args.rule_path,
            "gate_path": args.gate_path,
            "min_hold_bars": args.min_hold_bars,
            "improve_margin": args.improve_margin,
            "adverse_gap": args.adverse_gap,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "val_ap": val_ap,
            "test_ap": test_ap,
            "classification_report": report,
            "hazard_rows": int(len(haz)),
            "trade_rows": int(len(trades)),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    logger.info("saved model: %s", model_path)
    logger.info("saved meta: %s", meta_path)
    logger.info("hazard_rows=%d trade_rows=%d val_auc=%.4f test_auc=%.4f val_ap=%.4f test_ap=%.4f thr=%.4f", len(haz), len(trades), val_auc, test_auc, val_ap, test_ap, threshold)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train unified exit hazard model on conformal-gated TSFM trades")
    p.add_argument("--csv-path", default=DEFAULT_CSV)
    p.add_argument("--rule-path", default=DEFAULT_RULE)
    p.add_argument("--gate-path", default=DEFAULT_GATE)
    p.add_argument("--save-path", default=DEFAULT_SAVE)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--min-hold-bars", type=int, default=2)
    p.add_argument("--improve-margin", type=float, default=0.0015)
    p.add_argument("--adverse-gap", type=float, default=0.0035)
    p.add_argument("--exit-quantile", type=float, default=0.60)
    p.add_argument("--iterations", type=int, default=700)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--l2-leaf-reg", type=float, default=8.0)
    p.add_argument("--od-wait", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
