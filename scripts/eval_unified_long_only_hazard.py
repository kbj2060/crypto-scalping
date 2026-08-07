from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ensemble.supervised.common import median_fill_by_train, time_split_indices
from ensemble.supervised.unified_stack_utils import build_hazard_rows
from scripts.build_unified_sparse_candidates_v2 import build_candidates
from scripts.eval_unified_long_only_specialists import (
    BacktestResult,
    _apply_specialists,
    _load_frame,
    _load_pickle,
    _predict_direction_probs,
)


DEFAULT_TRAIN_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_dircat_oof.csv"
DEFAULT_TEST_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2026_unified.csv"
DEFAULT_DIR_MODEL = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_direction_catboost.pkl"
DEFAULT_SPECIALISTS = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_regime_specialists.json"
DEFAULT_CONFIG = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_long_only_specialists_eval.json"
DEFAULT_OUT = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_long_only_hazard_eval.json"
FEE = 0.0005
SLIP = 0.0002
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
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
]


def _simulate_trade_rows(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    hold_scale_by_regime: dict[str, float],
    exit_min_hold_bars: int,
    exit_prob_flip_margin: float,
    exit_quality_floor: float | None,
) -> tuple[pd.DataFrame, BacktestResult]:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    cand = pd.to_numeric(df["ud2_cand_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    side = pd.to_numeric(df["ud2_cand_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    hold_arr = pd.to_numeric(df["ud2_cand_hold"], errors="coerce").fillna(6).astype(np.int32).to_numpy()
    regime_arr = df["ud2_cand_regime"].astype(str).to_numpy()
    spec_key = df["ud2_spec_key"].astype(str).to_numpy()
    spec_prob = pd.to_numeric(df["ud2_spec_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    q = pd.to_numeric(df["m7_target_quality"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    lp = pd.to_numeric(df["ud_cat_long_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    sp = pd.to_numeric(df["ud_cat_short_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    raw_side = np.sign(pd.to_numeric(df["m7_action"], errors="coerce").fillna(0.0).to_numpy(np.float64)).astype(np.int8)

    trade_rows: list[dict[str, Any]] = []
    pos = 0
    entry_idx = -1
    entry_fill = 0.0
    hold = 0
    target_hold = 0
    balance = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    wins = 0
    longs = 0

    for i in range(len(df)):
        allowed = False
        if cand[i] == 1 and side[i] == 1 and spec_key[i] in thresholds:
            allowed = spec_prob[i] >= float(thresholds[spec_key[i]])
        if pos == 0:
            if allowed:
                pos = 1
                entry_idx = i
                entry_fill = close[i] * (1.0 + SLIP)
                balance *= (1.0 - FEE)
                hold = 0
                target_hold = max(2, int(round(float(hold_arr[i]) * float(hold_scale_by_regime.get(regime_arr[i], 0.8)))))
                longs += 1
        else:
            hold += 1
            prob_flip = sp[i] >= (lp[i] + float(exit_prob_flip_margin))
            raw_flip = raw_side[i] < 0
            quality_break = exit_quality_floor is not None and q[i] <= float(exit_quality_floor)
            early_exit = hold >= int(exit_min_hold_bars) and (prob_flip or raw_flip or quality_break)
            if early_exit or hold >= target_hold:
                fill = close[i] * (1.0 - SLIP)
                pnl = (fill - entry_fill) / max(entry_fill, 1e-8)
                balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
                trades += 1
                wins += int(pnl > 0)
                trade_rows.append(
                    {
                        "entry_idx": int(entry_idx),
                        "exit_idx": int(i),
                        "side": 1,
                        "entry_fill": float(entry_fill),
                        "exit_fill": float(fill),
                        "target_hold": int(target_hold),
                        "pnl": float(pnl),
                    }
                )
                pos = 0
                entry_idx = -1
                entry_fill = 0.0
                hold = 0
                target_hold = 0
        peak = max(peak, balance)
        mdd = min(mdd, balance / max(peak, 1e-8) - 1.0)

    if pos != 0:
        fill = close[-1] * (1.0 - SLIP)
        pnl = (fill - entry_fill) / max(entry_fill, 1e-8)
        balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
        trades += 1
        wins += int(pnl > 0)
        trade_rows.append(
            {
                "entry_idx": int(entry_idx),
                "exit_idx": int(len(df) - 1),
                "side": 1,
                "entry_fill": float(entry_fill),
                "exit_fill": float(fill),
                "target_hold": int(target_hold),
                "pnl": float(pnl),
            }
        )

    return pd.DataFrame(trade_rows), BacktestResult(
        pnl_pct=float((balance - 1.0) * 100.0),
        trades=int(trades),
        wr_pct=float(wins / max(trades, 1) * 100.0),
        mdd_pct=float(mdd * 100.0),
        longs=int(longs),
        shorts=0,
    )


def _run_with_hazard(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    hold_scale_by_regime: dict[str, float],
    exit_min_hold_bars: int,
    exit_prob_flip_margin: float,
    exit_quality_floor: float | None,
    hazard_payload: dict[str, Any] | None,
    hazard_threshold: float,
) -> BacktestResult:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    cand = pd.to_numeric(df["ud2_cand_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    side = pd.to_numeric(df["ud2_cand_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    hold_arr = pd.to_numeric(df["ud2_cand_hold"], errors="coerce").fillna(6).astype(np.int32).to_numpy()
    regime_arr = df["ud2_cand_regime"].astype(str).to_numpy()
    spec_key = df["ud2_spec_key"].astype(str).to_numpy()
    spec_prob = pd.to_numeric(df["ud2_spec_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    q = pd.to_numeric(df["m7_target_quality"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    lp = pd.to_numeric(df["ud_cat_long_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    sp = pd.to_numeric(df["ud_cat_short_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    fp = pd.to_numeric(df["ud_cat_flat_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    raw_side = np.sign(pd.to_numeric(df["m7_action"], errors="coerce").fillna(0.0).to_numpy(np.float64)).astype(np.int8)

    hazard_model = None
    hazard_features: list[str] = []
    if hazard_payload is not None:
        hazard_model = hazard_payload["model"]
        hazard_features = list(hazard_payload["feature_cols"])

    pos = 0
    entry_idx = -1
    entry_fill = 0.0
    hold = 0
    target_hold = 0
    balance = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    wins = 0
    longs = 0

    for i in range(len(df)):
        allowed = False
        if cand[i] == 1 and side[i] == 1 and spec_key[i] in thresholds:
            allowed = spec_prob[i] >= float(thresholds[spec_key[i]])
        if pos == 0:
            if allowed:
                pos = 1
                entry_idx = i
                entry_fill = close[i] * (1.0 + SLIP)
                balance *= (1.0 - FEE)
                hold = 0
                target_hold = max(2, int(round(float(hold_arr[i]) * float(hold_scale_by_regime.get(regime_arr[i], 0.8)))))
                longs += 1
        else:
            hold += 1
            prob_flip = sp[i] >= (lp[i] + float(exit_prob_flip_margin))
            raw_flip = raw_side[i] < 0
            quality_break = exit_quality_floor is not None and q[i] <= float(exit_quality_floor)
            early_exit = hold >= int(exit_min_hold_bars) and (prob_flip or raw_flip or quality_break)
            hazard_exit = False
            if hazard_model is not None and hold >= int(exit_min_hold_bars):
                cur_fill = close[i] * (1.0 - SLIP)
                cur_pnl = (cur_fill - entry_fill) / max(entry_fill, 1e-8)
                sofar = (close[entry_idx : i + 1] * (1.0 - SLIP) - entry_fill) / max(entry_fill, 1e-8)
                row = {
                    "haz_side": 1,
                    "haz_bars_held_norm": float(hold / max(target_hold, 1)),
                    "haz_remaining_norm": float(max(target_hold - hold, 0) / max(target_hold, 1)),
                    "haz_current_pnl": float(cur_pnl),
                    "haz_mfe_sofar": float(np.max(sofar)),
                    "haz_mae_sofar": float(np.min(sofar)),
                    "haz_long_prob": float(lp[i]),
                    "haz_flat_prob": float(fp[i]),
                    "haz_short_prob": float(sp[i]),
                    "haz_prob_max": float(max(lp[i], fp[i], sp[i])),
                    "m7_target_quality": float(q[i]),
                    "smart_money_flow": float(pd.to_numeric(df.iloc[i]["smart_money_flow"], errors="coerce")),
                    "taker_acceleration": float(pd.to_numeric(df.iloc[i]["taker_acceleration"], errors="coerce")),
                    "trade_intensity": float(pd.to_numeric(df.iloc[i]["trade_intensity"], errors="coerce")),
                    "garch_vol_z": float(pd.to_numeric(df.iloc[i]["garch_vol_z"], errors="coerce")),
                    "rogers_satchell_vol": float(pd.to_numeric(df.iloc[i]["rogers_satchell_vol"], errors="coerce")),
                    "amihud_illiquidity_z": float(pd.to_numeric(df.iloc[i]["amihud_illiquidity_z"], errors="coerce")),
                    "regime_bull": float(pd.to_numeric(df.iloc[i]["regime_bull"], errors="coerce")),
                    "regime_bear": float(pd.to_numeric(df.iloc[i]["regime_bear"], errors="coerce")),
                    "regime_chop": float(pd.to_numeric(df.iloc[i]["regime_chop"], errors="coerce")),
                    "regime_whipsaw": float(pd.to_numeric(df.iloc[i]["regime_whipsaw"], errors="coerce")),
                    "regime_normal": float(pd.to_numeric(df.iloc[i]["regime_normal"], errors="coerce")),
                }
                x = pd.DataFrame([{k: row.get(k, 0.0) for k in hazard_features}])
                prob = float(hazard_model.predict_proba(x)[0, 1])
                hazard_exit = prob >= float(hazard_threshold)
            if early_exit or hold >= target_hold or hazard_exit:
                fill = close[i] * (1.0 - SLIP)
                pnl = (fill - entry_fill) / max(entry_fill, 1e-8)
                balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
                trades += 1
                wins += int(pnl > 0)
                pos = 0
                entry_idx = -1
                entry_fill = 0.0
                hold = 0
                target_hold = 0
        peak = max(peak, balance)
        mdd = min(mdd, balance / max(peak, 1e-8) - 1.0)

    if pos != 0:
        fill = close[-1] * (1.0 - SLIP)
        pnl = (fill - entry_fill) / max(entry_fill, 1e-8)
        balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
        trades += 1
        wins += int(pnl > 0)

    return BacktestResult(
        pnl_pct=float((balance - 1.0) * 100.0),
        trades=int(trades),
        wr_pct=float(wins / max(trades, 1) * 100.0),
        mdd_pct=float(mdd * 100.0),
        longs=int(longs),
        shorts=0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate long-only hazard exit on top of regime specialists")
    ap.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    ap.add_argument("--dir-model", default=DEFAULT_DIR_MODEL)
    ap.add_argument("--specialists", default=DEFAULT_SPECIALISTS)
    ap.add_argument("--config-path", default=DEFAULT_CONFIG)
    ap.add_argument("--output-path", default=DEFAULT_OUT)
    args = ap.parse_args()

    train_df = _load_frame(args.train_csv)
    test_df = _load_frame(args.test_csv)
    dir_payload = _load_pickle(args.dir_model)
    with open(args.specialists, "r", encoding="utf-8") as f:
        spec_meta = json.load(f)
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)["best"]
    model_root = os.path.dirname(args.specialists)

    train_scored = build_candidates(train_df, config["candidate_params"])
    train_scored = _apply_specialists(train_scored, spec_meta, model_root)
    calib = train_scored[train_scored["ud_cat_is_holdout"] == 0].copy().reset_index(drop=True)
    holdout = train_scored[train_scored["ud_cat_is_holdout"] == 1].copy().reset_index(drop=True)
    test_scored = _predict_direction_probs(test_df, train_df, dir_payload)
    test_scored = build_candidates(test_scored, config["candidate_params"])
    test_scored = _apply_specialists(test_scored, spec_meta, model_root)

    trade_rows, calib_base = _simulate_trade_rows(
        calib,
        thresholds=config["thresholds"],
        hold_scale_by_regime=config["hold_scale_by_regime"],
        exit_min_hold_bars=int(config["exit_min_hold_bars"]),
        exit_prob_flip_margin=float(config["exit_prob_flip_margin"]),
        exit_quality_floor=config["exit_quality_floor"],
    )
    haz = build_hazard_rows(
        calib,
        trade_rows,
        prob_cols={"long": "ud_cat_long_prob", "flat": "ud_cat_flat_prob", "short": "ud_cat_short_prob"},
        min_hold_bars=2,
        improve_margin=0.0015,
        adverse_gap=0.0030,
    )
    haz = haz.replace([np.inf, -np.inf], np.nan).copy()
    if haz.empty:
        raise RuntimeError("no hazard rows generated")
    x = haz.loc[:, FEATURE_COLS].copy()
    y = pd.to_numeric(haz["haz_exit_label"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    tr_idx, va_idx, te_idx = time_split_indices(len(haz), 0.70, 0.15)
    x_train = x.iloc[tr_idx].copy()
    x_val = x.iloc[va_idx].copy()
    x_test = x.iloc[te_idx].copy()
    x_train, x_val = median_fill_by_train(x_train, x_val)
    x_train, x_test = median_fill_by_train(x_train, x_test)
    y_train, y_val, y_test = y[tr_idx], y[va_idx], y[te_idx]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=500,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=8.0,
        random_seed=42,
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=40,
        verbose=False,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    hazard_payload = {"model": model, "feature_cols": FEATURE_COLS}

    thr_rows = []
    for hazard_thr in (0.55, 0.60, 0.65, 0.70):
        res = _run_with_hazard(
            holdout,
            thresholds=config["thresholds"],
            hold_scale_by_regime=config["hold_scale_by_regime"],
            exit_min_hold_bars=int(config["exit_min_hold_bars"]),
            exit_prob_flip_margin=float(config["exit_prob_flip_margin"]),
            exit_quality_floor=config["exit_quality_floor"],
            hazard_payload=hazard_payload,
            hazard_threshold=float(hazard_thr),
        )
        score = float(res.pnl_pct) - 0.25 * abs(min(float(res.mdd_pct), 0.0))
        thr_rows.append({"hazard_threshold": float(hazard_thr), "holdout_2025": asdict(res), "score": score})
    best = max(thr_rows, key=lambda x: x["score"])

    holdout_res = _run_with_hazard(
        holdout,
        thresholds=config["thresholds"],
        hold_scale_by_regime=config["hold_scale_by_regime"],
        exit_min_hold_bars=int(config["exit_min_hold_bars"]),
        exit_prob_flip_margin=float(config["exit_prob_flip_margin"]),
        exit_quality_floor=config["exit_quality_floor"],
        hazard_payload=hazard_payload,
        hazard_threshold=float(best["hazard_threshold"]),
    )
    oos_res = _run_with_hazard(
        test_scored,
        thresholds=config["thresholds"],
        hold_scale_by_regime=config["hold_scale_by_regime"],
        exit_min_hold_bars=int(config["exit_min_hold_bars"]),
        exit_prob_flip_margin=float(config["exit_prob_flip_margin"]),
        exit_quality_floor=config["exit_quality_floor"],
        hazard_payload=hazard_payload,
        hazard_threshold=float(best["hazard_threshold"]),
    )

    out = {
        "config_path": args.config_path,
        "base_calibration_result": asdict(calib_base),
        "hazard_rows": int(len(haz)),
        "val_auc": float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else None,
        "val_ap": float(average_precision_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else None,
        "test_auc": float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else None,
        "test_ap": float(average_precision_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else None,
        "threshold_search": thr_rows,
        "best_threshold": best,
        "holdout_2025": asdict(holdout_res),
        "oos_2026": asdict(oos_res),
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
