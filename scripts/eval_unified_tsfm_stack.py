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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ensemble.supervised.unified_stack_utils import (
    apply_regime_veto,
    build_sparse_candidates,
    run_backtest_with_hazard,
)

DEFAULT_TRAIN_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified.csv"
DEFAULT_OOF_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_dircat_tsfm_oof.csv"
DEFAULT_TEST_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2026_unified.csv"
DEFAULT_DIRECTION = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_direction_catboost_tsfm.pkl"
DEFAULT_RULE = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_shadow_rule.json"
DEFAULT_GATE = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_tsfm_conformal_gate.json"
DEFAULT_HAZARD = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_exit_hazard_catboost.pkl"
DEFAULT_OUT = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_tsfm_stack_eval.json"


def _load_pickle(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def _predict_direction_probs(test_df: pd.DataFrame, train_df: pd.DataFrame, model_payload: dict[str, Any]) -> pd.DataFrame:
    feature_cols = list(model_payload["feature_cols"])
    model = model_payload["model"]
    missing_train = [c for c in feature_cols if c not in train_df.columns]
    if missing_train:
        raise ValueError(f"missing features in train frame for direction model: {missing_train}")

    x_train = train_df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    x_test = test_df.copy()
    for c in feature_cols:
        if c not in x_test.columns:
            x_test[c] = np.nan
    x_test = x_test.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    med = x_train.median(numeric_only=True)
    x_train = x_train.fillna(med).fillna(0.0)
    x_test = x_test.fillna(med).fillna(0.0)

    probs = model.predict_proba(x_test)
    out = test_df.copy()
    out["ud_tsfm_short_prob"] = probs[:, 0]
    out["ud_tsfm_flat_prob"] = probs[:, 1]
    out["ud_tsfm_long_prob"] = probs[:, 2]
    out["ud_tsfm_edge"] = out["ud_tsfm_long_prob"] - out["ud_tsfm_short_prob"]
    out["ud_tsfm_prob_max"] = np.max(probs, axis=1)
    out["ud_tsfm_pred_class"] = np.argmax(probs, axis=1)
    return out


def _apply_gate(df: pd.DataFrame, gate_cfg: dict[str, Any], veto_mask: np.ndarray) -> np.ndarray:
    thresholds = dict(gate_cfg["best"]["thresholds"])
    mode = str(gate_cfg["best"]["mode"])
    pred_prob = (
        df[["ud_tsfm_long_prob", "ud_tsfm_flat_prob", "ud_tsfm_short_prob"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .max(axis=1)
        .to_numpy(np.float64)
    )
    out = np.zeros(len(df), dtype=bool)
    if mode == "global":
        thr = float(thresholds["global"])
        out = pred_prob >= thr
    else:
        fallback = float(thresholds.get("_fallback", 1.0))
        regimes = df["ud_stack_regime"].astype(str).to_numpy()
        for i, rg in enumerate(regimes):
            out[i] = pred_prob[i] >= float(thresholds.get(rg, fallback))
    return out & veto_mask & (pd.to_numeric(df["ud_stack_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1)


def _score(res: dict[str, Any]) -> float:
    return float(res["pnl_pct"]) - 0.35 * abs(min(float(res["mdd_pct"]), 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate TSFM direction + conformal gate + exit hazard stack")
    ap.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--oof-csv", default=DEFAULT_OOF_CSV)
    ap.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    ap.add_argument("--direction-path", default=DEFAULT_DIRECTION)
    ap.add_argument("--rule-path", default=DEFAULT_RULE)
    ap.add_argument("--gate-path", default=DEFAULT_GATE)
    ap.add_argument("--hazard-path", default=DEFAULT_HAZARD)
    ap.add_argument("--output-path", default=DEFAULT_OUT)
    args = ap.parse_args()

    train_df = _load_frame(args.train_csv)
    oof_df = _load_frame(args.oof_csv)
    test_df = _load_frame(args.test_csv)
    direction_payload = _load_pickle(args.direction_path)
    hazard_payload = _load_pickle(args.hazard_path)
    rule_cfg = _load_json(args.rule_path)
    gate_cfg = _load_json(args.gate_path)

    calib = oof_df[oof_df["ud_tsfm_is_holdout"] == 0].copy().reset_index(drop=True)
    holdout = oof_df[oof_df["ud_tsfm_is_holdout"] == 1].copy().reset_index(drop=True)
    calib = build_sparse_candidates(
        calib,
        long_col="ud_tsfm_long_prob",
        flat_col="ud_tsfm_flat_prob",
        short_col="ud_tsfm_short_prob",
        params=rule_cfg["candidate_params"],
        prefix="ud_stack",
    )
    holdout = build_sparse_candidates(
        holdout,
        long_col="ud_tsfm_long_prob",
        flat_col="ud_tsfm_flat_prob",
        short_col="ud_tsfm_short_prob",
        params=rule_cfg["candidate_params"],
        prefix="ud_stack",
    )
    calib_veto = apply_regime_veto(calib, rule_cfg["veto_rule"], prefix="ud_stack")
    holdout_veto = apply_regime_veto(holdout, rule_cfg["veto_rule"], prefix="ud_stack")
    calib_take = _apply_gate(calib, gate_cfg, calib_veto)
    holdout_take = _apply_gate(holdout, gate_cfg, holdout_veto)

    fold_ids = sorted(int(x) for x in pd.to_numeric(calib["ud_tsfm_oof_fold"], errors="coerce").dropna().unique() if x >= 0)
    recent_folds = fold_ids[-4:]
    windows: list[tuple[int, int]] = []
    fold_col = pd.to_numeric(calib["ud_tsfm_oof_fold"], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
    for fid in recent_folds:
        idx = np.flatnonzero(fold_col == fid)
        if len(idx):
            windows.append((int(idx[0]), int(idx[-1] + 1)))

    best: dict[str, Any] | None = None
    for hazard_thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        for min_hold in (2, 3, 4):
            window_res = []
            scores = []
            for ws, we in windows:
                res = asdict(
                    run_backtest_with_hazard(
                        calib.iloc[ws:we].reset_index(drop=True),
                        take_mask=calib_take[ws:we],
                        prefix="ud_stack",
                        hold_scale=float(rule_cfg["hold_scale"]),
                        close_on_opp=bool(rule_cfg["close_on_opp"]),
                        hazard_payload=hazard_payload,
                        hazard_threshold=float(hazard_thr),
                        min_hold_bars=int(min_hold),
                    )
                )
                sc = _score(res)
                scores.append(sc)
                window_res.append({"start": ws, "end": we, "result": res, "score": sc})
            row = {
                "hazard_threshold": float(hazard_thr),
                "min_hold_bars": int(min_hold),
                "avg_score": float(np.mean(scores)),
                "avg_pnl_pct": float(np.mean([w["result"]["pnl_pct"] for w in window_res])),
                "avg_trades": float(np.mean([w["result"]["trades"] for w in window_res])),
                "windows": window_res,
            }
            if best is None or row["avg_score"] > best["avg_score"]:
                best = row

    assert best is not None
    holdout_res = asdict(
        run_backtest_with_hazard(
            holdout,
            take_mask=holdout_take,
            prefix="ud_stack",
            hold_scale=float(rule_cfg["hold_scale"]),
            close_on_opp=bool(rule_cfg["close_on_opp"]),
            hazard_payload=hazard_payload,
            hazard_threshold=float(best["hazard_threshold"]),
            min_hold_bars=int(best["min_hold_bars"]),
        )
    )

    scored_2026 = _predict_direction_probs(test_df, train_df, direction_payload)
    scored_2026 = build_sparse_candidates(
        scored_2026,
        long_col="ud_tsfm_long_prob",
        flat_col="ud_tsfm_flat_prob",
        short_col="ud_tsfm_short_prob",
        params=rule_cfg["candidate_params"],
        prefix="ud_stack",
    )
    test_veto = apply_regime_veto(scored_2026, rule_cfg["veto_rule"], prefix="ud_stack")
    test_take = _apply_gate(scored_2026, gate_cfg, test_veto)
    test_res = asdict(
        run_backtest_with_hazard(
            scored_2026,
            take_mask=test_take,
            prefix="ud_stack",
            hold_scale=float(rule_cfg["hold_scale"]),
            close_on_opp=bool(rule_cfg["close_on_opp"]),
            hazard_payload=hazard_payload,
            hazard_threshold=float(best["hazard_threshold"]),
            min_hold_bars=int(best["min_hold_bars"]),
        )
    )

    out = {
        "train_csv": args.train_csv,
        "oof_csv": args.oof_csv,
        "test_csv": args.test_csv,
        "direction_path": args.direction_path,
        "rule_path": args.rule_path,
        "gate_path": args.gate_path,
        "hazard_path": args.hazard_path,
        "candidate_params": rule_cfg["candidate_params"],
        "veto_rule": rule_cfg["veto_rule"],
        "hold_scale": rule_cfg["hold_scale"],
        "close_on_opp": rule_cfg["close_on_opp"],
        "hazard_choice": best,
        "holdout_2025": {
            "candidate_rows": int(pd.to_numeric(holdout["ud_stack_flag"], errors="coerce").fillna(0).sum()),
            "take_rows": int(holdout_take.sum()),
            "backtest": holdout_res,
        },
        "oos_2026": {
            "candidate_rows": int(pd.to_numeric(scored_2026["ud_stack_flag"], errors="coerce").fillna(0).sum()),
            "take_rows": int(test_take.sum()),
            "backtest": test_res,
        },
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
