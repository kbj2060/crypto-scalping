from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ensemble.supervised.unified_stack_utils import apply_regime_veto, build_sparse_candidates, run_backtest_with_hazard

DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_dircat_tsfm_oof.csv"
DEFAULT_RULE = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_shadow_rule.json"
DEFAULT_OUT_JSON = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_tsfm_conformal_gate.json"


def _label_prob(row: pd.Series) -> float:
    label = int(row["ud_tsfm_tb_label"])
    if label == 0:
        return float(row["ud_tsfm_short_prob"])
    if label == 1:
        return float(row["ud_tsfm_flat_prob"])
    return float(row["ud_tsfm_long_prob"])


def _pred_prob(row: pd.Series) -> float:
    pred = int(row["ud_tsfm_pred_class"])
    if pred == 0:
        return float(row["ud_tsfm_short_prob"])
    if pred == 1:
        return float(row["ud_tsfm_flat_prob"])
    return float(row["ud_tsfm_long_prob"])


def _quantile_higher(arr: np.ndarray, q: float) -> float:
    if len(arr) == 0:
        return 1.0
    q = min(max(q, 0.0), 1.0)
    idx = int(math.ceil(q * len(arr)) - 1)
    idx = min(max(idx, 0), len(arr) - 1)
    arr2 = np.sort(arr)
    return float(arr2[idx])


def _compute_thresholds(df: pd.DataFrame, mode: str, alpha: float) -> dict[str, float]:
    work = df.copy()
    work["ud_conf_nonconf"] = 1.0 - work.apply(_label_prob, axis=1)
    if mode == "global":
        qhat = _quantile_higher(work["ud_conf_nonconf"].to_numpy(np.float64), 1.0 - alpha)
        return {"global": 1.0 - qhat}
    thresholds: dict[str, float] = {}
    for regime, sub in work.groupby("ud_stack_regime"):
        qhat = _quantile_higher(sub["ud_conf_nonconf"].to_numpy(np.float64), 1.0 - alpha)
        thresholds[str(regime)] = 1.0 - qhat
    global_qhat = _quantile_higher(work["ud_conf_nonconf"].to_numpy(np.float64), 1.0 - alpha)
    thresholds["_fallback"] = 1.0 - global_qhat
    return thresholds


def _accept_mask(df: pd.DataFrame, thresholds: dict[str, float], mode: str) -> np.ndarray:
    pred_prob = df.apply(_pred_prob, axis=1).to_numpy(np.float64)
    if mode == "global":
        thr = float(thresholds["global"])
        return pred_prob >= thr
    regimes = df["ud_stack_regime"].astype(str).to_numpy()
    fallback = float(thresholds.get("_fallback", 1.0))
    out = np.zeros(len(df), dtype=bool)
    for i, rg in enumerate(regimes):
        out[i] = pred_prob[i] >= float(thresholds.get(rg, fallback))
    return out


def _score(res: dict[str, Any]) -> float:
    s = float(res["pnl_pct"]) - 0.35 * abs(min(float(res["mdd_pct"]), 0.0))
    if int(res["trades"]) < 8:
        s -= 8.0
    if int(res["trades"]) > 120:
        s -= 0.05 * (int(res["trades"]) - 120)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Build conformal/selective gate for TSFM direction candidates")
    ap.add_argument("--csv-path", default=DEFAULT_CSV)
    ap.add_argument("--rule-path", default=DEFAULT_RULE)
    ap.add_argument("--output-json", default=DEFAULT_OUT_JSON)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    with open(args.rule_path, "r", encoding="utf-8") as f:
        rule_cfg = json.load(f)

    full = build_sparse_candidates(
        df,
        long_col="ud_tsfm_long_prob",
        flat_col="ud_tsfm_flat_prob",
        short_col="ud_tsfm_short_prob",
        params=rule_cfg["candidate_params"],
        prefix="ud_stack",
    )
    veto_mask = apply_regime_veto(full, rule_cfg["veto_rule"], prefix="ud_stack")

    calib = full[full["ud_tsfm_is_holdout"] == 0].copy().reset_index(drop=True)
    calib_veto = veto_mask[full["ud_tsfm_is_holdout"].to_numpy(np.int8) == 0]
    fold_ids = sorted(int(x) for x in pd.to_numeric(calib["ud_tsfm_oof_fold"], errors="coerce").dropna().unique() if x >= 0)
    recent_folds = fold_ids[-4:]
    windows: list[tuple[int, int]] = []
    fold_col = pd.to_numeric(calib["ud_tsfm_oof_fold"], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
    for fid in recent_folds:
        idx = np.flatnonzero(fold_col == fid)
        if len(idx):
            windows.append((int(idx[0]), int(idx[-1] + 1)))

    candidate_calib = calib[(pd.to_numeric(calib["ud_stack_flag"], errors="coerce").fillna(0).astype(np.int8) == 1) & calib_veto].copy()
    best: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for mode in ("global", "regime"):
        for alpha in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            thresholds = _compute_thresholds(candidate_calib, mode=mode, alpha=float(alpha))
            accept = np.zeros(len(calib), dtype=bool)
            accept_sub = _accept_mask(candidate_calib, thresholds, mode=mode)
            accept[np.flatnonzero((pd.to_numeric(calib["ud_stack_flag"], errors="coerce").fillna(0).astype(np.int8) == 1) & calib_veto)] = accept_sub

            window_res = []
            scores = []
            for ws, we in windows:
                res = asdict(
                    run_backtest_with_hazard(
                        calib.iloc[ws:we].reset_index(drop=True),
                        take_mask=accept[ws:we],
                        prefix="ud_stack",
                        hold_scale=float(rule_cfg["hold_scale"]),
                        close_on_opp=bool(rule_cfg["close_on_opp"]),
                    )
                )
                sc = _score(res)
                scores.append(sc)
                window_res.append({"start": ws, "end": we, "result": res, "score": sc})
            row = {
                "mode": mode,
                "alpha": float(alpha),
                "thresholds": thresholds,
                "avg_score": float(np.mean(scores)),
                "avg_pnl_pct": float(np.mean([w["result"]["pnl_pct"] for w in window_res])),
                "avg_trades": float(np.mean([w["result"]["trades"] for w in window_res])),
                "windows": window_res,
            }
            rows.append(row)
            if best is None or row["avg_score"] > best["avg_score"]:
                best = row

    assert best is not None
    holdout = full[full["ud_tsfm_is_holdout"] == 1].copy().reset_index(drop=True)
    holdout_veto = veto_mask[full["ud_tsfm_is_holdout"].to_numpy(np.int8) == 1]
    holdout_accept = np.zeros(len(holdout), dtype=bool)
    holdout_candidates = holdout[(pd.to_numeric(holdout["ud_stack_flag"], errors="coerce").fillna(0).astype(np.int8) == 1) & holdout_veto].copy()
    if len(holdout_candidates):
        holdout_accept_sub = _accept_mask(holdout_candidates, best["thresholds"], mode=str(best["mode"]))
        holdout_accept[np.flatnonzero((pd.to_numeric(holdout["ud_stack_flag"], errors="coerce").fillna(0).astype(np.int8) == 1) & holdout_veto)] = holdout_accept_sub
    best["holdout"] = asdict(
        run_backtest_with_hazard(
            holdout,
            take_mask=holdout_accept,
            prefix="ud_stack",
            hold_scale=float(rule_cfg["hold_scale"]),
            close_on_opp=bool(rule_cfg["close_on_opp"]),
        )
    )
    best["holdout_take_rows"] = int(holdout_accept.sum())

    out = {
        "csv_path": args.csv_path,
        "rule_path": args.rule_path,
        "candidate_params": rule_cfg["candidate_params"],
        "veto_rule": rule_cfg["veto_rule"],
        "hold_scale": rule_cfg["hold_scale"],
        "close_on_opp": rule_cfg["close_on_opp"],
        "best": best,
        "top10": sorted(rows, key=lambda x: x["avg_score"], reverse=True)[:10],
        "windows": windows,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
