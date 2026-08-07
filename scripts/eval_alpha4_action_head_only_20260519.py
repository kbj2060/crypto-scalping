#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402
from scripts import eval_alpha4_new_features_full_retrain_20260517 as alpha4  # noqa: E402
from scripts.train_eval_alpha5_5_lgbm_supervised_parent_20260518 import (  # noqa: E402
    _backtest_actions,
    _compact,
    _decide_actions,
    _predict_proba_3,
    _score,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha4_action_head_only_20260519"
DEFAULT_ROOT = ROOT / "tmp/causal_regen_20260516/alpha4_new_features_full_retrain_20260517"
DEFAULT_TRAIN = ROOT / "tmp/causal_regen_20260516/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/causal_regen_20260516/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_PARENT = DEFAULT_ROOT / "artifacts/hgb/parent.pkl"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha4_action_head_only_20260519"


def _metrics(frame: pd.DataFrame, proba: np.ndarray, *, prob: float, margin: float, fee: float, slip: float, exposure: float, max_hold: int) -> dict[str, Any]:
    actions = _decide_actions(proba, prob, margin)
    return {
        "actions": actions,
        "cost1": _backtest_actions(frame, actions, fee=fee, slip=slip, unit_exposure=exposure, max_hold_bars=max_hold),
        "cost2": _backtest_actions(frame, actions, fee=fee * 2.0, slip=slip * 2.0, unit_exposure=exposure, max_hold_bars=max_hold),
        "cost3": _backtest_actions(frame, actions, fee=fee * 3.0, slip=slip * 3.0, unit_exposure=exposure, max_hold_bars=max_hold),
    }


def _predict_parent_action_proba(parent: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    feat_cols = list(parent["feature_cols"])
    x = prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=feat_cols)
    return _predict_proba_3(parent["action_model"], x)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Alpha4 HGB parent action head only.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--prob-thresholds", default="0.34,0.38,0.42,0.46,0.50,0.55,0.60")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12,0.16,0.20")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--max-hold-bars", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    # Historical artifacts may resolve wrapper classes via this module namespace.
    import __main__  # noqa: PLC0415

    __main__.FillNAWrapper = alpha4.FillNAWrapper
    __main__.EncodedClassifierWrapper = alpha4.EncodedClassifierWrapper
    __main__.SoftVotingClassifierWrapper = alpha4.SoftVotingClassifierWrapper
    __main__.MeanRegressorWrapper = alpha4.MeanRegressorWrapper

    parent = joblib.load(args.parent_model)
    proba_val = _predict_parent_action_proba(parent, val_df)
    proba_oos = _predict_parent_action_proba(parent, eval_df)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "parent_model": str(args.parent_model),
        "feature_count": int(len(parent["feature_cols"])),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "oos_rows": int(len(eval_df)),
        "max_hold_bars": int(args.max_hold_bars),
    }, ensure_ascii=False, default=_json_default), flush=True)

    prob_grid = [float(x.strip()) for x in str(args.prob_thresholds).split(",") if x.strip()]
    margin_grid = [float(x.strip()) for x in str(args.margin_thresholds).split(",") if x.strip()]
    for prob in prob_grid:
        for margin in margin_grid:
            val = _metrics(val_df, proba_val, prob=prob, margin=margin, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
            oos = _metrics(eval_df, proba_oos, prob=prob, margin=margin, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
            score = _score(val["cost1"], val["cost2"], val["cost3"])
            row = {
                "prob_threshold": float(prob),
                "margin_threshold": float(margin),
                "val_score": float(score),
                "val": _compact(val),
                "oos": _compact(oos),
            }
            rows.append(row)
            if best is None or float(row["val_score"]) > float(best["val_score"]):
                best = copy.deepcopy(row)
    assert best is not None

    summary = {
        "model_id": MODEL_ID,
        "parent_model": str(args.parent_model),
        "design": "Alpha4 HGB parent action_model only, action-only lifecycle backtest with flat/flip exits.",
        "max_hold_bars": int(args.max_hold_bars),
        "grid_size": int(len(rows)),
        "best": best,
        "top10": sorted(rows, key=lambda r: float(r["val_score"]), reverse=True)[:10],
    }
    summary_path = args.out_dir / "alpha4_action_head_only_summary.json"
    grid_path = args.out_dir / "alpha4_action_head_only_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([
        {
            "prob_threshold": r["prob_threshold"],
            "margin_threshold": r["margin_threshold"],
            "val_score": r["val_score"],
            "val_cost1_pnl": r["val"]["cost1"]["pnl"],
            "val_cost1_mdd": r["val"]["cost1"]["mdd"],
            "val_cost1_trades": r["val"]["cost1"]["trades"],
            "oos_cost1_pnl": r["oos"]["cost1"]["pnl"],
            "oos_cost1_mdd": r["oos"]["cost1"]["mdd"],
            "oos_cost1_trades": r["oos"]["cost1"]["trades"],
        }
        for r in rows
    ]).sort_values("val_score", ascending=False).to_csv(grid_path, index=False)
    print(json.dumps({
        "stage": "complete",
        "summary": str(summary_path),
        "grid": str(grid_path),
        "best": best,
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
