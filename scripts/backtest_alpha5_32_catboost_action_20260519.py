#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_alpha5_23_hgb_direction_refined_20260519 import _eval, _x  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha5_32_catboost_action_backtest_20260519"
DEFAULT_MODEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_32_catboost_contracts_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable005_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_32_catboost_action_backtest_20260519"


def _load_classifier(path: Path, multiclass: bool) -> Any:
    model = CatBoostClassifier()
    model.load_model(str(path))
    meta = joblib.load(path.parent / (path.stem.replace("_catboost_gpu", "") + "_meta.joblib"))
    return model, meta["feature_cols"]


def _load_regressor(path: Path) -> Any:
    model = CatBoostRegressor()
    model.load_model(str(path))
    meta = joblib.load(path.parent / (path.stem.replace("_catboost_gpu", "") + "_meta.joblib"))
    return model, meta["feature_cols"]


def _entry_trade_prob(model: Any, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(model.predict_proba(x), dtype=np.float64)
    pred = np.asarray(model.predict(x)).reshape(-1).astype(np.int64)
    trade_prob = p[:, 2] if p.shape[1] >= 3 else p[:, -1]
    non_trade_best = np.max(p[:, :2], axis=1) if p.shape[1] >= 2 else np.zeros(len(p), dtype=np.float64)
    return pred, np.clip(trade_prob - non_trade_best, -1.0, 1.0)


def _direction_long_prob(model: Any, x: pd.DataFrame) -> np.ndarray:
    p = np.asarray(model.predict_proba(x), dtype=np.float64)
    if p.shape[1] == 1:
        return np.clip(p.reshape(-1), 0.0, 1.0)
    return np.clip(p[:, 1], 0.0, 1.0)


def _compose(
    frame: pd.DataFrame,
    entry_pred: np.ndarray,
    trade_margin: np.ndarray,
    p_long: np.ndarray,
    q_pred: np.ndarray,
    *,
    require_trade_argmax: bool,
    trade_margin_min: float,
    side_threshold: float,
    side_margin_min: float,
    quality_min: float,
    score_abs_min: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    regime = frame["regime4_state"].astype(str).to_numpy()
    score_abs = pd.to_numeric(frame.get("current_direction_score", 0.0), errors="coerce").fillna(0.0).abs().to_numpy(np.float64)
    p_long = np.clip(p_long, 0.0, 1.0)
    p_short = 1.0 - p_long
    side_margin = np.abs(p_long - p_short)
    best_side = np.maximum(p_long, p_short)
    actions = np.where(p_long >= p_short, 1, 2).astype(np.int64)
    if require_trade_argmax:
        actions = np.where(entry_pred == 2, actions, 0)
    actions = np.where(trade_margin >= float(trade_margin_min), actions, 0)
    actions = np.where(best_side >= float(side_threshold), actions, 0)
    actions = np.where(side_margin >= float(side_margin_min), actions, 0)
    actions = np.where(q_pred >= float(quality_min), actions, 0)
    actions = np.where(score_abs >= float(score_abs_min), actions, 0)
    actions = np.where(regime == "whipsaw", 0, actions)
    return actions, {
        "trade_margin": trade_margin,
        "p_long": p_long,
        "p_short": p_short,
        "side_margin": side_margin,
        "best_side": best_side,
        "q_pred": q_pred,
        "score_abs": score_abs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest alpha5_32 CatBoost 3-head action composition.")
    p.add_argument(
        "--allow-deprecated-action-model",
        action="store_true",
        help="Allow historical reproduction of the deprecated CatBoost direct-action backtest.",
    )
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--max-hold-bars", type=int, default=96)
    args = p.parse_args()
    if not bool(args.allow_deprecated_action_model):
        p.error(
            "CatBoost direct-action backtests are deprecated and not allowed in active paths. "
            "Use Router5 a5dir_* as auxiliary DSAC features, or pass "
            "--allow-deprecated-action-model for historical reproduction."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_df = pd.read_parquet(args.data_dir / "alpha5_30_direction_learnable_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_30_direction_learnable_oos.parquet")

    entry_model, entry_cols = _load_classifier(args.model_dir / "entry_state_catboost_gpu.cbm", multiclass=True)
    dir_model, dir_cols = _load_classifier(args.model_dir / "direction_catboost_gpu.cbm", multiclass=False)
    q_model, q_cols = _load_regressor(args.model_dir / "quality_catboost_gpu.cbm")

    x_val_entry = _x(val_df, entry_cols)
    x_oos_entry = _x(oos_df, entry_cols)
    x_val_dir = _x(val_df, dir_cols)
    x_oos_dir = _x(oos_df, dir_cols)
    x_val_q = _x(val_df, q_cols)
    x_oos_q = _x(oos_df, q_cols)

    labels_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0.0).to_numpy(np.int64)
    labels_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0.0).to_numpy(np.int64)

    entry_pred_val, trade_margin_val = _entry_trade_prob(entry_model, x_val_entry)
    entry_pred_oos, trade_margin_oos = _entry_trade_prob(entry_model, x_oos_entry)
    p_long_val = _direction_long_prob(dir_model, x_val_dir)
    p_long_oos = _direction_long_prob(dir_model, x_oos_dir)
    q_pred_val = np.asarray(q_model.predict(x_val_q), dtype=np.float64).reshape(-1)
    q_pred_oos = np.asarray(q_model.predict(x_oos_q), dtype=np.float64).reshape(-1)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for require_trade_argmax, trade_margin_min, side_threshold, side_margin_min, quality_min, score_abs_min in product(
        [True, False],
        [0.00, 0.02, 0.05, 0.08, 0.12],
        [0.50, 0.55, 0.60, 0.65],
        [0.00, 0.05, 0.10, 0.15],
        [-0.05, 0.00, 0.05, 0.10],
        [0.00, 0.05, 0.10],
    ):
        actions_val, aux_val = _compose(
            val_df, entry_pred_val, trade_margin_val, p_long_val, q_pred_val,
            require_trade_argmax=require_trade_argmax,
            trade_margin_min=trade_margin_min,
            side_threshold=side_threshold,
            side_margin_min=side_margin_min,
            quality_min=quality_min,
            score_abs_min=score_abs_min,
        )
        val_eval = _eval(val_df, actions_val, labels_val, fee=args.fee, slip=args.slip, exposure=args.exposure, max_hold=args.max_hold_bars)
        row = {
            "require_trade_argmax": bool(require_trade_argmax),
            "trade_margin_min": float(trade_margin_min),
            "side_threshold": float(side_threshold),
            "side_margin_min": float(side_margin_min),
            "quality_min": float(quality_min),
            "score_abs_min": float(score_abs_min),
            "val_score": float(val_eval["score"]),
            "val_cost1_pnl": float(val_eval["backtest"]["cost1"]["pnl"]),
            "val_cost1_mdd": float(val_eval["backtest"]["cost1"]["mdd"]),
            "val_trades": int(val_eval["backtest"]["cost1"]["trades"]),
            "val_trade_precision": float(val_eval["direction"]["trade_precision"]),
            "val_balanced_trade_precision": float(val_eval["direction"]["balanced_trade_precision"]),
            "val_coverage": float(val_eval["direction"]["coverage"]),
        }
        rows.append(row)
        if best is None or row["val_score"] > best["val_score"]:
            best = row

    assert best is not None
    actions_oos, aux_oos = _compose(
        oos_df, entry_pred_oos, trade_margin_oos, p_long_oos, q_pred_oos,
        require_trade_argmax=bool(best["require_trade_argmax"]),
        trade_margin_min=float(best["trade_margin_min"]),
        side_threshold=float(best["side_threshold"]),
        side_margin_min=float(best["side_margin_min"]),
        quality_min=float(best["quality_min"]),
        score_abs_min=float(best["score_abs_min"]),
    )
    oos_eval = _eval(oos_df, actions_oos, labels_oos, fee=args.fee, slip=args.slip, exposure=args.exposure, max_hold=args.max_hold_bars)

    summary = {
        "model_id": MODEL_ID,
        "selection": best,
        "validation": best,
        "oos": {
            "cost1": oos_eval["backtest"]["cost1"],
            "cost2": oos_eval["backtest"]["cost2"],
            "cost3": oos_eval["backtest"]["cost3"],
            "direction": oos_eval["direction"],
            "score": float(oos_eval["score"]),
        },
    }
    (args.out_dir / "alpha5_32_catboost_action_backtest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    pd.DataFrame(rows).sort_values("val_score", ascending=False).to_csv(
        args.out_dir / "alpha5_32_catboost_action_backtest_grid.csv",
        index=False,
    )
    print(json.dumps({
        "stage": "alpha5_32_action_backtest_done",
        "summary_path": str(args.out_dir / "alpha5_32_catboost_action_backtest_summary.json"),
        "val_trades": int(best["val_trades"]),
        "oos_pnl": float(oos_eval["backtest"]["cost1"]["pnl"]),
        "oos_trades": int(oos_eval["backtest"]["cost1"]["trades"]),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
