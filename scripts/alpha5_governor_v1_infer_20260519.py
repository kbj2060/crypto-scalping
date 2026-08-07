#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from scripts.train_eval_alpha5_23_hgb_direction_refined_20260519 import _x  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha5_governor_v1_infer_20260519"
DEFAULT_MODEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_governor_v1_contracts_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable005_20260519"
DEFAULT_OUT_PATH = ROOT / "tmp/causal_regen_20260516/alpha5_governor_v1_infer_20260519/predictions.parquet"


def load_classifier(path: Path) -> tuple[Any, list[str]]:
    model = CatBoostClassifier()
    model.load_model(str(path))
    meta = joblib.load(path.parent / f"{path.stem.replace('_catboost_gpu', '')}_meta.joblib")
    return model, meta["feature_cols"]


def load_regressor(path: Path) -> tuple[Any, list[str]]:
    model = CatBoostRegressor()
    model.load_model(str(path))
    meta = joblib.load(path.parent / f"{path.stem.replace('_catboost_gpu', '')}_meta.joblib")
    return model, meta["feature_cols"]


def load_governor(model_dir: Path) -> dict[str, Any]:
    entry_model, entry_cols = load_classifier(model_dir / "entry_state_catboost_gpu.cbm")
    direction_model, direction_cols = load_classifier(model_dir / "direction_catboost_gpu.cbm")
    quality_model, quality_cols = load_regressor(model_dir / "quality_catboost_gpu.cbm")
    governor = {
        "entry_model": entry_model,
        "entry_cols": entry_cols,
        "direction_model": direction_model,
        "direction_cols": direction_cols,
        "quality_model": quality_model,
        "quality_cols": quality_cols,
    }
    ambiguous_path = model_dir / "ambiguous_subtype_catboost_gpu.cbm"
    if ambiguous_path.exists():
        ambiguous_model, ambiguous_cols = load_classifier(ambiguous_path)
        governor["ambiguous_model"] = ambiguous_model
        governor["ambiguous_cols"] = ambiguous_cols
    return governor


def predict_heads(governor: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    x_entry = _x(frame, governor["entry_cols"])
    x_dir = _x(frame, governor["direction_cols"])
    x_q = _x(frame, governor["quality_cols"])

    entry_p = np.asarray(governor["entry_model"].predict_proba(x_entry), dtype=np.float64)
    p_clean = entry_p[:, 0]
    p_amb = entry_p[:, 1]
    p_trade = entry_p[:, 2]

    dir_p = np.asarray(governor["direction_model"].predict_proba(x_dir), dtype=np.float64)
    p_long = np.clip(dir_p[:, 1], 0.0, 1.0)
    p_short = 1.0 - p_long

    q_pred = np.asarray(governor["quality_model"].predict(x_q), dtype=np.float64).reshape(-1)
    if "ambiguous_model" in governor:
        x_amb = _x(frame, governor["ambiguous_cols"])
        amb_p = np.asarray(governor["ambiguous_model"].predict_proba(x_amb), dtype=np.float64)[:, 1]
    else:
        amb_p = np.full(len(frame), 0.5, dtype=np.float64)
    current_direction_score = pd.to_numeric(frame.get("current_direction_score", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)

    return pd.DataFrame(
        {
            "p_clean_wait": p_clean,
            "p_ambiguous": p_amb,
            "p_amb_trade_like": amb_p,
            "p_amb_structural": 1.0 - amb_p,
            "p_trade": p_trade,
            "p_long": p_long,
            "p_short": p_short,
            "quality_pred": q_pred,
            "current_direction_score": current_direction_score,
        },
        index=frame.index,
    )


def compose_rank_veto(
    frame: pd.DataFrame,
    head_pred: pd.DataFrame,
    *,
    amb_struct_penalty: float = 0.95,
    amb_trade_like_penalty: float = 0.30,
    clean_penalty: float = 0.50,
    quality_gain: float = 0.25,
    trade_score_min: float = 0.05,
    ambiguous_struct_cap: float = 0.45,
    ambiguous_trade_like_cap: float = 0.75,
    quality_min: float = 0.00,
    side_threshold: float = 0.50,
    side_margin_min: float = 0.00,
    score_abs_min: float = 0.05,
) -> tuple[np.ndarray, pd.DataFrame]:
    regime = frame.get("regime4_state", "unknown").astype(str).to_numpy()
    p_clean = head_pred["p_clean_wait"].to_numpy(np.float64)
    p_amb = head_pred["p_ambiguous"].to_numpy(np.float64)
    p_amb_trade_like = head_pred["p_amb_trade_like"].to_numpy(np.float64)
    p_amb_struct = head_pred["p_amb_structural"].to_numpy(np.float64)
    p_trade = head_pred["p_trade"].to_numpy(np.float64)
    p_long = head_pred["p_long"].to_numpy(np.float64)
    p_short = head_pred["p_short"].to_numpy(np.float64)
    q_pred = head_pred["quality_pred"].to_numpy(np.float64)
    score_abs = np.abs(head_pred["current_direction_score"].to_numpy(np.float64))

    amb_struct_score = p_amb * p_amb_struct
    amb_trade_like_score = p_amb * p_amb_trade_like
    trade_score = (
        p_trade
        - float(amb_struct_penalty) * amb_struct_score
        - float(amb_trade_like_penalty) * amb_trade_like_score
        - float(clean_penalty) * p_clean
        + float(quality_gain) * q_pred
    )
    best_side = np.maximum(p_long, p_short)
    side_margin = np.abs(p_long - p_short)

    actions = np.where(p_long >= p_short, 1, 2).astype(np.int64)
    actions = np.where(regime == "whipsaw", 0, actions)
    actions = np.where(trade_score >= float(trade_score_min), actions, 0)
    actions = np.where(amb_struct_score <= float(ambiguous_struct_cap), actions, 0)
    actions = np.where(amb_trade_like_score <= float(ambiguous_trade_like_cap), actions, 0)
    actions = np.where(q_pred >= float(quality_min), actions, 0)
    actions = np.where(best_side >= float(side_threshold), actions, 0)
    actions = np.where(side_margin >= float(side_margin_min), actions, 0)
    actions = np.where(score_abs >= float(score_abs_min), actions, 0)

    diag = pd.DataFrame(
        {
            "trade_score": trade_score,
            "amb_struct_score": amb_struct_score,
            "amb_trade_like_score": amb_trade_like_score,
            "best_side": best_side,
            "side_margin": side_margin,
            "score_abs": score_abs,
            "selected_action": actions,
        },
        index=frame.index,
    )
    return actions, diag


def main() -> None:
    p = argparse.ArgumentParser(description="Infer alpha5 governor v1 head outputs and compose rank/veto actions.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA_DIR / "alpha5_30_direction_learnable_oos.parquet")
    p.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    p.add_argument("--trade-score-min", type=float, default=0.05)
    p.add_argument("--ambiguous-struct-cap", type=float, default=0.45)
    p.add_argument("--ambiguous-trade-like-cap", type=float, default=0.75)
    p.add_argument("--quality-min", type=float, default=0.00)
    p.add_argument("--side-threshold", type=float, default=0.50)
    p.add_argument("--side-margin-min", type=float, default=0.00)
    p.add_argument("--score-abs-min", type=float, default=0.05)
    args = p.parse_args()

    frame = pd.read_parquet(args.data_path)
    governor = load_governor(args.model_dir)
    head_pred = predict_heads(governor, frame)
    actions, diag = compose_rank_veto(
        frame,
        head_pred,
        trade_score_min=args.trade_score_min,
        ambiguous_struct_cap=args.ambiguous_struct_cap,
        ambiguous_trade_like_cap=args.ambiguous_trade_like_cap,
        quality_min=args.quality_min,
        side_threshold=args.side_threshold,
        side_margin_min=args.side_margin_min,
        score_abs_min=args.score_abs_min,
    )
    frame_out = frame.reset_index(drop=True).copy()
    if "current_direction_score" in frame_out.columns:
        frame_out = frame_out.rename(columns={"current_direction_score": "current_direction_score_input"})
    out = pd.concat([frame_out, head_pred.reset_index(drop=True), diag.reset_index(drop=True)], axis=1)
    out["governor_action"] = actions
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_path, index=False)
    print(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "out_path": str(args.out_path),
                "rows": int(len(out)),
                "trades": int(np.sum(actions != 0)),
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
