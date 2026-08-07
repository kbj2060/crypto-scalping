#!/usr/bin/env python3
"""Build SOL fixed TP=1.8%/SL=1.0% barrier candidate frame + Regime3 current probs.

This is a SOL-specific analog of
scripts/build_fixed_regime4_tp_sl_preprocess_20260517.py +
scripts/build_alpha4_tp_sl_path_edge_feature_20260517.py, but:

  - built directly from SOL's own raw 5m feature files
    (data/splits/year_oos/sol_features_{2025,2026}.csv), fresh through
    2026-07-21, instead of an ETH ai_feature_combo_grid source.
  - uses SOL's existing, currently-live Regime3 artifact
    (sol_regime3_current_hmm_sensitive_wide24_20260707) instead of the
    deprecated Regime4 HMM surface (no SOL Regime4 artifact exists and
    none should be built - Regime4 is deprecated project-wide).
  - the "directional/flow" feature set is FEATURE_COLS from
    ensemble.fully_learned_governor_policy minus DROP_RETRAIN_FEATURES
    minus columns that don't exist anywhere in SOL's raw data (ETH-only
    AI-ensemble outputs: m7_*, pred_patchtst/conf_patchtst,
    patchtst_median/patchtst_regime_sim, ai_dir_*, ai_adverse_risk,
    ai_reward_risk, ai_vol_regime_pct, tide_vol_*, ai_flow_*,
    dlinear_smf_*) and minus regime_*_id (raw regime_bull/bear/... columns
    don't exist in SOL data either) - these are dropped gracefully rather
    than fabricated as all-zero columns.

tp_sl_action_score keeps the exact fixed TP=1.8%/SL=1.0%, horizon=48 bars,
entry=next-bar-open, same-bar-tie=SL-wins barrier definition and walk-forward
OOF quantile-regression scoring policy as the ETH build
(scripts/build_alpha4_tp_sl_path_edge_feature_20260517.py), reusing its
private helper functions directly (they are asset-agnostic - operate only
on open/high/low/close arrays).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import FEATURE_COLS, prepare_features  # noqa: E402
from scripts.build_alpha4_tp_sl_path_edge_feature_20260517 import (  # noqa: E402
    DROP_RETRAIN_FEATURES,
    _barriers,
    _close as _close_a4,
    _fit_hgb_pair,
    _fit_quantile_pair,
    _predict_action_score,
    _predict_hgb_action_score,
    _targets,
    _walk_forward_oof,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


MODEL_ID = "sol_regime3_tp18_sl10_preprocess_20260722"
DEFAULT_TRAIN_RAW = ROOT / "data/splits/year_oos/sol_features_2025.csv"
DEFAULT_EVAL_RAW = ROOT / "data/splits/year_oos/sol_features_2026.csv"
DEFAULT_REGIME3_2025 = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707/sol_features_2025_regime3_current_sensitive_hmm_wide24.csv"
DEFAULT_REGIME3_2026 = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707/sol_features_2026_regime3_current_sensitive_hmm_wide24.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_fixed_regime3_tp18_sl10_preprocess_20260722"

REGIME3_PREFIX = "regime3_current_sensitive_wide24_"
REGIMES3 = ("bull", "bear", "chop")
TP = 0.018
SL = 0.010
HORIZON = 48

# ETH-only AI-ensemble outputs that don't exist anywhere in SOL's raw data.
MISSING_AI_FAMILY = {
    "m7_gate_block", "m7_tail_risk", "m7_expected_ret", "m7_composite_score",
    "m7_confidence", "m7_qwidth",
    "pred_patchtst", "conf_patchtst", "ai_dir_edge", "ai_dir_p_up",
    "ai_dir_p_down", "ai_dir_p_flat", "ai_dir_entropy",
    "patchtst_median", "patchtst_regime_sim", "ai_adverse_risk",
    "ai_reward_risk", "ai_vol_regime_pct", "tide_vol_raw", "tide_vol_zscore",
    "ai_flow_pressure", "ai_flow_exhaustion", "ai_flow_flip_prob",
    "ai_flow_slope", "dlinear_smf_ema", "dlinear_smf_slope",
}
# Raw regime_bull/regime_bear/... columns don't exist in SOL data (legacy
# regime taxonomy handled elsewhere) - drop rather than fabricate zeros.
MISSING_REGIME_ID_FAMILY = {
    "regime_bull_id", "regime_bear_id", "regime_chop_id",
    "regime_whipsaw_id", "regime_normal_id",
}


def _sol_feature_cols() -> list[str]:
    out = [
        c for c in FEATURE_COLS
        if c not in DROP_RETRAIN_FEATURES
        and c not in MISSING_AI_FAMILY
        and c not in MISSING_REGIME_ID_FAMILY
    ]
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_tp_sl_action_score(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_train_rows: int,
    n_estimators: int,
    risk_penalty: float,
    seed: int,
    model_family: str = "lgbm_quantile",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    close_train = _close_a4(train_df)
    close_eval = _close_a4(eval_df)
    x_train = prepare_features(train_df, side_hint=0, close=close_train, feature_cols=feature_cols).replace([np.inf, -np.inf], np.nan)
    x_eval = prepare_features(eval_df, side_hint=0, close=close_eval, feature_cols=feature_cols).replace([np.inf, -np.inf], np.nan)

    tp_arr, sl_arr, barrier_meta = _barriers(
        train_df, mode="fixed", fixed_tp=TP, fixed_sl=SL, tp_atr_mult=0.0, sl_atr_mult=0.0, atr_window=14,
    )
    y_long, y_short = _targets(train_df, horizon=HORIZON, tp=tp_arr, sl=sl_arr)

    train_edge, fold_rows = _walk_forward_oof(
        train_df, x_train, y_long, y_short,
        horizon=HORIZON, min_train_rows=min_train_rows, risk_penalty=risk_penalty, deadband=0.0,
        seed=seed, n_estimators=n_estimators, model_family=model_family,
    )
    if model_family == "hgb":
        final_model = _fit_hgb_pair(x_train, y_long, y_short, seed + 9999, n_estimators)
        eval_edge = _predict_hgb_action_score(final_model, x_eval)
    else:
        final_model = _fit_quantile_pair(x_train, y_long, y_short, seed + 9999, n_estimators)
        eval_edge = _predict_action_score(final_model, x_eval, risk_penalty=risk_penalty, deadband=0.0)

    meta = {
        "model_family": model_family,
        "barrier": barrier_meta,
        "horizon_bars": HORIZON,
        "tp": TP,
        "sl": SL,
        "folds": fold_rows,
        "train_stats": {
            "mean": float(np.mean(train_edge)),
            "std": float(np.std(train_edge)),
            "zero_rate": float(np.mean(train_edge == 0.0)),
        },
        "eval_stats": {
            "mean": float(np.mean(eval_edge)),
            "std": float(np.std(eval_edge)),
            "zero_rate": float(np.mean(eval_edge == 0.0)),
        },
    }
    return train_edge, eval_edge, meta


def _merge_regime3(base: pd.DataFrame, sidecar_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    sidecar = _read(sidecar_path)
    cols = [c for c in sidecar.columns if c.startswith(REGIME3_PREFIX)]
    if not cols:
        raise ValueError(f"{sidecar_path} has no {REGIME3_PREFIX}* columns")
    overlap = set(base.columns) & set(cols)
    if overlap:
        raise ValueError(f"{sidecar_path} would overwrite existing columns: {sorted(overlap)}")
    out = base.merge(sidecar[["timestamp"] + cols], on="timestamp", how="left")
    missing_rows = int(out[cols].isna().any(axis=1).sum())
    if missing_rows:
        raise ValueError(f"{sidecar_path} failed exact timestamp alignment; missing rows={missing_rows}")
    return out, {"source": str(sidecar_path), "rows": int(len(sidecar)), "columns": cols, "sha256": _sha256(sidecar_path)}


def _prob_audit(frame: pd.DataFrame) -> dict[str, Any]:
    prob_cols = [f"{REGIME3_PREFIX}{r}_prob" for r in REGIMES3]
    sums = frame[prob_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return {
        "probability_columns": prob_cols,
        "prob_sum_min": float(sums.min()),
        "prob_sum_max": float(sums.max()),
        "nan_count": int(frame[prob_cols].isna().sum().sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SOL fixed TP18/SL10 barrier frame + Regime3 current probs.")
    p.add_argument("--train-raw", type=Path, default=DEFAULT_TRAIN_RAW)
    p.add_argument("--eval-raw", type=Path, default=DEFAULT_EVAL_RAW)
    p.add_argument("--regime3-2025", type=Path, default=DEFAULT_REGIME3_2025)
    p.add_argument("--regime3-2026", type=Path, default=DEFAULT_REGIME3_2026)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--min-train-rows", type=int, default=20000)
    p.add_argument("--n-estimators", type=int, default=80)
    p.add_argument("--risk-penalty", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=417)
    p.add_argument("--model-family", choices=["hgb", "lgbm_quantile"], default="lgbm_quantile")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = _read(args.train_raw)
    eval_raw = _read(args.eval_raw)
    feature_cols = _sol_feature_cols()

    train_edge, eval_edge, tp_sl_meta = _build_tp_sl_action_score(
        train_raw, eval_raw, feature_cols,
        min_train_rows=int(args.min_train_rows), n_estimators=int(args.n_estimators),
        risk_penalty=float(args.risk_penalty), seed=int(args.seed), model_family=str(args.model_family),
    )
    train_out = train_raw.copy()
    eval_out = eval_raw.copy()
    train_out["tp_sl_action_score"] = train_edge
    eval_out["tp_sl_action_score"] = eval_edge

    train_out, train_r3_meta = _merge_regime3(train_out, args.regime3_2025)
    eval_out, eval_r3_meta = _merge_regime3(eval_out, args.regime3_2026)

    train_path = args.out_dir / "trade_candidates_2025_sol_regime3_tp18_sl10_fixed.csv"
    eval_path = args.out_dir / "trade_candidates_2026_sol_regime3_tp18_sl10_fixed.csv"
    train_out.to_csv(train_path, index=False)
    eval_out.to_csv(eval_path, index=False)

    manifest = {
        "model_id": MODEL_ID,
        "asset": "SOL",
        "fixed_preprocessing_contract": {
            "regime_taxonomy": list(REGIMES3),
            "current_regime_prefix": REGIME3_PREFIX,
            "tp_sl_feature": "tp_sl_action_score",
            "tp": TP,
            "sl": SL,
            "tp_sl_horizon_bars": HORIZON,
            "tp_sl_entry_reference": "next_bar_open",
            "same_bar_tp_sl_tie": "sl_wins",
            "timestamp_join": "exact_left_join_no_missing_rows",
            "regime4_used": False,
            "regime4_reason": "no SOL Regime4 artifact exists; Regime4 is deprecated project-wide, use Regime3 instead",
        },
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "dropped_missing_ai_family": sorted(MISSING_AI_FAMILY),
        "dropped_missing_regime_id_family": sorted(MISSING_REGIME_ID_FAMILY),
        "tp_sl_action_score": tp_sl_meta,
        "train": {
            "output": str(train_path),
            "output_sha256": _sha256(train_path),
            "source": {"path": str(args.train_raw), "sha256": _sha256(args.train_raw)},
            "rows": int(len(train_out)),
            "range": [str(train_out["timestamp"].iloc[0]), str(train_out["timestamp"].iloc[-1])],
            "regime3": train_r3_meta,
            "regime3_prob_audit": _prob_audit(train_out),
        },
        "eval": {
            "output": str(eval_path),
            "output_sha256": _sha256(eval_path),
            "source": {"path": str(args.eval_raw), "sha256": _sha256(args.eval_raw)},
            "rows": int(len(eval_out)),
            "range": [str(eval_out["timestamp"].iloc[0]), str(eval_out["timestamp"].iloc[-1])],
            "regime3": eval_r3_meta,
            "regime3_prob_audit": _prob_audit(eval_out),
        },
        "warnings": [
            "eval covers all of 2026 through the fresh data boundary (2026-07-21); "
            "downstream training script must itself slice canonical OOS "
            "(2026-01-01..2026-02-28) from the fresh-forward extension "
            "(2026-03-01..2026-07-21) - both are never used for selection.",
        ],
    }
    manifest_path = args.out_dir / "sol_regime3_tp18_sl10_preprocess_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "train": str(train_path), "eval": str(eval_path), "feature_count": len(feature_cols)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
