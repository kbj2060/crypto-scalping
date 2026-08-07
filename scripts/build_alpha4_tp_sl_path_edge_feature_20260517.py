#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import FEATURE_COLS, prepare_features  # noqa: E402
from scripts.eval_alpha4_new_features_full_retrain_20260517 import DROP_RETRAIN_FEATURES  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha4_tp_sl_path_edge_feature_20260517"
DEFAULT_TRAIN = ROOT / "tmp/causal_regen_20260516/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/causal_regen_20260516/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha4_3_purged_lgbm_quantile_tp_sl_action_score_20260517"
LEGACY_CLEAN_REGIME_PREFIX = "clean_regime_2024_unsup_v4_"
STICKY_REGIME_PREFIXES = (
    "clean_regime4_2024_unsup_v1_",
    "clean_regime4_state24_sticky090_v2_",
)


def _price_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    open_ = pd.to_numeric(df["open"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(df["high"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)
    close = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)
    return open_, high, low, close


def _first_path_outcome(
    entry: float,
    future_high: np.ndarray,
    future_low: np.ndarray,
    *,
    side: int,
    tp: float,
    sl: float,
) -> float:
    if not np.isfinite(entry) or entry <= 0.0 or len(future_high) == 0:
        return 0.0
    if side > 0:
        tp_hit = future_high >= entry * (1.0 + tp)
        sl_hit = future_low <= entry * (1.0 - sl)
    else:
        tp_hit = future_low <= entry * (1.0 - tp)
        sl_hit = future_high >= entry * (1.0 + sl)
    tp_idx = np.flatnonzero(tp_hit)
    sl_idx = np.flatnonzero(sl_hit)
    first_tp = int(tp_idx[0]) if tp_idx.size else 10**9
    first_sl = int(sl_idx[0]) if sl_idx.size else 10**9
    if first_tp < first_sl:
        return 1.0
    if first_sl <= first_tp and first_sl < 10**9:
        return -1.0
    terminal_ref = float((future_high[-1] + future_low[-1]) * 0.5)
    ret = (terminal_ref - entry) / max(entry, 1e-12)
    signed_ret = ret if side > 0 else -ret
    scale = max(tp, sl, 1e-6)
    return float(np.clip(signed_ret / scale, -0.5, 0.5))


def _atr_pct(df: pd.DataFrame, *, window: int) -> np.ndarray:
    open_, high, low, close = _price_arrays(df)
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(int(window), min_periods=max(2, min(5, int(window)))).mean()
    atr = atr.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
    return np.clip(atr / np.maximum(close, 1e-12), 1e-6, 0.50)


def _barriers(
    df: pd.DataFrame,
    *,
    mode: str,
    fixed_tp: float,
    fixed_sl: float,
    tp_atr_mult: float,
    sl_atr_mult: float,
    atr_window: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(df)
    if mode == "fixed":
        tp = np.full(n, float(fixed_tp), dtype=np.float64)
        sl = np.full(n, float(fixed_sl), dtype=np.float64)
        meta = {"mode": "fixed", "fixed_tp": float(fixed_tp), "fixed_sl": float(fixed_sl)}
    elif mode == "atr":
        atr = _atr_pct(df, window=int(atr_window))
        tp = atr * float(tp_atr_mult)
        sl = atr * float(sl_atr_mult)
        meta = {
            "mode": "atr",
            "atr_window": int(atr_window),
            "tp_atr_mult": float(tp_atr_mult),
            "sl_atr_mult": float(sl_atr_mult),
            "atr_pct_mean": float(np.mean(atr)),
            "atr_pct_median": float(np.median(atr)),
            "tp_mean": float(np.mean(tp)),
            "tp_median": float(np.median(tp)),
            "sl_mean": float(np.mean(sl)),
            "sl_median": float(np.median(sl)),
        }
    else:
        raise ValueError(f"unsupported barrier mode: {mode}")
    return np.clip(tp, 1e-6, 0.50), np.clip(sl, 1e-6, 0.50), meta


def _targets(
    df: pd.DataFrame,
    *,
    horizon: int,
    tp: np.ndarray,
    sl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    open_, high, low, close = _price_arrays(df)
    n = len(df)
    long_y = np.zeros(n, dtype=np.float32)
    short_y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        entry_i = min(i + 1, n - 1)
        end = min(entry_i + int(horizon), n)
        if end <= entry_i:
            continue
        entry = open_[entry_i] if np.isfinite(open_[entry_i]) and open_[entry_i] > 0 else close[i]
        fh = high[entry_i:end]
        fl = low[entry_i:end]
        long_y[i] = _first_path_outcome(entry, fh, fl, side=1, tp=float(tp[i]), sl=float(sl[i]))
        short_y[i] = _first_path_outcome(entry, fh, fl, side=-1, tp=float(tp[i]), sl=float(sl[i]))
    tail = int(min(max(horizon + 2, 1), n))
    if tail:
        long_y[-tail:] = 0.0
        short_y[-tail:] = 0.0
    return long_y, short_y


def _feature_cols(train: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    base = [c for c in FEATURE_COLS if c not in DROP_RETRAIN_FEATURES]
    extras = [
        c
        for c in train.columns
        if c.startswith(STICKY_REGIME_PREFIXES)
        or c in {
            "garch_vol_z",
            "liquidity_vacuum",
            "execution_quality",
            "jump_z",
            "jump_flag",
            "evt_tail_flag",
            "evt_excess_z",
            "funding_abs",
            "funding_pressure",
            "crowding_pressure",
            "whale_conviction",
        }
    ]
    out: list[str] = []
    for c in base + extras:
        lc = c.lower()
        if c in out or c == "tp_sl_path_edge":
            continue
        if c.startswith(LEGACY_CLEAN_REGIME_PREFIX):
            continue
        if any(tok in lc for tok in ("target", "label", "future", "cash_after")):
            continue
        if c in train.columns and c in eval_df.columns:
            out.append(c)
    return out


def _action_score_from_relative_edges(long_edge: np.ndarray, short_edge: np.ndarray, *, deadband: float) -> np.ndarray:
    out = np.asarray(long_edge, dtype=np.float64) - np.asarray(short_edge, dtype=np.float64)
    if deadband > 0.0:
        out[np.abs(out) < float(deadband)] = 0.0
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _action_score_from_sparse_edges(long_edge: np.ndarray, short_edge: np.ndarray) -> np.ndarray:
    long_edge = np.asarray(long_edge, dtype=np.float64)
    short_edge = np.asarray(short_edge, dtype=np.float64)
    best = np.maximum(long_edge, short_edge)
    out = np.zeros(len(long_edge), dtype=np.float64)
    long_mask = (best > 0.0) & (long_edge >= short_edge)
    short_mask = (best > 0.0) & (short_edge > long_edge)
    out[long_mask] = long_edge[long_mask]
    out[short_mask] = -short_edge[short_mask]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _hgb_model(seed: int, n_estimators: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=int(n_estimators),
            learning_rate=0.04,
            l2_regularization=0.12,
            min_samples_leaf=35,
            early_stopping=False,
            random_state=int(seed),
        ),
    )


def _fit_hgb_pair(x: pd.DataFrame, y_long: np.ndarray, y_short: np.ndarray, seed: int, n_estimators: int) -> dict[str, Any]:
    long_model = _hgb_model(seed, n_estimators)
    short_model = _hgb_model(seed + 1000, n_estimators)
    long_model.fit(x, y_long)
    short_model.fit(x, y_short)
    return {"long_model": long_model, "short_model": short_model}


def _predict_hgb_action_score(model: dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    long_edge = np.asarray(model["long_model"].predict(x), dtype=np.float64)
    short_edge = np.asarray(model["short_model"].predict(x), dtype=np.float64)
    return _action_score_from_sparse_edges(long_edge, short_edge)


def _quantile_model(alpha: float, seed: int, n_estimators: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        LGBMRegressor(
            objective="quantile",
            alpha=float(alpha),
            n_estimators=int(n_estimators),
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=35,
            reg_alpha=0.02,
            reg_lambda=0.18,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            extra_trees=True,
            random_state=int(seed),
            n_jobs=-1,
            verbosity=-1,
        ),
    )


def _fit_quantile_side(x: pd.DataFrame, y: np.ndarray, seed: int, n_estimators: int) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for j, q in enumerate((0.25, 0.50, 0.75)):
        model = _quantile_model(q, seed + j * 137, n_estimators)
        model.fit(x, y)
        models[f"q{int(q * 100):02d}"] = model
    return models


def _fit_quantile_pair(x: pd.DataFrame, y_long: np.ndarray, y_short: np.ndarray, seed: int, n_estimators: int) -> dict[str, Any]:
    return {
        "long": _fit_quantile_side(x, y_long, seed, n_estimators),
        "short": _fit_quantile_side(x, y_short, seed + 1000, n_estimators),
    }


def _side_edge(models: dict[str, Any], x: pd.DataFrame, *, risk_penalty: float) -> np.ndarray:
    q25 = np.asarray(models["q25"].predict(x), dtype=np.float64)
    q50 = np.asarray(models["q50"].predict(x), dtype=np.float64)
    q75 = np.asarray(models["q75"].predict(x), dtype=np.float64)
    downside_gap = np.maximum(0.0, q50 - q25)
    width = np.maximum(0.0, q75 - q25)
    edge = q50 - float(risk_penalty) * downside_gap - 0.10 * width
    return np.clip(edge, -1.0, 1.0)


def _predict_action_score(model: dict[str, Any], x: pd.DataFrame, *, risk_penalty: float, deadband: float) -> np.ndarray:
    long_edge = _side_edge(model["long"], x, risk_penalty=risk_penalty)
    short_edge = _side_edge(model["short"], x, risk_penalty=risk_penalty)
    return _action_score_from_relative_edges(long_edge, short_edge, deadband=deadband)


def _walk_forward_oof(
    train_df: pd.DataFrame,
    x_all: pd.DataFrame,
    y_long: np.ndarray,
    y_short: np.ndarray,
    *,
    horizon: int,
    min_train_rows: int,
    risk_penalty: float,
    deadband: float,
    seed: int,
    n_estimators: int,
    model_family: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    ts = pd.to_datetime(train_df["timestamp"])
    periods = pd.PeriodIndex(ts, freq="M")
    edge = np.zeros(len(train_df), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for fold_no, period in enumerate(sorted(periods.unique()), start=1):
        pred_mask = periods == period
        pred_arr = np.asarray(pred_mask, dtype=bool)
        first_pred_pos = int(np.flatnonzero(pred_arr)[0]) if bool(pred_arr.any()) else 0
        # Purge rows whose fixed-horizon TP/SL label can touch the prediction month.
        # Without this gap, late rows from the previous month can train on labels
        # that include the first bars of the month being predicted.
        purge_cutoff_pos = max(0, first_pred_pos - int(horizon) - 2)
        train_mask = np.arange(len(train_df)) <= purge_cutoff_pos
        n_train = int(train_mask.sum())
        n_pred = int(pred_mask.sum())
        if n_train < int(min_train_rows):
            rows.append({"fold": fold_no, "period": str(period), "train_rows": n_train, "pred_rows": n_pred, "purge_gap_rows": int(first_pred_pos - purge_cutoff_pos), "status": "zero_fill_insufficient_history"})
            continue
        if model_family == "hgb":
            model = _fit_hgb_pair(x_all.loc[train_mask], y_long[train_mask], y_short[train_mask], seed + fold_no * 101, n_estimators)
            edge[pred_arr] = _predict_hgb_action_score(model, x_all.loc[pred_arr])
        elif model_family == "lgbm_quantile":
            model = _fit_quantile_pair(x_all.loc[train_mask], y_long[train_mask], y_short[train_mask], seed + fold_no * 101, n_estimators)
            edge[pred_arr] = _predict_action_score(
                model,
                x_all.loc[pred_arr],
                risk_penalty=float(risk_penalty),
                deadband=float(deadband),
            )
        else:
            raise ValueError(f"unsupported model family: {model_family}")
        rows.append(
            {
                "fold": fold_no,
                "period": str(period),
                "train_rows": n_train,
                "pred_rows": n_pred,
                "purge_gap_rows": int(first_pred_pos - purge_cutoff_pos),
                "status": "predicted",
                "edge_mean": float(np.mean(edge[pred_mask])),
                "edge_std": float(np.std(edge[pred_mask])),
            }
        )
    return edge, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build leak-safe single signed TP/SL path feature for Alpha4 parent training.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--barrier-mode", choices=["fixed", "atr"], default="fixed")
    p.add_argument("--tp", type=float, default=0.018)
    p.add_argument("--sl", type=float, default=0.010)
    p.add_argument("--tp-atr-mult", type=float, default=3.0)
    p.add_argument("--sl-atr-mult", type=float, default=1.5)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--min-train-rows", type=int, default=20000)
    p.add_argument("--model-family", choices=["hgb", "lgbm_quantile"], default="lgbm_quantile")
    p.add_argument("--risk-penalty", type=float, default=0.50)
    p.add_argument("--deadband", type=float, default=0.0)
    p.add_argument("--n-estimators", type=int, default=80)
    p.add_argument("--seed", type=int, default=417)
    p.add_argument("--feature-name", default="tp_sl_action_score")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    cols = _feature_cols(train_df, eval_df)
    x_train = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=cols).replace([np.inf, -np.inf], np.nan)
    x_eval = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=cols).replace([np.inf, -np.inf], np.nan)
    train_tp, train_sl, train_barrier_meta = _barriers(
        train_df,
        mode=str(args.barrier_mode),
        fixed_tp=float(args.tp),
        fixed_sl=float(args.sl),
        tp_atr_mult=float(args.tp_atr_mult),
        sl_atr_mult=float(args.sl_atr_mult),
        atr_window=int(args.atr_window),
    )
    y_long, y_short = _targets(train_df, horizon=int(args.horizon), tp=train_tp, sl=train_sl)

    train_edge, fold_rows = _walk_forward_oof(
        train_df,
        x_train,
        y_long,
        y_short,
        horizon=int(args.horizon),
        min_train_rows=int(args.min_train_rows),
        risk_penalty=float(args.risk_penalty),
        deadband=float(args.deadband),
        seed=int(args.seed),
        n_estimators=int(args.n_estimators),
        model_family=str(args.model_family),
    )
    if str(args.model_family) == "hgb":
        final_model = _fit_hgb_pair(x_train, y_long, y_short, int(args.seed) + 9999, int(args.n_estimators))
        eval_edge = _predict_hgb_action_score(final_model, x_eval)
    else:
        final_model = _fit_quantile_pair(x_train, y_long, y_short, int(args.seed) + 9999, int(args.n_estimators))
        eval_edge = _predict_action_score(final_model, x_eval, risk_penalty=float(args.risk_penalty), deadband=float(args.deadband))

    train_out = train_df.copy()
    eval_out = eval_df.copy()
    train_out[str(args.feature_name)] = train_edge
    eval_out[str(args.feature_name)] = eval_edge
    train_path = args.out_dir / args.train_csv.name
    eval_path = args.out_dir / args.eval_csv.name
    train_out.to_csv(train_path, index=False)
    eval_out.to_csv(eval_path, index=False)
    model_path = args.out_dir / "tp_sl_path_edge_predictor.pkl"
    if str(args.model_family) == "hgb":
        model_payload = {
            "model_id": MODEL_ID,
            "model_family": "HistGradientBoostingRegressor_pair_sparse_hold_aware",
            "long_model": final_model["long_model"],
            "short_model": final_model["short_model"],
            "feature_cols": cols,
            "horizon": int(args.horizon),
            "barrier_policy": train_barrier_meta,
            "n_estimators": int(args.n_estimators),
        }
    else:
        model_payload = {"model_id": MODEL_ID, "model_family": "LightGBM_quantile_regression_pair", "model": final_model, "feature_cols": cols, "horizon": int(args.horizon), "tp": float(args.tp), "sl": float(args.sl), "risk_penalty": float(args.risk_penalty), "deadband": float(args.deadband), "n_estimators": int(args.n_estimators)}
    joblib.dump(model_payload, model_path)
    if str(args.model_family) == "hgb":
        model_family_desc = "HGB sparse hold-aware pair: long/short HistGradientBoostingRegressor"
        score_policy: dict[str, Any] = {"action_score": "0 if max(long_edge, short_edge) <= 0 else +long_edge when long_edge>=short_edge else -short_edge"}
    else:
        model_family_desc = "LightGBM quantile regression pair: long/short x q25/q50/q75"
        score_policy = {
            "quantiles": [0.25, 0.50, 0.75],
            "side_edge": "q50 - risk_penalty*(q50-q25) - 0.10*(q75-q25)",
            "action_score": "long_side_edge - short_side_edge",
            "risk_penalty": float(args.risk_penalty),
            "deadband": float(args.deadband),
        }
    audit = {
        "model_id": MODEL_ID,
        "status": "pass",
        "model_family": model_family_desc,
        "selection_uses_2026": False,
        "feature_name": str(args.feature_name),
        "contract": "single signed hold/long/short parent input feature; not a post-entry reject gate and not based on candidate TP/SL/max_hold",
        "label_policy": {"entry": "next_bar_open", "horizon_bars": int(args.horizon), "barrier": train_barrier_meta, "same_bar_tie": "SL_conservative"},
        "train_generation": "monthly walk-forward OOF; periods without enough prior rows are zero-filled",
        "eval_generation": "fit final predictor on all 2025 labels only; predict 2026 without 2026 labels",
        "score_policy": score_policy,
        "n_estimators": int(args.n_estimators),
        "feature_cols": cols,
        "legacy_clean_regime_feature_count": int(sum(c.startswith(LEGACY_CLEAN_REGIME_PREFIX) for c in cols)),
        "sticky_regime_feature_count": int(sum(c.startswith(STICKY_REGIME_PREFIXES) for c in cols)),
        "folds": fold_rows,
        "stats": {
            "train_mean": float(np.mean(train_edge)),
            "train_std": float(np.std(train_edge)),
            "train_zero_rate": float(np.mean(train_edge == 0.0)),
            "eval_mean": float(np.mean(eval_edge)),
            "eval_std": float(np.std(eval_edge)),
            "eval_zero_rate": float(np.mean(eval_edge == 0.0)),
            "long_target_mean": float(np.mean(y_long)),
            "short_target_mean": float(np.mean(y_short)),
        },
        "artifacts": {"train_csv": str(train_path), "eval_csv": str(eval_path), "model": str(model_path), "audit": str(args.out_dir / "tp_sl_path_edge_feature_audit.json")},
    }
    audit_path = args.out_dir / "tp_sl_path_edge_feature_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"train_csv": str(train_path), "eval_csv": str(eval_path), "audit": str(audit_path), "stats": audit["stats"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
