#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_5head_policy_20260522 import (  # noqa: E402
    DEFAULT_FEATURE_CSV,
    DEFAULT_LABEL_DIR,
    DEFAULT_SPEC_DIR,
    _label_frame,
    _numeric_matrix,
    _read_feature_frame,
    _read_spec,
)
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    CONTEXT_COLS,
    EQEConfig,
    _apply_label_preset,
    _bucket_horizon,
    _build_entry_labels,
    _build_exit_dataset,
    _estimate_expected_return_by_bucket,
    _exit_close_prob,
    _exit_state_vec,
    _fit_entry_models,
    _fit_exit_model,
    _predict_entry,
    _target_horizon_bucket,
)


MODEL_SPECS = [
    (
        "primary_hreg",
        ROOT
        / "tmp/causal_regen_20260516/alpha6_target_mode_abc_cpu_midgrid_selected_20260523/short_horizon_robust_horizon_reg/current_tail111",
    ),
    (
        "coverage_current",
        ROOT / "data/ensemble/supervised/alpha6_entry_quality_exit_5bucket_main_20260522/current_tail111",
    ),
    (
        "high_precision",
        ROOT
        / "tmp/causal_regen_20260516/alpha6_target_mode_abc_cpu_midgrid_selected_20260523/high_precision_robust/current_tail111",
    ),
    (
        "perturbation",
        ROOT
        / "tmp/causal_regen_20260516/alpha6_target_mode_abc_cpu_midgrid_selected_20260523/perturbation_robust/current_tail111",
    ),
    (
        "adverse_veto",
        ROOT
        / "tmp/causal_regen_20260516/alpha6_target_mode_abc_cpu_midgrid_selected_20260523/adverse_conformal/current_tail111",
    ),
    (
        "sam_veto",
        ROOT / "tmp/causal_regen_20260516/alpha6_target_mode_abc_cpu_midgrid_selected_20260523/sam_conformal/current_tail111",
    ),
]

EXPERT_PRESETS = [
    ("short_horizon_robust", "horizon_reg"),
    ("current_quality", "bucket5"),
    ("high_precision_robust", "bucket5"),
    ("perturbation_robust", "bucket5"),
    ("adverse_conformal", "bucket5"),
    ("sam_conformal", "bucket5"),
]

MODE_NAMES = [
    "skip",
    "primary_only",
    "primary_coverage",
    "primary_precision_confirm",
    "primary_perturbation_confirm",
    "primary_coverage_confirm",
    "risk_veto_filtered",
    "sniper_all_confirm",
    "soft_blend_low_risk",
    "soft_blend_high_conviction",
]


def _class_prob(model: Any, x: np.ndarray, cls: int) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    if cls not in classes:
        return np.zeros(len(x), dtype=np.float64)
    return np.asarray(proba[:, int(np.flatnonzero(classes == cls)[0])], dtype=np.float64)


def _predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    cols = list(bundle["feature_cols"])
    x_raw = _numeric_matrix(frame, cols)
    x = bundle["pipeline"].transform(x_raw)
    models = bundle["entry_models"]
    action_model = models["action_model"]
    cash_p = _class_prob(action_model, x, 0)
    long_p = _class_prob(action_model, x, 1)
    short_p = _class_prob(action_model, x, 2)
    proba = np.vstack([cash_p, long_p, short_p]).T
    action = np.argmax(proba, axis=1).astype(np.int64)
    quality = np.asarray(models["quality_model"].predict(x), dtype=np.float64)
    target_head_mode = str(models.get("target_head_mode", "bucket5")).strip().lower()
    if target_head_mode == "horizon_reg":
        horizon_model = models.get("target_horizon_model") or models.get("target_model")
        max_horizon = int(models.get("max_target_horizon") or 96)
        pred_horizon = np.expm1(np.asarray(horizon_model.predict(x), dtype=np.float64))
        target_horizon = np.clip(np.rint(pred_horizon), 2, max(2, max_horizon)).astype(np.int64)
        target_horizon = np.where(action == 0, 0, target_horizon)
        target_bucket = np.where(action == 0, 0, _target_horizon_bucket(target_horizon)).astype(np.int64)
    else:
        bucket_model = models.get("target_bucket_model") or models.get("target_model")
        if bucket_model is None:
            target_bucket = np.zeros(len(x), dtype=np.int64)
        else:
            bucket_proba = bucket_model.predict_proba(x)
            bucket_classes = np.asarray(bucket_model.classes_, dtype=int)
            target_bucket = bucket_classes[np.argmax(bucket_proba, axis=1)].astype(np.int64)
        target_bucket = np.where(action == 0, 0, np.clip(target_bucket, 0, 4)).astype(np.int64)
        target_horizon = np.asarray([_bucket_horizon(int(v)) if a != 0 else 0 for v, a in zip(target_bucket, action)], dtype=np.int64)
    out = pd.DataFrame(
        {
            "action": action,
            "cash_prob": cash_p,
            "long_prob": long_p,
            "short_prob": short_p,
            "quality": quality,
            "confidence": np.max(proba, axis=1),
            "target_bucket": target_bucket,
            "target_horizon": target_horizon,
        }
    )
    return out, x


def _load_frame(variant: str) -> pd.DataFrame:
    spec = _read_spec(DEFAULT_SPEC_DIR, variant)
    feat, _, _ = _read_feature_frame(DEFAULT_FEATURE_CSV, list(spec["features"]), CONTEXT_COLS)
    frame = feat.merge(_label_frame(DEFAULT_LABEL_DIR), on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    return frame


def _safe_col(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in frame.columns:
        return np.full(len(frame), float(default), dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().fillna(float(default)).to_numpy(dtype=np.float64)


def _build_base_features(frame: pd.DataFrame, preds: list[pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
    cols: list[np.ndarray] = []
    names: list[str] = []
    n = len(frame)
    actions = np.vstack([p["action"].to_numpy(dtype=np.int64) for p in preds]).T
    qualities = np.vstack([p["quality"].to_numpy(dtype=np.float64) for p in preds]).T
    horizons = np.vstack([p["target_horizon"].to_numpy(dtype=np.float64) for p in preds]).T
    for mi, p in enumerate(preds):
        for col in ("cash_prob", "long_prob", "short_prob", "confidence", "quality"):
            cols.append(p[col].to_numpy(dtype=np.float64))
            names.append(f"m{mi}_{col}")
        cols.append(p["target_horizon"].to_numpy(dtype=np.float64) / 96.0)
        names.append(f"m{mi}_target_horizon_frac")
        q = p["quality"].to_numpy(dtype=np.float64)
        for win in (5, 10):
            s = pd.Series(q)
            cols.append(s.rolling(win, min_periods=1).mean().to_numpy(dtype=np.float64))
            names.append(f"m{mi}_quality_roll{win}_mean")
            cols.append(s.rolling(win, min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float64))
            names.append(f"m{mi}_quality_roll{win}_std")
    long_agree = (actions == 1).mean(axis=1)
    short_agree = (actions == 2).mean(axis=1)
    active = actions != 0
    q_active = np.where(active, qualities, 0.0)
    q_abs = np.abs(qualities)
    long_q = np.where(actions == 1, qualities, 0.0).sum(axis=1)
    short_q = np.where(actions == 2, qualities, 0.0).sum(axis=1)
    vote_p = np.stack([(actions == 0).mean(axis=1), long_agree, short_agree], axis=1)
    entropy = -(vote_p * np.log(vote_p + 1e-9)).sum(axis=1) / math.log(3.0)
    h_nonzero = np.where(horizons > 0, horizons, np.nan)
    has_horizon = np.isfinite(h_nonzero).any(axis=1)
    h_sum = np.nan_to_num(h_nonzero, nan=0.0).sum(axis=1)
    h_count = np.maximum(np.isfinite(h_nonzero).sum(axis=1), 1)
    h_mean_raw = np.where(has_horizon, h_sum / h_count, 0.0)
    h_mean = h_mean_raw / 96.0
    h_var = np.zeros(n, dtype=np.float64)
    valid_h = np.flatnonzero(has_horizon)
    if len(valid_h):
        centered = np.where(np.isfinite(h_nonzero[valid_h]), h_nonzero[valid_h] - h_mean_raw[valid_h, None], 0.0)
        h_var[valid_h] = (centered * centered).sum(axis=1) / h_count[valid_h]
    h_std = np.sqrt(h_var) / 96.0
    h_gap = np.abs(horizons[:, 0] - h_mean_raw) / 96.0
    for arr, name in (
        (long_agree, "agreement_long"),
        (short_agree, "agreement_short"),
        (q_active.max(axis=1), "quality_top"),
        (q_abs.std(axis=1), "quality_dispersion"),
        (long_q, "quality_weighted_long"),
        (short_q, "quality_weighted_short"),
        (entropy, "model_disagreement_entropy"),
        (h_mean, "horizon_mean"),
        (h_std, "horizon_std"),
        (h_gap, "primary_horizon_gap"),
    ):
        cols.append(arr)
        names.append(name)
    for i, j, name in (
        (0, 4, "divergence_primary_adverse"),
        (0, 5, "divergence_primary_sam"),
        (1, 4, "divergence_coverage_adverse"),
        (1, 5, "divergence_coverage_sam"),
    ):
        if i < qualities.shape[1] and j < qualities.shape[1]:
            cols.append(qualities[:, i] - qualities[:, j])
            names.append(name)
    for th in (0.0, 0.25, 0.50):
        cols.append(((actions != 0) & (qualities > th)).sum(axis=1).astype(np.float64))
        names.append(f"consensus_active_q_gt_{str(th).replace('.', '_')}")
    for side, side_name in ((1, "long"), (2, "short")):
        for lo, hi, group in ((1, 12, "short"), (13, 48, "mid"), (49, 10_000, "long")):
            mask = (actions == side) & (horizons >= lo) & (horizons <= hi)
            cols.append(np.where(mask, qualities, 0.0).sum(axis=1))
            names.append(f"{group}_{side_name}_edge_score")
    context_cols = [
        ("obi", 0.0),
        ("taker_buy_ratio", 0.5),
        ("nif_whale", 0.0),
        ("eai", 0.0),
        ("oi_delta_pct", 0.0),
        ("oi_change_rate", 0.0),
        ("sig_oi_divergence", 0.0),
        ("funding_rate", 0.0),
        ("last_funding_rate", 0.0),
        ("funding_roc_288", 0.0),
        ("funding_price_divergence", 0.0),
        ("mta_funding", 0.0),
        ("ou_funding_z", 0.0),
        ("rsi", 50.0),
        ("mean_reversion_z", 0.0),
        ("atr14_pct", 0.003),
        ("clean_regime4_state24_sticky090_v2_instability_prob", 0.0),
        ("clean_regime4_state24_sticky090_v2_whipsaw_prob", 0.0),
        ("clean_regime4_state24_sticky090_v2_confidence", 0.0),
        ("regime4_pred_instability_prob", 0.0),
        ("regime4_pred_whipsaw_prob", 0.0),
    ]
    regime_named = [
        c
        for c in frame.columns
        if "regime" in c.lower()
        and c
        not in {
            "clean_regime4_state24_sticky090_v2_instability_prob",
            "clean_regime4_state24_sticky090_v2_whipsaw_prob",
            "clean_regime4_state24_sticky090_v2_confidence",
            "regime4_pred_instability_prob",
            "regime4_pred_whipsaw_prob",
        }
    ]
    for col in sorted(regime_named):
        context_cols.append((col, 0.0))
    for col, default in context_cols:
        vals = _safe_col(frame, col, default)
        if col == "taker_buy_ratio":
            vals = vals - 0.5
        elif col == "rsi":
            vals = (vals - 50.0) / 50.0
        cols.append(vals)
        names.append(col)
    x = np.vstack(cols).T.astype(np.float32)
    assert x.shape[0] == n
    return x, names


@dataclass
class RouterData:
    frame: pd.DataFrame
    preds: list[pd.DataFrame]
    xs: list[np.ndarray]
    bundles: list[dict[str, Any]]
    exit_models: list[list[Any]]
    exit_model_ids: list[np.ndarray]
    base_x: np.ndarray
    base_names: list[str]
    thresholds: np.ndarray
    exit_thresholds: np.ndarray


def _exit_model_for(data: RouterData, expert_idx: int, row_idx: int) -> Any:
    model_id = int(data.exit_model_ids[expert_idx][row_idx])
    return data.exit_models[expert_idx][model_id]


def _load_router_data(variant: str) -> RouterData:
    frame = _load_frame(variant)
    preds: list[pd.DataFrame] = []
    xs: list[np.ndarray] = []
    bundles: list[dict[str, Any]] = []
    exit_models: list[list[Any]] = []
    exit_model_ids: list[np.ndarray] = []
    thresholds: list[float] = []
    exit_thresholds: list[float] = []
    for _, prefix in MODEL_SPECS:
        bundle = joblib.load(f"{prefix}_bundle.joblib")
        summary = json.loads(Path(f"{prefix}_summary.json").read_text())
        pred, x = _predict_bundle(bundle, frame)
        preds.append(pred)
        xs.append(x)
        bundles.append(bundle)
        exit_models.append([bundle["exit_model"]])
        exit_model_ids.append(np.zeros(len(frame), dtype=np.int64))
        thresholds.append(float(summary["best"]["entry_threshold"]))
        exit_thresholds.append(float(summary["best"].get("exit_threshold", 0.55)))
    base_x, names = _build_base_features(frame, preds)
    return RouterData(
        frame,
        preds,
        xs,
        bundles,
        exit_models,
        exit_model_ids,
        base_x,
        names,
        np.asarray(thresholds),
        np.asarray(exit_thresholds),
    )


def _train_expert_fold(
    frame: pd.DataFrame,
    spec_features: list[str],
    fit_pos: np.ndarray,
    *,
    label_preset: str,
    target_head_mode: str,
    iterations: int,
    exit_iterations: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    import argparse as _argparse
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    cfg = _apply_label_preset(EQEConfig(), label_preset)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    x_fit_raw = _numeric_matrix(frame.iloc[fit_pos], spec_features)
    pipe.fit(x_fit_raw)
    x_all = pipe.transform(_numeric_matrix(frame, spec_features))
    valid, y, _ = _build_entry_labels(
        frame,
        cfg,
        stride_bars=3,
        batch_size=1024,
        adaptive_sampling=False,
        label_preset=label_preset,
        session_topk=2,
    )
    fit_set = set(int(v) for v in fit_pos)
    keep = np.asarray([int(v) in fit_set for v in valid], dtype=bool)
    valid_fit = valid[keep]
    y_fit = {k: v[keep] if len(v) == len(valid) else v for k, v in y.items()}
    args = _argparse.Namespace(
        iterations=int(iterations),
        learning_rate=0.055,
        depth=4,
        l2_leaf_reg=5.0,
        exit_iterations=int(exit_iterations),
        exit_learning_rate=0.045,
        exit_depth=5,
        task_type="CPU",
        seed=int(seed),
        verbose=0,
        target_head_mode=target_head_mode,
        fixed_target_horizon=0,
        max_target_horizon=int(cfg.max_train_horizon_bars),
        cash_action_weight=0.35,
    )
    entry_models = _fit_entry_models(x_all[valid_fit], y_fit, args)
    train_dec = _predict_entry(entry_models, x_all, cfg)
    expected = _estimate_expected_return_by_bucket(frame, valid_fit, y_fit, cfg)
    x_exit, y_exit, w_exit, exit_meta = _build_exit_dataset(
        frame,
        x_all,
        valid_fit,
        y_fit,
        train_dec,
        cfg,
        max_samples=1000,
        step=8,
        cost_mult=3.0,
        weight_scale=80.0,
        target_head_mode=target_head_mode,
        expected_return_by_bucket=expected,
    )
    exit_model = _fit_exit_model(x_exit, y_exit, w_exit, args)
    bundle = {
        "config": cfg.__dict__,
        "feature_cols": spec_features,
        "model_features": spec_features,
        "pipeline": pipe,
        "entry_models": entry_models,
        "exit_model": exit_model,
        "exit_meta": exit_meta,
        "expected_return_by_bucket": expected,
    }
    return bundle, x_all


def _load_router_data_oof(
    variant: str,
    *,
    folds: int,
    iterations: int,
    exit_iterations: int,
    purge_bars: int,
    seed: int,
) -> RouterData:
    frame = _load_frame(variant)
    spec = _read_spec(DEFAULT_SPEC_DIR, variant)
    spec_features = [c for c in spec["features"] if c in frame.columns]
    split = frame["dataset_split"].astype(str).str.lower().to_numpy()
    train_pos = np.flatnonzero(split == "train")
    val_pos = np.flatnonzero(split != "train")
    fold_parts = np.array_split(train_pos, int(folds))
    preds: list[pd.DataFrame] = []
    xs: list[np.ndarray] = []
    bundles: list[dict[str, Any]] = []
    exit_models_all: list[list[Any]] = []
    exit_ids_all: list[np.ndarray] = []
    thresholds: list[float] = []
    exit_thresholds: list[float] = []
    for expert_idx, ((_, full_prefix), (preset, mode)) in enumerate(zip(MODEL_SPECS, EXPERT_PRESETS)):
        print(f"[oof] expert={expert_idx} preset={preset} mode={mode}", flush=True)
        full_bundle = joblib.load(f"{full_prefix}_bundle.joblib")
        full_summary = json.loads(Path(f"{full_prefix}_summary.json").read_text())
        full_pred, full_x = _predict_bundle(full_bundle, frame)
        pred = full_pred.copy()
        x_all_rows = np.asarray(full_x, dtype=np.float64).copy()
        exit_models = [full_bundle["exit_model"]]
        exit_ids = np.zeros(len(frame), dtype=np.int64)
        for fold_id, fold_pos in enumerate(fold_parts, start=1):
            lo, hi = int(fold_pos.min()), int(fold_pos.max())
            purge_lo = max(int(train_pos.min()), lo - int(purge_bars))
            purge_hi = min(int(train_pos.max()), hi + int(purge_bars))
            fit_pos = train_pos[(train_pos < purge_lo) | (train_pos > purge_hi)]
            if len(fit_pos) < 5000:
                raise RuntimeError(f"too few OOF fit rows for {preset} fold {fold_id}: {len(fit_pos)}")
            fold_bundle, fold_x_all = _train_expert_fold(
                frame.iloc[train_pos].reset_index(drop=True),
                spec_features,
                np.searchsorted(train_pos, fit_pos),
                label_preset=preset,
                target_head_mode=mode,
                iterations=iterations,
                exit_iterations=exit_iterations,
                seed=seed + 1000 * expert_idx + fold_id,
            )
            local_fold_pos = np.searchsorted(train_pos, fold_pos)
            fold_pred, _ = _predict_bundle(fold_bundle, frame.iloc[fold_pos].reset_index(drop=True))
            for col in pred.columns:
                pred.loc[fold_pos, col] = fold_pred[col].to_numpy()
            x_all_rows[fold_pos] = fold_x_all[local_fold_pos]
            exit_models.append(fold_bundle["exit_model"])
            exit_ids[fold_pos] = len(exit_models) - 1
            print(
                f"[oof] expert={expert_idx} fold={fold_id}/{folds} fit_rows={len(fit_pos)} pred_rows={len(fold_pos)}",
                flush=True,
            )
        preds.append(pred.reset_index(drop=True))
        xs.append(x_all_rows)
        bundles.append(full_bundle)
        exit_models_all.append(exit_models)
        exit_ids_all.append(exit_ids)
        thresholds.append(float(full_summary["best"]["entry_threshold"]))
        exit_thresholds.append(float(full_summary["best"].get("exit_threshold", 0.55)))
    base_x, names = _build_base_features(frame, preds)
    return RouterData(
        frame,
        preds,
        xs,
        bundles,
        exit_models_all,
        exit_ids_all,
        base_x,
        names,
        np.asarray(thresholds),
        np.asarray(exit_thresholds),
    )


def _mode_weights(mode: int) -> np.ndarray:
    w = np.zeros(6, dtype=np.float64)
    if mode == 1:
        w[0] = 1.0
    elif mode == 2:
        w[[0, 1]] = [0.65, 0.35]
    elif mode == 3:
        w[[0, 2]] = [0.65, 0.35]
    elif mode == 4:
        w[[0, 3]] = [0.65, 0.35]
    elif mode == 5:
        w[[0, 1, 2, 3]] = [0.50, 0.20, 0.15, 0.15]
    elif mode == 6:
        w[[0, 4, 5]] = [0.70, 0.15, 0.15]
    elif mode == 7:
        w[[0, 1, 2, 3, 4, 5]] = [0.40, 0.18, 0.16, 0.16, 0.05, 0.05]
    elif mode == 8:
        w[:] = [0.46, 0.22, 0.12, 0.10, 0.05, 0.05]
    elif mode == 9:
        w[:] = [0.50, 0.16, 0.14, 0.12, 0.04, 0.04]
    return w


class Alpha6RouterEnv:
    def __init__(
        self,
        data: RouterData,
        indices: np.ndarray,
        base_mean: np.ndarray,
        base_std: np.ndarray,
        *,
        fee: float = 0.0004,
        slip: float = 0.00015,
        notional: float = 0.25,
        min_exit_hold: int = 2,
        holding_theta: float = 0.00008,
        skip_penalty: float = 0.00015,
        entry_bonus: float = 0.0020,
    ) -> None:
        self.data = data
        self.indices = indices.astype(np.int64)
        self.base_mean = base_mean.astype(np.float32)
        self.base_std = np.where(base_std <= 1e-6, 1.0, base_std).astype(np.float32)
        self.fee = float(fee)
        self.slip = float(slip)
        self.notional = float(notional)
        self.min_exit_hold = int(min_exit_hold)
        self.holding_theta = float(holding_theta)
        self.skip_penalty = float(skip_penalty)
        self.entry_bonus = float(entry_bonus)
        self.close = pd.to_numeric(data.frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
        self.reset()

    @property
    def state_dim(self) -> int:
        return int(self.data.base_x.shape[1] + 8)

    def reset(self) -> np.ndarray:
        self.ptr = 0
        self.i = int(self.indices[self.ptr])
        self.cash = 1.0
        self.peak = 1.0
        self.mdd = 0.0
        self.side = 0
        self.entry_idx = -1
        self.entry_px = 0.0
        self.target_horizon = 0
        self.target_bucket = 0
        self.mae = 0.0
        self.mfe = 0.0
        self.trades = 0
        self.wins = 0
        self.long_entries = 0
        self.short_entries = 0
        self.exit_counts: dict[str, int] = {}
        return self._state()

    def _equity(self, idx: int | None = None) -> float:
        if idx is None:
            idx = self.i
        if self.side == 0:
            return float(self.cash)
        raw = (self.close[idx] - self.entry_px) / max(self.entry_px, 1e-12)
        pnl = raw * self.side * self.notional
        return float(self.cash + pnl)

    def _position_features(self) -> np.ndarray:
        if self.side == 0:
            return np.zeros(8, dtype=np.float32)
        px = self.close[self.i]
        raw = (px - self.entry_px) / max(self.entry_px, 1e-12) * self.side
        hold = max(0, self.i - self.entry_idx)
        giveback = max(0.0, self.mfe - max(raw * self.notional, 0.0))
        return np.asarray(
            [
                self.side,
                hold / max(float(self.target_horizon), 1.0),
                raw,
                raw / max(_safe_col(self.data.frame, "atr14_pct", 0.003)[self.i], 1e-9),
                self.mae,
                self.mfe,
                giveback / max(self.mfe, 1e-9),
                self.target_horizon / 96.0,
            ],
            dtype=np.float32,
        )

    def _state(self) -> np.ndarray:
        base = (self.data.base_x[self.i] - self.base_mean) / self.base_std
        return np.concatenate([base, self._position_features()]).astype(np.float32)

    def snapshot(self) -> dict[str, Any]:
        return {
            "ptr": self.ptr,
            "i": self.i,
            "cash": self.cash,
            "peak": self.peak,
            "mdd": self.mdd,
            "side": self.side,
            "entry_idx": self.entry_idx,
            "entry_px": self.entry_px,
            "target_horizon": self.target_horizon,
            "target_bucket": self.target_bucket,
            "mae": self.mae,
            "mfe": self.mfe,
            "trades": self.trades,
            "wins": self.wins,
            "long_entries": self.long_entries,
            "short_entries": self.short_entries,
            "exit_counts": dict(self.exit_counts),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.ptr = int(state["ptr"])
        self.i = int(state["i"])
        self.cash = float(state["cash"])
        self.peak = float(state["peak"])
        self.mdd = float(state["mdd"])
        self.side = int(state["side"])
        self.entry_idx = int(state["entry_idx"])
        self.entry_px = float(state["entry_px"])
        self.target_horizon = int(state["target_horizon"])
        self.target_bucket = int(state["target_bucket"])
        self.mae = float(state["mae"])
        self.mfe = float(state["mfe"])
        self.trades = int(state["trades"])
        self.wins = int(state["wins"])
        self.long_entries = int(state["long_entries"])
        self.short_entries = int(state["short_entries"])
        self.exit_counts = dict(state["exit_counts"])

    def _signal_flags(self) -> dict[str, Any]:
        actions = np.asarray([int(p.iloc[self.i]["action"]) for p in self.data.preds], dtype=np.int64)
        qualities = np.asarray([float(p.iloc[self.i]["quality"]) for p in self.data.preds], dtype=np.float64)
        active = qualities >= self.data.thresholds
        primary = int(actions[0]) if active[0] and actions[0] != 0 else 0
        risk_opposite = bool(primary != 0 and any(active[j] and actions[j] not in (0, primary) for j in (4, 5)))
        coverage_ok = bool(primary != 0 and active[1] and actions[1] == primary)
        precision_ok = bool(primary != 0 and active[2] and actions[2] == primary)
        perturb_ok = bool(primary != 0 and active[3] and actions[3] == primary)
        confirms = int(coverage_ok) + int(precision_ok) + int(perturb_ok)
        long_votes = float(np.mean(actions == 1))
        short_votes = float(np.mean(actions == 2))
        agreement = max(long_votes, short_votes)
        any_active = bool(np.any(active & (actions != 0)))
        return {
            "actions": actions,
            "qualities": qualities,
            "active": active,
            "primary": primary,
            "risk_opposite": risk_opposite,
            "coverage_ok": coverage_ok,
            "precision_ok": precision_ok,
            "perturb_ok": perturb_ok,
            "confirms": confirms,
            "agreement": agreement,
            "any_active": any_active,
        }

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(len(MODE_NAMES), dtype=bool)
        mask[0] = True
        if self.side != 0:
            mask[:] = True
            return mask
        f = self._signal_flags()
        if f["risk_opposite"]:
            mask[6] = True
            return mask
        if not f["primary"]:
            mask[8] = bool(f["any_active"])
            return mask
        mask[1] = True
        mask[2] = bool(f["coverage_ok"])
        mask[3] = bool(f["precision_ok"])
        mask[4] = bool(f["perturb_ok"])
        mask[5] = bool(f["coverage_ok"] and (f["precision_ok"] or f["perturb_ok"]))
        mask[6] = True
        mask[7] = bool(f["confirms"] >= 2)
        mask[8] = True
        mask[9] = bool(f["confirms"] >= 2 and f["agreement"] >= 0.50)
        return mask

    def _coerce_action(self, action: int) -> int:
        action = int(np.clip(action, 0, len(MODE_NAMES) - 1))
        mask = self.valid_action_mask()
        if mask[action]:
            return action
        return 6 if mask[6] else 0

    def _mode_entry(self, mode: int) -> tuple[int, int, int]:
        if mode == 0:
            return 0, 0, 0
        actions = [int(p.iloc[self.i]["action"]) for p in self.data.preds]
        qualities = np.asarray([float(p.iloc[self.i]["quality"]) for p in self.data.preds])
        active = qualities >= self.data.thresholds
        primary = actions[0] if active[0] and actions[0] != 0 else 0
        if primary == 0 and mode != 7:
            return 0, 0, 0
        veto_opposite = any(active[j] and actions[j] not in (0, primary) for j in (4, 5))
        if mode == 1:
            side = primary
        elif mode == 2:
            side = primary if actions[1] in (0, primary) or (active[1] and actions[1] == primary) else 0
        elif mode == 3:
            side = primary if active[2] and actions[2] == primary else 0
        elif mode == 4:
            side = primary if active[3] and actions[3] == primary else 0
        elif mode == 5:
            confirms = sum(active[j] and actions[j] == primary for j in (1, 2, 3))
            side = primary if confirms >= 2 else 0
        elif mode == 6:
            side = 0 if veto_opposite else primary
        elif mode == 7:
            confirms = sum(active[j] and actions[j] == primary for j in (1, 2, 3))
            side = primary if confirms >= 2 and not veto_opposite else 0
        else:
            w = _mode_weights(mode)
            long_score = sum(w[j] * max(qualities[j], 0.0) for j in range(6) if actions[j] == 1 and active[j])
            short_score = sum(w[j] * max(qualities[j], 0.0) for j in range(6) if actions[j] == 2 and active[j])
            if long_score <= 0 and short_score <= 0:
                return 0, 0, 0
            side = 1 if long_score > short_score else 2
        if side == 0:
            return 0, 0, 0
        hvals = [int(p.iloc[self.i]["target_horizon"]) for p in self.data.preds if int(p.iloc[self.i]["action"]) == side]
        horizon = int(np.clip(np.median(hvals) if hvals else 24, 2, 96))
        bucket = int(_target_horizon_bucket(np.asarray([horizon]))[0])
        return side, horizon, bucket

    def _exit_prob(self, mode: int) -> tuple[float, float]:
        if self.side == 0:
            return 0.0, 1.0
        w = _mode_weights(mode)
        if w.sum() <= 0:
            return 0.0, 1.0
        hold = max(0, self.i - self.entry_idx)
        px = self.close[self.i]
        state = _exit_state_vec(
            self.data.frame,
            side=self.side,
            entry_idx=self.entry_idx,
            current_idx=self.i,
            entry_px=self.entry_px,
            px=px,
            hold=hold,
            horizon=max(self.target_horizon, 2),
            mae=self.mae,
            mfe=self.mfe,
            target_bucket=self.target_bucket,
            expected_return=0.01,
        )
        probs_arr = np.zeros(6, dtype=np.float64)
        for j in np.flatnonzero(w > 0):
            probs_arr[j] = _exit_close_prob(_exit_model_for(self.data, int(j), self.i), self.data.xs[int(j)][self.i], state)
        threshold = float(np.sum(w * self.data.exit_thresholds) / max(w.sum(), 1e-9))
        return float(np.sum(w * probs_arr) / max(w.sum(), 1e-9)), threshold

    def _close(self, reason: str) -> None:
        px = self.close[self.i]
        raw = (px - self.entry_px) / max(self.entry_px, 1e-12) * self.side
        pnl = raw * self.notional - (self.fee + self.slip) * self.notional
        self.cash += pnl
        self.trades += 1
        self.wins += int(pnl > 0)
        self.exit_counts[reason] = self.exit_counts.get(reason, 0) + 1
        self.side = 0
        self.entry_idx = -1
        self.entry_px = 0.0
        self.target_horizon = 0
        self.target_bucket = 0
        self.mae = 0.0
        self.mfe = 0.0

    def _enter(self, action_side: int, horizon: int, bucket: int) -> None:
        self.side = 1 if action_side == 1 else -1
        self.entry_idx = self.i
        self.entry_px = self.close[self.i]
        self.target_horizon = int(horizon)
        self.target_bucket = int(bucket)
        self.mae = 0.0
        self.mfe = 0.0
        self.cash -= (self.fee + self.slip) * self.notional
        if self.side > 0:
            self.long_entries += 1
        else:
            self.short_entries += 1

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        before = self._equity(self.i)
        requested_mode = int(np.clip(action, 0, len(MODE_NAMES) - 1))
        mode = self._coerce_action(requested_mode)
        held_before = self.side != 0
        valid_before = self.valid_action_mask()
        had_entry_option = bool(np.any(valid_before[1:]))
        if self.side != 0:
            raw = (self.close[self.i] - self.entry_px) / max(self.entry_px, 1e-12) * self.side * self.notional
            self.mae = max(self.mae, max(0.0, -raw))
            self.mfe = max(self.mfe, max(0.0, raw))
            close_prob, close_th = self._exit_prob(mode)
            hold = self.i - self.entry_idx
            if hold >= self.min_exit_hold and close_prob >= close_th:
                self._close("exit_model")
        if self.side == 0:
            side, horizon, bucket = self._mode_entry(mode)
            if side != 0:
                self._enter(side, horizon, bucket)
        self.ptr += 1
        done = self.ptr >= len(self.indices) - 1
        if done and self.side != 0:
            self._close("end")
        else:
            self.i = int(self.indices[self.ptr])
        eq = self._equity(self.i)
        self.peak = max(self.peak, eq)
        self.mdd = min(self.mdd, eq / max(self.peak, 1e-12) - 1.0)
        reward = float(np.clip((eq - before) * 100.0, -2.0, 2.0))
        if held_before or self.side != 0:
            reward -= float(self.holding_theta) * max(1.0, float(self.i - self.entry_idx if self.entry_idx >= 0 else 1))
        if (not held_before) and mode == 0 and had_entry_option:
            reward -= self.skip_penalty
        if (not held_before) and self.side != 0:
            reward += self.entry_bonus
        return self._state(), reward, done, {"requested_action": requested_mode, "actual_action": mode}

    def summary(self) -> dict[str, Any]:
        return {
            "pnl": float((self.cash - 1.0) * 100.0),
            "mdd": float(self.mdd * 100.0),
            "trades": int(self.trades),
            "wr": float(self.wins / max(self.trades, 1)),
            "long_entries": int(self.long_entries),
            "short_entries": int(self.short_entries),
            "exits": dict(self.exit_counts),
        }


class Replay:
    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = int(capacity)
        self.buf: list[tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.pos = 0

    def add(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, d: bool) -> None:
        item = (s.copy(), int(a), float(r), ns.copy(), float(d))
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buf, int(n))
        s, a, r, ns, d = zip(*batch)
        rewards = np.asarray(r, dtype=np.float32)
        r_mean = float(rewards.mean())
        r_std = float(max(rewards.std(), 0.05))
        return (
            torch.tensor(np.asarray(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor((rewards - r_mean) / r_std, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.asarray(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buf)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))
        self.q2 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x), self.q2(x)


class DiscreteSAC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        lr: float = 3e-4,
        gamma: float = 0.995,
        tau: float = 0.01,
        alpha_init: float = 0.03,
        alpha_min: float = 0.003,
        alpha_max: float = 0.20,
    ) -> None:
        self.device = torch.device(device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target = copy.deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.log_alpha = torch.tensor([math.log(alpha_init)], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = 0.45 * math.log(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)

    def act(self, state: np.ndarray, deterministic: bool = False, mask: np.ndarray | None = None) -> int:
        x = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.actor(x)
            if mask is not None:
                mask_t = torch.tensor(mask[None, :], dtype=torch.bool, device=self.device)
                logits = logits.masked_fill(~mask_t, -1e9)
            probs = torch.softmax(logits, dim=-1)[0]
        if deterministic:
            return int(torch.argmax(probs).item())
        return int(torch.distributions.Categorical(probs=probs).sample().item())

    def update(self, replay: Replay, batch_size: int) -> dict[str, float]:
        if len(replay) < batch_size:
            return {}
        s, a, r, ns, d, raw_r = replay.sample(batch_size)
        s, a, r, ns, d = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device), d.to(self.device)
        raw_r = raw_r.to(self.device)
        alpha = self.log_alpha.exp().clamp(float(self.alpha_min), float(self.alpha_max))
        with torch.no_grad():
            next_probs = torch.softmax(self.actor(ns), dim=-1)
            next_logp = torch.log(next_probs + 1e-8)
            tq1, tq2 = self.target(ns)
            next_v = (next_probs * (torch.minimum(tq1, tq2) - alpha * next_logp)).sum(dim=1, keepdim=True)
            target_q = r + self.gamma * (1.0 - d) * next_v
        q1, q2 = self.critic(s)
        q1_a = q1.gather(1, a.unsqueeze(1))
        q2_a = q2.gather(1, a.unsqueeze(1))
        critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        probs = torch.softmax(self.actor(s), dim=-1)
        logp = torch.log(probs + 1e-8)
        q1_pi, q2_pi = self.critic(s)
        actor_loss = (probs * (alpha.detach() * logp - torch.minimum(q1_pi, q2_pi))).sum(dim=1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        entropy = -(probs * logp).sum(dim=1, keepdim=True)
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(math.log(self.alpha_min), math.log(self.alpha_max))

        with torch.no_grad():
            for tp, p in zip(self.target.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(alpha.item()),
            "entropy": float(entropy.mean().item()),
            "raw_reward_mean": float(raw_r.mean().item()),
        }


def _run_policy(env: Alpha6RouterEnv, agent: DiscreteSAC | None = None, fixed_action: int | None = None) -> dict[str, Any]:
    s = env.reset()
    actions: list[int] = []
    while True:
        if fixed_action is not None:
            a = int(fixed_action)
        elif agent is not None:
            a = agent.act(s, deterministic=True, mask=env.valid_action_mask())
        else:
            a = 1
        ns, _, done, info = env.step(a)
        actions.append(int(info.get("actual_action", a)))
        s = ns
        if done:
            break
    out = env.summary()
    out["action_counts"] = {MODE_NAMES[k]: int(v) for k, v in pd.Series(actions).value_counts().sort_index().items()}
    return out


def _add_counterfactual_transitions(
    env: Alpha6RouterEnv,
    replay: Replay,
    state: np.ndarray,
    *,
    max_actions: int,
) -> None:
    mask = env.valid_action_mask()
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        return
    if int(max_actions) > 0 and len(candidates) > int(max_actions):
        candidates = np.random.choice(candidates, size=int(max_actions), replace=False)
    snap = env.snapshot()
    for a in candidates:
        env.restore(snap)
        ns_cf, r_cf, d_cf, info_cf = env.step(int(a))
        replay.add(state, int(info_cf.get("actual_action", int(a))), r_cf, ns_cf, d_cf)
    env.restore(snap)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/test a discrete SAC router over six Alpha6 CatBoost experts.")
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_dsac_ensemble_router_20260523")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--updates-per-step", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--oof-folds", type=int, default=0, help="If >1, train CatBoost experts out-of-fold on 2025 train for DSAC training.")
    ap.add_argument("--oof-iterations", type=int, default=250)
    ap.add_argument("--oof-exit-iterations", type=int, default=80)
    ap.add_argument("--oof-purge-bars", type=int, default=96)
    ap.add_argument("--max-train-rows", type=int, default=0, help="Optional tail crop for fast DSAC/router smoke tests.")
    ap.add_argument("--max-val-rows", type=int, default=0, help="Optional head crop for fast validation smoke tests.")
    ap.add_argument(
        "--deterministic-modes",
        default="all",
        help="Comma-separated mode names/ids to baseline before DSAC, 'all', or 'none'.",
    )
    ap.add_argument("--cer-actions", type=int, default=0, help="Counterfactual replay actions per visited state; 0 disables CER.")
    ap.add_argument("--holding-theta", type=float, default=0.00008)
    ap.add_argument("--skip-penalty", type=float, default=0.00015)
    ap.add_argument("--entry-bonus", type=float, default=0.0020)
    ap.add_argument("--alpha-init", type=float, default=0.03)
    ap.add_argument("--alpha-min", type=float, default=0.003)
    ap.add_argument("--alpha-max", type=float, default=0.20)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if int(args.oof_folds) > 1:
        data = _load_router_data_oof(
            args.variant,
            folds=int(args.oof_folds),
            iterations=int(args.oof_iterations),
            exit_iterations=int(args.oof_exit_iterations),
            purge_bars=int(args.oof_purge_bars),
            seed=int(args.seed),
        )
    else:
        data = _load_router_data(args.variant)
    split = data.frame["dataset_split"].astype(str).str.lower().to_numpy()
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split != "train")
    if int(args.max_train_rows) > 0:
        train_idx = train_idx[-int(args.max_train_rows) :]
    if int(args.max_val_rows) > 0:
        val_idx = val_idx[: int(args.max_val_rows)]
    base_mean = data.base_x[train_idx].mean(axis=0)
    base_std = data.base_x[train_idx].std(axis=0)
    env_kwargs = {
        "holding_theta": float(args.holding_theta),
        "skip_penalty": float(args.skip_penalty),
        "entry_bonus": float(args.entry_bonus),
    }
    train_env = Alpha6RouterEnv(data, train_idx, base_mean, base_std, **env_kwargs)
    val_env = Alpha6RouterEnv(data, val_idx, base_mean, base_std, **env_kwargs)

    deterministic = {}
    raw_modes = str(args.deterministic_modes).strip().lower()
    if raw_modes == "all":
        det_ids = list(range(len(MODE_NAMES)))
    elif raw_modes in {"", "none", "skip"}:
        det_ids = []
    else:
        det_ids = []
        for item in raw_modes.split(","):
            item = item.strip()
            if not item:
                continue
            det_ids.append(int(item) if item.isdigit() else MODE_NAMES.index(item))
    for a in det_ids:
        deterministic[MODE_NAMES[a]] = _run_policy(
            Alpha6RouterEnv(data, val_idx, base_mean, base_std, **env_kwargs),
            fixed_action=a,
        )

    agent = DiscreteSAC(
        train_env.state_dim,
        len(MODE_NAMES),
        args.device,
        alpha_init=float(args.alpha_init),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
    )
    replay = Replay()
    step = 0
    last_info: dict[str, float] = {}
    for ep in range(int(args.episodes)):
        s = train_env.reset()
        while True:
            if int(args.cer_actions) > 0:
                _add_counterfactual_transitions(train_env, replay, s, max_actions=int(args.cer_actions))
            if step < int(args.warmup):
                valid = np.flatnonzero(train_env.valid_action_mask())
                a = int(np.random.choice(valid)) if len(valid) else 0
            else:
                a = agent.act(s, deterministic=False, mask=train_env.valid_action_mask())
            ns, r, done, info = train_env.step(a)
            replay.add(s, int(info.get("actual_action", a)), r, ns, done)
            s = ns
            step += 1
            if step >= int(args.warmup):
                for _ in range(int(args.updates_per_step)):
                    info = agent.update(replay, int(args.batch_size))
                    if info:
                        last_info = info
            if done:
                break
        print(f"[dsac-router] episode={ep+1}/{args.episodes} train={train_env.summary()} update={last_info}", flush=True)

    dsac_val = _run_policy(Alpha6RouterEnv(data, val_idx, base_mean, base_std, **env_kwargs), agent=agent)
    result = {
        "model_id": "alpha6_dsac_ensemble_router_20260523",
        "variant": args.variant,
        "mode_names": MODE_NAMES,
        "model_specs": [(name, str(prefix)) for name, prefix in MODEL_SPECS],
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "state_dim": int(train_env.state_dim),
        "base_state_dim": int(data.base_x.shape[1]),
        "base_feature_names": data.base_names,
        "episodes": int(args.episodes),
        "warmup": int(args.warmup),
        "oof_folds": int(args.oof_folds),
        "oof_iterations": int(args.oof_iterations),
        "oof_exit_iterations": int(args.oof_exit_iterations),
        "oof_purge_bars": int(args.oof_purge_bars),
        "deterministic_modes": str(args.deterministic_modes),
        "cer_actions": int(args.cer_actions),
        "holding_theta": float(args.holding_theta),
        "skip_penalty": float(args.skip_penalty),
        "entry_bonus": float(args.entry_bonus),
        "alpha_init": float(args.alpha_init),
        "alpha_min": float(args.alpha_min),
        "alpha_max": float(args.alpha_max),
        "last_update": last_info,
        "deterministic_val": deterministic,
        "dsac_val": dsac_val,
        "audit": {
            "router_train_uses_expert_in_sample_predictions": False if int(args.oof_folds) > 1 else True,
            "router_eval_split": "2025 validation only",
            "fixed24_excluded": True,
            "cost_model": "entry and exit each subtract (fee+slip)*notional; notional=0.25",
            "oof_note": "OOF mode trains each CatBoost expert fold excluding the predicted fold plus purge bars; validation uses full-train expert bundles.",
            "risk_action_mask": "If adverse/sam experts strongly oppose primary while flat, only skip/risk_veto_filtered remain valid.",
            "counterfactual_replay": "Optional CER snapshots the environment and adds one-step transitions for valid alternate modes.",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(), "config": result}, args.out_dir / "dsac_router.pt")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
