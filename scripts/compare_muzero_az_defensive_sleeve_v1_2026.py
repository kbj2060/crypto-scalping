#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, FEATURE_COLS  # noqa: E402
from scripts.compare_muzero_az_vs_dt_lifecycle_2026 import (  # noqa: E402
    _build_zero_style_current,
    _clamp_decisions,
    _date_range,
    _run,
)
from scripts.eval_hf_entry_overlay_grid import _audit  # noqa: E402
from scripts.train_eval_alphazero_style_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_AZ_MODEL  # noqa: E402
from scripts.train_eval_dsac_replacement_heads_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_EXIT_BUNDLE,
    DEFAULT_POLICY,
    DEFAULT_SELECTION,
    DEFAULT_TRAIN_CSV,
    _load_selected,
    _read,
)
from scripts.train_eval_muzero_style_exit_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_MZ_EXIT_MODEL  # noqa: E402
from scripts.train_eval_muzero_style_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_MZ_ENTRY_MODEL, _load_az_exit  # noqa: E402
from scripts.train_eval_zero_style_remaining_layers_2026 import _load_mz_exit, _load_mz_risk, _load_pv  # noqa: E402
from scripts.train_eval_zero_style_risk_overlay_2026 import (  # noqa: E402
    DEFAULT_AZ_RISK_OUT,
    DEFAULT_MZ_RISK_OUT,
    RISK_ACTIONS,
)


DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/muzero_az_defensive_sleeve_v1_2026.json"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/muzero_az_defensive_sleeve_v1"


SLEEVE_EXTRA_COLS = [
    "baseline_action",
    "baseline_side",
    "baseline_notional",
    "baseline_leverage",
    "baseline_position_fraction",
    "baseline_quality",
    "baseline_confidence",
    "baseline_cooldown",
    "baseline_active",
    "recent_resize_pressure",
    "recent_active_rate",
    "notional_to_cap",
    "leverage_to_cap",
]
SLEEVE_FEATURE_COLS = list(FEATURE_COLS) + SLEEVE_EXTRA_COLS


@dataclass(frozen=True)
class DefensiveSleeveConfig:
    horizon: int = 144
    cvar_alpha: float = 0.10
    adverse_threshold: float = 0.020
    min_edge: float = 0.0
    max_train_samples: int = 50000
    seed: int = 42
    min_notional: float = 0.05
    scale_floor: float = 0.35
    hazard_scale_gap: float = 0.15
    cost_scale: float = 0.75


class ConstantBinaryProb:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self, p_one: float):
        self.p_one = float(np.clip(p_one, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        p1 = np.full(n, self.p_one, dtype=np.float64)
        return np.column_stack([1.0 - p1, p1])


def _close(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _limit(df: pd.DataFrame, rows: int | None) -> pd.DataFrame:
    if rows is None or int(rows) <= 0:
        return df.reset_index(drop=True)
    return df.head(int(rows)).reset_index(drop=True)


def _decision_array(dec: pd.DataFrame, col: str, default: float) -> np.ndarray:
    if col not in dec.columns:
        return np.full(len(dec), float(default), dtype=np.float64)
    return pd.to_numeric(dec[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).to_numpy(dtype=np.float64)


def _sleeve_features(
    feat: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    max_notional: float,
    leverage_cap: float,
) -> pd.DataFrame:
    out = feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    action = _decision_array(dec, "action", ACTION_CASH).astype(np.int64)
    side = _decision_array(dec, "side", 0.0)
    notional = _decision_array(dec, "notional_exposure", 0.0)
    leverage = _decision_array(dec, "leverage", 1.0)
    position_fraction = _decision_array(dec, "position_fraction", 0.0)
    quality = _decision_array(dec, "quality_score", 0.0)
    confidence = _decision_array(dec, "confidence", 0.0)
    cooldown = _decision_array(dec, "cooldown_bars", 0.0)
    active = ((action != ACTION_CASH) & (side != 0) & (notional > 0.0)).astype(np.float64)
    notional_s = pd.Series(notional)
    resize = notional_s.diff().abs().fillna(0.0) / np.maximum(notional_s.shift(1).abs().fillna(0.0), 1e-6)
    out["baseline_action"] = action.astype(np.float64)
    out["baseline_side"] = side.astype(np.float64)
    out["baseline_notional"] = notional
    out["baseline_leverage"] = leverage
    out["baseline_position_fraction"] = position_fraction
    out["baseline_quality"] = quality
    out["baseline_confidence"] = confidence
    out["baseline_cooldown"] = cooldown
    out["baseline_active"] = active
    out["recent_resize_pressure"] = resize.shift(1).fillna(0.0).rolling(24, min_periods=1).mean().clip(0.0, 10.0).to_numpy(dtype=np.float64)
    out["recent_active_rate"] = pd.Series(active).shift(1).fillna(0.0).rolling(72, min_periods=1).mean().to_numpy(dtype=np.float64)
    out["notional_to_cap"] = notional / max(float(max_notional), 1e-12)
    out["leverage_to_cap"] = leverage / max(float(leverage_cap), 1e-12)
    return out.reindex(columns=SLEEVE_FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _active_mask(dec: pd.DataFrame) -> np.ndarray:
    action = _decision_array(dec, "action", ACTION_CASH).astype(np.int64)
    side = _decision_array(dec, "side", 0.0)
    notional = _decision_array(dec, "notional_exposure", 0.0)
    return (action != ACTION_CASH) & (side != 0) & (notional > 0.0)


def _diagnostic_targets(
    df: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    cfg: DefensiveSleeveConfig,
    fee: float,
    slip: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    close = _close(df)
    n = len(close)
    side = _decision_array(dec, "side", 0.0).astype(np.int64)
    notional = _decision_array(dec, "notional_exposure", 0.0)
    active = _active_mask(dec)
    usable = np.flatnonzero(active)
    usable = usable[usable < n - int(cfg.horizon) - 2]
    rows: list[dict[str, float]] = []
    for i in usable:
        base = max(float(close[int(i)]), 1e-12)
        fut = close[int(i) + 1 : int(i) + 1 + int(cfg.horizon)]
        if len(fut) == 0:
            continue
        if int(side[int(i)]) > 0:
            raw_path = fut / base - 1.0
        else:
            raw_path = base / np.maximum(fut, 1e-12) - 1.0
        path = raw_path * float(notional[int(i)]) - 2.0 * float(fee + slip) * float(notional[int(i)])
        lower_q = float(np.quantile(path, float(cfg.cvar_alpha)))
        cvar = float(np.mean(path[path <= lower_q])) if np.any(path <= lower_q) else lower_q
        worst = float(np.min(path))
        final_net = float(path[-1])
        hazard = int(worst <= -abs(float(cfg.adverse_threshold)) or final_net < float(cfg.min_edge))
        rows.append(
            {
                "row_idx": float(i),
                "hazard": float(hazard),
                "edge": final_net,
                "lower_quantile": lower_q,
                "cvar": cvar,
                "worst": worst,
            }
        )
    target = pd.DataFrame(rows)
    idx = target["row_idx"].to_numpy(dtype=np.int64) if len(target) else np.zeros(0, dtype=np.int64)
    meta = {
        "horizon": int(cfg.horizon),
        "cvar_alpha": float(cfg.cvar_alpha),
        "adverse_threshold": float(cfg.adverse_threshold),
        "min_edge": float(cfg.min_edge),
        "active_rows": int(active.sum()),
        "usable_rows": int(len(idx)),
        "hazard_rate": float(target["hazard"].mean()) if len(target) else 0.0,
        "edge_quantiles": target["edge"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).round(8).tolist() if len(target) else [],
        "cvar_quantiles": target["cvar"].quantile([0.0, 0.05, 0.25, 0.5, 1.0]).round(8).tolist() if len(target) else [],
    }
    return idx, target, meta


def _fit_hazard(x: np.ndarray, y: np.ndarray, *, seed: int) -> Any:
    if len(x) == 0:
        return ConstantBinaryProb(0.0)
    if len(np.unique(y)) < 2:
        return ConstantBinaryProb(float(np.mean(y)))
    hazard_rate = float(np.mean(y))
    pos_weight = min(8.0, max(0.5, (1.0 - hazard_rate) / max(hazard_rate, 1e-6)))
    sample_weight = np.where(y == 1, pos_weight, 1.0).astype(np.float64)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=160,
            max_leaf_nodes=24,
            l2_regularization=0.05,
            random_state=int(seed),
        ),
    )
    model.fit(x, y.astype(np.int64), histgradientboostingclassifier__sample_weight=sample_weight)
    return model


def _fit_regressor(x: np.ndarray, y: np.ndarray, *, seed: int, loss: str = "squared_error", quantile: float | None = None) -> Any:
    if len(x) == 0:
        model = DummyRegressor(strategy="constant", constant=0.0)
        model.fit(np.zeros((1, len(SLEEVE_FEATURE_COLS)), dtype=np.float32), np.zeros(1, dtype=np.float32))
        return model
    if float(np.nanstd(y)) < 1e-12:
        model = DummyRegressor(strategy="constant", constant=float(np.nanmean(y)))
        model.fit(np.zeros((1, x.shape[1]), dtype=np.float32), np.zeros(1, dtype=np.float32))
        return model
    kwargs: dict[str, Any] = {
        "learning_rate": 0.045,
        "max_iter": 180,
        "max_leaf_nodes": 24,
        "l2_regularization": 0.05,
        "random_state": int(seed),
        "loss": loss,
    }
    if quantile is not None:
        kwargs["quantile"] = float(quantile)
    model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(**kwargs))
    model.fit(x, y.astype(np.float64))
    return model


def _train_sleeve(
    x_all: pd.DataFrame,
    idx: np.ndarray,
    target: pd.DataFrame,
    cfg: DefensiveSleeveConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(idx) != len(target):
        raise ValueError("diagnostic index and target length mismatch")
    rng = np.random.default_rng(int(cfg.seed))
    take = np.arange(len(idx), dtype=np.int64)
    if len(take) > int(cfg.max_train_samples):
        take = np.sort(rng.choice(take, size=int(cfg.max_train_samples), replace=False))
    idx_take = idx[take]
    x = x_all.iloc[idx_take].to_numpy(dtype=np.float32, copy=False)
    y_hazard = target["hazard"].to_numpy(dtype=np.int64)[take]
    y_edge = target["edge"].to_numpy(dtype=np.float64)[take]
    y_lq = target["lower_quantile"].to_numpy(dtype=np.float64)[take]
    y_cvar = target["cvar"].to_numpy(dtype=np.float64)[take]
    y_worst = target["worst"].to_numpy(dtype=np.float64)[take]
    models = {
        "hazard": _fit_hazard(x, y_hazard, seed=int(cfg.seed)),
        "edge": _fit_regressor(x, y_edge, seed=int(cfg.seed) + 1),
        "lower_quantile": _fit_regressor(x, y_lq, seed=int(cfg.seed) + 2, loss="quantile", quantile=float(cfg.cvar_alpha)),
        "cvar": _fit_regressor(x, y_cvar, seed=int(cfg.seed) + 3),
        "worst": _fit_regressor(x, y_worst, seed=int(cfg.seed) + 4, loss="quantile", quantile=0.05),
    }
    meta = {
        "samples": int(len(x)),
        "hazard_labels": int(y_hazard.sum()),
        "hazard_rate": float(y_hazard.mean()) if len(y_hazard) else 0.0,
        "edge_mean": float(np.mean(y_edge)) if len(y_edge) else 0.0,
        "cvar_mean": float(np.mean(y_cvar)) if len(y_cvar) else 0.0,
    }
    return models, meta


def _predict_sleeve(models: dict[str, Any], x: pd.DataFrame) -> dict[str, np.ndarray]:
    arr = x.to_numpy(dtype=np.float32, copy=False)
    hazard_model = models["hazard"]
    proba = hazard_model.predict_proba(arr)
    classes = np.asarray(getattr(hazard_model, "classes_", [0, 1]), dtype=np.int64)
    if 1 in classes:
        hazard = proba[:, int(np.flatnonzero(classes == 1)[0])]
    else:
        hazard = np.zeros(len(arr), dtype=np.float64)
    return {
        "hazard_prob": np.asarray(hazard, dtype=np.float64),
        "edge": np.asarray(models["edge"].predict(arr), dtype=np.float64),
        "lower_quantile": np.asarray(models["lower_quantile"].predict(arr), dtype=np.float64),
        "cvar": np.asarray(models["cvar"].predict(arr), dtype=np.float64),
        "worst": np.asarray(models["worst"].predict(arr), dtype=np.float64),
    }


def _regime_codes(feat: pd.DataFrame) -> np.ndarray:
    cols = ["regime_bull_id", "regime_bear_id", "regime_chop_id", "regime_whipsaw_id", "regime_normal_id"]
    present = [c for c in cols if c in feat.columns]
    if len(present) != len(cols):
        return np.full(len(feat), 4, dtype=np.int64)
    vals = feat[present].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    return np.argmax(vals, axis=1).astype(np.int64)


def _thresholds_for_regime(config: dict[str, Any], regime: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(regime)
    hazard_veto = np.full(n, float(config["hazard_veto"]), dtype=np.float64)
    hazard_scale = np.full(n, float(config["hazard_scale"]), dtype=np.float64)
    max_tail_loss = np.full(n, float(config["max_tail_loss"]), dtype=np.float64)
    if config.get("regime_mode") == "strict_chop":
        strict = np.isin(regime, np.asarray([2, 3], dtype=np.int64))
        hazard_veto[strict] = np.maximum(0.50, hazard_veto[strict] - 0.08)
        hazard_scale[strict] = np.maximum(0.35, hazard_scale[strict] - 0.08)
        max_tail_loss[strict] = np.maximum(0.004, max_tail_loss[strict] * 0.75)
    return hazard_veto, hazard_scale, max_tail_loss


def _apply_defensive_sleeve(
    baseline_dec: pd.DataFrame,
    pred: dict[str, np.ndarray],
    feat: pd.DataFrame,
    config: dict[str, Any],
    cfg: DefensiveSleeveConfig,
    *,
    max_notional: float,
    leverage_cap: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = baseline_dec.copy()
    active = _active_mask(out)
    base_action = _decision_array(out, "action", ACTION_CASH).astype(np.int64)
    base_side = _decision_array(out, "side", 0.0).astype(np.int64)
    base_notional = _decision_array(out, "notional_exposure", 0.0)
    base_leverage = _decision_array(out, "leverage", 1.0)
    base_conf = _decision_array(out, "confidence", 0.0)
    base_quality = _decision_array(out, "quality_score", 0.0)
    resize_pressure = _decision_array(feat, "recent_resize_pressure", 0.0)

    regime = _regime_codes(feat)
    hazard_veto, hazard_scale, max_tail_loss = _thresholds_for_regime(config, regime)
    edge = pred["edge"]
    cvar = pred["cvar"]
    lower_q = pred["lower_quantile"]
    worst = pred["worst"]
    hazard = pred["hazard_prob"]
    edge_floor = float(config["edge_floor"])
    min_confidence = float(config["min_confidence"])
    high_resize_pressure = float(config["high_resize_pressure"])
    leverage_ceiling = float(config["leverage_ceiling"])

    veto = active & (
        (hazard >= hazard_veto)
        | (edge < edge_floor)
        | (cvar <= -max_tail_loss)
        | (lower_q <= -max_tail_loss)
        | (worst <= -np.maximum(float(cfg.adverse_threshold), max_tail_loss))
    )
    scale = np.ones(len(out), dtype=np.float64)
    moderate_hazard = active & ~veto & (hazard >= hazard_scale)
    scale[moderate_hazard] *= np.clip(1.0 - (hazard[moderate_hazard] - hazard_scale[moderate_hazard]) / np.maximum(hazard_veto[moderate_hazard] - hazard_scale[moderate_hazard], 1e-6), float(cfg.scale_floor), 1.0)
    tail_pressure = active & ~veto & ((cvar < -0.5 * max_tail_loss) | (lower_q < -0.5 * max_tail_loss))
    scale[tail_pressure] *= 0.65
    cost_pressure = active & ~veto & (
        (base_conf < min_confidence)
        | (base_quality < edge_floor)
        | (edge < max(edge_floor, 2.0 * float(config["fee_plus_slip"])))
        | (resize_pressure >= high_resize_pressure)
    )
    scale[cost_pressure] *= float(cfg.cost_scale)
    scale = np.clip(scale, 0.0, 1.0)

    notional = np.where(active, np.minimum(base_notional, float(max_notional)) * scale, 0.0)
    notional[veto] = 0.0
    too_small = active & (notional < float(cfg.min_notional))
    notional[too_small] = 0.0
    final_active = active & (notional > 0.0)
    leverage = np.where(final_active, np.clip(base_leverage, 1.0, min(float(leverage_cap), leverage_ceiling)), 1.0)
    base_position_fraction = _decision_array(out, "position_fraction", 0.0)
    max_notional_by_margin = np.where(final_active, np.maximum(base_position_fraction, 0.0) * np.maximum(leverage, 1e-12), 0.0)
    notional = np.where(final_active, np.minimum(notional, max_notional_by_margin), 0.0)
    too_small_after_margin_cap = final_active & (notional < float(cfg.min_notional))
    notional[too_small_after_margin_cap] = 0.0
    final_active = active & (notional > 0.0)
    leverage = np.where(final_active, leverage, 1.0)
    action = np.where(final_active, base_action, ACTION_CASH)
    side = np.where(final_active, base_side, 0)

    out.loc[:, "action"] = action.astype(int)
    out.loc[:, "side"] = side.astype(int)
    out.loc[:, "notional_exposure"] = notional.astype(np.float64)
    out.loc[:, "leverage"] = leverage.astype(np.float64)
    out.loc[:, "position_fraction"] = notional / np.maximum(leverage, 1e-12)
    out.loc[:, "quality_score"] = np.minimum(base_quality, edge).astype(np.float64)
    out.loc[:, "confidence"] = np.minimum(base_conf, 1.0 - hazard).astype(np.float64)
    out = _clamp_decisions(out, max_notional=float(max_notional), leverage_cap=min(float(leverage_cap), leverage_ceiling))
    final_notional = _decision_array(out, "notional_exposure", 0.0)
    final_leverage = _decision_array(out, "leverage", 1.0)
    final_position_fraction = np.minimum(
        _decision_array(out, "position_fraction", 0.0),
        np.maximum(base_position_fraction, 0.0),
    )
    final_notional = np.minimum(final_notional, final_position_fraction * np.maximum(final_leverage, 1e-12))
    flat_after_invariant = final_notional < float(cfg.min_notional)
    out.loc[:, "notional_exposure"] = final_notional
    out.loc[:, "position_fraction"] = final_position_fraction
    out.loc[flat_after_invariant, ["action", "side", "notional_exposure", "position_fraction"]] = 0
    out.loc[flat_after_invariant, "leverage"] = 1.0
    final_active_after = _active_mask(out)
    final_notional_after = _decision_array(out, "notional_exposure", 0.0)
    final_leverage_after = _decision_array(out, "leverage", 1.0)
    telemetry = {
        "baseline_active_rows": int(active.sum()),
        "final_active_rows": int(final_active_after.sum()),
        "veto_rows": int(veto.sum()),
        "scale_down_rows": int((active & ~veto & (scale < 0.999)).sum()),
        "cost_pressure_rows": int(cost_pressure.sum()),
        "tail_pressure_rows": int(tail_pressure.sum()),
        "moderate_hazard_rows": int(moderate_hazard.sum()),
        "avg_notional_scale_active": float(np.mean(np.divide(final_notional_after[active], np.maximum(base_notional[active], 1e-12)))) if active.any() else 0.0,
        "max_leverage_after_cap": float(np.max(final_leverage_after[final_active_after])) if final_active_after.any() else 1.0,
    }
    return out, telemetry


def _invariant_audit(baseline_dec: pd.DataFrame, candidate_dec: pd.DataFrame, cfg: DefensiveSleeveConfig) -> dict[str, Any]:
    base_action = _decision_array(baseline_dec, "action", ACTION_CASH).astype(np.int64)
    base_side = _decision_array(baseline_dec, "side", 0.0).astype(np.int64)
    base_notional = _decision_array(baseline_dec, "notional_exposure", 0.0)
    base_leverage = _decision_array(baseline_dec, "leverage", 1.0)
    base_pf = _decision_array(baseline_dec, "position_fraction", 0.0)
    cand_action = _decision_array(candidate_dec, "action", ACTION_CASH).astype(np.int64)
    cand_side = _decision_array(candidate_dec, "side", 0.0).astype(np.int64)
    cand_notional = _decision_array(candidate_dec, "notional_exposure", 0.0)
    cand_leverage = _decision_array(candidate_dec, "leverage", 1.0)
    cand_pf = _decision_array(candidate_dec, "position_fraction", 0.0)
    baseline_flat = (base_action == ACTION_CASH) | (base_side == 0) | (base_notional <= 0.0)
    side_reversal = (~baseline_flat) & (cand_side != 0) & (cand_side != base_side)
    created_side = baseline_flat & (cand_side != 0)
    notional_increase = cand_notional > base_notional + 1e-9
    leverage_increase = cand_leverage > base_leverage + 1e-9
    position_fraction_increase = cand_pf > base_pf + 1e-9
    invalid_small_active = (cand_notional > 0.0) & (cand_notional < float(cfg.min_notional) - 1e-12)
    nonfinite = ~(np.isfinite(cand_notional) & np.isfinite(cand_leverage) & np.isfinite(cand_pf))
    violations = {
        "created_side": int(created_side.sum()),
        "side_reversal": int(side_reversal.sum()),
        "notional_increase": int(notional_increase.sum()),
        "leverage_increase": int(leverage_increase.sum()),
        "position_fraction_increase": int(position_fraction_increase.sum()),
        "invalid_small_active": int(invalid_small_active.sum()),
        "nonfinite": int(nonfinite.sum()),
    }
    return {
        "passed": bool(sum(violations.values()) == 0),
        "violations": violations,
        "rows": int(len(candidate_dec)),
    }


def _state_audit(x: pd.DataFrame) -> dict[str, Any]:
    arr = x.to_numpy(dtype=np.float64, copy=False)
    missing_cols = [c for c in SLEEVE_FEATURE_COLS if c not in x.columns]
    return {
        "rows": int(len(x)),
        "feature_count": int(len(SLEEVE_FEATURE_COLS)),
        "missing_cols": missing_cols,
        "nan_count_after_fill": int(np.isnan(arr).sum()),
        "nonfinite_count_after_fill": int((~np.isfinite(arr)).sum()),
        "trailing_context_shifted": True,
    }


def _grid_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for leverage_ceiling in (2.0, 2.2, 2.5):
        for hazard_veto in (0.58, 0.66, 0.74):
            for edge_floor in (0.0, 0.0005, 0.0010):
                for max_tail_loss in (0.012, 0.020, 0.035):
                    for regime_mode in ("neutral", "strict_chop"):
                        configs.append(
                            {
                                "leverage_ceiling": float(leverage_ceiling),
                                "hazard_veto": float(hazard_veto),
                                "hazard_scale": float(max(0.35, hazard_veto - float(args.hazard_scale_gap))),
                                "edge_floor": float(edge_floor),
                                "max_tail_loss": float(max_tail_loss),
                                "min_confidence": float(args.min_confidence),
                                "high_resize_pressure": float(args.high_resize_pressure),
                                "fee_plus_slip": float(args.fee + args.slip),
                                "regime_mode": regime_mode,
                            }
                        )
    return configs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare frozen MuZero/AZ current stack with defensive-only diagnostic sleeve v1.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--mz-entry-model", type=Path, default=DEFAULT_MZ_ENTRY_MODEL)
    p.add_argument("--az-model", type=Path, default=DEFAULT_AZ_MODEL)
    p.add_argument("--az-risk-model", type=Path, default=DEFAULT_AZ_RISK_OUT)
    p.add_argument("--mz-risk-model", type=Path, default=DEFAULT_MZ_RISK_OUT)
    p.add_argument("--mz-exit-model", type=Path, default=DEFAULT_MZ_EXIT_MODEL)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--validation-start", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--max-notional", type=float, default=None)
    p.add_argument("--leverage-cap", type=float, default=5.0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--horizon", type=int, default=144)
    p.add_argument("--cvar-alpha", type=float, default=0.10)
    p.add_argument("--adverse-threshold", type=float, default=0.020)
    p.add_argument("--max-train-samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mdd-weight", type=float, default=3.0)
    p.add_argument("--stage2-gamma", type=float, default=0.55)
    p.add_argument("--stage2-prior", type=float, default=0.0)
    p.add_argument("--stage2-depth", type=int, default=1)
    p.add_argument("--stage2-score-floor", type=float, default=0.12)
    p.add_argument("--hazard-scale-gap", type=float, default=0.15)
    p.add_argument("--min-confidence", type=float, default=0.35)
    p.add_argument("--high-resize-pressure", type=float, default=0.35)
    p.add_argument("--limit-train-rows", type=int, default=None, help="Development/smoke only: cap post-split train rows.")
    p.add_argument("--limit-val-rows", type=int, default=None, help="Development/smoke only: cap validation rows.")
    p.add_argument("--limit-eval-rows", type=int, default=None, help="Development/smoke only: cap eval rows.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(args.seed))

    policy = joblib.load(args.policy)
    entry_cfg, risk_cfg, exit_cfg = _load_selected(args.selection_report)
    max_notional = float(args.max_notional if args.max_notional is not None else risk_cfg.get("max_notional", entry_cfg.get("max_notional", 3.6)))
    entry_cfg = dict(entry_cfg)
    risk_cfg = dict(risk_cfg)
    exit_cfg = dict(exit_cfg)
    entry_cfg["max_notional"] = max_notional
    risk_cfg["max_notional"] = max_notional

    train_all = _read(args.train_csv)
    eval_df = _limit(_read(args.eval_csv), args.limit_eval_rows)
    split_ts = pd.Timestamp(args.validation_start)
    ts = pd.to_datetime(train_all["timestamp"], errors="coerce") if "timestamp" in train_all.columns else pd.Series(np.arange(len(train_all)))
    train_df = _limit(train_all.loc[ts < split_ts].reset_index(drop=True), args.limit_train_rows)
    val_df = _limit(train_all.loc[ts >= split_ts].reset_index(drop=True), args.limit_val_rows)

    mz_entry = __import__("scripts.train_eval_zero_style_risk_overlay_2026", fromlist=["_load_mz_entry"])._load_mz_entry(args.mz_entry_model, device)
    az_risk = _load_pv(args.az_risk_model, len(RISK_ACTIONS), RISK_ACTIONS, device)
    mz_risk = _load_mz_risk(args.mz_risk_model, device)
    az_exit = _load_az_exit(args.az_model, device)
    if az_exit is None:
        raise FileNotFoundError(f"AZ exit model not found: {args.az_model}")
    _ = _load_mz_exit(args.mz_exit_model, device)

    current_kwargs = dict(
        policy=policy,
        entry_cfg=entry_cfg,
        mz_entry=mz_entry,
        az_risk=az_risk,
        mz_risk=mz_risk,
        device=device,
        max_notional=max_notional,
        leverage_cap=float(args.leverage_cap),
        stage2_gamma=float(args.stage2_gamma),
        stage2_prior=float(args.stage2_prior),
        stage2_depth=int(args.stage2_depth),
        stage2_score_floor=float(args.stage2_score_floor),
    )
    train_current_pre = _build_zero_style_current(train_df, **current_kwargs)
    val_current_pre = _build_zero_style_current(val_df, **current_kwargs)
    eval_current_pre = _build_zero_style_current(eval_df, **current_kwargs)
    zero_exit_cfg = {"exit_threshold": 0.45, "min_exit_age": int(exit_cfg["min_exit_age"])}
    zero_val = _run("current_muzero_az_val", val_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, val_current_pre, fee=args.fee, slip=args.slip, mdd_weight=args.mdd_weight)
    zero_eval = _run("current_muzero_az_eval", eval_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, eval_current_pre, fee=args.fee, slip=args.slip, monthly=True, mdd_weight=args.mdd_weight)

    cfg = DefensiveSleeveConfig(
        horizon=int(args.horizon),
        cvar_alpha=float(args.cvar_alpha),
        adverse_threshold=float(args.adverse_threshold),
        min_edge=0.0,
        max_train_samples=int(args.max_train_samples),
        seed=int(args.seed),
        hazard_scale_gap=float(args.hazard_scale_gap),
    )
    train_feat, train_dec, _, _ = train_current_pre
    val_feat, val_dec, val_close, val_fill = val_current_pre
    eval_feat, eval_dec, eval_close, eval_fill = eval_current_pre
    train_x = _sleeve_features(train_feat, train_dec, max_notional=max_notional, leverage_cap=float(args.leverage_cap))
    val_x = _sleeve_features(val_feat, val_dec, max_notional=max_notional, leverage_cap=float(args.leverage_cap))
    eval_x = _sleeve_features(eval_feat, eval_dec, max_notional=max_notional, leverage_cap=float(args.leverage_cap))

    target_idx, target, target_meta = _diagnostic_targets(train_df, train_dec, cfg=cfg, fee=float(args.fee), slip=float(args.slip))
    models, train_meta = _train_sleeve(train_x, target_idx, target, cfg)
    val_pred = _predict_sleeve(models, val_x)
    eval_pred = _predict_sleeve(models, eval_x)

    grid: list[dict[str, Any]] = []
    telemetry_by_key: dict[str, Any] = {}
    for i, sleeve_cfg in enumerate(_grid_configs(args)):
        val_candidate_dec, telemetry = _apply_defensive_sleeve(
            val_dec,
            val_pred,
            val_x,
            sleeve_cfg,
            cfg,
            max_notional=max_notional,
            leverage_cap=float(args.leverage_cap),
        )
        val_pre = (val_feat, val_candidate_dec, val_close, val_fill)
        row = _run(
            f"defensive_sleeve_v1_grid_{i:03d}_val",
            val_df,
            policy,
            az_exit,
            entry_cfg,
            risk_cfg,
            zero_exit_cfg,
            val_pre,
            fee=float(args.fee),
            slip=float(args.slip),
            mdd_weight=float(args.mdd_weight),
        )
        row["config"] = sleeve_cfg
        row["telemetry"] = telemetry
        grid.append(row)
        telemetry_by_key[str(i)] = telemetry
    selected = sorted(grid, key=lambda r: float(r["score"]), reverse=True)[0]
    selected_cfg = dict(selected["config"])
    val_selected_dec, _ = _apply_defensive_sleeve(
        val_dec,
        val_pred,
        val_x,
        selected_cfg,
        cfg,
        max_notional=max_notional,
        leverage_cap=float(args.leverage_cap),
    )
    eval_candidate_dec, eval_telemetry = _apply_defensive_sleeve(
        eval_dec,
        eval_pred,
        eval_x,
        selected_cfg,
        cfg,
        max_notional=max_notional,
        leverage_cap=float(args.leverage_cap),
    )
    eval_candidate_pre = (eval_feat, eval_candidate_dec, eval_close, eval_fill)
    candidate_eval = _run(
        "muzero_az_defensive_sleeve_v1_eval",
        eval_df,
        policy,
        az_exit,
        entry_cfg,
        risk_cfg,
        zero_exit_cfg,
        eval_candidate_pre,
        fee=float(args.fee),
        slip=float(args.slip),
        monthly=True,
        mdd_weight=float(args.mdd_weight),
    )
    eval_invariant_audit = _invariant_audit(eval_dec, eval_candidate_dec, cfg)
    val_invariant_audit = _invariant_audit(val_dec, val_selected_dec, cfg)

    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = [
            _run("current_muzero_az", eval_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, eval_current_pre, fee=float(args.fee) * mult, slip=float(args.slip) * mult, mdd_weight=args.mdd_weight),
            _run("muzero_az_defensive_sleeve_v1", eval_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, eval_candidate_pre, fee=float(args.fee) * mult, slip=float(args.slip) * mult, mdd_weight=args.mdd_weight),
        ]

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "type": "muzero_az_defensive_sleeve_v1",
            "models": models,
            "feature_cols": SLEEVE_FEATURE_COLS,
            "config": asdict(cfg),
            "selected_config": selected_cfg,
            "target_meta": target_meta,
            "train_meta": train_meta,
        },
        args.model_dir / "defensive_sleeve_v1.pkl",
    )
    selector_payload = {
        "type": "muzero_az_defensive_sleeve_v1_regime_threshold_selector",
        "selected_config": selected_cfg,
        "validation_score": selected.get("score"),
        "validation_eval": selected.get("eval"),
        "regime_modes": ["neutral", "strict_chop"],
        "hard_leverage_governor_candidates": [2.0, 2.2, 2.5],
    }
    (args.model_dir / "regime_threshold_selector.json").write_text(json.dumps(selector_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    calibration_payload = {
        "type": "muzero_az_defensive_sleeve_v1_calibration_report",
        "target_meta": target_meta,
        "train_meta": train_meta,
        "state_audit": {
            "train": _state_audit(train_x),
            "validation": _state_audit(val_x),
            "eval": _state_audit(eval_x),
        },
        "invariant_audit": {
            "validation_selected": val_invariant_audit,
            "eval_selected": eval_invariant_audit,
        },
    }
    (args.model_dir / "calibration_report.json").write_text(json.dumps(calibration_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "type": "muzero_az_defensive_sleeve_v1_2026",
        "note": "Defensive-only sleeve over frozen Current MuZero Entry Planner + AZ Risk Overlay + Stage2 MuZero Sleeve + AZ Exit Governor. The sleeve never creates a new side/action; it only vetoes, scales down, preserves/lowers direction, and caps leverage.",
        "policy": str(args.policy),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "model_dir": str(args.model_dir),
        "report_out": str(args.report_out),
        "audit": {
            "source_audit": _audit(args.train_csv, args.eval_csv, policy),
            "train_range": _date_range(train_df),
            "validation_range": _date_range(val_df),
            "eval_range": _date_range(eval_df),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "eval_rows": int(len(eval_df)),
            "smoke_limits": {
                "limit_train_rows": args.limit_train_rows,
                "limit_val_rows": args.limit_val_rows,
                "limit_eval_rows": args.limit_eval_rows,
            },
        },
        "cost_and_caps": {
            "fee": float(args.fee),
            "slip": float(args.slip),
            "max_notional": float(max_notional),
            "original_leverage_cap": float(args.leverage_cap),
            "hard_leverage_governor_candidates": [2.0, 2.2, 2.5],
        },
        "frozen_current_config": {
            "entry": "MuZero entry planner",
            "risk": "AZ risk overlay",
            "stage2": {
                "model": "MuZero sleeve overlay",
                "gamma": float(args.stage2_gamma),
                "prior": float(args.stage2_prior),
                "depth": int(args.stage2_depth),
                "score_floor": float(args.stage2_score_floor),
            },
            "exit": {"model": "AZ exit governor", "threshold": 0.45, "min_exit_age": int(exit_cfg["min_exit_age"])},
        },
        "candidate_config": {
            "architecture": [
                "Lifecycle Hazard / DT Diagnostic Head",
                "Calibrated Quantile + CVaR Tail Head",
                "Cost / Turnover Monitor",
                "Regime Threshold Selector",
                "Hard Leverage Governor",
            ],
            "diagnostic": asdict(cfg),
            "selected_config": selected_cfg,
            "forbidden_gate_check": "No min_lower_edge < 0 gate is used; edge_floor grid is non-negative only.",
        },
        "target_meta": target_meta,
        "train_meta": train_meta,
        "validation": {
            "current_muzero_az": zero_val,
            "grid_ranked_top20": sorted(grid, key=lambda r: float(r["score"]), reverse=True)[:20],
            "selected_candidate": selected,
        },
        "eval": {
            "current_muzero_az": zero_eval,
            "muzero_az_defensive_sleeve_v1": candidate_eval,
            "defensive_telemetry": eval_telemetry,
            "delta": {
                "pnl": float(candidate_eval["eval"]["pnl"] - zero_eval["eval"]["pnl"]),
                "mdd": float(candidate_eval["eval"]["mdd"] - zero_eval["eval"]["mdd"]),
                "trades": int(candidate_eval["eval"]["trades"] - zero_eval["eval"]["trades"]),
                "trades_per_day": float(candidate_eval["eval"]["trades_per_day"] - zero_eval["eval"]["trades_per_day"]),
                "avg_leverage": float(candidate_eval["eval"]["avg_leverage"] - zero_eval["eval"]["avg_leverage"]),
            },
        },
        "state_audit": calibration_payload["state_audit"],
        "invariant_audit": calibration_payload["invariant_audit"],
        "calibration_audit": {
            "target_meta": target_meta,
            "train_meta": train_meta,
            "selector_artifact": str(args.model_dir / "regime_threshold_selector.json"),
            "calibration_artifact": str(args.model_dir / "calibration_report.json"),
        },
        "monthly": {
            "current_muzero_az": zero_eval.get("monthly", {}),
            "muzero_az_defensive_sleeve_v1": candidate_eval.get("monthly", {}),
        },
        "cost_stress": cost_stress,
        "red_team_required": [
            "Training labels use forward path diagnostics; run OOF/embargo leakage audit before promotion.",
            "This sleeve is defensive-only, but validation still selects thresholds; require walk-forward stability by week/month.",
            "Backtest reuses the same no-limit accounting path; funding, liquidation proximity, and margin path remain approximations.",
            "Confirm no baseline action inversion in downstream live wiring; sleeve may only keep same side or flat.",
            "Run fee/slippage stress and resize accounting audit before live shadow deployment.",
        ],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "model_dir": str(args.model_dir),
                "current": zero_eval["eval"],
                "candidate": candidate_eval["eval"],
                "delta": report["eval"]["delta"],
                "selected_config": selected_cfg,
                "defensive_telemetry": eval_telemetry,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
