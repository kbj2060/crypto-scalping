#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_5head_policy_20260522 import (  # noqa: E402
    DEFAULT_FEATURE_CSV,
    DEFAULT_LABEL_DIR,
    DEFAULT_SPEC_DIR,
    _days,
    _feature_matrix,
    _fill_price,
    _json_default,
    _label_frame,
    _read_feature_frame,
    _read_spec,
)


MODEL_ID = "alpha6_catboost_entry_quality_exit_policy_20260522"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_entry_quality_exit_current_tail111_20260522"
TARGET_BUCKET_TO_HORIZON = {0: 6, 1: 12, 2: 24, 3: 48, 4: 96}

CONTEXT_COLS = [
    "obi",
    "taker_buy_ratio",
    "nif_whale",
    "eai",
    "oi_delta_pct",
    "funding_rate",
    "clean_regime4_state24_sticky090_v2_instability_prob",
    "clean_regime4_state24_sticky090_v2_whipsaw_prob",
    "clean_regime4_state24_sticky090_v2_confidence",
    "regime4_pred_instability_prob",
    "regime4_pred_whipsaw_prob",
]

EXIT_STATE_FEATURES = [
    "side",
    "ret",
    "hold_frac",
    "remaining_frac",
    "target_horizon_frac",
    "mae",
    "mfe",
    "giveback",
    "giveback_ratio",
    "current_atr_pct",
    "entry_atr_pct",
    "ret_atr",
    "mae_atr",
    "mfe_atr",
    "side_obi",
    "side_obi_delta",
    "side_taker_delta",
    "side_nif_whale",
    "side_nif_whale_delta",
    "side_eai",
    "side_eai_delta",
    "side_oi_delta_pct",
    "side_funding_rate",
    "risk_off_prob",
    "whipsaw_prob",
    "regime_confidence",
    "target_bucket",
]


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    """Select thresholds by robust Cost3 Calmar; trade count is only a statistical floor."""
    if int(c3["trades"]) < 30:
        return -1e6 + float(c3["pnl"])
    return float(c3["pnl"]) / max(abs(float(c3["mdd"])), 1e-12)


def _parse_cost_multipliers(raw: str) -> tuple[int, ...]:
    vals = tuple(sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()}))
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"invalid eval cost multipliers: {raw!r}")
    return vals


def _exit_state_features(*, regime_drift: bool = False, capture_ratio: bool = False) -> list[str]:
    features = list(EXIT_STATE_FEATURES)
    if regime_drift:
        features.extend(
            [
                "risk_off_delta",
                "whipsaw_delta",
                "regime_confidence_delta",
                "risk_mode_flipped",
            ]
        )
    if capture_ratio:
        features.extend(["capture_ratio", "mfe_expected_ratio"])
    return features


@dataclass(frozen=True)
class EQEConfig:
    fixed_notional: float = 0.25
    max_train_horizon_bars: int = 96
    score_horizons: tuple[int, ...] = (6, 12, 24, 48, 96)
    fee: float = 0.0004
    slip: float = 0.00015
    cash_score: float = 0.0008
    min_net_edge: float = 0.00025
    dynamic_min_edge_atr_frac: float = 0.08
    direction_margin: float = 0.00015
    terminal_weight: float = 0.50
    mfe_weight: float = 0.50
    mae_penalty_lambda: float = 0.70
    path_vol_penalty_lambda: float = 0.06
    hold_penalty: float = 0.004
    exit_adverse_penalty: float = 1.20
    exit_giveback_penalty: float = 0.35


class _ConstantClassifier:
    def __init__(self, cls: int) -> None:
        self.cls = int(cls)
        self.classes_ = np.asarray([self.cls], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.ones((len(x), 1), dtype=np.float64)


class _ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value, dtype=np.float64)


def _target_horizon_bucket(horizon: np.ndarray | int) -> np.ndarray:
    h = np.asarray(horizon, dtype=np.int64)
    buckets = np.full(h.shape, 4, dtype=np.int64)
    buckets = np.where(h <= 6, 0, buckets)
    buckets = np.where((h > 6) & (h <= 12), 1, buckets)
    buckets = np.where((h > 12) & (h <= 24), 2, buckets)
    buckets = np.where((h > 24) & (h <= 48), 3, buckets)
    return buckets.astype(np.int64)


def _bucket_horizon(bucket: int) -> int:
    return int(TARGET_BUCKET_TO_HORIZON.get(int(bucket), 96))


def _parse_bucket_thresholds(value: str) -> tuple[float, float, float, float, float]:
    parts = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if len(parts) != 5:
        raise ValueError(f"bucket threshold config must contain 5 comma-separated values, got: {value!r}")
    return tuple(parts)  # type: ignore[return-value]


def _threshold_for_bucket(exit_threshold: float | tuple[float, ...], bucket: int) -> float:
    if isinstance(exit_threshold, tuple):
        return float(exit_threshold[int(np.clip(bucket, 0, len(exit_threshold) - 1))])
    return float(exit_threshold)


def _frame_value(frame: pd.DataFrame, col: str, idx: int, default: float = 0.0) -> float:
    if col not in frame.columns:
        return float(default)
    val = pd.to_numeric(frame[col], errors="coerce").iloc[int(np.clip(idx, 0, len(frame) - 1))]
    if pd.isna(val) or not np.isfinite(float(val)):
        return float(default)
    return float(val)


def _robust_ood_score(frame: pd.DataFrame) -> np.ndarray:
    cols = [
        "atr14_pct",
        "realized_vol_ratio",
        "volatility_z",
        "jump_z",
        "obi",
        "cvp_volume_imbalance",
        "taker_buy_ratio",
        "funding_price_divergence",
        "ou_funding_z",
        "clean_regime4_state24_sticky090_v2_transition_risk",
        "clean_regime4_state24_sticky090_v2_instability_prob",
        "clean_regime4_state24_sticky090_v2_whipsaw_prob",
    ]
    scores: list[np.ndarray] = []
    for col in cols:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(dtype=np.float64)
        med = float(np.nanmedian(vals))
        mad = float(np.nanmedian(np.abs(vals - med)))
        scale = max(1.4826 * mad, 1e-9)
        scores.append(np.abs(vals - med) / scale)
    if not scores:
        return np.zeros(len(frame), dtype=np.float64)
    return np.nanmax(np.vstack(scores), axis=0)


def _score_direction_paths(
    path: np.ndarray,
    cfg: EQEConfig,
    horizon_idx: list[int],
    max_horizon: int,
    *,
    cost: float,
) -> dict[str, np.ndarray]:
    n = int(path.shape[0])
    best_score = np.full(n, -np.inf, dtype=np.float64)
    best_horizon = np.full(n, int(cfg.score_horizons[-1]), dtype=np.int64)
    best_mae = np.full(n, np.inf, dtype=np.float64)
    best_mfe = np.zeros(n, dtype=np.float64)
    best_vol = np.zeros(n, dtype=np.float64)
    for hi in horizon_idx:
        sub = path[:, : hi + 1]
        terminal = sub[:, -1]
        mfe = np.max(sub, axis=1)
        mae = np.maximum(0.0, -np.min(sub, axis=1))
        vol = np.nanstd(sub, axis=1)
        score = (
            (float(cfg.terminal_weight) * terminal + float(cfg.mfe_weight) * mfe) * float(cfg.fixed_notional)
            - float(cfg.mae_penalty_lambda) * mae * float(cfg.fixed_notional)
            - float(cfg.path_vol_penalty_lambda) * vol * float(cfg.fixed_notional)
            - float(cfg.hold_penalty) * ((hi + 1.0) / max(float(max_horizon), 1.0))
            - cost
        )
        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_horizon = np.where(better, hi + 1, best_horizon)
        best_mae = np.where(better, mae, best_mae)
        best_mfe = np.where(better, mfe, best_mfe)
        best_vol = np.where(better, vol, best_vol)
    return {
        "score": best_score,
        "horizon": best_horizon,
        "mae": best_mae,
        "mfe": best_mfe,
        "vol": best_vol,
    }


def _entry_event_indices(
    frame: pd.DataFrame,
    *,
    max_horizon: int,
    event_quantile: float,
    max_extra: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    limit = max(0, len(frame) - int(max_horizon) - 1)
    if limit <= 0 or int(max_extra) <= 0:
        return np.asarray([], dtype=np.int64), {"event_counts": {}, "max_extra": int(max_extra)}

    def series(col: str, default: float = 0.0) -> pd.Series:
        if col not in frame.columns:
            return pd.Series(np.full(len(frame), float(default), dtype=np.float64))
        return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().fillna(float(default))

    def high_abs(col: str) -> pd.Series:
        s = series(col).abs()
        if float(s.max()) <= 0.0:
            return pd.Series(np.zeros(len(frame), dtype=bool))
        return s >= float(s.iloc[:limit].quantile(float(event_quantile)))

    def high(col: str) -> pd.Series:
        s = series(col)
        if float(s.max()) <= 0.0:
            return pd.Series(np.zeros(len(frame), dtype=bool))
        return s >= float(s.iloc[:limit].quantile(float(event_quantile)))

    regime_cols = [
        "clean_regime4_state24_sticky090_v2_risk_off_prob",
        "clean_regime4_state24_sticky090_v2_whipsaw_prob",
        "clean_regime4_state24_sticky090_v2_instability_prob",
        "clean_regime4_state24_sticky090_v2_confidence",
        "clean_regime4_state24_sticky090_v2_transition_risk",
    ]
    regime_delta = pd.Series(np.zeros(len(frame), dtype=np.float64))
    for col in regime_cols:
        regime_delta = np.maximum(regime_delta, series(col).diff().abs().fillna(0.0))
    regime_transition = regime_delta >= float(regime_delta.iloc[:limit].quantile(float(event_quantile)))
    regime_transition = regime_transition | high("clean_regime4_state24_sticky090_v2_transition_risk")

    volatility_expansion = high("atr14_pct") | high("garch_vol_z") | high("volatility_z") | high("realized_vol_ratio") | high_abs("jump_z")
    trend_breakout = (
        high_abs("breakout_strength")
        | high_abs("dual_momentum")
        | high_abs("trend_accel")
        | (high("sig_trend_health") & high("sig_volume_confirm"))
    )
    flow_extreme = high_abs("obi") | high_abs("cvp_volume_imbalance") | high_abs("nif_whale") | high_abs("eai")
    if "taker_buy_ratio" in frame.columns:
        flow_extreme = flow_extreme | ((series("taker_buy_ratio", 0.5) - 0.5).abs() >= float((series("taker_buy_ratio", 0.5) - 0.5).abs().iloc[:limit].quantile(float(event_quantile))))
    funding_event = high_abs("funding_price_divergence") | high_abs("ou_funding_z") | high_abs("last_funding_rate")
    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], errors="coerce")
        minute_of_day = ts.dt.hour.fillna(0).astype(int) * 60 + ts.dt.minute.fillna(0).astype(int)
        funding_window = pd.Series(np.zeros(len(frame), dtype=bool))
        for hour in (0, 8, 16):
            center = hour * 60
            pre = (center - 30) % (24 * 60)
            post = center + 15
            if pre > center:
                funding_window = funding_window | (minute_of_day >= pre) | (minute_of_day <= post)
            else:
                funding_window = funding_window | ((minute_of_day >= pre) & (minute_of_day <= post))
        funding_event = funding_event | funding_window

    masks = {
        "regime_transition": regime_transition,
        "volatility_expansion": volatility_expansion,
        "trend_breakout": trend_breakout,
        "flow_extreme": flow_extreme,
        "funding_event": funding_event,
    }
    combined = pd.Series(np.zeros(len(frame), dtype=bool))
    event_counts: dict[str, int] = {}
    for name, mask in masks.items():
        clean = mask.fillna(False).iloc[:limit].astype(bool)
        event_counts[name] = int(clean.sum())
        combined.iloc[:limit] = combined.iloc[:limit] | clean

    idx = np.flatnonzero(combined.iloc[:limit].to_numpy(dtype=bool)).astype(np.int64)
    if idx.size > int(max_extra):
        take = np.linspace(0, idx.size - 1, int(max_extra), dtype=np.int64)
        idx = idx[take]
    meta = {
        "event_quantile": float(event_quantile),
        "max_extra": int(max_extra),
        "event_counts": event_counts,
        "selected_extra_before_union": int(idx.size),
    }
    return idx, meta


def _build_entry_labels(
    frame: pd.DataFrame,
    cfg: EQEConfig,
    *,
    stride_bars: int,
    batch_size: int,
    adaptive_sampling: bool = False,
    event_quantile: float = 0.85,
    max_extra: int = 12000,
    label_preset: str = "current_quality",
    session_topk: int = 2,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    open_px = pd.to_numeric(frame.get("open", frame["close"]), errors="coerce").ffill().to_numpy(dtype=np.float64)
    atr_src = frame["atr14_pct"] if "atr14_pct" in frame.columns else pd.Series(0.003, index=frame.index)
    atr = pd.to_numeric(atr_src, errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    preset = str(label_preset).strip().lower()
    regime_preset = preset == "regime_conditional"
    perturbation_preset = preset in {
        "perturbation_robust",
        "sam_conformal",
        "short_horizon_robust",
        "high_precision_robust",
        "turnover_balanced_robust",
        "diffusion_stress_proxy",
    }
    adverse_preset = preset in {
        "adverse_conformal",
        "sam_conformal",
        "short_horizon_robust",
        "high_precision_robust",
        "diffusion_stress_proxy",
    }
    ood_preset = preset == "ts2vec_ood_proxy"
    psr_preset = preset == "psr_path_quality"
    ts2vec_preset = preset == "ts2vec_ood"
    cost_preset = preset == "cost_beta_neutral"
    mamba_preset = preset == "mamba_regime_filter"
    timegrad_preset = preset == "timegrad_mc"
    timellm_preset = preset == "timellm_uncertainty"
    adverse_atr_limit = {
        "adverse_conformal": 1.20,
        "sam_conformal": 1.10,
        "short_horizon_robust": 1.15,
        "high_precision_robust": 0.95,
        "diffusion_stress_proxy": 1.05,
    }.get(preset, 1.25)
    def _optional_col(name: str, default: float) -> pd.Series:
        if name in frame.columns:
            return frame[name]
        return pd.Series(float(default), index=frame.index)

    ood_score = _robust_ood_score(frame) if ood_preset else np.zeros(len(frame), dtype=np.float64)
    ts2vec_ood = pd.to_numeric(_optional_col("rep_ts2vec_ood_z", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    cost_beta = pd.to_numeric(_optional_col("rep_cost_beta_neutral_score", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64)
    mamba_toxic = pd.to_numeric(_optional_col("rep_mamba_toxic_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    timegrad_long = pd.to_numeric(_optional_col("rep_timegrad_long_win_prob", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64)
    timegrad_short = pd.to_numeric(_optional_col("rep_timegrad_short_win_prob", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64)
    timegrad_uncert = pd.to_numeric(_optional_col("rep_timegrad_uncertainty", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    timellm_uncert = pd.to_numeric(_optional_col("rep_timellm_uncertainty", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if regime_preset:
        instability = np.maximum(
            pd.to_numeric(_optional_col("clean_regime4_state24_sticky090_v2_instability_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            pd.to_numeric(_optional_col("regime4_pred_instability_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        )
        whipsaw = np.maximum(
            pd.to_numeric(_optional_col("clean_regime4_state24_sticky090_v2_whipsaw_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            pd.to_numeric(_optional_col("regime4_pred_whipsaw_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        )
        trend = pd.to_numeric(_optional_col("clean_regime4_state24_sticky090_v2_trend_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        instability = whipsaw = trend = np.zeros(len(frame), dtype=np.float64)
    h = int(cfg.max_train_horizon_bars)
    base_valid = np.arange(0, max(0, len(frame) - h - 1), max(1, int(stride_bars)), dtype=np.int64)
    event_valid = np.asarray([], dtype=np.int64)
    event_meta: dict[str, Any] = {"enabled": bool(adaptive_sampling)}
    if adaptive_sampling:
        event_valid, event_meta = _entry_event_indices(
            frame,
            max_horizon=h,
            event_quantile=float(event_quantile),
            max_extra=int(max_extra),
        )
        event_meta["enabled"] = True
    valid = np.unique(np.concatenate([base_valid, event_valid])).astype(np.int64)
    if valid.size == 0:
        raise ValueError("no train candidates for entry labels")
    y = {
        "action": np.zeros(valid.size, dtype=np.int64),
        "quality": np.full(valid.size, float(cfg.cash_score), dtype=np.float64),
        "target_bucket": np.zeros(valid.size, dtype=np.int64),
        "target_horizon": np.zeros(valid.size, dtype=np.int64),
        "score_long": np.full(valid.size, -np.inf, dtype=np.float64),
        "score_short": np.full(valid.size, -np.inf, dtype=np.float64),
        "score_margin": np.zeros(valid.size, dtype=np.float64),
        "selected_mae_atr": np.zeros(valid.size, dtype=np.float64),
        "selected_mfe_atr": np.zeros(valid.size, dtype=np.float64),
    }
    horizons = np.arange(1, h + 1, dtype=np.int64)
    horizon_idx = [int(np.clip(v, 1, h)) - 1 for v in cfg.score_horizons]
    cost = 2.0 * float(cfg.fee + cfg.slip) * float(cfg.fixed_notional)
    for start in range(0, valid.size, int(batch_size)):
        end = min(start + int(batch_size), valid.size)
        idx = valid[start:end]
        entry = np.maximum(close[idx], 1e-12)
        fut = close[idx[:, None] + horizons[None, :]]
        raw_ret = fut / entry[:, None] - 1.0
        if perturbation_preset:
            delayed_entry = np.maximum(open_px[idx + 1], 1e-12)
            delayed_fut = close[idx[:, None] + 1 + horizons[None, :]]
            delayed_ret = delayed_fut / delayed_entry[:, None] - 1.0
        atr_now = atr[idx]
        min_edge = np.maximum(float(cfg.min_net_edge), atr_now * float(cfg.dynamic_min_edge_atr_frac) * float(cfg.fixed_notional))
        if regime_preset:
            regime_mult = np.where(
                instability[idx] > 0.65,
                np.inf,
                np.where(whipsaw[idx] > 0.55, 3.0, np.where(trend[idx] > 0.60, 0.70, 1.0)),
            )
            min_edge = min_edge * regime_mult
        if ood_preset:
            min_edge = np.where(ood_score[idx] > 6.0, np.inf, min_edge * np.where(ood_score[idx] > 4.0, 2.0, 1.0))
        if ts2vec_preset:
            min_edge = np.where(ts2vec_ood[idx] > 2.0, np.inf, min_edge * np.where(ts2vec_ood[idx] > 1.5, 1.75, 1.0))
        if mamba_preset:
            min_edge = np.where(mamba_toxic[idx] > 0.65, np.inf, min_edge * (1.0 + 1.5 * np.clip(mamba_toxic[idx], 0.0, 1.0)))
        if timellm_preset:
            min_edge = min_edge * (1.0 + 1.25 * np.clip(timellm_uncert[idx], 0.0, 1.0))
        scores: dict[int, np.ndarray] = {}
        target_horizons: dict[int, np.ndarray] = {}
        best_mae: dict[int, np.ndarray] = {}
        best_mfe: dict[int, np.ndarray] = {}
        for action, side in ((1, 1.0), (2, -1.0)):
            path = raw_ret * side
            stats = _score_direction_paths(path, cfg, horizon_idx, h, cost=cost)
            if perturbation_preset:
                delayed_stats = _score_direction_paths(delayed_ret * side, cfg, horizon_idx, h, cost=cost)
                base_score = stats["score"]
                delayed_score = delayed_stats["score"]
                penalty = 1.10 if preset == "diffusion_stress_proxy" else 0.75
                robust_score = 0.5 * (base_score + delayed_score) - float(penalty) * np.abs(base_score - delayed_score)
                delayed_worse = delayed_score < base_score
                stats = {
                    "score": robust_score,
                    "horizon": np.where(delayed_worse, delayed_stats["horizon"], stats["horizon"]),
                    "mae": np.maximum(stats["mae"], delayed_stats["mae"]),
                    "mfe": np.minimum(stats["mfe"], delayed_stats["mfe"]),
                    "vol": np.maximum(stats["vol"], delayed_stats["vol"]),
                }
            if psr_preset:
                downside = np.maximum(0.0, stats["mae"] - 0.5 * stats["mfe"])
                stats["score"] = (
                    stats["score"]
                    - 0.35 * stats["vol"] * float(cfg.fixed_notional)
                    - 0.45 * downside * float(cfg.fixed_notional)
                )
            if cost_preset:
                stats["score"] = stats["score"] * np.clip(0.45 + cost_beta[idx], 0.35, 1.25)
            if timegrad_preset:
                win_prob = timegrad_long[idx] if action == 1 else timegrad_short[idx]
                stats["score"] = np.where(
                    win_prob >= 0.58,
                    stats["score"] * np.clip(0.70 + win_prob, 0.70, 1.45) - 0.10 * timegrad_uncert[idx] * float(cfg.fixed_notional),
                    -np.inf,
                )
            if adverse_preset:
                mae_atr = stats["mae"] / np.maximum(atr_now, 1e-9)
                excess = np.maximum(0.0, mae_atr - float(adverse_atr_limit))
                stats["score"] = np.where(
                    mae_atr <= float(adverse_atr_limit) * 1.50,
                    stats["score"] - excess * float(cfg.fixed_notional) * 0.35,
                    -np.inf,
                )
            scores[action] = stats["score"]
            target_horizons[action] = stats["horizon"]
            best_mae[action] = stats["mae"]
            best_mfe[action] = stats["mfe"]
        long_score = scores[1]
        short_score = scores[2]
        both_finite = np.isfinite(long_score) & np.isfinite(short_score)
        long_delta = np.full(len(idx), -np.inf, dtype=np.float64)
        short_delta = np.full(len(idx), -np.inf, dtype=np.float64)
        long_delta[both_finite] = long_score[both_finite] - short_score[both_finite]
        short_delta[both_finite] = -long_delta[both_finite]
        long_delta[np.isfinite(long_score) & ~np.isfinite(short_score)] = np.inf
        short_delta[np.isfinite(short_score) & ~np.isfinite(long_score)] = np.inf
        choose_long = (long_delta > float(cfg.direction_margin)) & (long_score > min_edge)
        choose_short = (short_delta > float(cfg.direction_margin)) & (short_score > min_edge)
        y["action"][start:end] = np.where(choose_long, 1, np.where(choose_short, 2, 0)).astype(np.int64)
        y["quality"][start:end] = np.where(choose_long, long_score, np.where(choose_short, short_score, float(cfg.cash_score)))
        y["score_long"][start:end] = long_score
        y["score_short"][start:end] = short_score
        score_margin = np.maximum(long_delta, short_delta)
        score_margin[both_finite] = np.abs(long_score[both_finite] - short_score[both_finite])
        y["score_margin"][start:end] = score_margin
        selected_mae = np.where(choose_long, best_mae[1], np.where(choose_short, best_mae[2], 0.0))
        selected_mfe = np.where(choose_long, best_mfe[1], np.where(choose_short, best_mfe[2], 0.0))
        y["selected_mae_atr"][start:end] = selected_mae / np.maximum(atr_now, 1e-9)
        y["selected_mfe_atr"][start:end] = selected_mfe / np.maximum(atr_now, 1e-9)
        best_horizon = np.where(
            choose_long,
            target_horizons[1],
            np.where(choose_short, target_horizons[2], 0),
        ).astype(np.int64)
        y["target_horizon"][start:end] = np.where(y["action"][start:end] == 0, 0, best_horizon).astype(np.int64)
        y["target_bucket"][start:end] = np.where(y["action"][start:end] == 0, 0, _target_horizon_bucket(best_horizon)).astype(np.int64)
    if preset.startswith("session_topk"):
        ts = pd.to_datetime(frame["timestamp"], errors="coerce").iloc[valid]
        active = np.flatnonzero(y["action"] != 0)
        keep = np.zeros(valid.size, dtype=bool)
        if active.size:
            active_dates = ts.iloc[active].dt.floor("D")
            topk = max(1, int(session_topk))
            for _, pos in pd.Series(active, index=active_dates).groupby(level=0):
                idx = pos.to_numpy(dtype=np.int64)
                if idx.size <= topk:
                    keep[idx] = True
                    continue
                order = np.argsort(y["quality"][idx])[::-1][:topk]
                keep[idx[order]] = True
        drop = ~keep
        y["action"][drop] = 0
        y["quality"][drop] = float(cfg.cash_score)
        y["target_bucket"][drop] = 0
        y["target_horizon"][drop] = 0
        y["selected_mae_atr"][drop] = 0.0
        y["selected_mfe_atr"][drop] = 0.0
    meta = {
        "candidates": int(valid.size),
        "base_candidates": int(base_valid.size),
        "adaptive_extra_candidates": int(max(0, valid.size - base_valid.size)),
        "entry_adaptive_sampling": event_meta,
        "stride_bars": int(stride_bars),
        "max_train_horizon_bars": int(h),
        "score_horizons": list(cfg.score_horizons),
        "trained_heads": ["action", "quality", "target_bucket", "exit"],
        "removed_heads": ["notional", "take_profit", "stop_loss", "max_hold", "cooldown"],
        "labeling_basis": "multi_horizon_risk_adjusted_direction_no_tp_sl",
        "label_preset": str(label_preset),
        "session_topk": int(session_topk),
        "perturbation_robust": bool(perturbation_preset),
        "adverse_conformal": bool(adverse_preset),
        "adverse_atr_limit": float(adverse_atr_limit) if adverse_preset else None,
        "ood_guard": bool(ood_preset),
        "psr_path_quality": bool(psr_preset),
        "representation_preset": {
            "ts2vec_ood": bool(ts2vec_preset),
            "cost_beta_neutral": bool(cost_preset),
            "mamba_regime_filter": bool(mamba_preset),
            "timegrad_mc": bool(timegrad_preset),
            "timellm_uncertainty": bool(timellm_preset),
        },
        **asdict(cfg),
    }
    return valid, y, meta


def _apply_label_preset(cfg: EQEConfig, preset: str) -> EQEConfig:
    name = str(preset).strip().lower()
    if name in {"current", "current_quality", "baseline"}:
        return cfg
    if name == "density_balanced":
        return replace(
            cfg,
            min_net_edge=0.00016,
            dynamic_min_edge_atr_frac=0.055,
            direction_margin=0.00010,
        )
    if name == "scalp_short_horizon":
        return replace(
            cfg,
            max_train_horizon_bars=24,
            score_horizons=(6, 12, 24),
            min_net_edge=0.00018,
            dynamic_min_edge_atr_frac=0.060,
            hold_penalty=0.008,
        )
    if name == "regime_conditional":
        return cfg
    if name == "pullback_entry":
        return cfg
    if name == "perturbation_robust":
        return replace(
            cfg,
            min_net_edge=0.00020,
            dynamic_min_edge_atr_frac=0.070,
            direction_margin=0.00012,
            path_vol_penalty_lambda=0.075,
        )
    if name == "adverse_conformal":
        return replace(
            cfg,
            min_net_edge=0.00018,
            dynamic_min_edge_atr_frac=0.060,
            direction_margin=0.00010,
            mae_penalty_lambda=0.95,
            path_vol_penalty_lambda=0.085,
        )
    if name == "sam_conformal":
        return replace(
            cfg,
            min_net_edge=0.00016,
            dynamic_min_edge_atr_frac=0.055,
            direction_margin=0.00010,
            mae_penalty_lambda=0.90,
            path_vol_penalty_lambda=0.090,
        )
    if name == "high_precision_robust":
        return replace(
            cfg,
            min_net_edge=0.00035,
            dynamic_min_edge_atr_frac=0.100,
            direction_margin=0.00022,
            mae_penalty_lambda=1.10,
            path_vol_penalty_lambda=0.110,
        )
    if name == "turnover_balanced_robust":
        return replace(
            cfg,
            min_net_edge=0.00012,
            dynamic_min_edge_atr_frac=0.045,
            direction_margin=0.00007,
            hold_penalty=0.003,
            path_vol_penalty_lambda=0.065,
        )
    if name == "short_horizon_robust":
        return replace(
            cfg,
            max_train_horizon_bars=24,
            score_horizons=(6, 12, 24),
            min_net_edge=0.00014,
            dynamic_min_edge_atr_frac=0.050,
            direction_margin=0.00008,
            hold_penalty=0.006,
            mae_penalty_lambda=0.95,
            path_vol_penalty_lambda=0.085,
        )
    if name == "ts2vec_ood_proxy":
        return replace(
            cfg,
            min_net_edge=0.00018,
            dynamic_min_edge_atr_frac=0.060,
            direction_margin=0.00010,
            mae_penalty_lambda=0.90,
        )
    if name == "diffusion_stress_proxy":
        return replace(
            cfg,
            min_net_edge=0.00022,
            dynamic_min_edge_atr_frac=0.070,
            direction_margin=0.00013,
            mae_penalty_lambda=1.00,
            path_vol_penalty_lambda=0.095,
        )
    if name == "psr_path_quality":
        return replace(
            cfg,
            min_net_edge=0.00020,
            dynamic_min_edge_atr_frac=0.065,
            direction_margin=0.00012,
            terminal_weight=0.65,
            mfe_weight=0.35,
            mae_penalty_lambda=1.05,
            path_vol_penalty_lambda=0.110,
        )
    if name == "ts2vec_ood":
        return replace(
            cfg,
            min_net_edge=0.00018,
            dynamic_min_edge_atr_frac=0.060,
            direction_margin=0.00010,
            mae_penalty_lambda=0.90,
        )
    if name == "cost_beta_neutral":
        return replace(
            cfg,
            min_net_edge=0.00018,
            dynamic_min_edge_atr_frac=0.060,
            direction_margin=0.00010,
            terminal_weight=0.45,
            mfe_weight=0.55,
            path_vol_penalty_lambda=0.080,
        )
    if name == "mamba_regime_filter":
        return replace(
            cfg,
            min_net_edge=0.00016,
            dynamic_min_edge_atr_frac=0.055,
            direction_margin=0.00009,
            mae_penalty_lambda=0.85,
        )
    if name == "timegrad_mc":
        return replace(
            cfg,
            min_net_edge=0.00014,
            dynamic_min_edge_atr_frac=0.050,
            direction_margin=0.00008,
            hold_penalty=0.003,
        )
    if name == "timellm_uncertainty":
        return replace(
            cfg,
            min_net_edge=0.00018,
            dynamic_min_edge_atr_frac=0.060,
            direction_margin=0.00010,
            mae_penalty_lambda=0.95,
            path_vol_penalty_lambda=0.090,
        )
    if name.startswith("session_topk"):
        return cfg
    raise ValueError(f"unknown label preset: {preset!r}")


def _classifier_params(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "MultiClass",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_seed": int(seed),
        "allow_writing_files": False,
        "verbose": int(args.verbose),
        "thread_count": -1,
    }
    if args.task_type.upper() == "GPU":
        params.update({"task_type": "GPU", "devices": "0"})
    return params


def _regressor_params(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "RMSE",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_seed": int(seed),
        "allow_writing_files": False,
        "verbose": int(args.verbose),
        "thread_count": -1,
    }
    if args.task_type.upper() == "GPU":
        params.update({"task_type": "GPU", "devices": "0"})
    return params


def _fit_entry_models(x: np.ndarray, y: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    target_head_mode = str(args.target_head_mode).strip().lower()
    trade = y["action"] != 0
    q_w = np.clip(np.abs(y["quality"]), 0.03, 1.0)
    weight = np.maximum(np.where(trade, 1.0, float(args.cash_action_weight)), q_w)
    if np.unique(y["action"]).size < 2:
        action_model: Any = _ConstantClassifier(int(y["action"][0]) if len(y["action"]) else 0)
    else:
        action_model = CatBoostClassifier(**_classifier_params(args, args.seed))
        action_model.fit(Pool(x, y["action"], weight=weight))
    if np.unique(y["quality"]).size < 2:
        quality_model: Any = _ConstantRegressor(float(y["quality"][0]) if len(y["quality"]) else 0.0)
    else:
        quality_model = CatBoostRegressor(**_regressor_params(args, args.seed + 99))
        quality_model.fit(Pool(x, y["quality"], weight=weight))
    target_model: Any | None = None
    if target_head_mode == "bucket5":
        if trade.sum() == 0 or np.unique(y["target_bucket"][trade]).size < 2:
            target_model = _ConstantClassifier(int(y["target_bucket"][trade][0]) if trade.sum() else 0)
        else:
            target_model = CatBoostClassifier(**_classifier_params(args, args.seed + 199))
            target_model.fit(Pool(x[trade], y["target_bucket"][trade], weight=weight[trade]))
    elif target_head_mode == "horizon_reg":
        horizon_target = np.log1p(np.clip(y["target_horizon"][trade].astype(np.float64), 1.0, float(args.max_target_horizon)))
        if trade.sum() == 0 or np.unique(horizon_target).size < 2:
            target_model = _ConstantRegressor(float(horizon_target[0]) if len(horizon_target) else np.log1p(float(args.fixed_target_horizon or 12)))
        else:
            target_model = CatBoostRegressor(**_regressor_params(args, args.seed + 199))
            target_model.fit(Pool(x[trade], horizon_target, weight=weight[trade]))
    elif target_head_mode != "fixed":
        raise ValueError(f"unknown target head mode: {target_head_mode!r}")
    target_horizon = y["target_horizon"][trade].astype(np.float64) if trade.sum() else np.asarray([], dtype=np.float64)
    return {
        "action_model": action_model,
        "quality_model": quality_model,
        "target_head_mode": target_head_mode,
        "target_model": target_model,
        "target_bucket_model": target_model if target_head_mode == "bucket5" else None,
        "target_horizon_model": target_model if target_head_mode == "horizon_reg" else None,
        "fixed_target_horizon": int(args.fixed_target_horizon),
        "max_target_horizon": int(args.max_target_horizon),
        "label_distribution": {
            "action": pd.Series(y["action"]).value_counts().sort_index().to_dict(),
            "target_bucket": pd.Series(y["target_bucket"][trade]).value_counts().sort_index().to_dict(),
            "target_horizon": {
                "mean": float(np.mean(target_horizon)) if len(target_horizon) else 0.0,
                "p50": float(np.quantile(target_horizon, 0.50)) if len(target_horizon) else 0.0,
                "p90": float(np.quantile(target_horizon, 0.90)) if len(target_horizon) else 0.0,
            },
            "quality_mean": float(np.mean(y["quality"])),
            "quality_p95": float(np.quantile(y["quality"], 0.95)),
        },
    }


def _predict_entry(models: dict[str, Any], x: np.ndarray, cfg: EQEConfig) -> pd.DataFrame:
    action_proba = models["action_model"].predict_proba(x)
    classes = np.asarray(models["action_model"].classes_, dtype=int)
    action = classes[np.argmax(action_proba, axis=1)].astype(np.int64)
    target_head_mode = str(models.get("target_head_mode", "bucket5")).strip().lower()
    if target_head_mode == "bucket5":
        bucket_model = models.get("target_bucket_model") or models.get("target_model")
        bucket_proba = bucket_model.predict_proba(x)
        bucket_classes = np.asarray(bucket_model.classes_, dtype=int)
        target_bucket = bucket_classes[np.argmax(bucket_proba, axis=1)].astype(np.int64)
        target_bucket = np.where(action == 0, 0, np.clip(target_bucket, 0, 4)).astype(np.int64)
        target_horizon = np.asarray([_bucket_horizon(int(v)) if a != 0 else 0 for v, a in zip(target_bucket, action)], dtype=np.int64)
    elif target_head_mode == "horizon_reg":
        horizon_model = models.get("target_horizon_model") or models.get("target_model")
        max_horizon = int(models.get("max_target_horizon") or cfg.max_train_horizon_bars)
        pred_horizon = np.expm1(np.asarray(horizon_model.predict(x), dtype=np.float64))
        target_horizon = np.clip(np.rint(pred_horizon), 2, max(2, max_horizon)).astype(np.int64)
        target_horizon = np.where(action == 0, 0, target_horizon).astype(np.int64)
        target_bucket = np.where(action == 0, 0, _target_horizon_bucket(target_horizon)).astype(np.int64)
    elif target_head_mode == "fixed":
        fixed_horizon = int(models.get("fixed_target_horizon") or cfg.score_horizons[-1])
        target_horizon = np.where(action == 0, 0, fixed_horizon).astype(np.int64)
        target_bucket = np.where(action == 0, 0, _target_horizon_bucket(target_horizon)).astype(np.int64)
    else:
        raise ValueError(f"unknown target head mode: {target_head_mode!r}")
    out = pd.DataFrame(
        {
            "action": action,
            "quality_score": np.asarray(models["quality_model"].predict(x), dtype=np.float64),
            "confidence": np.max(action_proba, axis=1),
            "target_bucket": target_bucket,
            "target_horizon": target_horizon,
            "notional": np.full(len(x), float(cfg.fixed_notional), dtype=np.float64),
        }
    )
    return out


def _estimate_expected_return_by_bucket(
    frame: pd.DataFrame,
    valid: np.ndarray,
    y: dict[str, np.ndarray],
    cfg: EQEConfig,
) -> dict[int, float]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    table: dict[int, list[float]] = {k: [] for k in TARGET_BUCKET_TO_HORIZON}
    for j in np.flatnonzero(y["action"] != 0):
        idx = int(valid[j])
        bucket = int(np.clip(y["target_bucket"][j], 0, 4))
        horizon = min(_bucket_horizon(bucket), len(frame) - idx - 2)
        if horizon <= 1:
            continue
        side = 1.0 if int(y["action"][j]) == 1 else -1.0
        entry = max(float(close[idx]), 1e-12)
        path = (close[idx : idx + horizon + 1] / entry - 1.0) * side
        table[bucket].append(max(float(np.max(path)), 1e-6))
    out: dict[int, float] = {}
    fallback = 0.01
    for bucket, vals in table.items():
        if vals:
            out[bucket] = float(np.clip(np.quantile(vals, 0.60), 0.001, 0.08))
            fallback = out[bucket]
        else:
            out[bucket] = float(fallback)
    return out


def _exit_state_vec(
    frame: pd.DataFrame,
    *,
    side: int,
    entry_idx: int,
    current_idx: int,
    entry_px: float,
    px: float,
    hold: int,
    horizon: int,
    mae: float,
    mfe: float,
    target_bucket: int = 4,
    regime_drift: bool = False,
    capture_ratio: bool = False,
    expected_return: float = 0.0,
) -> np.ndarray:
    eps = 1e-9
    raw = (px - entry_px) / max(entry_px, 1e-12) if side > 0 else (entry_px - px) / max(entry_px, 1e-12)
    current_atr = max(_frame_value(frame, "atr14_pct", current_idx, 0.003), eps)
    entry_atr = max(_frame_value(frame, "atr14_pct", entry_idx, current_atr), eps)
    giveback = max(0.0, float(mfe) - max(float(raw), 0.0))
    obi = _frame_value(frame, "obi", current_idx, 0.0)
    entry_obi = _frame_value(frame, "obi", entry_idx, obi)
    taker = _frame_value(frame, "taker_buy_ratio", current_idx, 0.5) - 0.5
    entry_taker = _frame_value(frame, "taker_buy_ratio", entry_idx, taker + 0.5) - 0.5
    nif = _frame_value(frame, "nif_whale", current_idx, 0.0)
    entry_nif = _frame_value(frame, "nif_whale", entry_idx, nif)
    eai = _frame_value(frame, "eai", current_idx, 0.0)
    entry_eai = _frame_value(frame, "eai", entry_idx, eai)
    risk_off = max(
        _frame_value(frame, "clean_regime4_state24_sticky090_v2_instability_prob", current_idx, 0.0),
        _frame_value(frame, "regime4_pred_instability_prob", current_idx, 0.0),
    )
    entry_risk_off = max(
        _frame_value(frame, "clean_regime4_state24_sticky090_v2_instability_prob", entry_idx, 0.0),
        _frame_value(frame, "regime4_pred_instability_prob", entry_idx, 0.0),
    )
    whipsaw = max(
        _frame_value(frame, "clean_regime4_state24_sticky090_v2_whipsaw_prob", current_idx, 0.0),
        _frame_value(frame, "regime4_pred_whipsaw_prob", current_idx, 0.0),
    )
    entry_whipsaw = max(
        _frame_value(frame, "clean_regime4_state24_sticky090_v2_whipsaw_prob", entry_idx, 0.0),
        _frame_value(frame, "regime4_pred_whipsaw_prob", entry_idx, 0.0),
    )
    confidence = _frame_value(frame, "clean_regime4_state24_sticky090_v2_confidence", current_idx, 0.0)
    entry_confidence = _frame_value(frame, "clean_regime4_state24_sticky090_v2_confidence", entry_idx, 0.0)
    values = [
            float(side),
            float(raw),
            float(hold) / max(float(horizon), 1.0),
            float(max(horizon - hold, 0)) / max(float(horizon), 1.0),
            float(horizon) / 96.0,
            float(mae),
            float(mfe),
            giveback,
            giveback / max(float(mfe), eps),
            current_atr,
            entry_atr,
            float(raw) / current_atr,
            float(mae) / current_atr,
            float(mfe) / current_atr,
            float(side) * obi,
            float(side) * (obi - entry_obi),
            float(side) * (taker - entry_taker),
            float(side) * nif,
            float(side) * (nif - entry_nif),
            float(side) * eai,
            float(side) * (eai - entry_eai),
            float(side) * _frame_value(frame, "oi_delta_pct", current_idx, 0.0),
            float(side) * _frame_value(frame, "funding_rate", current_idx, 0.0),
            risk_off,
            whipsaw,
            confidence,
            float(target_bucket),
    ]
    if regime_drift:
        values.extend(
            [
                float(risk_off - entry_risk_off),
                float(whipsaw - entry_whipsaw),
                float(confidence - entry_confidence),
                float((risk_off >= whipsaw) != (entry_risk_off >= entry_whipsaw)),
            ]
        )
    if capture_ratio:
        expected = max(float(expected_return), eps)
        values.extend(
            [
                float(np.clip(raw / expected, -2.0, 3.0)),
                float(np.clip((float(mfe) / max(float(EQEConfig.fixed_notional), eps)) / expected, 0.0, 3.0)),
            ]
        )
    return np.asarray(values, dtype=np.float64)


def _build_exit_dataset(
    frame: pd.DataFrame,
    x_all: np.ndarray,
    valid: np.ndarray,
    y: dict[str, np.ndarray],
    entry_dec: pd.DataFrame,
    cfg: EQEConfig,
    *,
    max_samples: int,
    step: int,
    cost_mult: float,
    weight_scale: float,
    regime_drift: bool = False,
    capture_ratio: bool = False,
    adaptive_sampling: bool = False,
    expected_return_by_bucket: dict[int, float] | None = None,
    target_head_mode: str = "bucket5",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    rng = np.random.default_rng(42)
    trade_idx = np.flatnonzero(y["action"] != 0)
    if max_samples > 0 and len(trade_idx) > max_samples:
        trade_idx = rng.choice(trade_idx, size=int(max_samples), replace=False)
        trade_idx.sort()
    h = int(cfg.max_train_horizon_bars)
    round_trip_cost = 2.0 * (float(cfg.fee) + float(cfg.slip)) * float(cfg.fixed_notional) * float(cost_mult)
    used_horizons: list[int] = []
    state_features = _exit_state_features(regime_drift=regime_drift, capture_ratio=capture_ratio)
    expected_return_by_bucket = expected_return_by_bucket or {k: 0.01 for k in TARGET_BUCKET_TO_HORIZON}
    target_head_mode = str(target_head_mode).strip().lower()
    for j in trade_idx:
        idx = int(valid[j])
        side = 1 if int(y["action"][j]) == 1 else -1
        target_bucket = int(np.clip(y["target_bucket"][j], 0, 4))
        if target_head_mode == "bucket5":
            target_horizon = _bucket_horizon(target_bucket)
        else:
            target_horizon = int(y.get("target_horizon", np.zeros_like(y["target_bucket"]))[j])
            if target_horizon <= 0:
                target_horizon = _bucket_horizon(target_bucket)
        horizon = min(target_horizon, len(frame) - idx - 2)
        if horizon <= 2:
            continue
        used_horizons.append(int(horizon))
        entry = close[idx]
        side_ret = (close[idx : idx + horizon + 1] / max(entry, 1e-12) - 1.0) * side
        sample_stride = 1 if adaptive_sampling else max(1, int(step))
        for k in range(1, horizon, sample_stride):
            cur_path = side_ret[: k + 1]
            fut_path = side_ret[k : horizon + 1]
            cur_ret = float(cur_path[-1]) * float(cfg.fixed_notional)
            mae = max(0.0, -float(np.min(cur_path))) * float(cfg.fixed_notional)
            mfe = max(0.0, float(np.max(cur_path))) * float(cfg.fixed_notional)
            current_atr = max(_frame_value(frame, "atr14_pct", idx + k, 0.003), 1e-9)
            hold_frac = float(k) / max(float(horizon), 1.0)
            giveback = max(0.0, mfe - max(cur_ret, 0.0))
            giveback_ratio = giveback / max(mfe, 1e-9)
            include_sample = True
            if adaptive_sampling:
                include_sample = (
                    (k % max(1, int(step)) == 0)
                    or (giveback_ratio > 0.5)
                    or (hold_frac > 0.85)
                    or ((mae / float(cfg.fixed_notional)) / current_atr > 0.8)
                )
            if not include_sample:
                continue
            future_best = float(np.max(fut_path)) * float(cfg.fixed_notional)
            future_terminal = float(fut_path[-1]) * float(cfg.fixed_notional)
            future_adverse = max(0.0, cur_ret - float(np.min(fut_path)) * float(cfg.fixed_notional))
            close_score = cur_ret - round_trip_cost
            continue_score = (
                0.45 * future_terminal
                + 0.55 * future_best
                - float(cfg.exit_adverse_penalty) * future_adverse
                - float(cfg.exit_giveback_penalty) * giveback
                - float(cfg.hold_penalty) * ((horizon - k) / max(float(horizon), 1.0))
            )
            margin = close_score - continue_score
            close_label = int(margin >= 0.0)
            state = _exit_state_vec(
                frame,
                side=side,
                entry_idx=idx,
                current_idx=idx + k,
                entry_px=entry,
                px=close[idx + k],
                hold=k,
                horizon=horizon,
                mae=mae,
                mfe=mfe,
                target_bucket=target_bucket,
                regime_drift=regime_drift,
                capture_ratio=capture_ratio,
                expected_return=float(expected_return_by_bucket.get(target_bucket, 0.01)),
            )
            rows.append(np.concatenate([x_all[idx + k], state]))
            labels.append(close_label)
            weights.append(float(np.clip(abs(margin) * float(weight_scale) + 0.25, 0.25, 6.0)))
    if not rows:
        raise RuntimeError("empty exit dataset")
    meta = {
        "samples": int(len(rows)),
        "close_rate": float(np.mean(labels)),
        "trade_entries_used": int(len(trade_idx)),
        "state_dim": int(len(state_features)),
        "state_features": state_features,
        "target_horizon_distribution": pd.Series(used_horizons).value_counts().sort_index().to_dict(),
        "step": int(step),
        "cost_mult": float(cost_mult),
        "weight_scale": float(weight_scale),
        "regime_drift": bool(regime_drift),
        "capture_ratio": bool(capture_ratio),
        "adaptive_sampling": bool(adaptive_sampling),
        "target_head_mode": target_head_mode,
        "horizon_source": "target_bucket_to_bucket_horizon" if target_head_mode == "bucket5" else "target_horizon_label",
        "expected_return_by_bucket": {int(k): float(v) for k, v in expected_return_by_bucket.items()},
    }
    return np.vstack(rows), np.asarray(labels, dtype=np.int64), np.asarray(weights, dtype=np.float64), meta


def _fit_exit_model(x: np.ndarray, y: np.ndarray, w: np.ndarray, args: argparse.Namespace) -> Any:
    if np.unique(y).size < 2:
        return _ConstantClassifier(int(y[0]) if len(y) else 0)
    params = _classifier_params(args, args.seed + 777)
    params["loss_function"] = "Logloss"
    params["iterations"] = int(args.exit_iterations)
    params["learning_rate"] = float(args.exit_learning_rate)
    params["depth"] = int(args.exit_depth)
    model = CatBoostClassifier(**params)
    model.fit(Pool(x, y, weight=w))
    return model


def _exit_close_prob(model: Any, x_row: np.ndarray, state: np.ndarray) -> float:
    probs = model.predict_proba(np.concatenate([x_row, state])[None, :])[0]
    classes = np.asarray(model.classes_, dtype=int)
    if 1 not in classes:
        return 0.0
    return float(probs[int(np.flatnonzero(classes == 1)[0])])


def _backtest(
    frame: pd.DataFrame,
    x_val: np.ndarray,
    dec: pd.DataFrame,
    exit_model: Any,
    *,
    entry_threshold: float,
    exit_threshold: float | tuple[float, ...],
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
    exit_on_flip: bool,
    regime_drift: bool = False,
    capture_ratio: bool = False,
    expected_return_by_bucket: dict[int, float] | None = None,
    guard_max_target_hold: bool = False,
    guard_adverse_atr: float = 0.0,
    guard_giveback_ratio: float = 0.0,
    guard_min_mfe: float = 0.0,
    entry_pullback_atr: float = 0.0,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    open_px = pd.to_numeric(frame["open"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_idx = 0
    entry_equity = 1.0
    hold = 0
    mae = mfe = 0.0
    exposure = 0.0
    target_horizon = int(state_horizon)
    target_bucket = 4
    expected_return_by_bucket = expected_return_by_bucket or {k: 0.01 for k in TARGET_BUCKET_TO_HORIZON}
    trades = wins = long_entries = short_entries = exit_model_closes = missed_entries = 0
    exposure_sum = 0.0
    exits: dict[str, int] = {}

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int, notional: float, horizon: int, bucket: int) -> None:
        nonlocal side, entry, entry_idx, entry_equity, hold, mae, mfe, exposure, target_horizon, target_bucket, cash, exposure_sum, long_entries, short_entries, missed_entries
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        entry_idx = int(i)
        exposure = float(np.clip(notional, 0.01, 2.0))
        target_horizon = int(np.clip(horizon, 2, state_horizon))
        target_bucket = int(np.clip(bucket, 0, 4))
        if float(entry_pullback_atr) > 0.0:
            pullback = float(entry_pullback_atr) * max(float(atr[fill_i]), 0.0)
            if side > 0:
                limit_px = float(open_px[fill_i]) * (1.0 - pullback)
                if float(low[fill_i]) > limit_px:
                    side = 0
                    exposure = 0.0
                    missed_entries += 1
                    return
                entry = limit_px * (1.0 + slip)
            else:
                limit_px = float(open_px[fill_i]) * (1.0 + pullback)
                if float(high[fill_i]) < limit_px:
                    side = 0
                    exposure = 0.0
                    missed_entries += 1
                    return
                entry = limit_px * (1.0 - slip)
        else:
            entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        hold = 0
        mae = mfe = 0.0
        exposure_sum += exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str) -> None:
        nonlocal side, entry, cash, hold, mae, mfe, exposure, target_horizon, target_bucket, trades, wins
        fill_px = _fill_price(frame, min(i + 1, len(frame) - 1), side, slip, entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        hold = 0
        mae = mfe = exposure = 0.0
        target_horizon = int(state_horizon)
        target_bucket = 4

    for i in range(len(frame) - 2):
        row = dec.iloc[i]
        desired = int(row.action) if float(row.quality_score) >= float(entry_threshold) else 0
        closed_this_bar = False
        if side != 0:
            hold += 1
            px = close[i]
            raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
            mae = max(mae, max(0.0, -raw * exposure))
            mfe = max(mfe, max(0.0, raw * exposure))
            if hold >= int(min_exit_hold):
                current_atr = max(_frame_value(frame, "atr14_pct", i, 0.003), 1e-9)
                giveback = max(0.0, mfe - max(raw * exposure, 0.0))
                giveback_ratio = giveback / max(mfe, 1e-9)
                adverse_atr = max(0.0, -raw) / current_atr
                if guard_max_target_hold and hold >= int(target_horizon):
                    exit_pos(i, "guard_target_hold")
                    closed_this_bar = True
                elif float(guard_adverse_atr) > 0.0 and adverse_atr >= float(guard_adverse_atr):
                    exit_pos(i, "guard_adverse_atr")
                    closed_this_bar = True
                elif (
                    float(guard_giveback_ratio) > 0.0
                    and mfe >= float(guard_min_mfe)
                    and giveback_ratio >= float(guard_giveback_ratio)
                ):
                    exit_pos(i, "guard_giveback")
                    closed_this_bar = True
                if closed_this_bar:
                    eq = equity(i)
                    peak = max(peak, eq)
                    mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
                    continue
                state = _exit_state_vec(
                    frame,
                    side=side,
                    entry_idx=entry_idx,
                    current_idx=i,
                    entry_px=entry,
                    px=px,
                    hold=hold,
                    horizon=int(target_horizon),
                    mae=mae,
                    mfe=mfe,
                    target_bucket=target_bucket,
                    regime_drift=regime_drift,
                    capture_ratio=capture_ratio,
                    expected_return=float(expected_return_by_bucket.get(target_bucket, 0.01)),
                )
                close_prob = _exit_close_prob(exit_model, x_val[i], state)
                if close_prob >= _threshold_for_bucket(exit_threshold, target_bucket):
                    exit_model_closes += 1
                    exit_pos(i, "exit_model")
                    closed_this_bar = True
                elif exit_on_flip and desired != 0 and ((desired == 1 and side < 0) or (desired == 2 and side > 0)):
                    exit_pos(i, "model_flip")
                    closed_this_bar = True
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and desired != 0 and not closed_this_bar:
            enter(
                i,
                1 if desired == 1 else -1,
                float(row.notional),
                int(getattr(row, "target_horizon", state_horizon)),
                int(getattr(row, "target_bucket", 4)),
            )
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "missed_entries": int(missed_entries),
        "avg_notional": float(exposure_sum / max(trades, 1)),
        "exit_model_closes": int(exit_model_closes),
        "exits": exits,
    }


def _entry_threshold_grid(dec: pd.DataFrame, n: int) -> np.ndarray:
    active = dec.loc[dec["action"] != 0, "quality_score"].to_numpy(dtype=np.float64)
    active = active[np.isfinite(active)]
    if active.size == 0:
        return np.array([np.inf])
    return np.unique(np.quantile(active, np.linspace(0.10, 0.995, int(n))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha6 entry/quality/exit-only CatBoost policy for DSAC risk handoff.")
    ap.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    ap.add_argument("--representation-feature-file", type=Path, default=None)
    ap.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--iterations", type=int, default=650)
    ap.add_argument("--learning-rate", type=float, default=0.055)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--l2-leaf-reg", type=float, default=5.0)
    ap.add_argument("--exit-iterations", type=int, default=500)
    ap.add_argument("--exit-learning-rate", type=float, default=0.045)
    ap.add_argument("--exit-depth", type=int, default=5)
    ap.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride-bars", type=int, default=3)
    ap.add_argument(
        "--label-preset",
        choices=[
            "current_quality",
            "density_balanced",
            "session_topk_day2",
            "scalp_short_horizon",
            "regime_conditional",
            "pullback_entry",
            "perturbation_robust",
            "adverse_conformal",
            "sam_conformal",
            "high_precision_robust",
            "turnover_balanced_robust",
            "short_horizon_robust",
            "ts2vec_ood_proxy",
            "diffusion_stress_proxy",
            "psr_path_quality",
            "ts2vec_ood",
            "cost_beta_neutral",
            "mamba_regime_filter",
            "timegrad_mc",
            "timellm_uncertainty",
        ],
        default="current_quality",
    )
    ap.add_argument("--session-topk", type=int, default=2)
    ap.add_argument("--adaptive-entry-sampling", action="store_true")
    ap.add_argument("--entry-event-quantile", type=float, default=0.85)
    ap.add_argument("--entry-adaptive-max-extra", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--entry-thresholds", type=int, default=50)
    ap.add_argument("--eval-costs", default="1,2,3", help="Comma-separated cost multipliers to replay during threshold search.")
    ap.add_argument("--exit-threshold-grid", default="0.45,0.55,0.65,0.75,0.85")
    ap.add_argument(
        "--exit-bucket-threshold-grid",
        default="",
        help="Optional semicolon-separated 5-bucket threshold configs, e.g. '0.28,0.30,0.35,0.42,0.48;0.35,0.35,0.35,0.35,0.35'.",
    )
    ap.add_argument("--fixed-notional", type=float, default=0.25)
    ap.add_argument("--exit-step", type=int, default=2)
    ap.add_argument("--exit-max-trades", type=int, default=9000)
    ap.add_argument("--exit-cost-mult", type=float, default=3.0)
    ap.add_argument("--exit-weight-scale", type=float, default=80.0)
    ap.add_argument("--min-exit-hold", type=int, default=2)
    ap.add_argument("--exit-on-flip", action="store_true")
    ap.add_argument("--enable-regime-drift-state", action="store_true")
    ap.add_argument("--enable-capture-ratio-state", action="store_true")
    ap.add_argument("--adaptive-exit-sampling", action="store_true")
    ap.add_argument("--guard-max-target-hold", action="store_true")
    ap.add_argument("--guard-adverse-atr", type=float, default=0.0)
    ap.add_argument("--guard-giveback-ratio", type=float, default=0.0)
    ap.add_argument("--guard-min-mfe", type=float, default=0.0)
    ap.add_argument("--entry-pullback-atr", type=float, default=0.0)
    ap.add_argument("--target-head-mode", choices=["bucket5", "horizon_reg", "fixed"], default="bucket5")
    ap.add_argument("--fixed-target-horizon", type=int, default=0)
    ap.add_argument("--max-target-horizon", type=int, default=96)
    ap.add_argument("--cash-action-weight", type=float, default=0.35)
    ap.add_argument("--verbose", type=int, default=100)
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _apply_label_preset(replace(EQEConfig(), fixed_notional=float(args.fixed_notional)), str(args.label_preset))
    args.max_target_horizon = int(max(2, min(int(args.max_target_horizon), int(cfg.max_train_horizon_bars))))
    if str(args.target_head_mode).strip().lower() == "fixed" and int(args.fixed_target_horizon) <= 0:
        name = str(args.label_preset).strip().lower()
        if "scalp" in name:
            args.fixed_target_horizon = 12
        elif "short_horizon" in name or "pullback" in name:
            args.fixed_target_horizon = 24
        else:
            args.fixed_target_horizon = int(cfg.score_horizons[-1])
    if int(args.fixed_target_horizon) > 0:
        args.fixed_target_horizon = int(max(2, min(int(args.fixed_target_horizon), int(cfg.max_train_horizon_bars))))
    entry_pullback_atr = float(args.entry_pullback_atr)
    if str(args.label_preset).strip().lower() == "pullback_entry" and entry_pullback_atr <= 0.0:
        entry_pullback_atr = 0.30
    spec = _read_spec(args.spec_dir, args.variant)
    use_pca = bool(spec.get("extra_pca_enable")) and not args.no_pca and int(spec.get("extra_pca_components") or 0) > 0
    feat, present, missing = _read_feature_frame(args.feature_csv, list(spec["features"]), CONTEXT_COLS)
    if args.representation_feature_file is not None:
        rep_path = Path(args.representation_feature_file)
        if not rep_path.exists():
            raise FileNotFoundError(rep_path)
        rep = pd.read_parquet(rep_path) if rep_path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(rep_path)
        rep_cols = ["timestamp", *[c for c in rep.columns if str(c).startswith("rep_")]]
        feat = feat.merge(rep[rep_cols], on="timestamp", how="left")
    frame = feat.merge(_label_frame(args.label_dir), on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    train = frame[frame["dataset_split"].astype(str).str.lower().eq("train")].copy()
    val = frame[frame["dataset_split"].astype(str).str.lower().ne("train")].copy()
    if args.smoke:
        train = train.iloc[: min(len(train), 5000)].copy()
        val = val.iloc[: min(len(val), 3000)].copy()
        args.iterations = min(args.iterations, 20)
        args.exit_iterations = min(args.exit_iterations, 20)
        args.entry_thresholds = min(args.entry_thresholds, 8)
        args.stride_bars = max(args.stride_bars, 6)
        args.entry_adaptive_max_extra = min(args.entry_adaptive_max_extra, 1000)
        args.exit_max_trades = min(args.exit_max_trades, 1000)
    x_train_all, x_val, model_features, pipe = _feature_matrix(
        train,
        val,
        present,
        use_pca=use_pca,
        pca_components=int(spec.get("extra_pca_components") or 0),
    )
    valid, y, label_meta = _build_entry_labels(
        train,
        cfg,
        stride_bars=args.stride_bars,
        batch_size=args.batch_size,
        adaptive_sampling=bool(args.adaptive_entry_sampling),
        event_quantile=float(args.entry_event_quantile),
        max_extra=int(args.entry_adaptive_max_extra),
        label_preset=str(args.label_preset),
        session_topk=int(args.session_topk),
    )
    target_head_mode = str(args.target_head_mode).strip().lower()
    if target_head_mode == "bucket5":
        label_meta["trained_heads"] = ["action", "quality", "target_bucket", "exit"]
    elif target_head_mode == "horizon_reg":
        label_meta["trained_heads"] = ["action", "quality", "target_horizon_reg", "exit"]
    else:
        label_meta["trained_heads"] = ["action", "quality", "fixed_target_horizon", "exit"]
    label_meta["target_head_mode"] = target_head_mode
    label_meta["fixed_target_horizon"] = int(args.fixed_target_horizon)
    label_meta["max_target_horizon"] = int(args.max_target_horizon)
    prefix = args.out_dir / args.variant
    label_cols = [c for c in ("timestamp", "open", "high", "low", "close", "atr14_pct") if c in train.columns]
    label_audit = train.iloc[valid][label_cols].copy()
    for col, values in y.items():
        if len(values) == len(valid):
            label_audit[col] = values
    label_audit.to_csv(f"{prefix}_train_labels.csv", index=False)
    expected_return_by_bucket = _estimate_expected_return_by_bucket(train, valid, y, cfg)
    x_entry = x_train_all[valid]
    print(
        f"[alpha6-eqe] variant={args.variant} train_rows={len(train)} val_rows={len(val)} labels={len(valid)} features={len(model_features)} use_pca={use_pca}",
        flush=True,
    )
    entry_models = _fit_entry_models(x_entry, y, args)
    train_dec = _predict_entry(entry_models, x_train_all, cfg)
    x_exit, y_exit, w_exit, exit_meta = _build_exit_dataset(
        train,
        x_train_all,
        valid,
        y,
        train_dec,
        cfg,
        max_samples=int(args.exit_max_trades),
        step=int(args.exit_step),
        cost_mult=float(args.exit_cost_mult),
        weight_scale=float(args.exit_weight_scale),
        regime_drift=bool(args.enable_regime_drift_state),
        capture_ratio=bool(args.enable_capture_ratio_state),
        adaptive_sampling=bool(args.adaptive_exit_sampling),
        expected_return_by_bucket=expected_return_by_bucket,
        target_head_mode=str(args.target_head_mode),
    )
    print(f"[alpha6-eqe] exit_samples={len(y_exit)} close_rate={np.mean(y_exit):.3f}", flush=True)
    exit_model = _fit_exit_model(x_exit, y_exit, w_exit, args)
    dec = _predict_entry(entry_models, x_val, cfg)
    exit_thresholds: list[float | tuple[float, ...]] = [float(x.strip()) for x in str(args.exit_threshold_grid).split(",") if x.strip()]
    if str(args.exit_bucket_threshold_grid).strip():
        exit_thresholds.extend(
            _parse_bucket_thresholds(x.strip())
            for x in str(args.exit_bucket_threshold_grid).split(";")
            if x.strip()
        )
    rows = []
    best: dict[str, Any] | None = None
    eval_costs = _parse_cost_multipliers(str(args.eval_costs))
    for eth in _entry_threshold_grid(dec, args.entry_thresholds):
        for xth in exit_thresholds:
            bt = {
                f"cost{m}": _backtest(
                    val,
                    x_val,
                    dec,
                    exit_model,
                    entry_threshold=float(eth),
                    exit_threshold=xth,
                    fee=cfg.fee * m,
                    slip=cfg.slip * m,
                    min_exit_hold=int(args.min_exit_hold),
                    state_horizon=int(cfg.max_train_horizon_bars),
                    exit_on_flip=bool(args.exit_on_flip),
                    regime_drift=bool(args.enable_regime_drift_state),
                    capture_ratio=bool(args.enable_capture_ratio_state),
                    expected_return_by_bucket=expected_return_by_bucket,
                    guard_max_target_hold=bool(args.guard_max_target_hold),
                    guard_adverse_atr=float(args.guard_adverse_atr),
                    guard_giveback_ratio=float(args.guard_giveback_ratio),
                    guard_min_mfe=float(args.guard_min_mfe),
                    entry_pullback_atr=float(entry_pullback_atr),
                )
                for m in eval_costs
            }
            primary_bt = bt.get("cost3") or bt[f"cost{eval_costs[-1]}"]
            score = _score(
                bt.get("cost1", primary_bt),
                bt.get("cost2", primary_bt),
                primary_bt,
            )
            row = {
                "entry_threshold": float(eth),
                "exit_threshold": ",".join(f"{v:.6g}" for v in xth) if isinstance(xth, tuple) else float(xth),
                "exit_threshold_type": "bucket" if isinstance(xth, tuple) else "scalar",
                "score": float(score),
                "pnl": float(primary_bt["pnl"]),
                "mdd": float(primary_bt["mdd"]),
                "trades": int(primary_bt["trades"]),
                "trades_per_day": float(primary_bt["trades_per_day"]),
                "wr": float(primary_bt["wr"]),
                "long_entries": int(primary_bt["long_entries"]),
                "short_entries": int(primary_bt["short_entries"]),
                "avg_notional": float(primary_bt["avg_notional"]),
                "exit_model_closes": int(primary_bt["exit_model_closes"]),
                "missed_entries": int(primary_bt.get("missed_entries", 0)),
                "exits": json.dumps(primary_bt["exits"], sort_keys=True),
            }
            rows.append(row)
            if best is None or row["score"] > best["summary"]["score"]:
                best = {"summary": row, "backtest": bt}
    assert best is not None

    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(f"{prefix}_threshold_grid.csv", index=False)
    pred = val[["timestamp", "open", "high", "low", "close", "label_action"]].copy()
    for col in dec.columns:
        pred[col] = dec[col].to_numpy()
    pred.to_csv(f"{prefix}_val_predictions.csv", index=False)
    artifact = {
        "model_id": MODEL_ID,
        "variant": args.variant,
        "config": asdict(cfg),
        "feature_cols": present,
        "model_features": model_features,
        "missing_features": missing,
        "use_pca": use_pca,
        "pipeline": pipe,
        "entry_models": entry_models,
        "exit_model": exit_model,
        "exit_meta": exit_meta,
        "exit_state_features": _exit_state_features(
            regime_drift=bool(args.enable_regime_drift_state),
            capture_ratio=bool(args.enable_capture_ratio_state),
        ),
        "active_exit_state_features": _exit_state_features(
            regime_drift=bool(args.enable_regime_drift_state),
            capture_ratio=bool(args.enable_capture_ratio_state),
        ),
        "expected_return_by_bucket": expected_return_by_bucket,
    }
    joblib.dump(artifact, f"{prefix}_bundle.joblib")
    summary = {
        "model_id": MODEL_ID,
        "variant": args.variant,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "label_meta": label_meta,
        "entry_label_distribution": entry_models["label_distribution"],
        "exit_meta": exit_meta,
        "raw_feature_count": int(len(present)),
        "missing_features": missing,
        "model_feature_count": int(len(model_features)),
        "use_pca": bool(use_pca),
        "best": best["summary"],
        "best_backtest": best["backtest"],
        "params": vars(args),
        "effective_entry_pullback_atr": float(entry_pullback_atr),
    }
    Path(f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
