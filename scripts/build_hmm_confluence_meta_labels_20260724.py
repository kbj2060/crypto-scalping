#!/usr/bin/env python3
"""Build causal HMM-routed confluence meta-labels for ETH 5-minute bars.

The current Regime3 HMM and transition-risk sidecars are decision-time context.
Future bars are used only after a candidate has been fixed, to determine whether
that setup reaches its frozen target before its frozen stop or time limit.

This is a research label path. It never reads a saved trade ledger or parent
exit timestamp, and it does not modify a live model or runtime configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_ID = "eth_hmm_confluence_meta_labels_v1_20260724"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

MARKET_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
MARKET_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
HMM_ARTIFACT = (
    ROOT
    / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
    / "regime3_current_sensitive_hmm_wide24_2024.joblib"
)
HMM_2025 = HMM_ARTIFACT.parent / "training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
HMM_2026 = HMM_ARTIFACT.parent / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
RISK_DIR = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530"
RISK_ARTIFACT = RISK_DIR / "regime3_stability_risk_h6.joblib"
FUNDING_DIR = ROOT / "binance_data/funding_rate"

HMM_PREFIX = "regime3_current_sensitive_wide24_"
HMM_CLASSES = ("bull", "bear", "chop")
BAR_MINUTES = 5
VPVR_WINDOW = 288
VPVR_BINS = 24
VALUE_AREA_PCT = 0.70
VWMA_FAST = 100
VWMA_SLOW = 288
ATR_WINDOW = 192
RANGE_HORIZON = 24
TREND_HORIZON = 96
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
CHART_START = pd.Timestamp("2026-01-01")
CHART_END = pd.Timestamp("2026-03-31 23:55:00")

REQUIRED_MARKET_COLS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "cvp_cluster_position",
    "cvp_volume_imbalance",
    "sig_volume_confirm",
    "sig_liquidity_trap",
    "oi_change_rate",
    "funding_z_score",
    "lower_wick_z",
    "upper_wick_z",
    "sweep_prev_high_reclaim",
    "sweep_prev_low_reclaim",
    "failed_breakout_up",
    "failed_breakout_down",
]


@dataclass(frozen=True)
class RouteThresholds:
    confidence_min: float
    margin_min: float
    entropy_max: float
    transition_risk_max: float
    churn_risk_max: float
    fit_start: str
    fit_end: str


@dataclass(frozen=True)
class FundingTape:
    timestamp_ns: np.ndarray
    rate_x_price_cumsum: np.ndarray


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_unique(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=usecols, parse_dates=["timestamp"], low_memory=False)
    if frame["timestamp"].isna().any():
        raise RuntimeError(f"{path} has invalid timestamps")
    if frame["timestamp"].duplicated().any():
        raise RuntimeError(f"{path} has duplicate timestamps")
    if not frame["timestamp"].is_monotonic_increasing:
        raise RuntimeError(f"{path} timestamps are not sorted")
    return frame.reset_index(drop=True)


def infer_transition_risk(market: pd.DataFrame, artifact_path: Path) -> pd.DataFrame:
    from scripts.train_regime3_stability_risk_20260530 import (
        _add_rolling_stable_features,
        _add_stability_features,
        _output,
        _prepare,
        _proba2,
    )

    payload = joblib.load(artifact_path)
    required = {
        "feature_cols",
        "feature_medians",
        "scaler",
        "transition_model",
        "risk_model",
        "threshold",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"{artifact_path} missing payload keys: {missing}")
    transformed = _add_stability_features(_add_rolling_stable_features(market.copy()))
    feature_cols = list(payload["feature_cols"])
    missing_features = sorted(set(feature_cols) - set(transformed.columns))
    if missing_features:
        raise RuntimeError(f"transition-risk inputs missing features: {missing_features}")
    medians = pd.Series(payload["feature_medians"], dtype=float)
    x, _, _ = _prepare(
        transformed,
        feature_cols,
        scaler=payload["scaler"],
        medians=medians,
    )
    transition_probability = _proba2(payload["transition_model"], x)
    risk_score = np.clip(payload["risk_model"].predict(x), 0.0, 1.0)
    return _output(
        transformed["timestamp"],
        transition_probability,
        risk_score,
        float(payload["threshold"]),
    )


def load_frame(market_path: Path, hmm_path: Path, risk_artifact: Path) -> pd.DataFrame:
    market = _read_unique(market_path)
    missing_market = sorted(set(REQUIRED_MARKET_COLS) - set(market.columns))
    if missing_market:
        raise RuntimeError(f"{market_path} missing market columns: {missing_market}")
    hmm = _read_unique(hmm_path)
    common_start = max(market["timestamp"].iloc[0], hmm["timestamp"].iloc[0])
    common_end = min(market["timestamp"].iloc[-1], hmm["timestamp"].iloc[-1])
    if common_start > common_end:
        raise RuntimeError("market/HMM/risk sidecars have no common timestamp range")
    if common_start != market["timestamp"].iloc[0] or common_end != market["timestamp"].iloc[-1]:
        print(
            json.dumps(
                {
                    "stage": "exact_sidecar_range",
                    "market": str(market_path),
                    "common_start": str(common_start),
                    "common_end": str(common_end),
                }
            ),
            flush=True,
        )
    market = market.loc[market["timestamp"].between(common_start, common_end)].reset_index(drop=True)
    risk = infer_transition_risk(market, risk_artifact)
    expected_hmm = {
        f"{HMM_PREFIX}{name}_prob" for name in HMM_CLASSES
    } | {
        f"{HMM_PREFIX}confidence",
        f"{HMM_PREFIX}margin",
        f"{HMM_PREFIX}entropy",
    }
    missing_hmm = sorted(expected_hmm - set(hmm.columns))
    if missing_hmm:
        raise RuntimeError(f"{hmm_path} missing HMM columns: {missing_hmm}")
    expected_risk = {
        "regime3_stability_h6_score",
        "regime3_transition_h6_risk_prob",
        "regime3_churn_h6_risk_score",
    }
    missing_risk = sorted(expected_risk - set(risk.columns))
    if missing_risk:
        raise RuntimeError(f"{risk_artifact} missing risk columns: {missing_risk}")
    out = market.merge(hmm, on="timestamp", how="left", validate="one_to_one")
    out = out.merge(risk, on="timestamp", how="left", validate="one_to_one")
    joined = sorted(expected_hmm | expected_risk)
    missing_rows = out[joined].isna().any(axis=1)
    if missing_rows.any():
        sample = out.loc[missing_rows, "timestamp"].head(10).astype(str).tolist()
        raise RuntimeError(f"sidecars do not cover market timestamps: {sample}")
    keep = REQUIRED_MARKET_COLS + sorted(expected_hmm | expected_risk)
    out = out[keep]
    numeric = out.drop(columns="timestamp").apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        bad = [c for c in numeric if not np.isfinite(numeric[c].to_numpy(dtype=np.float64)).all()]
        raise RuntimeError(f"non-finite input columns: {bad[:20]}")
    return out


def validate_hmm_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = joblib.load(path)
    required = {"model_id", "classes", "feature_cols", "scaler", "model", "state_class_matrix"}
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"{path} missing payload keys: {missing}")
    if tuple(payload["classes"]) != HMM_CLASSES:
        raise RuntimeError(f"HMM class contract mismatch: {payload['classes']}")
    if payload.get("prefix_stem") != "regime3_current_sensitive":
        raise RuntimeError(f"HMM prefix contract mismatch: {payload.get('prefix_stem')}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "model_id": str(payload["model_id"]),
        "classes": list(payload["classes"]),
        "feature_cols": list(payload["feature_cols"]),
        "state_count": int(payload.get("state_count", 0)),
        "sticky": float(payload.get("sticky", 0.0)),
    }


def derive_route_thresholds(frame_2025: pd.DataFrame) -> RouteThresholds:
    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2025-08-31 23:55:00")
    fit = frame_2025.loc[frame_2025["timestamp"].between(start, end)].copy()
    if len(fit) < 1000:
        raise RuntimeError("insufficient threshold-fit rows")
    return RouteThresholds(
        confidence_min=float(fit[f"{HMM_PREFIX}confidence"].quantile(0.25)),
        margin_min=float(fit[f"{HMM_PREFIX}margin"].quantile(0.25)),
        entropy_max=float(fit[f"{HMM_PREFIX}entropy"].quantile(0.75)),
        transition_risk_max=float(fit["regime3_transition_h6_risk_prob"].quantile(0.80)),
        churn_risk_max=float(fit["regime3_churn_h6_risk_score"].quantile(0.80)),
        fit_start=str(start),
        fit_end=str(end),
    )


def compute_vwma(close: pd.Series, volume: pd.Series, window: int) -> np.ndarray:
    cv = (close * volume).rolling(window, min_periods=window).sum()
    vv = volume.rolling(window, min_periods=window).sum()
    return (cv / vv.where(vv > 1.0e-12)).to_numpy(dtype=np.float64)


def compute_atr(frame: pd.DataFrame, window: int = ATR_WINDOW) -> np.ndarray:
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    previous = np.r_[np.nan, close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    return pd.Series(tr).rolling(window, min_periods=window).mean().to_numpy(dtype=np.float64)


def compute_confirmed_rsi_divergence(
    high: np.ndarray,
    low: np.ndarray,
    rsi: np.ndarray,
    *,
    left: int = 3,
    right: int = 3,
    active_bars: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Release divergence only after the second price pivot is confirmed."""
    n = len(rsi)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    last_low = -1
    last_high = -1
    bull_until = -1
    bear_until = -1
    for release in range(left + right, n):
        pivot = release - right
        lo_slice = low[pivot - left : pivot + right + 1]
        hi_slice = high[pivot - left : pivot + right + 1]
        if np.isfinite(low[pivot]) and low[pivot] <= float(np.min(lo_slice)):
            if last_low >= 0 and low[pivot] < low[last_low] and rsi[pivot] > rsi[last_low]:
                bull_until = release + active_bars - 1
            last_low = pivot
        if np.isfinite(high[pivot]) and high[pivot] >= float(np.max(hi_slice)):
            if last_high >= 0 and high[pivot] > high[last_high] and rsi[pivot] < rsi[last_high]:
                bear_until = release + active_bars - 1
            last_high = pivot
        bull[release] = release <= bull_until
        bear[release] = release <= bear_until
        if release + 1 < n:
            bull[release + 1] = release + 1 <= bull_until
            bear[release + 1] = release + 1 <= bear_until
    return bull, bear


def compute_rolling_vpvr(
    frame: pd.DataFrame,
    *,
    window: int = VPVR_WINDOW,
    n_bins: int = VPVR_BINS,
    value_area_pct: float = VALUE_AREA_PCT,
    chunk_size: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return causal POC/VAH/VAL using bars [i-window, i-1] for row i."""
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    volume = frame["volume"].to_numpy(dtype=np.float64)
    typical = (high + low + close) / 3.0
    n = len(frame)
    poc = np.full(n, np.nan, dtype=np.float64)
    vah = np.full(n, np.nan, dtype=np.float64)
    val = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    sw = np.lib.stride_tricks.sliding_window_view

    for target_start in range(window, n, chunk_size):
        target_end = min(n, target_start + chunk_size)
        source_start = target_start - window
        source_end = target_end - 1
        tp_windows = sw(typical[source_start:source_end], window)
        vol_windows = sw(volume[source_start:source_end], window)
        low_windows = sw(low[source_start:source_end], window)
        high_windows = sw(high[source_start:source_end], window)
        lo_min = low_windows.min(axis=1)
        hi_max = high_windows.max(axis=1)
        price_range = np.maximum(hi_max - lo_min, 1.0e-9)
        bin_idx = np.clip(
            ((tp_windows - lo_min[:, None]) / price_range[:, None] * n_bins).astype(np.int32),
            0,
            n_bins - 1,
        )
        rows = len(bin_idx)
        row_ids = np.repeat(np.arange(rows, dtype=np.int32), window)
        flat = row_ids * n_bins + bin_idx.ravel()
        hist = np.bincount(flat, weights=vol_windows.ravel(), minlength=rows * n_bins).reshape(rows, n_bins)
        row_index = np.arange(rows)
        poc_idx = hist.argmax(axis=1)
        va_lo = poc_idx.copy()
        va_hi = poc_idx.copy()
        cumulative = hist[row_index, poc_idx].astype(np.float64)
        target = hist.sum(axis=1) * float(value_area_pct)
        for _ in range(n_bins):
            active = (cumulative < target) & ~((va_lo == 0) & (va_hi == n_bins - 1))
            if not active.any():
                break
            below_idx = np.clip(va_lo - 1, 0, n_bins - 1)
            above_idx = np.clip(va_hi + 1, 0, n_bins - 1)
            below = np.where(va_lo > 0, hist[row_index, below_idx], -1.0)
            above = np.where(va_hi < n_bins - 1, hist[row_index, above_idx], -1.0)
            take_above = active & (above >= below) & (above >= 0.0)
            take_below = active & ~take_above & (below >= 0.0)
            va_hi = np.where(take_above, va_hi + 1, va_hi)
            va_lo = np.where(take_below, va_lo - 1, va_lo)
            cumulative += np.where(take_above, above, 0.0) + np.where(take_below, below, 0.0)
        sl = slice(target_start, target_end)
        poc[sl] = lo_min + (poc_idx + 0.5) / n_bins * price_range
        val[sl] = lo_min + va_lo / n_bins * price_range
        vah[sl] = lo_min + (va_hi + 1.0) / n_bins * price_range
        valid[sl] = True
    return poc, vah, val, valid


def append_causal_context(frame: pd.DataFrame, thresholds: RouteThresholds) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)
    out["context_vwma100"] = compute_vwma(close, volume, VWMA_FAST)
    out["context_vwma288"] = compute_vwma(close, volume, VWMA_SLOW)
    out["context_atr192"] = compute_atr(out)
    poc, vah, val, vpvr_valid = compute_rolling_vpvr(out)
    out["context_vpvr_poc"] = poc
    out["context_vpvr_vah"] = vah
    out["context_vpvr_val"] = val
    out["context_vpvr_valid"] = vpvr_valid.astype(np.int8)
    bull_div, bear_div = compute_confirmed_rsi_divergence(
        out["high"].to_numpy(float),
        out["low"].to_numpy(float),
        out["rsi"].to_numpy(float),
    )
    out["context_bull_divergence"] = bull_div.astype(np.int8)
    out["context_bear_divergence"] = bear_div.astype(np.int8)

    pcols = [f"{HMM_PREFIX}{name}_prob" for name in HMM_CLASSES]
    probabilities = out[pcols].to_numpy(dtype=np.float64)
    best = probabilities.argmax(axis=1)
    hmm_ok = (
        (out[f"{HMM_PREFIX}confidence"].to_numpy(float) >= thresholds.confidence_min)
        & (out[f"{HMM_PREFIX}margin"].to_numpy(float) >= thresholds.margin_min)
        & (out[f"{HMM_PREFIX}entropy"].to_numpy(float) <= thresholds.entropy_max)
    )
    transition_veto = (
        (out["regime3_transition_h6_risk_prob"].to_numpy(float) > thresholds.transition_risk_max)
        | (out["regime3_churn_h6_risk_score"].to_numpy(float) > thresholds.churn_risk_max)
    )
    route = np.full(len(out), "uncertain", dtype=object)
    for index, name in enumerate(HMM_CLASSES):
        route[hmm_ok & ~transition_veto & (best == index)] = name
    out["context_regime_route"] = route
    out["context_transition_veto"] = transition_veto.astype(np.int8)
    out["context_regime_sample_weight"] = np.clip(
        out[f"{HMM_PREFIX}confidence"].to_numpy(float)
        * (1.0 - out["regime3_transition_h6_risk_prob"].to_numpy(float)),
        0.05,
        1.0,
    )
    out["context_swing_low24"] = out["low"].rolling(24, min_periods=24).min().shift(1)
    out["context_swing_high24"] = out["high"].rolling(24, min_periods=24).max().shift(1)
    out["context_swing_low48"] = out["low"].rolling(48, min_periods=48).min().shift(1)
    out["context_swing_high48"] = out["high"].rolling(48, min_periods=48).max().shift(1)
    return out


def build_candidate_plans(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"].to_numpy(float)
    atr = frame["context_atr192"].to_numpy(float)
    poc = frame["context_vpvr_poc"].to_numpy(float)
    vah = frame["context_vpvr_vah"].to_numpy(float)
    val = frame["context_vpvr_val"].to_numpy(float)
    vwma_fast = frame["context_vwma100"].to_numpy(float)
    vwma_slow = frame["context_vwma288"].to_numpy(float)
    cluster = frame["cvp_cluster_position"].to_numpy(float)
    imbalance = frame["cvp_volume_imbalance"].to_numpy(float)
    rsi = frame["rsi"].to_numpy(float)
    route = frame["context_regime_route"].astype(str).to_numpy()

    valid = (
        frame["context_vpvr_valid"].to_numpy(bool)
        & np.isfinite(atr)
        & (atr > 0.0)
        & np.isfinite(vwma_fast)
        & np.isfinite(vwma_slow)
    )
    near_poc = np.abs(close - poc) <= 0.75 * atr
    near_vwma = np.abs(close - vwma_fast) <= 0.75 * atr
    lower_zone = (close <= val + 0.25 * atr) | (cluster <= 0.20)
    upper_zone = (close >= vah - 0.25 * atr) | (cluster >= 0.80)

    reclaim_long = (
        (frame["sweep_prev_low_reclaim"].to_numpy(float) > 0.0)
        | (frame["failed_breakout_down"].to_numpy(float) >= 0.25)
        | (frame["lower_wick_z"].to_numpy(float) >= 0.25)
        | (frame["sig_liquidity_trap"].to_numpy(float) >= 0.20)
    )
    reclaim_short = (
        (frame["sweep_prev_high_reclaim"].to_numpy(float) > 0.0)
        | (frame["failed_breakout_up"].to_numpy(float) >= 0.25)
        | (frame["upper_wick_z"].to_numpy(float) >= 0.25)
        | (frame["sig_liquidity_trap"].to_numpy(float) <= -0.20)
    )
    flow_long = (
        (frame["sig_volume_confirm"].to_numpy(float) >= 0.10)
        | ((frame["oi_change_rate"].to_numpy(float) > 0.0) & (imbalance > 0.0))
        | (frame["funding_z_score"].to_numpy(float) <= -1.5)
    )
    flow_short = (
        (frame["sig_volume_confirm"].to_numpy(float) <= -0.10)
        | ((frame["oi_change_rate"].to_numpy(float) > 0.0) & (imbalance < 0.0))
        | (frame["funding_z_score"].to_numpy(float) >= 1.5)
    )
    bull_div = frame["context_bull_divergence"].to_numpy(bool)
    bear_div = frame["context_bear_divergence"].to_numpy(bool)
    trend_long = (
        valid
        & (route == "bull")
        & (frame["mtf_trend_1h"].to_numpy(float) > 0.0)
        & (frame["mtf_trend_4h"].to_numpy(float) >= 0.0)
        & (close >= vwma_slow)
        & (near_poc | near_vwma | reclaim_long)
        & (rsi >= 40.0)
        & (rsi <= 68.0)
        & (reclaim_long | flow_long)
    )
    trend_short = (
        valid
        & (route == "bear")
        & (frame["mtf_trend_1h"].to_numpy(float) < 0.0)
        & (frame["mtf_trend_4h"].to_numpy(float) <= 0.0)
        & (close <= vwma_slow)
        & (near_poc | near_vwma | reclaim_short)
        & (rsi >= 32.0)
        & (rsi <= 60.0)
        & (reclaim_short | flow_short)
    )
    range_long = (
        valid
        & (route == "chop")
        & lower_zone
        & ((rsi <= 38.0) | bull_div)
        & (reclaim_long | flow_long | bull_div)
    )
    range_short = (
        valid
        & (route == "chop")
        & upper_zone
        & ((rsi >= 62.0) | bear_div)
        & (reclaim_short | flow_short | bear_div)
    )

    rows: list[dict[str, Any]] = []
    for i in np.flatnonzero(trend_long | trend_short | range_long | range_short):
        candidates: list[tuple[str, int]] = []
        if trend_long[i]:
            candidates.append(("trend_pullback", 1))
        if trend_short[i]:
            candidates.append(("trend_pullback", -1))
        if range_long[i]:
            candidates.append(("range_reversal", 1))
        if range_short[i]:
            candidates.append(("range_reversal", -1))
        if len(candidates) != 1:
            continue
        setup, side = candidates[0]
        decision_price = float(close[i])
        atr_i = float(atr[i])
        if setup == "range_reversal":
            if side > 0:
                stop = min(float(val[i]), float(frame["context_swing_low48"].iloc[i])) - 0.25 * atr_i
            else:
                stop = max(float(vah[i]), float(frame["context_swing_high48"].iloc[i])) + 0.25 * atr_i
            target = float(poc[i])
            horizon = RANGE_HORIZON
            min_rr = 0.75
        else:
            if side > 0:
                anchors = [x for x in (vwma_fast[i], poc[i], frame["context_swing_low24"].iloc[i]) if np.isfinite(x) and x < decision_price]
                if not anchors:
                    continue
                support = max(anchors)
                stop = float(support - 0.25 * atr_i)
                risk = decision_price - stop
                target = float(max(vah[i], decision_price + 1.5 * risk))
            else:
                anchors = [x for x in (vwma_fast[i], poc[i], frame["context_swing_high24"].iloc[i]) if np.isfinite(x) and x > decision_price]
                if not anchors:
                    continue
                resistance = min(anchors)
                stop = float(resistance + 0.25 * atr_i)
                risk = stop - decision_price
                target = float(min(val[i], decision_price - 1.5 * risk))
            horizon = TREND_HORIZON
            min_rr = 1.25
        reward = side * (target / decision_price - 1.0)
        risk = -side * (stop / decision_price - 1.0)
        if not np.isfinite([target, stop, reward, risk]).all() or reward <= 0.0 or risk <= 0.0:
            continue
        risk_atr = abs(stop - decision_price) / atr_i
        if risk_atr < 0.25 or risk_atr > 4.0 or reward / risk < min_rr:
            continue
        rows.append(
            {
                "decision_index": int(i),
                "decision_timestamp": frame["timestamp"].iloc[i],
                "setup_family": setup,
                "candidate_side": int(side),
                "candidate_side_name": "LONG" if side > 0 else "SHORT",
                "horizon_bars": int(horizon),
                "planned_target_price": target,
                "planned_stop_price": stop,
                "planned_tp_price_move": reward,
                "planned_sl_price_move": risk,
                "planned_rr": reward / risk,
                "context_regime_route": str(route[i]),
                "context_regime_confidence": float(frame[f"{HMM_PREFIX}confidence"].iloc[i]),
                "context_regime_margin": float(frame[f"{HMM_PREFIX}margin"].iloc[i]),
                "context_regime_entropy": float(frame[f"{HMM_PREFIX}entropy"].iloc[i]),
                "context_transition_risk": float(frame["regime3_transition_h6_risk_prob"].iloc[i]),
                "context_churn_risk": float(frame["regime3_churn_h6_risk_score"].iloc[i]),
                "context_sample_weight": float(frame["context_regime_sample_weight"].iloc[i]),
                "context_rsi": float(rsi[i]),
                "context_vwma100": float(vwma_fast[i]),
                "context_vwma288": float(vwma_slow[i]),
                "context_vpvr_poc": float(poc[i]),
                "context_vpvr_vah": float(vah[i]),
                "context_vpvr_val": float(val[i]),
                "context_atr192": atr_i,
                "context_bull_divergence": int(bull_div[i]),
                "context_bear_divergence": int(bear_div[i]),
            }
        )
    return pd.DataFrame(rows)


def load_funding_tape(frame: pd.DataFrame, *, funding_dir: Path = FUNDING_DIR) -> tuple[FundingTape, dict[str, str]]:
    start = frame["timestamp"].iloc[0]
    end = frame["timestamp"].iloc[-1] + pd.Timedelta(minutes=BAR_MINUTES)
    parts: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for path in sorted(funding_dir.glob("ETHUSDT-fundingRate-*.zip")):
        year_month = path.stem.rsplit("-", 2)[-2:]
        if len(year_month) != 2:
            continue
        month_start = pd.Timestamp(f"{year_month[0]}-{year_month[1]}-01")
        if month_start > end or month_start + pd.offsets.MonthEnd(1) < start:
            continue
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        with zipfile.ZipFile(path) as archive:
            with archive.open(archive.namelist()[0]) as handle:
                part = pd.read_csv(handle, usecols=["calc_time", "last_funding_rate"])
        part["timestamp"] = pd.to_datetime(part.pop("calc_time"), unit="ms")
        parts.append(part)
    if not parts:
        raise RuntimeError(f"no ETH funding files cover {start} -> {end}")
    funding = pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    funding = funding.loc[funding["timestamp"].between(start, end, inclusive="both")].reset_index(drop=True)
    gaps = funding["timestamp"].diff().dropna()
    if not gaps.between(pd.Timedelta(hours=7, minutes=59), pd.Timedelta(hours=8, minutes=1)).all():
        raise RuntimeError("ETH funding tape violates the 8-hour interval contract")
    bar_ns = frame["timestamp"].astype("int64").to_numpy()
    funding_ns = funding["timestamp"].astype("int64").to_numpy()
    completed = np.searchsorted(bar_ns, funding_ns, side="left") - 1
    keep = completed >= 0
    rates = funding.loc[keep, "last_funding_rate"].to_numpy(dtype=np.float64)
    settlement_close = frame["close"].to_numpy(dtype=np.float64)[completed[keep]]
    rate_x_price = rates * settlement_close
    return FundingTape(funding_ns[keep], np.r_[0.0, np.cumsum(rate_x_price)]), hashes


def funding_return(tape: FundingTape, entry_ns: int, exit_ns: int, entry_fill: float, side: int) -> float:
    left = int(np.searchsorted(tape.timestamp_ns, entry_ns, side="right"))
    right = int(np.searchsorted(tape.timestamp_ns, exit_ns, side="right"))
    value = tape.rate_x_price_cumsum[right] - tape.rate_x_price_cumsum[left]
    return float(-side * value / max(entry_fill, 1.0e-12))


def split_contract(timestamp: pd.Timestamp) -> tuple[str, pd.Timestamp]:
    if timestamp <= pd.Timestamp("2025-08-31 23:55:00"):
        return "train", pd.Timestamp("2025-08-31 23:55:00")
    if timestamp <= pd.Timestamp("2025-12-31 23:55:00"):
        return "validation", pd.Timestamp("2025-12-31 23:55:00")
    if timestamp <= pd.Timestamp("2026-03-31 23:55:00"):
        return "oos", pd.Timestamp("2026-03-31 23:55:00")
    return "fresh", pd.Timestamp("2026-07-20 00:00:00")


def simulate_candidate(
    frame: pd.DataFrame,
    row: pd.Series,
    tape: FundingTape,
) -> dict[str, Any]:
    i = int(row["decision_index"])
    side = int(row["candidate_side"])
    entry_i = i + 1
    horizon = int(row["horizon_bars"])
    split, split_end = split_contract(pd.Timestamp(row["decision_timestamp"]))
    if entry_i + horizon >= len(frame):
        return {"split": split, "label_valid": 0, "label_invalid_reason": "right_censored"}
    timeout_i = entry_i + horizon
    if frame["timestamp"].iloc[timeout_i] > split_end:
        return {"split": split, "label_valid": 0, "label_invalid_reason": "split_boundary_censored"}

    open_px = frame["open"].to_numpy(float)
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    target = float(row["planned_target_price"])
    stop = float(row["planned_stop_price"])
    entry_fill = float(open_px[entry_i] * (1.0 + side * SLIPPAGE_RATE))
    if side * (target / entry_fill - 1.0) <= 0.0 or -side * (stop / entry_fill - 1.0) <= 0.0:
        return {"split": split, "label_valid": 0, "label_invalid_reason": "entry_gap_invalid"}

    mfe = 0.0
    mae = 0.0
    outcome = "TIMEOUT"
    exit_i = timeout_i
    exit_level = float(open_px[timeout_i])
    for j in range(entry_i, timeout_i):
        favorable = (high[j] / entry_fill - 1.0) if side > 0 else (1.0 - low[j] / entry_fill)
        adverse = (low[j] / entry_fill - 1.0) if side > 0 else (1.0 - high[j] / entry_fill)
        mfe = max(mfe, float(favorable))
        mae = min(mae, float(adverse))
        open_stop = open_px[j] <= stop if side > 0 else open_px[j] >= stop
        open_target = open_px[j] >= target if side > 0 else open_px[j] <= target
        if open_stop:
            outcome, exit_i, exit_level = "SL", j, float(open_px[j])
            break
        if open_target:
            outcome, exit_i, exit_level = "TP", j, target
            break
        hit_stop = low[j] <= stop if side > 0 else high[j] >= stop
        hit_target = high[j] >= target if side > 0 else low[j] <= target
        if hit_stop and hit_target:
            outcome, exit_i, exit_level = "AMBIGUOUS", j, stop
            break
        if hit_stop:
            outcome, exit_i, exit_level = "SL", j, stop
            break
        if hit_target:
            outcome, exit_i, exit_level = "TP", j, target
            break

    exit_fill = float(exit_level * (1.0 - side * SLIPPAGE_RATE))
    gross = float(side * (exit_fill / entry_fill - 1.0))
    entry_fee = FEE_RATE
    exit_fee = FEE_RATE * exit_fill / entry_fill
    entry_ns = int(pd.Timestamp(frame["timestamp"].iloc[entry_i]).value)
    if outcome in {"TP", "SL", "AMBIGUOUS"}:
        exit_timestamp = frame["timestamp"].iloc[exit_i] + pd.Timedelta(minutes=BAR_MINUTES)
    else:
        exit_timestamp = frame["timestamp"].iloc[exit_i]
    exit_ns = int(pd.Timestamp(exit_timestamp).value)
    funding = funding_return(tape, entry_ns, exit_ns, entry_fill, side)
    net = gross - entry_fee - exit_fee + funding
    success = int(outcome == "TP" and net > 0.0)
    return {
        "split": split,
        "entry_index": int(entry_i),
        "entry_timestamp": frame["timestamp"].iloc[entry_i],
        "entry_fill_price": entry_fill,
        "event_end_index": int(exit_i),
        "event_end_timestamp": exit_timestamp,
        "exit_fill_price": exit_fill,
        "label_valid": int(outcome != "AMBIGUOUS"),
        "label_invalid_reason": "same_bar_tp_sl" if outcome == "AMBIGUOUS" else "",
        "label_outcome": outcome,
        "label_success": success,
        "label_net_return_per_notional": net,
        "label_gross_return_per_notional": gross,
        "label_funding_return_per_notional": funding,
        "label_mfe_price_move": mfe,
        "label_mae_price_move": mae,
        "label_bars_to_exit": int(max(exit_i - entry_i + (outcome != "TIMEOUT"), 0)),
    }


def build_labels(frame: pd.DataFrame, candidates: pd.DataFrame, tape: FundingTape) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    outcomes = [simulate_candidate(frame, row, tape) for _, row in candidates.iterrows()]
    return pd.concat([candidates.reset_index(drop=True), pd.DataFrame(outcomes)], axis=1)


def replay_non_overlapping(labels: pd.DataFrame, *, cooldown_bars: int = 6) -> pd.DataFrame:
    valid = labels.loc[(labels["label_valid"] == 1) & labels["label_outcome"].notna()].copy()
    valid = valid.sort_values(
        ["decision_timestamp", "context_sample_weight"], ascending=[True, False]
    )
    selected: list[pd.Series] = []
    next_decision_timestamp = pd.Timestamp.min
    for _, row in valid.iterrows():
        decision_timestamp = pd.Timestamp(row["decision_timestamp"])
        if decision_timestamp < next_decision_timestamp:
            continue
        selected.append(row)
        next_decision_timestamp = pd.Timestamp(row["event_end_timestamp"]) + pd.Timedelta(
            minutes=BAR_MINUTES * int(cooldown_bars)
        )
    if not selected:
        return valid.iloc[:0].copy()
    trades = pd.DataFrame(selected).reset_index(drop=True)
    trades["equity_per_notional"] = (1.0 + trades["label_net_return_per_notional"].astype(float)).cumprod()
    return trades


def summary(labels: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    valid = labels.loc[labels["label_valid"] == 1]
    returns = trades["label_net_return_per_notional"].to_numpy(float) if len(trades) else np.empty(0)
    equity = np.cumprod(1.0 + returns) if len(returns) else np.asarray([1.0])
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1.0e-12) - 1.0
    return {
        "candidate_rows": int(len(labels)),
        "valid_label_rows": int(len(valid)),
        "invalid_reason_counts": labels["label_invalid_reason"].fillna("missing").value_counts().to_dict(),
        "outcome_counts": valid["label_outcome"].value_counts().to_dict(),
        "success_rate": float(valid["label_success"].mean()) if len(valid) else 0.0,
        "setup_counts": valid["setup_family"].value_counts().to_dict(),
        "side_counts": valid["candidate_side_name"].value_counts().to_dict(),
        "policy_trades": int(len(trades)),
        "policy_wins": int((returns > 0.0).sum()),
        "policy_win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
        "policy_compounded_return_per_notional": float(equity[-1] - 1.0),
        "policy_max_drawdown_per_notional": float(drawdown.min()),
    }


def choose_chart_window(trades: pd.DataFrame, *, days: int = 14) -> tuple[pd.Timestamp, pd.Timestamp]:
    if trades.empty:
        return CHART_START, min(CHART_START + pd.Timedelta(days=days), CHART_END)
    decisions = pd.to_datetime(trades["decision_timestamp"])
    best_start = decisions.iloc[0].floor("D")
    best_count = -1
    for timestamp in decisions:
        start = timestamp.floor("D")
        count = int(decisions.between(start, start + pd.Timedelta(days=days)).sum())
        if count > best_count:
            best_start, best_count = start, count
    return best_start, best_start + pd.Timedelta(days=days)


def plot_trade_chart(frame: pd.DataFrame, trades: pd.DataFrame, path: Path) -> dict[str, str]:
    start, end = choose_chart_window(trades)
    view = frame.loc[frame["timestamp"].between(start, end)].copy()
    shown = trades.loc[pd.to_datetime(trades["decision_timestamp"]).between(start, end)].copy()
    fig, (ax, eq_ax) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    ax.plot(view["timestamp"], view["close"], color="#334155", linewidth=1.0, label="ETH close")
    ax.plot(view["timestamp"], view["context_vwma100"], color="#2563eb", linewidth=0.9, label="VWMA100")
    ax.plot(view["timestamp"], view["context_vpvr_poc"], color="#a855f7", linewidth=0.8, alpha=0.8, label="VPVR POC")
    for _, trade in shown.iterrows():
        entry_ts = pd.Timestamp(trade["entry_timestamp"])
        exit_ts = pd.Timestamp(trade["event_end_timestamp"])
        side = int(trade["candidate_side"])
        won = float(trade["label_net_return_per_notional"]) > 0.0
        entry_marker = "^" if side > 0 else "v"
        ax.scatter(entry_ts, trade["entry_fill_price"], marker=entry_marker, s=65, color="#16a34a" if side > 0 else "#dc2626", zorder=5)
        ax.scatter(exit_ts, trade["exit_fill_price"], marker="x", s=50, color="#16a34a" if won else "#dc2626", zorder=5)
        ax.plot([entry_ts, exit_ts], [trade["entry_fill_price"], trade["exit_fill_price"]], color="#16a34a" if won else "#dc2626", linewidth=1.0, alpha=0.65)
    if len(shown):
        shown = shown.sort_values("event_end_timestamp").copy()
        shown["window_equity"] = (1.0 + shown["label_net_return_per_notional"].astype(float)).cumprod()
        eq_ax.step(pd.to_datetime(shown["event_end_timestamp"]), shown["window_equity"], where="post", color="#0f766e", linewidth=1.5)
    else:
        eq_ax.plot(view["timestamp"], np.ones(len(view)), color="#0f766e", linewidth=1.0)
    ax.set_ylabel("ETHUSDT price")
    eq_ax.set_ylabel("Equity")
    eq_ax.set_xlabel("UTC")
    ax.set_title(f"HMM-routed confluence trades: {start.date()} to {end.date()}")
    ax.legend(loc="upper left", ncols=4)
    ax.grid(alpha=0.15)
    eq_ax.grid(alpha=0.15)
    eq_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"start": str(start), "end": str(end), "trades": str(len(shown))}


def _write_split_artifacts(labels: pd.DataFrame, trades: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for split in ("train", "validation", "oos", "fresh"):
        split_labels = labels.loc[labels["split"] == split].copy()
        split_trades = trades.loc[trades["split"] == split].copy()
        label_path = out_dir / f"{split}_meta_labels.parquet"
        trade_path = out_dir / f"{split}_diagnostic_trades.csv"
        split_labels.to_parquet(label_path, index=False)
        split_trades.to_csv(trade_path, index=False)
        artifacts[split] = {
            "labels": str(label_path),
            "trades": str(trade_path),
            "summary": summary(split_labels, split_trades),
        }
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--market-2025", type=Path, default=MARKET_2025)
    parser.add_argument("--market-2026", type=Path, default=MARKET_2026)
    parser.add_argument("--hmm-2025", type=Path, default=HMM_2025)
    parser.add_argument("--hmm-2026", type=Path, default=HMM_2026)
    parser.add_argument("--risk-artifact", type=Path, default=RISK_ARTIFACT)
    parser.add_argument("--hmm-artifact", type=Path, default=HMM_ARTIFACT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hmm_manifest = validate_hmm_artifact(args.hmm_artifact)
    frame_2025 = load_frame(args.market_2025, args.hmm_2025, args.risk_artifact)
    frame_2026 = load_frame(args.market_2026, args.hmm_2026, args.risk_artifact)
    thresholds = derive_route_thresholds(frame_2025)
    all_labels: list[pd.DataFrame] = []
    all_frames: list[pd.DataFrame] = []
    input_hashes: dict[str, str] = {}
    funding_hashes: dict[str, str] = {}
    for year, frame, market_path, hmm_path in (
        (2025, frame_2025, args.market_2025, args.hmm_2025),
        (2026, frame_2026, args.market_2026, args.hmm_2026),
    ):
        print(json.dumps({"stage": "context", "year": year, "rows": len(frame)}), flush=True)
        context = append_causal_context(frame, thresholds)
        candidates = build_candidate_plans(context)
        tape, year_funding_hashes = load_funding_tape(context)
        labels = build_labels(context, candidates, tape)
        labels["source_year"] = int(year)
        all_labels.append(labels)
        all_frames.append(context)
        funding_hashes.update(year_funding_hashes)
        for path in (market_path, hmm_path):
            input_hashes[str(path.relative_to(ROOT))] = sha256(path)
        print(json.dumps({"stage": "labels", "year": year, "candidates": len(candidates), "valid": int(labels.get("label_valid", pd.Series(dtype=int)).sum())}), flush=True)

    combined_labels = pd.concat(all_labels, ignore_index=True).sort_values("decision_timestamp").reset_index(drop=True)
    combined_context = pd.concat(all_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    trades = replay_non_overlapping(combined_labels)
    artifacts = _write_split_artifacts(combined_labels, trades, args.out_dir)
    chart_trades = trades.loc[trades["split"] == "oos"].copy()
    chart_path = args.out_dir / "oos_trade_chart.png"
    chart_window = plot_trade_chart(combined_context, chart_trades, chart_path)

    report = {
        "model_id": MODEL_ID,
        "status": "research_labels_generated",
        "asset": "ETHUSDT",
        "bar_minutes": BAR_MINUTES,
        "hmm": hmm_manifest,
        "transition_risk_artifact": {
            "path": str(args.risk_artifact),
            "sha256": sha256(args.risk_artifact),
            "inference": "causal_full_range_from_frozen_artifact",
        },
        "route_thresholds": asdict(thresholds),
        "label_contract": {
            "entry": "next_bar_open_with_adverse_slippage",
            "range_horizon_bars": RANGE_HORIZON,
            "trend_horizon_bars": TREND_HORIZON,
            "same_bar_tp_sl": "AMBIGUOUS_and_label_valid_0",
            "fee_rate_per_side": FEE_RATE,
            "slippage_rate_per_side": SLIPPAGE_RATE,
            "funding": "actual_ETHUSDT_8h_settlements",
            "primary_target": "label_success",
            "label_success": "TP_first_and_net_return_positive",
        },
        "splits": {
            "train_end": "2025-08-31 23:55:00",
            "validation": ["2025-09-01", "2025-12-31 23:55:00"],
            "oos": ["2026-01-01", "2026-03-31 23:55:00"],
            "fresh": ["2026-04-01", str(combined_context["timestamp"].iloc[-1])],
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "stored_trade_ledger_is_diagnostic_only": True,
        "input_hashes": input_hashes,
        "funding_hashes": funding_hashes,
        "artifacts": artifacts,
        "chart": {"path": str(chart_path), **chart_window},
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path), "chart": str(chart_path), "summaries": {k: v["summary"] for k, v in artifacts.items()}}, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
