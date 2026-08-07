#!/usr/bin/env python3
"""Screen alternative label families on the BTC v1 research data contract.

This is deliberately a research screen, not a promotion artifact.  It keeps one
stationary BTC-only feature set and one execution contract fixed so the observed
difference is attributable mainly to the target/sampling design.  It does not
overwrite the live BTC v1 parent, risk sidecar, or runtime configuration.

The six cases are:
1. fixed_horizon: sign of the 24-hour forward return;
2. meta_label: a primary fixed-horizon side plus a separately trained take/skip model;
3. dollar_event: 5m-derived dollar-activity events with a 24-event forward sign;
4. directional_change: causal 1% reversal-confirmation events;
5. denoised_ssl: paper-inspired train-only convolutional denoiser, then a 24-hour sign;
6. reward_shaping: no class label; regress net long/short reward and choose the best action.

All model inputs at timestamp t are trailing features available at t.  Offline
labels may use the future, but entries are shifted to the next completed hourly
bar and the replay advances sequentially without a saved trade ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
HOURLY_DIR = ROOT / "tmp/causal_regen_20260516/sigma9_1h_btc_20260706"
FIVE_MINUTE_DIR = ROOT / "data/splits/year_oos"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_v1_label_family_suite_20260715"

TRAIN_END = pd.Timestamp("2025-08-31 23:59:59")
VALIDATION_START = pd.Timestamp("2025-09-01 00:00:00")
VALIDATION_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01 00:00:00")
OOS_END = pd.Timestamp("2026-03-31 23:59:59")
HOLDOUT_START = pd.Timestamp("2026-07-14 00:00:00")

HORIZON_HOURS = 24
RESEARCH_DATA_END = OOS_END + pd.Timedelta(hours=HORIZON_HOURS + 1)
DC_THRESHOLD = 0.010
MARGIN_FRACTION = 0.30
LEVERAGE = 2.0
NOTIONAL = MARGIN_FRACTION * LEVERAGE
ROUND_TRIP_COST = 0.0014
SEEDS = (310713, 310719, 310727)
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}


@dataclass(frozen=True)
class LabelCase:
    name: str
    label: np.ndarray
    eligible: np.ndarray
    detail: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_hourly() -> tuple[pd.DataFrame, list[str]]:
    frames = []
    expected: list[str] | None = None
    for year in (2024, 2025, 2026):
        path = HOURLY_DIR / f"sigma9_btc_1h_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        if expected is None:
            expected = list(frame.columns)
        elif list(frame.columns) != expected:
            raise RuntimeError(f"hourly feature contract mismatch: {path}")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    out = out.loc[out["timestamp"] <= RESEARCH_DATA_END].reset_index(drop=True)
    delta = out["timestamp"].diff().dropna()
    if not delta.eq(pd.Timedelta(hours=1)).all():
        raise RuntimeError("hourly BTC frame is not continuous")
    features = [column for column in out.columns if column not in NON_FEATURE]
    if len(features) != 28:
        raise RuntimeError(f"expected 28 BTC-only stationary features, got {len(features)}")
    forbidden = [column for column in features if any(token in column.lower() for token in ("target", "future", "label", "pnl"))]
    if forbidden:
        raise RuntimeError(f"forbidden feature columns: {forbidden}")
    finite_rows = np.isfinite(out[features].to_numpy(dtype=np.float64)).all(axis=1)
    if not finite_rows.any():
        raise RuntimeError("hourly features contain no fully initialized row")
    first_complete = int(np.flatnonzero(finite_rows)[0])
    if not finite_rows[first_complete:].all():
        bad = out.loc[~pd.Series(finite_rows, index=out.index) & out.index.to_series().ge(first_complete), "timestamp"]
        raise RuntimeError(f"non-finite hourly features after warm-up: {bad.head(10).tolist()}")
    out = out.iloc[first_complete:].reset_index(drop=True)
    return out, features


def _read_hourly_dollar_volume() -> pd.Series:
    frames = []
    for year in (2024, 2025, 2026):
        path = FIVE_MINUTE_DIR / f"btc_features_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=["timestamp", "close", "volume"], parse_dates=["timestamp"])
        frame = frame.loc[frame["timestamp"] <= RESEARCH_DATA_END]
        frame["dollar_volume"] = pd.to_numeric(frame["close"], errors="raise") * pd.to_numeric(frame["volume"], errors="raise")
        frames.append(frame[["timestamp", "dollar_volume"]])
    five = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    return five.set_index("timestamp")["dollar_volume"].resample("1h", label="left", closed="left").sum()


def fixed_horizon_labels(close: np.ndarray, horizon: int = HORIZON_HOURS) -> tuple[np.ndarray, np.ndarray]:
    close = np.asarray(close, dtype=np.float64)
    future = np.full(len(close), np.nan, dtype=np.float64)
    future[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    label = np.zeros(len(close), dtype=np.int8)
    label[future > 0.0] = 1
    label[future <= 0.0] = 2
    eligible = np.isfinite(future)
    return label, eligible


def directional_change_events(close: np.ndarray, threshold: float = DC_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """Return causal reversal-confirmation event labels (LONG=1, SHORT=2)."""
    prices = np.asarray(close, dtype=np.float64)
    label = np.zeros(len(prices), dtype=np.int8)
    eligible = np.zeros(len(prices), dtype=bool)
    if not len(prices):
        return label, eligible
    mode = 0  # 0=uninitialized, +1=upturn/track high, -1=downturn/track low
    high = low = float(prices[0])
    for i in range(1, len(prices)):
        price = float(prices[i])
        if mode == 0:
            high = max(high, price)
            low = min(low, price)
            if price >= low * (1.0 + threshold):
                mode = 1
                high = price
                label[i] = 1
                eligible[i] = True
            elif price <= high * (1.0 - threshold):
                mode = -1
                low = price
                label[i] = 2
                eligible[i] = True
        elif mode > 0:
            high = max(high, price)
            if price <= high * (1.0 - threshold):
                mode = -1
                low = price
                label[i] = 2
                eligible[i] = True
        else:
            low = min(low, price)
            if price >= low * (1.0 + threshold):
                mode = 1
                high = price
                label[i] = 1
                eligible[i] = True
    return label, eligible


def dollar_event_labels(
    close: np.ndarray,
    dollar_volume: np.ndarray,
    train_mask: np.ndarray,
    horizon_events: int = HORIZON_HOURS,
) -> tuple[np.ndarray, np.ndarray, float]:
    close = np.asarray(close, dtype=np.float64)
    activity = np.asarray(dollar_volume, dtype=np.float64)
    threshold = float(np.median(activity[np.asarray(train_mask, dtype=bool)]))
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise RuntimeError(f"invalid dollar-bar threshold: {threshold}")
    event_indices: list[int] = []
    accumulator = 0.0
    for i, value in enumerate(activity):
        accumulator += max(float(value), 0.0)
        if accumulator >= threshold:
            event_indices.append(i)
            accumulator %= threshold
    label = np.zeros(len(close), dtype=np.int8)
    eligible = np.zeros(len(close), dtype=bool)
    for position, i in enumerate(event_indices[:-horizon_events]):
        j = event_indices[position + horizon_events]
        label[i] = 1 if close[j] > close[i] else 2
        eligible[i] = True
    return label, eligible, threshold


class CausalConvDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 4, kernel_size=5, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(4, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def denoise_close_train_only(
    close: np.ndarray,
    train_mask: np.ndarray,
    *,
    seed: int = SEEDS[0],
    window: int = 48,
    epochs: int = 8,
) -> tuple[np.ndarray, dict[str, Any], CausalConvDenoiser]:
    """Fit only on training windows and reconstruct each t from a trailing window.

    The self-supervised target is a causal SMA/EMA consensus, mirroring the
    reconstruction target family in arXiv:2112.10139 while avoiding future input.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    log_close = np.log(np.asarray(close, dtype=np.float64))
    train_values = log_close[np.asarray(train_mask, dtype=bool)]
    mean, std = float(train_values.mean()), float(train_values.std())
    normalized = ((log_close - mean) / max(std, 1e-8)).astype(np.float32)
    series = pd.Series(normalized)
    smooth = ((series.rolling(6, min_periods=1).mean() + series.ewm(span=12, adjust=False).mean()) / 2.0).to_numpy(dtype=np.float32)
    train_end = int(np.flatnonzero(train_mask)[-1])
    starts = np.arange(0, train_end - window + 2, dtype=np.int64)
    x = np.stack([normalized[start : start + window] for start in starts])[:, None, :]
    y = np.stack([smooth[start : start + window] for start in starts])[:, None, :]
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)
    model = CausalConvDenoiser()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    last_loss = np.nan
    for _ in range(epochs):
        losses = []
        for pure, target in loader:
            noisy = pure + 0.08 * torch.randn_like(pure)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(noisy), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        last_loss = float(np.mean(losses))
    reconstructed = np.full(len(close), np.nan, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(close) - window + 1, 512):
            ends = np.arange(start + window - 1, min(start + 512 + window - 1, len(close)), dtype=np.int64)
            batch = np.stack([normalized[end - window + 1 : end + 1] for end in ends])[:, None, :]
            prediction = model(torch.from_numpy(batch))[:, 0, -1].numpy()
            reconstructed[ends] = prediction
    reconstructed[: window - 1] = smooth[: window - 1]
    calibration_rows = np.asarray(train_mask, dtype=bool) & np.isfinite(reconstructed)
    slope, intercept = np.polyfit(reconstructed[calibration_rows], smooth[calibration_rows], deg=1)
    reconstructed = slope * reconstructed + intercept
    denoised_close = np.exp(reconstructed * max(std, 1e-8) + mean)
    if not np.isfinite(denoised_close).all():
        raise RuntimeError("denoiser produced non-finite prices")
    diag = {
        "fit_rows": int(train_end + 1),
        "window": int(window),
        "epochs": int(epochs),
        "final_reconstruction_loss": last_loss,
        "train_only_scale_calibration": {"slope": float(slope), "intercept": float(intercept)},
        "train_only_fit": True,
        "causal_trailing_reconstruction": True,
    }
    return denoised_close, diag, model


def _fit_classifiers(x: np.ndarray, y: np.ndarray, seeds: tuple[int, ...] = SEEDS) -> list[HistGradientBoostingClassifier]:
    models = []
    for seed in seeds:
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.04,
            max_iter=220,
            max_depth=4,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            class_weight="balanced",
            random_state=int(seed),
        )
        model.fit(x, y)
        models.append(model)
    return models


def _predict_action_probability(models: list[HistGradientBoostingClassifier], x: np.ndarray) -> np.ndarray:
    probability = np.zeros((len(x), 3), dtype=np.float64)
    for model in models:
        raw = model.predict_proba(x)
        for column, cls in enumerate(model.classes_):
            probability[:, int(cls)] += raw[:, column]
    probability /= len(models)
    return probability


def _fit_binary_models(x: np.ndarray, y: np.ndarray) -> list[HistGradientBoostingClassifier]:
    return _fit_classifiers(x, y)


def _predict_binary(models: list[HistGradientBoostingClassifier], x: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x), dtype=np.float64)
    for model in models:
        raw = model.predict_proba(x)
        class_index = {int(cls): i for i, cls in enumerate(model.classes_)}
        out += raw[:, class_index[1]]
    return out / len(models)


def _future_return(close: np.ndarray, horizon: int = HORIZON_HOURS) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype=np.float64)
    out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    return out


def fresh_forward_replay(frame: pd.DataFrame, signal: np.ndarray, mask: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Sequential one-position replay with next-bar entry and a fixed 24h exit."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise RuntimeError("empty replay split")
    start, end = int(indices[0]), int(indices[-1])
    open_price = frame["open"].to_numpy(dtype=np.float64)
    close_price = frame["close"].to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    busy_until = -1
    rows = []
    curve = np.ones(end - start + 1, dtype=np.float64)
    for i in range(start, end + 1):
        curve[i - start] = cash
        if i <= busy_until or signal[i] == 0 or i + 1 > end:
            continue
        entry_i = i + 1
        exit_i = min(entry_i + HORIZON_HOURS, end)
        if exit_i <= entry_i:
            continue
        side = 1 if int(signal[i]) == 1 else -1
        raw_return = side * (close_price[exit_i] / open_price[entry_i] - 1.0)
        account_return = NOTIONAL * (raw_return - ROUND_TRIP_COST)
        cash *= max(1.0 + account_return, 1e-9)
        peak = max(peak, cash)
        mdd = min(mdd, cash / peak - 1.0)
        busy_until = exit_i
        rows.append(
            {
                "signal_timestamp": frame["timestamp"].iloc[i],
                "entry_timestamp": frame["timestamp"].iloc[entry_i],
                "exit_timestamp": frame["timestamp"].iloc[exit_i],
                "side": side,
                "entry_price": float(open_price[entry_i]),
                "exit_price": float(close_price[exit_i]),
                "raw_return": float(raw_return),
                "account_return": float(account_return),
                "equity": float(cash),
                "exit_reason": "fixed_24h",
            }
        )
        curve[entry_i - start : exit_i - start + 1] = cash
    ledger = pd.DataFrame(rows)
    metrics = {
        "pnl": float(cash - 1.0),
        "mdd": float(mdd),
        "calmar": float((cash - 1.0) / abs(mdd)) if mdd < 0 else 0.0,
        "trades": int(len(ledger)),
        "win_rate": float((ledger["account_return"] > 0).mean()) if len(ledger) else 0.0,
        "long_trades": int((ledger["side"] > 0).sum()) if len(ledger) else 0,
        "short_trades": int((ledger["side"] < 0).sum()) if len(ledger) else 0,
    }
    return metrics, ledger, curve


def _threshold_signal(probability: np.ndarray, eligible: np.ndarray, threshold: float) -> np.ndarray:
    action = probability.argmax(axis=1).astype(np.int8)
    confidence = probability[np.arange(len(probability)), action]
    signal = np.where(np.asarray(eligible, dtype=bool) & (action != 0) & (confidence >= threshold), action, 0).astype(np.int8)
    return signal


def _select_threshold(frame: pd.DataFrame, probability: np.ndarray, eligible: np.ndarray, validation_mask: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for threshold in THRESHOLDS:
        signal = _threshold_signal(probability, eligible, threshold)
        metrics, _, _ = fresh_forward_replay(frame, signal, validation_mask)
        rows.append({"threshold": threshold, **metrics})
    eligible_rows = [row for row in rows if row["trades"] >= 5]
    pool = eligible_rows or rows
    selected = max(pool, key=lambda row: (row["calmar"], row["pnl"], -row["threshold"]))
    return float(selected["threshold"]), rows


def _classification_diagnostic(label: np.ndarray, probability: np.ndarray, eligible: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    use = np.asarray(eligible, dtype=bool) & np.asarray(mask, dtype=bool)
    if not use.any():
        return {"rows": 0, "balanced_accuracy": None}
    pred = probability[use].argmax(axis=1)
    return {
        "rows": int(use.sum()),
        "balanced_accuracy": float(balanced_accuracy_score(label[use], pred)),
        "label_counts": {str(int(k)): int(v) for k, v in pd.Series(label[use]).value_counts().sort_index().items()},
    }


def _fit_meta_label(
    x: np.ndarray,
    fixed_label: np.ndarray,
    fixed_eligible: np.ndarray,
    future_return: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[list[HistGradientBoostingClassifier], list[HistGradientBoostingClassifier], np.ndarray, np.ndarray, dict[str, Any]]:
    train_indices = np.flatnonzero(train_mask & fixed_eligible)
    splitter = TimeSeriesSplit(n_splits=5)
    oof_probability = np.full((len(x), 3), np.nan, dtype=np.float64)
    fold_rows = []
    for fold, (fit_local, test_local) in enumerate(splitter.split(train_indices), start=1):
        fit_idx, test_idx = train_indices[fit_local], train_indices[test_local]
        models = _fit_classifiers(x[fit_idx], fixed_label[fit_idx], seeds=(SEEDS[fold % len(SEEDS)],))
        oof_probability[test_idx] = _predict_action_probability(models, x[test_idx])
        fold_rows.append({"fold": fold, "train_rows": int(len(fit_idx)), "oof_rows": int(len(test_idx))})
    primary_models = _fit_classifiers(x[train_indices], fixed_label[train_indices])
    primary_probability = _predict_action_probability(primary_models, x)
    oof_rows = train_mask & np.isfinite(oof_probability).all(axis=1)
    oof_side = oof_probability.argmax(axis=1).astype(np.int8)
    signed_return = np.where(oof_side == 1, future_return, -future_return)
    meta_target = (signed_return > ROUND_TRIP_COST).astype(np.int8)
    oof_conf = np.zeros(len(x), dtype=np.float64)
    oof_conf[oof_rows] = np.max(oof_probability[oof_rows], axis=1)
    oof_margin = np.nan_to_num(np.abs(oof_probability[:, 1] - oof_probability[:, 2]), nan=0.0)
    meta_x = np.column_stack([x, np.nan_to_num(oof_conf), np.nan_to_num(oof_margin), np.where(oof_side == 1, 1.0, -1.0)])
    meta_models = _fit_binary_models(meta_x[oof_rows], meta_target[oof_rows])
    primary_side = primary_probability.argmax(axis=1).astype(np.int8)
    primary_conf = primary_probability.max(axis=1)
    primary_margin = np.abs(primary_probability[:, 1] - primary_probability[:, 2])
    inference_x = np.column_stack([x, primary_conf, primary_margin, np.where(primary_side == 1, 1.0, -1.0)])
    meta_probability = _predict_binary(meta_models, inference_x)
    detail = {
        "primary_label": "fixed_horizon_24h_sign",
        "meta_target": "primary_side_net_24h_return_positive",
        "genuine_primary_oof_rows": int(oof_rows.sum()),
        "folds": fold_rows,
        "primary_and_meta_models_separate": True,
    }
    return primary_models, meta_models, primary_probability, meta_probability, detail


def _fit_reward_models(x: np.ndarray, future_return: np.ndarray, train_mask: np.ndarray) -> tuple[dict[str, HistGradientBoostingRegressor], np.ndarray]:
    models = {}
    rewards = np.column_stack(
        [
            np.zeros(len(x), dtype=np.float64),
            NOTIONAL * (future_return - ROUND_TRIP_COST),
            NOTIONAL * (-future_return - ROUND_TRIP_COST),
        ]
    )
    finite_train = np.asarray(train_mask, dtype=bool) & np.isfinite(future_return)
    predicted = np.zeros_like(rewards)
    for action, name in ((1, "long"), (2, "short")):
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=220,
            max_depth=4,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=SEEDS[action],
        )
        model.fit(x[finite_train], np.clip(rewards[finite_train, action], -0.20, 0.20))
        predicted[:, action] = model.predict(x)
        models[name] = model
    return models, predicted


def _plot_label_chart(
    frame: pd.DataFrame,
    name: str,
    label: np.ndarray,
    eligible: np.ndarray,
    signal: np.ndarray,
    output: Path,
    *,
    denoised_close: np.ndarray | None = None,
    meta_probability: np.ndarray | None = None,
) -> None:
    chart_mask = (frame["timestamp"] >= OOS_START) & (frame["timestamp"] < OOS_START + pd.Timedelta(days=21))
    idx = np.flatnonzero(chart_mask)
    timestamps = frame["timestamp"].iloc[idx]
    close = frame["close"].to_numpy(dtype=float)[idx]
    local_label = label[idx]
    local_eligible = eligible[idx]
    local_signal = signal[idx]
    colors = {0: "#8b949e", 1: "#2ca02c", 2: "#d62728"}
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [4, 1.25, 1.4]})
    ax = axes[0]
    ax.plot(timestamps, close, color="#1f77b4", linewidth=1.25, label="BTC close")
    if denoised_close is not None:
        ax.plot(timestamps, denoised_close[idx], color="#ff7f0e", linewidth=1.0, label="denoised close")
    for action, marker, label_name in ((1, "^", "LONG label"), (2, "v", "SHORT label")):
        use = local_eligible & (local_label == action)
        if use.any():
            ax.scatter(timestamps.iloc[np.flatnonzero(use)], close[use], s=24, marker=marker, color=colors[action], alpha=0.55, label=label_name)
    ax.set_ylabel("BTC price (USD)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    ax.set_title(f"{name}: offline labels over BTC price (2026-01-01 to 2026-01-21)")

    strip = np.where(local_eligible, local_label, 0)
    axes[1].step(timestamps, strip, where="post", color="#6f42c1", linewidth=1.0)
    axes[1].set_yticks([0, 1, 2], ["NONE/CASH", "LONG", "SHORT"])
    axes[1].set_ylabel("Label")
    axes[1].grid(alpha=0.2)

    accepted = local_signal != 0
    axes[2].scatter(timestamps, np.zeros(len(idx)), s=8, color="#c7c7c7", alpha=0.35, label="no action")
    for action, marker, label_name in ((1, "^", "executed LONG signal"), (2, "v", "executed SHORT signal")):
        use = accepted & (local_signal == action)
        if use.any():
            y = np.ones(use.sum()) if action == 1 else -np.ones(use.sum())
            axes[2].scatter(timestamps.iloc[np.flatnonzero(use)], y, s=34, marker=marker, color=colors[action], label=label_name)
    if meta_probability is not None:
        axes[2].plot(timestamps, 2.0 * meta_probability[idx] - 1.0, color="#9467bd", linewidth=0.8, alpha=0.8, label="meta probability (scaled)")
    axes[2].set_yticks([-1, 0, 1], ["SHORT", "SKIP", "LONG"])
    axes[2].set_ylabel("Model")
    axes[2].grid(alpha=0.2)
    axes[2].legend(loc="upper left", ncol=3, fontsize=8)
    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--denoiser-epochs", type=int, default=8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_data", flush=True)
    frame, feature_columns = _read_hourly()
    dollar = _read_hourly_dollar_volume()
    frame = frame.merge(dollar.rename("dollar_volume"), left_on="timestamp", right_index=True, how="left", validate="one_to_one")
    if frame["dollar_volume"].isna().any():
        raise RuntimeError("missing 5m-derived hourly dollar volume")
    timestamps = frame["timestamp"]
    train_mask = timestamps.le(TRAIN_END).to_numpy()
    validation_mask = timestamps.between(VALIDATION_START, VALIDATION_END).to_numpy()
    oos_mask = timestamps.between(OOS_START, OOS_END).to_numpy()
    if timestamps[oos_mask].max() >= HOLDOUT_START:
        raise RuntimeError("OOS crosses frozen holdout")
    x = frame[feature_columns].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    future_return = _future_return(close)

    fixed_label, fixed_eligible = fixed_horizon_labels(close)
    dc_label, dc_eligible = directional_change_events(close)
    event_label, event_eligible, event_threshold = dollar_event_labels(
        close, frame["dollar_volume"].to_numpy(dtype=np.float64), train_mask
    )
    print("stage=fit_denoiser", flush=True)
    denoised_close, denoiser_diag, denoiser = denoise_close_train_only(
        close, train_mask, epochs=int(args.denoiser_epochs)
    )
    denoised_label, denoised_eligible = fixed_horizon_labels(denoised_close)

    cases = [
        LabelCase("fixed_horizon", fixed_label, fixed_eligible, {"horizon_hours": HORIZON_HOURS, "threshold": 0.0}),
        LabelCase("dollar_event", event_label, event_eligible, {"dollar_threshold": event_threshold, "horizon_events": HORIZON_HOURS, "source": "5m close_x_volume"}),
        LabelCase("directional_change", dc_label, dc_eligible, {"reversal_threshold": DC_THRESHOLD, "event_timestamp_is_confirmation": True}),
        LabelCase("denoised_ssl", denoised_label, denoised_eligible, denoiser_diag),
    ]

    report_cases: dict[str, Any] = {}
    all_signals: dict[str, np.ndarray] = {}
    bundles: dict[str, Any] = {"feature_columns": feature_columns}
    for case in cases:
        print(f"stage=train_case case={case.name}", flush=True)
        fit_rows = train_mask & case.eligible & (case.label != 0)
        if fit_rows.sum() < 100:
            raise RuntimeError(f"{case.name}: insufficient training labels: {int(fit_rows.sum())}")
        models = _fit_classifiers(x[fit_rows], case.label[fit_rows])
        probability = _predict_action_probability(models, x)
        selected_threshold, grid = _select_threshold(frame, probability, case.eligible, validation_mask)
        signal = _threshold_signal(probability, case.eligible, selected_threshold)
        val_metrics, val_ledger, _ = fresh_forward_replay(frame, signal, validation_mask)
        oos_metrics, oos_ledger, _ = fresh_forward_replay(frame, signal, oos_mask)
        val_ledger.to_csv(args.out_dir / f"{case.name}_validation_ledger.csv", index=False)
        oos_ledger.to_csv(args.out_dir / f"{case.name}_oos_ledger.csv", index=False)
        chart = args.out_dir / f"{case.name}_labels.png"
        _plot_label_chart(
            frame,
            case.name,
            case.label,
            case.eligible,
            signal,
            chart,
            denoised_close=denoised_close if case.name == "denoised_ssl" else None,
        )
        report_cases[case.name] = {
            "detail": case.detail,
            "train_label_rows": int(fit_rows.sum()),
            "validation_event_rows": int((validation_mask & case.eligible).sum()),
            "oos_event_rows": int((oos_mask & case.eligible).sum()),
            "selected_validation_threshold": selected_threshold,
            "validation_threshold_grid": grid,
            "validation_classifier": _classification_diagnostic(case.label, probability, case.eligible, validation_mask),
            "oos_classifier": _classification_diagnostic(case.label, probability, case.eligible, oos_mask),
            "validation": val_metrics,
            "oos": oos_metrics,
            "label_chart": str(chart),
        }
        all_signals[case.name] = signal
        bundles[case.name] = models

    print("stage=train_case case=meta_label", flush=True)
    primary_models, meta_models, primary_probability, meta_probability, meta_detail = _fit_meta_label(
        x, fixed_label, fixed_eligible, future_return, train_mask
    )
    primary_side = primary_probability.argmax(axis=1).astype(np.int8)
    meta_grid = []
    for threshold in THRESHOLDS:
        signal = np.where((primary_side != 0) & (meta_probability >= threshold), primary_side, 0).astype(np.int8)
        metrics, _, _ = fresh_forward_replay(frame, signal, validation_mask)
        meta_grid.append({"threshold": threshold, **metrics})
    meta_pool = [row for row in meta_grid if row["trades"] >= 5] or meta_grid
    meta_selected = max(meta_pool, key=lambda row: (row["calmar"], row["pnl"], -row["threshold"]))
    meta_signal = np.where((primary_side != 0) & (meta_probability >= meta_selected["threshold"]), primary_side, 0).astype(np.int8)
    val_metrics, val_ledger, _ = fresh_forward_replay(frame, meta_signal, validation_mask)
    oos_metrics, oos_ledger, _ = fresh_forward_replay(frame, meta_signal, oos_mask)
    val_ledger.to_csv(args.out_dir / "meta_label_validation_ledger.csv", index=False)
    oos_ledger.to_csv(args.out_dir / "meta_label_oos_ledger.csv", index=False)
    meta_target_all = np.where(primary_side == 1, future_return, -future_return) > ROUND_TRIP_COST
    meta_chart = args.out_dir / "meta_label_labels.png"
    _plot_label_chart(
        frame,
        "meta_label",
        np.where(meta_target_all, primary_side, 0).astype(np.int8),
        np.isfinite(future_return),
        meta_signal,
        meta_chart,
        meta_probability=meta_probability,
    )
    report_cases["meta_label"] = {
        "detail": meta_detail,
        "selected_validation_threshold": float(meta_selected["threshold"]),
        "validation_threshold_grid": meta_grid,
        "validation": val_metrics,
        "oos": oos_metrics,
        "label_chart": str(meta_chart),
    }
    all_signals["meta_label"] = meta_signal
    bundles["meta_label"] = {"primary": primary_models, "meta": meta_models}

    print("stage=train_case case=reward_shaping", flush=True)
    reward_models, predicted_reward = _fit_reward_models(x, future_return, train_mask)
    reward_signal = predicted_reward.argmax(axis=1).astype(np.int8)
    reward_signal[predicted_reward.max(axis=1) <= 0.0] = 0
    val_metrics, val_ledger, _ = fresh_forward_replay(frame, reward_signal, validation_mask)
    oos_metrics, oos_ledger, _ = fresh_forward_replay(frame, reward_signal, oos_mask)
    val_ledger.to_csv(args.out_dir / "reward_shaping_validation_ledger.csv", index=False)
    oos_ledger.to_csv(args.out_dir / "reward_shaping_oos_ledger.csv", index=False)
    oracle_reward = np.column_stack(
        [np.zeros(len(frame)), NOTIONAL * (future_return - ROUND_TRIP_COST), NOTIONAL * (-future_return - ROUND_TRIP_COST)]
    )
    reward_label = np.nan_to_num(oracle_reward, nan=-np.inf).argmax(axis=1).astype(np.int8)
    reward_eligible = np.isfinite(future_return)
    reward_chart = args.out_dir / "reward_shaping_labels.png"
    _plot_label_chart(frame, "reward_shaping", reward_label, reward_eligible, reward_signal, reward_chart)
    report_cases["reward_shaping"] = {
        "detail": {
            "class_labels_used": False,
            "method": "one_step_offline_reward_regression",
            "reward": "notional * (signed_24h_return - round_trip_cost)",
            "cash_reward": 0.0,
            "note": "With a fixed terminal horizon this is a contextual reward model, not a sequential DSAC reproduction.",
        },
        "validation": val_metrics,
        "oos": oos_metrics,
        "label_chart": str(reward_chart),
    }
    all_signals["reward_shaping"] = reward_signal
    bundles["reward_shaping"] = reward_models

    labels = pd.DataFrame({"timestamp": frame["timestamp"], "close": close})
    for case in cases:
        labels[f"{case.name}_label"] = case.label
        labels[f"{case.name}_eligible"] = case.eligible
    labels["meta_label"] = np.where(meta_target_all, primary_side, 0).astype(np.int8)
    labels["meta_probability"] = meta_probability
    labels["reward_oracle_action"] = reward_label
    for name, signal in all_signals.items():
        labels[f"{name}_model_signal"] = signal
    label_path = args.out_dir / "labels_and_signals.parquet"
    labels.to_parquet(label_path, index=False)

    bundle_path = args.out_dir / "research_models.joblib"
    joblib.dump(bundles, bundle_path)
    torch.save({"state_dict": denoiser.state_dict(), "diagnostic": denoiser_diag}, args.out_dir / "denoiser.pt")
    report = {
        "model_id": "btc_v1_label_family_suite_20260715",
        "status": "research_screen_not_promotion_artifact",
        "interpretation": "BTC v1 data/split screen with a common lightweight parent; live TabM/risk/exit artifacts unchanged.",
        "splits": {
            "train_end": TRAIN_END,
            "validation": [VALIDATION_START, VALIDATION_END],
            "oos": [OOS_START, OOS_END],
            "holdout_start": HOLDOUT_START,
        },
        "execution_contract": {
            "entry": "next_hour_open",
            "exit": "fixed_24h_close",
            "margin_fraction": MARGIN_FRACTION,
            "leverage": LEVERAGE,
            "notional": NOTIONAL,
            "round_trip_cost": ROUND_TRIP_COST,
            "single_position": True,
        },
        "feature_contract": {"source": str(HOURLY_DIR), "columns": feature_columns, "count": len(feature_columns)},
        "cases": report_cases,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "post_holdout_data_read": False,
        "live_artifacts_modified": False,
        "promotion_grade": False,
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default))
    manifest = {
        path.name: _sha256(path)
        for path in sorted(args.out_dir.iterdir())
        if path.is_file() and path.name != "manifest.sha256.json"
    }
    (args.out_dir / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved report={report_path}", flush=True)
    for name, case_report in report_cases.items():
        print(f"case={name} validation={case_report['validation']} oos={case_report['oos']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
