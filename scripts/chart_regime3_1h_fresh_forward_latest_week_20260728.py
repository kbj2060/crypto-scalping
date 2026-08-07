#!/usr/bin/env python3
"""Train a research-only 1h Regime3 HMM on 2024 and chart the latest 2026 week.

Five-minute source rows are aggregated into completed one-hour bars. The 2026
filter advances exactly once per completed hour and never uses future rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from retrain_clean_regime_hmm_20260517 import GaussianStateModel  # noqa: E402


TRAIN_2024 = ROOT / "data/splits/year_oos/training_features_2024.csv"
FORWARD_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/regime3_1h_fresh_forward_chart_20260728"
MODEL_OUT = OUT_DIR / "eth_regime3_1h_research_hmm_2024.joblib"
CHART = OUT_DIR / "eth_regime3_1h_fresh_forward_latest_week.png"
FEATURES_OUT = OUT_DIR / "eth_regime3_1h_fresh_forward_latest_week.csv"
AUDIT = OUT_DIR / "eth_regime3_1h_fresh_forward_latest_week_audit.json"

START = pd.Timestamp("2026-07-13 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2026-07-20 00:00:00")
CLASSES = ["bull", "bear", "chop"]
COLORS = {"bull": "#20c997", "bear": "#ff5c77", "chop": "#f4c95d"}
N_STATES = 6
STICKY = 0.93
SEED = 7529

FEATURE_COLS = [
    "log_return_1h",
    "return_3h",
    "return_6h",
    "return_12h",
    "return_24h",
    "ema_slope_5h",
    "trend_efficiency_12h",
    "trend_efficiency_24h",
    "realized_vol_12h",
    "realized_vol_24h",
    "bb_width_20h",
    "bb_width_z_168h",
    "rsi_14h",
    "macd_hist_1h",
    "adx_14h",
    "wick_ratio_1h",
    "volume_z_168h",
    "net_taker_ratio_1h",
    "oi_change_1h",
    "btc_corr_24h",
    "eth_btc_spread_6h",
]


def _read_source(path: Path) -> pd.DataFrame:
    needed = [
        "timestamp", "open", "high", "low", "close", "volume", "taker_buy_base",
        "sum_open_interest_value", "last_funding_rate", "close_btc",
    ]
    frame = pd.read_csv(path, usecols=needed)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def _aggregate_completed_hours(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate [hour, hour+1) and timestamp it at hour+1, when it is known."""
    work = frame.set_index("timestamp")
    rule = dict(label="right", closed="left")
    bars = work.resample("1h", **rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        taker_buy_base=("taker_buy_base", "sum"),
        open_interest=("sum_open_interest_value", "last"),
        funding=("last_funding_rate", "last"),
        close_btc=("close_btc", "last"),
    )
    counts = work["close"].resample("1h", **rule).count()
    bars = bars.loc[counts.eq(12)].dropna(subset=["open", "high", "low", "close"])
    return bars.reset_index()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + gain / (loss + 1e-12))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0.0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0.0), down, 0.0), index=high.index)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-12)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def _trend_efficiency(close: pd.Series, window: int) -> pd.Series:
    direction = close.diff(window).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return direction / (path + 1e-12)


def _with_1h_features(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    log_return = np.log(close / close.shift(1))
    out["log_return_1h"] = log_return
    for hours in (3, 6, 12, 24):
        out[f"return_{hours}h"] = close / close.shift(hours) - 1.0
    ema21 = close.ewm(span=21, adjust=False).mean()
    out["ema_slope_5h"] = (ema21 - ema21.shift(5)) / (close * 5.0 + 1e-12)
    out["trend_efficiency_12h"] = _trend_efficiency(close, 12)
    out["trend_efficiency_24h"] = _trend_efficiency(close, 24)
    out["realized_vol_12h"] = log_return.rolling(12, min_periods=12).std()
    out["realized_vol_24h"] = log_return.rolling(24, min_periods=24).std()
    sma20 = close.rolling(20, min_periods=20).mean()
    out["bb_width_20h"] = 4.0 * close.rolling(20, min_periods=20).std() / (sma20 + 1e-12)
    bb_mean = out["bb_width_20h"].rolling(168, min_periods=48).mean()
    bb_std = out["bb_width_20h"].rolling(168, min_periods=48).std()
    out["bb_width_z_168h"] = (out["bb_width_20h"] - bb_mean) / (bb_std + 1e-12)
    out["rsi_14h"] = _rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist_1h"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (close + 1e-12)
    out["adx_14h"] = _adx(out["high"], out["low"], close, 14)
    candle_range = out["high"] - out["low"]
    out["wick_ratio_1h"] = (candle_range - (out["close"] - out["open"]).abs()) / (candle_range + 1e-12)
    log_volume = np.log1p(out["volume"].clip(lower=0.0))
    out["volume_z_168h"] = (
        (log_volume - log_volume.rolling(168, min_periods=48).mean())
        / (log_volume.rolling(168, min_periods=48).std() + 1e-12)
    )
    out["net_taker_ratio_1h"] = 2.0 * out["taker_buy_base"] / (out["volume"] + 1e-12) - 1.0
    out["oi_change_1h"] = out["open_interest"].pct_change()
    btc_return = np.log(out["close_btc"] / out["close_btc"].shift(1))
    out["btc_corr_24h"] = log_return.rolling(24, min_periods=24).corr(btc_return)
    out["eth_btc_spread_6h"] = out["return_6h"] - (out["close_btc"] / out["close_btc"].shift(6) - 1.0)
    out[FEATURE_COLS] = out[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    return out


def _labels(frame: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(frame), 2, dtype=np.int64)
    adx = frame["adx_14h"].fillna(0.0).to_numpy()
    slope = frame["ema_slope_5h"].fillna(0.0).to_numpy()
    bb_width = frame["bb_width_20h"].fillna(0.0).to_numpy()
    trending = adx >= 16.0
    labels[trending & (slope > 0.00015)] = 0
    labels[trending & (slope < -0.00015)] = 1
    labels[(adx < 12.0) | (bb_width < 0.012)] = 2
    return labels


def _state_class_matrix(state_prob: np.ndarray, labels: np.ndarray, smoothing: float = 0.02) -> np.ndarray:
    matrix = np.full((state_prob.shape[1], len(CLASSES)), smoothing, dtype=np.float64)
    for class_index in range(len(CLASSES)):
        mask = labels == class_index
        matrix[:, class_index] += state_prob[mask].sum(axis=0) / max(int(mask.sum()), 1)
    return matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-300, None)


def _class_probability(state_prob: np.ndarray, state_class: np.ndarray) -> np.ndarray:
    probability = state_prob @ state_class
    return probability / np.clip(probability.sum(axis=1, keepdims=True), 1e-300, None)


def _fit(train: pd.DataFrame) -> dict:
    raw = train[FEATURE_COLS].copy()
    medians = raw.median(numeric_only=True).fillna(0.0)
    x = raw.fillna(medians).fillna(0.0)
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    observations = scaler.fit_transform(x)
    model = GaussianStateModel(N_STATES, 30, SEED, sticky=STICKY).fit(observations)
    state_probability = model.filter_proba(observations)
    state_class = _state_class_matrix(state_probability, _labels(train))
    return {
        "model_id": "eth_regime3_1h_research_hmm_2024_20260728",
        "classes": CLASSES,
        "feature_cols": FEATURE_COLS,
        "feature_medians": medians.to_dict(),
        "scaler": scaler,
        "model": model,
        "state_class_matrix": state_class,
        "state_count": N_STATES,
        "sticky": STICKY,
        "research_only": True,
    }


def _fresh_forward(payload: dict, frame: pd.DataFrame) -> pd.DataFrame:
    medians = pd.Series(payload["feature_medians"])
    raw = frame[FEATURE_COLS].fillna(medians).fillna(0.0)
    observations = payload["scaler"].transform(raw)
    model = payload["model"]
    log_emission = model._log_emission(observations)
    log_transition = np.log(model.A_ + 1e-300)
    state_probability = np.empty((len(frame), model.n_states), dtype=np.float64)
    previous: np.ndarray | None = None
    for bar_index in range(len(frame)):
        if previous is None:
            current = np.log(model.pi_ + 1e-300) + log_emission[bar_index]
        else:
            current = log_emission[bar_index] + model._logsumexp(
                previous[:, None] + log_transition, axis=0
            )
        current -= model._logsumexp(current, axis=0)
        state_probability[bar_index] = np.exp(current)
        previous = current
    probability = _class_probability(state_probability, payload["state_class_matrix"])
    out = frame[["timestamp", "open", "high", "low", "close"]].copy()
    for class_index, name in enumerate(CLASSES):
        out[f"{name}_prob"] = probability[:, class_index]
    sorted_probability = np.sort(probability, axis=1)
    out["confidence"] = sorted_probability[:, -1]
    out["margin"] = sorted_probability[:, -1] - sorted_probability[:, -2]
    out["entropy"] = -np.sum(
        probability * np.log(np.clip(probability, 1e-12, None)), axis=1
    ) / np.log(len(CLASSES))
    out["regime"] = np.asarray(CLASSES, dtype=object)[np.argmax(probability, axis=1)]
    return out


def _shade(ax: plt.Axes, frame: pd.DataFrame) -> None:
    regimes = frame["regime"].to_numpy()
    timestamps = frame["timestamp"].to_numpy()
    starts = np.r_[0, np.flatnonzero(regimes[1:] != regimes[:-1]) + 1]
    stops = np.r_[starts[1:], len(frame)]
    for start, stop in zip(starts, stops, strict=True):
        ax.axvspan(
            pd.Timestamp(timestamps[start]) - pd.Timedelta(hours=1),
            pd.Timestamp(timestamps[stop - 1]),
            color=COLORS[str(regimes[start])],
            alpha=0.15,
            lw=0,
        )


def _plot(frame: pd.DataFrame) -> None:
    plt.style.use("dark_background")
    fig, (ax_price, ax_probability, ax_certainty) = plt.subplots(
        3,
        1,
        figsize=(20, 11),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.45, 1.0], "hspace": 0.1},
    )
    fig.patch.set_facecolor("#0b1018")
    for ax in (ax_price, ax_probability, ax_certainty):
        ax.set_facecolor("#101824")
        ax.grid(True, color="#718096", alpha=0.14, lw=0.6)
        for spine in ax.spines.values():
            spine.set_color("#263449")

    _shade(ax_price, frame)
    ax_price.plot(frame["timestamp"], frame["close"], color="#e7edf6", lw=1.15)
    ax_price.set_ylabel("ETH 1h close (USDT)")
    ax_price.set_title(
        "ETH Research Regime3 — 1-hour Causal Fresh-Forward",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=13,
    )
    ax_price.text(
        0.0,
        1.012,
        "2024-trained 6-state HMM · one update per completed hour · research-only, not promotion-tested",
        transform=ax_price.transAxes,
        color="#93a4b8",
        fontsize=9.5,
    )

    for name in CLASSES:
        ax_probability.plot(
            frame["timestamp"], frame[f"{name}_prob"], color=COLORS[name], lw=1.35,
            label=f"{name} probability",
        )
    ax_probability.set_ylim(0.0, 1.0)
    ax_probability.set_ylabel("Regime probability")
    ax_probability.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)

    ax_certainty.plot(frame["timestamp"], frame["confidence"], color="#76a9fa", lw=1.0, label="confidence")
    ax_certainty.plot(frame["timestamp"], frame["margin"], color="#c084fc", lw=1.0, label="top-2 margin")
    ax_certainty.plot(frame["timestamp"], frame["entropy"], color="#94a3b8", lw=1.0, label="entropy")
    ax_certainty.set_ylim(0.0, 1.0)
    ax_certainty.set_ylabel("Certainty")
    ax_certainty.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
    ax_certainty.xaxis.set_major_locator(mdates.DayLocator())
    ax_certainty.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_certainty.set_xlabel("Completed 1-hour bars (UTC timestamps from feature tape)")

    fig.text(
        0.99, 0.01, "Background: bull (green), bear (red), chop (yellow)",
        ha="right", color="#8190a5", fontsize=8.5,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.94, bottom=0.065)
    fig.savefig(CHART, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _run_lengths(regime: pd.Series) -> np.ndarray:
    values = regime.to_numpy()
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    return np.diff(np.r_[starts, len(values)])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_bars = _with_1h_features(_aggregate_completed_hours(_read_source(TRAIN_2024)))
    forward_bars = _with_1h_features(_aggregate_completed_hours(_read_source(FORWARD_2026)))
    payload = _fit(train_bars)
    joblib.dump(payload, MODEL_OUT)
    forward = _fresh_forward(payload, forward_bars)
    plotted = forward[(forward["timestamp"] > START) & (forward["timestamp"] <= END_EXCLUSIVE)].copy()
    if len(plotted) != 168:
        raise RuntimeError(f"expected 168 complete hourly bars in latest week, got {len(plotted)}")
    plotted.to_csv(FEATURES_OUT, index=False)
    runs = _run_lengths(plotted["regime"])
    audit = {
        "model_id": payload["model_id"],
        "status": "RESEARCH_ONLY_NOT_VALIDATED_NOT_PROMOTION_ELIGIBLE",
        "train_source": str(TRAIN_2024),
        "forward_source": str(FORWARD_2026),
        "model_path": str(MODEL_OUT),
        "chart_path": str(CHART),
        "train_completed_1h_bars": int(len(train_bars)),
        "forward_completed_1h_bars": int(len(forward_bars)),
        "plot_range": [str(plotted["timestamp"].iloc[0]), str(plotted["timestamp"].iloc[-1])],
        "plot_completed_1h_bars": int(len(plotted)),
        "state_count": N_STATES,
        "sticky_initialization": STICKY,
        "feature_count": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "fresh_forward_bar_by_bar": True,
        "future_rows_used_for_current_regime": False,
        "saved_5m_regime_probabilities_used": False,
        "trade_ledgers_used_as_input": False,
        "regime_bar_share": {name: float((plotted["regime"] == name).mean()) for name in CLASSES},
        "regime_flips": int(plotted["regime"].ne(plotted["regime"].shift()).sum() - 1),
        "mean_run_hours": float(runs.mean()),
        "median_run_hours": float(np.median(runs)),
        "close_return_pct": float(100.0 * (plotted["close"].iloc[-1] / plotted["open"].iloc[0] - 1.0)),
    }
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _plot(plotted)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
