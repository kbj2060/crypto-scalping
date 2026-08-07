#!/usr/bin/env python3
"""Plot the active ETH Regime3 surface with an explicit causal bar-by-bar filter.

The frozen 2024 HMM is initialized at the first available 2026 bar. Each next
probability uses only the previous filtered state and the current bar emission.
The saved Regime3 sidecar is used only as an output-parity audit, never as input.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import experiment_regime3_current_hmm_wide24_20260529 as regime_builder  # noqa: E402


FEATURES = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
MODEL = (
    ROOT
    / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
    / "regime3_current_sensitive_hmm_wide24_2024.joblib"
)
SIDECAR = (
    ROOT
    / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
    / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
)
OUT_DIR = ROOT / "tmp/causal_regen_20260516/regime3_fresh_forward_chart_20260728"
CHART = OUT_DIR / "eth_regime3_fresh_forward_2026_june_july.png"
DAILY = OUT_DIR / "eth_regime3_fresh_forward_daily_summary.csv"
AUDIT = OUT_DIR / "eth_regime3_fresh_forward_audit.json"
START = pd.Timestamp("2026-06-01 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2026-08-01 00:00:00")

CLASSES = ["bull", "bear", "chop"]
COLORS = {"bull": "#20c997", "bear": "#ff5c77", "chop": "#f4c95d"}


def _fresh_forward(payload: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Run one causal HMM filtering update per bar, without saved regime inputs."""
    cols = payload["feature_cols"]
    work = regime_builder._with_features(frame, cols)
    medians = pd.Series(payload["feature_medians"])
    x_raw = (
        work[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .fillna(0.0)
    )
    obs = payload["scaler"].transform(x_raw)
    model = payload["model"]
    log_emit = model._log_emission(obs)
    log_trans = np.log(model.A_ + 1e-300)

    state_prob = np.empty((len(obs), model.n_states), dtype=np.float64)
    previous_log_alpha: np.ndarray | None = None
    for bar_index in range(len(obs)):
        if previous_log_alpha is None:
            current = np.log(model.pi_ + 1e-300) + log_emit[bar_index]
        else:
            current = log_emit[bar_index] + model._logsumexp(
                previous_log_alpha[:, None] + log_trans,
                axis=0,
            )
        current -= model._logsumexp(current, axis=0)
        state_prob[bar_index] = np.exp(current)
        previous_log_alpha = current

    class_prob = regime_builder._class_proba(state_prob, payload["state_class_matrix"])
    sorted_prob = np.sort(class_prob, axis=1)
    out = pd.DataFrame({"timestamp": work["timestamp"].reset_index(drop=True)})
    for class_index, name in enumerate(CLASSES):
        out[f"{name}_prob"] = class_prob[:, class_index]
    out["confidence"] = sorted_prob[:, -1]
    out["entropy"] = -np.sum(
        class_prob * np.log(np.clip(class_prob, 1e-12, None)), axis=1
    ) / np.log(len(CLASSES))
    out["margin"] = sorted_prob[:, -1] - sorted_prob[:, -2]
    out["regime"] = np.asarray(CLASSES, dtype=object)[np.argmax(class_prob, axis=1)]
    return out


def _shade_regimes(ax: plt.Axes, frame: pd.DataFrame) -> None:
    regime = frame["regime"].to_numpy()
    timestamps = frame["timestamp"].to_numpy()
    starts = np.r_[0, np.flatnonzero(regime[1:] != regime[:-1]) + 1]
    stops = np.r_[starts[1:], len(frame)]
    bar_delta = pd.Timedelta(minutes=5)
    for start, stop in zip(starts, stops, strict=True):
        left = pd.Timestamp(timestamps[start])
        right = pd.Timestamp(timestamps[stop - 1]) + bar_delta
        ax.axvspan(left, right, color=COLORS[str(regime[start])], alpha=0.13, lw=0)


def _daily_summary(frame: pd.DataFrame) -> pd.DataFrame:
    daily_prob = frame.set_index("timestamp")[[f"{name}_prob" for name in CLASSES]].resample("1D").mean()
    shares = (
        pd.crosstab(frame["timestamp"].dt.floor("1D"), frame["regime"], normalize="index")
        .reindex(columns=CLASSES, fill_value=0.0)
        .rename(columns={name: f"{name}_bar_share" for name in CLASSES})
    )
    daily = daily_prob.join(shares, how="left")
    daily.index.name = "date"
    return daily.reset_index()


def _plot(frame: pd.DataFrame, daily: pd.DataFrame, source_end: pd.Timestamp) -> None:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 13), dpi=150, facecolor="#0b1018")
    grid = fig.add_gridspec(4, 1, height_ratios=[2.5, 1.55, 1.25, 1.1], hspace=0.12)
    axes = [fig.add_subplot(grid[i, 0]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor("#101824")
        ax.grid(True, color="#718096", alpha=0.13, lw=0.6)
        ax.tick_params(colors="#b8c4d4", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#263449")

    ax_price, ax_prob, ax_risk, ax_daily = axes
    _shade_regimes(ax_price, frame)
    ax_price.plot(frame["timestamp"], frame["close"], color="#e7edf6", lw=0.85)
    ax_price.set_ylabel("ETH close (USDT)", color="#dce6f2")
    ax_price.set_title(
        "ETH Active Regime3 — Causal 5-minute Fresh-Forward",
        loc="left",
        fontsize=18,
        fontweight="bold",
        color="#f4f7fb",
        pad=13,
    )
    ax_price.text(
        0.0,
        1.012,
        f"Frozen 2024 HMM  |  one state update per bar  |  plotted through {source_end:%Y-%m-%d %H:%M}",
        transform=ax_price.transAxes,
        fontsize=9.5,
        color="#93a4b8",
    )

    for name in CLASSES:
        values = frame[f"{name}_prob"]
        trailing_hour = values.rolling(12, min_periods=1).mean()
        ax_prob.plot(frame["timestamp"], values, color=COLORS[name], lw=0.28, alpha=0.10)
        ax_prob.plot(
            frame["timestamp"],
            trailing_hour,
            color=COLORS[name],
            lw=1.05,
            alpha=0.92,
            label=f"{name} probability (trailing 1h view)",
        )
    ax_prob.set_ylim(0.0, 1.0)
    ax_prob.set_ylabel("Regime probability")
    ax_prob.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)

    risk_lines = (("confidence", "#76a9fa"), ("margin", "#c084fc"), ("entropy", "#94a3b8"))
    for name, color in risk_lines:
        values = frame[name]
        ax_risk.plot(frame["timestamp"], values, color=color, lw=0.25, alpha=0.08)
        ax_risk.plot(
            frame["timestamp"],
            values.rolling(12, min_periods=1).mean(),
            color=color,
            lw=0.9,
            alpha=0.9,
            label=f"{name} (trailing 1h view)",
        )
    ax_risk.set_ylim(0.0, 1.0)
    ax_risk.set_ylabel("Certainty / uncertainty")
    ax_risk.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)

    dates = pd.to_datetime(daily["date"])
    bottom = np.zeros(len(daily), dtype=float)
    for name in CLASSES:
        values = daily[f"{name}_bar_share"].to_numpy(dtype=float)
        ax_daily.bar(
            dates,
            values,
            bottom=bottom,
            width=0.86,
            color=COLORS[name],
            alpha=0.88,
            label=name,
        )
        bottom += values
    ax_daily.set_ylim(0.0, 1.0)
    ax_daily.set_ylabel("Daily 5m-bar share")
    ax_daily.set_xlabel("2026 (UTC timestamps from feature tape)")

    for ax in axes:
        ax.set_xlim(frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1])
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    fig.text(
        0.995,
        0.006,
        "Background: bull (green), bear (red), chop (yellow) · Saved regime sidecar excluded from inference",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#8190a5",
    )
    fig.subplots_adjust(left=0.068, right=0.985, top=0.95, bottom=0.055)
    fig.savefig(CHART, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = joblib.load(MODEL)
    source = regime_builder._read(FEATURES)
    fresh = _fresh_forward(payload, source)
    joined = source[["timestamp", "close"]].merge(fresh, on="timestamp", validate="one_to_one")

    plot_frame = joined[(joined["timestamp"] >= START) & (joined["timestamp"] < END_EXCLUSIVE)].copy()
    if plot_frame.empty:
        raise RuntimeError("no source bars in the requested June-July 2026 window")
    daily = _daily_summary(plot_frame)
    daily.to_csv(DAILY, index=False)

    saved = pd.read_csv(SIDECAR, parse_dates=["timestamp"])
    prefix = "regime3_current_sensitive_wide24_"
    parity = fresh.merge(saved, on="timestamp", how="inner", validate="one_to_one")
    parity_diffs = {
        name: float(np.max(np.abs(parity[f"{name}_prob"] - parity[f"{prefix}{name}_prob"])))
        for name in CLASSES
    }
    parity_diffs.update(
        {
            name: float(np.max(np.abs(parity[name] - parity[f"{prefix}{name}"])))
            for name in ("confidence", "entropy", "margin")
        }
    )

    month_stats = {}
    for month, part in plot_frame.groupby(plot_frame["timestamp"].dt.to_period("M")):
        month_stats[str(month)] = {
            "bars": int(len(part)),
            "start": str(part["timestamp"].iloc[0]),
            "end": str(part["timestamp"].iloc[-1]),
            "dominant_bar_share": {
                name: float((part["regime"] == name).mean()) for name in CLASSES
            },
            "mean_probability": {
                name: float(part[f"{name}_prob"].mean()) for name in CLASSES
            },
            "close_return_pct": float(100.0 * (part["close"].iloc[-1] / part["close"].iloc[0] - 1.0)),
        }

    audit = {
        "model_id": payload["model_id"],
        "model_fit_period": "2024",
        "feature_source": str(FEATURES),
        "model_path": str(MODEL),
        "chart_path": str(CHART),
        "source_range": [str(source["timestamp"].iloc[0]), str(source["timestamp"].iloc[-1])],
        "plot_range": [str(plot_frame["timestamp"].iloc[0]), str(plot_frame["timestamp"].iloc[-1])],
        "source_rows_processed_sequentially": int(len(source)),
        "plotted_rows": int(len(plot_frame)),
        "bar_interval": "5m",
        "fresh_forward_bar_by_bar": True,
        "future_rows_used_for_current_regime": False,
        "saved_regime_sidecar_used_as_input": False,
        "saved_regime_sidecar_used_for_output_parity_audit_only": True,
        "saved_trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "probability_sum_max_abs_error": float(
            np.max(np.abs(plot_frame[[f"{name}_prob" for name in CLASSES]].sum(axis=1) - 1.0))
        ),
        "sidecar_parity_rows": int(len(parity)),
        "sidecar_parity_max_abs_diff": parity_diffs,
        "monthly_summary": month_stats,
    }
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _plot(plot_frame, daily, source["timestamp"].iloc[-1])
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
