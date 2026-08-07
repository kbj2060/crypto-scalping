#!/usr/bin/env python3
"""Plot BTC and ETH Regime3 HMM features for March 2026 by causal 5m filtering.

The HMMs are frozen 2024 artifacts.  Each 2026 bar is processed once in time
order; no saved regime sidecar or ledger is read as an inference input.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_regime3_current_hmm_wide24_20260529 as regime_builder  # noqa: E402


OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_eth_regime3_fresh_forward_oos_month_20260730"
CHART = OUT_DIR / "btc_eth_regime3_fresh_forward_oos_march_2026.png"
FEATURES_CSV = OUT_DIR / "btc_eth_regime3_fresh_forward_oos_march_2026.csv"
REPORT = OUT_DIR / "btc_eth_regime3_fresh_forward_oos_march_2026.json"
START = pd.Timestamp("2026-03-01 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2026-04-01 00:00:00")
CLASSES = ("bull", "bear", "chop")
COLORS = {"bull": "#20c997", "bear": "#ff5c77", "chop": "#f4c95d"}
ASSETS = {
    "BTC": {
        "features": ROOT / "data/splits/year_oos/btc_features_2026.csv",
        "model": ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_20260708/regime3_current_sensitive_hmm_wide24_2024.joblib",
    },
    "ETH": {
        "features": ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
        "model": ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/regime3_current_sensitive_hmm_wide24_2024.joblib",
    },
}


def _fresh_forward(payload: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """One alpha-filter update per timestamp; every row sees only t and t-1."""
    cols = payload["feature_cols"]
    work = regime_builder._with_features(frame, cols)
    medians = pd.Series(payload["feature_medians"])
    raw = work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    obs = payload["scaler"].transform(raw.fillna(medians).fillna(0.0))
    model = payload["model"]
    log_emit = model._log_emission(obs)
    log_trans = np.log(model.A_ + 1e-300)
    state_prob = np.empty((len(obs), model.n_states), dtype=np.float64)
    previous: np.ndarray | None = None
    for index in range(len(obs)):
        current = np.log(model.pi_ + 1e-300) + log_emit[index] if previous is None else (
            log_emit[index] + model._logsumexp(previous[:, None] + log_trans, axis=0)
        )
        current -= model._logsumexp(current, axis=0)
        state_prob[index] = np.exp(current)
        previous = current
    probabilities = regime_builder._class_proba(state_prob, payload["state_class_matrix"])
    ordered = np.sort(probabilities, axis=1)
    out = pd.DataFrame({"timestamp": work["timestamp"].reset_index(drop=True)})
    for index, name in enumerate(CLASSES):
        out[f"{name}_prob"] = probabilities[:, index]
    out["confidence"] = ordered[:, -1]
    out["margin"] = ordered[:, -1] - ordered[:, -2]
    out["entropy"] = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None)), axis=1) / np.log(len(CLASSES))
    out["regime"] = np.asarray(CLASSES, dtype=object)[np.argmax(probabilities, axis=1)]
    return out


def _shade(ax: plt.Axes, frame: pd.DataFrame) -> None:
    values = frame["regime"].to_numpy()
    times = frame["timestamp"].to_numpy()
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    stops = np.r_[starts[1:], len(frame)]
    for start, stop in zip(starts, stops, strict=True):
        ax.axvspan(pd.Timestamp(times[start]), pd.Timestamp(times[stop - 1]) + pd.Timedelta(minutes=5),
                   color=COLORS[str(values[start])], alpha=0.14, lw=0)


def _plot(frames: dict[str, pd.DataFrame]) -> None:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(6, 1, figsize=(20, 16), dpi=150, sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1.2, 0.72, 1.25, 1.2, 0.72], "hspace": 0.10})
    fig.patch.set_facecolor("#0b1018")
    for ax in axes:
        ax.set_facecolor("#101824")
        ax.grid(True, color="#718096", alpha=0.13, lw=0.6)
        ax.tick_params(colors="#b8c4d4", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#263449")
    for offset, asset in ((0, "BTC"), (3, "ETH")):
        frame = frames[asset]
        price, probability, diagnostics = axes[offset: offset + 3]
        _shade(price, frame)
        price.plot(frame["timestamp"], frame["close"], color="#e7edf6", lw=0.9)
        price.set_ylabel(f"{asset} close\n(USDT)")
        price.set_title(f"{asset} active Regime3 HMM — March 2026 OOS, causal 5-minute filtering",
                        loc="left", fontsize=14, fontweight="bold", pad=9)
        for name in CLASSES:
            probability.plot(frame["timestamp"], frame[f"{name}_prob"], color=COLORS[name], lw=0.48, alpha=0.35)
            probability.plot(frame["timestamp"], frame[f"{name}_prob"].rolling(12, min_periods=1).mean(),
                             color=COLORS[name], lw=1.05, label=f"{name} (1h trailing)")
        probability.set_ylim(0, 1)
        probability.set_ylabel("Class probability")
        probability.legend(loc="upper left", ncol=3, frameon=False, fontsize=8.5)
        for name, color in (("confidence", "#76a9fa"), ("margin", "#c084fc"), ("entropy", "#94a3b8")):
            diagnostics.plot(frame["timestamp"], frame[name].rolling(12, min_periods=1).mean(), color=color, lw=0.95, label=f"{name} (1h trailing)")
        diagnostics.set_ylim(0, 1)
        diagnostics.set_ylabel("HMM diagnostic")
        diagnostics.legend(loc="upper left", ncol=3, frameon=False, fontsize=8.5)
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("2026 UTC timestamp")
    fig.text(0.99, 0.008, "Price background: bull (green), bear (red), chop (yellow) · no saved regime sidecar or ledger used as input",
             ha="right", va="bottom", color="#8190a5", fontsize=8.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.045)
    fig.savefig(CHART, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plotted: dict[str, pd.DataFrame] = {}
    assets: dict[str, dict] = {}
    for asset, cfg in ASSETS.items():
        source = regime_builder._read(cfg["features"])
        fresh = _fresh_forward(joblib.load(cfg["model"]), source)
        joined = source[["timestamp", "close"]].merge(fresh, on="timestamp", validate="one_to_one")
        part = joined[(joined["timestamp"] >= START) & (joined["timestamp"] < END_EXCLUSIVE)].copy()
        if len(part) != 31 * 24 * 12:
            raise RuntimeError(f"{asset}: expected 8,928 March 5m bars, found {len(part):,}")
        plotted[asset] = part
        assets[asset] = {
            "feature_source": str(cfg["features"]), "model_path": str(cfg["model"]),
            "source_rows_processed_sequentially": int(len(source)), "plotted_rows": int(len(part)),
            "plot_range": [str(part["timestamp"].iloc[0]), str(part["timestamp"].iloc[-1])],
            "dominant_bar_share": {name: float((part["regime"] == name).mean()) for name in CLASSES},
            "mean_probability": {name: float(part[f"{name}_prob"].mean()) for name in CLASSES},
        }
    output = pd.concat([frame.assign(asset=asset) for asset, frame in plotted.items()], ignore_index=True)
    output.to_csv(FEATURES_CSV, index=False)
    _plot(plotted)
    report = {"oos_window": [str(START), str(END_EXCLUSIVE)], "bar_interval": "5m", "fresh_forward_bar_by_bar": True,
              "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False,
              "future_rows_used_for_entry": False, "saved_regime_sidecar_used_as_input": False,
              "assets": assets, "outputs": {"chart": str(CHART), "features_csv": str(FEATURES_CSV)}}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
