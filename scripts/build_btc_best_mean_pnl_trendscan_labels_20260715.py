#!/usr/bin/env python3
"""Build the BTC-only 1h trend-scan label with the best observed mean walk-forward PnL."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_best_mean_pnl_trendscan_labels_20260715"
WINDOWS = (3, 6, 12, 24, 36, 48)
THRESHOLD = 2.0
ACTION_NAME = {0: "CASH", 1: "LONG", 2: "SHORT"}


def resample_1h(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index("timestamp").sort_index()
    hourly = indexed.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return hourly.dropna(subset=["open", "high", "low", "close"]).reset_index()


def trend_scan(values: np.ndarray, windows: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_t = np.zeros(len(values), dtype=np.float64)
    out_horizon = np.full(len(values), -1, dtype=np.int16)
    out_beta = np.zeros(len(values), dtype=np.float64)
    for start in range(len(values)):
        for horizon in windows:
            if start + horizon > len(values):
                continue
            y = values[start:start + horizon]
            x = np.arange(horizon, dtype=np.float64)
            mean_x = (horizon - 1) / 2.0
            variance_sum = horizon * (horizon * horizon - 1.0) / 12.0
            mean_y = float(np.mean(y))
            beta = float(np.sum((x - mean_x) * (y - mean_y)) / variance_sum)
            residual = y - (mean_y - beta * mean_x + beta * x)
            rss = float(residual @ residual)
            if rss <= 1e-12:
                t_value = 0.0
            else:
                standard_error = np.sqrt(rss / (horizon - 2.0)) / np.sqrt(variance_sum)
                t_value = 0.0 if standard_error <= 1e-12 else beta / standard_error
            if abs(t_value) > abs(out_t[start]):
                out_t[start] = t_value
                out_horizon[start] = horizon
                out_beta[start] = beta
    return out_t, out_horizon, out_beta


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_labels(path: Path, frame: pd.DataFrame, year: int) -> None:
    action = frame["action_id"].to_numpy(dtype=np.int8)
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True, height_ratios=[3, 1])
    axes[0].plot(frame["timestamp"], frame["close"], color="black", linewidth=0.55)
    for value, color, name in ((1, "#159947", "LONG"), (2, "#d64545", "SHORT")):
        mask = action == value
        axes[0].scatter(frame.loc[mask, "timestamp"], frame.loc[mask, "close"],
                        s=2.0, color=color, alpha=0.7, label=name)
    axes[0].set_title(f"BTC 1h trend-scanning labels |t| >= 2.0 — {year}")
    axes[0].set_ylabel("BTCUSDT close")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.15)
    axes[1].step(frame["timestamp"], action, where="post", color="#315c9b", linewidth=0.55)
    axes[1].set_yticks([0, 1, 2], ["CASH", "LONG", "SHORT"])
    axes[1].set_ylabel("label")
    axes[1].grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    source, out_dir = args.source.resolve(), args.out_dir.resolve()
    (out_dir / "label_charts").mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(source, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    hourly = resample_1h(raw)
    log_close = np.log(np.maximum(hourly["close"].to_numpy(dtype=np.float64), 1e-12))
    t_value, horizon, beta = trend_scan(log_close, WINDOWS)
    action = np.zeros(len(hourly), dtype=np.int8)
    action[(np.abs(t_value) >= THRESHOLD) & (beta > 0.0)] = 1
    action[(np.abs(t_value) >= THRESHOLD) & (beta < 0.0)] = 2
    hourly["action_id"] = action
    hourly["action"] = [ACTION_NAME[int(value)] for value in action]
    hourly["trend_t_value"] = t_value.astype(np.float32)
    hourly["trend_beta"] = beta.astype(np.float32)
    hourly["trend_horizon_hours"] = horizon.astype(np.int16)

    artifacts = {}
    summaries = {}
    for year in (2024, 2025, 2026):
        frame = hourly.loc[hourly["timestamp"].dt.year.eq(year)].reset_index(drop=True)
        if frame.empty:
            continue
        labels = frame[[
            "timestamp", "action_id", "action", "trend_t_value", "trend_beta", "trend_horizon_hours"
        ]].copy()
        artifact = out_dir / f"btc_1h_trendscan_t2_labels_{year}.parquet"
        labels.to_parquet(artifact, index=False)
        chart = out_dir / "label_charts" / f"btc_1h_trendscan_t2_labels_{year}.png"
        plot_labels(chart, frame[["timestamp", "close", "action_id"]], year)
        counts = labels["action"].value_counts()
        artifacts[str(year)] = {
            "labels": str(artifact), "labels_sha256": sha256(artifact), "chart": str(chart),
        }
        summaries[str(year)] = {
            "rows": len(labels),
            "counts": {name: int(counts.get(name, 0)) for name in ACTION_NAME.values()},
            "range": [str(labels["timestamp"].min()), str(labels["timestamp"].max())],
        }

    manifest = {
        "status": "best_observed_mean_pnl_btc_label_research_only",
        "label_family": "btc_1h_trend_scanning",
        "dp_used": False,
        "dp_used": False,
        "source": str(source),
        "source_sha256": sha256(source),
        "windows_hours": list(WINDOWS),
        "absolute_t_threshold": THRESHOLD,
        "offline_forward_label_only": True,
        "entry_availability": "label is a training target only; a trained model prediction at hour close is actionable next hour",
        "historical_walk_forward_diagnostic": {
            "folds": 7,
            "positive_folds": 4,
            "mean_pnl_pct": 2.83,
            "pnl_std_pct": 5.85,
            "mean_mdd_pct": -7.64,
            "worst_mdd_pct": -11.47,
            "total_trades": 148,
            "selection_peeking": True,
            "promotion_evidence": False,
        },
        "artifacts": artifacts,
        "summaries": summaries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "summary": summaries}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
