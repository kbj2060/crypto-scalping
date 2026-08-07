#!/usr/bin/env python3
"""Package the strongest historical non-DP BTC label family with charts and provenance."""
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
SOURCE_DIR = ROOT / "tmp/causal_regen_20260516/trend_scanning_action_labels_smoothed_20260620"
SCREEN = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/parent_candidate_summary_by_validation.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_best_non_dp_label_20260715"
ACTION_NAME = {0: "CASH", 1: "LONG", 2: "SHORT"}


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
    timestamp = frame["timestamp"]
    close = frame["close"].to_numpy(dtype=np.float64)
    action = frame["action_id"].to_numpy(dtype=np.int8)
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True, height_ratios=[3, 1])
    axes[0].plot(timestamp, close, color="black", linewidth=0.45)
    for value, color, name in ((1, "#159947", "LONG"), (2, "#d64545", "SHORT")):
        mask = action == value
        axes[0].scatter(timestamp[mask], close[mask], s=1.2, color=color, label=name, alpha=0.75)
    axes[0].set_title(f"BTC smoothed trend-scanning training labels — {year}")
    axes[0].set_ylabel("BTCUSDT close")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.15)
    axes[1].step(timestamp, action, where="post", linewidth=0.5, color="#315c9b")
    axes[1].set_yticks([0, 1, 2], ["CASH", "LONG", "SHORT"])
    axes[1].set_ylabel("label")
    axes[1].grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--screen", type=Path, default=SCREEN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    screen_path = args.screen.resolve()
    out_dir = args.out_dir.resolve()
    (out_dir / "label_charts").mkdir(parents=True, exist_ok=True)

    screen = pd.read_csv(screen_path)
    selected = screen.loc[screen["name"].eq("trend_scanning_action_labels_smoothed_20260620")]
    if len(selected) != 1:
        raise RuntimeError(f"expected one selected screen row, got {len(selected)}")
    row = selected.iloc[0]
    robust_min = min(float(row["best_val_pnl"]), float(row["best_val_oos_pnl"]))
    all_robust = np.minimum(screen["best_val_pnl"], screen["best_val_oos_pnl"])
    if robust_min < float(all_robust.max()):
        raise RuntimeError("smoothed trend-scanning is no longer the robust label-only PnL winner")

    artifacts = {}
    summaries = {}
    for year in (2024, 2025, 2026):
        source = source_dir / f"zigzag_action_labels_{year}.csv"
        raw = pd.read_csv(
            source,
            usecols=["timestamp", "close", "ts_t_value", "ts_beta", "zigzag_action"],
            parse_dates=["timestamp"],
        )
        if raw["timestamp"].duplicated().any() or not raw["timestamp"].is_monotonic_increasing:
            raise RuntimeError(f"{year}: timestamp contract failed")
        action = pd.to_numeric(raw["zigzag_action"], errors="raise").to_numpy(dtype=np.int8)
        if not set(np.unique(action)).issubset({0, 1, 2}):
            raise RuntimeError(f"{year}: invalid actions")
        labels = pd.DataFrame({
            "timestamp": raw["timestamp"],
            "action_id": action,
            "action": [ACTION_NAME[int(value)] for value in action],
            "trend_t_value": pd.to_numeric(raw["ts_t_value"], errors="raise"),
            "trend_beta": pd.to_numeric(raw["ts_beta"], errors="raise"),
        })
        artifact = out_dir / f"btc_smoothed_trendscan_labels_{year}.parquet"
        labels.to_parquet(artifact, index=False)
        chart = out_dir / "label_charts" / f"btc_smoothed_trendscan_labels_{year}.png"
        plot_labels(chart, pd.DataFrame({"timestamp": raw["timestamp"], "close": raw["close"], "action_id": action}), year)
        artifacts[str(year)] = {
            "source": str(source), "source_sha256": sha256(source),
            "label_artifact": str(artifact), "label_sha256": sha256(artifact), "chart": str(chart),
        }
        counts = pd.Series(action).value_counts().sort_index()
        summaries[str(year)] = {
            "rows": len(labels),
            "counts": {ACTION_NAME[int(key)]: int(value) for key, value in counts.items()},
        }

    manifest = {
        "status": "best_observed_non_dp_label_pack_research_only",
        "label_family": "smoothed_trend_scanning",
        "dp_used": False,
        "training_label_is_forward_looking_offline_only": True,
        "live_feature_or_signal": False,
        "parameters": {
            "enter_t": 9.0, "exit_t": 5.0, "flip_t": 10.0,
            "min_active_len": 12, "max_same_side_cash_gap": 6, "transition_buffer": 1,
        },
        "historical_parent_screen": {
            "selection": "maximum min(validation_pnl, oos_pnl) among 27 label-only candidates",
            "quality_threshold_selected_on_validation": float(row["best_val_q"]),
            "validation_pnl_pct": float(row["best_val_pnl"]),
            "validation_mdd_pct": float(row["best_val_mdd"]),
            "validation_trades": int(row["best_val_trades"]),
            "oos_pnl_pct": float(row["best_val_oos_pnl"]),
            "oos_mdd_pct": float(row["best_val_oos_mdd"]),
            "oos_trades": int(row["best_val_oos_trades"]),
            "robust_min_pnl_pct": robust_min,
            "promotion_evidence": False,
            "note": "The historical OOS was previously observed and is diagnostic-only under the current fresh-forward rule.",
        },
        "artifacts": artifacts,
        "summaries": summaries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "historical_parent_screen": manifest["historical_parent_screen"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
