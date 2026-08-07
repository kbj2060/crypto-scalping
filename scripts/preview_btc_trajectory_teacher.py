#!/usr/bin/env python3
"""Render a latest-complete-week diagnostic chart for BTC oracle teacher labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.btc_trajectory_teacher import TeacherConfig, build_teacher_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/splits/year_oos/btc_features_2026.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp/btc_trajectory_teacher_preview")
    args = parser.parse_args()
    frame = pd.read_csv(args.data, usecols=["timestamp", "open", "close"], low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    config = TeacherConfig()
    # Limit DP work to the requested diagnostic week plus its future label horizon.
    latest_complete_decision = frame["timestamp"].max() - pd.Timedelta(minutes=5 * config.horizon_bars)
    start = latest_complete_decision - pd.Timedelta(days=7)
    frame = frame[(frame["timestamp"] > start) & (frame["timestamp"] <= latest_complete_decision + pd.Timedelta(minutes=5 * config.horizon_bars))]
    labels = build_teacher_path(frame, config)
    end = labels["decision_timestamp"].max()
    week = labels[(labels["decision_timestamp"] > start) & (labels["decision_timestamp"] <= end)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "btc_teacher_latest_complete_week.csv"
    png_path = args.output_dir / "btc_teacher_latest_complete_week.png"
    report_path = args.output_dir / "btc_teacher_latest_complete_week.json"
    week.to_csv(csv_path, index=False)

    fig, (price_ax, exposure_ax, equity_ax) = plt.subplots(3, 1, figsize=(16, 10), sharex=True, height_ratios=[2.2, 1, 1.2])
    price_ax.plot(week["execution_timestamp"], week["execution_close"], color="#d8dee9", linewidth=1.1, label="BTC close")
    events = week[week["turnover_margin_fraction"] > 1e-9]
    for direction, color, marker in [("long", "#2ecc71", "^"), ("short", "#ff5c5c", "v"), ("flat", "#f1c40f", "x")]:
        selected = events[events["direction_label"] == direction]
        price_ax.scatter(selected["execution_timestamp"], selected["execution_open"], color=color, marker=marker, s=30, label=direction)
    price_ax.set_title("BTC trajectory-teacher diagnostic — latest complete 7 days (oracle labels, not model PnL)")
    price_ax.set_ylabel("BTCUSDT")
    price_ax.legend(ncol=4, loc="upper left")
    exposure_ax.step(week["execution_timestamp"], week["hard_target_margin_fraction"], where="post", color="#4c566a", alpha=.45, label="hard path (PnL diagnostic)")
    exposure_ax.plot(week["execution_timestamp"], week["teacher_signed_margin_fraction"], color="#88c0d0", linewidth=1.2, label="soft-label expected margin")
    exposure_ax.axhline(0, color="#4c566a", linewidth=.8)
    exposure_ax.set_ylim(-.33, .33)
    exposure_ax.set_ylabel("target\nmargin")
    exposure_ax.legend(loc="upper left")
    equity_ax.plot(week["execution_timestamp"], week["equity"], color="#a3be8c")
    equity_ax.set_ylabel("teacher equity")
    equity_ax.set_xlabel("UTC")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    report = {
        "diagnostic_only": True, "teacher_uses_future_prices_for_labels": True,
        "period_start": str(week["decision_timestamp"].min()), "period_end": str(week["decision_timestamp"].max()),
        "bars": int(len(week)), "trade_events": int(len(events)), "teacher_pnl_pct": float((week["equity"].iloc[-1] / week["equity"].iloc[0] - 1.0) * 100.0),
        "config": {"leverage": config.leverage, "max_margin_fraction": config.max_margin_fraction, "horizon_bars": config.horizon_bars, "one_way_cost_rate": config.one_way_cost_rate, "soft_label_temperature": config.soft_label_temperature},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"chart": str(png_path), "labels": str(csv_path), "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
