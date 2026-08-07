#!/usr/bin/env python3
"""Compare the active ETH ZigZag/H48 components with the hindsight oracle labels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp/causal_regen_20260516"
OUT = TMP / "eth_zigzag_h48_oracle_comparison_20260724"

COMPONENTS = {
    "H48 q0.50": TMP
    / "omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630",
    "ZigZag q0.75": TMP
    / "omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_zig075_q075_precomputed_20260630",
}
ORACLE_LEDGER = (
    TMP
    / "eth_full_oracle_strategy_labels_v1_20260724"
    / "oracle_selected_trades.csv"
)
ROUTER_AUDIT = (
    TMP / "omega4_6_1_phase1_robustness_20260707" / "result.json"
)

WINDOWS = {
    "validation": (pd.Timestamp("2025-10-01"), pd.Timestamp("2026-01-01")),
    "oos": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-01")),
}


def load_component() -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict]]:
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    reports: dict[str, dict] = {}
    for name, directory in COMPONENTS.items():
        reports[name] = json.loads((directory / "report.json").read_text())
        frames[name] = {}
        for split in WINDOWS:
            path = directory / f"{split}_selected_risk_replayed_trade_ledger.csv"
            frame = pd.read_csv(
                path, parse_dates=["entry_timestamp", "exit_timestamp"]
            ).sort_values("exit_timestamp")
            frames[name][split] = frame
    return frames, reports


def equity_curve(frame: pd.DataFrame) -> pd.DataFrame:
    values = np.cumprod(1.0 + frame["trade_return"].to_numpy(float))
    return pd.DataFrame({"timestamp": frame["exit_timestamp"], "equity": values})


def oracle_window(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    start, end = WINDOWS[split]
    return frame.loc[
        (frame["decision_timestamp"] >= start)
        & (frame["event_end_timestamp"] < end)
    ].copy()


def component_metrics(report: dict, split: str) -> dict:
    metrics = report["selected"][split]
    replay = report["selected"]["selected_full_replay"][split]
    return {
        "pnl_pct": float(metrics["pnl"]),
        "mdd_pct": float(metrics["mdd"]),
        "trades": int(metrics["trades"]),
        "win_rate": float(metrics["wr"]),
        "avg_notional": float(metrics["avg_notional"]),
        "avg_leverage": float(metrics["avg_leverage"]),
        "replay_pnl_pct": float(replay["pnl"]),
        "replay_mdd_pct": float(replay["mdd"]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames, reports = load_component()
    oracle = pd.read_csv(
        ORACLE_LEDGER,
        parse_dates=["decision_timestamp", "event_end_timestamp"],
    )
    router = json.loads(ROUTER_AUDIT.read_text())["cost_stress"]

    summary: dict[str, object] = {
        "comparison_contract": {
            "component_validation_window": ["2025-10-01", "2025-12-31"],
            "component_oos_window": ["2026-01-01", "2026-02-28"],
            "component_return_basis": "variable account notional and leverage",
            "oracle_return_basis": "one unit notional, no leverage",
            "oracle_future_rows_used_for_label": True,
            "oracle_is_model_validation": False,
            "direct_return_ranking_valid": False,
        },
        "components": {},
        "router": {
            "validation": router["val"]["cost1x"],
            "extended_oos_2026_h1": router["oos"]["cost1x"],
        },
        "oracle": {},
        "caveats": [
            "H48 changes the quality head target; its direction head still uses ZigZag labels.",
            "The router numbers are historical diagnostic replays, not a new fresh-forward promotion test.",
            "The hindsight oracle uses future paths to create targets and cannot be interpreted as expected live return.",
        ],
    }

    for name, report in reports.items():
        summary["components"][name] = {
            split: component_metrics(report, split) for split in WINDOWS
        }

    for split in WINDOWS:
        selected = oracle_window(oracle, split)
        log_return = float(np.log1p(selected["net_return_per_notional"]).sum())
        summary["oracle"][split] = {
            "selected_trades": int(len(selected)),
            "long_trades": int((selected["side"] == 1).sum()),
            "short_trades": int((selected["side"] == -1).sum()),
            "mean_return_per_notional": float(
                selected["net_return_per_notional"].mean()
            ),
            "median_return_per_notional": float(
                selected["net_return_per_notional"].median()
            ),
            "log_return": log_return,
            "equity_multiple": float(np.exp(log_return)),
        }

    colors = {"H48 q0.50": "#3478c7", "ZigZag q0.75": "#d66b2c"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    for ax, split, title in [
        (axes[0, 0], "validation", "Validation: 2025-10 to 2025-12"),
        (axes[0, 1], "oos", "OOS readout: 2026-01 to 2026-02"),
    ]:
        for name in COMPONENTS:
            curve = equity_curve(frames[name][split])
            ax.step(
                curve["timestamp"],
                curve["equity"],
                where="post",
                label=f"{name}  {curve.equity.iloc[-1]:.2f}x",
                color=colors[name],
                linewidth=2.2,
            )
        ax.axhline(1.0, color="#777777", linewidth=1, alpha=0.55)
        ax.set_title(title)
        ax.set_ylabel("Account equity multiple")
        ax.grid(alpha=0.18)
        ax.legend(loc="best")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax = axes[1, 0]
    labels = list(COMPONENTS) + ["Current router"]
    val_pnl = [
        reports[name]["selected"]["validation"]["pnl"] for name in COMPONENTS
    ] + [router["val"]["cost1x"]["pnl"]]
    oos_pnl = [reports[name]["selected"]["oos"]["pnl"] for name in COMPONENTS] + [
        router["oos"]["cost1x"]["pnl"]
    ]
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width / 2, val_pnl, width, label="Validation", color="#4c78a8")
    bars2 = ax.bar(x + width / 2, oos_pnl, width, label="OOS*", color="#f28e2b")
    ax.bar_label(bars1, fmt="%.1f%%", padding=3)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Historical account return (%)")
    ax.set_title("Current model results (*router OOS is Jan-Jun)")
    ax.grid(axis="y", alpha=0.18)
    ax.legend()

    ax = axes[1, 1]
    h48_n = reports["H48 q0.50"]["selected"]["oos"]["trades"]
    zig_n = reports["ZigZag q0.75"]["selected"]["oos"]["trades"]
    router_n = router["oos"]["cost1x"]["trades"]
    oracle_n = summary["oracle"]["oos"]["selected_trades"]
    density_labels = ["H48\nJan-Feb", "ZigZag\nJan-Feb", "Router\nJan-Jun", "Oracle\nJan-Feb"]
    density_values = [h48_n, zig_n, router_n, oracle_n]
    bars = ax.bar(density_labels, density_values, color=["#3478c7", "#d66b2c", "#5b8c5a", "#8b5fbf"])
    ax.set_yscale("log")
    ax.set_ylabel("Selected trades (log scale)")
    ax.set_title("Signal density: deployable models vs hindsight labels")
    ax.grid(axis="y", alpha=0.18, which="both")
    for bar, value in zip(bars, density_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:,}", ha="center", va="bottom")

    fig.suptitle("ETH ZigZag / H48 validation versus full-history oracle labels", fontsize=16)
    chart = OUT / "zigzag_h48_oracle_comparison.png"
    fig.savefig(chart, dpi=170)
    plt.close(fig)

    summary["artifacts"] = {"chart": str(chart), "report": str(OUT / "report.json")}
    (OUT / "report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
