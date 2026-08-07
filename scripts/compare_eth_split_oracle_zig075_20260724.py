#!/usr/bin/env python3
"""Compare Oracle and Zig075 students under the identical new split contract."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_split_oracle_vs_zig075_20260724"
RUNS = {
    "Oracle DP student": ROOT / "tmp/causal_regen_20260516/eth_split_oracle_3head_noleak_20260724",
    "Zig075 student": ROOT / "tmp/causal_regen_20260516/eth_split_zig075_3head_noleak_20260724",
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = {}
    reports = {}
    for name, run_dir in RUNS.items():
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        chart_report = json.loads((run_dir / "oos_chart_report.json").read_text(encoding="utf-8"))
        reports[name] = report
        curve = pd.read_csv(run_dir / "oos_fresh_forward_equity.csv", parse_dates=["timestamp"])
        curves[name] = curve
        val = report["selection"]["validation_metrics"]
        oos = report["oos"]
        rows.append(
            {
                "model": name,
                "quality_threshold": report["selection"]["quality_threshold"],
                "notional_scale": report["selection"]["notional_scale"],
                "validation_pnl": val["pnl"],
                "validation_mdd": val["mdd"],
                "validation_trades": val["trades"],
                "oos_pnl": oos["pnl"],
                "oos_mdd": oos["mdd"],
                "oos_trades": oos["trades"],
                "oos_win_rate": oos["wr"],
                "replay_exact_match": chart_report["metrics"] == {
                    "pnl": oos["pnl"], "mdd": oos["mdd"], "trades": oos["trades"], "wr": oos["wr"]
                },
            }
        )
    comparison = pd.DataFrame(rows).sort_values("oos_pnl", ascending=False).reset_index(drop=True)
    comparison.to_csv(OUT_DIR / "comparison.csv", index=False)

    fig, (axis, drawdown_axis) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    colors = {"Oracle DP student": "#7c3aed", "Zig075 student": "#0f766e"}
    for name, curve in curves.items():
        axis.plot(curve["timestamp"], curve["equity"], label=name, color=colors[name], linewidth=1.2)
        drawdown_axis.plot(curve["timestamp"], curve["drawdown"] * 100.0, label=name, color=colors[name], linewidth=1.0)
    axis.axhline(1.0, color="#64748b", linestyle="--", linewidth=0.8)
    axis.set_title("Identical-contract ETH OOS fresh-forward comparison")
    axis.set_ylabel("Equity")
    axis.legend()
    axis.grid(alpha=0.15)
    drawdown_axis.set_ylabel("Drawdown %")
    drawdown_axis.set_xlabel("UTC")
    drawdown_axis.grid(alpha=0.15)
    fig.tight_layout()
    chart_path = OUT_DIR / "oracle_vs_zig075_oos_equity.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    winner = comparison.iloc[0]
    report = {
        "comparison_contract": {
            "train": "2024-01-01 <= timestamp < 2026-01-01",
            "validation": "2026-01-01 <= timestamp < 2026-04-01",
            "oos_period": "2026-04-01 <= timestamp <= 2026-07-20 00:00",
            "architecture": "same 3-head TabM, 3 HMM-routed experts, 158 causal features",
            "training": "2 epochs, Train labels only, 30000 Train-only Exit-head samples",
            "selection": "quality threshold and notional scale selected on Validation PnL only",
            "execution": "same maker-limit/fallback engine, fees, slippage, leverage, notional grid",
            "oos_evaluation": "one frozen bar-by-bar fresh-forward pass per model",
        },
        "results": rows,
        "winner_by_oos_pnl": str(winner["model"]),
        "winner_oos_pnl": float(winner["oos_pnl"]),
        "both_promotion_eligible": False,
        "reason": "both candidates have negative OOS PnL and excessive drawdown",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "artifacts": {"comparison": str(OUT_DIR / "comparison.csv"), "chart": str(chart_path)},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
