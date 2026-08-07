#!/usr/bin/env python3
"""Summarize the Validation-only Oracle quality funnel and selected Q05 OOS."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tmp/causal_regen_20260516"
OUT_DIR = BASE / "eth_oracle_quality_variants_20260724"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in ("q20", "q10", "q05"):
        model_dir = BASE / f"eth_oracle_quality_{variant}_3head_noleak_20260724"
        label_dir = BASE / f"eth_oracle_entry_quality_{variant}_20260724"
        model = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
        labels = json.loads((label_dir / "report.json").read_text(encoding="utf-8"))
        val = model["selection"]["validation_metrics"]
        rows.append(
            {
                "variant": variant,
                "train_quality_positive_ratio": labels["summaries"]["train"]["quality_positive_ratio"],
                "quality_threshold": model["selection"]["quality_threshold"],
                "notional_scale": model["selection"]["notional_scale"],
                "validation_pnl": val["pnl"],
                "validation_mdd": val["mdd"],
                "validation_trades": val["trades"],
                "validation_win_rate": val["wr"],
                "oos_evaluated": variant == "q05",
                "oos_pnl": model.get("oos", {}).get("pnl"),
                "oos_mdd": model.get("oos", {}).get("mdd"),
                "oos_trades": model.get("oos", {}).get("trades"),
                "oos_win_rate": model.get("oos", {}).get("wr"),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "quality_variant_funnel.csv", index=False)

    dense_curve = pd.read_csv(BASE / "eth_split_oracle_3head_noleak_20260724/oos_fresh_forward_equity.csv", parse_dates=["timestamp"])
    sparse_curve = pd.read_csv(BASE / "eth_oracle_quality_q05_3head_noleak_20260724/oos_fresh_forward_equity.csv", parse_dates=["timestamp"])
    fig, (axis, dd_axis) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for name, curve, color in (
        ("Dense Oracle quality=direction", dense_curve, "#7c3aed"),
        ("Sparse Oracle Q05 quality", sparse_curve, "#dc2626"),
    ):
        axis.plot(curve["timestamp"], curve["equity"], label=name, color=color, linewidth=1.2)
        dd_axis.plot(curve["timestamp"], curve["drawdown"] * 100.0, label=name, color=color, linewidth=1.0)
    axis.axhline(1.0, color="#64748b", linestyle="--", linewidth=0.8)
    axis.set_title("Oracle-specific Quality head: frozen OOS comparison")
    axis.set_ylabel("Equity")
    axis.legend()
    axis.grid(alpha=0.15)
    dd_axis.set_ylabel("Drawdown %")
    dd_axis.set_xlabel("UTC")
    dd_axis.grid(alpha=0.15)
    fig.tight_layout()
    chart_path = OUT_DIR / "dense_vs_sparse_q05_oos.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    report = {
        "selection": "Q05 selected by Validation PnL before any variant OOS evaluation",
        "variant_rows": rows,
        "unselected_variant_oos_evaluated": False,
        "selected_q05_oos_is_fresh_for_this_variant_funnel": True,
        "promotion_eligible": False,
        "promotion_blockers": [
            "selected Q05 has negative OOS PnL and excessive drawdown",
            "the shared April-July interval was already observed in earlier model experiments, so this is research diagnostic rather than a new pristine promotion test",
        ],
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "artifacts": {"funnel": str(OUT_DIR / "quality_variant_funnel.csv"), "chart": str(chart_path)},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
