#!/usr/bin/env python3
"""Consolidate the corrected 2025-only 5m-to-1h HMM replacement experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516/regime3_1h_parent_replacement_report_20260728"
REPORT = OUT_DIR / "report.json"
SUMMARY = OUT_DIR / "parent_comparison.csv"
CHART = OUT_DIR / "parent_comparison.png"
MATERIALIZATION_AUDIT = ROOT / "tmp/causal_regen_20260516/regime3_1h_as_5m_contract_20260728/materialization_audit.json"
PATHS = {
    "h48qual_control_5m": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_matchedrow_5m_control_20260728_h48qual/report.json",
    "h48qual_replace_1h": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime1h_replace_2025only_20260728_h48qual/report.json",
    "zig075_control_5m": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_matchedrow_5m_control_20260728_zig075/report.json",
    "zig075_replace_1h": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime1h_replace_2025only_20260728_zig075/report.json",
}


def _metrics(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    result = next(iter(report["results"].values()))
    return {
        "input_contract": report["input_contract"],
        "validation": result["validation"],
        "oos": result["oos"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = {name: _metrics(path) for name, path in PATHS.items()}
    rows = []
    for component in ("h48qual", "zig075"):
        for contract in ("control_5m", "replace_1h"):
            name = f"{component}_{contract}"
            one = loaded[name]
            rows.append({
                "component": component,
                "contract": contract,
                "validation_pnl": one["validation"]["pnl"],
                "validation_mdd": one["validation"]["mdd"],
                "validation_trades": one["validation"]["trades"],
                "validation_wr": one["validation"]["wr"],
                "oos_pnl": one["oos"]["pnl"],
                "oos_mdd": one["oos"]["mdd"],
                "oos_trades": one["oos"]["trades"],
                "oos_wr": one["oos"]["wr"],
            })
    table = pd.DataFrame(rows)
    table.to_csv(SUMMARY, index=False)

    deltas = {}
    for component in ("h48qual", "zig075"):
        control = table[(table.component == component) & (table.contract == "control_5m")].iloc[0]
        candidate = table[(table.component == component) & (table.contract == "replace_1h")].iloc[0]
        deltas[component] = {
            key: float(candidate[key] - control[key])
            for key in ("validation_pnl", "validation_mdd", "validation_trades", "oos_pnl", "oos_mdd", "oos_trades")
        }

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor("#0b1018")
    colors = {"control_5m": "#94a3b8", "replace_1h": "#20c997"}
    for ax, metric, title in zip(axes, ("validation_pnl", "oos_pnl"), ("Validation PnL", "2026 OOS PnL"), strict=True):
        pivot = table.pivot(index="component", columns="contract", values=metric).reindex(["h48qual", "zig075"])
        pivot[["control_5m", "replace_1h"]].plot.bar(
            ax=ax,
            color=[colors["control_5m"], colors["replace_1h"]],
            width=0.72,
        )
        ax.axhline(0.0, color="#e7edf6", lw=0.8, alpha=0.7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("PnL (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis="y", alpha=0.15)
        ax.legend(["existing 5m HMM", "replacement 1h HMM"], frameon=False, fontsize=8)
    fig.suptitle("Omega4 Parent — 5m HMM Features Replaced by 1h HMM Features", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    materialization = json.loads(MATERIALIZATION_AUDIT.read_text(encoding="utf-8"))
    report = {
        "experiment_id": "omega4_parent_regime3_1h_feature_replacement_20260728",
        "status": "REJECTED_RESEARCH_ONLY",
        "contract": {
            "hmm_fit": "2024 completed 1h bars only",
            "parent_train": "2025-01-01 through 2025-09-30 only",
            "parent_validation": "2025-10-01 through 2025-12-31",
            "parent_oos": "2026 held out from parent selection",
            "replacement": "exactly six existing current-regime feature values; base column names/order remain pinned",
            "matched_row_control": "both contracts use the same 78,443 parent-train rows after identical exclusions",
            "base_feature_count": 102,
            "live_wiring_changed": False,
        },
        "materialization_audit": materialization,
        "parent_results": rows,
        "replacement_delta_vs_5m_control": deltas,
        "risk_sidecar": {
            "component_attempted": "zig075 only; h48qual rejected at parent gate",
            "precomputed_parent_prediction_tag": "q075",
            "result": "FAIL_FAST_NO_ELIGIBLE_RISK_MAPPING",
            "reason": "no validation mapping satisfied required average notional range 0.45..0.95",
            "constraints_relaxed": False,
        },
        "verdict": "Do not replace the existing 5m HMM features. h48qual worsened; zig075 did not clear validation and failed the risk-sidecar exposure gate.",
        "invalidated_run": {
            "path": str(ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime1h_replace_20260728_h48qual"),
            "reason": "superseded run incorrectly trained parent on HMM-fit year 2024; marked INVALID_DO_NOT_USE",
        },
        "outputs": {"summary_csv": str(SUMMARY), "chart": str(CHART)},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "deltas": deltas, "report": str(REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
