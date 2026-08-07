#!/usr/bin/env python3
"""Red-team style audit for Omega 4.6 borrowed-upgrade diagnostics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_borrowed_version_upgrade_tests_20260630"
DEFAULT_VARIANT = "short_rsi_skip_ge_56p656189__short_bias_cap180__time_stop_120h"
BASELINE_VARIANT = "none__none__none"


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def row_for(df: pd.DataFrame, variant: str) -> dict[str, Any]:
    rows = df.loc[df["variant"].eq(variant)]
    if rows.empty:
        raise ValueError(f"variant not found: {variant}")
    return rows.iloc[0].to_dict()


def gate_checks(row: dict[str, Any], artifact_pass: bool) -> dict[str, bool]:
    return {
        "artifact_integrity_pass": bool(artifact_pass),
        "validation_mdd_lte_20_abs": abs(float(row["validation_mdd"])) <= 20.0 + 1.0e-9,
        "oos_mdd_lte_20_abs": abs(float(row["oos_mdd"])) <= 20.0 + 1.0e-9,
        "validation_leverage_lte_5": float(row["validation_max_leverage"]) <= 5.0 + 1.0e-9,
        "oos_leverage_lte_5": float(row["oos_max_leverage"]) <= 5.0 + 1.0e-9,
        "validation_no_overlap": int(row["validation_overlap_count"]) == 0,
        "oos_no_overlap": int(row["oos_overlap_count"]) == 0,
        "validation_accounting_consistent": float(row["validation_accounting_error_max_abs"]) <= 1.0e-10,
        "oos_accounting_consistent": float(row["oos_accounting_error_max_abs"]) <= 1.0e-10,
        "validation_notional_contract_consistent": float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10,
        "oos_notional_contract_consistent": float(row["oos_notional_contract_error_max_abs"]) <= 1.0e-10,
    }


def hard_live_checks(row: dict[str, Any], *, clean_oos_selection_claim_allowed: bool) -> dict[str, bool]:
    return {
        "validation_max_hold_lte_24h": float(row["validation_max_hold_hours"]) <= 24.0 + 1.0e-9,
        "oos_max_hold_lte_24h": float(row["oos_max_hold_hours"]) <= 24.0 + 1.0e-9,
        "validation_pnl_gte_100pct": float(row["validation_pnl"]) >= 100.0,
        "oos_pnl_gte_100pct": float(row["oos_pnl"]) >= 100.0,
        "runtime_native_replay_available": False,
        "clean_oos_selection_claim_allowed": bool(clean_oos_selection_claim_allowed),
    }


def metric_subset(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "validation_pnl",
        "validation_mdd",
        "validation_trades",
        "validation_wr",
        "validation_max_hold_hours",
        "validation_hold_over_24h_count",
        "validation_max_leverage",
        "validation_max_notional",
        "validation_monthly_min_pnl",
        "oos_pnl",
        "oos_mdd",
        "oos_trades",
        "oos_wr",
        "oos_max_hold_hours",
        "oos_hold_over_24h_count",
        "oos_max_leverage",
        "oos_max_notional",
    ]
    return {key: row[key] for key in keys if key in row}


def deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in (
        "validation_pnl",
        "validation_mdd",
        "validation_trades",
        "validation_max_hold_hours",
        "oos_pnl",
        "oos_mdd",
        "oos_trades",
        "oos_max_hold_hours",
    ):
        out[key] = float(candidate[key]) - float(baseline[key])
    return out


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    c = audit["candidate_metrics"]
    b = audit["baseline_metrics"]
    lines = [
        "# Omega 4.6 Borrowed Upgrade Candidate Red-Team Audit - 2026-06-30",
        "",
        "## Verdict",
        "",
        f"`{audit['verdict']}`",
        "",
        audit["verdict_detail"],
        "",
        "## Candidate",
        "",
        f"- Variant: `{audit['variant']}`",
        f"- Base model: `{audit['base_model_id']}`",
        f"- Report: `{audit['source_report']}`",
        f"- Ranking: `{audit['ranking']}`",
        f"- Artifact audit: `{audit['artifact_audit']}`",
        "",
        "## Metrics",
        "",
        "| Split | Baseline PnL | Candidate PnL | Baseline MDD | Candidate MDD | Trades | Max hold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Validation | `{b['validation_pnl']:+.2f}%` | `{c['validation_pnl']:+.2f}%` | `{b['validation_mdd']:.2f}%` | `{c['validation_mdd']:.2f}%` | `{int(c['validation_trades'])}` | `{c['validation_max_hold_hours']:.2f}h` |",
        f"| OOS readout | `{b['oos_pnl']:+.2f}%` | `{c['oos_pnl']:+.2f}%` | `{b['oos_mdd']:.2f}%` | `{c['oos_mdd']:.2f}%` | `{int(c['oos_trades'])}` | `{c['oos_max_hold_hours']:.2f}h` |",
        "",
        "## Passed Conditional Gates",
        "",
    ]
    for name, ok in audit["conditional_gate_checks"].items():
        if ok:
            lines.append(f"- `{name}`")
    lines.extend(["", "## Failed / Excluded Full-Live Gates", ""])
    for name, ok in audit["full_live_checks"].items():
        if not ok:
            lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "## Red-Team Notes",
            "",
            "- This is a ledger-level diagnostic, not a runtime-native promoted model.",
            "- The balanced candidate is not a clean-OOS promotion claim because final preference considered OOS readout versus the pure validation-score winner.",
            "- Full live/day-trading PASS remains blocked by max hold above 24h and OOS PnL below 100%.",
            "- Promotion would require a frozen runtime contract, native replay, and fresh holdout or walk-forward confirmation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    report_path = run_dir / "report.json"
    ranking_path = run_dir / "borrowed_upgrade_ranking.csv"
    artifact_audit_path = run_dir / "omega_artifact_integrity_audit_20260630.json"
    report = read_json(report_path)
    artifact_audit = read_json(artifact_audit_path)
    df = pd.read_csv(ranking_path)
    candidate = row_for(df, args.variant)
    baseline = row_for(df, BASELINE_VARIANT)
    script_winner = report.get("selected_variant", {}).get("variant")
    clean_selection = args.variant == script_winner

    conditional = gate_checks(candidate, artifact_pass=bool(artifact_audit["promotion_pass"]))
    live = hard_live_checks(candidate, clean_oos_selection_claim_allowed=clean_selection)
    conditional_pass = all(conditional.values())
    full_live_pass = conditional_pass and all(live.values())
    if full_live_pass:
        verdict = "FULL_LIVE_PASS"
    elif conditional_pass:
        verdict = "CONDITIONAL_DIAGNOSTIC_PASS_FULL_LIVE_FAIL_FRESH_HOLDOUT_REQUIRED"
    else:
        verdict = "FAIL"
    verdict_detail = (
        "The candidate passes artifact, accounting, overlap, MDD, leverage, and notional-contract checks, "
        "but it is not a full live/day-trading pass."
        if conditional_pass
        else "The candidate fails at least one non-excluded conditional gate."
    )
    audit = {
        "audit_id": "omega4_6_borrowed_upgrade_candidate_redteam_20260630",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "base_model_id": report["base_model_id"],
        "source_report": str(report_path),
        "ranking": str(ranking_path),
        "artifact_audit": str(artifact_audit_path),
        "artifact_integrity_promotion_pass": bool(artifact_audit["promotion_pass"]),
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "conditional_pass": bool(conditional_pass),
        "full_live_pass": bool(full_live_pass),
        "candidate_metrics": metric_subset(candidate),
        "baseline_metrics": metric_subset(baseline),
        "delta_vs_baseline": deltas(candidate, baseline),
        "conditional_gate_checks": conditional,
        "full_live_checks": live,
        "selection_caveat": {
            "diagnostic_selection_scope": report.get("selection_scope"),
            "script_validation_only_winner": script_winner,
            "audited_variant": args.variant,
            "clean_oos_selection_claim_allowed": bool(clean_selection),
            "clean_oos_promotion_claim_allowed": False,
            "reason": (
                "audited variant is the diagnostic validation-only winner, but promotion still needs runtime-native replay"
                if clean_selection
                else "audited balanced candidate was preferred after considering OOS readout; needs fresh holdout/walk-forward"
            ),
        },
        "required_before_promotion": [
            "Freeze this rule into a runtime contract.",
            "Implement native replay/runtime path instead of ledger-level diagnostic only.",
            "Run fresh holdout or walk-forward because OOS readout influenced candidate preference.",
            "Keep max-hold 24h and OOS 100% PnL gates explicit if claiming full live/day-trading PASS.",
        ],
    }
    out_json = run_dir / f"{args.variant}_redteam_audit_20260630.json"
    out_md = run_dir / f"{args.variant}_redteam_audit_20260630.md"
    write_json(out_json, audit)
    write_markdown(out_md, audit)
    print(json.dumps({"verdict": verdict, "conditional_pass": conditional_pass, "full_live_pass": full_live_pass, "json": str(out_json), "markdown": str(out_md)}, ensure_ascii=False, indent=2))
    return 0 if conditional_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
