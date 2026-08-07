#!/usr/bin/env python3
"""Red-team audit for the Omega 4.6.2 v5 roll24 daytrade overlay."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_V1_PATH = ROOT / "scripts/audit_omega4_6_2_loss_cluster_governor_redteam_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll24_daytrade_overlay_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_roll24_daytrade_overlay_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REPORT = OUT_DIR / "report.json"
EVAL_SCRIPT = ROOT / "scripts/eval_omega4_6_2_v5_roll24_daytrade_overlay_20260701.py"
PRIOR_RUNTIME_AUDIT = (
    ROOT / "docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json"
)
AUDIT_JSON = OUT_DIR / "redteam_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll24_daytrade_redteam_20260701.md"
TOL = 1.0e-8


def load_audit_v1() -> Any:
    spec = importlib.util.spec_from_file_location("omega462_loss_cluster_audit_v1_for_v5_roll24", AUDIT_V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {AUDIT_V1_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(payload: dict[str, Any]) -> None:
    reference = payload["reference_variant"]
    observed = payload["observed"]
    blockers = [c for c in payload["checks"] if not c["pass"] and c["severity"] == "live_blocker"]
    research_fails = [
        c for c in payload["checks"] if not c["pass"] and c["severity"] == "research_blocker"
    ]
    lines = [
        "# Omega 4.6.2 v5 Roll24 Daytrade Red-Team Audit - 2026-07-01",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Parent: `{payload['parent_model_id']}`",
        f"- Reference daytrade model: `{payload['reference_model_id']}`",
        f"- Verdict: `{payload['verdict']}`",
        f"- Daytrade research pass: `{payload['daytrade_research_pass']}`",
        f"- PnL upgrade vs reference: `{payload['pnl_upgrade_vs_reference']}`",
        f"- Full live pass: `{payload['full_live_pass']}`",
        "",
        "| Split | Reference PnL | v5 Roll24 PnL | Reference MDD | v5 Roll24 MDD | Reference Avg Hold | v5 Roll24 Avg Hold | Reference Max Hold | v5 Roll24 Max Hold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["validation", "oos"]:
        lines.append(
            "| "
            f"{split} | "
            f"{reference[f'{split}_pnl']:.4f}% | {observed[split]['pnl']:.4f}% | "
            f"{reference[f'{split}_mdd']:.4f}% | {observed[split]['mdd']:.4f}% | "
            f"{reference[f'{split}_avg_hold_hours']:.4f}h | {observed[split]['avg_hold_hours']:.4f}h | "
            f"{reference[f'{split}_max_hold_hours']:.4f}h | {observed[split]['max_hold_hours']:.4f}h |"
        )
    lines.extend(["", "## Blocking Items", ""])
    lines.extend([f"- `{c['name']}`: {c['details']}" for c in blockers] or ["- None for full live pass."])
    lines.extend(["", "## Research Failures", ""])
    lines.extend([f"- `{c['name']}`: {c['details']}" for c in research_fails] or ["- None."])
    lines.extend(
        [
            "",
            "## Contract Checks",
            "",
            f"- Validation accounting error max abs: `{observed['validation']['accounting_error_max_abs']}`",
            f"- OOS accounting error max abs: `{observed['oos']['accounting_error_max_abs']}`",
            f"- Validation notional contract error max abs: `{observed['validation']['notional_contract_error_max_abs']}`",
            f"- OOS notional contract error max abs: `{observed['oos']['notional_contract_error_max_abs']}`",
            f"- Validation trades: `{observed['validation']['trades']}`",
            f"- OOS trades: `{observed['oos']['trades']}`",
            "",
            "## Artifacts",
            "",
            f"- Audit JSON: `{AUDIT_JSON}`",
            f"- Candidate report: `{REPORT}`",
            f"- Validation ledger: `{payload['artifacts']['selected_validation_ledger']}`",
            f"- OOS ledger: `{payload['artifacts']['selected_oos_ledger']}`",
        ]
    )
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = load_audit_v1()
    report = read_json(REPORT)
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    artifacts = report["artifacts"]
    val_ledger = Path(artifacts["selected_validation_ledger"])
    oos_ledger = Path(artifacts["selected_oos_ledger"])
    checks: list[dict[str, Any]] = []

    missing = {
        name: str(path)
        for name, path in {
            "report": REPORT,
            "eval_script": EVAL_SCRIPT,
            "validation_ledger": val_ledger,
            "oos_ledger": oos_ledger,
            "prior_runtime_audit": PRIOR_RUNTIME_AUDIT,
        }.items()
        if not path.exists()
    }
    audit.add_check(checks, "required_artifacts_exist", not missing, "research_blocker", missing)
    audit.add_check(
        checks,
        "model_id_matches",
        report.get("model_id") == MODEL_ID,
        "research_blocker",
        {"observed": report.get("model_id"), "expected": MODEL_ID},
    )
    audit.add_check(
        checks,
        "reference_model_id_matches",
        report.get("reference_model_id") == REFERENCE_MODEL_ID,
        "research_blocker",
        {"observed": report.get("reference_model_id"), "expected": REFERENCE_MODEL_ID},
    )
    audit.add_check(
        checks,
        "parent_model_id_matches",
        report.get("parent_model_id") == PARENT_MODEL_ID,
        "research_blocker",
        {"observed": report.get("parent_model_id"), "expected": PARENT_MODEL_ID},
    )

    validation = audit.metrics(pd.read_csv(val_ledger))
    oos = audit.metrics(pd.read_csv(oos_ledger))
    audit.compare_metric_block(checks, "validation", validation, selected)
    audit.compare_metric_block(checks, "oos", oos, selected)
    audit.add_check(
        checks,
        "daytrade_hold_contract",
        validation["max_hold_hours"] <= 24.0 and oos["max_hold_hours"] <= 24.0,
        "research_blocker",
        {"validation": validation["max_hold_hours"], "oos": oos["max_hold_hours"]},
    )
    audit.add_check(
        checks,
        "pnl_over_100_percent_validation_and_oos",
        validation["pnl"] >= 100.0 and oos["pnl"] >= 100.0,
        "research_blocker",
        {"validation": validation["pnl"], "oos": oos["pnl"]},
    )
    audit.add_check(
        checks,
        "mdd_within_20_percent_validation_and_oos",
        validation["mdd"] >= -20.0 and oos["mdd"] >= -20.0,
        "research_blocker",
        {"validation": validation["mdd"], "oos": oos["mdd"]},
    )
    audit.add_check(
        checks,
        "accounting_contract_pass",
        validation["accounting_error_max_abs"] <= TOL and oos["accounting_error_max_abs"] <= TOL,
        "research_blocker",
        {"validation": validation["accounting_error_max_abs"], "oos": oos["accounting_error_max_abs"]},
    )
    audit.add_check(
        checks,
        "notional_contract_pass",
        validation["notional_contract_error_max_abs"] <= TOL
        and oos["notional_contract_error_max_abs"] <= TOL,
        "research_blocker",
        {
            "validation": validation["notional_contract_error_max_abs"],
            "oos": oos["notional_contract_error_max_abs"],
        },
    )
    audit.add_check(
        checks,
        "no_overlapping_active_trades",
        validation["overlap_count"] == 0 and oos["overlap_count"] == 0,
        "research_blocker",
        {"validation": validation["overlap_count"], "oos": oos["overlap_count"]},
    )
    audit.add_check(
        checks,
        "pnl_upgrade_vs_daytrade_reference",
        validation["pnl"] > reference["validation_pnl"] and oos["pnl"] > reference["oos_pnl"],
        "research_blocker",
        {
            "reference_validation": reference["validation_pnl"],
            "candidate_validation": validation["pnl"],
            "reference_oos": reference["oos_pnl"],
            "candidate_oos": oos["pnl"],
        },
    )

    prior = audit.prior_runtime_status()
    native_status = prior["overall"].get("runtime_native_replay_status", "MISSING")
    fresh = prior["fresh_holdout_walkforward"]
    fresh_available = bool(fresh.get("fresh_holdout_available", False))
    audit.add_check(
        checks,
        "runtime_native_replay_complete",
        native_status.startswith("PASS"),
        "live_blocker",
        {"inherited_status": native_status, "prior_audit": str(PRIOR_RUNTIME_AUDIT)},
    )
    audit.add_check(
        checks,
        "fresh_holdout_walkforward_complete",
        fresh_available,
        "live_blocker",
        {
            "fresh_holdout_available": fresh_available,
            "reason": fresh.get("fresh_holdout_unavailable_reason"),
            "prior_audit": str(PRIOR_RUNTIME_AUDIT),
        },
    )

    daytrade_pass = all(c["pass"] for c in checks if c["severity"] == "research_blocker")
    full_live_pass = daytrade_pass and all(
        c["pass"] for c in checks if c["severity"] in {"research_blocker", "live_blocker"}
    )
    if full_live_pass:
        verdict = "FULL_LIVE_PASS"
    elif daytrade_pass:
        verdict = "DAYTRADE_RESEARCH_PASS_FULL_LIVE_BLOCKED"
    else:
        verdict = "REDTEAM_FAIL"
    payload = {
        "audit_id": "omega4_6_2_v5_roll24_daytrade_redteam_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "verdict": verdict,
        "daytrade_research_pass": daytrade_pass,
        "pnl_upgrade_vs_reference": report.get("pnl_upgrade_vs_reference"),
        "full_live_pass": full_live_pass,
        "selected_variant": selected,
        "reference_variant": reference,
        "observed": {"validation": validation, "oos": oos},
        "checks": checks,
        "prior_runtime_audit": prior,
        "artifacts": {**artifacts, "audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD)},
    }
    audit.write_json(AUDIT_JSON, payload)
    write_markdown(payload)
    print(
        json.dumps(
            {
                "audit_json": str(AUDIT_JSON),
                "audit_md": str(AUDIT_MD),
                "verdict": verdict,
                "daytrade_research_pass": daytrade_pass,
                "full_live_pass": full_live_pass,
            },
            ensure_ascii=False,
            default=audit.json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
