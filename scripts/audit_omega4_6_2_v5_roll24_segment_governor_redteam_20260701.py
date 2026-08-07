#!/usr/bin/env python3
"""Red-team audit for the Omega 4.6.2 v5 roll24 segment governor."""

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
MODEL_ID = "omega4_6_2_v5_roll24_segment_governor_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll24_daytrade_overlay_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REPORT = OUT_DIR / "report.json"
RANKING = OUT_DIR / "v5_roll24_segment_governor_ranking.csv"
EVAL_SCRIPT = ROOT / "scripts/eval_omega4_6_2_v5_roll24_segment_governor_20260701.py"
PRIOR_RUNTIME_AUDIT = (
    ROOT / "docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json"
)
AUDIT_JSON = OUT_DIR / "redteam_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll24_segment_governor_redteam_20260701.md"
EPS = 1.0e-12
TOL = 1.0e-8


def load_audit_v1() -> Any:
    spec = importlib.util.spec_from_file_location("omega462_loss_cluster_audit_v1_for_v5_segment", AUDIT_V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {AUDIT_V1_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def segment_governor_replay(df: pd.DataFrame, selected: dict[str, Any]) -> dict[str, Any]:
    work = df.copy()
    work["entry_timestamp_dt"] = pd.to_datetime(work["entry_timestamp"], errors="raise")
    work["exit_timestamp_dt"] = pd.to_datetime(work["exit_timestamp"], errors="raise")
    active = work[work["notional"].astype(float) > EPS].sort_values(["entry_i", "exit_i"]).reset_index(drop=True)
    loss1 = float(selected["segment_governor_loss1_scale"])
    loss2 = float(selected["segment_governor_loss2_scale"])
    loss_window_hours = float(selected["segment_governor_loss_window_hours"])
    loss_streak = 0
    last_loss_exit_ts: pd.Timestamp | None = None
    errors: list[dict[str, Any]] = []
    max_multiplier_diff = 0.0
    max_notional_diff = 0.0
    max_return_diff = 0.0
    for i, row in active.iterrows():
        entry_ts = pd.Timestamp(row["entry_timestamp_dt"])
        effective_streak = loss_streak
        if last_loss_exit_ts is not None:
            hours_from_loss = (entry_ts - last_loss_exit_ts).total_seconds() / 3600.0
            if hours_from_loss > loss_window_hours:
                effective_streak = 0
        if effective_streak >= 2:
            expected_multiplier = loss2
        elif effective_streak == 1:
            expected_multiplier = loss1
        else:
            expected_multiplier = 1.0
        expected_notional = float(row["pre_governor_notional"]) * expected_multiplier
        expected_return = float(row["net_per_notional"]) * expected_notional
        multiplier_diff = abs(float(row["roll24_segment_multiplier"]) - expected_multiplier)
        notional_diff = abs(float(row["notional"]) - expected_notional)
        return_diff = abs(float(row["trade_return"]) - expected_return)
        max_multiplier_diff = max(max_multiplier_diff, multiplier_diff)
        max_notional_diff = max(max_notional_diff, notional_diff)
        max_return_diff = max(max_return_diff, return_diff)
        if multiplier_diff > TOL or notional_diff > TOL or return_diff > TOL:
            errors.append(
                {
                    "row": int(i),
                    "entry_timestamp": str(row["entry_timestamp"]),
                    "expected_multiplier": expected_multiplier,
                    "observed_multiplier": float(row["roll24_segment_multiplier"]),
                    "expected_notional": expected_notional,
                    "observed_notional": float(row["notional"]),
                    "expected_return": expected_return,
                    "observed_return": float(row["trade_return"]),
                }
            )
        if float(row["net_per_notional"]) < 0.0:
            loss_streak += 1
            last_loss_exit_ts = pd.Timestamp(row["exit_timestamp_dt"])
        else:
            loss_streak = 0
            last_loss_exit_ts = None
    return {
        "pass": not errors,
        "errors": errors[:10],
        "error_count": len(errors),
        "max_multiplier_diff": max_multiplier_diff,
        "max_notional_diff": max_notional_diff,
        "max_return_diff": max_return_diff,
        "tolerance": TOL,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    selected = payload["selected_variant"]
    reference = payload["reference_variant"]
    observed = payload["observed"]
    blockers = [c for c in payload["checks"] if not c["pass"] and c["severity"] == "live_blocker"]
    research_fails = [
        c for c in payload["checks"] if not c["pass"] and c["severity"] == "research_blocker"
    ]
    lines = [
        "# Omega 4.6.2 v5 Roll24 Segment Governor Red-Team Audit - 2026-07-01",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Parent: `{payload['parent_model_id']}`",
        f"- Reference daytrade model: `{payload['reference_model_id']}`",
        f"- Verdict: `{payload['verdict']}`",
        f"- Daytrade research pass: `{payload['daytrade_research_pass']}`",
        f"- Full live pass: `{payload['full_live_pass']}`",
        "",
        "## Selected Candidate",
        "",
        f"- Exposure spec: `{selected['exposure_spec']}`",
        f"- Segment governor: `{selected['segment_governor_spec']}`",
        "",
        "| Split | Reference PnL | Candidate PnL | Reference MDD | Candidate MDD | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |",
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
            f"- Validation max notional: `{observed['validation']['max_notional']}`",
            f"- OOS max notional: `{observed['oos']['max_notional']}`",
            f"- Segment governor replay: `{payload['segment_governor_replay_pass']}`",
            "",
            "## Artifacts",
            "",
            f"- Audit JSON: `{AUDIT_JSON}`",
            f"- Candidate report: `{REPORT}`",
            f"- Ranking: `{RANKING}`",
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
            "ranking": RANKING,
            "eval_script": EVAL_SCRIPT,
            "validation_ledger": val_ledger,
            "oos_ledger": oos_ledger,
            "prior_runtime_audit": PRIOR_RUNTIME_AUDIT,
        }.items()
        if not path.exists()
    }
    audit.add_check(checks, "required_artifacts_exist", not missing, "research_blocker", missing)
    audit.add_check(checks, "model_id_matches", report.get("model_id") == MODEL_ID, "research_blocker", {"observed": report.get("model_id"), "expected": MODEL_ID})
    audit.add_check(checks, "reference_model_id_matches", report.get("reference_model_id") == REFERENCE_MODEL_ID, "research_blocker", {"observed": report.get("reference_model_id"), "expected": REFERENCE_MODEL_ID})
    audit.add_check(checks, "parent_model_id_matches", report.get("parent_model_id") == PARENT_MODEL_ID, "research_blocker", {"observed": report.get("parent_model_id"), "expected": PARENT_MODEL_ID})
    audit.add_check(
        checks,
        "selection_scope_discloses_oos_safety_gate",
        report.get("selection_scope") == "validation_primary_with_oos_safety_gate; fresh_holdout_required",
        "research_blocker",
        {"observed": report.get("selection_scope")},
    )
    audit.add_check(checks, "selected_validation_upgrade_gate_pass", selected.get("validation_upgrade_gate_pass") is True, "research_blocker", {"observed": selected.get("validation_upgrade_gate_pass")})
    audit.add_check(checks, "selected_oos_research_gate_pass", selected.get("oos_research_gate_pass") is True, "research_blocker", {"observed": selected.get("oos_research_gate_pass")})
    audit.add_check(checks, "selected_research_upgrade_gate_pass", selected.get("research_upgrade_gate_pass") is True, "research_blocker", {"observed": selected.get("research_upgrade_gate_pass")})

    validation_df = pd.read_csv(val_ledger)
    oos_df = pd.read_csv(oos_ledger)
    validation = audit.metrics(validation_df)
    oos = audit.metrics(oos_df)
    audit.compare_metric_block(checks, "validation", validation, selected)
    audit.compare_metric_block(checks, "oos", oos, selected)
    val_replay = segment_governor_replay(validation_df, selected)
    oos_replay = segment_governor_replay(oos_df, selected)
    audit.add_check(
        checks,
        "segment_governor_replay_parity",
        val_replay["pass"] and oos_replay["pass"],
        "research_blocker",
        {"validation": val_replay, "oos": oos_replay},
    )
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
    exposure_cap = float(selected["exposure_cap_notional"])
    audit.add_check(
        checks,
        "notional_within_declared_cap",
        validation["max_notional"] <= exposure_cap + TOL and oos["max_notional"] <= exposure_cap + TOL,
        "research_blocker",
        {"declared_cap": exposure_cap, "validation": validation["max_notional"], "oos": oos["max_notional"]},
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
        "audit_id": "omega4_6_2_v5_roll24_segment_governor_redteam_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "verdict": verdict,
        "daytrade_research_pass": daytrade_pass,
        "full_live_pass": full_live_pass,
        "selected_variant": selected,
        "reference_variant": reference,
        "observed": {"validation": validation, "oos": oos},
        "checks": checks,
        "segment_governor_replay_pass": val_replay["pass"] and oos_replay["pass"],
        "segment_governor_replay": {"validation": val_replay, "oos": oos_replay},
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
