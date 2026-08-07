#!/usr/bin/env python3
"""Red-team audit for Omega 4.6.2 v5 roll4 hold-compressed branch."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_HELPER_PATH = ROOT / "scripts/audit_omega4_6_2_loss_cluster_governor_redteam_20260701.py"
EVAL_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REPORT = OUT_DIR / "report.json"
RANKING = OUT_DIR / "roll4_two_stage_exposure_hold_compressed_ranking.csv"
TOP20 = OUT_DIR / "roll4_two_stage_exposure_hold_compressed_top20.csv"
PRIOR_RUNTIME_AUDIT = (
    ROOT / "docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json"
)
AUDIT_JSON = OUT_DIR / "redteam_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_redteam_20260701.md"
TOL = 1.0e-8


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def max_abs_diff(left: pd.Series, right: pd.Series) -> float:
    if len(left) == 0:
        return 0.0
    return float((left.astype(float).reset_index(drop=True) - right.astype(float).reset_index(drop=True)).abs().max())


def replay_compare(expected: pd.DataFrame, observed: pd.DataFrame) -> dict[str, Any]:
    id_cols = ["entry_i", "exit_i", "side", "entry_timestamp", "exit_timestamp"]
    id_match = bool((expected[id_cols].astype(str) == observed[id_cols].astype(str)).all().all())
    cols = [
        col
        for col in [
            "notional",
            "leverage",
            "margin_fraction",
            "risk_notional",
            "risk_leverage",
            "risk_margin_fraction",
            "exit_input_notional",
            "exit_input_leverage",
            "exit_input_exposure",
            "trade_return",
        ]
        if col in expected.columns and col in observed.columns
    ]
    diffs = {col: max_abs_diff(expected[col], observed[col]) for col in cols}
    return {
        "pass": bool(id_match and all(value <= TOL for value in diffs.values())),
        "id_match": id_match,
        "max_abs_diffs": diffs,
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
        "# Omega 4.6.2 v5 Roll4 Hold-Compressed Red-Team Audit - 2026-07-01",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Reference: `{payload['reference_model_id']}`",
        f"- Parent: `{payload['parent_model_id']}`",
        f"- Verdict: `{payload['verdict']}`",
        f"- Research pass: `{payload['research_pass']}`",
        f"- Full live pass: `{payload['full_live_pass']}`",
        "",
        "## Selected Candidate",
        "",
        f"- Exposure spec: `{selected['exposure_spec']}`",
        f"- Max roll hold: `{selected['roll4_max_hours']}h`",
        f"- OOS MDD buffer to -20%: `{20.0 + observed['oos']['mdd']:.4f}pp`",
        "",
        "| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["validation", "oos"]:
        lines.append(
            "| "
            f"{split} | "
            f"{reference[f'{split}_pnl']:.4f}% | {observed[split]['pnl']:.4f}% | "
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
            "## Replay Checks",
            "",
            f"- Validation replay: `{payload['roll4_replay']['validation']['pass']}`",
            f"- OOS replay: `{payload['roll4_replay']['oos']['pass']}`",
            "",
            "## Artifacts",
            "",
            f"- Audit JSON: `{AUDIT_JSON}`",
            f"- Candidate report: `{REPORT}`",
            f"- Ranking: `{RANKING}`",
            f"- Top 20: `{TOP20}`",
            f"- Validation ledger: `{payload['artifacts']['selected_validation_ledger']}`",
            f"- OOS ledger: `{payload['artifacts']['selected_oos_ledger']}`",
        ]
    )
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = load_module("omega462_loss_cluster_audit_for_roll4_hold_compressed", AUDIT_HELPER_PATH)
    eval_mod = load_module("omega462_roll4_hold_compressed_eval_for_audit", EVAL_MODULE_PATH)
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
            "top20": TOP20,
            "eval_script": EVAL_MODULE_PATH,
            "validation_ledger": val_ledger,
            "oos_ledger": oos_ledger,
            "prior_runtime_audit": PRIOR_RUNTIME_AUDIT,
        }.items()
        if not path.exists()
    }
    audit.add_check(checks, "required_artifacts_exist", not missing, "research_blocker", missing)
    audit.add_check(checks, "model_id_matches", report.get("model_id") == MODEL_ID, "research_blocker", {"observed": report.get("model_id")})
    audit.add_check(checks, "reference_model_id_matches", report.get("reference_model_id") == REFERENCE_MODEL_ID, "research_blocker", {"observed": report.get("reference_model_id")})
    audit.add_check(checks, "parent_model_id_matches", report.get("parent_model_id") == PARENT_MODEL_ID, "research_blocker", {"observed": report.get("parent_model_id")})
    audit.add_check(
        checks,
        "selection_scope_declared",
        report.get("selection_scope")
        == "validation_primary_roll4_two_stage_veto_exposure_overlay_with_oos_safety_gate; fresh_holdout_required",
        "research_blocker",
        {"observed": report.get("selection_scope")},
    )
    audit.add_check(
        checks,
        "selected_gate_pass",
        selected.get("research_roll4_hold_compressed_gate_pass") is True,
        "research_blocker",
        {"observed": selected.get("research_roll4_hold_compressed_gate_pass")},
    )
    validation_df = pd.read_csv(val_ledger)
    oos_df = pd.read_csv(oos_ledger)
    validation = audit.metrics(validation_df)
    oos = audit.metrics(oos_df)
    audit.compare_metric_block(checks, "validation", validation, selected)
    audit.compare_metric_block(checks, "oos", oos, selected)
    replay_val, replay_oos = eval_mod.build_selected_ledgers(selected)
    val_replay = replay_compare(replay_val, validation_df)
    oos_replay = replay_compare(replay_oos, oos_df)
    audit.add_check(
        checks,
        "roll4_replay_parity",
        val_replay["pass"] and oos_replay["pass"],
        "research_blocker",
        {"validation": val_replay, "oos": oos_replay},
    )
    audit.add_check(
        checks,
        "hold_compression_contract",
        validation["pnl"] >= 100.0
        and oos["pnl"] >= 100.0
        and validation["mdd"] >= -20.0
        and oos["mdd"] >= -20.0
        and validation["avg_hold_hours"] < reference["validation_avg_hold_hours"]
        and oos["avg_hold_hours"] < reference["oos_avg_hold_hours"]
        and validation["max_hold_hours"] <= 4.0
        and oos["max_hold_hours"] <= 4.0,
        "research_blocker",
        {"reference": reference, "candidate": {"validation": validation, "oos": oos}},
    )
    audit.add_check(
        checks,
        "exposure_accounting_contract",
        validation["max_leverage"] <= 5.0 + TOL
        and oos["max_leverage"] <= 5.0 + TOL
        and validation["max_notional"] <= 5.0 + TOL
        and oos["max_notional"] <= 5.0 + TOL
        and validation["max_margin_fraction"] <= 1.0 + TOL
        and oos["max_margin_fraction"] <= 1.0 + TOL
        and validation["accounting_error_max_abs"] <= TOL
        and oos["accounting_error_max_abs"] <= TOL
        and validation["notional_contract_error_max_abs"] <= TOL
        and oos["notional_contract_error_max_abs"] <= TOL
        and validation["overlap_count"] == 0
        and oos["overlap_count"] == 0,
        "research_blocker",
        {"validation": validation, "oos": oos},
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
    research_pass = all(c["pass"] for c in checks if c["severity"] == "research_blocker")
    full_live_pass = research_pass and all(
        c["pass"] for c in checks if c["severity"] in {"research_blocker", "live_blocker"}
    )
    verdict = (
        "FULL_LIVE_PASS"
        if full_live_pass
        else "RESEARCH_ROLL4_HOLD_COMPRESSED_PASS_FULL_LIVE_BLOCKED"
        if research_pass
        else "REDTEAM_FAIL"
    )
    payload = {
        "audit_id": "omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_redteam_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "verdict": verdict,
        "research_pass": research_pass,
        "full_live_pass": full_live_pass,
        "selected_variant": selected,
        "reference_variant": reference,
        "observed": {"validation": validation, "oos": oos},
        "checks": checks,
        "roll4_replay": {"validation": val_replay, "oos": oos_replay},
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
                "research_pass": research_pass,
                "full_live_pass": full_live_pass,
            },
            ensure_ascii=False,
            default=audit.json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
