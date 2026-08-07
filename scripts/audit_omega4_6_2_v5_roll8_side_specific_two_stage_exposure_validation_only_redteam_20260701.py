#!/usr/bin/env python3
"""Red-team audit for the Omega 4.6.2 validation-only exposure branch."""

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
BASE_EXPOSURE_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
REPORT = OUT_DIR / "report.json"
AUDIT_JSON = OUT_DIR / "redteam_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_redteam_20260701.md"
RUNTIME_REPLAY_AUDIT = OUT_DIR / "runtime_replay_audit_20260701.json"
TOL = 1.0e-8


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def max_abs_diff(left: pd.Series, right: pd.Series) -> float:
    if len(left) == 0:
        return 0.0
    return float((left.astype(float).reset_index(drop=True) - right.astype(float).reset_index(drop=True)).abs().max())


def exposure_replay(reference_df: pd.DataFrame, candidate_df: pd.DataFrame, selected: dict[str, Any], eval_mod: Any) -> dict[str, Any]:
    replay = eval_mod.apply_exposure_overlay(
        reference_df,
        float(selected["exposure_long_factor"]),
        float(selected["exposure_short_factor"]),
        float(selected["exposure_cap_notional"]),
    )
    id_cols = ["entry_i", "exit_i", "side", "entry_timestamp", "exit_timestamp"]
    id_match = bool((replay[id_cols].astype(str) == candidate_df[id_cols].astype(str)).all().all())
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
        if col in replay.columns and col in candidate_df.columns
    ]
    diffs = {col: max_abs_diff(replay[col], candidate_df[col]) for col in cols}
    spec_match = bool((candidate_df["two_stage_exposure_spec"].astype(str) == str(selected["exposure_spec"])).all())
    return {
        "pass": bool(id_match and spec_match and all(value <= TOL for value in diffs.values())),
        "id_match": id_match,
        "spec_match": spec_match,
        "max_abs_diffs": diffs,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    selected = payload["selected_variant"]
    observed = payload["observed"]
    failed = [c for c in payload["checks"] if not c["pass"]]
    lines = [
        "# Omega 4.6.2 v5 Roll8 Two-Stage Exposure Validation-Only Red-Team - 2026-07-01",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Verdict: `{payload['verdict']}`",
        f"- Research pass: `{payload['research_pass']}`",
        f"- Full live pass: `{payload['full_live_pass']}`",
        f"- OOS used in selection: `{selected.get('oos_used_in_selection')}`",
        "",
        "## Selected Exposure",
        "",
        f"- Exposure spec: `{selected['exposure_spec']}`",
        f"- Long/short factor: `{selected['exposure_long_factor']}` / `{selected['exposure_short_factor']}`",
        f"- Cap notional: `{selected['exposure_cap_notional']}`",
        "",
        "| Split | PnL | MDD | Avg Hold | Max Hold | Trades |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["validation", "oos"]:
        row = observed[split]
        lines.append(
            f"| `{split}` | `{row['pnl']:.4f}%` | `{row['mdd']:.4f}%` | "
            f"`{row['avg_hold_hours']:.4f}h` | `{row['max_hold_hours']:.4f}h` | `{row['trades']}` |"
        )
    lines.extend(["", "## Failed Checks", ""])
    lines.extend([f"- `{c['name']}`: {c['details']}" for c in failed] or ["- None."])
    lines.extend(
        [
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


def main() -> int:
    audit = load_module("omega462_loss_cluster_audit_for_validation_only", AUDIT_HELPER_PATH)
    eval_mod = load_module("omega462_two_stage_exposure_eval_for_validation_only_audit", BASE_EXPOSURE_PATH)
    report = read_json(REPORT)
    selected = report["selected_variant"]
    reference_report = read_json(REFERENCE_REPORT)
    artifacts = report["artifacts"]
    source_artifacts = reference_report["artifacts"]
    val_ledger = Path(artifacts["selected_validation_ledger"])
    oos_ledger = Path(artifacts["selected_oos_ledger"])
    source_val_ledger = Path(source_artifacts["selected_validation_ledger"])
    source_oos_ledger = Path(source_artifacts["selected_oos_ledger"])
    validation_df = pd.read_csv(val_ledger)
    oos_df = pd.read_csv(oos_ledger)
    source_validation_df = pd.read_csv(source_val_ledger)
    source_oos_df = pd.read_csv(source_oos_ledger)
    validation = audit.metrics(validation_df)
    oos = audit.metrics(oos_df)
    val_replay = exposure_replay(source_validation_df, validation_df, selected, eval_mod)
    oos_replay = exposure_replay(source_oos_df, oos_df, selected, eval_mod)
    runtime_replay = read_json(RUNTIME_REPLAY_AUDIT) if RUNTIME_REPLAY_AUDIT.exists() else {}
    checks: list[dict[str, Any]] = []
    missing = {
        name: str(path)
        for name, path in {
            "report": REPORT,
            "validation_ledger": val_ledger,
            "oos_ledger": oos_ledger,
            "source_validation_ledger": source_val_ledger,
            "source_oos_ledger": source_oos_ledger,
        }.items()
        if not path.exists()
    }
    audit.add_check(checks, "required_artifacts_exist", not missing, "research_blocker", missing)
    audit.add_check(checks, "model_id_matches", report.get("model_id") == MODEL_ID, "research_blocker", {"observed": report.get("model_id")})
    audit.add_check(checks, "reference_model_id_matches", report.get("reference_model_id") == REFERENCE_MODEL_ID, "research_blocker", {"observed": report.get("reference_model_id")})
    audit.add_check(checks, "parent_model_id_matches", report.get("parent_model_id") == PARENT_MODEL_ID, "research_blocker", {"observed": report.get("parent_model_id")})
    audit.add_check(
        checks,
        "selection_scope_declared_validation_only",
        report.get("selection_scope") == "validation_only_two_stage_exposure_overlay_with_validation_mdd_floor; oos_readout_after_selection",
        "research_blocker",
        {"observed": report.get("selection_scope")},
    )
    audit.add_check(
        checks,
        "selection_rule_declares_no_oos_filter_or_tiebreak",
        report.get("selection_rule")
        == "require validation_two_stage_exposure_gate_pass and validation_mdd >= -17.50; sort by validation_pnl, validation_mdd, validation_avg_hold_hours; OOS is not used as filter, ordering key, or tie-breaker",
        "research_blocker",
        {"observed": report.get("selection_rule")},
    )
    audit.add_check(
        checks,
        "oos_not_used_in_selection_flag",
        selected.get("oos_used_in_selection") is False and selected.get("research_validation_only_gate_pass") is True,
        "research_blocker",
        {"oos_used_in_selection": selected.get("oos_used_in_selection"), "research_validation_only_gate_pass": selected.get("research_validation_only_gate_pass")},
    )
    audit.compare_metric_block(checks, "validation", validation, selected)
    audit.compare_metric_block(checks, "oos", oos, selected)
    audit.add_check(checks, "exposure_replay_parity", val_replay["pass"] and oos_replay["pass"], "research_blocker", {"validation": val_replay, "oos": oos_replay})
    audit.add_check(
        checks,
        "pnl_mdd_hold_research_contract",
        validation["pnl"] >= 100.0
        and oos["pnl"] >= 100.0
        and validation["mdd"] >= -20.0
        and oos["mdd"] >= -20.0
        and validation["max_hold_hours"] <= 8.0
        and oos["max_hold_hours"] <= 8.0,
        "research_blocker",
        {"validation": validation, "oos": oos},
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
    audit.add_check(
        checks,
        "runtime_owned_replay_pass",
        runtime_replay.get("verdict") == "RUNTIME_REPLAY_PASS"
        and runtime_replay.get("runtime_replay_pass") is True,
        "live_blocker",
        {"runtime_replay_audit": str(RUNTIME_REPLAY_AUDIT), "observed_verdict": runtime_replay.get("verdict")},
    )
    research_pass = all(c["pass"] for c in checks if c["severity"] == "research_blocker")
    full_live_pass = research_pass and all(
        c["pass"] for c in checks if c["severity"] in {"research_blocker", "live_blocker"}
    )
    payload = {
        "audit_id": "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_redteam_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "verdict": "FULL_LIVE_PASS_VALIDATION_ONLY" if full_live_pass else "RESEARCH_VALIDATION_ONLY_PASS" if research_pass else "REDTEAM_FAIL",
        "research_pass": research_pass,
        "full_live_pass": full_live_pass,
        "selected_variant": selected,
        "observed": {"validation": validation, "oos": oos},
        "checks": checks,
        "exposure_replay": {"validation": val_replay, "oos": oos_replay},
        "artifacts": {**artifacts, "audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD)},
    }
    audit.write_json(AUDIT_JSON, payload)
    write_markdown(payload)
    print(json.dumps({"audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD), "verdict": payload["verdict"], "research_pass": research_pass}, ensure_ascii=False, default=audit.json_default))
    return 0 if research_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
