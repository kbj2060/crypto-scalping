#!/usr/bin/env python3
"""Red-team audit for Omega 4.6.2 v5 roll8 side-specific feature veto branch."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_HELPER_PATH = ROOT / "scripts/audit_omega4_6_2_loss_cluster_governor_redteam_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll8_side_specific_feature_veto_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REFERENCE_OUT_DIR = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID
REPORT = OUT_DIR / "report.json"
REFERENCE_REPORT = REFERENCE_OUT_DIR / "report.json"
RANKING = OUT_DIR / "roll8_side_specific_feature_veto_ranking.csv"
TOP20 = OUT_DIR / "roll8_side_specific_feature_veto_top20.csv"
EVAL_SCRIPT = ROOT / "scripts/eval_omega4_6_2_v5_roll8_side_specific_feature_veto_20260701.py"
PRIOR_RUNTIME_AUDIT = (
    ROOT / "docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json"
)
AUDIT_JSON = OUT_DIR / "redteam_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll8_side_specific_feature_veto_redteam_20260701.md"
EPS = 1.0e-12
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


def feature_veto_mask(df: pd.DataFrame, selected: dict[str, Any]) -> pd.Series:
    feature = str(selected["feature_name"])
    op = str(selected["feature_op"])
    threshold = float(selected["feature_threshold"])
    active = df["notional"].astype(float) > EPS
    is_short = df["side"].astype(int) < 0
    values = pd.to_numeric(df[feature], errors="coerce")
    if op == "<=":
        condition = values <= threshold
    elif op == ">=":
        condition = values >= threshold
    else:
        raise ValueError(f"unsupported feature op: {op}")
    return active & is_short & condition.fillna(False)


def feature_veto_replay(
    split: str,
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    selected: dict[str, Any],
) -> dict[str, Any]:
    if len(reference_df) != len(candidate_df):
        return {
            "split": split,
            "pass": False,
            "reason": "row_count_mismatch",
            "reference_rows": len(reference_df),
            "candidate_rows": len(candidate_df),
        }
    id_cols = ["entry_i", "exit_i", "side", "entry_timestamp", "exit_timestamp"]
    id_match = bool((reference_df[id_cols].astype(str) == candidate_df[id_cols].astype(str)).all().all())
    expected = feature_veto_mask(reference_df, selected)
    actual = (
        (candidate_df["notional"].astype(float) <= EPS)
        & (candidate_df["reason"].astype(str) == "feature_veto")
    )
    if "feature_veto_spec" in candidate_df.columns:
        expected_spec = (
            f"{selected['feature_name']}_{selected['feature_op']}_"
            f"{float(selected['feature_threshold']):.8g}"
        )
        spec_ok = bool((candidate_df.loc[actual, "feature_veto_spec"].astype(str) == expected_spec).all())
        non_veto_spec_ok = bool(
            (candidate_df.loc[~actual, "feature_veto_spec"].astype(str) == "none").all()
        )
    else:
        expected_spec = None
        spec_ok = False
        non_veto_spec_ok = False
    mask_match = bool((expected.to_numpy() == actual.to_numpy()).all())
    zero_cols = [
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
        if col in candidate_df.columns
    ]
    veto_zero_ok = bool((candidate_df.loc[actual, zero_cols].astype(float).abs() <= TOL).all().all())
    common_cols = [c for c in reference_df.columns if c in candidate_df.columns and c != "feature_veto_spec"]
    non_veto_match = bool(
        (
            reference_df.loc[~expected, common_cols].astype(str).reset_index(drop=True)
            == candidate_df.loc[~actual, common_cols].astype(str).reset_index(drop=True)
        )
        .all()
        .all()
    )
    return {
        "split": split,
        "pass": bool(
            id_match
            and mask_match
            and spec_ok
            and non_veto_spec_ok
            and veto_zero_ok
            and non_veto_match
        ),
        "id_match": id_match,
        "mask_match": mask_match,
        "spec_ok": spec_ok,
        "non_veto_spec_ok": non_veto_spec_ok,
        "veto_zero_ok": veto_zero_ok,
        "non_veto_match": non_veto_match,
        "expected_spec": expected_spec,
        "expected_veto_count": int(expected.sum()),
        "actual_veto_count": int(actual.sum()),
        "selected_veto_count": int(selected[f"{split}_vetoed"]),
        "long_veto_count": int((actual & (candidate_df["side"].astype(int) > 0)).sum()),
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
        "# Omega 4.6.2 v5 Roll8 Side-Specific Feature Veto Red-Team Audit - 2026-07-01",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Reference: `{payload['reference_model_id']}`",
        f"- Parent: `{payload['parent_model_id']}`",
        f"- Verdict: `{payload['verdict']}`",
        f"- Research pass: `{payload['research_pass']}`",
        f"- Full live pass: `{payload['full_live_pass']}`",
        "",
        "## Selected Veto",
        "",
        f"- Feature: `{selected['feature_name']}`",
        f"- Rule: `{selected['feature_name']} {selected['feature_op']} {float(selected['feature_threshold']):.8g}`",
        f"- Quantile: `{selected['feature_quantile']}`",
        f"- Validation/OOS vetoed shorts: `{selected['validation_vetoed']}` / `{selected['oos_vetoed']}`",
        f"- Search scope: `{payload['selection_scope']}`",
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
            f"- Validation feature-veto replay: `{payload['feature_veto_replay']['validation']['pass']}`",
            f"- OOS feature-veto replay: `{payload['feature_veto_replay']['oos']['pass']}`",
            f"- Lookahead regex: `{payload['lookahead_exclude_regex']}`",
            "",
            "## Contract Checks",
            "",
            f"- Validation accounting error max abs: `{observed['validation']['accounting_error_max_abs']}`",
            f"- OOS accounting error max abs: `{observed['oos']['accounting_error_max_abs']}`",
            f"- Validation notional contract error max abs: `{observed['validation']['notional_contract_error_max_abs']}`",
            f"- OOS notional contract error max abs: `{observed['oos']['notional_contract_error_max_abs']}`",
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
    audit = load_module("omega462_loss_cluster_audit_for_roll8_feature_veto", AUDIT_HELPER_PATH)
    report = read_json(REPORT)
    reference_report = read_json(REFERENCE_REPORT)
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    artifacts = report["artifacts"]
    ref_artifacts = reference_report["artifacts"]
    val_ledger = Path(artifacts["selected_validation_ledger"])
    oos_ledger = Path(artifacts["selected_oos_ledger"])
    ref_val_ledger = Path(ref_artifacts["selected_validation_ledger"])
    ref_oos_ledger = Path(ref_artifacts["selected_oos_ledger"])
    checks: list[dict[str, Any]] = []
    missing = {
        name: str(path)
        for name, path in {
            "report": REPORT,
            "reference_report": REFERENCE_REPORT,
            "ranking": RANKING,
            "top20": TOP20,
            "eval_script": EVAL_SCRIPT,
            "validation_ledger": val_ledger,
            "oos_ledger": oos_ledger,
            "reference_validation_ledger": ref_val_ledger,
            "reference_oos_ledger": ref_oos_ledger,
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
        == "validation_primary_single_entry_feature_short_veto_with_oos_reference_safety_gate; fresh_holdout_required",
        "research_blocker",
        {"observed": report.get("selection_scope")},
    )
    audit.add_check(
        checks,
        "selection_rule_declares_oos_not_ordering_key",
        report.get("selection_rule")
        == "search single non-lookahead-named numeric entry feature thresholds; among research-gated variants sort by validation_pnl, validation_avg_hold, validation_vetoed; OOS is not an ordering key",
        "research_blocker",
        {"observed": report.get("selection_rule")},
    )
    audit.add_check(
        checks,
        "selected_feature_veto_gate_pass",
        selected.get("research_feature_veto_gate_pass") is True,
        "research_blocker",
        {"observed": selected.get("research_feature_veto_gate_pass")},
    )
    feature = str(selected.get("feature_name"))
    lookahead_re = re.compile(str(report.get("lookahead_exclude_regex", "")), re.IGNORECASE)
    audit.add_check(
        checks,
        "selected_feature_not_lookahead_named",
        bool(feature) and lookahead_re.search(feature) is None,
        "research_blocker",
        {"feature": feature, "regex": report.get("lookahead_exclude_regex")},
    )
    audit.add_check(
        checks,
        "veto_fraction_limited",
        int(selected["validation_vetoed"])
        <= int(reference["validation_short_trades"] * selected["max_validation_short_veto_fraction"]),
        "research_blocker",
        {
            "validation_vetoed": selected["validation_vetoed"],
            "validation_short_trades": reference["validation_short_trades"],
            "max_fraction": selected["max_validation_short_veto_fraction"],
        },
    )

    validation_df = pd.read_csv(val_ledger)
    oos_df = pd.read_csv(oos_ledger)
    ref_validation_df = pd.read_csv(ref_val_ledger)
    ref_oos_df = pd.read_csv(ref_oos_ledger)
    validation = audit.metrics(validation_df)
    oos = audit.metrics(oos_df)
    audit.compare_metric_block(checks, "validation", validation, selected)
    audit.compare_metric_block(checks, "oos", oos, selected)
    val_replay = feature_veto_replay("validation", ref_validation_df, validation_df, selected)
    oos_replay = feature_veto_replay("oos", ref_oos_df, oos_df, selected)
    audit.add_check(
        checks,
        "feature_veto_replay_parity",
        val_replay["pass"] and oos_replay["pass"],
        "research_blocker",
        {"validation": val_replay, "oos": oos_replay},
    )
    audit.add_check(
        checks,
        "feature_veto_count_parity",
        val_replay["actual_veto_count"] == int(selected["validation_vetoed"])
        and oos_replay["actual_veto_count"] == int(selected["oos_vetoed"])
        and val_replay["long_veto_count"] == 0
        and oos_replay["long_veto_count"] == 0,
        "research_blocker",
        {"validation": val_replay, "oos": oos_replay},
    )
    audit.add_check(
        checks,
        "pnl_mdd_hold_research_contract",
        validation["pnl"] > reference["validation_pnl"]
        and oos["pnl"] >= reference["oos_pnl"]
        and validation["mdd"] >= -20.0
        and oos["mdd"] >= reference["oos_mdd"]
        and validation["avg_hold_hours"] < reference["validation_avg_hold_hours"]
        and oos["avg_hold_hours"] <= reference["oos_avg_hold_hours"] + TOL
        and validation["max_hold_hours"] <= 8.0
        and oos["max_hold_hours"] <= 8.0,
        "research_blocker",
        {"reference": reference, "candidate": {"validation": validation, "oos": oos}},
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
        {"validation": validation["notional_contract_error_max_abs"], "oos": oos["notional_contract_error_max_abs"]},
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
        "exposure_caps_respected",
        validation["max_leverage"] <= 5.0 + TOL
        and oos["max_leverage"] <= 5.0 + TOL
        and validation["max_notional"] <= 5.0 + TOL
        and oos["max_notional"] <= 5.0 + TOL,
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
        else "RESEARCH_ROLL8_FEATURE_VETO_PASS_FULL_LIVE_BLOCKED"
        if research_pass
        else "REDTEAM_FAIL"
    )
    payload = {
        "audit_id": "omega4_6_2_v5_roll8_side_specific_feature_veto_redteam_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "verdict": verdict,
        "research_pass": research_pass,
        "full_live_pass": full_live_pass,
        "selection_scope": report.get("selection_scope"),
        "selection_rule": report.get("selection_rule"),
        "lookahead_exclude_regex": report.get("lookahead_exclude_regex"),
        "features_evaluated": report.get("features_evaluated"),
        "variants_evaluated": report.get("variants_evaluated"),
        "selected_variant": selected,
        "reference_variant": reference,
        "observed": {"validation": validation, "oos": oos},
        "checks": checks,
        "feature_veto_replay": {"validation": val_replay, "oos": oos_replay},
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
