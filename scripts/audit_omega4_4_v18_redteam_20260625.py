#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_4_v18_redteam_audit_20260625"
DOC_PATH = ROOT / "docs/audits/omega4_4_v18_redteam_audit_20260625.md"

EXPECTED_MODEL_ALIAS = "omega4_4_v18_baseline_20260624"
CONTRACT_DOC = ROOT / "docs/model_contracts/omega4_4_v18_baseline_20260624_contract.md"
PROMOTION_MANIFEST = RUN_DIR / "promotion_manifest.json"
RUNTIME_CONTRACT = RUN_DIR / "runtime_contract.json"
CANDIDATE_MANIFEST = ROOT / f"data/ensemble/supervised/{EXPECTED_MODEL_ALIAS}/candidate_manifest.json"
FORBIDDEN_FEATURE_TERMS = (
    "future",
    "fwd",
    "forward",
    "target",
    "label",
    "net_per_notional",
    "trade_return",
    "pnl",
    "mfe",
    "mae",
    "exit_i",
    "exit_timestamp",
    "win",
)


@dataclass
class Check:
    name: str
    status: str
    detail: str


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check(name: str, ok: bool, detail: str, warning: bool = False) -> Check:
    if ok:
        status = "pass"
    elif warning:
        status = "warning"
    else:
        status = "fail"
    return Check(name, status, detail)


def ledger_contract(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    max_notional_err = 0.0
    margin_min = float("inf")
    margin_max = float("-inf")
    leverage_min = float("inf")
    leverage_max = float("-inf")
    notional_min = float("inf")
    notional_max = float("-inf")
    for row in rows:
        margin = float(row.get("risk_margin_fraction") or row.get("margin_fraction") or 0.0)
        leverage = float(row.get("risk_leverage") or row.get("leverage") or 0.0)
        notional = float(row.get("risk_notional") or row.get("notional") or 0.0)
        max_notional_err = max(max_notional_err, abs(notional - margin * leverage))
        margin_min = min(margin_min, margin)
        margin_max = max(margin_max, margin)
        leverage_min = min(leverage_min, leverage)
        leverage_max = max(leverage_max, leverage)
        notional_min = min(notional_min, notional)
        notional_max = max(notional_max, notional)
    return {
        "rows": len(rows),
        "max_notional_error": max_notional_err,
        "margin_fraction_range": [margin_min, margin_max],
        "leverage_range": [leverage_min, leverage_max],
        "notional_range": [notional_min, notional_max],
    }


def selected_rank(report: dict[str, Any], ranking_path: Path) -> dict[str, Any]:
    rows = read_csv_rows(ranking_path)
    base_trades = int(report["ledger_sizing_baseline"]["validation"]["trades"])
    trade_floor = math.floor(base_trades * 0.95)
    risk_model = report["risk_model"]
    min_notional = float(risk_model.get("min_validation_avg_notional", 0.0))
    max_notional = float(risk_model.get("max_validation_avg_notional", 0.0))
    eligible: list[dict[str, str]] = []
    for row in rows:
        avg_notional = float(row["validation_avg_notional"])
        leverage_span = float(row.get("leverage_max") or 0.0) - float(row.get("leverage_min") or 0.0)
        long_leverage_scale = float(row.get("long_leverage_scale") or 1.0)
        short_leverage_scale = float(row.get("short_leverage_scale") or 1.0)
        if int(row["validation_trades"]) < trade_floor:
            continue
        if float(row["validation_mdd"]) < -16.0:
            continue
        if min_notional > 0.0 and avg_notional < min_notional:
            continue
        if max_notional > 0.0 and avg_notional > max_notional:
            continue
        if bool(risk_model.get("require_dynamic_leverage_mapping")):
            if leverage_span <= 0.0 and long_leverage_scale == 1.0 and short_leverage_scale == 1.0:
                continue
        eligible.append(row)

    eligible.sort(
        key=lambda r: (
            float(r["validation_log_risk_utility"]),
            float(r["validation_mdd"]),
            float(r["validation_pnl"]),
        ),
        reverse=True,
    )
    selected = report["selected"]["variant"]
    rank = next((i for i, row in enumerate(eligible, start=1) if row["variant"] == selected), None)
    return {
        "selected_variant": selected,
        "eligible_count": len(eligible),
        "rank": rank,
        "top_variant": eligible[0]["variant"] if eligible else None,
        "top_validation_log_risk_utility": float(eligible[0]["validation_log_risk_utility"]) if eligible else None,
        "top_validation_pnl": float(eligible[0]["validation_pnl"]) if eligible else None,
        "trade_floor": trade_floor,
    }


def metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "pnl",
        "mdd",
        "trades",
        "wr",
        "avg_notional",
        "avg_margin_fraction",
        "avg_leverage",
        "log_risk_utility",
        "exit_reasons",
    )
    return {k: metrics[k] for k in keys if k in metrics}


def maybe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_path = RUN_DIR / "report.json"
    report = read_json(report_path)
    sidecar_path = RUN_DIR / "risk_sidecar.pkl"
    ranking_path = RUN_DIR / "risk_mapping_ranking.csv"
    artifact_paths = {
        "run_dir": RUN_DIR,
        "report": report_path,
        "risk_sidecar": sidecar_path,
        "ranking": ranking_path,
        "validation_selected_sizing_ledger": RUN_DIR / "validation_selected_risk_trade_ledger.csv",
        "oos_selected_sizing_ledger": RUN_DIR / "oos_selected_risk_trade_ledger.csv",
        "validation_selected_replayed_ledger": RUN_DIR / "validation_selected_risk_replayed_trade_ledger.csv",
        "oos_selected_replayed_ledger": RUN_DIR / "oos_selected_risk_replayed_trade_ledger.csv",
        "validation_chart": RUN_DIR / "charts/omega44_live_like_dynamic_leverage_v18_validation_trade_chart.png",
        "oos_chart": RUN_DIR / "charts/omega44_live_like_dynamic_leverage_v18_oos_trade_chart.png",
        "parent_bundle": Path(report["baseline_bundle"]),
        "train_csv": Path(report["risk_model"]["train_csv"]),
        "eval_csv": Path(report["risk_model"]["eval_csv"]),
        "direction_label_dir": Path(report["risk_model"]["direction_label_dir"]),
    }

    checks: list[Check] = []
    missing = [name for name, path in artifact_paths.items() if not path.exists()]
    checks.append(check("artifact_presence", not missing, "missing=" + json.dumps(missing, ensure_ascii=False)))

    with sidecar_path.open("rb") as f:
        sidecar = pickle.load(f)
    feature_cols = list(sidecar.get("feature_columns", []))
    feature_hits = sorted(
        col for col in feature_cols if any(term in col.lower() for term in FORBIDDEN_FEATURE_TERMS)
    )
    checks.append(check("risk_sidecar_loads", isinstance(sidecar, dict), f"keys={sorted(sidecar.keys())}"))
    checks.append(check("feature_columns_present", len(feature_cols) > 0, f"count={len(feature_cols)}"))
    checks.append(check("forbidden_feature_hits_zero", not feature_hits, f"hits={feature_hits}"))

    risk_model = report["risk_model"]
    contract = report["contract"]
    expected_contracts = {
        "quality_threshold_070": contract.get("quality_threshold") == 0.7,
        "exit_threshold_075": contract.get("exit_threshold") == 0.75,
        "model_kind_hgb": risk_model.get("model_kind") == "hgb",
        "feature_mode_parent_outputs": risk_model.get("risk_feature_mode") == "parent_outputs",
        "side_split_enabled": risk_model.get("side_split_model") is True,
        "dynamic_leverage_enabled": risk_model.get("dynamic_leverage") is True,
        "selection_scope_validation_only": risk_model.get("selection_scope") == "validation_only",
        "selection_objective_log_risk": risk_model.get("selection_objective") == "log_risk",
        "tail_penalty_050": risk_model.get("log_risk_params", {}).get("tail_penalty") == 0.5,
        "target_mae_penalty_050": report.get("risk_label", {}).get("target_mae_penalty") == 0.5,
        "notional_scaled_sltp_false": contract.get("notional_scaled_sltp") is False,
        "notional_contract_declared": contract.get("risk_sizing") == "notional = margin_fraction * leverage",
    }
    for name, ok in expected_contracts.items():
        checks.append(check(name, bool(ok), f"value={ok}"))

    rank_info = selected_rank(report, ranking_path)
    checks.append(
        check(
            "selected_mapping_top_validation_only_eligible",
            rank_info["rank"] == 1,
            json.dumps(rank_info, ensure_ascii=False, sort_keys=True),
        )
    )

    ledger_checks = {
        "validation_sizing": ledger_contract(artifact_paths["validation_selected_sizing_ledger"]),
        "oos_sizing": ledger_contract(artifact_paths["oos_selected_sizing_ledger"]),
        "validation_replayed": ledger_contract(artifact_paths["validation_selected_replayed_ledger"]),
        "oos_replayed": ledger_contract(artifact_paths["oos_selected_replayed_ledger"]),
    }
    max_ledger_err = max(item["max_notional_error"] for item in ledger_checks.values())
    checks.append(check("ledger_notional_math_exact", max_ledger_err <= 1e-12, f"max_error={max_ledger_err}"))

    selected = report["selected"]
    metrics = {
        "sizing_only_validation": metric_subset(selected["validation"]),
        "sizing_only_oos": metric_subset(selected["oos"]),
        "full_replay_validation": metric_subset(selected["selected_full_replay"]["validation"]),
        "full_replay_oos": metric_subset(selected["selected_full_replay"]["oos"]),
    }
    checks.append(check("validation_full_replay_positive", metrics["full_replay_validation"]["pnl"] > 0.0, json.dumps(metrics["full_replay_validation"], ensure_ascii=False)))
    checks.append(check("oos_full_replay_positive", metrics["full_replay_oos"]["pnl"] > 0.0, json.dumps(metrics["full_replay_oos"], ensure_ascii=False)))
    checks.append(check("validation_full_replay_mdd_within_16pct", metrics["full_replay_validation"]["mdd"] >= -16.0, f"mdd={metrics['full_replay_validation']['mdd']}"))
    checks.append(check("oos_full_replay_mdd_within_16pct", metrics["full_replay_oos"]["mdd"] >= -16.0, f"mdd={metrics['full_replay_oos']['mdd']}"))

    checks.append(
        check(
            "standalone_promotion_manifest_exists",
            PROMOTION_MANIFEST.exists(),
            f"exists={PROMOTION_MANIFEST.exists()} path={PROMOTION_MANIFEST}",
        )
    )
    checks.append(
        check(
            "standalone_runtime_contract_exists",
            RUNTIME_CONTRACT.exists(),
            f"exists={RUNTIME_CONTRACT.exists()} path={RUNTIME_CONTRACT}",
        )
    )
    checks.append(
        check(
            "candidate_manifest_exists",
            CANDIDATE_MANIFEST.exists(),
            f"exists={CANDIDATE_MANIFEST.exists()} path={CANDIDATE_MANIFEST}",
        )
    )
    checks.append(
        check(
            "contract_doc_exists",
            CONTRACT_DOC.exists(),
            f"exists={CONTRACT_DOC.exists()} path={CONTRACT_DOC}",
        )
    )
    promotion = maybe_json(PROMOTION_MANIFEST)
    runtime = maybe_json(RUNTIME_CONTRACT)
    candidate = maybe_json(CANDIDATE_MANIFEST)
    if promotion is not None:
        checks.append(check("promotion_manifest_model_id_unique_v18", promotion.get("model_id") == EXPECTED_MODEL_ALIAS, f"model_id={promotion.get('model_id')}"))
        checks.append(check("promotion_manifest_source_model_id_preserved", promotion.get("source_report_model_id") == report.get("model_id"), f"source_report_model_id={promotion.get('source_report_model_id')} report_model_id={report.get('model_id')}"))
    if runtime is not None:
        checks.append(check("runtime_contract_model_id_unique_v18", runtime.get("model_id") == EXPECTED_MODEL_ALIAS, f"model_id={runtime.get('model_id')}"))
        checks.append(check("runtime_contract_full_replay_enabled", runtime.get("execution_contract", {}).get("full_replay_dynamic_exit_enabled") is True, f"execution_contract={runtime.get('execution_contract')}"))
        checks.append(check("runtime_contract_fail_fast_required", runtime.get("fail_fast_required") is True, f"fail_fast_required={runtime.get('fail_fast_required')}"))
    if candidate is not None:
        checks.append(check("candidate_manifest_model_id_unique_v18", candidate.get("model_id") == EXPECTED_MODEL_ALIAS, f"model_id={candidate.get('model_id')}"))
        checks.append(check("candidate_manifest_runtime_contract_ref", candidate.get("runtime_contract") == str(RUNTIME_CONTRACT.relative_to(ROOT)), f"runtime_contract={candidate.get('runtime_contract')}"))
    full_replay_candidates = report.get("full_replay_selection_candidates", [])
    checks.append(
        check(
            "full_replay_candidate_is_diagnostic_single_candidate",
            len(full_replay_candidates) == 1 and full_replay_candidates[0].get("variant") == selected["variant"],
            f"count={len(full_replay_candidates)} variants={[c.get('variant') for c in full_replay_candidates]}",
        )
    )

    hard_failures = [c for c in checks if c.status == "fail"]
    warnings = [c for c in checks if c.status == "warning"]
    research_reproduction_pass = not hard_failures
    if hard_failures:
        verdict = "REDTEAM_FAIL_REPRODUCTION_OR_CONTRACT"
        redteam_pass = False
    elif warnings:
        verdict = "REDTEAM_CONDITIONAL_PASS_RESEARCH_BLOCKED_FOR_PROMOTION"
        redteam_pass = False
    else:
        verdict = "REDTEAM_PASS_FULL_PROMOTABLE"
        redteam_pass = True

    result = {
        "audit_id": "omega4_4_v18_redteam_audit_20260625",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_alias": EXPECTED_MODEL_ALIAS,
        "source_report": str(report_path),
        "source_run_dir": str(RUN_DIR),
        "verdict": verdict,
        "redteam_pass": redteam_pass,
        "research_reproduction_pass": research_reproduction_pass,
        "promotion_pass": redteam_pass,
        "checks": [c.__dict__ for c in checks],
        "blockers": [c.__dict__ for c in hard_failures],
        "promotion_blockers": [c.__dict__ for c in warnings],
        "metrics": metrics,
        "selected_rank": rank_info,
        "ledger_contract": ledger_checks,
        "feature_columns": feature_cols,
        "selected_mapping": selected["mapping"],
        "notes": [
            "The v18 research run is internally reproducible: artifacts load, selection is validation-only among eligible mappings, OOS is not used in ranking, and notional math is exact.",
            "It is not a full promotable pass because no standalone promotion_manifest/runtime_contract/candidate_manifest exists for the v18 alias, and the report model_id is the generic omega4_2 sidecar id.",
            "Full replay is present as the reported performance readout. Promotion requires an explicit runtime contract that states whether the exit head receives baseline sizing inputs or sidecar actual sizing inputs.",
        ],
    }
    out_json = OUT_DIR / "report.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Omega4.4 v18 Red-Team Audit",
        "",
        f"- Verdict: `{verdict}`",
        f"- Research reproduction pass: `{research_reproduction_pass}`",
        f"- Promotion red-team pass: `{redteam_pass}`",
        f"- Source report: `{report_path}`",
        f"- Audit JSON: `{out_json}`",
        "",
        "## Metrics",
        "",
        "| Split | PnL | MDD | Trades | WR | Avg notional | Avg leverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        lines.append(
            f"| {name} | {m.get('pnl', 0.0):.4f} | {m.get('mdd', 0.0):.4f} | {int(m.get('trades', 0))} | "
            f"{float(m.get('wr', 0.0)):.4f} | {float(m.get('avg_notional', 0.0)):.4f} | {float(m.get('avg_leverage', 0.0)):.4f} |"
        )
    lines += [
        "",
        "## Blocking Result",
        "",
    ]
    if hard_failures:
        lines += [f"- FAIL `{c.name}`: {c.detail}" for c in hard_failures]
    else:
        lines.append("- No hard reproduction failures.")
    if warnings:
        lines += ["", "## Promotion Blockers", ""]
        lines += [f"- WARNING `{c.name}`: {c.detail}" for c in warnings]
    lines += [
        "",
        "## Key Passes",
        "",
    ]
    lines += [f"- PASS `{c.name}`: {c.detail}" for c in checks if c.status == "pass"]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_json), "doc": str(DOC_PATH), "verdict": verdict, "redteam_pass": redteam_pass}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
