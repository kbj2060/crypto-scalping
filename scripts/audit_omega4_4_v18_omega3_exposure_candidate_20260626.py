#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_ID = "omega4_4_v18_baseline_20260624"
SOURCE_MODEL_ID = "omega3_aggressive_compensated_scale200_cap090_20260618"
DEFAULT_SWEEP_REPORT = ROOT / "tmp/causal_regen_20260516/omega44_v18_omega3_exposure_fine_sweep_20260626/report.json"
BASE_CANDIDATE_MANIFEST = ROOT / f"data/ensemble/supervised/{BASE_MODEL_ID}/candidate_manifest.json"
BASE_RUNTIME_CONTRACT = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624"
    / "runtime_contract.json"
)
BASE_REDTEAM_JSON = ROOT / "tmp/causal_regen_20260516/omega4_4_v18_redteam_audit_20260625/report.json"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


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


def close_enough(a: Any, b: Any, tol: float = 1.0e-9) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return a == b


def metric_subset(row: dict[str, Any], split: str) -> dict[str, Any]:
    keys = ("pnl", "mdd", "trades", "wr", "overlay_hits", "avg_notional", "avg_margin_fraction", "avg_leverage", "log_risk_utility", "exit_reasons")
    return {key: row[f"{split}_{key}"] for key in keys if f"{split}_{key}" in row}


def ledger_contract(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    max_notional_err = 0.0
    partial_done = 0
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
        partial_done += int(str(row.get("partial_done", "")).lower() == "true")
    return {
        "rows": len(rows),
        "partial_done": partial_done,
        "max_notional_error": max_notional_err,
        "margin_fraction_range": [margin_min, margin_max],
        "leverage_range": [leverage_min, leverage_max],
        "notional_range": [notional_min, notional_max],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    model_id = args.model_id
    run_dir = ROOT / f"tmp/causal_regen_20260516/{model_id}"
    report_json = run_dir / "redteam_report.json"
    doc_path = ROOT / f"docs/audits/{model_id}_redteam.md"
    candidate_manifest_path = ROOT / f"data/ensemble/supervised/{model_id}/candidate_manifest.json"
    promotion_manifest_path = run_dir / "promotion_manifest.json"
    runtime_contract_path = run_dir / "runtime_contract.json"
    contract_doc_path = ROOT / f"docs/model_contracts/{model_id}_contract.md"

    checks: list[Check] = []
    paths = {
        "candidate_manifest": candidate_manifest_path,
        "promotion_manifest": promotion_manifest_path,
        "runtime_contract": runtime_contract_path,
        "contract_doc": contract_doc_path,
        "base_candidate_manifest": BASE_CANDIDATE_MANIFEST,
        "base_runtime_contract": BASE_RUNTIME_CONTRACT,
        "base_redteam_json": BASE_REDTEAM_JSON,
        "fine_sweep_report": DEFAULT_SWEEP_REPORT,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    checks.append(check("artifact_presence", not missing, "missing=" + json.dumps(missing, ensure_ascii=False)))
    if missing:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(
            json.dumps(
                {
                    "audit_id": f"{model_id}_redteam",
                    "verdict": "REDTEAM_FAIL_CONTRACT_OR_REPRODUCTION",
                    "redteam_pass": False,
                    "blockers": [checks[-1].__dict__],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 1

    manifest = read_json(candidate_manifest_path)
    promotion = read_json(promotion_manifest_path)
    runtime = read_json(runtime_contract_path)
    sweep_report_path = ROOT / str(manifest.get("fine_sweep_report", rel(DEFAULT_SWEEP_REPORT)))
    paths["fine_sweep_report"] = sweep_report_path
    base_manifest = read_json(BASE_CANDIDATE_MANIFEST)
    base_runtime = read_json(BASE_RUNTIME_CONTRACT)
    base_redteam = read_json(BASE_REDTEAM_JSON)
    fine_report = read_json(sweep_report_path)
    baseline = fine_report["baseline"]
    variant = str(manifest["variant"])
    top_rows = [row for row in fine_report["top20"] if row.get("variant") == variant]
    selected_rows = [row for row in fine_report["selected"].values() if isinstance(row, dict) and row.get("variant") == variant]
    fine_row = selected_rows[0] if selected_rows else top_rows[0] if top_rows else None

    validation_ledger = ROOT / manifest["ledgers"]["validation"]
    oos_ledger = ROOT / manifest["ledgers"]["oos"]
    paths["validation_ledger"] = validation_ledger
    paths["oos_ledger"] = oos_ledger
    checks.append(check("ledger_presence", validation_ledger.exists() and oos_ledger.exists(), f"validation={validation_ledger.exists()} oos={oos_ledger.exists()}"))

    checks.append(check("base_model_id", manifest.get("base_model_id") == BASE_MODEL_ID, f"base_model_id={manifest.get('base_model_id')}"))
    checks.append(check("source_model_id", manifest.get("source_model_id") == SOURCE_MODEL_ID, f"source_model_id={manifest.get('source_model_id')}"))
    checks.append(check("base_redteam_full_pass", base_redteam.get("verdict") == "REDTEAM_PASS_FULL_PROMOTABLE" and base_redteam.get("redteam_pass") is True, f"base_verdict={base_redteam.get('verdict')} pass={base_redteam.get('redteam_pass')}"))
    checks.append(check("candidate_manifest_model_id", manifest.get("model_id") == model_id, f"model_id={manifest.get('model_id')}"))
    checks.append(check("promotion_manifest_model_id", promotion.get("model_id") == model_id, f"model_id={promotion.get('model_id')}"))
    checks.append(check("runtime_contract_model_id", runtime.get("model_id") == model_id, f"model_id={runtime.get('model_id')}"))
    checks.append(check("runtime_fail_fast_required", runtime.get("fail_fast_required") is True, f"fail_fast_required={runtime.get('fail_fast_required')}"))
    checks.append(check("base_runtime_not_mutated", base_runtime.get("model_id") == BASE_MODEL_ID, f"base_runtime_model_id={base_runtime.get('model_id')}"))
    checks.append(check("base_manifest_redteam_pass", base_manifest.get("redteam_pass") is True, f"base_manifest_redteam_pass={base_manifest.get('redteam_pass')}"))
    checks.append(check("fine_sweep_variant_found", fine_row is not None, f"variant={variant}"))

    risk_remap = runtime.get("risk_remap", {})
    checks.append(check("risk_remap_enabled", risk_remap.get("enabled") is True, f"risk_remap={risk_remap}"))
    checks.append(check("risk_remap_leverage_fixed_2", close_enough(risk_remap.get("leverage"), 2.0), f"leverage={risk_remap.get('leverage')}"))
    checks.append(check("risk_remap_notional_contract", risk_remap.get("notional_math") == "notional = margin_fraction * leverage", f"notional_math={risk_remap.get('notional_math')}"))
    checks.append(check("sltp_no_double_leverage", "not multiplied twice" in str(risk_remap.get("sltp_contract", "")), f"sltp_contract={risk_remap.get('sltp_contract')}"))

    overlay = runtime.get("lifecycle_overlay", {})
    if manifest.get("lifecycle_overlay", {}).get("enabled"):
        expected_overlay = {
            "enabled": True,
            "mode": "short_aged_profit_partial_deleverage",
            "side_value": -1,
            "cap_bars": 1152,
            "min_unrealized_price_move": 0.035,
            "partial_fraction": 0.5,
            "fires_once_per_position": True,
        }
        for key, expected in expected_overlay.items():
            checks.append(check(f"overlay_{key}", overlay.get(key) == expected, f"value={overlay.get(key)!r} expected={expected!r}"))

    for split, manifest_key in (("validation", "candidate_validation"), ("oos", "candidate_oos_readout")):
        m = manifest[manifest_key]
        for key in ("pnl", "mdd", "trades", "wr", "avg_notional", "avg_margin_fraction", "avg_leverage", "overlay_hits", "log_risk_utility"):
            if fine_row is not None:
                report_value = fine_row[f"{split}_{key}"]
                checks.append(check(f"manifest_{split}_{key}_matches_fine_sweep", close_enough(m[key], report_value), f"manifest={m[key]} fine={report_value}"))

    checks.append(check("validation_pnl_improves", manifest["candidate_validation"]["pnl"] > baseline["validation_pnl"], f"candidate={manifest['candidate_validation']['pnl']} baseline={baseline['validation_pnl']}"))
    checks.append(check("oos_pnl_improves_diagnostic", manifest["candidate_oos_readout"]["pnl"] > baseline["oos_pnl"], f"candidate={manifest['candidate_oos_readout']['pnl']} baseline={baseline['oos_pnl']}"))
    checks.append(check("validation_mdd_improves_for_strict_promotion", manifest["candidate_validation"]["mdd"] >= baseline["validation_mdd"], f"candidate={manifest['candidate_validation']['mdd']} baseline={baseline['validation_mdd']}", warning=True))
    checks.append(check("oos_mdd_improves_diagnostic", manifest["candidate_oos_readout"]["mdd"] >= baseline["oos_mdd"], f"candidate={manifest['candidate_oos_readout']['mdd']} baseline={baseline['oos_mdd']}", warning=True))

    ledger = {}
    if validation_ledger.exists() and oos_ledger.exists():
        ledger = {
            "validation": ledger_contract(validation_ledger),
            "oos": ledger_contract(oos_ledger),
        }
        max_err = max(item["max_notional_error"] for item in ledger.values())
        checks.append(check("ledger_notional_math_exact", max_err <= 1e-12, f"max_error={max_err}"))
        checks.append(check("validation_partial_done_matches_hits", ledger["validation"]["partial_done"] == int(manifest["candidate_validation"]["overlay_hits"]), f"partial_done={ledger['validation']['partial_done']} hits={manifest['candidate_validation']['overlay_hits']}"))
        checks.append(check("oos_partial_done_matches_hits", ledger["oos"]["partial_done"] == int(manifest["candidate_oos_readout"]["overlay_hits"]), f"partial_done={ledger['oos']['partial_done']} hits={manifest['candidate_oos_readout']['overlay_hits']}"))

    selection = runtime.get("candidate_selection", {})
    checks.append(check("selection_oos_informed_declared", selection.get("selection_oos_informed") is True, f"selection={selection}"))
    checks.append(
        check(
            "clean_oos_promotion_blocker",
            selection.get("selection_oos_informed") is False,
            "candidate was chosen after OOS diagnostic comparison; fresh holdout/walk-forward is required before clean-OOS promotion",
            warning=True,
        )
    )

    hard_failures = [c for c in checks if c.status == "fail"]
    warnings = [c for c in checks if c.status == "warning"]
    research_reproduction_pass = not hard_failures
    clean_oos_promotion_pass = not hard_failures and not warnings
    redteam_pass = research_reproduction_pass
    promotion_pass = clean_oos_promotion_pass
    if hard_failures:
        verdict = "REDTEAM_FAIL_CONTRACT_OR_REPRODUCTION"
    elif warnings:
        verdict = "REDTEAM_PASS_RESEARCH_CANDIDATE_CLEAN_OOS_PROMOTION_BLOCKED"
    else:
        verdict = "REDTEAM_PASS_FULL_PROMOTABLE"

    metrics = {
        "baseline_validation": metric_subset(baseline, "validation"),
        "candidate_validation": manifest["candidate_validation"],
        "baseline_oos_readout": metric_subset(baseline, "oos"),
        "candidate_oos_readout": manifest["candidate_oos_readout"],
    }
    result = {
        "audit_id": f"{model_id}_redteam",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "base_model_id": BASE_MODEL_ID,
        "source_model_id": SOURCE_MODEL_ID,
        "variant": variant,
        "verdict": verdict,
        "redteam_pass": redteam_pass,
        "research_reproduction_pass": research_reproduction_pass,
        "clean_oos_promotion_pass": clean_oos_promotion_pass,
        "promotion_pass": promotion_pass,
        "checks": [c.__dict__ for c in checks],
        "blockers": [c.__dict__ for c in hard_failures],
        "promotion_blockers": [c.__dict__ for c in warnings],
        "metrics": metrics,
        "ledger_contract": ledger,
        "artifacts": {name: rel(path) for name, path in paths.items()},
        "notes": [
            "The risk remap and lifecycle overlay contracts are explicit and fail-fast; the base v18 runtime and artifacts remain unchanged.",
            "PnL and OOS readout are diagnostics here because the candidate was selected after a fine sweep that included OOS readout.",
            "Clean-OOS promotion requires a fresh holdout or walk-forward confirmation.",
        ],
    }
    report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Omega4.4 v18 Omega3 Exposure Transfer Red-Team Audit",
        "",
        f"- Verdict: `{verdict}`",
        f"- Red-team reproduction pass: `{redteam_pass}`",
        f"- Clean-OOS promotion pass: `{clean_oos_promotion_pass}`",
        f"- Model id: `{model_id}`",
        f"- Variant: `{variant}`",
        f"- Audit JSON: `{rel(report_json)}`",
        "",
        "## Metrics",
        "",
        "| Split | PnL | MDD | Trades | WR | Avg notional | Overlay hits | Log-risk utility |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in metrics.items():
        lines.append(
            f"| {label} | {row.get('pnl', 0.0):.4f} | {row.get('mdd', 0.0):.4f} | {int(row.get('trades', 0))} | "
            f"{float(row.get('wr', 0.0)):.4f} | {float(row.get('avg_notional', 0.0)):.4f} | "
            f"{int(row.get('overlay_hits', 0))} | {float(row.get('log_risk_utility', 0.0)):.6f} |"
        )
    lines += ["", "## Blockers", ""]
    if hard_failures:
        lines += [f"- FAIL `{c.name}`: {c.detail}" for c in hard_failures]
    else:
        lines.append("- No hard contract or reproduction blockers.")
    if warnings:
        lines += ["", "## Promotion Blockers", ""]
        lines += [f"- WARNING `{c.name}`: {c.detail}" for c in warnings]
    lines += ["", "## Key Passes", ""]
    lines += [f"- PASS `{c.name}`: {c.detail}" for c in checks if c.status == "pass"]
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(report_json),
                "doc": str(doc_path),
                "verdict": verdict,
                "redteam_pass": redteam_pass,
                "clean_oos_promotion_pass": clean_oos_promotion_pass,
                "blockers": len(hard_failures),
                "promotion_blockers": len(warnings),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
