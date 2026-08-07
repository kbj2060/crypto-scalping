#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "omega4_4_v18_short_partial_cap1152_u0035_p050_20260626"
BASE_MODEL_ID = "omega4_4_v18_baseline_20260624"
SOURCE_VARIANT = "short_partial_cap1152_u0.035_p0.50"

RUN_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
REPORT_JSON = RUN_DIR / "redteam_report.json"
DOC_PATH = ROOT / f"docs/audits/{MODEL_ID}_redteam.md"

CANDIDATE_MANIFEST = ROOT / f"data/ensemble/supervised/{MODEL_ID}/candidate_manifest.json"
PROMOTION_MANIFEST = RUN_DIR / "promotion_manifest.json"
RUNTIME_CONTRACT = RUN_DIR / "runtime_contract.json"
CONTRACT_DOC = ROOT / f"docs/model_contracts/{MODEL_ID}_contract.md"

BASE_CANDIDATE_MANIFEST = ROOT / f"data/ensemble/supervised/{BASE_MODEL_ID}/candidate_manifest.json"
BASE_RUNTIME_CONTRACT = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624"
    / "runtime_contract.json"
)
BASE_REDTEAM_JSON = ROOT / "tmp/causal_regen_20260516/omega4_4_v18_redteam_audit_20260625/report.json"

OVERLAY_DIR = ROOT / "tmp/causal_regen_20260516/omega4_4_v18_short_aged_profit_overlay_full_replay_20260625"
OVERLAY_REPORT = OVERLAY_DIR / "report.json"
OVERLAY_GRID = OVERLAY_DIR / "full_replay_overlay_results.csv"
VALIDATION_LEDGER = OVERLAY_DIR / f"validation_{SOURCE_VARIANT}_ledger.csv"
OOS_LEDGER = OVERLAY_DIR / f"oos_{SOURCE_VARIANT}_ledger.csv"


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


def find_result(report: dict[str, Any], variant: str) -> dict[str, Any]:
    for row in report["results"]:
        if row.get("variant") == variant:
            return row
    raise KeyError(f"variant not found: {variant}")


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


def metric_subset(row: dict[str, Any], split: str) -> dict[str, Any]:
    keys = ("pnl", "mdd", "trades", "wr", "overlay_hits", "log_risk_utility", "exit_reasons")
    return {key: row[f"{split}_{key}"] for key in keys if f"{split}_{key}" in row}


def close_enough(a: Any, b: Any, tol: float = 1.0e-10) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return a == b


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "candidate_manifest": CANDIDATE_MANIFEST,
        "promotion_manifest": PROMOTION_MANIFEST,
        "runtime_contract": RUNTIME_CONTRACT,
        "contract_doc": CONTRACT_DOC,
        "base_candidate_manifest": BASE_CANDIDATE_MANIFEST,
        "base_runtime_contract": BASE_RUNTIME_CONTRACT,
        "base_redteam_json": BASE_REDTEAM_JSON,
        "overlay_report": OVERLAY_REPORT,
        "overlay_grid": OVERLAY_GRID,
        "validation_ledger": VALIDATION_LEDGER,
        "oos_ledger": OOS_LEDGER,
    }
    checks: list[Check] = []
    missing = [name for name, path in paths.items() if not path.exists()]
    checks.append(check("artifact_presence", not missing, "missing=" + json.dumps(missing, ensure_ascii=False)))

    candidate_manifest = read_json(CANDIDATE_MANIFEST)
    promotion_manifest = read_json(PROMOTION_MANIFEST)
    runtime = read_json(RUNTIME_CONTRACT)
    base_manifest = read_json(BASE_CANDIDATE_MANIFEST)
    base_runtime = read_json(BASE_RUNTIME_CONTRACT)
    base_redteam = read_json(BASE_REDTEAM_JSON)
    overlay_report = read_json(OVERLAY_REPORT)
    baseline = overlay_report["baseline"]
    candidate = find_result(overlay_report, SOURCE_VARIANT)
    overlay = runtime.get("lifecycle_overlay", {})
    selection = runtime.get("candidate_selection", {})

    checks.append(check("base_model_id", candidate_manifest.get("base_model_id") == BASE_MODEL_ID, f"base_model_id={candidate_manifest.get('base_model_id')}"))
    checks.append(check("base_redteam_full_pass", base_redteam.get("verdict") == "REDTEAM_PASS_FULL_PROMOTABLE" and base_redteam.get("redteam_pass") is True, f"base_verdict={base_redteam.get('verdict')} pass={base_redteam.get('redteam_pass')}"))
    checks.append(check("candidate_manifest_model_id", candidate_manifest.get("model_id") == MODEL_ID, f"model_id={candidate_manifest.get('model_id')}"))
    checks.append(check("promotion_manifest_model_id", promotion_manifest.get("model_id") == MODEL_ID, f"model_id={promotion_manifest.get('model_id')}"))
    checks.append(check("runtime_contract_model_id", runtime.get("model_id") == MODEL_ID, f"model_id={runtime.get('model_id')}"))
    checks.append(check("runtime_fail_fast_required", runtime.get("fail_fast_required") is True, f"fail_fast_required={runtime.get('fail_fast_required')}"))
    checks.append(check("base_runtime_not_mutated", base_runtime.get("model_id") == BASE_MODEL_ID, f"base_runtime_model_id={base_runtime.get('model_id')}"))
    checks.append(check("base_manifest_redteam_pass", base_manifest.get("redteam_pass") is True, f"base_manifest_redteam_pass={base_manifest.get('redteam_pass')}"))

    expected_overlay = {
        "enabled": True,
        "mode": "short_aged_profit_partial_deleverage",
        "side": "short",
        "side_value": -1,
        "cap_bars": 1152,
        "min_unrealized_price_move": 0.035,
        "partial_fraction": 0.5,
        "fires_once_per_position": True,
    }
    for key, expected in expected_overlay.items():
        checks.append(check(f"overlay_{key}", overlay.get(key) == expected, f"value={overlay.get(key)!r} expected={expected!r}"))

    result_expected = {
        "variant": SOURCE_VARIANT,
        "mode": "partial_deleverage",
        "side": -1,
        "cap_bars": 1152,
        "min_unreal": 0.035,
        "partial_fraction": 0.5,
    }
    for key, expected in result_expected.items():
        checks.append(check(f"overlay_result_{key}", candidate.get(key) == expected, f"value={candidate.get(key)!r} expected={expected!r}"))

    for split in ("validation", "oos"):
        for key in ("pnl", "mdd", "trades", "wr", "overlay_hits", "log_risk_utility"):
            manifest_value = candidate_manifest[f"candidate_{split if split == 'validation' else 'oos_readout'}"][key]
            report_value = candidate[f"{split}_{key}"]
            checks.append(check(f"manifest_{split}_{key}_matches_report", close_enough(manifest_value, report_value), f"manifest={manifest_value} report={report_value}"))

    checks.append(check("validation_pnl_improves", candidate["validation_pnl"] > baseline["validation_pnl"], f"candidate={candidate['validation_pnl']} baseline={baseline['validation_pnl']}"))
    checks.append(check("validation_mdd_improves", candidate["validation_mdd"] >= baseline["validation_mdd"], f"candidate={candidate['validation_mdd']} baseline={baseline['validation_mdd']}"))
    checks.append(check("oos_pnl_improves_diagnostic", candidate["oos_pnl"] > baseline["oos_pnl"], f"candidate={candidate['oos_pnl']} baseline={baseline['oos_pnl']}"))
    checks.append(check("oos_mdd_improves_diagnostic", candidate["oos_mdd"] >= baseline["oos_mdd"], f"candidate={candidate['oos_mdd']} baseline={baseline['oos_mdd']}"))
    checks.append(check("validation_overlay_hits_positive", int(candidate["validation_overlay_hits"]) > 0, f"hits={candidate['validation_overlay_hits']}"))
    checks.append(check("oos_overlay_hits_positive", int(candidate["oos_overlay_hits"]) > 0, f"hits={candidate['oos_overlay_hits']}"))

    ledger = {
        "validation": ledger_contract(VALIDATION_LEDGER),
        "oos": ledger_contract(OOS_LEDGER),
    }
    max_err = max(item["max_notional_error"] for item in ledger.values())
    checks.append(check("ledger_notional_math_exact", max_err <= 1e-12, f"max_error={max_err}"))
    checks.append(check("validation_partial_done_matches_hits", ledger["validation"]["partial_done"] == int(candidate["validation_overlay_hits"]), f"partial_done={ledger['validation']['partial_done']} hits={candidate['validation_overlay_hits']}"))
    checks.append(check("oos_partial_done_matches_hits", ledger["oos"]["partial_done"] == int(candidate["oos_overlay_hits"]), f"partial_done={ledger['oos']['partial_done']} hits={candidate['oos_overlay_hits']}"))

    checks.append(
        check(
            "selection_oos_informed_declared",
            selection.get("selection_oos_informed") is True,
            f"selection={selection}",
        )
    )
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
        "candidate_validation": metric_subset(candidate, "validation"),
        "baseline_oos_readout": metric_subset(baseline, "oos"),
        "candidate_oos_readout": metric_subset(candidate, "oos"),
    }
    result = {
        "audit_id": f"{MODEL_ID}_redteam",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "source_variant": SOURCE_VARIANT,
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
            "The lifecycle overlay contract is explicit and fail-fast; the base v18 runtime and artifacts remain unchanged.",
            "The candidate improves validation PnL/MDD and OOS readout PnL/MDD versus v18 baseline in exact full replay.",
            "Because the candidate was chosen after OOS diagnostic comparison, red-team marks research reproduction as pass but blocks clean-OOS promotion until a fresh holdout or walk-forward confirmation is run.",
        ],
    }
    REPORT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Omega4.4 v18 Short Partial Lifecycle Overlay Red-Team Audit",
        "",
        f"- Verdict: `{verdict}`",
        f"- Red-team reproduction pass: `{redteam_pass}`",
        f"- Clean-OOS promotion pass: `{clean_oos_promotion_pass}`",
        f"- Model id: `{MODEL_ID}`",
        f"- Base model: `{BASE_MODEL_ID}`",
        f"- Audit JSON: `{rel(REPORT_JSON)}`",
        "",
        "## Metrics",
        "",
        "| Split | PnL | MDD | Trades | WR | Overlay hits | Log-risk utility |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in metrics.items():
        lines.append(
            f"| {label} | {row.get('pnl', 0.0):.4f} | {row.get('mdd', 0.0):.4f} | {int(row.get('trades', 0))} | "
            f"{float(row.get('wr', 0.0)):.4f} | {int(row.get('overlay_hits', 0))} | {float(row.get('log_risk_utility', 0.0)):.6f} |"
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
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(REPORT_JSON),
                "doc": str(DOC_PATH),
                "verdict": verdict,
                "redteam_pass": redteam_pass,
                "clean_oos_promotion_pass": clean_oos_promotion_pass,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
