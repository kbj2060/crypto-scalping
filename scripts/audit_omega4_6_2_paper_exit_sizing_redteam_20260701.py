#!/usr/bin/env python3
"""Red-team audit for the Omega 4.6.2 paper exit+sizing upgrade candidate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega4_6_2_paper_optstop_exit_sizing_overlay_20260701"
BASE_MODEL_ID = "omega4_6_2_cap220_short_boost125_time_stop120h_20260630"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REPORT = OUT_DIR / "report.json"
RANKING = OUT_DIR / "paper_exit_sizing_ranking.csv"
EVAL_SCRIPT = ROOT / "scripts/eval_omega4_6_2_paper_exit_sizing_stopping_20260701.py"
PRIOR_RUNTIME_AUDIT = (
    ROOT / "docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json"
)
AUDIT_JSON = OUT_DIR / "redteam_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_paper_exit_sizing_redteam_20260701.md"
EPS = 1.0e-12
TOL = 1.0e-8


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(type(obj).__name__)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def active_ledger(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["notional"].astype(float) > EPS].copy()


def ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entry_timestamp_dt"] = pd.to_datetime(out["entry_timestamp"], errors="raise")
    out["exit_timestamp_dt"] = pd.to_datetime(out["exit_timestamp"], errors="raise")
    if "hold_hours" not in out.columns:
        out["hold_hours"] = (
            out["exit_timestamp_dt"] - out["entry_timestamp_dt"]
        ).dt.total_seconds() / 3600.0
    return out


def overlap_count(df: pd.DataFrame) -> int:
    active = active_ledger(df)
    if len(active) <= 1:
        return 0
    ordered = active.sort_values(["entry_i", "exit_i"]).reset_index(drop=True)
    prev_exit = -1
    overlaps = 0
    for _, row in ordered.iterrows():
        entry_i = int(row["entry_i"])
        exit_i = int(row["exit_i"])
        if entry_i <= prev_exit:
            overlaps += 1
        prev_exit = max(prev_exit, exit_i)
    return overlaps


def metrics(df: pd.DataFrame) -> dict[str, Any]:
    df = ensure_time_columns(df)
    active = active_ledger(df)
    if active.empty:
        return {
            "pnl": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "wr": 0.0,
            "avg_hold_hours": 0.0,
            "max_hold_hours": 0.0,
            "hold_over_24h_count": 0,
            "max_leverage": 0.0,
            "avg_notional": 0.0,
            "max_notional": 0.0,
            "max_margin_fraction": 0.0,
            "skipped": int(len(df)),
            "overlap_count": 0,
            "accounting_error_max_abs": 0.0,
            "notional_contract_error_max_abs": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "reason_counts": {},
        }
    returns = active["trade_return"].astype(float).to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    drawdown = curve / np.maximum(peak, EPS) - 1.0
    accounting_error = (
        active["trade_return"].astype(float)
        - active["net_per_notional"].astype(float) * active["notional"].astype(float)
    ).abs()
    notional_contract_error = (
        active["notional"].astype(float)
        - active["margin_fraction"].astype(float) * active["leverage"].astype(float)
    ).abs()
    hold_hours = active["hold_hours"].astype(float)
    return {
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(np.min(drawdown) * 100.0),
        "trades": int(len(active)),
        "wr": float((active["trade_return"].astype(float) > 0.0).mean()),
        "avg_hold_hours": float(hold_hours.mean()),
        "max_hold_hours": float(hold_hours.max()),
        "hold_over_24h_count": int((hold_hours > 24.0 + 1.0e-9).sum()),
        "max_leverage": float(active["leverage"].astype(float).max()),
        "avg_notional": float(active["notional"].astype(float).mean()),
        "max_notional": float(active["notional"].astype(float).max()),
        "max_margin_fraction": float(active["margin_fraction"].astype(float).max()),
        "skipped": int((df["notional"].astype(float) <= EPS).sum()),
        "overlap_count": int(overlap_count(df)),
        "accounting_error_max_abs": float(accounting_error.max()),
        "notional_contract_error_max_abs": float(notional_contract_error.max()),
        "long_trades": int((active["side"].astype(int) > 0).sum()),
        "short_trades": int((active["side"].astype(int) < 0).sum()),
        "reason_counts": {
            str(k): int(v)
            for k, v in active["reason"].value_counts().sort_index().to_dict().items()
        },
    }


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    severity: str,
    details: dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "name": name,
            "pass": bool(passed),
            "severity": severity,
            "details": details or {},
        }
    )


def compare_metric_block(
    checks: list[dict[str, Any]],
    split: str,
    observed: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        "pnl",
        "mdd",
        "trades",
        "wr",
        "avg_hold_hours",
        "max_hold_hours",
        "hold_over_24h_count",
        "max_leverage",
        "avg_notional",
        "max_notional",
        "skipped",
        "overlap_count",
        "accounting_error_max_abs",
        "notional_contract_error_max_abs",
        "long_trades",
        "short_trades",
    ]
    rows: dict[str, Any] = {}
    passed = True
    for key in keys:
        report_key = f"{split}_{key}"
        obs = observed[key]
        exp = selected[report_key]
        diff = float(obs) - float(exp)
        ok = abs(diff) <= TOL
        passed = passed and ok
        rows[key] = {"observed": obs, "expected": exp, "diff": diff, "pass": ok}

    reason_raw = selected.get(f"{split}_reason_counts", "{}")
    expected_reasons = json.loads(reason_raw) if isinstance(reason_raw, str) else reason_raw
    reason_ok = observed["reason_counts"] == expected_reasons
    rows["reason_counts"] = {
        "observed": observed["reason_counts"],
        "expected": expected_reasons,
        "pass": reason_ok,
    }
    passed = passed and reason_ok

    add_check(
        checks,
        f"{split}_ledger_report_metric_parity",
        passed,
        "research_blocker",
        {"checks": rows, "tolerance": TOL},
    )
    return rows


def prior_runtime_status() -> dict[str, Any]:
    if not PRIOR_RUNTIME_AUDIT.exists():
        return {
            "available": False,
            "overall": {},
            "fresh_holdout_walkforward": {},
        }
    audit = read_json(PRIOR_RUNTIME_AUDIT)
    return {
        "available": True,
        "overall": audit.get("overall", {}),
        "fresh_holdout_walkforward": audit.get("fresh_holdout_walkforward", {}),
    }


def write_markdown(payload: dict[str, Any]) -> None:
    selected = payload["selected_variant"]
    baseline = payload["baseline"]
    observed = payload["observed"]
    blockers = [c for c in payload["checks"] if not c["pass"] and c["severity"] == "live_blocker"]
    research_fails = [
        c for c in payload["checks"] if not c["pass"] and c["severity"] == "research_blocker"
    ]
    warnings = [c for c in payload["checks"] if not c["pass"] and c["severity"] == "warning"]

    lines = [
        "# Omega 4.6.2 Paper Exit + Sizing Red-Team Audit - 2026-07-01",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Base: `{payload['base_model_id']}`",
        f"- Verdict: `{payload['verdict']}`",
        f"- Research upgrade pass: `{payload['research_upgrade_pass']}`",
        f"- Full live pass: `{payload['full_live_pass']}`",
        "",
        "## Selected Candidate",
        "",
        f"- Stop spec: `{selected['stop_spec']}`",
        f"- Exposure spec: `{selected['exposure_spec']}`",
        f"- Selection scope: `{payload['selection_scope']}`",
        "",
        "| Split | Baseline PnL | Candidate PnL | Baseline MDD | Candidate MDD | Baseline Avg Hold | Candidate Avg Hold | Baseline Max Hold | Candidate Max Hold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["validation", "oos"]:
        base = baseline[split]
        cand = observed[split]
        lines.append(
            "| "
            f"{split} | "
            f"{base['pnl']:.4f}% | {cand['pnl']:.4f}% | "
            f"{base['mdd']:.4f}% | {cand['mdd']:.4f}% | "
            f"{base['avg_hold_hours']:.4f}h | {cand['avg_hold_hours']:.4f}h | "
            f"{base['max_hold_hours']:.4f}h | {cand['max_hold_hours']:.4f}h |"
        )

    lines.extend(["", "## Blocking Items", ""])
    if not blockers:
        lines.append("- None for full live pass.")
    for check in blockers:
        lines.append(f"- `{check['name']}`: {check['details']}")

    lines.extend(["", "## Research Failures", ""])
    if not research_fails:
        lines.append("- None.")
    for check in research_fails:
        lines.append(f"- `{check['name']}`: {check['details']}")

    lines.extend(["", "## Warnings", ""])
    if not warnings:
        lines.append("- None.")
    for check in warnings:
        lines.append(f"- `{check['name']}`: {check['details']}")

    lines.extend(
        [
            "",
            "## Contract Checks",
            "",
            f"- Validation accounting error max abs: `{observed['validation']['accounting_error_max_abs']}`",
            f"- OOS accounting error max abs: `{observed['oos']['accounting_error_max_abs']}`",
            f"- Validation notional contract error max abs: `{observed['validation']['notional_contract_error_max_abs']}`",
            f"- OOS notional contract error max abs: `{observed['oos']['notional_contract_error_max_abs']}`",
            f"- Validation max leverage: `{observed['validation']['max_leverage']}`",
            f"- OOS max leverage: `{observed['oos']['max_leverage']}`",
            f"- Validation max notional: `{observed['validation']['max_notional']}`",
            f"- OOS max notional: `{observed['oos']['max_notional']}`",
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
    report = read_json(REPORT)
    artifacts = report["artifacts"]
    selected = report["selected_variant"]
    val_ledger = Path(artifacts["selected_validation_ledger"])
    oos_ledger = Path(artifacts["selected_oos_ledger"])

    checks: list[dict[str, Any]] = []
    required_artifacts = {
        "report": REPORT,
        "ranking": RANKING,
        "eval_script": EVAL_SCRIPT,
        "validation_ledger": val_ledger,
        "oos_ledger": oos_ledger,
        "prior_runtime_audit": PRIOR_RUNTIME_AUDIT,
    }
    missing = {name: str(path) for name, path in required_artifacts.items() if not path.exists()}
    add_check(checks, "required_artifacts_exist", not missing, "research_blocker", missing)
    add_check(
        checks,
        "model_id_matches",
        report.get("model_id") == MODEL_ID,
        "research_blocker",
        {"observed": report.get("model_id"), "expected": MODEL_ID},
    )
    add_check(
        checks,
        "base_model_id_matches",
        report.get("base_model_id") == BASE_MODEL_ID,
        "research_blocker",
        {"observed": report.get("base_model_id"), "expected": BASE_MODEL_ID},
    )
    add_check(
        checks,
        "selection_scope_validation_only",
        report.get("selection_scope") == "validation_only; OOS readout only",
        "research_blocker",
        {"observed": report.get("selection_scope")},
    )
    add_check(
        checks,
        "selected_validation_gate_pass",
        selected.get("validation_gate_pass") is True,
        "research_blocker",
        {"observed": selected.get("validation_gate_pass")},
    )

    validation = metrics(pd.read_csv(val_ledger))
    oos = metrics(pd.read_csv(oos_ledger))
    compare_metric_block(checks, "validation", validation, selected)
    compare_metric_block(checks, "oos", oos, selected)

    exposure_cap = float(selected["exposure_cap_notional"])
    add_check(
        checks,
        "accounting_contract_pass",
        validation["accounting_error_max_abs"] <= TOL and oos["accounting_error_max_abs"] <= TOL,
        "research_blocker",
        {
            "validation": validation["accounting_error_max_abs"],
            "oos": oos["accounting_error_max_abs"],
            "tolerance": TOL,
        },
    )
    add_check(
        checks,
        "notional_contract_pass",
        validation["notional_contract_error_max_abs"] <= TOL
        and oos["notional_contract_error_max_abs"] <= TOL,
        "research_blocker",
        {
            "validation": validation["notional_contract_error_max_abs"],
            "oos": oos["notional_contract_error_max_abs"],
            "tolerance": TOL,
        },
    )
    add_check(
        checks,
        "no_overlapping_active_trades",
        validation["overlap_count"] == 0 and oos["overlap_count"] == 0,
        "research_blocker",
        {"validation": validation["overlap_count"], "oos": oos["overlap_count"]},
    )
    add_check(
        checks,
        "leverage_cap_5x",
        validation["max_leverage"] <= 5.0 + TOL and oos["max_leverage"] <= 5.0 + TOL,
        "research_blocker",
        {"validation": validation["max_leverage"], "oos": oos["max_leverage"]},
    )
    add_check(
        checks,
        "margin_fraction_cap_1",
        validation["max_margin_fraction"] <= 1.0 + TOL and oos["max_margin_fraction"] <= 1.0 + TOL,
        "research_blocker",
        {
            "validation": validation["max_margin_fraction"],
            "oos": oos["max_margin_fraction"],
        },
    )
    add_check(
        checks,
        "notional_within_declared_cap",
        validation["max_notional"] <= exposure_cap + TOL and oos["max_notional"] <= exposure_cap + TOL,
        "research_blocker",
        {
            "declared_cap": exposure_cap,
            "validation": validation["max_notional"],
            "oos": oos["max_notional"],
        },
    )

    baseline = report["baseline"]
    add_check(
        checks,
        "validation_pnl_and_hold_improve",
        validation["pnl"] > baseline["validation"]["pnl"]
        and validation["avg_hold_hours"] < baseline["validation"]["avg_hold_hours"]
        and validation["max_hold_hours"] < baseline["validation"]["max_hold_hours"],
        "research_blocker",
        {"baseline": baseline["validation"], "candidate": validation},
    )
    add_check(
        checks,
        "oos_pnl_and_hold_improve_readout",
        oos["pnl"] > baseline["oos"]["pnl"]
        and oos["avg_hold_hours"] < baseline["oos"]["avg_hold_hours"]
        and oos["max_hold_hours"] < baseline["oos"]["max_hold_hours"],
        "research_blocker",
        {"baseline": baseline["oos"], "candidate": oos},
    )
    add_check(
        checks,
        "pnl_over_100_percent_validation_and_oos",
        validation["pnl"] >= 100.0 and oos["pnl"] >= 100.0,
        "research_blocker",
        {"validation": validation["pnl"], "oos": oos["pnl"]},
    )
    add_check(
        checks,
        "mdd_within_20_percent_validation_and_oos",
        validation["mdd"] >= -20.0 and oos["mdd"] >= -20.0,
        "research_blocker",
        {"validation": validation["mdd"], "oos": oos["mdd"]},
    )
    add_check(
        checks,
        "validation_mdd_buffer_over_1pp",
        validation["mdd"] >= -19.0,
        "warning",
        {"validation_mdd": validation["mdd"], "buffer_to_20pct": 20.0 + validation["mdd"]},
    )

    prior = prior_runtime_status()
    overall = prior["overall"]
    fresh = prior["fresh_holdout_walkforward"]
    native_status = overall.get("runtime_native_replay_status", "MISSING")
    fresh_available = bool(fresh.get("fresh_holdout_available", False))
    add_check(
        checks,
        "runtime_native_replay_complete",
        native_status.startswith("PASS"),
        "live_blocker",
        {"inherited_status": native_status, "prior_audit": str(PRIOR_RUNTIME_AUDIT)},
    )
    add_check(
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
    add_check(
        checks,
        "max_hold_24h_daytrading_requirement",
        validation["max_hold_hours"] <= 24.0 and oos["max_hold_hours"] <= 24.0,
        "live_blocker",
        {"validation": validation["max_hold_hours"], "oos": oos["max_hold_hours"]},
    )

    research_pass = all(
        check["pass"] for check in checks if check["severity"] == "research_blocker"
    )
    full_live_pass = research_pass and all(
        check["pass"] for check in checks if check["severity"] in {"research_blocker", "live_blocker"}
    )
    if full_live_pass:
        verdict = "FULL_LIVE_PASS"
    elif research_pass:
        verdict = "RESEARCH_UPGRADE_PASS_FULL_LIVE_BLOCKED"
    else:
        verdict = "REDTEAM_FAIL"

    payload = {
        "audit_id": "omega4_6_2_paper_exit_sizing_redteam_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "verdict": verdict,
        "research_upgrade_pass": research_pass,
        "full_live_pass": full_live_pass,
        "selection_scope": report.get("selection_scope"),
        "selected_variant": selected,
        "baseline": baseline,
        "observed": {"validation": validation, "oos": oos},
        "prior_runtime_audit": prior,
        "checks": checks,
        "artifacts": {
            **artifacts,
            "audit_json": str(AUDIT_JSON),
            "audit_md": str(AUDIT_MD),
        },
    }
    write_json(AUDIT_JSON, payload)
    write_markdown(payload)
    print(
        json.dumps(
            {
                "audit_json": str(AUDIT_JSON),
                "audit_md": str(AUDIT_MD),
                "verdict": verdict,
                "research_upgrade_pass": research_pass,
                "full_live_pass": full_live_pass,
            },
            ensure_ascii=False,
            default=json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
