#!/usr/bin/env python3
"""Leakage/data-contamination red-team audit for final Omega 4.6.2 frontier models."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / "omega4_6_2_frontier_leakage_redteam_20260701"
AUDIT_JSON = OUT_DIR / "frontier_leakage_redteam_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_frontier_leakage_redteam_20260701.md"
ROLL7_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701.py"
RUNTIME_BLOCKER_JSON = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_runtime_wiring_blockers_20260701"
    / "runtime_wiring_blockers_20260701.json"
)
CVP_FEATURE_AUDIT_JSON = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "cvp_feature_causality_20260701"
    / "cvp_feature_causality_20260701.json"
)

FRONTIER_MODELS = [
    {
        "model_id": "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701",
        "label": "validation_only_final_frontier",
        "declared_risk": "validation-only exposure selection; OOS readout after selection",
        "expected_oos_selection": "readout_after_selection",
        "entry_features": ["volume", "cvp_vah_val_width"],
    },
]

LOOKAHEAD_NAME_RE = re.compile(
    r"(future|fwd|forward|target|label|oracle|mfe|mae|exit_|raw_exit|trade_return|"
    r"net_per_notional|win$|reason|hold|exit_i|pnl|profit|loss|stop|take_profit|"
    r"stop_loss|cum|peak|dd|drawdown|return)",
    re.IGNORECASE,
)
TOL = 1.0e-8


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def report_path(model_id: str) -> Path:
    return ROOT / "tmp/causal_regen_20260516" / model_id / "report.json"


def redteam_path(model_id: str) -> Path:
    return ROOT / "tmp/causal_regen_20260516" / model_id / "redteam_audit_20260701.json"


def load_report_ledgers(model_id: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    report = read_json(report_path(model_id))
    artifacts = report["artifacts"]
    val = pd.read_csv(artifacts["selected_validation_ledger"])
    oos = pd.read_csv(artifacts["selected_oos_ledger"])
    return report, val, oos


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, severity: str, details: dict[str, Any]) -> None:
    checks.append({"name": name, "pass": bool(passed), "severity": severity, "details": details})


def timestamp_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce", utc=False)


def temporal_checks(model_id: str, val: pd.DataFrame, oos: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for split, df in [("validation", val), ("oos", oos)]:
        entry_ts = timestamp_series(df, "entry_timestamp")
        exit_ts = timestamp_series(df, "exit_timestamp")
        entry_i = pd.to_numeric(df["entry_i"], errors="coerce")
        exit_i = pd.to_numeric(df["exit_i"], errors="coerce")
        active = pd.to_numeric(df["notional"], errors="coerce").fillna(0.0) > TOL
        add_check(
            checks,
            f"{split}_entry_exit_order",
            bool(((exit_ts >= entry_ts) & (exit_i >= entry_i)).all()),
            "leakage_blocker",
            {
                "bad_rows": int((~((exit_ts >= entry_ts) & (exit_i >= entry_i))).sum()),
                "rows": int(len(df)),
            },
        )
        if "hold_hours" in df.columns:
            hold = pd.to_numeric(df["hold_hours"], errors="coerce")
        else:
            hold = (exit_ts - entry_ts).dt.total_seconds() / 3600.0
        add_check(
            checks,
            f"{split}_nonnegative_hold",
            bool((hold[active] >= -TOL).all()),
            "leakage_blocker",
            {"min_active_hold_hours": float(hold[active].min()) if bool(active.any()) else None},
        )
    val_entry = timestamp_series(val, "entry_timestamp")
    val_exit = timestamp_series(val, "exit_timestamp")
    oos_entry = timestamp_series(oos, "entry_timestamp")
    duplicate_entries = sorted(set(val_entry.astype(str)).intersection(set(oos_entry.astype(str))))
    add_check(
        checks,
        "validation_oos_time_order",
        bool(val_exit.max() < oos_entry.min()),
        "contamination_blocker",
        {
            "validation_max_exit": str(val_exit.max()),
            "oos_min_entry": str(oos_entry.min()),
        },
    )
    add_check(
        checks,
        "validation_oos_no_duplicate_entry_timestamps",
        not duplicate_entries,
        "contamination_blocker",
        {"duplicate_count": len(duplicate_entries), "sample": duplicate_entries[:5]},
    )
    return checks


def oos_selection_risk(report: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        str(report.get(key, ""))
        for key in ["selection_scope", "selection_rule", "status"]
    ).lower()
    if "oos is not used as filter" in text or "oos not used as filter" in text:
        mode = "readout_after_selection"
    elif "highest oos" in text or "maximize oos" in text or "oos-balanced" in text:
        mode = "ordering"
    elif "oos" in text and ("sort" in text or "tie" in text):
        mode = "safety_tiebreak"
    elif "oos" in text:
        mode = "safety_or_disclosure"
    else:
        mode = "none_declared"
    return {
        "mode": mode,
        "fresh_holdout_declared": "fresh_holdout_required" in text or "fresh holdout" in text,
        "selection_scope": report.get("selection_scope"),
        "selection_rule": report.get("selection_rule"),
    }


def market_paths() -> tuple[Path, Path]:
    roll7 = load_module("omega462_roll7_for_frontier_leakage", ROLL7_PATH)
    mods = roll7.modules()
    v1 = mods["v1"]
    stop_mod = mods["stop_mod"]
    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    return Path(train_csv), Path(eval_csv)


def feature_parity(
    model_id: str,
    val: pd.DataFrame,
    oos: pd.DataFrame,
    train_market: pd.DataFrame,
    eval_market: pd.DataFrame,
    features: list[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not features:
        add_check(checks, "entry_feature_parity_not_applicable", True, "info", {"features": []})
        return checks
    for feature in features:
        risky_name = bool(LOOKAHEAD_NAME_RE.search(feature))
        add_check(
            checks,
            f"feature_name_not_lookahead_like_{feature}",
            not risky_name,
            "leakage_blocker",
            {"feature": feature, "lookahead_regex": LOOKAHEAD_NAME_RE.pattern},
        )
        for split, ledger, market in [
            ("validation", val, train_market),
            ("oos", oos, eval_market),
        ]:
            if feature not in ledger.columns or feature not in market.columns:
                add_check(
                    checks,
                    f"{split}_{feature}_column_present",
                    False,
                    "contamination_blocker",
                    {"ledger_has": feature in ledger.columns, "market_has": feature in market.columns},
                )
                continue
            active = pd.to_numeric(ledger["notional"], errors="coerce").fillna(0.0) > TOL
            ledger_active = ledger.loc[active, ["entry_timestamp", feature]].copy()
            ledger_active["entry_timestamp_key"] = pd.to_datetime(
                ledger_active["entry_timestamp"], errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
            market_lookup = market[["timestamp", feature]].copy()
            market_lookup["entry_timestamp_key"] = pd.to_datetime(
                market_lookup["timestamp"], errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
            market_lookup = market_lookup.drop_duplicates("entry_timestamp_key", keep="first")
            merged = ledger_active.merge(
                market_lookup[["entry_timestamp_key", feature]],
                on="entry_timestamp_key",
                how="left",
                suffixes=("_ledger", "_market"),
            )
            unmatched = int(merged[f"{feature}_market"].isna().sum())
            ledger_values = pd.to_numeric(merged[f"{feature}_ledger"], errors="coerce")
            market_values = pd.to_numeric(merged[f"{feature}_market"], errors="coerce")
            diffs = (ledger_values - market_values).abs()
            max_abs_diff = float(diffs.max()) if len(diffs) else 0.0
            mismatch_count = int((diffs > 1.0e-9).sum())
            sample = (
                merged.loc[diffs > 1.0e-9, ["entry_timestamp", f"{feature}_ledger", f"{feature}_market"]]
                .head(5)
                .to_dict(orient="records")
            )
            add_check(
                checks,
                f"{split}_{feature}_matches_market_entry_timestamp",
                bool(unmatched == 0 and max_abs_diff <= 1.0e-9),
                "contamination_blocker",
                {
                    "active_rows_checked": int(len(merged)),
                    "unmatched_timestamps": unmatched,
                    "mismatch_count": mismatch_count,
                    "max_abs_diff": max_abs_diff,
                    "sample_mismatches": sample,
                },
            )
    return checks


def selected_metrics(report: dict[str, Any]) -> dict[str, Any]:
    selected = report["selected_variant"]
    return {
        "validation_pnl": selected.get("validation_pnl"),
        "oos_pnl": selected.get("oos_pnl"),
        "validation_avg_hold_hours": selected.get("validation_avg_hold_hours"),
        "oos_avg_hold_hours": selected.get("oos_avg_hold_hours"),
        "validation_max_hold_hours": selected.get("validation_max_hold_hours"),
        "oos_max_hold_hours": selected.get("oos_max_hold_hours"),
        "validation_mdd": selected.get("validation_mdd"),
        "oos_mdd": selected.get("oos_mdd"),
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Omega 4.6.2 Frontier Leakage Red-Team Audit - 2026-07-01",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Direct future-data leak found: `{payload['direct_future_leak_found']}`",
        f"- Data contamination found: `{payload['data_contamination_found']}`",
        f"- OOS selection contamination blockers: `{payload['oos_selection_contamination_blockers']}`",
        f"- Full live pass: `{payload['full_live_pass']}`",
        "",
        "## Model Verdicts",
        "",
        "| Model | Direct Leak | Split Clean | Entry Feature Parity | OOS Selection Risk | Verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["models"]:
        lines.append(
            f"| `{row['model_id']}` | `{row['direct_future_leak_found']}` | "
            f"`{row['split_clean']}` | `{row['entry_feature_parity_pass']}` | "
            f"`{row['oos_selection_risk']['mode']}` | `{row['verdict']}` |"
        )
    lines.extend(["", "## Critical Findings", ""])
    if payload["critical_findings"]:
        lines.extend(f"- {item}" for item in payload["critical_findings"])
    else:
        lines.append("- No direct future-data leakage was found in the audited ledgers/features.")
    if payload["data_contamination_findings"]:
        lines.extend(["", "## Data Contamination Findings", ""])
        lines.extend(f"- {item}" for item in payload["data_contamination_findings"])
    lines.extend(["", "## OOS/Fresh-Holdout Findings", ""])
    for row in payload["models"]:
        risk = row["oos_selection_risk"]
        lines.append(
            f"- `{row['model_id']}`: OOS selection mode `{risk['mode']}`, "
            f"fresh holdout declared `{risk['fresh_holdout_declared']}`."
        )
    lines.extend(
        [
            "",
            "## Entry Feature Causality",
            "",
            "- For models using `volume` and `cvp_vah_val_width`, the audit compared ledger feature values against the source market CSV at each active trade's `entry_timestamp`.",
            "- The corrected check uses `entry_timestamp`, because rolled segment `entry_i` values are synthetic and can exceed the source CSV row count.",
            f"- Upstream CVP feature causality audit: `{payload['cvp_feature_audit_verdict']}` at `{payload['cvp_feature_audit']}`.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{AUDIT_JSON}`",
        ]
    )
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_csv, eval_csv = market_paths()
    train_market = pd.read_csv(train_csv)
    eval_market = pd.read_csv(eval_csv)
    runtime = read_json(RUNTIME_BLOCKER_JSON) if RUNTIME_BLOCKER_JSON.exists() else {}
    cvp_feature_audit = read_json(CVP_FEATURE_AUDIT_JSON) if CVP_FEATURE_AUDIT_JSON.exists() else {}
    models: list[dict[str, Any]] = []
    critical_findings: list[str] = []
    data_contamination_findings: list[str] = []

    for config in FRONTIER_MODELS:
        model_id = config["model_id"]
        report, val, oos = load_report_ledgers(model_id)
        checks: list[dict[str, Any]] = []
        add_check(
            checks,
            "report_and_redteam_exist",
            report_path(model_id).exists() and redteam_path(model_id).exists(),
            "research_blocker",
            {"report": str(report_path(model_id)), "redteam": str(redteam_path(model_id))},
        )
        checks.extend(temporal_checks(model_id, val, oos))
        checks.extend(feature_parity(model_id, val, oos, train_market, eval_market, config["entry_features"]))
        risk = oos_selection_risk(report)
        fresh_holdout_required = risk["mode"] in {"ordering", "safety_tiebreak"}
        add_check(
            checks,
            "fresh_holdout_requirement_disclosed_if_oos_selected",
            (not fresh_holdout_required) or bool(risk["fresh_holdout_declared"]),
            "contamination_blocker",
            {**risk, "fresh_holdout_required": fresh_holdout_required},
        )
        direct_leak = any((not c["pass"]) and c["severity"] == "leakage_blocker" for c in checks)
        split_clean = all(c["pass"] for c in checks if c["name"].startswith("validation_oos_"))
        entry_feature_checks = [
            c for c in checks if "matches_market_entry_timestamp" in c["name"] or "feature_name_not_lookahead_like" in c["name"]
        ]
        entry_feature_parity_pass = all(c["pass"] for c in entry_feature_checks)
        contamination_failures = [
            c
            for c in checks
            if (not c["pass"]) and c["severity"] == "contamination_blocker"
        ]
        data_contamination = bool(contamination_failures)
        oos_blocker = risk["mode"] in {"ordering", "safety_tiebreak"}
        if direct_leak:
            verdict = "FAIL_DIRECT_FUTURE_LEAK"
            critical_findings.append(f"`{model_id}` has failed leakage-blocker checks.")
        elif data_contamination:
            verdict = "FAIL_DATA_CONTAMINATION"
            data_contamination_findings.append(
                f"`{model_id}` has contamination-blocker failures: "
                + ", ".join(f"`{c['name']}`" for c in contamination_failures[:6])
            )
        elif oos_blocker:
            verdict = "NO_DIRECT_LEAK_FOUND_RESEARCH_ONLY_OOS_CONTAMINATION_BLOCKED"
        else:
            verdict = "NO_DIRECT_LEAK_FOUND"
        models.append(
            {
                "model_id": model_id,
                "label": config["label"],
                "declared_risk": config["declared_risk"],
                "metrics": selected_metrics(report),
                "oos_selection_risk": risk,
                "direct_future_leak_found": direct_leak,
                "data_contamination_found": data_contamination,
                "split_clean": split_clean,
                "entry_feature_parity_pass": entry_feature_parity_pass,
                "verdict": verdict,
                "checks": checks,
                "artifacts": report["artifacts"],
            }
        )

    direct_future_leak_found = any(row["direct_future_leak_found"] for row in models)
    data_contamination_found = any(row["data_contamination_found"] for row in models)
    oos_selection_contamination_blockers = [
        row["model_id"]
        for row in models
        if row["oos_selection_risk"]["mode"] in {"ordering", "safety_tiebreak"}
    ]
    full_live_pass = (
        not direct_future_leak_found
        and not data_contamination_found
        and not oos_selection_contamination_blockers
        and runtime.get("verdict") != "RUNTIME_WIRING_BLOCKED"
        and cvp_feature_audit.get("verdict") == "CVP_FEATURE_CAUSALITY_PASS"
    )
    if direct_future_leak_found:
        verdict = "FRONTIER_LEAKAGE_REDTEAM_FAIL"
    elif data_contamination_found:
        verdict = "NO_DIRECT_FUTURE_LEAK_FOUND_BUT_DATA_CONTAMINATION_FOUND"
    elif oos_selection_contamination_blockers:
        verdict = "NO_DIRECT_FUTURE_LEAK_FOUND_BUT_OOS_SELECTION_CONTAMINATION_BLOCKS_FULL_LIVE"
    elif full_live_pass:
        verdict = "FRONTIER_LEAKAGE_RUNTIME_PASS"
    else:
        verdict = "NO_DIRECT_FUTURE_LEAK_FOUND"
    if runtime.get("verdict") == "RUNTIME_WIRING_BLOCKED":
        critical_findings.append(
            "Runtime-native replay remains blocked, so even leakage-clean research candidates are not full-live pass."
        )
    if cvp_feature_audit.get("verdict") != "CVP_FEATURE_CAUSALITY_PASS":
        critical_findings.append(
            "CVP feature causality audit is missing or failing, so cvp_vah_val_width provenance is not full-live pass."
        )
    if oos_selection_contamination_blockers:
        critical_findings.append(
            "OOS readout was used in candidate selection/tie-break for: "
            + ", ".join(f"`{model_id}`" for model_id in oos_selection_contamination_blockers)
            + ". These require fresh holdout/walk-forward before live promotion."
        )

    payload = {
        "audit_id": "omega4_6_2_frontier_leakage_redteam_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "direct_future_leak_found": direct_future_leak_found,
        "data_contamination_found": data_contamination_found,
        "oos_selection_contamination_blockers": oos_selection_contamination_blockers,
        "full_live_pass": full_live_pass,
        "train_market_csv": str(train_csv),
        "eval_market_csv": str(eval_csv),
        "runtime_blocker_verdict": runtime.get("verdict"),
        "cvp_feature_audit": str(CVP_FEATURE_AUDIT_JSON),
        "cvp_feature_audit_verdict": cvp_feature_audit.get("verdict"),
        "critical_findings": critical_findings,
        "data_contamination_findings": data_contamination_findings,
        "models": models,
        "artifacts": {"audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD)},
    }
    write_json(AUDIT_JSON, payload)
    write_markdown(payload)
    print(
        json.dumps(
            {
                "audit_json": str(AUDIT_JSON),
                "audit_md": str(AUDIT_MD),
                "verdict": verdict,
                "direct_future_leak_found": direct_future_leak_found,
                "data_contamination_found": data_contamination_found,
                "oos_selection_contamination_blockers": oos_selection_contamination_blockers,
                "full_live_pass": full_live_pass,
            },
            ensure_ascii=False,
            default=json_default,
        )
    )
    return 0 if not direct_future_leak_found else 1


if __name__ == "__main__":
    raise SystemExit(main())
