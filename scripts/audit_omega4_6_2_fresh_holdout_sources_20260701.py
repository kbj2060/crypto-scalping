#!/usr/bin/env python3
"""Audit local sources for a fresh Omega 4.6.2 post-OOS holdout."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONTRACT = (
    ROOT
    / "tmp/causal_regen_20260516/omega4_6_2_cap220_short_boost125_time_stop120h_20260630/runtime_contract.json"
)
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_2_fresh_holdout_sources_20260701"
AUDIT_JSON = OUT_DIR / "fresh_holdout_sources_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_fresh_holdout_sources_20260701.md"
EPS = 1.0e-12


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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def csv_time_range(path: Path) -> dict[str, Any] | None:
    try:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception as exc:
        return {"path": str(path), "error": type(exc).__name__, "message": str(exc)}
    time_col = next(
        (col for col in ["timestamp", "entry_timestamp", "exit_timestamp", "open_time", "datetime", "date"] if col in columns),
        None,
    )
    if time_col is None:
        return None
    try:
        values = pd.read_csv(path, usecols=[time_col])
        ts = pd.to_datetime(values[time_col], errors="coerce")
    except Exception as exc:
        return {"path": str(path), "time_col": time_col, "error": type(exc).__name__, "message": str(exc)}
    return {
        "path": str(path),
        "time_col": time_col,
        "rows": int(len(ts)),
        "columns": int(len(columns)),
        "size_mb": float(path.stat().st_size / 1_000_000.0),
        "min_timestamp": ts.min(),
        "max_timestamp": ts.max(),
        "na_timestamps": int(ts.isna().sum()),
    }


def candidate_paths() -> list[Path]:
    paths: list[Path] = []
    for path in (ROOT / "tmp/causal_regen_20260516").rglob("trade_candidates_2026*.csv"):
        paths.append(path)
    for path in (ROOT / "data/ensemble/supervised").rglob("*2026*.csv"):
        name = str(path)
        if "training_features_2026" in name or path.name == "features_2026.csv":
            paths.append(path)
    seen: dict[str, Path] = {}
    for path in paths:
        seen[str(path)] = path
    return sorted(seen.values(), key=lambda item: str(item))


def write_markdown(payload: dict[str, Any]) -> None:
    top = payload["top_sources_by_max_timestamp"][:20]
    lines = [
        "# Omega 4.6.2 Fresh Holdout Source Audit - 2026-07-01",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Runtime eval source max timestamp: `{payload['runtime_eval_source']['max_timestamp']}`",
        f"- Existing OOS ledger max entry timestamp: `{payload['oos_ledger']['max_timestamp']}`",
        f"- Local candidate CSVs scanned: `{payload['sources_scanned']}`",
        f"- Post-OOS local sources found: `{len(payload['post_oos_sources'])}`",
        "",
        "## Top Local Sources",
        "",
        "| Max Timestamp | Min Timestamp | Rows | Columns | Path |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in top:
        lines.append(
            f"| `{row['max_timestamp']}` | `{row['min_timestamp']}` | "
            f"`{row['rows']}` | `{row['columns']}` | `{row['path']}` |"
        )
    lines.extend(
        [
            "",
            "## Finding",
            "",
            "No local 2026 candidate or supervised feature CSV extends beyond the current Omega4.6.2 eval source window. A clean fresh holdout or walk-forward therefore cannot be claimed from existing local artifacts.",
            "",
            "## Required Follow-Up",
            "",
            "- Generate the exact Omega4.6.2 candidate feature contract on data after the current eval source max timestamp.",
            "- Rebuild parent predictions, risk sidecar inputs, ledgers, and runtime replay on that post-OOS window.",
            "- Keep this as a fresh readout only; do not tune candidate parameters on it.",
            "",
            "## Artifacts",
            "",
            f"- Audit JSON: `{AUDIT_JSON}`",
        ]
    )
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    runtime = read_json(RUNTIME_CONTRACT)
    h48_report = read_json(resolve_path(runtime["components"]["h48qual"]["report"]))
    eval_csv = resolve_path(h48_report["risk_model"]["eval_csv"])
    eval_range = csv_time_range(eval_csv)
    if eval_range is None:
        raise RuntimeError(f"eval csv has no timestamp column: {eval_csv}")

    source_report = read_json(resolve_path(runtime["source_report"]))
    variant = runtime["variant"].replace(".", "p").replace("/", "_")
    source_dir = resolve_path(runtime["source_report"]).parent
    oos_ledger = source_dir / f"oos_{variant}_ledger.csv"
    oos_range = csv_time_range(oos_ledger)
    if oos_range is None:
        raise RuntimeError(f"oos ledger has no timestamp column: {oos_ledger}")

    rows = []
    for path in candidate_paths():
        row = csv_time_range(path)
        if row is not None and "max_timestamp" in row:
            rows.append(row)
    rows.sort(key=lambda row: (pd.Timestamp(row["max_timestamp"]), row["path"]), reverse=True)
    eval_max = pd.Timestamp(eval_range["max_timestamp"])
    post_oos = [row for row in rows if pd.Timestamp(row["max_timestamp"]) > eval_max]
    verdict = (
        "FRESH_HOLDOUT_AVAILABLE"
        if post_oos
        else "FRESH_HOLDOUT_NOT_AVAILABLE_IN_LOCAL_ARTIFACTS"
    )
    payload = {
        "audit_id": "omega4_6_2_fresh_holdout_sources_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "runtime_contract": str(RUNTIME_CONTRACT),
        "source_report": str(resolve_path(runtime["source_report"])),
        "source_report_model_id": source_report.get("model_id"),
        "runtime_eval_source": eval_range,
        "oos_ledger": oos_range,
        "fresh_holdout_required_after": eval_range["max_timestamp"],
        "sources_scanned": int(len(rows)),
        "post_oos_sources": post_oos,
        "top_sources_by_max_timestamp": rows[:50],
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
                "sources_scanned": len(rows),
                "post_oos_sources": len(post_oos),
                "fresh_holdout_required_after": payload["fresh_holdout_required_after"],
            },
            ensure_ascii=False,
            default=json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
