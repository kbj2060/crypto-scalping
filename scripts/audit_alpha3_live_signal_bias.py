#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _is_alpha3(row: dict[str, Any]) -> bool:
    text = json.dumps(row, ensure_ascii=False).lower()
    return any(token in text for token in ("alpha3", "alpha2_1", "alpha2.1", "v31_deep_alpha", "deep_alpha"))


def _present(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    return value is not None and str(value) != ""


def build_report(root: Path) -> dict[str, Any]:
    journal_path = root / "data/live/trade_journal.jsonl"
    state_path = root / "data/live/dashboard_state.json"
    compact_path = root / "data/live/dashboard_state_dsac_compact.json"
    rows = _load_jsonl(journal_path)
    alpha = [r for r in rows if _is_alpha3(r)]
    opens = [r for r in alpha if str(r.get("kind", "")).upper() == "OPEN"]
    closes = [r for r in alpha if str(r.get("kind", "")).upper() == "CLOSE"]
    required_diag = [
        "v31_q_long",
        "v31_q_short",
        "v31_edge",
        "v31_margin",
        "parent_action",
        "teacher_gate_result",
    ]
    missing_counts = {key: sum(1 for row in alpha if not _present(row, key)) for key in required_diag}

    state = _load_json(state_path)
    compact = _load_json(compact_path)
    signal = state.get("signal") if isinstance(state.get("signal"), dict) else {}
    sleeve_trace = signal.get("sleeve_trace") if isinstance(signal.get("sleeve_trace"), dict) else {}
    v31_trace = sleeve_trace.get("v31") if isinstance(sleeve_trace.get("v31"), dict) else {}
    alpha2_trace = sleeve_trace.get("alpha2_1") if isinstance(sleeve_trace.get("alpha2_1"), dict) else {}

    long_entries = sum(1 for row in opens if str(row.get("side", "")).upper() == "LONG")
    short_entries = sum(1 for row in opens if str(row.get("side", "")).upper() == "SHORT")
    diagnosis: list[str] = []
    if opens and short_entries == 0:
        diagnosis.append("recent_alpha3_entries_are_long_only")
    if any(v > 0 for v in missing_counts.values()):
        diagnosis.append("historical_journal_missing_flat_v31_teacher_diagnostics")
    if v31_trace:
        ql = float(v31_trace.get("q_long", 0.0) or 0.0)
        qs = float(v31_trace.get("q_short", 0.0) or 0.0)
        if ql > qs:
            diagnosis.append("latest_dashboard_v31_utility_prefers_long")
        elif qs > ql:
            diagnosis.append("latest_dashboard_v31_utility_prefers_short")
    if alpha2_trace:
        reason = str(alpha2_trace.get("reason", "") or "")
        if "pruned" in reason:
            diagnosis.append("latest_teacher_gate_pruned_parent")
        elif "keep" in reason:
            diagnosis.append("latest_teacher_gate_kept_parent")

    return {
        "schema_version": "alpha3_live_signal_bias_audit.v1",
        "journal_path": str(journal_path),
        "dashboard_state_path": str(state_path),
        "compact_state_path": str(compact_path),
        "rows_total": len(rows),
        "alpha3_rows": len(alpha),
        "alpha3_open_rows": len(opens),
        "alpha3_close_rows": len(closes),
        "entry_side_counts": dict(Counter(str(r.get("side", "")).upper() for r in opens)),
        "close_reason_counts": dict(Counter(str(r.get("reason", "")) for r in closes)),
        "missing_diagnostic_field_counts": missing_counts,
        "latest_dashboard_v31": {
            "q_long": v31_trace.get("q_long"),
            "q_short": v31_trace.get("q_short"),
            "edge": v31_trace.get("edge"),
            "margin": v31_trace.get("margin"),
            "selected_side": v31_trace.get("selected_side"),
            "pass_gate": v31_trace.get("pass_gate"),
        },
        "latest_dashboard_teacher_gate": {
            "reason": alpha2_trace.get("reason"),
            "teacher_pred_action": alpha2_trace.get("teacher_pred_action"),
            "teacher_confidence": alpha2_trace.get("teacher_confidence"),
            "teacher_quality": alpha2_trace.get("teacher_quality"),
            "keep_parent": alpha2_trace.get("keep_parent"),
            "parent_action_before": alpha2_trace.get("parent_action_before"),
            "parent_side_before": alpha2_trace.get("parent_side_before"),
        },
        "diagnosis": diagnosis,
        "note": (
            "Rows created before the diagnostic patch can lack q_long/q_short/teacher fields. "
            "New OPEN/CLOSE rows will carry flattened V31 and teacher-gate diagnostics."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="data/ensemble/reports/alpha3_live_signal_bias_audit_latest.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = build_report(root)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
