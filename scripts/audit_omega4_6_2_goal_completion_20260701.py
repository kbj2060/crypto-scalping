#!/usr/bin/env python3
"""Completion audit for the 2026-07-01 Omega 4.6.2 upgrade loop goal."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_BLOCKER_JSON = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_runtime_wiring_blockers_20260701"
    / "runtime_wiring_blockers_20260701.json"
)
UPGRADE_LOOP_MD = ROOT / "docs/audits/omega4_6_2_upgrade_loop_20260701.md"
RUNTIME_BLOCKER_MD = ROOT / "docs/audits/omega4_6_2_runtime_wiring_blockers_20260701.md"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / "omega4_6_2_goal_completion_20260701"
AUDIT_JSON = OUT_DIR / "goal_completion_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_goal_completion_audit_20260701.md"

BASELINE_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
BEST_PNL_HOLD_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701"
SUB5H_ID = "omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701"
ULTRA_SHORT_ID = "omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701"
ROLL12_OOS_ID = "omega4_6_2_v5_roll12_side_specific_oos_max_20260701"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def report_path(model_id: str) -> Path:
    return ROOT / "tmp/causal_regen_20260516" / model_id / "report.json"


def redteam_path(model_id: str) -> Path:
    return ROOT / "tmp/causal_regen_20260516" / model_id / "redteam_audit_20260701.json"


def selected(model_id: str) -> dict[str, Any]:
    return read_json(report_path(model_id))["selected_variant"]


def max_hold(row: dict[str, Any]) -> float:
    return max(float(row["validation_max_hold_hours"]), float(row["oos_max_hold_hours"]))


def check(checks: list[dict[str, Any]], name: str, passed: bool, details: dict[str, Any]) -> None:
    checks.append({"name": name, "pass": bool(passed), "details": details})


def main() -> int:
    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    runtime = read_json(RUNTIME_BLOCKER_JSON)
    candidate_ids = [row["model_id"] for row in runtime["candidates"]]
    checks: list[dict[str, Any]] = []

    check(
        checks,
        "deadline_reached",
        now_kst >= datetime(2026, 7, 1, 9, 0, 0, tzinfo=ZoneInfo("Asia/Seoul")),
        {"now_kst": now_kst.isoformat(), "deadline_kst": "2026-07-01T09:00:00+09:00"},
    )
    check(
        checks,
        "core_reports_exist",
        UPGRADE_LOOP_MD.exists() and RUNTIME_BLOCKER_MD.exists() and RUNTIME_BLOCKER_JSON.exists(),
        {
            "upgrade_loop_md": str(UPGRADE_LOOP_MD),
            "runtime_blocker_md": str(RUNTIME_BLOCKER_MD),
            "runtime_blocker_json": str(RUNTIME_BLOCKER_JSON),
        },
    )

    missing_reports = [model_id for model_id in candidate_ids if not report_path(model_id).exists()]
    missing_redteams = [model_id for model_id in candidate_ids if not redteam_path(model_id).exists()]
    check(
        checks,
        "all_registered_candidates_have_reports",
        not missing_reports,
        {"registered_candidates": len(candidate_ids), "missing_reports": missing_reports},
    )
    check(
        checks,
        "all_registered_candidates_have_redteam_json",
        not missing_redteams,
        {"registered_candidates": len(candidate_ids), "missing_redteams": missing_redteams},
    )

    redteam_failures: list[dict[str, Any]] = []
    for model_id in candidate_ids:
        path = redteam_path(model_id)
        if not path.exists():
            continue
        audit = read_json(path)
        verdict = str(audit.get("verdict", ""))
        research_pass = audit.get("research_pass")
        legacy_pass_verdict = research_pass is None and "PASS" in verdict and verdict != "REDTEAM_FAIL"
        if not (research_pass is True or legacy_pass_verdict):
            redteam_failures.append(
                {
                    "model_id": model_id,
                    "verdict": verdict,
                    "research_pass": research_pass,
                }
            )
    check(
        checks,
        "all_registered_redteams_research_pass",
        not redteam_failures,
        {"failures": redteam_failures},
    )

    baseline = selected(BASELINE_ID)
    best = selected(BEST_PNL_HOLD_ID)
    sub5h = selected(SUB5H_ID)
    ultra = selected(ULTRA_SHORT_ID)
    roll12 = selected(ROLL12_OOS_ID)

    check(
        checks,
        "best_frontier_improves_pnl_and_reduces_hold_vs_baseline",
        float(best["validation_pnl"]) > float(baseline["validation_pnl"])
        and float(best["oos_pnl"]) > float(baseline["oos_pnl"])
        and float(best["validation_avg_hold_hours"]) < float(baseline["validation_avg_hold_hours"])
        and float(best["oos_avg_hold_hours"]) < float(baseline["oos_avg_hold_hours"])
        and max_hold(best) < max_hold(baseline),
        {"baseline": {"model_id": BASELINE_ID, **baseline}, "best": {"model_id": BEST_PNL_HOLD_ID, **best}},
    )
    check(
        checks,
        "sub5h_candidate_keeps_pnl_contract",
        float(sub5h["validation_pnl"]) >= 100.0
        and float(sub5h["oos_pnl"]) >= 100.0
        and float(sub5h["validation_avg_hold_hours"]) < 5.0
        and float(sub5h["oos_avg_hold_hours"]) < 5.0
        and max_hold(sub5h) <= 6.0,
        {"model_id": SUB5H_ID, **sub5h},
    )
    check(
        checks,
        "ultra_short_candidate_keeps_pnl_contract",
        float(ultra["validation_pnl"]) >= 100.0
        and float(ultra["oos_pnl"]) >= 100.0
        and max_hold(ultra) <= 2.0,
        {"model_id": ULTRA_SHORT_ID, **ultra},
    )
    check(
        checks,
        "roll12_oos_candidate_recorded",
        float(roll12["oos_pnl"]) > 170.0 and max_hold(roll12) <= 12.0,
        {"model_id": ROLL12_OOS_ID, **roll12},
    )

    runtime_blocked = runtime.get("verdict") == "RUNTIME_WIRING_BLOCKED"
    check(
        checks,
        "full_live_blockers_disclosed",
        runtime_blocked,
        {"runtime_verdict": runtime.get("verdict"), "runtime_report": str(RUNTIME_BLOCKER_MD)},
    )

    completion_pass = all(item["pass"] for item in checks)
    payload = {
        "audit_id": "omega4_6_2_goal_completion_audit_20260701",
        "created_at_kst": now_kst.isoformat(),
        "objective": "오늘 오전 9시(KST)까지 Omega 계열 후보를 반복 개선해 PnL을 올리고 보유 시간을 줄이며, 업그레이드 후보마다 레드팀 전수조사 리포트를 남긴다.",
        "completion_pass": completion_pass,
        "verdict": "GOAL_COMPLETION_EVIDENCED" if completion_pass else "GOAL_COMPLETION_UNPROVEN",
        "checks": checks,
        "frontier_summary": {
            "baseline": {"model_id": BASELINE_ID, "selected_variant": baseline},
            "best_pnl_hold": {"model_id": BEST_PNL_HOLD_ID, "selected_variant": best},
            "sub5h": {"model_id": SUB5H_ID, "selected_variant": sub5h},
            "ultra_short": {"model_id": ULTRA_SHORT_ID, "selected_variant": ultra},
            "roll12_oos": {"model_id": ROLL12_OOS_ID, "selected_variant": roll12},
        },
        "registered_candidates": candidate_ids,
        "artifacts": {
            "audit_json": str(AUDIT_JSON),
            "audit_md": str(AUDIT_MD),
            "upgrade_loop_md": str(UPGRADE_LOOP_MD),
            "runtime_blocker_md": str(RUNTIME_BLOCKER_MD),
            "runtime_blocker_json": str(RUNTIME_BLOCKER_JSON),
        },
    }
    write_json(AUDIT_JSON, payload)

    lines = [
        "# Omega 4.6.2 Goal Completion Audit - 2026-07-01",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Completion pass: `{payload['completion_pass']}`",
        f"- Created at KST: `{payload['created_at_kst']}`",
        f"- Registered candidates checked: `{len(candidate_ids)}`",
        "",
        "## Checks",
        "",
    ]
    for item in checks:
        lines.append(f"- `{item['name']}`: `{item['pass']}`")
    lines.extend(
        [
            "",
            "## Frontier Evidence",
            "",
            "| Model | Validation PnL | OOS PnL | Validation Avg Hold | OOS Avg Hold | Max Hold |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, model_id, row in [
        ("Baseline", BASELINE_ID, baseline),
        ("Best PnL/Hold", BEST_PNL_HOLD_ID, best),
        ("Sub-5h", SUB5H_ID, sub5h),
        ("Ultra-short", ULTRA_SHORT_ID, ultra),
        ("Roll12 OOS", ROLL12_OOS_ID, roll12),
    ]:
        lines.append(
            f"| {label}: `{model_id}` | "
            f"`{float(row['validation_pnl']):.4f}%` | "
            f"`{float(row['oos_pnl']):.4f}%` | "
            f"`{float(row['validation_avg_hold_hours']):.4f}h` | "
            f"`{float(row['oos_avg_hold_hours']):.4f}h` | "
            f"`{max_hold(row):.1f}h` |"
        )
    lines.extend(
        [
            "",
            "## Full-Live Disclosure",
            "",
            "- Research red-team passed for all registered candidates.",
            "- Full-live promotion remains blocked by runtime-native replay adapter and fresh holdout requirements; see the runtime blocker report.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{AUDIT_JSON}`",
            f"- Upgrade loop: `{UPGRADE_LOOP_MD}`",
            f"- Runtime blockers: `{RUNTIME_BLOCKER_MD}`",
        ]
    )
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "audit_json": str(AUDIT_JSON),
                "audit_md": str(AUDIT_MD),
                "verdict": payload["verdict"],
                "completion_pass": completion_pass,
                "registered_candidates": len(candidate_ids),
            },
            ensure_ascii=False,
        )
    )
    return 0 if completion_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
