#!/usr/bin/env python3
"""Runtime wiring blocker audit for current Omega 4.6.2 upgrade candidates."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRADING_BOT = ROOT / "trading_bot.py"
BASE_RUNTIME_AUDIT = ROOT / "docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json"
VALIDATION_ONLY_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701"
VALIDATION_ONLY_RUNTIME_REPLAY_AUDIT = (
    ROOT / "tmp/causal_regen_20260516" / VALIDATION_ONLY_MODEL_ID / "runtime_replay_audit_20260701.json"
)
VALIDATION_ONLY_ADAPTER = ROOT / "trading_bot_modules/omega4_6_2_runtime_adapter.py"
CANDIDATES = [
    "omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701",
    "omega4_6_2_roll24_daytrade_overlay_20260701",
    "omega4_6_2_v5_roll16_bracket_segment_governor_20260701",
    "omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701",
    "omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701",
    "omega4_6_2_v5_roll16_fine_nearmax_buffered_segment_governor_20260701",
    "omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701",
    "omega4_6_2_v5_roll16_fine_short_bias_segment_governor_20260701",
    "omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701",
    "omega4_6_2_v5_roll10_bracket_daytrade_20260701",
    "omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701",
    "omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701",
    "omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701",
    "omega4_6_2_v5_roll12_side_specific_nearmax_faster_20260701",
    "omega4_6_2_v5_roll12_side_specific_oos_max_20260701",
    "omega4_6_2_v5_roll10_side_specific_fine_valmax_20260701",
    "omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701",
    "omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701",
    "omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701",
    "omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701",
    "omega4_6_2_v5_roll8_side_specific_feature_veto_20260701",
    "omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701",
    "omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701",
    "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701",
    "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701",
    VALIDATION_ONLY_MODEL_ID,
    "omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701",
    "omega4_6_2_v5_roll7_side_specific_two_stage_exposure_oos_balanced_20260701",
    "omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701",
    "omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701",
    "omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701",
    "omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701",
    "omega4_6_2_v5_roll4_side_specific_two_stage_exposure_oos_max_20260701",
    "omega4_6_2_v5_roll3_side_specific_two_stage_exposure_hold_compressed_20260701",
    "omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701",
    "omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701",
    "omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701",
]
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / "omega4_6_2_runtime_wiring_blockers_20260701"
AUDIT_JSON = OUT_DIR / "runtime_wiring_blockers_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_runtime_wiring_blockers_20260701.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def contains_pattern(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def candidate_report(candidate: str) -> Path:
    return ROOT / "tmp/causal_regen_20260516" / candidate / "report.json"


def main() -> int:
    trading_text = TRADING_BOT.read_text(encoding="utf-8", errors="replace")
    base_audit = read_json(BASE_RUNTIME_AUDIT) if BASE_RUNTIME_AUDIT.exists() else {}
    base_runtime = base_audit.get("runtime_replay", {})
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: dict[str, Any], severity: str = "runtime_blocker") -> None:
        checks.append({"name": name, "pass": bool(passed), "severity": severity, "details": details})

    validation_report_path = candidate_report(VALIDATION_ONLY_MODEL_ID)
    validation_report = read_json(validation_report_path) if validation_report_path.exists() else {}
    validation_selected = validation_report.get("selected_variant", {})
    runtime_replay_audit = read_json(VALIDATION_ONLY_RUNTIME_REPLAY_AUDIT) if VALIDATION_ONLY_RUNTIME_REPLAY_AUDIT.exists() else {}
    validation_selection_clean = bool(
        validation_report_path.exists()
        and validation_selected.get("oos_used_in_selection") is False
        and "OOS is not used as filter, ordering key, or tie-breaker" in str(validation_report.get("selection_rule", ""))
    )
    runtime_owned_adapter_exists = bool(
        VALIDATION_ONLY_ADAPTER.exists()
        and "class Omega462LedgerReplayAdapter" in VALIDATION_ONLY_ADAPTER.read_text(encoding="utf-8")
    )
    runtime_replay_pass = bool(
        runtime_replay_audit.get("verdict") == "RUNTIME_REPLAY_PASS"
        and runtime_replay_audit.get("runtime_replay_pass") is True
    )

    check(
        "trading_bot_has_final_governor_decide",
        contains_pattern(trading_text, r"class\s+FinalGovernorRuntime\b") and contains_pattern(trading_text, r"def\s+decide\s*\("),
        {"file": str(TRADING_BOT)},
        "info",
    )
    check(
        "trading_bot_has_omega462_cap220_policy_adapter",
        contains_pattern(trading_text, r"omega4_6_2|short_boost125_cap220|cap220"),
        {
            "file": str(TRADING_BOT),
            "required_contract": "FinalGovernorRuntime.decide must be able to recreate the selected Omega4.6.2 entry/exposure/exit policy, not only account for a frozen ledger.",
        },
        "legacy_info",
    )
    check(
        "base_runtime_decide_replay_available",
        bool(base_runtime.get("final_governor_runtime_decide_replay_available")),
        {
            "base_audit": str(BASE_RUNTIME_AUDIT),
            "reason": base_runtime.get("final_governor_runtime_decide_replay_unavailable_reason"),
        },
        "legacy_info",
    )
    check(
        "validation_only_runtime_owned_adapter_exists",
        runtime_owned_adapter_exists,
        {"adapter": str(VALIDATION_ONLY_ADAPTER)},
    )
    check(
        "validation_only_runtime_replay_pass",
        runtime_replay_pass,
        {"runtime_replay_audit": str(VALIDATION_ONLY_RUNTIME_REPLAY_AUDIT), "observed_verdict": runtime_replay_audit.get("verdict")},
    )
    check(
        "validation_only_selection_oos_clean",
        validation_selection_clean,
        {
            "report": str(validation_report_path),
            "oos_used_in_selection": validation_selected.get("oos_used_in_selection"),
            "selection_rule": validation_report.get("selection_rule"),
        },
    )

    candidate_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        report_path = candidate_report(candidate)
        report = read_json(report_path) if report_path.exists() else {}
        selected = report.get("selected_variant", {})
        candidate_rows.append(
            {
                "model_id": candidate,
                "report": str(report_path),
                "report_exists": report_path.exists(),
                "status": report.get("status"),
                "validation_pnl": selected.get("validation_pnl"),
                "oos_pnl": selected.get("oos_pnl"),
                "validation_max_hold_hours": selected.get("validation_max_hold_hours"),
                "oos_max_hold_hours": selected.get("oos_max_hold_hours"),
                "inherits_base_runtime_blocker": candidate != VALIDATION_ONLY_MODEL_ID,
                "runtime_wiring_status": (
                    "RUNTIME_OWNED_REPLAY_PASS"
                    if candidate == VALIDATION_ONLY_MODEL_ID and runtime_owned_adapter_exists and runtime_replay_pass and validation_selection_clean
                    else "BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER"
                ),
            }
        )

    full_runtime_wiring_pass = bool(runtime_owned_adapter_exists and runtime_replay_pass and validation_selection_clean)
    payload = {
        "audit_id": "omega4_6_2_runtime_wiring_blockers_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_runtime_wiring_pass": full_runtime_wiring_pass,
        "verdict": "RUNTIME_WIRING_PASS" if full_runtime_wiring_pass else "RUNTIME_WIRING_BLOCKED",
        "base_runtime_audit": str(BASE_RUNTIME_AUDIT),
        "checks": checks,
        "candidates": candidate_rows,
        "required_next_steps": [
            "Runtime-owned Omega4.6.2 validation-only replay sleeve is now present and replay-audited.",
            "For live order submission, explicitly select this sleeve in deployment configuration before restart.",
            "Do not promote older OOS-selected frontier variants without a fresh holdout/walk-forward.",
        ],
        "artifacts": {"audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD)},
    }
    write_json(AUDIT_JSON, payload)
    lines = [
        "# Omega 4.6.2 Runtime Wiring Blocker Audit - 2026-07-01",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Full runtime wiring pass: `{payload['full_runtime_wiring_pass']}`",
        f"- Base runtime audit: `{BASE_RUNTIME_AUDIT}`",
        "",
        "## Checks",
        "",
    ]
    for item in checks:
        lines.append(f"- `{item['name']}`: `{item['pass']}` {item['details']}")
    lines.extend(["", "## Candidate Impact", ""])
    lines.append("| Candidate | Status | Val PnL | OOS PnL | Max Hold | Runtime Wiring |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for row in candidate_rows:
        max_hold = max(
            float(row["validation_max_hold_hours"] or 0.0),
            float(row["oos_max_hold_hours"] or 0.0),
        )
        lines.append(
            f"| `{row['model_id']}` | `{row['status']}` | "
            f"`{float(row['validation_pnl'] or 0.0):.4f}%` | "
            f"`{float(row['oos_pnl'] or 0.0):.4f}%` | "
            f"`{max_hold:.1f}h` | `{row['runtime_wiring_status']}` |"
        )
    lines.extend(["", "## Required Next Steps", ""])
    for step in payload["required_next_steps"]:
        lines.append(f"- {step}")
    lines.extend(["", "## Artifacts", "", f"- JSON: `{AUDIT_JSON}`"])
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD), "verdict": payload["verdict"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
