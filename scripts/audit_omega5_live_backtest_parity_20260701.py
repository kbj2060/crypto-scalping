#!/usr/bin/env python3
"""Audit whether the active Omega5 live path can reproduce the source backtest."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRADING_BOT = ROOT / "trading_bot.py"
OMEGA5_LIVE = ROOT / "trading_bot_modules" / "omega5_live.py"
OMEGA121_LIVE = ROOT / "trading_bot_modules" / "omega1_2_1_live.py"
OMEGA462_REPLAY = ROOT / "trading_bot_modules" / "omega4_6_2_runtime_adapter.py"
OMEGA462_SOURCE_PARENT = ROOT / "trading_bot_modules" / "omega4_6_2_source_parent_live.py"
SOURCE_REPORT = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701"
    / "report.json"
)
CAP220_RUNTIME_CONTRACT = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_cap220_short_boost125_time_stop120h_20260630"
    / "runtime_contract.json"
)
TRADE_JOURNAL = ROOT / "data/live/trade_journal.jsonl"
DASHBOARD_STATE = ROOT / "data/live/dashboard_state.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega5_live_backtest_parity_20260701"
AUDIT_JSON = OUT_DIR / "omega5_live_backtest_parity_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega5_live_backtest_parity_20260701.md"
RUNTIME_NATIVE_WALKFORWARD_REPORT = (
    ROOT
    / "tmp/causal_regen_20260516/omega5_source_parent_runtime_native_walkforward_20260701"
    / "report.json"
)
SOURCE_PARENT_LIVE_ADAPTER_AUDIT = (
    ROOT
    / "tmp/causal_regen_20260516/omega4_6_2_source_parent_live_adapter_audit_20260701"
    / "omega4_6_2_source_parent_live_adapter_audit_20260701.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_constant(text: str, name: str) -> str:
    match = re.search(rf'^{re.escape(name)}\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def check(name: str, passed: bool, severity: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "severity": severity, "details": details}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"_decode_error": line[:200]})
    return rows


def latest_omega5_opens(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    opens = [
        row
        for row in rows
        if row.get("kind") == "OPEN"
        and str(row.get("model_id", row.get("open_model_id", ""))) == "omega5_event_risk_governor_20260702"
    ]
    return opens[-limit:]


def active_omega5_opens(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active: dict[str, dict[str, Any]] = {}
    for row in rows:
        trade_id = str(row.get("trade_id", "") or "")
        if not trade_id:
            continue
        is_omega5 = str(row.get("model_id", row.get("open_model_id", ""))) == "omega5_event_risk_governor_20260702"
        if row.get("kind") == "OPEN" and is_omega5:
            active[trade_id] = row
        elif row.get("kind") == "CLOSE":
            active.pop(trade_id, None)
    return list(active.values())


def current_position() -> dict[str, Any]:
    if not DASHBOARD_STATE.exists():
        return {}
    state = read_json(DASHBOARD_STATE)
    pos = state.get("position", {})
    return pos if isinstance(pos, dict) else {}


def render_markdown(payload: dict[str, Any]) -> str:
    source = payload["source_model"]
    live = payload["live_model"]
    lines = [
        "# Omega5 Live Backtest Parity Audit - 2026-07-01",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Can reproduce source backtest in live: `{payload['can_live_reproduce_source_backtest']}`",
        f"- Runtime route pass: `{payload['runtime_route_pass']}`",
        "",
        "## Contract Snapshot",
        "",
        f"- Source model: `{source['model_id']}`",
        f"- Source backtest parent: `{source['parent_model_id']}`",
        f"- Live parent model: `{live['parent_model_id']}`",
        f"- Source validation PnL: `{source['validation_pnl']:.4f}%`",
        f"- Source OOS PnL: `{source['oos_pnl']:.4f}%`",
        f"- Source avg notional val/oos: `{source['validation_avg_notional']:.4f}` / `{source['oos_avg_notional']:.4f}`",
        f"- Recent live avg notional: `{live['recent_avg_notional']:.4f}`",
        "",
        "## Checks",
        "",
    ]
    for item in payload["checks"]:
        lines.append(f"- `{item['name']}`: `{item['pass']}` ({item['severity']}) {item['details']}")
    lines.extend(
        [
            "",
            "## Required Contract",
            "",
            "- Omega5 live parity is source-immediate, not next-open.",
            "- The live path must use the Omega4.6.2 loss-cluster parent for base notional and the Omega5 reference source-policy event artifact for entry/exit timing.",
            "- The final Omega5 ledger must not be used as the live decision provider.",
            "- The historical ledger replay adapter is acceptable for audits only; it must not be used as a live future-timestamp decision provider.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{AUDIT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    trading_text = TRADING_BOT.read_text(encoding="utf-8", errors="replace")
    omega5_text = OMEGA5_LIVE.read_text(encoding="utf-8", errors="replace")
    omega121_text = OMEGA121_LIVE.read_text(encoding="utf-8", errors="replace")
    replay_text = OMEGA462_REPLAY.read_text(encoding="utf-8", errors="replace") if OMEGA462_REPLAY.exists() else ""
    source_parent_text = (
        OMEGA462_SOURCE_PARENT.read_text(encoding="utf-8", errors="replace")
        if OMEGA462_SOURCE_PARENT.exists()
        else ""
    )
    source_report = read_json(SOURCE_REPORT)
    cap220_contract = read_json(CAP220_RUNTIME_CONTRACT)
    selected = dict(source_report["selected_variant"])
    successor_requirements = dict(cap220_contract.get("promotion_requirements_for_successors", {}) or {})
    source_parent_adapter_audit = (
        read_json(SOURCE_PARENT_LIVE_ADAPTER_AUDIT) if SOURCE_PARENT_LIVE_ADAPTER_AUDIT.exists() else {}
    )
    source_parent_predictive_available = bool(
        source_parent_adapter_audit.get("verdict") == "SOURCE_PARENT_LIVE_ADAPTER_PASS"
        and source_parent_adapter_audit.get("adapter_implementation_pass") is True
        and "source_parent_live_native_adapter" in source_parent_text
        and "ledger_replay_used" in source_parent_text
        and "pd.read_csv" not in source_parent_text
    )
    source_parent_model_id = str(source_report.get("parent_model_id", ""))
    source_parent_default_on = 'FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE", True' in trading_text
    live_parent_model_id = (
        source_parent_model_id
        if source_parent_default_on and "Omega1.2.1/Omega3 parent substitution is forbidden" in trading_text
        else extract_constant(omega121_text, "OMEGA121_MODEL_ID")
    )
    rows = read_jsonl(TRADE_JOURNAL)
    recent_opens = latest_omega5_opens(rows)
    active_opens = active_omega5_opens(rows)
    recent_notional = [float(row.get("notional_exposure", 0.0) or 0.0) for row in recent_opens]
    recent_avg_notional = sum(recent_notional) / len(recent_notional) if recent_notional else 0.0
    current_pos = current_position()
    walkforward_report = read_json(RUNTIME_NATIVE_WALKFORWARD_REPORT) if RUNTIME_NATIVE_WALKFORWARD_REPORT.exists() else {}
    walkforward_pass = bool(
        walkforward_report.get("verdict") == "OMEGA5_RUNTIME_NATIVE_PROOF_PASS"
        and walkforward_report.get("runtime_native_proof_pass") is True
    )

    checks = [
        check(
            "omega5_runtime_native_uses_signal_immediate_route",
            'REPLAY_EXECUTION_ROUTE = "runtime_native_signal_immediate_maker_limit"' in (
                ROOT / "scripts/prove_omega5_runtime_native_walkforward_20260701.py"
            ).read_text(encoding="utf-8", errors="replace")
            and 'FINAL_GOVERNOR_NEXT_OPEN_EXECUTION_ENABLE"] = "0"' in (
                ROOT / "scripts/prove_omega5_runtime_native_walkforward_20260701.py"
            ).read_text(encoding="utf-8", errors="replace"),
            "must",
            {"route": "runtime_native_signal_immediate_maker_limit"},
        ),
        check(
            "omega5_live_parent_matches_source_backtest_parent",
            live_parent_model_id == source_parent_model_id,
            "blocker",
            {"live_parent": live_parent_model_id, "source_parent": source_parent_model_id},
        ),
        check(
            "omega5_adapter_declares_overlay_on_external_parent",
            "parent_decision" in omega5_text and "parent_notional" in omega5_text,
            "blocker",
            {"adapter": str(OMEGA5_LIVE), "meaning": "Omega5 sizes an externally supplied parent decision"},
        ),
        check(
            "source_parent_predictive_artifact_available",
            source_parent_predictive_available,
            "blocker",
            {
                "adapter": str(OMEGA462_SOURCE_PARENT),
                "runtime_contract": str(CAP220_RUNTIME_CONTRACT),
                "source_parent_live_adapter_audit": str(SOURCE_PARENT_LIVE_ADAPTER_AUDIT),
                "source_parent_live_adapter_verdict": source_parent_adapter_audit.get("verdict"),
                "exact_threshold_parent_predictions_required": successor_requirements.get(
                    "exact_threshold_parent_predictions_required"
                ),
                "historical_trade_ledger_fallback_allowed": successor_requirements.get(
                    "historical_trade_ledger_fallback_allowed"
                ),
                "meaning": "live parity requires runtime-native source-parent forward inference, not validation/OOS event-window replay",
            },
        ),
        check(
            "omega462_adapter_is_historical_replay_only",
            "class Omega462LedgerReplayAdapter" in replay_text
            and "LIVE_DECISION_SUPPORTED = False" in replay_text
            and "def decide_live" in replay_text
            and "historical-only" in replay_text,
            "blocker",
            {"adapter": str(OMEGA462_REPLAY), "allowed_use": "historical validation/OOS replay only"},
        ),
        check(
            "omega5_source_parent_switch_default_enabled",
            source_parent_default_on and "Omega1.2.1/Omega3 parent substitution is forbidden" in trading_text,
            "evidence",
            {"reason": "Omega5 source parent is mandatory; Omega1.2.1/Omega3 substitution is forbidden"},
        ),
        check(
            "omega5_open_journal_persists_sizing_trace",
            "omega5_sizing_trace" in trading_text
            and "omega5_parent_notional_exposure" in trading_text
            and "omega5_reference_notional_exposure" in trading_text
            and "omega5_source_parent_policy_row" in trading_text
            and "omega5_source_parent_live_native_adapter" in trading_text,
            "blocker",
            {"reason": "OPEN/CLOSE journal rows must preserve Omega5 sizing provenance"},
        ),
        check(
            "active_omega5_open_journal_trace_contract",
            all(
                isinstance(row.get("omega5_sizing_trace"), dict)
                and bool(row.get("omega5_sizing_trace"))
                and row.get("omega5_trace_present") is True
                and (
                    "omega5_source_parent_policy_row" in row
                    or "omega5_source_parent_live_native_adapter" in row
                )
                for row in active_opens
            ),
            "blocker",
            {
                "active_open_count": len(active_opens),
                "bad_trade_ids": [
                    row.get("trade_id")
                    for row in active_opens
                    if not (
                        isinstance(row.get("omega5_sizing_trace"), dict)
                        and bool(row.get("omega5_sizing_trace"))
                        and row.get("omega5_trace_present") is True
                        and (
                            "omega5_source_parent_policy_row" in row
                            or "omega5_source_parent_live_native_adapter" in row
                        )
                    )
                ],
            },
        ),
        check(
            "runtime_native_walkforward_replay_completed",
            walkforward_pass,
            "blocker",
            {
                "expected_report": str(RUNTIME_NATIVE_WALKFORWARD_REPORT),
                "observed_verdict": walkforward_report.get("verdict") if walkforward_report else None,
                "observed_pass": walkforward_report.get("runtime_native_proof_pass") if walkforward_report else None,
                "reason": "static wiring is not enough to claim source backtest PnL parity",
            },
        ),
        check(
            "recent_live_notional_matches_source_scale",
            bool(recent_notional)
            and recent_avg_notional >= 0.75 * float(selected["oos_avg_notional"]),
            "evidence",
            {
                "recent_live_avg_notional": recent_avg_notional,
                "source_oos_avg_notional": float(selected["oos_avg_notional"]),
                "recent_open_count": len(recent_opens),
            },
        ),
        check(
            "current_open_position_created_after_next_open_repair",
            str(current_pos.get("entry_price_source", "")) == "next_bar_open",
            "evidence",
            {
                "trade_id": current_pos.get("trade_id"),
                "entry_price_source": current_pos.get("entry_price_source"),
            },
        ),
    ]
    runtime_route_pass = bool(checks[0]["pass"])
    parity_blockers = [
        item["name"]
        for item in checks
        if item["severity"] == "blocker" and not bool(item["pass"])
    ]
    can_reproduce = bool(runtime_route_pass and not parity_blockers)
    payload = {
        "audit_id": "omega5_live_backtest_parity_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "LIVE_BACKTEST_PARITY_PASS" if can_reproduce else "LIVE_BACKTEST_PARITY_BLOCKED",
        "can_live_reproduce_source_backtest": can_reproduce,
        "runtime_route_pass": runtime_route_pass,
        "parity_blockers": parity_blockers,
        "source_model": {
            "model_id": source_report.get("model_id"),
            "parent_model_id": source_parent_model_id,
            "validation_pnl": float(selected["validation_pnl"]),
            "oos_pnl": float(selected["oos_pnl"]),
            "validation_avg_notional": float(selected["validation_avg_notional"]),
            "oos_avg_notional": float(selected["oos_avg_notional"]),
        },
        "live_model": {
            "model_id": extract_constant(omega5_text, "OMEGA5_MODEL_ID"),
            "parent_model_id": live_parent_model_id,
            "recent_avg_notional": float(recent_avg_notional),
            "recent_open_count": len(recent_opens),
            "current_position": current_pos,
        },
        "checks": checks,
        "artifacts": {"audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD)},
    }
    write_json(AUDIT_JSON, payload)
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "json": str(AUDIT_JSON), "markdown": str(AUDIT_MD)}, ensure_ascii=False))
    return 0 if can_reproduce else 2


if __name__ == "__main__":
    raise SystemExit(main())
