#!/usr/bin/env python3
"""Static audit for exact Omega5 trading-bot wiring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRADING_BOT = ROOT / "trading_bot.py"
OMEGA5_LIVE = ROOT / "trading_bot_modules/omega5_live.py"
SOURCE_PARENT = ROOT / "trading_bot_modules/omega4_6_2_source_parent_live.py"
LEDGER_REPLAY = ROOT / "trading_bot_modules/omega4_6_2_runtime_adapter.py"
RUNTIME_NATIVE_WALKFORWARD_REPORT = (
    ROOT
    / "tmp/causal_regen_20260516/omega5_source_parent_runtime_native_walkforward_20260701"
    / "report.json"
)
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega5_trading_bot_exact_wiring_audit_20260701"
AUDIT_JSON = OUT_DIR / "omega5_trading_bot_exact_wiring_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega5_trading_bot_exact_wiring_20260701.md"


def check(name: str, passed: bool, severity: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "severity": severity, "details": details}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_method(text: str, name: str) -> str:
    marker = f"    def {name}"
    start = text.find(marker)
    if start < 0:
        return ""
    end = text.find("\n    def ", start + len(marker))
    if end < 0:
        end = len(text)
    return text[start:end]


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Omega5 Trading Bot Exact Wiring Audit - 2026-07-01",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Exact static wiring pass: `{payload['exact_static_wiring_pass']}`",
        "",
        "## Checks",
        "",
    ]
    for item in payload["checks"]:
        lines.append(f"- `{item['name']}`: `{item['pass']}` ({item['severity']}) {item['details']}")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- This audit verifies trading-bot wiring and fail-fast contracts.",
            "- It does not replace a runtime-native walk-forward backtest.",
            "- Historical ledger replay remains audit-only and is forbidden as a live decision provider.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{AUDIT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    trading = TRADING_BOT.read_text(encoding="utf-8", errors="replace")
    omega5 = OMEGA5_LIVE.read_text(encoding="utf-8", errors="replace")
    source_parent = SOURCE_PARENT.read_text(encoding="utf-8", errors="replace")
    ledger = LEDGER_REPLAY.read_text(encoding="utf-8", errors="replace")
    walkforward_report = read_json(RUNTIME_NATIVE_WALKFORWARD_REPORT)
    parent_method = extract_method(trading, "_omega5_parent_decision")
    decide_method = extract_method(trading, "decide")
    recover_method = extract_method(trading, "_recover_omega5_state_from_open_journal")
    manage_method = extract_method(trading, "_manage_omega5_position")

    omega5_idx = decide_method.find("if self.omega5_adapter is not None:")
    omega121_idx = decide_method.find("if self.omega1_2_1_adapter is not None:")

    checks = [
        check(
            "omega5_source_parent_default_on",
            'FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE", True' in trading,
            "must",
            {"expected": "source parent is enabled by default when Omega5 is enabled"},
        ),
        check(
            "omega5_requires_source_parent_no_substitute",
            "Omega1.2.1/Omega3 parent substitution is forbidden" in trading
            and "requires either Omega4.6.2 source parent or Omega1.2.1 parent adapter" not in trading,
            "must",
            {"forbidden_substitute": "Omega1.2.1/Omega3 parent"},
        ),
        check(
            "omega5_parent_provider_only_source_parent",
            "self.omega4_6_2_source_parent_adapter.decide_latest(frame)" in parent_method
            and "self.omega1_2_1_adapter.decide_latest(frame)" not in parent_method,
            "must",
            {"method": "FinalGovernorRuntime._omega5_parent_decision"},
        ),
        check(
            "omega5_adapter_validates_parent_identity",
            "parent_trace_model_id != OMEGA5_PARENT_MODEL_ID" in omega5
            and "ledger_replay_used" in omega5
            and "ledger replay parent decisions are forbidden" in omega5,
            "must",
            {"adapter": str(OMEGA5_LIVE)},
        ),
        check(
            "source_parent_uses_runtime_native_forward_artifact",
            "source_parent_live_native_adapter" in source_parent
            and "source_parent_predictive_artifact" in source_parent
            and "tabm_bundle+risk_sidecar_runtime_forward" in source_parent
            and "true_3head_tabm_bundle.pt" in source_parent
            and "torch.load" in source_parent
            and "risk_sidecar" in source_parent
            and "component_predictions" in source_parent
            and "CAP220_SHORT_RSI_THRESHOLD" in source_parent
            and "loss_governor_scale" in source_parent
            and '"ledger_replay_used": False' in source_parent,
            "must",
            {
                "adapter": str(SOURCE_PARENT),
                "meaning": "uses causal TabM bundle + risk sidecar runtime inference, not validation/OOS event replay",
            },
        ),
        check(
            "source_parent_forbids_historical_replay_provider",
            "pd.read_csv" not in source_parent
            and "selected_validation_ledger" not in source_parent
            and "_reference_entry_event_for_timestamp" not in source_parent
            and "_parent_interval_for_timestamp" not in source_parent
            and '"reference_policy_entry_event_adapter": True' not in source_parent
            and "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701" not in source_parent,
            "must",
            {
                "adapter": str(SOURCE_PARENT),
                "forbidden": "historical validation/OOS ledger or interval replay as live decision source",
            },
        ),
        check(
            "ledger_replay_live_path_blocked",
            "LIVE_DECISION_SUPPORTED = False" in ledger
            and "def decide_live" in ledger
            and "historical-only" in ledger,
            "must",
            {"adapter": str(LEDGER_REPLAY)},
        ),
        check(
            "omega5_open_journal_recovery_requires_contract_fields",
            "missing_contract_fields" in recover_method
            and "notional_exposure" in recover_method
            and "execution_leverage" in recover_method
            and "effective_take_profit" in recover_method
            and "effective_stop_loss" in recover_method
            and "omega5_sizing_trace" in recover_method
            and "missing source-parent policy provenance" in recover_method
            and "current_leverage" not in recover_method,
            "must",
            {"method": "FinalGovernorRuntime._recover_omega5_state_from_open_journal"},
        ),
        check(
            "omega5_runtime_state_load_fail_fast_no_trace_fallback",
            "Omega5 runtime state contract mismatch" in trading
            and 'data.get("active_omega5_sizing_trace", self.active_omega5_sizing_trace)' not in trading
            and 'open_row.get("omega5_sizing_trace", self.active_omega5_sizing_trace)' not in trading,
            "must",
            {
                "method": "FinalGovernorRuntime._load_runtime_state + _recover_omega5_state_from_open_journal",
                "reason": "Omega5 active state must fail fast when sizing provenance is absent",
            },
        ),
        check(
            "omega5_active_position_no_reconcile_fallback",
            "omega5_reconcile_close" not in manage_method
            and "reconcile close fallback is forbidden" in manage_method
            and "self.active_omega5_notional" in manage_method
            and "meta_router.current_leverage or self.active_omega5_notional" not in manage_method,
            "must",
            {"method": "FinalGovernorRuntime._manage_omega5_position"},
        ),
        check(
            "omega5_source_exit_event_is_runtime_owner",
            "active_omega5_source_exit_reason" in trading
            and "active_omega5_source_exit_price_move" in trading
            and "omega5_source_roundtrip_cost" in trading
            and "source_exit_armed" in manage_method,
            "must",
            {"method": "FinalGovernorRuntime._manage_omega5_position"},
        ),
        check(
            "omega5_decision_priority_before_omega121",
            omega5_idx >= 0 and omega121_idx >= 0 and omega5_idx < omega121_idx,
            "must",
            {"method": "FinalGovernorRuntime.decide"},
        ),
        check(
            "omega5_entry_persists_risk_contract",
            "self.active_omega5_take_profit = float(dec.take_profit)" in trading
            and "self.active_omega5_stop_loss = float(dec.stop_loss)" in trading
            and "self.active_omega5_max_hold_bars = int(dec.max_hold_bars)" in trading
            and "self.active_omega5_notional = float(dec.notional_exposure)" in trading
            and "self.active_omega5_leverage = float(dec.leverage)" in trading,
            "must",
            {"method": "FinalGovernorRuntime._decide_omega5_entry"},
        ),
        check(
            "omega5_entry_persists_sizing_trace_contract",
            "self.active_omega5_sizing_trace = dict(GovernorPositionRouter._journal_jsonable(trace))" in trading
            and "active_omega5_sizing_trace" in trading
            and "omega5_sizing_trace" in trading
            and "omega5_parent_notional_exposure" in trading
            and "omega5_reference_notional_exposure" in trading
            and "omega5_source_parent_policy_row" in trading,
            "must",
            {
                "method": "FinalGovernorRuntime._decide_omega5_entry + GovernorPositionRouter._journal_audit_fields",
                "reason": "Omega5 OPEN/CLOSE journal rows must retain sizing provenance",
            },
        ),
        check(
            "omega5_entry_persists_live_native_source_parent_provenance",
            "omega5_source_parent_live_native_adapter" in trading
            and "omega5_source_parent_predictive_artifact" in trading
            and "omega5_source_parent_component_bundle" in trading
            and "omega5_source_parent_component_sidecar" in trading
            and "omega5_source_parent_loss_governor_scale" in trading,
            "must",
            {
                "method": "GovernorPositionRouter._journal_audit_fields",
                "reason": "live-native Omega5 rows must retain source model and sidecar provenance",
            },
        ),
        check(
            "omega5_risk_math_uses_price_move_times_notional",
            "take_profit = tp_move * notional" in omega5
            and "stop_loss = sl_move * notional" in omega5
            and "margin_fraction = notional / max(leverage" in omega5,
            "must",
            {"adapter": str(OMEGA5_LIVE)},
        ),
        check(
            "trading_bot_does_not_import_ledger_replay_adapter",
            "Omega462LedgerReplayAdapter" not in trading,
            "must",
            {"file": str(TRADING_BOT)},
        ),
        check(
            "runtime_native_walkforward_full_pass",
            walkforward_report.get("verdict") == "OMEGA5_RUNTIME_NATIVE_PROOF_PASS"
            and bool(walkforward_report.get("runtime_native_proof_pass")),
            "must",
            {"report": str(RUNTIME_NATIVE_WALKFORWARD_REPORT)},
        ),
    ]
    failed = [item["name"] for item in checks if item["severity"] == "must" and not item["pass"]]
    payload = {
        "audit_id": "omega5_trading_bot_exact_wiring_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "OMEGA5_EXACT_STATIC_WIRING_PASS" if not failed else "OMEGA5_EXACT_STATIC_WIRING_BLOCKED",
        "exact_static_wiring_pass": not failed,
        "failed_checks": failed,
        "runtime_native_walkforward_executed": True,
        "checks": checks,
        "artifacts": {"audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD)},
    }
    write_json(AUDIT_JSON, payload)
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "json": str(AUDIT_JSON), "markdown": str(AUDIT_MD)}, ensure_ascii=False))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
