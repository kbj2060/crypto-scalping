#!/usr/bin/env python3
"""Audit Omega4.6.2 source-parent live-native adapter wiring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRADING_BOT = ROOT / "trading_bot.py"
REPLAY_ADAPTER = ROOT / "trading_bot_modules/omega4_6_2_runtime_adapter.py"
SOURCE_PARENT_ADAPTER = ROOT / "trading_bot_modules/omega4_6_2_source_parent_live.py"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_2_source_parent_live_adapter_audit_20260701"
AUDIT_JSON = OUT_DIR / "omega4_6_2_source_parent_live_adapter_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_source_parent_live_adapter_20260701.md"


def check(name: str, passed: bool, severity: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "severity": severity, "details": details}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Omega4.6.2 Source Parent Live Adapter Audit - 2026-07-01",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Adapter implementation pass: `{payload['adapter_implementation_pass']}`",
        "",
        "## Checks",
        "",
    ]
    for item in payload["checks"]:
        lines.append(f"- `{item['name']}`: `{item['pass']}` ({item['severity']}) {item['details']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The selected Omega4.6.2 ledger replay adapter is now explicitly historical-only.",
            "- The new source-parent adapter is a forward policy path: it loads the two TabM bundles and applies h48qual/zig075 routing, cap220 exposure, and source-parent exposure/governor logic without reading validation/OOS ledgers.",
            "- The trading bot switch is default-on and Omega5 fails fast if the source parent is disabled or missing.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{AUDIT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    trading_text = TRADING_BOT.read_text(encoding="utf-8", errors="replace")
    replay_text = REPLAY_ADAPTER.read_text(encoding="utf-8", errors="replace")
    source_text = SOURCE_PARENT_ADAPTER.read_text(encoding="utf-8", errors="replace")
    checks = [
        check(
            "ledger_replay_live_decision_blocked",
            "LIVE_DECISION_SUPPORTED = False" in replay_text
            and "def decide_live" in replay_text
            and "historical-only" in replay_text,
            "must",
            {"adapter": str(REPLAY_ADAPTER)},
        ),
        check(
            "source_parent_live_adapter_exists",
            "class Omega462SourceParentLiveAdapter" in source_text
            and "OMEGA462_SOURCE_PARENT_MODEL_ID" in source_text,
            "must",
            {"adapter": str(SOURCE_PARENT_ADAPTER)},
        ),
        check(
            "source_parent_adapter_does_not_read_ledgers",
            "selected_validation_ledger" not in source_text
            and "selected_oos_ledger" not in source_text
            and "pd.read_csv" not in source_text,
            "must",
            {"forbidden": ["selected_validation_ledger", "selected_oos_ledger", "pd.read_csv"]},
        ),
        check(
            "source_parent_uses_predictive_artifacts",
            "true_3head_tabm_bundle.pt" in source_text
            and "torch.load" in source_text
            and "self.components" in source_text
            and "predict(" in source_text,
            "must",
            {"components": ["h48qual", "zig075"]},
        ),
        check(
            "source_parent_reconstructs_policy_contract",
            ("cap220_notional" in source_text or "CAP220_NOTIONAL_CAP" in source_text)
            and "source_parent_side_factor" in source_text
            and "loss_governor_scale" in source_text
            and ("short_rsi_gate" in source_text or "CAP220_SHORT_RSI_THRESHOLD" in source_text),
            "must",
            {"policy_layers": ["cap220", "fine_exposure", "loss_governor", "short_rsi_gate"]},
        ),
        check(
            "trading_bot_has_guarded_source_parent_switch",
            'FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE", True' in trading_text
            and "Omega462SourceParentLiveAdapter" in trading_text
            and "Omega1.2.1/Omega3 parent substitution is forbidden" in trading_text,
            "must",
            {"default": "enabled; Omega5 fails fast if source parent is disabled or missing"},
        ),
        check(
            "omega5_entry_uses_selected_parent_provider",
            "def _omega5_parent_decision" in trading_text
            and "self.omega4_6_2_source_parent_adapter.decide_latest(frame)" in trading_text
            and "self.omega5_adapter.decide_latest(frame, parent_dec)" in trading_text,
            "must",
            {"entrypoint": "FinalGovernorRuntime._decide_omega5_entry"},
        ),
    ]
    failed = [item["name"] for item in checks if item["severity"] == "must" and not item["pass"]]
    payload = {
        "audit_id": "omega4_6_2_source_parent_live_adapter_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "SOURCE_PARENT_LIVE_ADAPTER_PASS" if not failed else "SOURCE_PARENT_LIVE_ADAPTER_BLOCKED",
        "adapter_implementation_pass": not failed,
        "failed_checks": failed,
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
