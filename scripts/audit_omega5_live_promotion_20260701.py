#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from trading_bot_modules.omega5_live import (
    OMEGA5_EVENT_RISK_POLICY_ID,
    OMEGA5_EVENT_RISK_SHOCK_NOTIONAL_SCALE,
    OMEGA5_FIRST_VETO_THRESHOLD,
    OMEGA5_LONG_SL_PRICE_MOVE,
    OMEGA5_LONG_TP_PRICE_MOVE,
    OMEGA5_MAX_HOLD_BARS,
    OMEGA5_MODEL_ID,
    OMEGA5_MODEL_VERSION,
    OMEGA5_OWNER,
    OMEGA5_PARENT_MODEL_ID,
    OMEGA5_SHORT_SL_PRICE_MOVE,
    OMEGA5_SHORT_TP_PRICE_MOVE,
    Omega5LiveAdapter,
)


OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega5_live_promotion_20260701"
AUDIT_JSON = OUT_DIR / "omega5_live_promotion_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega5_live_promotion_20260701.md"
MODEL_CONTRACT_MD = ROOT / "docs/model_contracts/omega5_event_risk_governor_20260702_contract.md"
ACTIVE_LIVE_MD = ROOT / "docs/active_live/omega5_live_stack.md"


PATHS = {
    "report_path": ROOT / "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/report.json",
    "feature_veto_report_path": ROOT / "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/report.json",
    "two_stage_veto_report_path": ROOT / "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/report.json",
    "pnl_tilt_report_path": ROOT / "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/report.json",
    "redteam_path": ROOT / "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/redteam_audit_20260701.json",
    "frontier_audit_path": ROOT / "tmp/causal_regen_20260516/omega4_6_2_frontier_leakage_redteam_20260701/frontier_leakage_redteam_20260701.json",
    "cvp_audit_path": ROOT / "tmp/causal_regen_20260516/cvp_feature_causality_20260701/cvp_feature_causality_20260701.json",
    "artifact_integrity_path": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/omega_artifact_integrity_audit_20260630.json",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "status": "pass" if ok else "fail", "detail": detail}


def decision_dict(decision: object) -> dict:
    if is_dataclass(decision):
        return asdict(decision)
    return dict(vars(decision))


def parent_decision(*, action: int, side: int, notional: float) -> SimpleNamespace:
    return SimpleNamespace(
        action=action,
        side=side,
        notional_exposure=notional,
        leverage=2.0 if side else 1.0,
        position_fraction=(notional / 2.0) if side else 0.0,
        take_profit=0.0,
        stop_loss=0.0,
        max_hold_bars=0,
        cooldown_bars=0,
        quality_score=0.75 if side else 0.0,
        confidence=0.80 if side else 0.0,
        router_expert="audit",
        trace={"audit_parent": True, "model_id": OMEGA5_PARENT_MODEL_ID, "ledger_replay_used": False},
    )


def neutral_frame(*, timestamp: str = "2026-03-10 00:00:00", close: float = 2000.0) -> pd.DataFrame:
    end = pd.Timestamp(timestamp)
    rows = []
    for i in range(60):
        rows.append(
            {
                "timestamp": end - pd.Timedelta(minutes=5 * (59 - i)),
                "close": float(close),
                "bb_width": 0.01,
                "m7_prob_up": 0.10,
                "jump_flag": 0.0,
                "evt_tail_flag": 0.0,
                "jump_z": 0.0,
            }
        )
    return pd.DataFrame(rows)


def short_veto_frame() -> pd.DataFrame:
    out = neutral_frame()
    out.loc[out.index[-1], "bb_width"] = OMEGA5_FIRST_VETO_THRESHOLD / 2.0
    return out


def macro_veto_frame() -> pd.DataFrame:
    return neutral_frame(timestamp="2026-06-23 13:15:00")


def shock_haircut_frame() -> pd.DataFrame:
    out = neutral_frame()
    out.loc[out.index[-1], "jump_z"] = 3.1
    return out


def write_docs(audit: dict) -> None:
    selected = audit["source_report"]["selected_variant"]
    lines = [
        "# Omega5 Live Promotion Audit",
        "",
        f"- Created: `{audit['created_at']}`",
        f"- Promotion pass: `{audit['promotion_pass']}`",
        f"- Live model id: `{OMEGA5_MODEL_ID}`",
        f"- Model version: `{OMEGA5_MODEL_VERSION}`",
        f"- Owner: `{OMEGA5_OWNER}`",
        "",
        "## Checks",
        "",
    ]
    for item in audit["checks"]:
        lines.append(f"- `{item['name']}`: `{item['status']}` - {item['detail']}")
    lines.extend(
        [
            "",
            "## Selected Contract",
            "",
            f"- Source model: `{audit['source_report']['model_id']}`",
            f"- Reference model: `{audit['source_report']['reference_model_id']}`",
            f"- Exposure spec: `{selected['exposure_spec']}`",
            f"- Long factor: `{selected['exposure_long_factor']}`",
            f"- Short factor: `{selected['exposure_short_factor']}`",
            f"- Notional cap: `{selected['exposure_cap_notional']}`",
            f"- Leverage cap: `{selected['exposure_leverage_cap']}`",
            f"- Max hold bars: `{OMEGA5_MAX_HOLD_BARS}`",
        ]
    )
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    contract = [
        "# Omega5 Event-Risk Governor Live Contract",
        "",
        f"- Model id: `{OMEGA5_MODEL_ID}`",
        f"- Model version: `{OMEGA5_MODEL_VERSION}`",
        "- Live role: Omega5 promotion layer on top of the Omega4.6.2 source parent policy.",
        f"- Source Omega4.6.2 model: `{audit['source_report']['model_id']}`",
        f"- Source red-team verdict: `{audit['redteam']['verdict']}`",
        f"- Artifact integrity promotion pass: `{audit['artifact_integrity']['promotion_pass']}`",
        "",
        "## Entry",
        "",
        "- Parent cash remains cash.",
        "- Parent long uses long exposure factor and no short veto.",
        "- Parent short is vetoed when `bb_width <= 0.003939593535185601`.",
        "- Parent short is also vetoed when `m7_prob_up >= 0.909727596`.",
        "- Scheduled macro entry veto blocks new entries from 30 minutes before to 120 minutes after rule-based NFP/ISM/S&P Global PMI/FOMC events.",
        "- Shock haircut scales new-entry notional by `0.50` when `jump_flag`, `evt_tail_flag`, `abs(jump_z) >= 3.0`, `abs(1h return) >= 3%`, or `abs(4h return) >= 4%` fires.",
        "",
        "## Risk",
        "",
        "- `reference_notional = min(parent_notional * reference_side_factor, 4.2, 5.0 * 1.0)`.",
        "- `notional = min(reference_notional * final_side_factor, 4.4, 5.0 * 1.0)`.",
        "- `leverage = 5.0`.",
        "- `margin_fraction = notional / leverage`.",
        "- Long TP/SL price moves: `0.020 / 0.030`.",
        "- Short TP/SL price moves: `0.025 / 0.0385`.",
        "- Runtime TP/SL thresholds are account-PnL thresholds: `price_move * notional`.",
        "- Max hold: `8h = 96 five-minute bars`.",
    ]
    MODEL_CONTRACT_MD.parent.mkdir(parents=True, exist_ok=True)
    MODEL_CONTRACT_MD.write_text("\n".join(contract) + "\n", encoding="utf-8")

    active = [
        "# Omega5 Live Stack",
        "",
        f"- Active promoted live model: `{OMEGA5_MODEL_ID}`",
        "- Trading bot integration: `FinalGovernorRuntime` loads `Omega5LiveAdapter` by default.",
        "- Decision priority: Omega5 runs before the legacy Omega1.2.1 entry path.",
        "- Existing open Omega5 positions are recovered from the trade journal by `omega5|...` source or Omega5 model id.",
        f"- Promotion audit: `{AUDIT_MD}`",
        f"- Model contract: `{MODEL_CONTRACT_MD}`",
    ]
    ACTIVE_LIVE_MD.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_LIVE_MD.write_text("\n".join(active) + "\n", encoding="utf-8")


def main() -> int:
    checks: list[dict] = []
    trading_bot_text = (ROOT / "trading_bot.py").read_text(encoding="utf-8")
    checks.append(check("trading_bot_imports_omega5", "Omega5LiveAdapter" in trading_bot_text, "Omega5 adapter import present"))
    decide_idx = trading_bot_text.index("    def decide(")
    omega5_idx = trading_bot_text.index("if self.omega5_adapter is not None:", decide_idx)
    omega1_idx = trading_bot_text.index("if self.omega1_2_1_adapter is not None:", decide_idx)
    checks.append(check("omega5_precedes_omega1_entry_path", omega5_idx < omega1_idx, "Omega5 decision block precedes Omega1.2.1"))

    adapter = Omega5LiveAdapter(**PATHS)
    checks.append(check("omega5_adapter_initializes", adapter.cap_notional == 4.4 and adapter.leverage_cap == 5.0, "adapter contract loaded"))

    cash = adapter.decide_latest(pd.DataFrame([{}]), parent_decision(action=0, side=0, notional=0.0))
    checks.append(check("parent_cash_stays_cash", cash.action == 0 and cash.side == 0, str(decision_dict(cash))))

    long_dec = adapter.decide_latest(neutral_frame(), parent_decision(action=1, side=1, notional=2.0))
    expected_reference_long = min(
        2.0 * adapter.reference_long_factor,
        adapter.reference_cap_notional,
        adapter.leverage_cap * adapter.max_margin_fraction,
    )
    expected_long_notional = min(
        expected_reference_long * adapter.long_factor,
        adapter.cap_notional,
        adapter.leverage_cap * adapter.max_margin_fraction,
    )
    long_ok = (
        long_dec.action == 1
        and abs(long_dec.notional_exposure - expected_long_notional) <= 1e-12
        and abs(long_dec.position_fraction - (expected_long_notional / adapter.leverage_cap)) <= 1e-12
        and abs(long_dec.take_profit - (OMEGA5_LONG_TP_PRICE_MOVE * expected_long_notional)) <= 1e-12
        and abs(long_dec.stop_loss - (OMEGA5_LONG_SL_PRICE_MOVE * expected_long_notional)) <= 1e-12
        and long_dec.max_hold_bars == OMEGA5_MAX_HOLD_BARS
    )
    checks.append(check("long_sizing_tp_sl_contract", long_ok, str(decision_dict(long_dec))))

    short_veto = adapter.decide_latest(short_veto_frame(), parent_decision(action=2, side=-1, notional=2.0))
    checks.append(
        check(
            "short_first_stage_veto_contract",
            short_veto.action == 0 and short_veto.trace.get("omega5_reason") == "short_first_stage_veto",
            str(decision_dict(short_veto)),
        )
    )

    macro_veto = adapter.decide_latest(macro_veto_frame(), parent_decision(action=1, side=1, notional=2.0))
    checks.append(
        check(
            "event_risk_macro_entry_veto_contract",
            macro_veto.action == 0
            and macro_veto.trace.get("omega5_reason") == "macro_event_entry_veto"
            and (macro_veto.trace.get("event_risk") or {}).get("macro_entry_veto") is True,
            str(decision_dict(macro_veto)),
        )
    )

    shock_dec = adapter.decide_latest(shock_haircut_frame(), parent_decision(action=1, side=1, notional=2.0))
    shock_expected = expected_long_notional * OMEGA5_EVENT_RISK_SHOCK_NOTIONAL_SCALE
    shock_trace = dict(shock_dec.trace or {})
    checks.append(
        check(
            "event_risk_shock_haircut_contract",
            shock_dec.action == 1
            and shock_trace.get("omega5_reason") == "entry_shock_haircut"
            and (shock_trace.get("event_risk") or {}).get("policy_id") == OMEGA5_EVENT_RISK_POLICY_ID
            and abs(shock_dec.notional_exposure - shock_expected) <= 1e-12
            and abs(shock_dec.position_fraction - (shock_expected / adapter.leverage_cap)) <= 1e-12,
            str(decision_dict(shock_dec)),
        )
    )

    source_report = read_json(PATHS["report_path"])
    redteam = read_json(PATHS["redteam_path"])
    artifact_integrity = read_json(PATHS["artifact_integrity_path"])
    checks.append(
        check(
            "source_selection_oos_independent",
            bool(source_report["selected_variant"].get("oos_used_in_selection")) is False,
            "selected_variant.oos_used_in_selection=false",
        )
    )
    checks.append(
        check(
            "source_redteam_full_live_pass",
            redteam.get("verdict") == "FULL_LIVE_PASS_VALIDATION_ONLY" and bool(redteam.get("full_live_pass")),
            str(redteam.get("verdict")),
        )
    )
    checks.append(
        check(
            "source_artifact_integrity_pass",
            bool(artifact_integrity.get("promotion_pass")),
            str(artifact_integrity.get("promotion_pass")),
        )
    )

    promotion_pass = all(item["status"] == "pass" for item in checks)
    audit = {
        "audit_id": "omega5_live_promotion_20260701",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "promotion_pass": bool(promotion_pass),
        "checks": checks,
        "source_report": source_report,
        "redteam": redteam,
        "artifact_integrity": artifact_integrity,
        "paths": {k: str(v) for k, v in PATHS.items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_docs(audit)
    print(json.dumps({"promotion_pass": promotion_pass, "json": str(AUDIT_JSON), "markdown": str(AUDIT_MD)}, ensure_ascii=False, indent=2))
    return 0 if promotion_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
