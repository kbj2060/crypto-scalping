from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


OMEGA5_MODEL_ID = "omega5_event_risk_governor_20260702"
OMEGA5_MODEL_VERSION = "Omega5-event-risk-governor-20260702"
OMEGA5_OWNER = "omega5"
OMEGA5_SOURCE_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701"
OMEGA5_PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
OMEGA5_REFERENCE_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701"
OMEGA5_MAX_HOLD_HOURS = 8.0
OMEGA5_BARS_PER_HOUR = 12
OMEGA5_MAX_HOLD_BARS = int(OMEGA5_MAX_HOLD_HOURS * OMEGA5_BARS_PER_HOUR)
OMEGA5_LONG_TP_PRICE_MOVE = 0.020
OMEGA5_LONG_SL_PRICE_MOVE = 0.030
OMEGA5_SHORT_TP_PRICE_MOVE = 0.025
OMEGA5_SHORT_SL_PRICE_MOVE = 0.0385
OMEGA5_FIRST_VETO_FEATURE = "bb_width"
OMEGA5_FIRST_VETO_OP = "<="
OMEGA5_FIRST_VETO_THRESHOLD = 0.003939593535185601
OMEGA5_SECOND_VETO_FEATURE = "m7_prob_up"
OMEGA5_SECOND_VETO_OP = ">="
OMEGA5_SECOND_VETO_THRESHOLD = 0.909727596
OMEGA5_EVENT_RISK_POLICY_ID = "omega5_event_risk_macro_skip_shock50_20260702"
OMEGA5_EVENT_RISK_MACRO_PRE_MINUTES = 30
OMEGA5_EVENT_RISK_MACRO_POST_MINUTES = 120
OMEGA5_EVENT_RISK_SHOCK_NOTIONAL_SCALE = 0.50
OMEGA5_EVENT_RISK_SHOCK_JUMP_Z_THRESHOLD = 3.0
OMEGA5_EVENT_RISK_SHOCK_RET_1H_THRESHOLD = 0.030
OMEGA5_EVENT_RISK_SHOCK_RET_4H_THRESHOLD = 0.040
OMEGA5_EVENT_RISK_FOMC_DECISION_DATES = {
    2026: ("2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09")
}
OMEGA5_LIVE_PROMOTION_STATUS = "blocked_ledger_contamination_20260702"
OMEGA5_LIVE_PROMOTION_BLOCK_REASON = (
    "Omega5 live adapter is blocked: side-thread audit found validation/test "
    "ledger dependence in the promoted model-selection path."
)


@dataclass(frozen=True)
class Omega5Decision:
    action: int
    side: int
    notional_exposure: float
    leverage: float
    position_fraction: float
    take_profit: float
    stop_loss: float
    max_hold_bars: int
    cooldown_bars: int
    quality_score: float
    confidence: float
    trace: dict[str, Any]


class Omega5LiveAdapter:
    def __init__(
        self,
        *,
        report_path: str | Path,
        feature_veto_report_path: str | Path,
        two_stage_veto_report_path: str | Path,
        pnl_tilt_report_path: str | Path,
        redteam_path: str | Path,
        frontier_audit_path: str | Path,
        cvp_audit_path: str | Path,
        artifact_integrity_path: str | Path,
    ) -> None:
        raise RuntimeError(OMEGA5_LIVE_PROMOTION_BLOCK_REASON)
        self.report_path = Path(report_path)
        self.feature_veto_report_path = Path(feature_veto_report_path)
        self.two_stage_veto_report_path = Path(two_stage_veto_report_path)
        self.pnl_tilt_report_path = Path(pnl_tilt_report_path)
        self.redteam_path = Path(redteam_path)
        self.frontier_audit_path = Path(frontier_audit_path)
        self.cvp_audit_path = Path(cvp_audit_path)
        self.artifact_integrity_path = Path(artifact_integrity_path)
        self.report = self._read_json(self.report_path)
        self.feature_veto_report = self._read_json(self.feature_veto_report_path)
        self.two_stage_veto_report = self._read_json(self.two_stage_veto_report_path)
        self.pnl_tilt_report = self._read_json(self.pnl_tilt_report_path)
        self.redteam = self._read_json(self.redteam_path)
        self.frontier_audit = self._read_json(self.frontier_audit_path)
        self.cvp_audit = self._read_json(self.cvp_audit_path)
        self.artifact_integrity = self._read_json(self.artifact_integrity_path)
        self._validate_contract()
        selected = dict(self.report["selected_variant"])
        pnl = dict(self.pnl_tilt_report["selected_variant"])
        self.reference_long_factor = float(pnl["exposure_long_factor"])
        self.reference_short_factor = float(pnl["exposure_short_factor"])
        self.reference_cap_notional = float(pnl["exposure_cap_notional"])
        self.long_factor = float(selected["exposure_long_factor"])
        self.short_factor = float(selected["exposure_short_factor"])
        self.cap_notional = float(selected["exposure_cap_notional"])
        self.leverage_cap = float(selected["exposure_leverage_cap"])
        self.max_margin_fraction = float(selected["exposure_max_margin_fraction"])

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise RuntimeError(f"Omega5 required artifact is missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _require(value: bool, message: str) -> None:
        if not bool(value):
            raise RuntimeError(message)

    def _validate_contract(self) -> None:
        self._require(self.report.get("model_id") == OMEGA5_SOURCE_MODEL_ID, "Omega5 source model id mismatch")
        self._require(self.report.get("parent_model_id") == OMEGA5_PARENT_MODEL_ID, "Omega5 parent model id mismatch")
        self._require(self.report.get("reference_model_id") == OMEGA5_REFERENCE_MODEL_ID, "Omega5 reference model id mismatch")
        selected = self.report.get("selected_variant")
        self._require(isinstance(selected, dict), "Omega5 report missing selected_variant")
        self._require(bool(selected.get("research_validation_only_gate_pass")), "Omega5 validation-only gate did not pass")
        self._require(bool(selected.get("validation_two_stage_exposure_gate_pass")), "Omega5 exposure gate did not pass")
        self._require(bool(selected.get("oos_safety_gate_pass")), "Omega5 OOS safety gate did not pass")
        self._require(bool(selected.get("oos_used_in_selection")) is False, "Omega5 selection used OOS")
        self._require(float(selected.get("exposure_leverage_cap", 0.0)) == 5.0, "Omega5 leverage cap contract mismatch")
        self._require(float(selected.get("exposure_cap_notional", 0.0)) == 4.4, "Omega5 notional cap contract mismatch")
        self._require(float(selected.get("exposure_max_margin_fraction", 0.0)) == 1.0, "Omega5 margin cap contract mismatch")

        self._require(
            self.redteam.get("verdict") == "FULL_LIVE_PASS_VALIDATION_ONLY" and bool(self.redteam.get("full_live_pass")),
            "Omega5 red-team full live pass is missing",
        )
        self._require(
            self.frontier_audit.get("verdict") == "FRONTIER_LEAKAGE_RUNTIME_PASS"
            and bool(self.frontier_audit.get("full_live_pass")),
            "Omega5 frontier leakage/runtime pass is missing",
        )
        self._require(
            self.cvp_audit.get("verdict") == "CVP_FEATURE_CAUSALITY_PASS",
            "Omega5 CVP feature causality pass is missing",
        )
        self._require(
            bool(self.artifact_integrity.get("promotion_pass")),
            "Omega5 source artifact integrity promotion_pass is missing",
        )

        first = dict(self.feature_veto_report.get("selected_variant") or {})
        second = dict(self.two_stage_veto_report.get("selected_variant") or {})
        pnl = dict(self.pnl_tilt_report.get("selected_variant") or {})
        self._require(
            first.get("feature_name") == OMEGA5_FIRST_VETO_FEATURE
            and first.get("feature_op") == OMEGA5_FIRST_VETO_OP
            and abs(float(first.get("feature_threshold")) - OMEGA5_FIRST_VETO_THRESHOLD) <= 1e-15,
            "Omega5 first-stage veto contract mismatch",
        )
        self._require(
            second.get("feature_name") == OMEGA5_SECOND_VETO_FEATURE
            and second.get("feature_op") == OMEGA5_SECOND_VETO_OP
            and abs(float(second.get("feature_threshold")) - OMEGA5_SECOND_VETO_THRESHOLD) <= 1e-15,
            "Omega5 second-stage veto contract mismatch",
        )
        self._require(float(pnl.get("roll8_max_hours", 0.0)) == OMEGA5_MAX_HOLD_HOURS, "Omega5 max-hold contract mismatch")
        self._require(float(pnl.get("roll8_long_tp_move", 0.0)) == OMEGA5_LONG_TP_PRICE_MOVE, "Omega5 long TP contract mismatch")
        self._require(float(pnl.get("roll8_long_sl_move", 0.0)) == OMEGA5_LONG_SL_PRICE_MOVE, "Omega5 long SL contract mismatch")
        self._require(float(pnl.get("roll8_short_tp_move", 0.0)) == OMEGA5_SHORT_TP_PRICE_MOVE, "Omega5 short TP contract mismatch")
        self._require(float(pnl.get("roll8_short_sl_move", 0.0)) == OMEGA5_SHORT_SL_PRICE_MOVE, "Omega5 short SL contract mismatch")
        self._require(float(pnl.get("exposure_cap_notional", 0.0)) == 4.2, "Omega5 reference cap contract mismatch")

    @staticmethod
    def _latest_float(frame: pd.DataFrame, feature: str) -> float:
        if not len(frame):
            raise RuntimeError("Omega5 received empty feature frame")
        if feature not in frame.columns:
            raise RuntimeError(f"Omega5 missing required feature: {feature}")
        value = float(frame.iloc[-1][feature])
        if not np.isfinite(value):
            raise RuntimeError(f"Omega5 non-finite required feature: {feature}")
        return value

    @staticmethod
    def _latest_timestamp(frame: pd.DataFrame) -> pd.Timestamp:
        if not len(frame):
            raise RuntimeError("Omega5 received empty feature frame")
        if "timestamp" not in frame.columns:
            raise RuntimeError("Omega5 event-risk governor missing required feature: timestamp")
        ts = pd.Timestamp(frame.iloc[-1]["timestamp"])
        if pd.isna(ts):
            raise RuntimeError("Omega5 event-risk governor received NaT timestamp")
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts

    @staticmethod
    def _latest_return(frame: pd.DataFrame, bars: int) -> float:
        if "close" not in frame.columns:
            raise RuntimeError("Omega5 event-risk governor missing required feature: close")
        if len(frame) <= int(bars):
            raise RuntimeError(f"Omega5 event-risk governor needs at least {int(bars) + 1} close rows")
        latest = float(frame.iloc[-1]["close"])
        previous = float(frame.iloc[-1 - int(bars)]["close"])
        if not np.isfinite(latest) or not np.isfinite(previous) or previous <= 0.0:
            raise RuntimeError("Omega5 event-risk governor received invalid close history")
        return float(latest / previous - 1.0)

    @staticmethod
    def _weekday_on_or_after(year: int, month: int, day: int) -> pd.Timestamp:
        out = pd.Timestamp(year=year, month=month, day=day)
        while out.weekday() >= 5:
            out += pd.Timedelta(days=1)
        return out

    @staticmethod
    def _nth_weekday(year: int, month: int, n: int) -> pd.Timestamp:
        out = pd.Timestamp(year=year, month=month, day=1)
        count = 0
        while True:
            if out.weekday() < 5:
                count += 1
                if count == int(n):
                    return out
            out += pd.Timedelta(days=1)

    @staticmethod
    def _first_friday(year: int, month: int) -> pd.Timestamp:
        out = pd.Timestamp(year=year, month=month, day=1)
        while out.weekday() != 4:
            out += pd.Timedelta(days=1)
        return out

    @staticmethod
    def _et_to_utc_naive(day: pd.Timestamp, hour: int, minute: int) -> pd.Timestamp:
        ny = ZoneInfo("America/New_York")
        dt = datetime(int(day.year), int(day.month), int(day.day), int(hour), int(minute), tzinfo=ny)
        return pd.Timestamp(dt.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))

    @classmethod
    def _macro_events_for_year(cls, year: int) -> list[tuple[str, pd.Timestamp]]:
        events: list[tuple[str, pd.Timestamp]] = []
        for month in range(1, 13):
            nfp = cls._first_friday(year, month)
            events.append(("NFP_8h30_ET_rule_based", cls._et_to_utc_naive(nfp, 8, 30)))
            manufacturing = cls._nth_weekday(year, month, 1)
            events.append(("SPGLOBAL_FINAL_MANUF_9h45_ET_rule_based", cls._et_to_utc_naive(manufacturing, 9, 45)))
            events.append(("ISM_MANUFACTURING_10h_ET_rule_based", cls._et_to_utc_naive(manufacturing, 10, 0)))
            services = cls._nth_weekday(year, month, 3)
            events.append(("SPGLOBAL_FINAL_SERVICES_9h45_ET_rule_based", cls._et_to_utc_naive(services, 9, 45)))
            events.append(("ISM_SERVICES_10h_ET_rule_based", cls._et_to_utc_naive(services, 10, 0)))
            flash = cls._weekday_on_or_after(year, month, 23)
            events.append(("SPGLOBAL_FLASH_PMI_9h45_ET_rule_based", cls._et_to_utc_naive(flash, 9, 45)))
        for raw in OMEGA5_EVENT_RISK_FOMC_DECISION_DATES.get(int(year), ()):
            day = pd.Timestamp(raw)
            events.append(("FOMC_14h_ET_static_calendar", cls._et_to_utc_naive(day, 14, 0)))
        return events

    @classmethod
    def _event_risk_latest(cls, frame: pd.DataFrame) -> dict[str, Any]:
        ts = cls._latest_timestamp(frame)
        event_hits: list[str] = []
        for year in (ts.year - 1, ts.year, ts.year + 1):
            for name, event_ts in cls._macro_events_for_year(int(year)):
                start = event_ts - pd.Timedelta(minutes=OMEGA5_EVENT_RISK_MACRO_PRE_MINUTES)
                end = event_ts + pd.Timedelta(minutes=OMEGA5_EVENT_RISK_MACRO_POST_MINUTES)
                if start <= ts <= end:
                    event_hits.append(name)

        jump_flag = cls._latest_optional_float(frame, "jump_flag")
        evt_tail_flag = cls._latest_optional_float(frame, "evt_tail_flag")
        jump_z = cls._latest_optional_float(frame, "jump_z")
        ret_1h = cls._latest_return(frame, 12)
        ret_4h = cls._latest_return(frame, 48)
        shock_hit = (
            jump_flag > 0.0
            or evt_tail_flag > 0.0
            or abs(jump_z) >= OMEGA5_EVENT_RISK_SHOCK_JUMP_Z_THRESHOLD
            or abs(ret_1h) >= OMEGA5_EVENT_RISK_SHOCK_RET_1H_THRESHOLD
            or abs(ret_4h) >= OMEGA5_EVENT_RISK_SHOCK_RET_4H_THRESHOLD
        )
        return {
            "policy_id": OMEGA5_EVENT_RISK_POLICY_ID,
            "timestamp": str(ts),
            "macro_pre_minutes": int(OMEGA5_EVENT_RISK_MACRO_PRE_MINUTES),
            "macro_post_minutes": int(OMEGA5_EVENT_RISK_MACRO_POST_MINUTES),
            "macro_entry_veto": bool(event_hits),
            "macro_event_names": event_hits,
            "shock_haircut": bool(shock_hit),
            "shock_notional_scale": float(OMEGA5_EVENT_RISK_SHOCK_NOTIONAL_SCALE if shock_hit else 1.0),
            "shock_jump_z_threshold": float(OMEGA5_EVENT_RISK_SHOCK_JUMP_Z_THRESHOLD),
            "shock_ret_1h_threshold": float(OMEGA5_EVENT_RISK_SHOCK_RET_1H_THRESHOLD),
            "shock_ret_4h_threshold": float(OMEGA5_EVENT_RISK_SHOCK_RET_4H_THRESHOLD),
            "jump_flag": float(jump_flag),
            "evt_tail_flag": float(evt_tail_flag),
            "jump_z": float(jump_z),
            "ret_1h_past": float(ret_1h),
            "ret_4h_past": float(ret_4h),
        }

    @staticmethod
    def _latest_optional_float(frame: pd.DataFrame, feature: str) -> float:
        if feature not in frame.columns:
            return 0.0
        value = float(frame.iloc[-1][feature])
        return value if np.isfinite(value) else 0.0

    @staticmethod
    def _cash(trace: dict[str, Any], reason: str) -> Omega5Decision:
        out = dict(trace)
        out["omega5_reason"] = reason
        return Omega5Decision(
            action=0,
            side=0,
            notional_exposure=0.0,
            leverage=1.0,
            position_fraction=0.0,
            take_profit=0.0,
            stop_loss=0.0,
            max_hold_bars=0,
            cooldown_bars=0,
            quality_score=float(out.get("parent_quality_score", 0.0) or 0.0),
            confidence=float(out.get("parent_confidence", 0.0) or 0.0),
            trace=out,
        )

    def decide_latest(self, frame: pd.DataFrame, parent_decision: Any) -> Omega5Decision:
        parent_side = int(getattr(parent_decision, "side"))
        parent_action = int(getattr(parent_decision, "action"))
        parent_notional = float(getattr(parent_decision, "notional_exposure"))
        parent_trace = dict(getattr(parent_decision, "trace", {}) or {})
        parent_trace_model_id = str(parent_trace.get("model_id", "") or "")
        if parent_trace_model_id != OMEGA5_PARENT_MODEL_ID:
            raise RuntimeError(
                f"Omega5 parent model mismatch: {parent_trace_model_id} != {OMEGA5_PARENT_MODEL_ID}"
            )
        if bool(parent_trace.get("ledger_replay_used", True)):
            raise RuntimeError("Omega5 parent decision must be live-native; ledger replay parent decisions are forbidden")
        trace: dict[str, Any] = {
            "model_id": OMEGA5_MODEL_ID,
            "model_version": OMEGA5_MODEL_VERSION,
            "source_model_id": OMEGA5_SOURCE_MODEL_ID,
            "parent_model_id": OMEGA5_PARENT_MODEL_ID,
            "reference_model_id": OMEGA5_REFERENCE_MODEL_ID,
            "parent_action": parent_action,
            "parent_side": parent_side,
            "parent_notional_exposure": parent_notional,
            "parent_quality_score": float(getattr(parent_decision, "quality_score", 0.0) or 0.0),
            "parent_confidence": float(getattr(parent_decision, "confidence", 0.0) or 0.0),
            "parent_router_expert": str(getattr(parent_decision, "router_expert", "") or ""),
            "parent_trace": parent_trace,
            "source_roundtrip_cost": float(parent_trace.get("reference_policy_roundtrip_cost", 0.0) or 0.0),
            "source_raw_exit_price_move": float(parent_trace.get("reference_policy_raw_exit_price_move", 0.0) or 0.0),
            "source_net_per_notional": float(parent_trace.get("reference_policy_net_per_notional", 0.0) or 0.0),
            "exposure_long_factor": float(self.long_factor),
            "exposure_short_factor": float(self.short_factor),
            "exposure_cap_notional": float(self.cap_notional),
            "reference_exposure_long_factor": float(self.reference_long_factor),
            "reference_exposure_short_factor": float(self.reference_short_factor),
            "reference_exposure_cap_notional": float(self.reference_cap_notional),
            "leverage_cap": float(self.leverage_cap),
            "max_margin_fraction": float(self.max_margin_fraction),
            "max_hold_bars": int(OMEGA5_MAX_HOLD_BARS),
            "first_veto": {
                "feature": OMEGA5_FIRST_VETO_FEATURE,
                "op": OMEGA5_FIRST_VETO_OP,
                "threshold": float(OMEGA5_FIRST_VETO_THRESHOLD),
            },
            "second_veto": {
                "feature": OMEGA5_SECOND_VETO_FEATURE,
                "op": OMEGA5_SECOND_VETO_OP,
                "threshold": float(OMEGA5_SECOND_VETO_THRESHOLD),
            },
        }
        if parent_action == 0 or parent_side == 0:
            return self._cash(trace, "parent_cash")
        if parent_side not in {-1, 1}:
            raise RuntimeError(f"Omega5 invalid parent side: {parent_side}")
        if parent_notional <= 0.0 or not np.isfinite(parent_notional):
            raise RuntimeError(f"Omega5 invalid parent notional for active decision: {parent_notional}")

        event_risk = self._event_risk_latest(frame)
        trace["event_risk"] = event_risk
        if bool(event_risk["macro_entry_veto"]):
            return self._cash(trace, "macro_event_entry_veto")

        if parent_side < 0:
            first_value = self._latest_float(frame, OMEGA5_FIRST_VETO_FEATURE)
            second_value = self._latest_float(frame, OMEGA5_SECOND_VETO_FEATURE)
            trace["first_veto_value"] = float(first_value)
            trace["second_veto_value"] = float(second_value)
            if first_value <= OMEGA5_FIRST_VETO_THRESHOLD:
                return self._cash(trace, "short_first_stage_veto")
            if second_value >= OMEGA5_SECOND_VETO_THRESHOLD:
                return self._cash(trace, "short_second_stage_veto")

        reference_factor = self.reference_long_factor if parent_side > 0 else self.reference_short_factor
        reference_notional = min(
            parent_notional * reference_factor,
            self.reference_cap_notional,
            self.leverage_cap * self.max_margin_fraction,
        )
        factor = self.long_factor if parent_side > 0 else self.short_factor
        notional = min(reference_notional * factor, self.cap_notional, self.leverage_cap * self.max_margin_fraction)
        pre_event_risk_notional = float(notional)
        notional *= float(event_risk["shock_notional_scale"])
        leverage = self.leverage_cap
        margin_fraction = notional / max(leverage, 1e-12)
        if margin_fraction > self.max_margin_fraction + 1e-12:
            raise RuntimeError(
                f"Omega5 margin fraction exceeds contract: margin={margin_fraction} cap={self.max_margin_fraction}"
            )
        tp_move = OMEGA5_LONG_TP_PRICE_MOVE if parent_side > 0 else OMEGA5_SHORT_TP_PRICE_MOVE
        sl_move = OMEGA5_LONG_SL_PRICE_MOVE if parent_side > 0 else OMEGA5_SHORT_SL_PRICE_MOVE
        take_profit = tp_move * notional
        stop_loss = sl_move * notional
        trace.update(
            {
                "omega5_reason": "entry_shock_haircut" if bool(event_risk["shock_haircut"]) else "entry",
                "side": int(parent_side),
                "reference_exposure_factor": float(reference_factor),
                "reference_notional_exposure": float(reference_notional),
                "exposure_factor": float(factor),
                "pre_event_risk_notional_exposure": float(pre_event_risk_notional),
                "event_risk_notional_scale": float(event_risk["shock_notional_scale"]),
                "notional_exposure": float(notional),
                "leverage": float(leverage),
                "position_fraction": float(margin_fraction),
                "tp_price_move": float(tp_move),
                "sl_price_move": float(sl_move),
                "take_profit": float(take_profit),
                "stop_loss": float(stop_loss),
            }
        )
        return Omega5Decision(
            action=1 if parent_side > 0 else 2,
            side=int(parent_side),
            notional_exposure=float(notional),
            leverage=float(leverage),
            position_fraction=float(margin_fraction),
            take_profit=float(take_profit),
            stop_loss=float(stop_loss),
            max_hold_bars=int(OMEGA5_MAX_HOLD_BARS),
            cooldown_bars=0,
            quality_score=float(getattr(parent_decision, "quality_score", 0.0) or 0.0),
            confidence=float(getattr(parent_decision, "confidence", 0.0) or 0.0),
            trace=trace,
        )
