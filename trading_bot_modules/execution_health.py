"""Shared execution-alert contract for dashboard and Telegram surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json


_ERROR_TOKENS = ("error", "failed", "mismatch", "bad_")
_BLOCK_TOKENS = ("blocked", "unavailable", "not_ready", "pending_reconcile")


def build_execution_alert(
    execution: dict[str, object] | None,
    *,
    decision_reason: str = "",
    observed_at: str = "",
) -> dict[str, object]:
    state = dict(execution or {})
    enabled = bool(state.get("enabled", False))
    requested_enabled = bool(state.get("requested_enabled", enabled))
    blocking = bool(state.get("blocking", False))
    status = str(state.get("status", "") or "")
    disabled_reason = str(state.get("disabled_reason", "") or "")
    error = str(state.get("error") or state.get("last_error") or "")
    decision_text = str(decision_reason or "")
    decision_basis = decision_text.lower()
    decision_is_issue = any(
        token in decision_basis for token in (*_ERROR_TOKENS, *_BLOCK_TOKENS)
    )
    reason = error or (decision_text if decision_is_issue else "") or disabled_reason or status
    basis = " ".join((status, disabled_reason, error, decision_text)).lower()

    if blocking or error or any(token in basis for token in _ERROR_TOKENS):
        severity = "error"
        title = "트레이딩봇 실행 오류"
    elif requested_enabled and not enabled:
        severity = "blocked"
        title = "실제 주문 실행 차단"
    elif not enabled:
        severity = "disabled"
        title = "실제 주문 실행 비활성"
    elif any(token in basis for token in _BLOCK_TOKENS):
        severity = "blocked"
        title = "트레이딩봇 진입 차단"
    else:
        return {
            "active": False,
            "severity": "ok",
            "title": "실제 주문 실행 정상",
            "reason": "",
            "occurred_at": "",
            "status": status or "ready",
        }

    occurred_at = str(
        state.get("last_error_at")
        or state.get("disabled_at")
        or state.get("occurred_at")
        or observed_at
        or ""
    )
    return {
        "active": True,
        "severity": severity,
        "title": title,
        "reason": reason or "unknown_execution_state",
        "occurred_at": occurred_at,
        "status": status or ("disabled" if not enabled else "blocked"),
    }


def execution_alert_fingerprint(alert: dict[str, object]) -> str:
    return json.dumps(
        {
            "severity": alert.get("severity"),
            "status": alert.get("status"),
            "reason": alert.get("reason"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


@dataclass
class ExecutionAlertDeduper:
    active_fingerprint: str = ""
    active_occurred_at: str = ""
    notified_fingerprint: str = ""

    def should_notify(self, alert: dict[str, object]) -> bool:
        if not bool(alert.get("active", False)):
            self.active_fingerprint = ""
            self.active_occurred_at = ""
            self.notified_fingerprint = ""
            return False
        fingerprint = execution_alert_fingerprint(alert)
        if fingerprint != self.active_fingerprint:
            self.active_fingerprint = fingerprint
            self.active_occurred_at = str(alert.get("occurred_at", "") or "")
            self.notified_fingerprint = ""
        elif self.active_occurred_at:
            alert["occurred_at"] = self.active_occurred_at
        if fingerprint == self.notified_fingerprint:
            return False
        self.notified_fingerprint = fingerprint
        return True
