#!/usr/bin/env python3
"""Current-live Omega5 source-parent contract proof.

This is intentionally a blocker proof, not a trading backtest.  Omega5 can
reproduce its selected validation/OOS ledgers through the interval adapter, but
the active live timestamp must not be treated as covered unless a predictive
source-parent artifact exists.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_bot_modules.omega4_6_2_source_parent_live import Omega462SourceParentLiveAdapter


SNAPSHOT_PATH = ROOT / "data/live/decision_feature_frame_snapshot.pkl.gz"
CAP220_RUNTIME_CONTRACT = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_cap220_short_boost125_time_stop120h_20260630"
    / "runtime_contract.json"
)
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega5_live_current_snapshot_contract_20260702"
REPORT_JSON = OUT_DIR / "report.json"
REPORT_MD = ROOT / "docs/audits/omega5_live_current_snapshot_contract_20260702.md"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"required JSON artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_snapshot_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"current live feature snapshot is missing: {path}")
    payload = pd.read_pickle(path)
    frame = payload.get("frame") if isinstance(payload, dict) else payload
    if not isinstance(frame, pd.DataFrame) or not len(frame):
        raise RuntimeError(f"current live feature snapshot did not contain a non-empty DataFrame: {path}")
    return frame.copy().reset_index(drop=True)


def latest_timestamp(frame: pd.DataFrame) -> str:
    if "timestamp" in frame.columns:
        return str(pd.Timestamp(frame.iloc[-1]["timestamp"]))
    if isinstance(frame.index, pd.DatetimeIndex):
        return str(pd.Timestamp(frame.index[-1]))
    return ""


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Omega5 Current Live Snapshot Contract - 2026-07-02",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Contract proof pass: `{payload['contract_proof_pass']}`",
        f"- Current snapshot timestamp: `{payload['current_snapshot']['timestamp']}`",
        f"- Current snapshot rows: `{payload['current_snapshot']['rows']}`",
        "",
        "## Checks",
        "",
    ]
    for item in payload["checks"]:
        lines.append(f"- `{item['name']}`: `{item['pass']}` ({item['severity']}) {item['details']}")
    lines.extend(
        [
            "",
            "## Meaning",
            "",
            "- PASS here means the current live timestamp is blocked as expected because the promoted source-parent artifacts only cover validation/OOS windows.",
            "- This prevents Omega5 from silently using historical event-window replay as a future live decision provider.",
            "- A future live promotion must provide a predictive source-parent artifact and update this proof accordingly.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{REPORT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def check(name: str, passed: bool, severity: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "severity": severity, "details": details}


def main() -> int:
    frame = load_snapshot_frame(SNAPSHOT_PATH)
    ts = latest_timestamp(frame)
    contract = read_json(CAP220_RUNTIME_CONTRACT)
    known_live_blockers = dict(contract.get("known_live_blockers", {}) or {})
    successor_requirements = dict(contract.get("promotion_requirements_for_successors", {}) or {})

    adapter_error = ""
    adapter_returned = False
    try:
        Omega462SourceParentLiveAdapter().decide_latest(frame)
        adapter_returned = True
    except RuntimeError as exc:
        adapter_error = str(exc)

    expected_failfast = (not adapter_returned) and "no promoted artifact coverage" in adapter_error
    predictive_artifact_missing = (
        known_live_blockers.get("runtime_native_replay_available") is False
        and successor_requirements.get("historical_trade_ledger_fallback_allowed") is False
        and successor_requirements.get("exact_threshold_parent_predictions_required") is True
    )
    checks = [
        check(
            "current_live_snapshot_source_parent_failfast",
            expected_failfast,
            "must",
            {
                "snapshot": str(SNAPSHOT_PATH),
                "timestamp": ts,
                "adapter_returned_decision": adapter_returned,
                "error": adapter_error,
            },
        ),
        check(
            "predictive_source_parent_artifact_not_declared",
            predictive_artifact_missing,
            "must",
            {
                "runtime_contract": str(CAP220_RUNTIME_CONTRACT),
                "runtime_native_replay_available": known_live_blockers.get("runtime_native_replay_available"),
                "historical_trade_ledger_fallback_allowed": successor_requirements.get(
                    "historical_trade_ledger_fallback_allowed"
                ),
                "exact_threshold_parent_predictions_required": successor_requirements.get(
                    "exact_threshold_parent_predictions_required"
                ),
            },
        ),
    ]
    failed = [item["name"] for item in checks if item["severity"] == "must" and not item["pass"]]
    payload = {
        "audit_id": "omega5_live_current_snapshot_contract_20260702",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "OMEGA5_LIVE_CURRENT_SNAPSHOT_BLOCKED_AS_EXPECTED" if not failed else "OMEGA5_LIVE_CURRENT_SNAPSHOT_CONTRACT_FAIL",
        "contract_proof_pass": not failed,
        "failed_checks": failed,
        "current_snapshot": {"path": str(SNAPSHOT_PATH), "timestamp": ts, "rows": int(len(frame))},
        "checks": checks,
        "artifacts": {"json": str(REPORT_JSON), "markdown": str(REPORT_MD)},
    }
    write_json(REPORT_JSON, payload)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "json": str(REPORT_JSON), "markdown": str(REPORT_MD)}, ensure_ascii=False))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
