#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRAME_SNAPSHOT_PATH = ROOT / "data/live/decision_feature_frame_snapshot.pkl.gz"
REPORT_OUT = ROOT / "data/ensemble/reports/replay_alpha3_live_snapshot_parity_20260515.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


def _restore_router(router: Any, snapshot: dict[str, Any]) -> None:
    pos = snapshot.get("pos")
    router.pos = str(pos).upper() if str(pos or "").upper() in {"LONG", "SHORT"} else None
    router.entry_price = float(snapshot.get("entry_price", 0.0) or 0.0)
    router.hold_count = int(snapshot.get("hold_bars", 0) or 0)
    router.position_fraction = float(snapshot.get("position_fraction", 0.0) or 0.0)
    router.execution_leverage = float(snapshot.get("execution_leverage", 1.0) or 1.0)
    router.current_leverage = float(snapshot.get("notional_exposure", 0.0) or 0.0)
    router.position_realized_pnl_frac = float(snapshot.get("position_realized_pnl_frac", 0.0) or 0.0)
    router.open_trade_id = str(snapshot.get("trade_id", "") or "")
    router.opened_at = str(snapshot.get("opened_at", "") or "")
    router.decision_at = str(snapshot.get("decision_at", "") or "")
    router.entry_price_source = str(snapshot.get("entry_price_source", "") or "")
    router.entry_decision_price = float(snapshot.get("entry_decision_price", 0.0) or 0.0)
    router.exchange_entry_price = float(snapshot.get("exchange_entry_price", 0.0) or 0.0)
    router.entry_execution_liquidity = str(snapshot.get("entry_execution_liquidity", "") or "")
    router.entry_execution_route = str(snapshot.get("entry_execution_route", "") or "")
    router.entry_execution_order_type = str(snapshot.get("entry_execution_order_type", "") or "")


def _decision_summary(decision_tuple: tuple[Any, ...]) -> dict[str, Any]:
    action, exposure, fraction, leverage, info, regime = decision_tuple
    info = dict(info or {})
    sleeve = dict(info.get("sleeve_trace", {}) or {})
    v31 = dict(sleeve.get("v31", {}) or {})
    alpha2 = dict(sleeve.get("alpha2_1", {}) or {})
    return {
        "action": int(action),
        "exposure": float(exposure),
        "fraction": float(fraction),
        "leverage": float(leverage),
        "regime": str(regime),
        "source": str(info.get("source", "")),
        "position_signal": str(info.get("position_signal", "")),
        "position_reason": str(info.get("position_reason", "")),
        "score": float(info.get("score", 0.0) or 0.0),
        "conviction": float(info.get("conviction", 0.0) or 0.0),
        "v31": {
            "q_long": float(v31.get("q_long", 0.0) or 0.0),
            "q_short": float(v31.get("q_short", 0.0) or 0.0),
            "q_long_raw": float(v31.get("q_long_raw", 0.0) or 0.0),
            "q_short_raw": float(v31.get("q_short_raw", 0.0) or 0.0),
            "edge": float(v31.get("edge", 0.0) or 0.0),
            "margin": float(v31.get("margin", 0.0) or 0.0),
            "selected_side": str(v31.get("selected_side", "")),
            "pass_gate": bool(v31.get("pass_gate", False)),
            "guard_reasons": list(v31.get("regime_long_guard_reasons", []) or []),
            "transition_risk": float(v31.get("transition_risk", 0.0) or 0.0),
        },
        "alpha2_1": {
            "parent_action_before": int(alpha2.get("parent_action_before", 0) or 0),
            "teacher_pred_action": int(alpha2.get("teacher_pred_action", 0) or 0),
            "teacher_confidence": float(alpha2.get("teacher_confidence", 0.0) or 0.0),
            "keep_parent": bool(alpha2.get("keep_parent", False)),
            "reason": str(alpha2.get("reason", "")),
        },
    }


def main() -> int:
    if not FRAME_SNAPSHOT_PATH.exists():
        raise SystemExit(f"missing frame snapshot: {FRAME_SNAPSHOT_PATH}")

    payload = pd.read_pickle(FRAME_SNAPSHOT_PATH, compression="gzip")
    if not isinstance(payload, dict) or not isinstance(payload.get("frame"), pd.DataFrame):
        raise SystemExit("invalid frame snapshot payload")

    frame = payload["frame"].copy()
    runtime_state = dict(payload.get("governor_runtime_state") or {})
    router_snapshot = dict(payload.get("router_snapshot") or {})
    active_info = dict(payload.get("active_info") or {})
    expected = dict(payload.get("decision") or {})
    expected_health = dict(payload.get("health_summary") or {})

    with tempfile.TemporaryDirectory(prefix="alpha3_replay_") as td:
        tmp_runtime = Path(td) / "final_governor_runtime_state.json"
        tmp_router = Path(td) / "governor_live_state.json"
        tmp_runtime.write_text(json.dumps(runtime_state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_router.write_text("{}", encoding="utf-8")

        os.environ["FINAL_GOVERNOR_RUNTIME_STATE_PATH"] = str(tmp_runtime)
        os.environ["GOVERNOR_LIVE_STATE_PATH"] = str(tmp_router)
        os.environ.setdefault("CONSOLE_LOG_COMPACT", "1")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        import trading_bot as tb  # noqa: WPS433

        governor = tb.FinalGovernorRuntime()
        router = tb.GovernorPositionRouter()
        _restore_router(router, router_snapshot)

        decision_price = float(active_info.get("decision_price", 0.0) or 0.0)
        if decision_price <= 0.0 and len(frame) and "close" in frame.columns:
            decision_price = float(pd.to_numeric(frame["close"], errors="coerce").iloc[-1])

        direct_v31 = {}
        try:
            pred = governor._v31_predict_latest(frame)
            if pred is not None:
                direct_v31 = {"q_long": float(pred[0]), "q_short": float(pred[1])}
        except Exception as exc:  # pragma: no cover - diagnostic script
            direct_v31 = {"error": str(exc)}

        try:
            replay_decision = governor.decide(
                processed_df=frame,
                meta_router=router,
                current_price=decision_price,
                m7_last=None,
                trend_signal=None,
            )
            replay = _decision_summary(replay_decision)
            replay_error = ""
        except Exception as exc:  # pragma: no cover - diagnostic script
            replay = {}
            replay_error = str(exc)

    expected_action = int(expected.get("action", active_info.get("final_action", 0)) or 0)
    expected_reason = str(expected.get("position_reason", active_info.get("position_reason", "")) or "")
    expected_source = str(expected.get("source", active_info.get("source", "")) or "")
    expected_v31 = dict(expected_health.get("v31", {}) or {})
    action_match = bool(replay and int(replay.get("action", -999)) == expected_action)
    reason_match = bool(replay and str(replay.get("position_reason", "")) == expected_reason)
    source_match = bool(replay and str(replay.get("source", "")) == expected_source)

    report = {
        "model_id": "replay_alpha3_live_snapshot_parity_20260515",
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "snapshot_path": str(FRAME_SNAPSHOT_PATH),
        "snapshot_created_at": payload.get("created_at"),
        "snapshot_timestamp": payload.get("timestamp"),
        "frame_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "router_snapshot": router_snapshot,
        "expected": {
            "action": expected_action,
            "source": expected_source,
            "position_reason": expected_reason,
            "health_v31": expected_v31,
        },
        "replay": replay,
        "replay_error": replay_error,
        "direct_v31_latest_from_frame": direct_v31,
        "parity": {
            "action_match": action_match,
            "source_match": source_match,
            "reason_match": reason_match,
            "full_decision_match": bool(action_match and source_match and reason_match),
            "note": "Direct V31 latest can differ from health_v31 while in-position, because health_v31 records active entry-state trace for lifecycle hold decisions.",
        },
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "parity": report["parity"], "replay_error": replay_error}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
