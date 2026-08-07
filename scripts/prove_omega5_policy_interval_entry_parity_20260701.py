#!/usr/bin/env python3
"""Entry-policy parity proof for Omega5 source-parent interval adapter."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prove_omega5_runtime_native_walkforward_20260701 import (
    EXPECTED_LEDGERS,
    load_enriched_frame,
    prepare_replay_frame,
)
from trading_bot_modules.omega4_6_2_source_parent_live import Omega462SourceParentLiveAdapter
from trading_bot_modules.omega5_live import Omega5LiveAdapter


OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega5_policy_interval_entry_parity_20260701"
REPORT_PATH = OUT_DIR / "report.json"
REPORT_MD = ROOT / "docs/audits/omega5_policy_interval_entry_parity_20260701.md"
EPS = 1.0e-12


def _path(raw: str) -> Path:
    return ROOT / raw


def _omega5_adapter() -> Omega5LiveAdapter:
    return Omega5LiveAdapter(
        report_path=_path("tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/report.json"),
        feature_veto_report_path=_path("tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/report.json"),
        two_stage_veto_report_path=_path("tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/report.json"),
        pnl_tilt_report_path=_path("tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/report.json"),
        redteam_path=_path("tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/redteam_audit_20260701.json"),
        frontier_audit_path=_path("tmp/causal_regen_20260516/omega4_6_2_frontier_leakage_redteam_20260701/frontier_leakage_redteam_20260701.json"),
        cvp_audit_path=_path("tmp/causal_regen_20260516/cvp_feature_causality_20260701/cvp_feature_causality_20260701.json"),
        artifact_integrity_path=_path("tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/omega_artifact_integrity_audit_20260630.json"),
    )


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def _active_expected(row: pd.Series) -> bool:
    return float(row["notional"]) > EPS


def _entry_signal_timestamp(row: pd.Series) -> pd.Timestamp:
    return pd.Timestamp(row["entry_timestamp"])


def _expected_take_profit(row: pd.Series) -> float:
    return float(row["roll8_tp_move"]) * float(row["notional"]) if "roll8_tp_move" in row else 0.0


def _expected_stop_loss(row: pd.Series) -> float:
    return float(row["roll8_sl_move"]) * float(row["notional"]) if "roll8_sl_move" in row else 0.0


def run_split(split: str, *, window_bars: int) -> dict[str, Any]:
    frame = prepare_replay_frame(load_enriched_frame(split))
    ts_to_idx = {pd.Timestamp(ts): int(i) for i, ts in enumerate(pd.to_datetime(frame["timestamp"], errors="raise"))}
    expected = pd.read_csv(EXPECTED_LEDGERS[split])
    source_parent = Omega462SourceParentLiveAdapter()
    omega5 = _omega5_adapter()

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for n, row in expected.reset_index(drop=True).iterrows():
        signal_ts = _entry_signal_timestamp(row)
        if signal_ts not in ts_to_idx:
            failed.append({"row": int(n), "check": "signal_timestamp_exists", "signal_timestamp": str(signal_ts)})
            continue
        idx = ts_to_idx[signal_ts]
        start = max(0, idx + 1 - int(window_bars))
        view = frame.iloc[start : idx + 1].copy().reset_index(drop=True)
        parent_dec = source_parent.decide_latest(view)
        dec = omega5.decide_latest(view, parent_dec)
        expected_active = _active_expected(row)
        observed_active = int(dec.action) != 0 and int(dec.side) != 0 and float(dec.notional_exposure) > EPS
        item = {
            "row": int(n),
            "entry_timestamp": str(row["entry_timestamp"]),
            "signal_timestamp": str(signal_ts),
            "expected_active": bool(expected_active),
            "observed_active": bool(observed_active),
            "expected_side": int(row["side"]),
            "observed_side": int(dec.side),
            "expected_notional": float(row["notional"]),
            "observed_notional": float(dec.notional_exposure),
            "expected_leverage": float(row["leverage"]),
            "observed_leverage": float(dec.leverage),
            "expected_margin_fraction": float(row["margin_fraction"]),
            "observed_margin_fraction": float(dec.position_fraction),
            "expected_take_profit": _expected_take_profit(row),
            "observed_take_profit": float(dec.take_profit),
            "expected_stop_loss": _expected_stop_loss(row),
            "observed_stop_loss": float(dec.stop_loss),
            "omega5_reason": str(dec.trace.get("omega5_reason", "")),
            "parent_reason": str((dec.trace.get("parent_trace") or {}).get("omega462_reason", "")),
        }
        rows.append(item)
        row_failed: list[str] = []
        if expected_active != observed_active:
            row_failed.append("active_flag")
        if expected_active and observed_active:
            if int(row["side"]) != int(dec.side):
                row_failed.append("side")
            for key in ("notional", "leverage", "margin_fraction", "take_profit", "stop_loss"):
                diff = abs(float(item[f"expected_{key}"]) - float(item[f"observed_{key}"]))
                if diff > 1.0e-8:
                    row_failed.append(key)
        if row_failed:
            bad = dict(item)
            bad["failed_checks"] = row_failed
            failed.append(bad)

    detail_path = OUT_DIR / f"{split}_entry_policy_rows.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(detail_path, index=False)
    return {
        "split": split,
        "rows": int(len(rows)),
        "failed": int(len(failed)),
        "pass": len(failed) == 0,
        "failed_examples": failed[:20],
        "detail_csv": str(detail_path),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Omega5 Policy Interval Entry Parity - 2026-07-01",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Entry parity pass: `{payload['entry_policy_parity_pass']}`",
        "",
        "## Splits",
        "",
        "| Split | Pass | Rows | Failed |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in payload["splits"]:
        lines.append(f"| `{row['split']}` | `{row['pass']}` | `{row['rows']}` | `{row['failed']}` |")
    lines.extend(["", "## Failed Examples", ""])
    examples: list[str] = []
    for row in payload["splits"]:
        for failed in row.get("failed_examples", []):
            examples.append(f"- `{row['split']}` row `{failed.get('row')}`: `{failed}`")
    lines.extend(examples or ["- None."])
    lines.extend(["", "## Artifacts", "", f"- JSON: `{REPORT_PATH}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    payload: dict[str, Any] = {
        "audit_id": "omega5_policy_interval_entry_parity_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_entrypoint": "Omega462SourceParentLiveAdapter.decide_latest + Omega5LiveAdapter.decide_latest",
        "splits": [run_split("validation", window_bars=7000), run_split("oos", window_bars=7000)],
    }
    passed = all(row["pass"] for row in payload["splits"])
    payload["entry_policy_parity_pass"] = bool(passed)
    payload["verdict"] = "OMEGA5_POLICY_INTERVAL_ENTRY_PARITY_PASS" if passed else "OMEGA5_POLICY_INTERVAL_ENTRY_PARITY_FAIL"
    write_json(REPORT_PATH, payload)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "json": str(REPORT_PATH), "markdown": str(REPORT_MD)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
