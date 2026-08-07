#!/usr/bin/env python3
"""Runtime-owned replay audit for the Omega 4.6.2 validation-only adapter."""

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

from trading_bot_modules.omega4_6_2_runtime_adapter import (
    DEFAULT_MODEL_ID,
    DEFAULT_REPORT,
    NUMERIC_DECISION_COLUMNS,
    STRING_DECISION_COLUMNS,
    Omega462LedgerReplayAdapter,
)


OUT_DIR = ROOT / "tmp/causal_regen_20260516" / DEFAULT_MODEL_ID
AUDIT_JSON = OUT_DIR / "runtime_replay_audit_20260701.json"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_validation_only_runtime_replay_20260701.md"
TOL = 1.0e-8


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def timestamp_key(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="raise").dt.strftime("%Y-%m-%d %H:%M:%S")


def compare_split(adapter: Omega462LedgerReplayAdapter, split: str) -> dict[str, Any]:
    expected = adapter.ledgers[split].reset_index(drop=True)
    replay = adapter.replay_split(split).reset_index(drop=True)
    checks: list[dict[str, Any]] = []
    checks.append({"name": "row_count", "pass": len(expected) == len(replay), "details": {"expected": len(expected), "replay": len(replay)}})
    numeric_diffs: dict[str, float] = {}
    for col in NUMERIC_DECISION_COLUMNS:
        if col in expected.columns and col in replay.columns:
            diff = (pd.to_numeric(expected[col], errors="coerce") - pd.to_numeric(replay[col], errors="coerce")).abs()
            numeric_diffs[col] = float(diff.max()) if len(diff) else 0.0
    checks.append({"name": "numeric_decision_parity", "pass": all(value <= TOL for value in numeric_diffs.values()), "details": numeric_diffs})
    string_mismatches: dict[str, int] = {}
    for col in STRING_DECISION_COLUMNS:
        if col in expected.columns and col in replay.columns:
            if "timestamp" in col:
                left = timestamp_key(expected[col])
                right = timestamp_key(replay[col])
            else:
                left = expected[col].astype(str).fillna("")
                right = replay[col].astype(str).fillna("")
            string_mismatches[col] = int((left.reset_index(drop=True) != right.reset_index(drop=True)).sum())
    checks.append({"name": "string_decision_parity", "pass": all(value == 0 for value in string_mismatches.values()), "details": string_mismatches})
    action_expected = np.where(pd.to_numeric(expected["notional"], errors="coerce").fillna(0.0) > 1.0e-12, "ENTER", "SKIP")
    action_mismatch = int((pd.Series(action_expected) != replay["action"].astype(str)).sum())
    checks.append({"name": "action_parity", "pass": action_mismatch == 0, "details": {"mismatch_count": action_mismatch}})
    contract_error = (
        pd.to_numeric(replay["notional"], errors="coerce")
        - pd.to_numeric(replay["margin_fraction"], errors="coerce") * pd.to_numeric(replay["leverage"], errors="coerce")
    ).abs()
    checks.append({"name": "notional_margin_leverage_contract", "pass": float(contract_error.max()) <= TOL, "details": {"max_abs_error": float(contract_error.max())}})
    return {
        "split": split,
        "pass": all(item["pass"] for item in checks),
        "rows": int(len(expected)),
        "active_rows": int((pd.to_numeric(expected["notional"], errors="coerce").fillna(0.0) > 1.0e-12).sum()),
        "checks": checks,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Omega 4.6.2 Validation-Only Runtime Replay Audit - 2026-07-01",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Verdict: `{payload['verdict']}`",
        f"- Runtime replay pass: `{payload['runtime_replay_pass']}`",
        f"- Report: `{payload['report']}`",
        "",
        "## Split Results",
        "",
        "| Split | Pass | Rows | Active Rows |",
        "| --- | --- | ---: | ---: |",
    ]
    for split in payload["splits"]:
        lines.append(f"| `{split['split']}` | `{split['pass']}` | `{split['rows']}` | `{split['active_rows']}` |")
    lines.extend(["", "## Failed Checks", ""])
    failed = [
        (split["split"], check)
        for split in payload["splits"]
        for check in split["checks"]
        if not check["pass"]
    ]
    lines.extend([f"- `{split}` / `{check['name']}`: {check['details']}" for split, check in failed] or ["- None."])
    lines.extend(["", "## Artifacts", "", f"- JSON: `{AUDIT_JSON}`"])
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    adapter = Omega462LedgerReplayAdapter.from_report(DEFAULT_REPORT)
    selected = adapter.selected_variant
    no_oos_selection = selected.get("oos_used_in_selection") is False
    splits = [compare_split(adapter, "validation"), compare_split(adapter, "oos")]
    runtime_replay_pass = bool(no_oos_selection and all(item["pass"] for item in splits))
    payload = {
        "audit_id": "omega4_6_2_validation_only_runtime_replay_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": DEFAULT_MODEL_ID,
        "report": str(DEFAULT_REPORT),
        "adapter_module": "trading_bot_modules.omega4_6_2_runtime_adapter",
        "adapter_class": "Omega462LedgerReplayAdapter",
        "selection_oos_used": selected.get("oos_used_in_selection"),
        "runtime_replay_pass": runtime_replay_pass,
        "verdict": "RUNTIME_REPLAY_PASS" if runtime_replay_pass else "RUNTIME_REPLAY_FAIL",
        "splits": splits,
        "artifacts": {"audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD)},
    }
    write_json(AUDIT_JSON, payload)
    write_markdown(payload)
    print(json.dumps({"audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD), "verdict": payload["verdict"], "runtime_replay_pass": runtime_replay_pass}, ensure_ascii=False))
    return 0 if runtime_replay_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
