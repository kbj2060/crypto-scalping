#!/usr/bin/env python3
"""Causality audit for CVP features used by Omega 4.6.2 frontier vetoes."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cvp import CVP_FEATURE_COLS, add_cvp_features


FRONTIER_AUDIT_JSON = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_frontier_leakage_redteam_20260701"
    / "frontier_leakage_redteam_20260701.json"
)
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / "cvp_feature_causality_20260701"
AUDIT_JSON = OUT_DIR / "cvp_feature_causality_20260701.json"
AUDIT_MD = ROOT / "docs/audits/cvp_feature_causality_20260701.md"
CVP_SOURCE = ROOT / "core/cvp.py"
FORBIDDEN_SOURCE_PATTERNS = [
    r"shift\s*\(\s*-",
    r"\.iloc\s*\[[^\]]*\+\s*[1-9]",
    r"\bfuture\b",
    r"\btarget\b",
    r"\blabel\b",
    r"\boracle\b",
    r"\bmfe\b",
    r"\bmae\b",
    r"\bexit_",
]
TOL = 1.0e-12


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def source_check() -> dict[str, Any]:
    text = CVP_SOURCE.read_text(encoding="utf-8")
    matches = {
        pattern: sorted(set(re.findall(pattern, text, flags=re.IGNORECASE)))
        for pattern in FORBIDDEN_SOURCE_PATTERNS
    }
    matches = {pattern: values for pattern, values in matches.items() if values}
    return {
        "pass": not matches,
        "source": str(CVP_SOURCE),
        "forbidden_matches": matches,
        "evidence": "add_cvp_features slices each row as start:i+1, so each output row uses only current and prior bars.",
    }


def prefix_stability_check(csv_path: Path, *, prefix_rows: int = 500, future_rows: int = 200) -> dict[str, Any]:
    frame = pd.read_csv(csv_path)
    required = ["high", "low", "close", "volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        return {"pass": False, "csv": str(csv_path), "missing": missing}
    needed = min(len(frame), prefix_rows + future_rows)
    if needed < 50:
        return {"pass": False, "csv": str(csv_path), "reason": "not enough rows", "rows": int(len(frame))}
    prefix_len = min(prefix_rows, needed // 2)
    extended_len = min(len(frame), prefix_len + future_rows)
    prefix = add_cvp_features(frame.iloc[:prefix_len].copy(), lookback=200, n_clusters=4, output_cols=CVP_FEATURE_COLS)
    extended = add_cvp_features(frame.iloc[:extended_len].copy(), lookback=200, n_clusters=4, output_cols=CVP_FEATURE_COLS)
    diffs = {}
    for col in CVP_FEATURE_COLS:
        left = pd.to_numeric(prefix[col], errors="coerce").reset_index(drop=True)
        right = pd.to_numeric(extended.iloc[:prefix_len][col], errors="coerce").reset_index(drop=True)
        diffs[col] = float((left - right).abs().max())
    return {
        "pass": all(value <= TOL for value in diffs.values()),
        "csv": str(csv_path),
        "prefix_rows": int(prefix_len),
        "extended_rows": int(extended_len),
        "max_abs_diffs": diffs,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# CVP Feature Causality Audit - 2026-07-01",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Source check pass: `{payload['source_check']['pass']}`",
        "",
        "## Prefix Stability",
        "",
        "| CSV | Pass | Prefix Rows | Extended Rows | Max Diff |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in payload["prefix_stability"]:
        max_diff = max(item.get("max_abs_diffs", {"missing": 1.0}).values())
        lines.append(
            f"| `{item['csv']}` | `{item['pass']}` | `{item.get('prefix_rows', 0)}` | "
            f"`{item.get('extended_rows', 0)}` | `{max_diff:.3e}` |"
        )
    lines.extend(["", "## Source Evidence", "", f"- {payload['source_check']['evidence']}"])
    if payload["source_check"]["forbidden_matches"]:
        lines.extend(["", "## Forbidden Matches", ""])
        lines.append(f"- `{payload['source_check']['forbidden_matches']}`")
    lines.extend(["", "## Artifacts", "", f"- JSON: `{AUDIT_JSON}`"])
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    frontier = read_json(FRONTIER_AUDIT_JSON)
    csv_paths = [Path(frontier["train_market_csv"]), Path(frontier["eval_market_csv"])]
    src = source_check()
    stability = [prefix_stability_check(path) for path in csv_paths]
    passed = bool(src["pass"] and all(item["pass"] for item in stability))
    payload = {
        "audit_id": "cvp_feature_causality_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "CVP_FEATURE_CAUSALITY_PASS" if passed else "CVP_FEATURE_CAUSALITY_FAIL",
        "source_check": src,
        "prefix_stability": stability,
        "artifacts": {"audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD)},
    }
    write_json(AUDIT_JSON, payload)
    write_markdown(payload)
    print(json.dumps({"audit_json": str(AUDIT_JSON), "audit_md": str(AUDIT_MD), "verdict": payload["verdict"]}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

