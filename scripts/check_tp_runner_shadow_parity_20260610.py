#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "data/live/tp_runner_shadow_parity.jsonl"
DEFAULT_BUNDLE = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"invalid JSONL at {path}:{line_no}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _predict(bundle: dict[str, Any], features: dict[str, Any]) -> tuple[float, str]:
    feature_cols = list(bundle["feature_cols"])
    missing = [c for c in feature_cols if c not in features]
    if missing:
        raise RuntimeError(f"missing logged feature columns: {missing}")
    x = np.asarray([[float(features[c]) for c in feature_cols]], dtype=np.float64)
    model = bundle["model"]
    prob = float(model.predict_proba(x)[0, 1]) if hasattr(model, "predict_proba") else float(model.predict(x)[0])
    template = dict(bundle.get("template") or {})
    quality_min = float(template.get("quality_min", 1.0) or 1.0)
    momentum_min = float(template.get("momentum_min", 0.0) or 0.0)
    proba_min = float(bundle.get("proba_min", 1.0) or 1.0)
    decision = (
        "extend"
        if float(features["quality"]) >= quality_min and float(features["ret3_side"]) > momentum_min and prob >= proba_min
        else "take_profit_now"
    )
    return prob, decision


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = ap.parse_args()

    if not args.bundle.exists():
        raise RuntimeError(f"missing selector bundle: {args.bundle}")
    bundle = joblib.load(args.bundle)
    rows = _read_jsonl(args.log)
    checked = 0
    mismatches: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if row.get("event") != "omega1_2_1_tp_runner_shadow":
            continue
        if row.get("feature_status") != "ok" or row.get("selector_status") != "loaded":
            continue
        prob, decision = _predict(bundle, dict(row.get("features") or {}))
        checked += 1
        logged_prob = float(row.get("selector_proba", np.nan))
        logged_decision = str(row.get("selector_decision", ""))
        if not np.isfinite(logged_prob) or abs(prob - logged_prob) > 1e-12 or decision != logged_decision:
            mismatches.append(
                {
                    "row_index": idx,
                    "logged_prob": logged_prob,
                    "recomputed_prob": prob,
                    "logged_decision": logged_decision,
                    "recomputed_decision": decision,
                }
            )
    result = {
        "log": str(args.log),
        "bundle": str(args.bundle),
        "rows": len(rows),
        "checked": checked,
        "mismatches": mismatches[:20],
        "status": "pass" if checked > 0 and not mismatches else ("no_rows" if checked == 0 else "fail"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
