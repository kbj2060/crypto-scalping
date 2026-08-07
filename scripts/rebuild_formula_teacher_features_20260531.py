#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.teacher_meta_side_features import (
    REQUIRED_TEACHER_INPUTS,
    TEACHER_FEATURE_COLS,
    append_side_teacher_features,
)


DEFAULT_IN_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01965_formula_teacher_v1_candidates_20260531"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} missing timestamp")
    return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _audit(frame: pd.DataFrame) -> dict[str, Any]:
    missing = [col for col in REQUIRED_TEACHER_INPUTS if col not in frame.columns]
    nonfinite = {
        col: int(pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).isna().sum())
        for col in REQUIRED_TEACHER_INPUTS
        if col in frame.columns
    }
    teacher_stats = {}
    for col in TEACHER_FEATURE_COLS:
        s = pd.to_numeric(frame[col], errors="coerce")
        teacher_stats[col] = {
            "min": float(s.min()),
            "p05": float(s.quantile(0.05)),
            "p50": float(s.quantile(0.50)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "na": int(s.isna().sum()),
        }
    return {
        "rows": int(len(frame)),
        "missing_required_inputs": missing,
        "required_input_na_counts": nonfinite,
        "teacher_stats": teacher_stats,
    }


def _process(src: Path, dst: Path) -> dict[str, Any]:
    frame = _read(src)
    out = append_side_teacher_features(frame)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return {"source": str(src), "output": str(dst), **_audit(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild deterministic Formula Teacher v1 candidate CSVs.")
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    jobs = [
        (
            args.in_dir / "trade_candidates_2025_alpha6_current_tail111_exact.csv",
            args.out_dir / "trade_candidates_2025_alpha6_current_tail111_exact.csv",
        ),
        (
            args.in_dir / "trade_candidates_2026_alpha6_current_tail111_exact.csv",
            args.out_dir / "trade_candidates_2026_alpha6_current_tail111_exact.csv",
        ),
    ]
    summary = {
        "model_id": "formula_teacher_v1_20260531",
        "contract": {
            "required_inputs": REQUIRED_TEACHER_INPUTS,
            "outputs": TEACHER_FEATURE_COLS,
            "forbidden_inputs": ["label_*", "target_*", "tp_sl_action_score", "future_*", "pnl_*"],
            "transform": "deterministic_no_fit_current_row_oos_model_outputs_only",
        },
        "datasets": [_process(src, dst) for src, dst in jobs],
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "formula_teacher_v1_audit.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
