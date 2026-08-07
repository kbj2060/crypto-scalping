#!/usr/bin/env python3
"""Pad SOL's h48_conservative triple-barrier action onto the SOL wave3 zigzag
label's exact per-year timestamp index (unmatched timestamps = CASH), matching
the ETH h48qual quality-label contract
(sltp_h48_conservative_padded_to_zigzag_timestamps).

Output: zigzag_action_labels_{2025,2026}.csv with a `zigzag_action` column
(the shape the parent trainer's --quality-label-dir expects), sourced from
tb_action_h48_conservative.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TB_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega1_2_triple_barrier_labels_20260707"
DEFAULT_ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/sol_zigzag_action_labels_20260707"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/sol_h48_conservative_padded_to_zigzag_timestamps_20260707"
BARRIER_COL = "tb_action_h48_conservative"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb-dir", type=Path, default=DEFAULT_TB_DIR)
    ap.add_argument("--zigzag-dir", type=Path, default=DEFAULT_ZIGZAG_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_tb = pd.read_csv(args.tb_dir / "train_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=["timestamp", BARRIER_COL])
    val_tb = pd.read_csv(args.tb_dir / "validation_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=["timestamp", BARRIER_COL])
    oos_tb = pd.read_csv(args.tb_dir / "oos_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=["timestamp", BARRIER_COL])
    tb_by_year = {
        2025: pd.concat([train_tb, val_tb], ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last"),
        2026: oos_tb.sort_values("timestamp").drop_duplicates("timestamp", keep="last"),
    }

    audit: dict[str, Any] = {"barrier_source_col": BARRIER_COL, "artifacts": {}, "summaries": {}}
    for year, tb in tb_by_year.items():
        zigzag_path = args.zigzag_dir / f"zigzag_action_labels_{year}.csv"
        zigzag = pd.read_csv(zigzag_path, parse_dates=["timestamp"], usecols=["timestamp"])
        merged = zigzag.merge(tb.rename(columns={BARRIER_COL: "zigzag_action"}), on="timestamp", how="left", validate="one_to_one")
        n_missing = int(merged["zigzag_action"].isna().sum())
        merged["zigzag_action"] = merged["zigzag_action"].fillna(0).astype(np.int64)
        out_path = args.out_dir / f"zigzag_action_labels_{year}.csv"
        merged.to_csv(out_path, index=False)
        counts = merged["zigzag_action"].value_counts().sort_index().to_dict()
        audit["artifacts"][str(year)] = str(out_path)
        audit["summaries"][str(year)] = {
            "rows": int(len(merged)),
            "missing_timestamps_filled_as_cash": n_missing,
            "counts": {str(int(k)): int(v) for k, v in counts.items()},
        }

    audit_path = args.out_dir / "sol_h48_padded_label_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
