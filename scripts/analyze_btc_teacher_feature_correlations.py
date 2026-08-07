#!/usr/bin/env python3
"""Compute feature-to-soft-teacher correlations on a pinned BTC feature frame."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.btc_trajectory_teacher import TeacherConfig, build_teacher_path  # noqa: E402

LABEL_COLS = [
    "teacher_short_probability", "teacher_flat_probability", "teacher_long_probability",
    "teacher_signed_margin_fraction", "teacher_margin_fraction",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, nargs="+")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-bars", type=int, default=0, help="Diagnostic subset before full-history label generation.")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    frame = pd.concat([pd.read_csv(path, low_memory=False) for path in args.data], ignore_index=True)
    if args.tail_bars:
        frame = frame.tail(args.tail_bars + 49).reset_index(drop=True)
    # Year files are independent label segments; concatenate their causal teacher outputs.
    labels = pd.concat([build_teacher_path(part[["timestamp", "open", "close"]], TeacherConfig()) for _, part in frame.groupby(pd.to_datetime(frame["timestamp"]).dt.year)], ignore_index=True)
    labels["decision_timestamp"] = pd.to_datetime(labels["decision_timestamp"], utc=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    merged = frame.merge(labels[["decision_timestamp"] + LABEL_COLS], left_on="timestamp", right_on="decision_timestamp", how="inner")
    drop = set(selection["drop_columns"])
    feature_cols = [c for c in frame.columns if c not in {"timestamp", *drop}]
    numeric = merged[feature_cols].select_dtypes(include="number")
    result = pd.concat(
        [numeric.corrwith(merged[label], method="spearman").rename(label) for label in LABEL_COLS],
        axis=1,
    )
    result.index.name = "feature"
    result = result.reset_index().melt(id_vars="feature", var_name="teacher_label", value_name="spearman_correlation")
    result["abs_spearman_correlation"] = result["spearman_correlation"].abs()
    result = result.sort_values(["teacher_label", "abs_spearman_correlation"], ascending=[True, False])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "btc_shared_policy_teacher_feature_spearman.csv"
    labels_path = args.output_dir / "btc_shared_policy_teacher_labels_2026.csv"
    result.to_csv(csv_path, index=False)
    labels.to_csv(labels_path, index=False)
    print(json.dumps({"rows": int(len(merged)), "features": int(len(feature_cols)), "tail_bars": int(args.tail_bars), "correlations": str(csv_path), "labels": str(labels_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
