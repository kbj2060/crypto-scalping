#!/usr/bin/env python3
"""Prune unused RL training columns and write a slimmer CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DROP_COLS = [
    "pred_ridge",
    "m7_hdb_label",
    "m7_hdb_prob",
    "m7_vae_threshold",
    "cada",
    "m7_prob_dn",
    "m7_prob_fl",
    "m7_prob_up",
    "m7_direction",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune unused columns from RL training CSV")
    p.add_argument("--input", required=True, help="Source CSV path")
    p.add_argument("--output", required=True, help="Destination CSV path")
    p.add_argument(
        "--drop-cols",
        nargs="*",
        default=DEFAULT_DROP_COLS,
        help="Columns to drop. Defaults to the vetted unused/redundant set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        raise FileNotFoundError(f"input csv not found: {src}")

    df = pd.read_csv(src)
    existing_drop = [c for c in args.drop_cols if c in df.columns]
    missing_drop = [c for c in args.drop_cols if c not in df.columns]
    pruned = df.drop(columns=existing_drop)

    dst.parent.mkdir(parents=True, exist_ok=True)
    pruned.to_csv(dst, index=False)

    print(f"input_rows={len(df)} input_cols={len(df.columns)}")
    print(f"output_rows={len(pruned)} output_cols={len(pruned.columns)}")
    print("dropped_existing=" + ",".join(existing_drop))
    print("dropped_missing=" + ",".join(missing_drop))


if __name__ == "__main__":
    main()
