#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze specialist predictive power on rl_meta csv")
    ap.add_argument("--csv", default="data/splits/year_oos/rl_meta_2026.csv")
    ap.add_argument("--lookahead", type=int, default=5)
    ap.add_argument("--min-abs-action", type=float, default=0.1)
    ap.add_argument(
        "--short-mode",
        choices=["raw", "negate", "signed_2x_minus_1"],
        default="negate",
        help="meta_short_raw 해석 모드",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    closes = df["close"].values

    print(f"[FILE] {args.csv}")
    print(
        f"[CFG] lookahead={args.lookahead} min_abs_action={args.min_abs_action} short_mode={args.short_mode}"
    )

    for name, col in [("primary", "meta_primary_raw"), ("long", "meta_long_raw"), ("short", "meta_short_raw")]:
        if col not in df.columns:
            print(f"  {col} 없음")
            continue
        actions = df[col].values
        if name == "short":
            if args.short_mode == "negate":
                actions = -actions
            elif args.short_mode == "signed_2x_minus_1":
                actions = (2.0 * actions) - 1.0
        correct = 0
        total = 0
        pnl = 0.0
        for i in range(len(df) - args.lookahead):
            a = actions[i]
            if abs(a) < args.min_abs_action:
                continue
            fut_ret = (closes[i + args.lookahead] - closes[i]) / closes[i]
            hit = (a > 0 and fut_ret > 0) or (a < 0 and fut_ret < 0)
            correct += int(hit)
            total += 1
            pnl += abs(a) * fut_ret * (1 if a > 0 else -1)

        if total > 0:
            print(f"  {name:10s}  accuracy={correct/total:.1%}  signals={total}  pnl={pnl*100:.2f}%")
        else:
            print(f"  {name:10s}  신호 없음")

    print("\n[ACTION DISTRIBUTION]")
    for col in ["meta_primary_raw", "meta_long_raw", "meta_short_raw"]:
        if col in df.columns:
            v = df[col].dropna()
            print(
                f"  {col:25s}  mean={v.mean():.4f}  std={v.std():.4f}  min={v.min():.4f}  max={v.max():.4f}"
            )


if __name__ == "__main__":
    main()
