#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_catboost_current_tail111_tune_20260521"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Alpha6 current_tail111 tuning runs.")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = ap.parse_args()
    rows = []
    for grid_path in sorted(args.run_dir.glob("*/current_tail111_threshold_grid.csv")):
        name = grid_path.parent.name
        df = pd.read_csv(grid_path)
        if df.empty:
            continue
        best = df.sort_values("score", ascending=False).iloc[0]
        band = df[(df["trades_per_day"] >= 5.0) & (df["trades_per_day"] <= 10.0)].sort_values("score", ascending=False)
        band_best = band.iloc[0] if len(band) else None
        row = {
            "run": name,
            "best_score": float(best["score"]),
            "best_pnl": float(best["pnl"]),
            "best_mdd": float(best["mdd"]),
            "best_trades": int(best["trades"]),
            "best_tpd": float(best["trades_per_day"]),
            "best_wr": float(best["wr"]),
            "best_long": int(best["long_entries"]),
            "best_short": int(best["short_entries"]),
        }
        if band_best is not None:
            row.update(
                {
                    "band_score": float(band_best["score"]),
                    "band_pnl": float(band_best["pnl"]),
                    "band_mdd": float(band_best["mdd"]),
                    "band_trades": int(band_best["trades"]),
                    "band_tpd": float(band_best["trades_per_day"]),
                    "band_wr": float(band_best["wr"]),
                    "band_long": int(band_best["long_entries"]),
                    "band_short": int(band_best["short_entries"]),
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["band_pnl", "best_pnl"], ascending=False, na_position="last")
    out_path = args.run_dir / "tune_summary.csv"
    json_path = args.run_dir / "tune_summary.json"
    out.to_csv(out_path, index=False)
    json_path.write_text(json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2))
    print(out.to_string(index=False) if not out.empty else "no completed runs")
    print(f"summary_csv={out_path}")


if __name__ == "__main__":
    main()
