#!/usr/bin/env python3
"""Create sparse Oracle entry-quality targets using Train-only DP advantage cutoffs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "tmp/causal_regen_20260516/eth_split_oracle_strategy_labels_20260724"
OUT_ROOT = ROOT / "tmp/causal_regen_20260516"
VARIANTS = {"q20": 0.80, "q10": 0.90, "q05": 0.95}


def load_split(split: str) -> pd.DataFrame:
    strategy = pd.read_parquet(SOURCE_DIR / f"{split}_oracle_strategy_labels.parquet")
    trajectory = pd.read_csv(
        SOURCE_DIR / f"{split}_oracle_trajectory_labels.csv",
        parse_dates=["timestamp"],
        low_memory=False,
    )
    value = pd.to_numeric(strategy["oracle_dp_log_value_from_here"], errors="raise").to_numpy(dtype=np.float64)
    margin = value - np.r_[value[1:], value[-1]]
    entry = strategy["oracle_dp_selected"].astype(bool).to_numpy()
    valid = strategy["label_evaluable"].astype(bool).to_numpy()
    net = pd.to_numeric(strategy["oracle_net_return_per_notional"], errors="raise").to_numpy(dtype=np.float64)
    side = pd.to_numeric(strategy["oracle_side"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(pd.to_datetime(strategy["decision_timestamp"]).to_numpy(), trajectory["timestamp"].to_numpy()):
        raise RuntimeError(f"{split}: strategy/trajectory timestamp mismatch")
    out = trajectory.copy()
    out["oracle_entry_selected"] = entry.astype(np.int8)
    out["oracle_entry_side"] = side.astype(np.int8)
    out["oracle_entry_net_return"] = net
    out["oracle_dp_entry_advantage"] = margin
    out["oracle_quality_candidate"] = (entry & valid & (net > 0.0) & (margin > 0.0)).astype(np.int8)
    return out


def main() -> int:
    splits = {name: load_split(name) for name in ("train", "validation", "oos")}
    train_candidates = splits["train"].loc[splits["train"]["oracle_quality_candidate"].astype(bool), "oracle_dp_entry_advantage"]
    if train_candidates.empty:
        raise RuntimeError("empty Train Oracle quality candidate set")
    for variant, quantile in VARIANTS.items():
        cutoff = float(train_candidates.quantile(quantile))
        out_dir = OUT_ROOT / f"eth_oracle_entry_quality_{variant}_20260724"
        out_dir.mkdir(parents=True, exist_ok=True)
        combined = []
        summaries = {}
        for split, frame in splits.items():
            out = frame.copy()
            positive = out["oracle_quality_candidate"].astype(bool) & (out["oracle_dp_entry_advantage"] >= cutoff)
            quality = np.zeros(len(out), dtype=np.int64)
            side = out["oracle_entry_side"].to_numpy(dtype=np.int64)
            quality[positive & (side > 0)] = 1
            quality[positive & (side < 0)] = 2
            out["oracle_quality_action"] = quality
            out["oracle_quality_variant"] = variant
            out["oracle_quality_train_advantage_cutoff"] = cutoff
            out.to_csv(out_dir / f"{split}_quality_labels.csv", index=False)
            combined.append(out)
            summaries[split] = {
                "rows": int(len(out)),
                "valid_rows": int(out["oracle_label_valid"].sum()),
                "candidate_entries": int(out["oracle_quality_candidate"].sum()),
                "quality_positive_rows": int((quality != 0).sum()),
                "quality_positive_ratio": float((quality != 0).mean()),
                "counts": {str(int(k)): int(v) for k, v in pd.Series(quality).value_counts().sort_index().items()},
            }
        all_rows = pd.concat(combined, ignore_index=True)
        artifacts = {}
        for year in (2024, 2025, 2026):
            year_frame = all_rows.loc[all_rows["timestamp"].dt.year == year].reset_index(drop=True)
            path = out_dir / f"zigzag_action_labels_{year}.csv"
            year_frame.to_csv(path, index=False)
            artifacts[str(year)] = str(path)
        report = {
            "model_id": f"eth_oracle_entry_quality_{variant}_20260724",
            "direction_target": "split-local Oracle trajectory",
            "quality_target": "split-local Oracle selected entry with positive net return and Train-only top DP advantage",
            "train_advantage_quantile": quantile,
            "train_advantage_cutoff": cutoff,
            "validation_or_oos_distribution_used_for_cutoff": False,
            "summaries": summaries,
            "artifacts": artifacts,
        }
        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"variant": variant, "cutoff": cutoff, "summaries": summaries}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
