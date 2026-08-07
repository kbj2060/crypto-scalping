#!/usr/bin/env python3
"""Rebuild ETH Zig075 labels independently inside the new Train/Val/OOS splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_wave3_action_labels_20260531 as zigzag  # noqa: E402


MODEL_ID = "eth_split_zig075_labels_20260724"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
RIGHT_CENSOR_ROWS = 97
SPLITS = {
    "train": (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")),
    "validation": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")),
    "oos": (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-07-21")),
}


def load_market() -> pd.DataFrame:
    parts = []
    for year, name in (
        (2024, "training_features_2024.csv"),
        (2025, "training_features_2025.csv"),
        (2026, "training_features_2026_rebuilt.csv"),
    ):
        frame = pd.read_csv(
            ROOT / "data/splits/year_oos" / name,
            parse_dates=["timestamp"],
            low_memory=False,
        )
        if sorted(frame["timestamp"].dt.year.unique().tolist()) != [year]:
            raise RuntimeError(f"{name}: year contract mismatch")
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    if combined["timestamp"].duplicated().any() or not combined["timestamp"].is_monotonic_increasing:
        raise RuntimeError("combined ZigZag market frame violates timestamp contract")
    return combined


def build_split(frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
    labels = zigzag.build_zigzag_action_labels(
        frame,
        min_reversal_pct=0.009,
        min_wave_bars=6,
        transition_buffer=1,
        atr_window=14,
        atr_multiplier=1.0,
        mae_penalty=1.1,
        softmax_temperature=1.9,
        min_risk_floor=0.001,
    )
    labels["oracle_label_valid"] = 1
    labels.loc[labels.index[-RIGHT_CENSOR_ROWS:], "oracle_label_valid"] = 0
    labels["oracle_split"] = split
    labels["label_source"] = "split_local_zig075"
    return labels


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    market = load_market()
    split_frames: list[pd.DataFrame] = []
    summaries = {}
    for split, (start, end) in SPLITS.items():
        frame = market.loc[(market["timestamp"] >= start) & (market["timestamp"] < end)].reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"{split}: empty market split")
        labels = build_split(frame, split=split)
        split_frames.append(labels)
        path = OUT_DIR / f"{split}_zig075_labels.csv"
        labels.to_csv(path, index=False)
        valid = labels["oracle_label_valid"].astype(bool)
        summaries[split] = {
            "range": [str(labels["timestamp"].iloc[0]), str(labels["timestamp"].iloc[-1])],
            "rows": int(len(labels)),
            "valid_rows": int(valid.sum()),
            "right_censored_rows": RIGHT_CENSOR_ROWS,
            "valid_counts": {str(int(k)): int(v) for k, v in labels.loc[valid, "zigzag_action"].value_counts().sort_index().items()},
        }
    combined = pd.concat(split_frames, ignore_index=True)
    if combined["timestamp"].duplicated().any():
        raise RuntimeError("split ZigZag output contains duplicate timestamps")
    artifacts = {}
    for year in (2024, 2025, 2026):
        year_frame = combined.loc[combined["timestamp"].dt.year == year].reset_index(drop=True)
        path = OUT_DIR / f"zigzag_action_labels_{year}.csv"
        year_frame.to_csv(path, index=False)
        artifacts[str(year)] = str(path)
    report = {
        "model_id": MODEL_ID,
        "parameters": {
            "zigzag_reversal_pct": 0.009,
            "min_wave_bars": 6,
            "transition_buffer": 1,
            "atr_window": 14,
            "atr_multiplier": 1.0,
            "mae_penalty": 1.1,
            "softmax_temperature": 1.9,
            "min_risk_floor": 0.001,
        },
        "split_local_generation": True,
        "cross_split_future_rows_used": False,
        "right_censor_rows_per_split": RIGHT_CENSOR_ROWS,
        "summaries": summaries,
        "artifacts": artifacts,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
