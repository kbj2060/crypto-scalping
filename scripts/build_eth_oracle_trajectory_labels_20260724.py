#!/usr/bin/env python3
"""Convert full-history ETH oracle trades into the current 3-head action schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    ROOT
    / "tmp/causal_regen_20260516/eth_full_oracle_strategy_labels_v1_20260724"
)
SOURCE_LABELS = SOURCE_DIR / "full_oracle_strategy_labels.parquet"
OUT_DIR = (
    ROOT / "tmp/causal_regen_20260516/eth_oracle_trajectory_action_labels_20260724"
)
SPLIT_TS = pd.Timestamp("2025-10-01 00:00:00")
PURGE_BARS = 96


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_trajectory(labels: pd.DataFrame) -> pd.DataFrame:
    required = {
        "decision_index",
        "decision_timestamp",
        "oracle_dp_selected",
        "oracle_side",
        "oracle_event_end_index",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise RuntimeError(f"oracle labels missing columns: {missing}")
    ordered = labels.sort_values("decision_index").reset_index(drop=True)
    expected = np.arange(len(ordered), dtype=np.int64)
    actual = ordered["decision_index"].to_numpy(np.int64)
    if not np.array_equal(actual, expected):
        raise RuntimeError("oracle decision_index must be contiguous and zero-based")

    action = np.zeros(len(ordered), dtype=np.int8)
    selected = ordered.loc[ordered["oracle_dp_selected"].astype(bool)]
    previous_end = -1
    for row in selected.itertuples(index=False):
        start = int(row.decision_index)
        end = int(row.oracle_event_end_index)
        if start < previous_end:
            raise RuntimeError(
                f"overlapping oracle trades: start={start}, previous_end={previous_end}"
            )
        if not 0 <= start < end <= len(ordered):
            raise RuntimeError(f"invalid oracle interval: [{start}, {end})")
        side = int(row.oracle_side)
        if side not in (-1, 1):
            raise RuntimeError(f"invalid selected oracle side: {side}")
        action[start:end] = 1 if side == 1 else 2
        previous_end = end

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ordered["decision_timestamp"]),
            "zigzag_action": action.astype(np.int64),
        }
    )
    if out["timestamp"].duplicated().any():
        raise RuntimeError("oracle trajectory contains duplicate timestamps")
    return out


def apply_train_boundary_purge(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = frame.copy()
    train_idx = np.flatnonzero(out["timestamp"].to_numpy() < np.datetime64(SPLIT_TS))
    purge_idx = train_idx[-PURGE_BARS:]
    active_before = int((out.loc[purge_idx, "zigzag_action"] != 0).sum())
    out.loc[purge_idx, "zigzag_action"] = 0
    return out, {
        "split_timestamp": str(SPLIT_TS),
        "purge_bars": int(PURGE_BARS),
        "purge_start": str(out.loc[purge_idx[0], "timestamp"]),
        "purge_end": str(out.loc[purge_idx[-1], "timestamp"]),
        "active_rows_replaced_with_cash": active_before,
    }


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(int(key)): int(value)
        for key, value in frame["zigzag_action"].value_counts().sort_index().items()
    }


def main() -> int:
    labels = pd.read_parquet(
        SOURCE_LABELS,
        columns=[
            "decision_index",
            "decision_timestamp",
            "oracle_dp_selected",
            "oracle_side",
            "oracle_event_end_index",
        ],
    )
    trajectory = build_trajectory(labels)
    trajectory, purge = apply_train_boundary_purge(trajectory)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict[str, object]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for year in (2025, 2026):
        year_frame = trajectory.loc[trajectory["timestamp"].dt.year == year].reset_index(
            drop=True
        )
        path = OUT_DIR / f"zigzag_action_labels_{year}.csv"
        year_frame.to_csv(path, index=False)
        artifacts[str(year)] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": int(len(year_frame)),
        }
        summaries[str(year)] = {
            "range": [
                str(year_frame["timestamp"].iloc[0]),
                str(year_frame["timestamp"].iloc[-1]),
            ],
            "counts": label_counts(year_frame),
            "active_ratio": float((year_frame["zigzag_action"] != 0).mean()),
        }

    train = trajectory.loc[trajectory["timestamp"] < SPLIT_TS]
    validation = trajectory.loc[
        (trajectory["timestamp"] >= SPLIT_TS)
        & (trajectory["timestamp"] < pd.Timestamp("2026-01-01"))
    ]
    report = {
        "model_id": "eth_oracle_trajectory_action_labels_20260724",
        "status": "labels_built",
        "source": {
            "path": str(SOURCE_LABELS),
            "sha256": sha256(SOURCE_LABELS),
            "future_rows_used_for_label": True,
        },
        "label_contract": {
            "classes": {"0": "CASH", "1": "LONG", "2": "SHORT"},
            "interval": "selected decision_index inclusive to oracle_event_end_index exclusive",
            "same_side_adjacent_trades": "continuous position-state segment",
            "purpose": "offline supervised target only",
            "entry_features_use_future_rows": False,
        },
        "external_split_purge": purge,
        "split_counts": {
            "train": {
                "rows": int(len(train)),
                "counts": label_counts(train),
            },
            "validation": {
                "rows": int(len(validation)),
                "counts": label_counts(validation),
            },
        },
        "year_summaries": summaries,
        "artifacts": artifacts,
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
