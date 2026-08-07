#!/usr/bin/env python3
"""Build split-local ETH Oracle-DP labels without cross-split DP recursion."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_eth_full_oracle_strategy_labels_20260724 as oracle  # noqa: E402


MODEL_ID = "eth_split_oracle_strategy_labels_20260724"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPLITS = {
    "train": (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")),
    "validation": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")),
    "oos": (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-07-21")),
}


def load_full_frame() -> pd.DataFrame:
    market_2024 = ROOT / "data/splits/year_oos/training_features_2024.csv"
    hmm_2024 = (
        oracle.base.HMM_ARTIFACT.parent
        / "training_features_2024_regime3_current_sensitive_hmm_wide24.csv"
    )
    frame_2024 = oracle.base.load_frame(
        market_2024, hmm_2024, oracle.base.RISK_ARTIFACT
    )
    frame_2025 = oracle.base.load_frame(
        oracle.base.MARKET_2025, oracle.base.HMM_2025, oracle.base.RISK_ARTIFACT
    )
    frame_2026 = oracle.base.load_frame(
        oracle.base.MARKET_2026, oracle.base.HMM_2026, oracle.base.RISK_ARTIFACT
    )
    parts = []
    for year, frame in ((2024, frame_2024), (2025, frame_2025), (2026, frame_2026)):
        part = frame.copy()
        part["source_year"] = year
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    if combined["timestamp"].duplicated().any() or not combined["timestamp"].is_monotonic_increasing:
        raise RuntimeError("combined 2024-2026 oracle frame violates timestamp contract")
    combined["oracle_context_atr192"] = oracle.base.compute_atr(combined)
    combined["oracle_context_vwma100"] = oracle.base.compute_vwma(
        combined["close"], combined["volume"], oracle.base.VWMA_FAST
    )
    combined["oracle_context_vwma288"] = oracle.base.compute_vwma(
        combined["close"], combined["volume"], oracle.base.VWMA_SLOW
    )
    return combined


def build_trajectory(labels: pd.DataFrame, *, split: str) -> pd.DataFrame:
    ordered = labels.sort_values("decision_index").reset_index(drop=True)
    expected = np.arange(len(ordered), dtype=np.int64)
    if not np.array_equal(ordered["decision_index"].to_numpy(np.int64), expected):
        raise RuntimeError(f"{split}: decision_index is not contiguous and zero-based")

    action = np.zeros(len(ordered), dtype=np.int8)
    previous_end = -1
    for row in ordered.loc[ordered["oracle_dp_selected"].astype(bool)].itertuples(index=False):
        start = int(row.decision_index)
        end = int(row.oracle_event_end_index)
        if start < previous_end:
            raise RuntimeError(f"{split}: overlapping DP trades at {start}")
        if not 0 <= start < end <= len(ordered):
            raise RuntimeError(f"{split}: invalid DP interval [{start}, {end})")
        side = int(row.oracle_side)
        if side not in (-1, 1):
            raise RuntimeError(f"{split}: invalid selected side {side}")
        action[start:end] = 1 if side > 0 else 2
        previous_end = end

    valid = ordered["label_evaluable"].to_numpy(dtype=np.int8)
    action[valid == 0] = 0
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ordered["decision_timestamp"]),
            "zigzag_action": action.astype(np.int64),
            "oracle_label_valid": valid,
            "oracle_label_invalid_reason": ordered["label_invalid_reason"].astype(str),
            "oracle_split": split,
        }
    )


def label_counts(frame: pd.DataFrame, *, valid_only: bool = False) -> dict[str, int]:
    source = frame.loc[frame["oracle_label_valid"].astype(bool)] if valid_only else frame
    return {
        str(int(key)): int(value)
        for key, value in source["zigzag_action"].value_counts().sort_index().items()
    }


def run_split(
    full_frame: pd.DataFrame,
    tape: oracle.base.FundingTape,
    *,
    split: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = full_frame.loc[
        (full_frame["timestamp"] >= start) & (full_frame["timestamp"] < end)
    ].copy().reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"{split}: no rows in requested interval {start} -> {end}")

    evaluation = oracle.evaluate_actions(frame, tape)
    value, selected, selected_action = oracle.dynamic_program(evaluation, len(frame))
    labels, trades = oracle.build_labels(
        frame, evaluation, value, selected, selected_action
    )
    row_index = labels["decision_index"].to_numpy(dtype=np.int64)
    finite_atr = np.isfinite(frame["oracle_context_atr192"].to_numpy(dtype=np.float64))
    valid_label = (row_index < evaluation.evaluable_rows) & finite_atr
    labels["label_evaluable"] = valid_label.astype(np.int8)
    labels["label_invalid_reason"] = np.where(
        row_index >= evaluation.evaluable_rows,
        "right_censored_max_horizon",
        np.where(~finite_atr, "atr_warmup", ""),
    )
    labels.insert(0, "oracle_split", split)
    if len(trades):
        trades.insert(0, "oracle_split", split)
    trajectory = build_trajectory(labels, split=split)

    invalid_tail = len(frame) - evaluation.evaluable_rows
    if invalid_tail != oracle.MAX_HORIZON + 1:
        raise RuntimeError(
            f"{split}: expected {oracle.MAX_HORIZON + 1} right-censored rows, got {invalid_tail}"
        )
    tail = trajectory.tail(invalid_tail)
    if tail["oracle_label_valid"].any() or (tail["zigzag_action"] != 0).any():
        raise RuntimeError(f"{split}: right-censored rows leaked into training targets")
    if len(trades) and pd.to_datetime(trades["event_end_timestamp"]).max() > end:
        raise RuntimeError(f"{split}: selected trade crosses the split boundary")

    log_return = float(value[0])
    summary = {
        "configured_start": str(start),
        "configured_end_exclusive": str(end),
        "actual_range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
        "rows": int(len(frame)),
        "valid_label_rows": int(trajectory["oracle_label_valid"].sum()),
        "right_censored_decision_rows": int(invalid_tail),
        "requested_last_horizon_bars_excluded": int(oracle.MAX_HORIZON),
        "additional_next_open_boundary_row_excluded": 1,
        "atr_warmup_rows_excluded": int((labels["label_invalid_reason"] == "atr_warmup").sum()),
        "counts_all_rows": label_counts(trajectory),
        "counts_valid_rows": label_counts(trajectory, valid_only=True),
        "selected_trades": int(len(trades)),
        "long_trades": int((trades["side"] > 0).sum()) if len(trades) else 0,
        "short_trades": int((trades["side"] < 0).sum()) if len(trades) else 0,
        "total_log_return": log_return,
        "oracle_equity_multiple": math.exp(log_return) if log_return < 700 else None,
    }
    return labels, trades, trajectory, summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_frame = load_full_frame()
    tape, funding_hashes = oracle.base.load_funding_tape(full_frame)
    trajectories: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}

    for split, (start, end) in SPLITS.items():
        print(json.dumps({"stage": "split_dp", "split": split}), flush=True)
        labels, trades, trajectory, summary = run_split(
            full_frame, tape, split=split, start=start, end=end
        )
        label_path = OUT_DIR / f"{split}_oracle_strategy_labels.parquet"
        trade_path = OUT_DIR / f"{split}_oracle_selected_trades.csv"
        trajectory_path = OUT_DIR / f"{split}_oracle_trajectory_labels.csv"
        labels.to_parquet(label_path, index=False)
        trades.to_csv(trade_path, index=False)
        trajectory.to_csv(trajectory_path, index=False)
        trajectories.append(trajectory)
        summaries[split] = summary
        artifacts[split] = {
            "strategy_labels": {"path": str(label_path), "sha256": oracle.base.sha256(label_path)},
            "selected_trades": {"path": str(trade_path), "sha256": oracle.base.sha256(trade_path)},
            "trajectory_labels": {"path": str(trajectory_path), "sha256": oracle.base.sha256(trajectory_path)},
        }

    combined = pd.concat(trajectories, ignore_index=True)
    if combined["timestamp"].duplicated().any():
        raise RuntimeError("split-local trajectory labels contain duplicate timestamps")
    for year in (2024, 2025, 2026):
        year_frame = combined.loc[combined["timestamp"].dt.year == year].reset_index(drop=True)
        path = OUT_DIR / f"zigzag_action_labels_{year}.csv"
        year_frame.to_csv(path, index=False)
        artifacts[f"training_schema_{year}"] = {
            "path": str(path),
            "sha256": oracle.base.sha256(path),
            "rows": int(len(year_frame)),
        }

    grid_path = OUT_DIR / "action_grid.json"
    grid_path.write_text(
        json.dumps([asdict(spec) for spec in oracle.action_grid()], indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "model_id": MODEL_ID,
        "status": "split_local_oracle_labels_generated",
        "purpose": "offline_supervised_targets_with_split_local_hindsight_only",
        "split_contract": {
            "train": "2024-01-01 <= timestamp < 2026-01-01",
            "validation": "2026-01-01 <= timestamp < 2026-04-01",
            "oos": "2026-04-01 <= timestamp < 2026-07-21 (actual source ends 2026-07-20 00:00)",
            "boundary_change_disclosure": "resplit on 2026-07-24 after the ETH source was extended through July",
            "dp_terminal_value": "zero independently at the end of every split",
            "cross_split_price_rows_used_by_dp": False,
            "cross_split_dp_value_recursion": False,
        },
        "right_censor_contract": {
            "requested_max_horizon_bars": int(oracle.MAX_HORIZON),
            "excluded_decision_rows_per_split": int(oracle.MAX_HORIZON + 1),
            "reason": "entry is next-bar open and a 96-bar timeout exits at the following open",
        },
        "training_contract": {
            "weight_updates": "train valid labels only",
            "validation_labels": "diagnostic only; threshold and sizing selection use validation realized performance",
            "oos_labels": "post-hoc agreement diagnostic only; never model input or selection input",
            "oos_evaluation": "single causal fresh-forward bar-by-bar pass after validation freeze",
        },
        "future_rows_used_for_label": True,
        "future_rows_used_for_entry_features": False,
        "future_rows_used_for_entry": False,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "promotion_eligible": False,
        "promotion_blocker": "model retraining, validation freeze, fresh-forward OOS, and artifact integrity audit remain",
        "action_grid": [asdict(spec) for spec in oracle.action_grid()],
        "split_summaries": summaries,
        "funding_hashes": funding_hashes,
        "artifacts": artifacts,
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=oracle.base._json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "splits": summaries}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
