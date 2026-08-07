#!/usr/bin/env python3
"""Build split-local SOL Zig075 and widened-H24 labels for a dual-parent study."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_omega1_2_triple_barrier_labels_sol_20260707 as triple_barrier  # noqa: E402
import build_wave3_action_labels_20260531 as zigzag  # noqa: E402


MODEL_ID = "sol_dual_zig075_h24wide_splitlocal_20260729"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DIRECTION_DIR = OUT_DIR / "zig075"
H24WIDE_DIR = OUT_DIR / "h24wide"
RAW_H24_DIR = OUT_DIR / "h24_raw"
SWITCH_PENALTY = 12
RIGHT_CENSOR_ROWS = 97
SPLITS = {
    "train": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-09-01")),
    "validation": (pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01")),
    "oos": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")),
}


def widen(actions: np.ndarray, switch_penalty: int) -> np.ndarray:
    if actions.ndim != 1 or len(actions) == 0:
        raise ValueError("actions must be a non-empty 1D array")
    if not np.isin(actions, [0, 1, 2]).all():
        raise ValueError("actions must contain only CASH=0, LONG=1, SHORT=2")
    costs = (np.arange(3) != int(actions[0])).astype(np.int64)
    parents = np.empty((len(actions) - 1, 3), dtype=np.int8)
    states = np.arange(3)
    for index, observed in enumerate(actions[1:]):
        transition = costs[:, None] + int(switch_penalty) * (states[:, None] != states[None, :])
        previous = np.argmin(transition, axis=0)
        costs = transition[previous, states] + (states != int(observed))
        parents[index] = previous
    output = np.empty(len(actions), dtype=np.int8)
    output[-1] = int(np.argmin(costs))
    for index in range(len(actions) - 2, -1, -1):
        output[index] = parents[index, output[index + 1]]
    return output


def label_stats(actions: np.ndarray) -> dict[str, Any]:
    starts = np.r_[0, np.flatnonzero(actions[1:] != actions[:-1]) + 1]
    ends = np.r_[starts[1:], len(actions)]
    lengths = ends - starts
    non_cash = actions[actions != 0]
    direct_flips = int(np.sum(non_cash[1:] != non_cash[:-1])) if len(non_cash) > 1 else 0
    return {
        "rows": int(len(actions)),
        "counts": {str(label): int(np.sum(actions == label)) for label in (0, 1, 2)},
        "runs": int(len(lengths)),
        "median_run_bars": float(np.median(lengths)),
        "p10_run_bars": float(np.quantile(lengths, 0.10)),
        "p90_run_bars": float(np.quantile(lengths, 0.90)),
        "all_action_changes": int(np.sum(actions[1:] != actions[:-1])),
        "direct_non_cash_flips": direct_flips,
        "hours_per_direct_non_cash_flip": float(len(actions) / 12.0 / max(direct_flips, 1)),
    }


def load_market() -> pd.DataFrame:
    parts = []
    for year in (2025, 2026):
        frame = pd.read_csv(
            ROOT / f"data/splits/year_oos/sol_features_{year}.csv",
            parse_dates=["timestamp"],
            low_memory=False,
        )
        if sorted(frame["timestamp"].dt.year.unique().tolist()) != [year]:
            raise RuntimeError(f"SOL {year} feature file violates year contract")
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if combined["timestamp"].duplicated().any() or not combined["timestamp"].is_monotonic_increasing:
        raise RuntimeError("combined SOL frame violates timestamp contract")
    return combined


def build_zigzag(frame: pd.DataFrame) -> pd.DataFrame:
    labels = zigzag.build_zigzag_action_labels(
        frame,
        min_reversal_pct=0.010,
        min_wave_bars=8,
        transition_buffer=2,
        atr_window=14,
        atr_multiplier=1.0,
        mae_penalty=1.25,
        softmax_temperature=1.75,
        min_risk_floor=0.001,
    )
    tail = labels.index[-RIGHT_CENSOR_ROWS:]
    labels.loc[tail, "zigzag_action"] = 0
    labels.loc[tail, "zigzag_action_name"] = "CASH"
    for column in ("zigzag_path_return", "zigzag_path_mae", "zigzag_path_mfe", "zigzag_path_calmar", "zigzag_path_edge"):
        labels.loc[tail, column] = 0.0
    labels.loc[tail, ["zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"]] = [1.0, 0.0, 0.0]
    labels["oracle_label_valid"] = 1
    labels.loc[tail, "oracle_label_valid"] = 0
    return labels


def build_h24wide(frame: pd.DataFrame, split: str, fee_cost: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_tb, audit = triple_barrier._build_split_labels(frame, fee_cost=fee_cost)
    raw = all_tb[["timestamp", "tb_action_h24_conservative"]].copy()
    raw_action = pd.to_numeric(raw["tb_action_h24_conservative"], errors="raise").to_numpy(dtype=np.int8)
    wide_action = widen(raw_action, SWITCH_PENALTY)
    wide_valid = pd.DataFrame({"timestamp": raw["timestamp"], "zigzag_action": wide_action})
    padded = frame[["timestamp"]].merge(wide_valid, on="timestamp", how="left", validate="one_to_one")
    padded["oracle_label_valid"] = padded["zigzag_action"].notna().astype(np.int8)
    padded["zigzag_action"] = padded["zigzag_action"].fillna(0).astype(np.int8)
    padded["oracle_split"] = split
    return padded, raw, {
        "raw": label_stats(raw_action),
        "wide_valid": label_stats(wide_action),
        "right_censored_cash_rows": int((padded["oracle_label_valid"] == 0).sum()),
        "triple_barrier_h24": audit["h24_conservative"],
        "switch_penalty_grid": {
            str(penalty): {
                **label_stats(widen(raw_action, penalty)),
                "agreement_with_raw": float(np.mean(widen(raw_action, penalty) == raw_action)),
            }
            for penalty in (0, 3, 6, 9, 12, 18, 24)
        },
    }


def main() -> int:
    for directory in (OUT_DIR, DIRECTION_DIR, H24WIDE_DIR, RAW_H24_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    market = load_market()
    fee_cost = float(triple_barrier.FEE_RATE + triple_barrier.SLIP_RATE) * 2.0 * 3.0
    zig_parts = []
    wide_parts = []
    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "split_local_generation": True,
        "cross_split_future_rows_used": False,
        "right_censor_rows_per_split": RIGHT_CENSOR_ROWS,
        "h24_switch_penalty": SWITCH_PENALTY,
        "h24_mismatch_cost": 1,
        "splits": {},
        "artifacts": {},
    }
    for split, (start, end) in SPLITS.items():
        frame = market.loc[(market["timestamp"] >= start) & (market["timestamp"] < end)].reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"{split}: empty frame")
        zig = build_zigzag(frame)
        wide, raw_h24, h24_diag = build_h24wide(frame, split, fee_cost)
        zig["oracle_split"] = split
        zig_parts.append(zig)
        wide_parts.append(wide)
        raw_path = RAW_H24_DIR / f"{split}_h24_conservative_raw.csv"
        raw_h24.to_csv(raw_path, index=False)
        report["splits"][split] = {
            "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
            "rows": int(len(frame)),
            "zig075": label_stats(zig["zigzag_action"].to_numpy(dtype=np.int8)),
            "zig075_right_censored_rows": int((zig["oracle_label_valid"] == 0).sum()),
            "h24wide": h24_diag,
        }

    zig_all = pd.concat(zig_parts, ignore_index=True)
    wide_all = pd.concat(wide_parts, ignore_index=True)
    for name, labels, directory in (("zig075", zig_all, DIRECTION_DIR), ("h24wide", wide_all, H24WIDE_DIR)):
        if labels["timestamp"].duplicated().any():
            raise RuntimeError(f"{name}: duplicate timestamps")
        report["artifacts"][name] = {}
        for year in (2025, 2026):
            year_frame = labels.loc[labels["timestamp"].dt.year == year].reset_index(drop=True)
            path = directory / f"zigzag_action_labels_{year}.csv"
            year_frame.to_csv(path, index=False)
            report["artifacts"][name][str(year)] = str(path)

    report_path = OUT_DIR / "report.json"
    report["artifacts"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
