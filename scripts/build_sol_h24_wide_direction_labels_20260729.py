#!/usr/bin/env python3
"""Build research-only wide SOL H24 direction labels.

The transform minimizes per-bar disagreement with ``tb_action_h24_conservative``
while charging a fixed cost for every state transition.  Train, validation, and
OOS are denoised independently so label construction cannot cross split bounds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TB_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega1_2_triple_barrier_labels_hysteresis_rebuild_20260719"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_h24_wide_direction_labels_penalty12_20260729"
DIRECTION_DIR = ROOT / "tmp/causal_regen_20260516/sol_zigzag_action_labels_20260707"
COLUMN = "tb_action_h24_conservative"
SWITCH_PENALTY = 12


def widen(actions: np.ndarray, switch_penalty: int) -> np.ndarray:
    if actions.ndim != 1 or len(actions) == 0:
        raise ValueError("actions must be a non-empty 1D array")
    if not np.isin(actions, [0, 1, 2]).all():
        raise ValueError("actions must contain only CASH=0, LONG=1, SHORT=2")

    costs = (np.arange(3) != int(actions[0])).astype(np.int64)
    parents = np.empty((len(actions) - 1, 3), dtype=np.int8)
    for index, observed in enumerate(actions[1:]):
        transition_costs = costs[:, None] + int(switch_penalty) * (np.arange(3)[:, None] != np.arange(3)[None, :])
        previous = np.argmin(transition_costs, axis=0)
        costs = transition_costs[previous, np.arange(3)] + (np.arange(3) != int(observed))
        parents[index] = previous

    output = np.empty(len(actions), dtype=np.int8)
    output[-1] = int(np.argmin(costs))
    for index in range(len(actions) - 2, -1, -1):
        output[index] = parents[index, output[index + 1]]
    return output


def diagnostics(actions: np.ndarray) -> dict[str, object]:
    starts = np.r_[0, np.flatnonzero(actions[1:] != actions[:-1]) + 1]
    ends = np.r_[starts[1:], len(actions)]
    lengths = ends - starts
    return {
        "rows": int(len(actions)),
        "counts": {str(label): int(np.sum(actions == label)) for label in (0, 1, 2)},
        "runs": int(len(lengths)),
        "median_run_bars": float(np.median(lengths)),
        "p10_run_bars": float(np.quantile(lengths, 0.10)),
        "p90_run_bars": float(np.quantile(lengths, 0.90)),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_year: dict[int, list[pd.DataFrame]] = {2025: [], 2026: []}
    audit: dict[str, object] = {
        "type": "potts_denoised_h24_conservative_direction_label",
        "source_column": COLUMN,
        "label_mapping": {"0": "CASH", "1": "LONG", "2": "SHORT"},
        "mismatch_cost": 1,
        "switch_penalty": SWITCH_PENALTY,
        "split_independent_transform": True,
        "research_only": True,
        "splits": {},
    }
    for split in ("train", "validation", "oos"):
        source = TB_DIR / f"{split}_triple_barrier_labels.csv"
        frame = pd.read_csv(source, usecols=["timestamp", COLUMN], parse_dates=["timestamp"])
        raw = pd.to_numeric(frame[COLUMN], errors="raise").to_numpy(dtype=np.int8)
        wide = widen(raw, SWITCH_PENALTY)
        output = pd.DataFrame({"timestamp": frame["timestamp"], "zigzag_action": wide})
        years = output["timestamp"].dt.year.unique().tolist()
        if len(years) != 1 or int(years[0]) not in by_year:
            raise RuntimeError(f"{split}: unexpected years {years}")
        by_year[int(years[0])].append(output)
        audit["splits"][split] = {
            "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
            "before": diagnostics(raw),
            "after": diagnostics(wide),
            "changed_rows": int(np.sum(raw != wide)),
        }

    for year, parts in by_year.items():
        output = pd.concat(parts, ignore_index=True).sort_values("timestamp")
        if output["timestamp"].duplicated().any():
            raise RuntimeError(f"{year}: duplicate timestamps")
        direction_index = pd.read_csv(
            DIRECTION_DIR / f"zigzag_action_labels_{year}.csv",
            usecols=["timestamp"],
            parse_dates=["timestamp"],
        ).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        padded = direction_index.merge(output, on="timestamp", how="left", validate="one_to_one")
        missing = int(padded["zigzag_action"].isna().sum())
        padded["zigzag_action"] = padded["zigzag_action"].fillna(0).astype(np.int8)
        padded.to_csv(OUT_DIR / f"zigzag_action_labels_{year}.csv", index=False)
        audit.setdefault("year_padding", {})[str(year)] = {
            "direction_index_rows": int(len(direction_index)),
            "h24_rows_before_padding": int(len(output)),
            "cash_padded_rows": missing,
        }

    (OUT_DIR / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
