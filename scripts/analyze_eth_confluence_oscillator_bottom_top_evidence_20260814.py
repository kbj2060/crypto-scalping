#!/usr/bin/env python3
"""Evidence study (NOT a trading algorithm): does "Slow %K + Williams %R both oversold/
overbought" actually mean the market is near a real swing low/high?

This is a retrospective diagnostic, not a live causal signal or a backtest. Ground truth for
"was this actually a bottom/top" comes from ETH's already-built zigzag swing labels
(tmp/zigzag_action_labels_extended_20260809/zigzag_action_labels_{year}.csv, zigzag_action:
0=CASH/transition-buffer, 1=LONG=inside an up-swing, 2=SHORT=inside a down-swing -- the same
label h48qual and every zigzag-based script in this repo uses), NOT a hand-rolled swing
detector. A SHORT-run's minimum-low bar is that down-swing's real bottom pivot; a LONG-run's
maximum-high bar is that up-swing's real top pivot. Looking forward from a trigger bar to see
whether a real pivot follows is intentional here (retrospective evidence-gathering about what
the indicator combination has historically meant), unlike the fresh-forward backtest rule,
which governs live-tradeable promotion claims -- this script makes no promotion or live-signal
claim, only reports historical hit-rates for a human to weigh.

Question decomposed into 4 sub-questions, each with a baseline comparison:
  1. Precision: of bars where the signal fires, what fraction are within K bars of a real pivot?
     Compared against the base rate (P(near a real pivot | a random bar)) for lift.
  2. Recall: of real pivots, what fraction were preceded by the signal within K bars?
  3. Lead time: among true positives, how many bars from signal to the real pivot?
  4. Excess move: among true positives, how much further does price move (in the "wrong"
     direction first) between the signal bar and the real pivot? This is the direct evidence
     for/against "it's the bottom right now" -- a big further move means the signal is early,
     not a bottom call.
  5. Combination check: does requiring BOTH indicators beat using EITHER alone? This is the
     literal premise behind combining Williams %R and Slow %K and gets tested directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
    compute_indicators,
    load_frame,
)

ZIGZAG_DIR = ROOT / "tmp" / "zigzag_action_labels_extended_20260809"
OOS_END = pd.Timestamp("2026-02-17 15:00:00")  # matches raw data's actual coverage, see backtest script
K_HORIZONS = {"K12_1h": 12, "K48_4h": 48, "K96_8h": 96}


def load_zigzag_pivots() -> pd.DataFrame:
    """Real swing bottom/top pivot bars from ETH's zigzag_action label, one file per year."""
    frames = []
    for year in (2025, 2026):
        path = ZIGZAG_DIR / f"zigzag_action_labels_{year}.csv"
        z = pd.read_csv(path, parse_dates=["timestamp"], usecols=["timestamp", "low", "high", "zigzag_action"])
        frames.append(z)
    zz = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    zz = zz.reset_index(drop=True)

    run_id = (zz["zigzag_action"] != zz["zigzag_action"].shift()).cumsum()
    pivots = []
    for _, run in zz.groupby(run_id):
        action = int(run["zigzag_action"].iloc[0])
        if action == 2:  # SHORT run -> its lowest low is that down-swing's real bottom
            pivot_row = run.loc[run["low"].idxmin()]
            pivots.append({"timestamp": pivot_row["timestamp"], "pivot_type": "bottom", "pivot_price": pivot_row["low"]})
        elif action == 1:  # LONG run -> its highest high is that up-swing's real top
            pivot_row = run.loc[run["high"].idxmax()]
            pivots.append({"timestamp": pivot_row["timestamp"], "pivot_type": "top", "pivot_price": pivot_row["high"]})
    return pd.DataFrame(pivots).sort_values("timestamp").reset_index(drop=True)


def event_study(trigger_pos: np.ndarray, pivot_pos: np.ndarray, all_pos: np.ndarray, K: int) -> dict:
    """precision/recall/lift/lead-time of `trigger_pos` predicting a pivot in `pivot_pos` within K bars forward."""
    pivot_pos_sorted = np.sort(pivot_pos)

    def next_pivot_dist(positions: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(pivot_pos_sorted, positions, side="left")
        dist = np.full(len(positions), np.inf)
        valid = idx < len(pivot_pos_sorted)
        dist[valid] = pivot_pos_sorted[idx[valid]] - positions[valid]
        return dist

    trig_dist = next_pivot_dist(trigger_pos)
    hits = trig_dist <= K
    precision = float(hits.mean()) if len(trigger_pos) else float("nan")
    lead_times = trig_dist[hits]
    median_lead = float(np.median(lead_times)) if hits.any() else float("nan")

    all_dist = next_pivot_dist(all_pos)
    baseline_rate = float((all_dist <= K).mean()) if len(all_pos) else float("nan")
    lift = precision / baseline_rate if baseline_rate and np.isfinite(baseline_rate) and baseline_rate > 0 else float("nan")

    trigger_pos_sorted = np.sort(trigger_pos)
    idx_before = np.searchsorted(trigger_pos_sorted, pivot_pos, side="right") - 1
    has_trigger_before = idx_before >= 0
    within_k = np.zeros(len(pivot_pos), dtype=bool)
    if has_trigger_before.any():
        cand = trigger_pos_sorted[idx_before[has_trigger_before]]
        within_k[has_trigger_before] = (pivot_pos[has_trigger_before] - cand) <= K
    recall = float(within_k.mean()) if len(pivot_pos) else float("nan")

    return {
        "n_triggers": int(len(trigger_pos)),
        "n_pivots": int(len(pivot_pos)),
        "precision": precision,
        "baseline_rate": baseline_rate,
        "lift": lift,
        "recall": recall,
        "median_lead_bars": median_lead,
        "n_true_positive": int(hits.sum()),
    }


def excess_move(trigger_pos: np.ndarray, pivot_pos: np.ndarray, close: np.ndarray, K: int) -> dict:
    """Among true positives, how far does price still move before the real pivot is hit?"""
    pivot_pos_sorted = np.sort(pivot_pos)
    idx = np.searchsorted(pivot_pos_sorted, trigger_pos, side="left")
    valid = idx < len(pivot_pos_sorted)
    dist = np.full(len(trigger_pos), np.inf)
    dist[valid] = pivot_pos_sorted[idx[valid]] - trigger_pos[valid]
    hit_mask = dist <= K
    if not hit_mask.any():
        return {"n": 0, "mean_pct": float("nan"), "median_pct": float("nan")}
    trig_hit = trigger_pos[hit_mask]
    pivot_hit = pivot_pos_sorted[idx[hit_mask]]
    pct_move = (close[pivot_hit] - close[trig_hit]) / close[trig_hit]
    return {"n": int(hit_mask.sum()), "mean_pct": float(pct_move.mean() * 100), "median_pct": float(np.median(pct_move) * 100)}


def run_side(frame: pd.DataFrame, window_mask: np.ndarray, pivots: pd.DataFrame, side: str) -> pd.DataFrame:
    close = frame["close"].to_numpy()
    all_pos = np.flatnonzero(window_mask)
    side_pivots = pivots.loc[pivots["pivot_type"] == side]
    pivot_pos = frame.index[frame["timestamp"].isin(side_pivots["timestamp"])].to_numpy()

    if side == "bottom":
        signals = {
            "fixed_both (%R<=20 AND SlowK<=20)": (frame["fast_k"] <= 20) & (frame["slow_k"] <= 20),
            "fixed_R_only (%R<=20)": frame["fast_k"] <= 20,
            "fixed_K_only (SlowK<=20)": frame["slow_k"] <= 20,
            "adaptive_both (p_fast<=.10 AND p_slow<=.10)": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10),
        }
    else:
        signals = {
            "fixed_both (%R>=80 AND SlowK>=80)": (frame["fast_k"] >= 80) & (frame["slow_k"] >= 80),
            "fixed_R_only (%R>=80)": frame["fast_k"] >= 80,
            "fixed_K_only (SlowK>=80)": frame["slow_k"] >= 80,
            "adaptive_both (p_fast>=.90 AND p_slow>=.90)": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90),
        }

    rows = []
    for sig_name, mask in signals.items():
        trigger_pos = np.flatnonzero(mask.to_numpy() & window_mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            move = excess_move(trigger_pos, pivot_pos, close, K)
            rows.append(
                {
                    "side": side,
                    "signal": sig_name,
                    "horizon": k_name,
                    **stats,
                    "excess_move_mean_pct": move["mean_pct"],
                    "excess_move_median_pct": move["median_pct"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_frame()
    frame = compute_indicators(raw).reset_index(drop=True)
    pivots = load_zigzag_pivots()

    ts = frame["timestamp"]
    window_mask = ((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))
    window_mask = window_mask.to_numpy()
    print(f"Study window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots ({(pivots['pivot_type']=='bottom').sum()} bottom / "
          f"{(pivots['pivot_type']=='top').sum()} top)")

    all_rows = pd.concat([run_side(frame, window_mask, pivots, "bottom"), run_side(frame, window_mask, pivots, "top")], ignore_index=True)

    pd.set_option("display.width", 160)
    for side in ("bottom", "top"):
        print(f"\n=== {side.upper()} evidence (does oversold/overbought precede a real {side}?) ===")
        sub = all_rows[all_rows["side"] == side]
        for horizon in K_HORIZONS:
            print(f"\n-- horizon {horizon} --")
            cols = ["signal", "n_triggers", "precision", "baseline_rate", "lift", "recall", "median_lead_bars", "excess_move_mean_pct"]
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_confluence_oscillator_bottom_top_evidence_20260814"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out_dir / "evidence_table.csv", index=False)
    print(f"\nWrote full table to {out_dir / 'evidence_table.csv'}")


if __name__ == "__main__":
    main()
