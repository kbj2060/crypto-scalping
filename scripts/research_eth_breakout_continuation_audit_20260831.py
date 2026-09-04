#!/usr/bin/env python3
"""Exhaustive audit of research_eth_breakout_continuation_raw_lift_check_20260831.py's numbers,
per user request ("수치에 어떤 문제가 있는지 전수조사해줘"). Checks, quantifies, and where possible
fixes each suspected issue separately so their individual contribution to the original 20x+ lift
is visible, not just a single "corrected" number with no paper trail.

Issues checked:
  1. Baseline tautology (already flagged before this script): "already-beyond-level" trigger bars
     vs unconditional random bars for a "stays beyond level" hit-rate. Confirmed structural, not
     re-litigated here -- this script focuses on the OTHER issues and reports the ATR-move metric
     (which never had this specific problem) as the primary corrected comparison instead.
  2. ATR self-inclusion: add_causal_columns()'s atr[i] includes bar i's OWN true range -- the
     exact bug pattern already documented+fixed for V자반등 v3->v4
     (build_eth_5m_liquidity_sweep_v_rebound_labels_20260829.py comment: "ATR was self-inclusive").
     The original precheck used atr[idx] (self-inclusive); this checks atr[idx-1] (pre-trigger)
     instead and reports how much the move_atr numbers shift.
  3. Un-clustered consecutive trigger bars: a single sustained breakout run gets counted as MANY
     separate "trigger" events (every bar the close stays beyond the level is its own row), which
     is a form of survivorship bias -- bars deep into an already-successful run are, by
     construction, exactly the ones that didn't fail early, inflating both the hit-rate and the
     move stat. Quantifies raw-bar-count vs distinct-run-count, and reruns the ATR-move comparison
     using ONLY the first bar of each run (cluster-anchored, matching this project's standard
     liquidity_sweep/taker/etc. anchoring convention) instead of every bar.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"

START = pd.Timestamp("2024-01-01", tz="UTC")
K_HORIZONS = {"K12_1h": 12, "K48_4h": 48, "K96_8h": 96}


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_breakout_audit_20260831", SWEEP_IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES, usecols=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df[df["timestamp"] >= START].reset_index(drop=True)


def run_lengths(idx: np.ndarray) -> np.ndarray:
    """Given sorted bar positions that satisfy the trigger condition, returns the length of each
    maximal consecutive-bar run (e.g. idx=[5,6,7,20,21] -> [3,2])."""
    if len(idx) == 0:
        return np.array([], dtype=int)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    return ends - starts + 1


def first_bar_of_each_run(idx: np.ndarray) -> np.ndarray:
    if len(idx) == 0:
        return idx
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    return idx[starts]


def move_atr_stats(idx: np.ndarray, is_up: bool, close: np.ndarray, atr_series: np.ndarray, K: int, n: int,
                    atr_lag: int) -> dict:
    """atr_lag=0 reproduces the ORIGINAL (self-inclusive, buggy) precheck; atr_lag=1 uses the
    pre-trigger-bar ATR (atr[idx-1]), matching this project's documented fix convention."""
    valid = idx[(idx + K < n) & (idx - atr_lag >= 0)]
    a = atr_series[valid - atr_lag]
    valid = valid[np.isfinite(a) & (a > 0)]
    a = atr_series[valid - atr_lag]
    if len(valid) == 0:
        return {"n": 0, "mean_move_atr": float("nan"), "median_move_atr": float("nan")}
    end_move = (close[valid + K] - close[valid]) if is_up else (close[valid] - close[valid + K])
    move_atr = end_move / a
    return {"n": int(len(valid)), "mean_move_atr": float(move_atr.mean()),
            "median_move_atr": float(np.median(move_atr))}


def main() -> None:
    impl = load_impl()
    kl = load_klines()
    causal = impl.add_causal_columns(kl[["timestamp", "open", "high", "low", "close"]].copy())

    close = kl["close"].to_numpy()
    level_high = causal["sweep_level_high"].to_numpy()
    level_low = causal["sweep_level_low"].to_numpy()
    atr = causal["atr"].to_numpy()
    n = len(kl)

    is_breakout_up = np.isfinite(level_high) & (close > level_high)
    is_breakout_down = np.isfinite(level_low) & (close < level_low)
    up_idx = np.flatnonzero(is_breakout_up)
    down_idx = np.flatnonzero(is_breakout_down)

    print("=" * 70)
    print("ISSUE 2 check: ATR self-inclusion -- how much does atr[i] vs atr[i-1] differ")
    print("=" * 70)
    for side, idx in (("up", up_idx), ("down", down_idx)):
        valid = idx[idx >= 1]
        atr_self = atr[valid]
        atr_prev = atr[valid - 1]
        ratio = atr_self / atr_prev
        print(f"  {side}: mean atr[i]/atr[i-1] = {ratio.mean():.4f} "
              f"(median {np.median(ratio):.4f}) -- {'INFLATED' if ratio.mean() > 1.02 else 'no material self-inclusion effect'} "
              f"at trigger bars specifically")

    print()
    print("=" * 70)
    print("ISSUE 3 check: consecutive-bar clustering (raw bar count vs distinct runs)")
    print("=" * 70)
    for side, idx in (("up", up_idx), ("down", down_idx)):
        runs = run_lengths(idx)
        first_bars = first_bar_of_each_run(idx)
        print(f"  {side}: raw triggers={len(idx)}, distinct runs={len(runs)} "
              f"({len(idx)/len(runs):.1f}x avg run length, max run={runs.max()} bars={runs.max()*5}min), "
              f"first-bar-only count={len(first_bars)}")

    print()
    print("=" * 70)
    print("Corrected ATR-move comparison: cluster-anchored (first bar of run only) "
          "AND pre-trigger ATR (atr[idx-1]), vs the ORIGINAL script's every-bar + atr[idx]")
    print("=" * 70)
    for side, idx, is_up in (("up", up_idx, True), ("down", down_idx, False)):
        first_bars = first_bar_of_each_run(idx)
        print(f"\n  --- {side} ---")
        for k_name, K in K_HORIZONS.items():
            orig = move_atr_stats(idx, is_up, close, atr, K, n, atr_lag=0)
            fixed_atr_only = move_atr_stats(idx, is_up, close, atr, K, n, atr_lag=1)
            fixed_both = move_atr_stats(first_bars, is_up, close, atr, K, n, atr_lag=1)
            print(f"    {k_name}: ORIGINAL(every-bar,atr[i])       mean={orig['mean_move_atr']:+.3f} n={orig['n']}")
            print(f"    {k_name}: +ATR fix only (every-bar,atr[i-1]) mean={fixed_atr_only['mean_move_atr']:+.3f} n={fixed_atr_only['n']}")
            print(f"    {k_name}: +cluster-anchor+ATR fix (first-bar,atr[i-1]) mean={fixed_both['mean_move_atr']:+.3f} "
                  f"median={fixed_both['median_move_atr']:+.3f} n={fixed_both['n']}")


if __name__ == "__main__":
    main()
