"""Stage 0 — does regime detection want COARSER BARS?  (2026-08-08)

Contract: docs/experiments/btc_regime_bar_timeframe_scale_20260808.json (pre-registered).

The user's hypothesis is that 30m/1h bars fit regimes better than 5m.  Bar size alone is not the
variable though: the zigzag oracle is defined on the PRICE PATH by a threshold theta, so changing
bar size trades RESOLUTION (a detector cannot turn faster than one bar) against FEATURE NOISE
(coarser bars average out 5m microstructure).  Bar scale therefore only pays when it is PAIRED
with an oracle scale, and this script measures that pairing directly with NO learning at all --
the causal zigzag is a definitional rule with no fitted parameter, so nothing is spent here.

The decisive arithmetic Stage 0 makes explicit: at theta=0.5% the oracle's median wave is ~16 bars
at 5m (80 min), i.e. 1.3 bars at 1h -- unresolvable.  At theta=4% it is ~692 bars at 5m (~2.4 d),
i.e. ~58 bars at 1h -- a normal sequence length.  `bars_per_oracle_wave` is the headline column;
cells below 8 are unresolvable by construction whatever their agreement score says.

Every bar size is scored against the SAME reference: the oracle computed on the 5m close, which is
the finest available resolution of the true price path.  Scoring is reported twice --
  at 5m resolution   the live-relevant number (decisions happen on 5m bars, so a 1h detector is
                     correctly penalised on the 11 intermediate bars where it cannot update)
  at native resolution  fair to the detector itself
-- because reporting only one of them would rig the comparison in whichever direction was picked.

Causality: 5m -> coarse aggregation is right-closed / right-labelled, so the bar stamped T closes
at T and uses only bars <= T; the coarse state is then forward-filled onto the 5m grid (state at
5m bar t = most recent coarse state at or before t).  No future bar is read anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_btc_regime_classifier_lag_20260808 import detection_lag  # noqa: E402
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from refine_btc_regime_classifier_theta005_20260808 import PANEL_PATH  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/btc_regime_bar_timeframe_20260808"
BAR_SIZES = {"5m": 1, "15m": 3, "30m": 6, "1h": 12}      # value = number of 5m bars
THETAS = [0.005, 0.010, 0.020, 0.030, 0.040]
MIN_BARS_PER_WAVE = 8.0                                   # resolvability floor from the contract


def resample_close(ts: pd.Series, close: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-closed / right-labelled aggregation to a coarser grid.

    Returns (coarse_close, map_5m_to_coarse) where map[t] is the index of the most recent coarse
    bar at or before 5m bar t -- i.e. what a live reader would have had at t.
    """
    if step == 1:
        return close.copy(), np.arange(len(close))
    n = len(close)
    # a coarse bar ends at every 5m index where (i+1) % step == 0; the bar stamped there closes there
    ends = np.arange(step - 1, n, step)
    coarse = close[ends]
    # 5m bar t sees the last coarse bar whose end index is <= t
    idx = np.searchsorted(ends, np.arange(n), side="right") - 1
    return coarse, idx


def summarize_runs(state_5m: np.ndarray, idx: np.ndarray) -> tuple[float, int]:
    runs = [e - s + 1 for s, e, _ in contiguous_runs(state_5m[idx])]
    return (float(np.median(runs)) if runs else float("nan")), max(len(runs) - 1, 0)


def agreement(det: np.ndarray, oracle: np.ndarray, idx: np.ndarray) -> float | None:
    m = idx[(det[idx] != 0) & (oracle[idx] != 0)]
    return None if len(m) < 50 else round(float(np.mean(det[m] == oracle[m])) * 100, 1)


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts, close = panel["timestamp"], panel["close"].to_numpy(dtype=np.float64)
    n = len(close)
    windows = {
        "full": np.arange(n),
        "val_2025Q4": np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()),
        "oos_2026Q1": np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()),
    }
    print(json.dumps({"bars_5m": n, "span": [str(ts.iloc[0]), str(ts.iloc[-1])]}), flush=True)

    coarse = {name: resample_close(ts, close, step) for name, step in BAR_SIZES.items()}
    for name, step in BAR_SIZES.items():
        print(json.dumps({"grid": name, "bars": int(len(coarse[name][0]))}), flush=True)

    cells: dict[str, dict] = {}
    for th in THETAS:
        o_dir, pivots = zigzag_oracle(close, threshold=th)
        wave_lens = np.diff(np.asarray(pivots + [n - 1])) if len(pivots) >= 2 else np.array([np.nan])
        med_wave_5m = float(np.median(wave_lens))
        base_key = None
        for name, step in BAR_SIZES.items():
            c_close, idx_map = coarse[name]
            c_state = causal_zigzag(c_close, threshold=th)          # native-grid detector
            det_5m = c_state[np.clip(idx_map, 0, len(c_state) - 1)]  # forward-filled to 5m
            det_5m[idx_map < 0] = 0

            bars_per_wave = round(med_wave_5m / step, 1)
            entry: dict = {
                "bars_per_oracle_wave": bars_per_wave,
                "resolvable": bool(bars_per_wave >= MIN_BARS_PER_WAVE),
                "oracle_median_wave_min": round(med_wave_5m * 5, 1),
                "n_oracle_waves": int(len(wave_lens)),
            }
            for wname, widx in windows.items():
                med_run_5m, flips = summarize_runs(det_5m, widx)
                lag = detection_lag(det_5m, o_dir, pivots, int(widx[0]), int(widx[-1]))
                entry[wname] = {
                    "agree_at_5m_res": agreement(det_5m, o_dir, widx),
                    "coverage_pct": round(float((det_5m[widx] != 0).mean()) * 100, 1),
                    "median_run_min": None if np.isnan(med_run_5m) else round(med_run_5m * 5, 1),
                    "flips": flips,
                    "detection_lag_min": None if lag["median_bars"] is None else round(lag["median_bars"] * 5, 1),
                }
            # native-resolution scoring: oracle sampled at the coarse bar ends
            ends = np.arange(BAR_SIZES[name] - 1, n, BAR_SIZES[name])
            o_native = o_dir[ends]
            for wname, widx in windows.items():
                lo, hi = int(widx[0]), int(widx[-1])
                sel = np.flatnonzero((ends >= lo) & (ends <= hi))
                entry[wname]["agree_at_native_res"] = agreement(c_state, o_native, sel)
            key = f"theta{th * 100:g}|{name}"
            cells[key] = entry
            if name == "5m":
                base_key = key
            a5 = entry["full"]["agree_at_5m_res"]
            base = cells[base_key]["full"]["agree_at_5m_res"] if base_key else None
            delta = None if (a5 is None or base is None) else round(a5 - base, 1)
            print(f"  {key:16} bars/wave {bars_per_wave:7}  agree5m {a5}  (vs 5m {delta:+})  "
                  f"cov {entry['full']['coverage_pct']}  run {entry['full']['median_run_min']}min  "
                  f"lag {entry['full']['detection_lag_min']}min", flush=True)

    # ---- verdict per the pre-registered rule
    wins = []
    for th in THETAS:
        base = cells[f"theta{th * 100:g}|5m"]["full"]["agree_at_5m_res"]
        for name in BAR_SIZES:
            if name == "5m":
                continue
            c = cells[f"theta{th * 100:g}|{name}"]
            a = c["full"]["agree_at_5m_res"]
            if c["resolvable"] and base is not None and a is not None and a >= base:
                wins.append({"cell": f"theta{th * 100:g}|{name}", "agree": a, "vs_5m": round(a - base, 1),
                             "bars_per_wave": c["bars_per_oracle_wave"]})
    verdict = {
        "rule": "proceed only if a resolvable coarse cell (bars_per_oracle_wave >= 8) matches or "
                "beats the 5m detector at the SAME theta, scored at 5m resolution",
        "coarse_cells_matching_or_beating_5m": wins,
        "proceed_to_modelling": bool(wins),
    }
    out = {"contract": "docs/experiments/btc_regime_bar_timeframe_scale_20260808.json",
           "kind": "no-learning definitional measurement; nothing spent",
           "min_bars_per_wave": MIN_BARS_PER_WAVE, "cells": cells, "verdict": verdict}
    (OUT_DIR / "stage0.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"wrote {OUT_DIR / 'stage0.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
