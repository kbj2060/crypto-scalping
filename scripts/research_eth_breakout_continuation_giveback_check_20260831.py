#!/usr/bin/env python3
"""돌파 지속 (breakout continuation) precheck, take 2 -- reruns using the ACTUAL v7b giveback
formula (fast_move_atr_mult>=1.5 within 30min AND giveback_ratio<=0.20 within 60min), not the
crude "never dips below the level for the whole window" check from the first precheck (which the
audit showed was a much stricter, different test than what actually worked for V자반등).

Cannot reuse research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py::realized_outcome()
verbatim -- that function is hardwired for REVERSAL (extreme -> OPPOSITE-direction move, e.g.
is_down=True anchors on the wick LOW and measures the UPWARD rebound away from it). Continuation
needs the mirror: anchor on the breakout bar's own CLOSE ("entry", matching this trigger's own
close-confirmed definition, not a wick) and measure movement in the SAME direction as the
breakout. The formula/thresholds/window lengths (1.5x, 0.20, 6/12 bars) are reused unchanged --
only the direction-mapping and anchor point differ, because the underlying question does.

Also carries forward BOTH audit fixes from research_eth_breakout_continuation_audit_20260831.py:
pre-trigger ATR (atr[idx-1], not self-inclusive atr[idx]) and cluster-anchoring (first bar of each
consecutive-breakout run only, not every bar). Baseline this time is NON-tautological: the exact
same giveback outcome formula applied to entry=that bar's own close for ALL bars (not conditioned
on already being beyond a level) -- matching how research_eth_sweep_v_rebound_random_bar_
baseline_20260829.py compared V_REBOUND's own trigger-rate against a random-bar rate using the
SAME label definition for both populations.
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
FAST_BARS = 6    # 30min, verbatim from v7b
FULL_BARS = 12   # 60min, verbatim from v7b
ATR_MULT = 1.5   # verbatim from v7b
T_SUSTAIN = 0.20  # verbatim from v7b


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_breakout_giveback_20260831", SWEEP_IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES, usecols=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df[df["timestamp"] >= START].reset_index(drop=True)


def first_bar_of_each_run(idx: np.ndarray) -> np.ndarray:
    if len(idx) == 0:
        return idx
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    return idx[starts]


def continuation_outcome(close: np.ndarray, high: np.ndarray, low: np.ndarray, atr: np.ndarray,
                          idx: int, is_up: bool, n: int) -> dict | None:
    """Mirror of v7b's realized_outcome(), but anchored on entry=close[idx] (not a wick extreme)
    and measuring movement in the SAME direction as is_up (not the opposite/reversal direction)."""
    if idx - 1 < 0 or idx + FULL_BARS >= n:
        return None
    pre_atr = atr[idx - 1]
    if not np.isfinite(pre_atr) or pre_atr <= 0:
        return None
    entry = close[idx]
    fast_close = close[idx + 1: idx + FAST_BARS + 1]
    full_high = high[idx + 1: idx + FULL_BARS + 1]
    full_low = low[idx + 1: idx + FULL_BARS + 1]
    full_close_end = close[idx + FULL_BARS]

    if is_up:
        fast_move = fast_close.max() - entry
        peak = full_high.max()
        denom = peak - entry
        giveback = (peak - full_close_end) / denom if denom > 1e-12 else np.nan
    else:
        fast_move = entry - fast_close.min()
        peak = full_low.min()
        denom = entry - peak
        giveback = (full_close_end - peak) / denom if denom > 1e-12 else np.nan

    fast_mult = fast_move / pre_atr
    if fast_mult >= ATR_MULT and np.isfinite(giveback) and giveback <= T_SUSTAIN:
        label = "지속"
    elif fast_mult < 1.0:
        label = "정체"
    else:
        label = "애매"
    return {"fast_mult": float(fast_mult), "giveback": float(giveback) if np.isfinite(giveback) else None,
            "label": label}


def main() -> None:
    impl = load_impl()
    kl = load_klines()
    causal = impl.add_causal_columns(kl[["timestamp", "open", "high", "low", "close"]].copy())

    close = kl["close"].to_numpy()
    high = kl["high"].to_numpy()
    low = kl["low"].to_numpy()
    level_high = causal["sweep_level_high"].to_numpy()
    level_low = causal["sweep_level_low"].to_numpy()
    atr = causal["atr"].to_numpy()
    n = len(kl)

    is_breakout_up = np.isfinite(level_high) & (close > level_high)
    is_breakout_down = np.isfinite(level_low) & (close < level_low)
    up_idx = first_bar_of_each_run(np.flatnonzero(is_breakout_up))
    down_idx = first_bar_of_each_run(np.flatnonzero(is_breakout_down))
    all_idx = np.arange(n)

    print(f"bars={n}, breakout runs: up={len(up_idx)} down={len(down_idx)} (cluster-anchored)\n")

    def score(idx_arr: np.ndarray, is_up: bool) -> dict:
        counts = {"지속": 0, "정체": 0, "애매": 0}
        n_scored = 0
        for i in idx_arr:
            o = continuation_outcome(close, high, low, atr, int(i), is_up, n)
            if o is None:
                continue
            counts[o["label"]] += 1
            n_scored += 1
        return {"n_scored": n_scored, **counts}

    print(f"{'':30s} {'n':>7s} {'지속':>8s} {'정체':>8s} {'애매':>8s} {'지속율':>8s}")
    results = {}
    for side, idx_arr, is_up in (("up(breakout,지속=계속상승)", up_idx, True),
                                  ("down(breakout,지속=계속하락)", down_idx, False)):
        r = score(idx_arr, is_up)
        rate = r["지속"] / r["n_scored"] if r["n_scored"] else float("nan")
        results[side] = {"trigger": r, "rate": rate}
        print(f"{side:30s} {r['n_scored']:7d} {r['지속']:8d} {r['정체']:8d} {r['애매']:8d} {rate:7.1%}")

    print("\n--- baseline: 동일 공식, ALL bars (트리거 조건 없이 entry=그 봉 자신의 종가) ---")
    for side, is_up in (("up방향(all bars)", True), ("down방향(all bars)", False)):
        r = score(all_idx, is_up)
        rate = r["지속"] / r["n_scored"] if r["n_scored"] else float("nan")
        results[side] = {"trigger": r, "rate": rate}
        print(f"{side:30s} {r['n_scored']:7d} {r['지속']:8d} {r['정체']:8d} {r['애매']:8d} {rate:7.1%}")

    up_lift = results["up(breakout,지속=계속상승)"]["rate"] / results["up방향(all bars)"]["rate"]
    down_lift = results["down(breakout,지속=계속하락)"]["rate"] / results["down방향(all bars)"]["rate"]
    print(f"\nlift: up={up_lift:.2f}x, down={down_lift:.2f}x")


if __name__ == "__main__":
    main()
