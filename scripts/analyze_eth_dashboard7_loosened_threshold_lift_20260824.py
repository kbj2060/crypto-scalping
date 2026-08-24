#!/usr/bin/env python3
"""One-notch-looser lift comparison for the 7 live dashboard signals (2026-08-24).

User explicitly directed this re-test after being warned it repeats a multiple-comparison risk
already flagged for this exact research line (docs/eth_snapshot_dashboard_12signal_decision_
support_assessment_20260823.md names these thresholds "registry-retest-off-limits absent a fresh
pre-registered OOS window, not available until 09-30"; the 2026-08-24 frequency-only sensitivity
check deliberately did NOT recompute lift at loosened thresholds for exactly this reason). Executed
anyway per explicit user direction. Results here are EXPLORATORY ONLY -- not a basis for
redeploying loosened thresholds to the live dashboard without further discussion, and not a
substitute for the proper re-registration this line's own prior assessment calls for.

Discipline preserved despite the override: every loosened threshold below is chosen ONCE, reusing
the exact convention already set by the 2026-08-24 frequency-only sensitivity check (orthogonal_
combo p<=0.10->0.15, |z|>=2.0->1.5) -- not swept or hand-tuned looking for a better cell. Two knob
types, since the 7 signals split into two structural families:
  (a) continuous z-score/percentile threshold: shift by the same increment as the precedent
      (percentile band +0.05, |z| cutoff -0.5) -- applies to orthogonal_combo, volume_wick_climax,
      short_term_return_z, taker_delta_z_climax.
  (b) swing-lookback window: 48->36 bars (25% shorter, roughly proportional to (a)'s shift size)
      -- applies to liquidity_sweep, smt_divergence, fib_extension_exhaustion, which have no
      z-score knob at all (they are structural/geometric pattern definitions).
Same VAL+OOS pooled window and event_study/excess_move harness as every sibling evidence-signal
script, for direct comparability with the lift numbers already shown on the live dashboard.
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

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    OOS_END,
    event_study,
    excess_move,
    load_zigzag_pivots,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
    compute_indicators,
)

ETH_PATH = ROOT / "data" / "eth_5m_1year.csv"
BTC_PATH = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp" / "eth_dashboard7_loosened_threshold_lift_20260824"

CURRENT_SWING, LOOSE_SWING = 48, 36
ZSCORE_WINDOW = 288
EPS = 1e-12


def load_frame() -> pd.DataFrame:
    """Same columns as scripts/live_evidence_signal_dashboard_20260823.py::fetch_klines uses
    (taker_buy_base, not taker_buy_quote) so delta_z here matches what the live dashboard
    actually computes, not a different sibling script's quote-volume variant."""
    cols = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]
    df = pd.read_csv(ETH_PATH, usecols=cols, parse_dates=["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def add_creative_cols(f: pd.DataFrame) -> pd.DataFrame:
    """delta_z / vol_z / wick ratios / ret3_z, copied verbatim from
    live_evidence_signal_dashboard_20260823.py::compute_signals (the actual live formulas)."""
    close, open_, high, low, volume = f["close"], f["open"], f["high"], f["low"], f["volume"]
    taker_buy = f["taker_buy_base"]
    delta = 2.0 * taker_buy - volume
    f["delta_z"] = (delta - delta.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        delta.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)
    f["vol_z"] = (volume - volume.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        volume.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)
    rng_body = (high - low).replace(0.0, np.nan)
    f["lower_wick_ratio"] = (np.minimum(open_, close) - low) / (rng_body + EPS)
    f["upper_wick_ratio"] = (high - np.maximum(open_, close)) / (rng_body + EPS)
    ret3 = close / close.shift(3) - 1.0
    f["ret3_z"] = (ret3 - ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)
    return f


def add_sweep_w(f: pd.DataFrame, window: int, suffix: str) -> pd.DataFrame:
    swing_low = f["low"].rolling(window, min_periods=window).min().shift(1)
    swing_high = f["high"].rolling(window, min_periods=window).max().shift(1)
    f[f"swing_low_prior{suffix}"] = swing_low
    f[f"swing_high_prior{suffix}"] = swing_high
    f[f"sweep_low{suffix}"] = (f["low"] < swing_low) & (f["close"] > swing_low)
    f[f"sweep_high{suffix}"] = (f["high"] > swing_high) & (f["close"] < swing_high)
    return f


def add_smt_w(f: pd.DataFrame, window: int, suffix: str) -> pd.DataFrame:
    btc = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    btc = (btc.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
           .rename(columns={"high": "btc_high", "low": "btc_low"}))
    if "btc_high" not in f.columns:
        merged = f.merge(btc, on="timestamp", how="left")
        f["btc_high"], f["btc_low"] = merged["btc_high"].to_numpy(), merged["btc_low"].to_numpy()
    btc_swing_low = f["btc_low"].rolling(window, min_periods=window).min().shift(1)
    btc_swing_high = f["btc_high"].rolling(window, min_periods=window).max().shift(1)
    btc_holds_low = (f["btc_low"] > btc_swing_low).fillna(False)
    btc_holds_high = (f["btc_high"] < btc_swing_high).fillna(False)
    f[f"smt_bottom{suffix}"] = (f["low"] < f[f"swing_low_prior{suffix}"]) & btc_holds_low
    f[f"smt_top{suffix}"] = (f["high"] > f[f"swing_high_prior{suffix}"]) & btc_holds_high
    return f


def add_leg_direction_w(f: pd.DataFrame, window: int, suffix: str) -> pd.DataFrame:
    low, high = f["low"].to_numpy(), f["high"].to_numpy()
    n = len(f)
    low_pos = np.full(n, -1, dtype=np.int64)
    high_pos = np.full(n, -1, dtype=np.int64)
    if n > window:
        lo_w = np.lib.stride_tricks.sliding_window_view(low, window)
        hi_w = np.lib.stride_tricks.sliding_window_view(high, window)
        idx = np.arange(window, n)
        j = idx - window
        low_pos[idx] = j + lo_w[j].argmin(axis=1)
        high_pos[idx] = j + hi_w[j].argmax(axis=1)
    leg_up = low_pos < high_pos
    leg_down = high_pos < low_pos
    lo, hi = f[f"swing_low_prior{suffix}"], f[f"swing_high_prior{suffix}"]
    rng = (hi - lo).replace(0.0, np.nan)
    f[f"fib_ext_top{suffix}"] = pd.Series(leg_up, index=f.index) & \
        f["high"].between(hi + 0.272 * rng, hi + 0.618 * rng)
    f[f"fib_ext_bottom{suffix}"] = pd.Series(leg_down, index=f.index) & \
        f["low"].between(lo - 0.618 * rng, lo - 0.272 * rng)
    return f


def build_variant(f: pd.DataFrame, window: int, suffix: str) -> pd.DataFrame:
    f = add_sweep_w(f, window, suffix)
    f = add_smt_w(f, window, suffix)
    f = add_leg_direction_w(f, window, suffix)
    return f


def components(f: pd.DataFrame, side: str, variant: str) -> dict[str, pd.Series]:
    suffix = "" if variant == "current" else "_loose"
    p_fast_th, p_slow_th = (0.10, 0.10) if variant == "current" else (0.15, 0.15)
    delta_th = -2.0 if variant == "current" else -1.5
    vol_th, wick_th = (2.0, 0.5) if variant == "current" else (1.5, 0.4)
    ret_th = -2.5 if variant == "current" else -2.0
    sign = 1 if side == "bottom" else -1

    p_fast, p_slow, delta_z = f["p_fast"], f["p_slow"], f["delta_z"]
    vol_z, ret3_z = f["vol_z"], f["ret3_z"]
    wick = f["lower_wick_ratio"] if side == "bottom" else f["upper_wick_ratio"]

    if side == "bottom":
        orth = (p_fast <= p_fast_th) & (p_slow <= p_slow_th) & (delta_z <= delta_th)
        taker = delta_z <= delta_th
        retz = ret3_z <= ret_th
    else:
        orth = (p_fast >= 1 - p_fast_th) & (p_slow >= 1 - p_slow_th) & (delta_z >= -delta_th)
        taker = delta_z >= -delta_th
        retz = ret3_z >= -ret_th
    volwick = (vol_z >= vol_th) & (wick >= wick_th)

    struct_key = f"sweep_{'low' if side == 'bottom' else 'high'}{suffix}"
    smt_key = f"smt_{side}{suffix}"
    fib_key = f"fib_ext_{side}{suffix}"
    return {
        "orthogonal_combo": orth, "liquidity_sweep": f[struct_key],
        "volume_wick_climax": volwick, "short_term_return_z": retz,
        "taker_delta_z_climax": taker, "smt_divergence": f[smt_key],
        "fib_extension_exhaustion": f[fib_key],
    }


def run_side(f: pd.DataFrame, mask: np.ndarray, pivots: pd.DataFrame, side: str, variant: str) -> pd.DataFrame:
    close = f["close"].to_numpy()
    all_pos = np.flatnonzero(mask)
    pivot_pos = f.index[f["timestamp"].isin(pivots.loc[pivots["pivot_type"] == side, "timestamp"])].to_numpy()
    rows = []
    for name, sig in components(f, side, variant).items():
        trigger_pos = np.flatnonzero(sig.fillna(False).to_numpy() & mask)
        stats = event_study(trigger_pos, pivot_pos, all_pos, K_HORIZONS["K12_1h"])
        move = excess_move(trigger_pos, pivot_pos, close, K_HORIZONS["K12_1h"])
        rows.append({"variant": variant, "side": side, "signal": name, **stats,
                     "excess_move_mean_pct": move["mean_pct"]})
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_frame()
    f = compute_indicators(raw).reset_index(drop=True)
    f = add_creative_cols(f)
    f = build_variant(f, CURRENT_SWING, "")
    f = build_variant(f, LOOSE_SWING, "_loose")
    pivots = load_zigzag_pivots()

    ts = f["timestamp"]
    mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"pooled bars={int(mask.sum())}, pivots={len(pivots)}, K=1h\n")
    print(f"Loosened knobs: p_fast/p_slow 0.10->0.15, |delta_z|/|ret3_z(2.5->2.0)| cutoff -0.5, "
          f"vol_z 2.0->1.5, wick_ratio 0.5->0.4, swing-lookback {CURRENT_SWING}->{LOOSE_SWING} bars\n")

    res = pd.concat(
        [run_side(f, mask, pivots, side, variant) for side in ("bottom", "top") for variant in ("current", "loose")],
        ignore_index=True,
    )

    pd.set_option("display.width", 200)
    piv = res.pivot_table(index=["side", "signal"], columns="variant",
                          values=["lift", "n_triggers", "precision"], aggfunc="first")
    piv = piv.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0)
    print(piv.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "loosened_threshold_lift_table.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'loosened_threshold_lift_table.csv'}")


if __name__ == "__main__":
    main()
