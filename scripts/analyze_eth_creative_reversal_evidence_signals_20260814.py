#!/usr/bin/env python3
"""More candidate "is this actually a bottom/top" evidence signals for ETH 5m, built from
recent literature + orthogonal information sources (order flow, VWAP, volume) rather than more
price-position oscillators. NOT a trading algorithm -- same retrospective evidence-study
methodology as analyze_eth_confluence_oscillator_bottom_top_evidence_20260814.py (reused
unmodified: event_study, excess_move, load_zigzag_pivots, K_HORIZONS), ground truth is ETH's
real zigzag swing pivots, not a hand-rolled detector.

Candidates and their literature grounding:
  1. momentum divergence   -- price makes a lower low while the oscillator's momentum improves
                               (classic RSI/stochastic bullish/bearish divergence; literature:
                               VT Markets/QuantifiedStrategies/tradingsim summaries -- divergence
                               reportedly precedes reversals by a few bars, weaker in strong trends).
  2. CVD divergence         -- price falls while cumulative taker buy/sell delta rises (hidden
                               accumulation via passive absorption of aggressive selling); this
                               repo's OHLCV already carries taker_buy_base, so CVD is computable
                               without new data. Grounding: Bookmap/MarketTrace/BackQuant CVD-
                               divergence guides, and the order-flow-toxicity literature
                               (VPIN-price-jump link, 2026 Bitcoin microstructure study).
  3. taker imbalance climax -- an extreme one-bar burst of aggressive net selling/buying (order-
                               flow analogue of a Wyckoff selling/buying climax), independent of
                               price-position oscillators entirely.
  4. VWAP deviation extreme -- price stretched far from a rolling VWAP; VWAP mean-reversion is a
                               standard intraday microstructure effect (2025 VWAP-band guides:
                               "2nd/3rd deviation band -> reversion to VWAP").
  5. volume + wick climax   -- Wyckoff selling/buying climax proxy: high-volume bar with a long
                               opposite-direction wick (institutional absorption signature).
  6. orthogonal combo       -- the original oscillator confluence (adaptive_both, already tested)
                               AND an orthogonal order-flow signal (taker imbalance climax) fired
                               together. Direct test of this sub-project's original "information-
                               family diversity, not indicator count" hypothesis: combining %R
                               with Slow %K (same info source) barely beat either alone; does
                               combining an oscillator with a genuinely different info source
                               (order flow) do better?
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

DATA_PATH = ROOT / "data" / "eth_5m_1year.csv"


def load_frame_with_orderflow() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"],
        parse_dates=["timestamp"],
    )
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df


def add_creative_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    close, open_, high, low, volume = frame["close"], frame["open"], frame["high"], frame["low"], frame["volume"]
    taker_buy = frame["taker_buy_base"]
    eps = 1e-12

    delta = 2.0 * taker_buy - volume  # net aggressive buy volume this bar
    delta_mean = delta.rolling(288, min_periods=288).mean()
    delta_std = delta.rolling(288, min_periods=288).std()
    delta_z = (delta - delta_mean) / delta_std.replace(0.0, np.nan)

    cvd_roll = delta.rolling(288, min_periods=288).sum()  # stationary 1-day rolling CVD proxy
    cvd_roll_roc_48 = cvd_roll - cvd_roll.shift(48)

    vwap_roll = (close * volume).rolling(288, min_periods=288).sum() / volume.rolling(288, min_periods=288).sum()
    vwap_dev = (close - vwap_roll) / vwap_roll
    vwap_dev_mean = vwap_dev.rolling(864, min_periods=864).mean()
    vwap_dev_std = vwap_dev.rolling(864, min_periods=864).std()
    vwap_dev_z = (vwap_dev - vwap_dev_mean) / vwap_dev_std.replace(0.0, np.nan)

    vol_mean = volume.rolling(288, min_periods=288).mean()
    vol_std = volume.rolling(288, min_periods=288).std()
    vol_z = (volume - vol_mean) / vol_std.replace(0.0, np.nan)
    rng = (high - low).replace(0.0, np.nan)
    lower_wick_ratio = (np.minimum(open_, close) - low) / (rng + eps)
    upper_wick_ratio = (high - np.maximum(open_, close)) / (rng + eps)

    price_roc_48 = close / close.shift(48) - 1.0
    fast_k_roc_48 = frame["fast_k"] - frame["fast_k"].shift(48)

    out = frame.copy()
    out["delta_z"] = delta_z
    out["cvd_roll_roc_48"] = cvd_roll_roc_48
    out["vwap_dev_z"] = vwap_dev_z
    out["vol_z"] = vol_z
    out["lower_wick_ratio"] = lower_wick_ratio
    out["upper_wick_ratio"] = upper_wick_ratio
    out["price_roc_48"] = price_roc_48
    out["fast_k_roc_48"] = fast_k_roc_48
    return out


def build_signals(frame: pd.DataFrame, side: str) -> dict:
    if side == "bottom":
        return {
            "momentum_divergence (price 4h-LL, %R momentum UP)": (frame["price_roc_48"] <= -0.01) & (frame["fast_k_roc_48"] >= 5),
            "cvd_divergence (price 4h-LL, CVD rising)": (frame["price_roc_48"] <= -0.01) & (frame["cvd_roll_roc_48"] >= 0),
            "taker_sell_climax (delta_z<=-2)": frame["delta_z"] <= -2.0,
            "vwap_extreme_low (dev_z<=-2)": frame["vwap_dev_z"] <= -2.0,
            "volume_wick_climax_low (vol_z>=2, lower_wick>=.5)": (frame["vol_z"] >= 2.0) & (frame["lower_wick_ratio"] >= 0.5),
            "orthogonal_combo (adaptive_OS AND taker_sell_climax)": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["delta_z"] <= -2.0),
        }
    return {
        "momentum_divergence (price 4h-HH, %R momentum DOWN)": (frame["price_roc_48"] >= 0.01) & (frame["fast_k_roc_48"] <= -5),
        "cvd_divergence (price 4h-HH, CVD falling)": (frame["price_roc_48"] >= 0.01) & (frame["cvd_roll_roc_48"] <= 0),
        "taker_buy_climax (delta_z>=2)": frame["delta_z"] >= 2.0,
        "vwap_extreme_high (dev_z>=2)": frame["vwap_dev_z"] >= 2.0,
        "volume_wick_climax_high (vol_z>=2, upper_wick>=.5)": (frame["vol_z"] >= 2.0) & (frame["upper_wick_ratio"] >= 0.5),
        "orthogonal_combo (adaptive_OB AND taker_buy_climax)": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & (frame["delta_z"] >= 2.0),
    }


def run_side(frame: pd.DataFrame, window_mask: np.ndarray, pivots: pd.DataFrame, side: str) -> pd.DataFrame:
    close = frame["close"].to_numpy()
    all_pos = np.flatnonzero(window_mask)
    side_pivots = pivots.loc[pivots["pivot_type"] == side]
    pivot_pos = frame.index[frame["timestamp"].isin(side_pivots["timestamp"])].to_numpy()

    rows = []
    for sig_name, mask in build_signals(frame, side).items():
        trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            move = excess_move(trigger_pos, pivot_pos, close, K)
            rows.append({"side": side, "signal": sig_name, "horizon": k_name, **stats,
                         "excess_move_mean_pct": move["mean_pct"], "excess_move_median_pct": move["median_pct"]})
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_frame_with_orderflow()
    frame = compute_indicators(raw).reset_index(drop=True)
    frame = add_creative_indicators(frame)
    pivots = load_zigzag_pivots()

    ts = frame["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    print(f"Study window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")

    all_rows = pd.concat([run_side(frame, window_mask, pivots, "bottom"), run_side(frame, window_mask, pivots, "top")], ignore_index=True)

    pd.set_option("display.width", 170)
    for side in ("bottom", "top"):
        print(f"\n=== {side.upper()} evidence ===")
        sub = all_rows[all_rows["side"] == side]
        for horizon in K_HORIZONS:
            print(f"\n-- horizon {horizon} --")
            cols = ["signal", "n_triggers", "precision", "baseline_rate", "lift", "recall", "median_lead_bars", "excess_move_mean_pct"]
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_creative_reversal_evidence_signals_20260814"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out_dir / "evidence_table.csv", index=False)
    print(f"\nWrote full table to {out_dir / 'evidence_table.csv'}")


if __name__ == "__main__":
    main()
