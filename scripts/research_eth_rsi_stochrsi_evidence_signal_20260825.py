#!/usr/bin/env python3
"""Evidence study (NOT a trading algorithm, NOT a re-derivation of the closed oscillator-
confluence axis -- see memory eth_oscillator_confluence_closed_20260814 /
eth_classical_technical_indicator_literature_check_20260817): does literal Wilder RSI(14) /
StochRSI(14,14,3,3) -- as opposed to the already-tested Williams %R / Slow Stochastic %K family
-- carry any real "near a pivot" evidence under this repo's established retrospective lift
methodology? Same ground truth (zigzag_action swing pivots), same event_study/excess_move code,
reused unmodified from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814.py, for
direct comparability with the existing 22-signal master ranking
(eth_reversal_evidence_signal_scorecard_20260814 memory).

Motivating question (user, 2026-08-25): "코인 가격이 안내려가면서 StochRSI가 80에서 20으로
줄어드는건 어떤 의미지?" -- operationalized below as `stochrsi_fade_from_extreme` (top side is
the literal question; bottom side is its structural mirror).

Prior art already closed this family's most literature-cited framing (classic RSI/stochastic
DIVERGENCE: price makes a NEW 4h high/low while momentum disagrees) at 1.24x bottom / 0.88x top
(WORSE than random), confirmed again out-of-window at 0.77x top -- see momentum_divergence in
eth_creative_reversal_evidence_signals_20260814.md / eth_evidence_signal_ranking_stability_
mar_jul_2026_20260814.md. This script does NOT re-run that; `rsi_momentum_divergence` below is
only a cheap completeness check (swap %R for literal RSI in the identical formula) since the
infrastructure is already loaded. The two genuinely untested framings are the plain level check
and the fade-from-extreme pattern.

Caveat carried over from the parent scorecard: this adds 10 more signal/side cells to an already
22-signal, uncorrected-for-multiple-comparisons search (falsification_audit not run). Treat as
exploratory, same as the rest of that scorecard.
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
)

DATA_PATH = ROOT / "data" / "eth_5m_1year.csv"
RSI_PERIOD = 14
STOCHRSI_PERIOD = 14
STOCHRSI_K_SMOOTH = 3
STOCHRSI_D_SMOOTH = 3
FADE_LOOKBACK = 12  # 1h, matches K12_1h -- StochRSI is fast enough to sweep 80->20 within this
FADE_NOCONFIRM_PCT = 0.002  # price must stay within +-0.2% of flat over that window to count as "didn't decline/rally"


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(avg_loss != 0.0, 100.0)


def add_rsi_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rsi = _rsi(out["close"], RSI_PERIOD)
    rsi_min = rsi.rolling(STOCHRSI_PERIOD, min_periods=STOCHRSI_PERIOD).min()
    rsi_max = rsi.rolling(STOCHRSI_PERIOD, min_periods=STOCHRSI_PERIOD).max()
    stoch_raw = 100.0 * (rsi - rsi_min) / (rsi_max - rsi_min).replace(0.0, np.nan)
    stoch_k = stoch_raw.rolling(STOCHRSI_K_SMOOTH, min_periods=STOCHRSI_K_SMOOTH).mean()
    stoch_d = stoch_k.rolling(STOCHRSI_D_SMOOTH, min_periods=STOCHRSI_D_SMOOTH).mean()

    out["rsi"] = rsi
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d
    out["rsi_roc_48"] = rsi - rsi.shift(48)
    out["price_roc_48"] = out["close"] / out["close"].shift(48) - 1.0
    out["price_roc_fade"] = out["close"] / out["close"].shift(FADE_LOOKBACK) - 1.0
    out["stoch_k_touched_high"] = stoch_k.rolling(FADE_LOOKBACK, min_periods=FADE_LOOKBACK).max() >= 80.0
    out["stoch_k_touched_low"] = stoch_k.rolling(FADE_LOOKBACK, min_periods=FADE_LOOKBACK).min() <= 20.0
    return out


def build_signals(frame: pd.DataFrame, side: str) -> dict:
    if side == "bottom":
        return {
            "rsi_extreme_30_70 (RSI<=30)": frame["rsi"] <= 30,
            "rsi_extreme_20_80 (RSI<=20)": frame["rsi"] <= 20,
            "stochrsi_extreme_20_80 (StochK<=20)": frame["stoch_k"] <= 20,
            "stochrsi_fade_from_extreme (touched<=20 in last 1h, now>=80, price NOT up>=0.2% over window)":
                frame["stoch_k_touched_low"] & (frame["stoch_k"] >= 80) & (frame["price_roc_fade"] <= FADE_NOCONFIRM_PCT),
            "rsi_momentum_divergence (price 4h-LL, RSI momentum UP)":
                (frame["price_roc_48"] <= -0.01) & (frame["rsi_roc_48"] >= 5),
        }
    return {
        "rsi_extreme_30_70 (RSI>=70)": frame["rsi"] >= 70,
        "rsi_extreme_20_80 (RSI>=80)": frame["rsi"] >= 80,
        "stochrsi_extreme_20_80 (StochK>=80)": frame["stoch_k"] >= 80,
        "stochrsi_fade_from_extreme (touched>=80 in last 1h, now<=20, price NOT down>=0.2% over window) [LITERAL USER QUESTION]":
            frame["stoch_k_touched_high"] & (frame["stoch_k"] <= 20) & (frame["price_roc_fade"] >= -FADE_NOCONFIRM_PCT),
        "rsi_momentum_divergence (price 4h-HH, RSI momentum DOWN)":
            (frame["price_roc_48"] >= 0.01) & (frame["rsi_roc_48"] <= -5),
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
    raw = load_frame()
    frame = add_rsi_indicators(raw).reset_index(drop=True)
    pivots = load_zigzag_pivots()

    ts = frame["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    print(f"Study window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots "
          f"({(pivots['pivot_type']=='bottom').sum()} bottom / {(pivots['pivot_type']=='top').sum()} top)")

    all_rows = pd.concat([run_side(frame, window_mask, pivots, "bottom"), run_side(frame, window_mask, pivots, "top")], ignore_index=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)
    for side in ("bottom", "top"):
        print(f"\n=== {side.upper()} evidence ===")
        sub = all_rows[all_rows["side"] == side]
        for horizon in K_HORIZONS:
            print(f"\n-- horizon {horizon} --")
            cols = ["signal", "n_triggers", "precision", "baseline_rate", "lift", "recall", "median_lead_bars", "excess_move_mean_pct"]
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_rsi_stochrsi_evidence_signal_20260825"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out_dir / "evidence_table.csv", index=False)
    print(f"\nWrote full table to {out_dir / 'evidence_table.csv'}")
    print("\nReference points from the existing master ranking (1h, same methodology, same repo):")
    print("  orthogonal_combo 3.51x (best overall) | liquidity_sweep 3.01x | volume_wick_climax 2.94x")
    print("  taker_sell_climax 2.75x | bollinger_pctb_extreme 2.34x | %R+SlowK oscillator 2.28x (closest analog)")
    print("  momentum_divergence(%R) 1.24x bottom / 0.88x top (WORSE than random, already CLOSED)")


if __name__ == "__main__":
    main()
