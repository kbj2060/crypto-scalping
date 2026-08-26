#!/usr/bin/env python3
"""Combo-candidate search using two info sources NOT yet combined with the oscillator leg in
this repo's zigzag-pivot lift methodology: funding rate (level) and cross-asset (BTC).
Liquidation was considered and excluded -- see the "why liquidation is skipped" note below,
this is a real data-availability wall, not an oversight.

Reuses event_study/excess_move/load_zigzag_pivots/compute_indicators/add_creative_indicators
unmodified, same VAL+OOS window, for direct comparability with the master ranking
(eth_reversal_evidence_signal_scorecard_20260814) and today's RSI combo check
(research_eth_rsi_orthogonal_combo_20260825.py).

Data notes:
  - Funding: `data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv` is the CORRECTED file (the original
    2026-08-14 funding_extreme lift result used data/TOTAL_ETHFIUSDT_fundingRate.csv by mistake
    -- that's ETHFI token funding, not ETH -- see eth_funding_ethfi_mislabel_20260824 memory).
    `funding_extreme_low`/`_high` below is a re-run on the CORRECT file, both standalone (fixes
    the mislabeled prior result) and combined with the oscillator leg. merge_asof(direction=
    "backward") onto the 5m frame -- only the most recently PUBLISHED funding rate as of each
    bar is used, no lookahead.
  - Cross-asset: `data/btc_5m_1year.csv` has full VAL+OOS coverage, exact same 5m timestamps as
    the ETH file (verified: both Binance klines, same source pipeline). BTC's own oscillator
    (via the SAME compute_indicators() call, not a re-derivation) is joined on exact timestamp.
    Two distinct hypotheses tested: "ETH oversold while BTC is NOT" (idiosyncratic/ETH-specific
    capitulation) vs "ETH and BTC both oversold together" (systematic/market-wide) -- plus the
    already-standalone-tested ETH/BTC ratio extreme (eth_reversal_evidence_signal_scorecard:
    1.61x bottom-only, flagged as a possible bear-market-window artifact) now combined with the
    oscillator leg for the first time.
  - Liquidation SKIPPED: tail_risk_1m/oi_lsratio (the only source for liq_net_z_12 / crowding
    legs) only has live history from 2026-07-18 / 2026-08-22 onward -- ZERO overlap with this
    scorecard's VAL+OOS window (2025-09-01..2026-02-17). Recomputing zigzag pivots on the short
    recent-only window to force a test would (a) not be comparable to any existing lift number
    here, and (b) re-peek at data still accumulating toward its own pre-registered 09-15 gate
    (eth_liquidation_crowding_conditional_fade_arm_preregistration_20260823,
    eth_liquidation_s13_s14_early_peek_status_20260825) -- both of which this repo has
    repeatedly flagged as invalid. Revisit after 09-15 (and after position-skew history clears
    its own 15-day minimum, currently ~3 days).
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
from analyze_eth_creative_reversal_evidence_signals_20260814 import (  # noqa: E402
    add_creative_indicators,
    load_frame_with_orderflow,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
    compute_indicators,
)

BTC_PATH = ROOT / "data" / "btc_5m_1year.csv"
FUNDING_PATH = ROOT / "data" / "TOTAL_ETHUSDT_fundingRate_2025_2026.csv"
RATIO_Z_WINDOW = 864  # 3 days, matches this repo's existing vwap_dev_z/percentile_window convention
FUNDING_Z_MIN_PERIODS = 30  # ~10 days of 8h funding observations before trusting a rolling z


def load_btc_oscillator() -> pd.DataFrame:
    btc = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    btc = compute_indicators(btc)
    return btc[["timestamp", "close", "p_fast", "p_slow"]].rename(
        columns={"close": "btc_close", "p_fast": "btc_p_fast", "p_slow": "btc_p_slow"}
    )


def load_funding_z() -> pd.DataFrame:
    f = pd.read_csv(FUNDING_PATH, parse_dates=["calc_time"])
    f = f.sort_values("calc_time").reset_index(drop=True)
    mean = f["last_funding_rate"].rolling(90, min_periods=FUNDING_Z_MIN_PERIODS).mean()
    std = f["last_funding_rate"].rolling(90, min_periods=FUNDING_Z_MIN_PERIODS).std()
    f["funding_z"] = (f["last_funding_rate"] - mean) / std.replace(0.0, np.nan)
    return f[["calc_time", "funding_z"]]


def build_frame() -> pd.DataFrame:
    raw = load_frame_with_orderflow()
    frame = compute_indicators(raw)
    frame = add_creative_indicators(frame)

    btc = load_btc_oscillator()
    frame = frame.merge(btc, on="timestamp", how="left")
    ratio = frame["close"] / frame["btc_close"]
    ratio_mean = ratio.rolling(RATIO_Z_WINDOW, min_periods=RATIO_Z_WINDOW).mean()
    ratio_std = ratio.rolling(RATIO_Z_WINDOW, min_periods=RATIO_Z_WINDOW).std()
    frame["ethbtc_ratio_z"] = (ratio - ratio_mean) / ratio_std.replace(0.0, np.nan)

    funding = load_funding_z()
    frame = pd.merge_asof(frame.sort_values("timestamp"), funding, left_on="timestamp", right_on="calc_time", direction="backward")
    return frame.reset_index(drop=True)


def build_signals(frame: pd.DataFrame, side: str) -> dict:
    if side == "bottom":
        return {
            "orthogonal_combo_live [reproduced baseline]": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["delta_z"] <= -2.0),
            "funding_extreme_low [CORRECTED FILE, standalone]": frame["funding_z"] <= -2.0,
            "oscillator_and_funding_low (osc oversold AND funding_z<=-2)": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["funding_z"] <= -2.0),
            "eth_oversold_btc_not (ETH osc oversold, BTC NOT: idiosyncratic)": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["btc_p_fast"] > 0.30),
            "eth_btc_both_oversold (systematic/market-wide)": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["btc_p_fast"] <= 0.10),
            "oscillator_and_ethbtc_ratio_extreme (osc oversold AND ETH cheap vs BTC)": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["ethbtc_ratio_z"] <= -2.0),
        }
    return {
        "orthogonal_combo_live [reproduced baseline]": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & (frame["delta_z"] >= 2.0),
        "funding_extreme_high [CORRECTED FILE, standalone]": frame["funding_z"] >= 2.0,
        "oscillator_and_funding_high (osc overbought AND funding_z>=2)": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & (frame["funding_z"] >= 2.0),
        "eth_overbought_btc_not (ETH osc overbought, BTC NOT: idiosyncratic)": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & (frame["btc_p_fast"] < 0.70),
        "eth_btc_both_overbought (systematic/market-wide)": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & (frame["btc_p_fast"] >= 0.90),
        "oscillator_and_ethbtc_ratio_extreme (osc overbought AND ETH rich vs BTC)": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & (frame["ethbtc_ratio_z"] >= 2.0),
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
    frame = build_frame()
    pivots = load_zigzag_pivots()

    ts = frame["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    print(f"Study window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")
    print(f"BTC join coverage: {frame['btc_p_fast'].notna().sum()}/{len(frame)} bars matched")
    print(f"Funding join coverage: {frame['funding_z'].notna().sum()}/{len(frame)} bars matched")

    all_rows = pd.concat([run_side(frame, window_mask, pivots, "bottom"), run_side(frame, window_mask, pivots, "top")], ignore_index=True)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 90)
    for side in ("bottom", "top"):
        print(f"\n=== {side.upper()} evidence ===")
        sub = all_rows[all_rows["side"] == side]
        for horizon in K_HORIZONS:
            print(f"\n-- horizon {horizon} --")
            cols = ["signal", "n_triggers", "precision", "lift", "recall", "median_lead_bars", "excess_move_mean_pct"]
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_funding_crossasset_combo_signal_20260825"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out_dir / "evidence_table.csv", index=False)
    print(f"\nWrote full table to {out_dir / 'evidence_table.csv'}")


if __name__ == "__main__":
    main()
