#!/usr/bin/env python3
"""Yush(@TraderYush) orderflow/LAF-model component evidence study for ETH 5m -- NOT a trading
algorithm and NOT a promotion claim. Same retrospective event-study methodology and the same
harness as analyze_eth_deep_evidence_signal_sweep_round2_20260814.py (imported unmodified);
ground truth is the repo's zigzag_action swing pivots.

Purpose: Yush is a discretionary NQ/ES futures orderflow trader. Several of his LAF-model
components already have a near-equivalent in this repo (trapped traders ~= A3_liquidity_sweep,
absorption-with-long-wick ~= the Wyckoff signal, delta ~= the orderflow net-sell surge,
value area ~= core/cvp.py). This script tests only the components that DO NOT yet exist
anywhere in the repo's feature set or evidence scorecard, so the question "is it absorbable"
gets a number instead of an opinion:

  Y1 prior_day_level      -- previous completed UTC day's high/low. Yush's "Previous Day High
                             and Low". The repo has NO market-generated session-level feature
                             (grep for prev_day_high/pdh/opening_range over features/ + core/
                             returns nothing); the existing A3_liquidity_sweep uses a ROLLING
                             N-bar swing, which is a different object.
  Y2 prior_asia_level     -- previous completed Asia session (00-08 UTC) high/low, the closest
                             24/7-crypto analogue of Yush's "Overnight High and Low".
  Y3 value_area_edge      -- previous UTC day's 70% Value Area High/Low from a real per-day
                             volume profile. Yush trades value-area EDGES only.
  Y4 value_area_middle    -- price parked near the prior day's POC. This is the direct test of
                             Yush's explicit rule "the middle of the range is unpredictable and
                             should be avoided" -- it is expected to UNDERperform, and a lift
                             near/below 1.0 is the confirmation, not a failure.
  Y5 lvn_touch            -- Low Volume Node of the prior day's profile (bin volume < 50% of the
                             in-range mean bin volume). Yush's trend model enters pullbacks into
                             LVNs. No LVN concept exists in core/cvp.py (it exposes POC /
                             VAH-VAL width / cluster position / volume imbalance, not nodes).
  Y6 absorption_no_move   -- Yush's literal absorption definition: "heavy buying or selling
                             activity but the price does not move". Encoded as volume z-score
                             high AND bar body tiny relative to ATR AND taker delta pushing the
                             wrong way. Distinct from the existing Wyckoff signal, which
                             REQUIRES a long wick (i.e. price did move and came back).
  Y7 level_sweep_reentry  -- breaks the prior-day level intrabar but closes back inside. Yush's
                             "confirmation entry: after breakout fails and price re-enters".
                             Same shape as A3 but anchored to a market-generated level.
  Y8 big_trade_proxy      -- Yush filters NQ 75+ / ES 200+ lot prints. 5m bars carry no tick
                             tape, so the closest available proxy is average trade notional
                             (quote_volume / trades) z-score = unusually large average print.
  Y9 confirmation_count   -- Yush's core execution rule is "at least two confirmations must
                             align". Tested as a ladder (>=1 / >=2 / >=3 of Y1,Y2,Y3,Y5,Y6,Y8)
                             so the rule itself, not any single component, gets measured.

Causality note: every level is built from a COMPLETED prior day / prior Asia session and applied
forward, so no same-bar lookahead is used to define a level. As with the other evidence-sweep
scripts, looking FORWARD from a trigger bar to a real pivot is the retrospective study design
itself; this makes no fresh-forward/promotion claim.
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
OUT_DIR = ROOT / "tmp" / "eth_yush_orderflow_component_evidence_20260815"

N_BINS = 60             # price bins per daily volume profile
VALUE_AREA_FRAC = 0.70  # Yush/AMT standard: value area = 70% of the day's volume
LVN_FRAC = 0.50         # a bin is a Low Volume Node if its volume < 50% of the in-range mean
TOUCH_ATR_MULT = 0.35   # "at the level" tolerance, in units of the bar's ATR%


def load_frame() -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]
    df = pd.read_csv(ETH_PATH, usecols=cols, parse_dates=["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _daily_profile(day_df: pd.DataFrame) -> dict:
    """One UTC day's volume profile -> POC / VAH / VAL / LVN bin edges."""
    lo, hi = float(day_df["low"].min()), float(day_df["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return {}
    edges = np.linspace(lo, hi, N_BINS + 1)
    hlc3 = ((day_df["high"] + day_df["low"] + day_df["close"]) / 3.0).to_numpy()
    vol = day_df["volume"].to_numpy(dtype=np.float64)
    idx = np.clip(np.digitize(hlc3, edges) - 1, 0, N_BINS - 1)
    binvol = np.bincount(idx, weights=vol, minlength=N_BINS)
    total = binvol.sum()
    if total <= 0:
        return {}

    poc_bin = int(binvol.argmax())
    # Standard value-area construction: expand from the POC, always taking the richer neighbour.
    lo_b = hi_b = poc_bin
    acc = binvol[poc_bin]
    while acc / total < VALUE_AREA_FRAC and (lo_b > 0 or hi_b < N_BINS - 1):
        below = binvol[lo_b - 1] if lo_b > 0 else -1.0
        above = binvol[hi_b + 1] if hi_b < N_BINS - 1 else -1.0
        if above >= below:
            hi_b += 1
            acc += binvol[hi_b]
        else:
            lo_b -= 1
            acc += binvol[lo_b]

    centers = (edges[:-1] + edges[1:]) / 2.0
    occupied = binvol > 0
    mean_occ = binvol[occupied].mean() if occupied.any() else 0.0
    lvn_mask = occupied & (binvol < LVN_FRAC * mean_occ)
    return {
        "poc": float(centers[poc_bin]),
        "vah": float(edges[hi_b + 1]),
        "val": float(edges[lo_b]),
        "lvn_lo": edges[:-1][lvn_mask].astype(float),
        "lvn_hi": edges[1:][lvn_mask].astype(float),
    }


def add_session_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Prior-day OHLC levels, prior-Asia-session levels, and prior-day volume-profile levels."""
    f = frame.copy()
    day = f["timestamp"].dt.floor("D")
    f["_day"] = day

    daily = f.groupby("_day").agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last"))
    daily = daily.shift(1)  # previous COMPLETED day
    f = f.merge(daily.rename(columns={"day_high": "pdh", "day_low": "pdl", "day_close": "pdc"}),
                left_on="_day", right_index=True, how="left")

    asia = f[f["timestamp"].dt.hour < 8].groupby("_day").agg(asia_high=("high", "max"), asia_low=("low", "min"))
    asia = asia.reindex(daily.index).shift(1)  # previous day's completed Asia session
    f = f.merge(asia.rename(columns={"asia_high": "onh", "asia_low": "onl"}),
                left_on="_day", right_index=True, how="left")

    profiles = {}
    for d, g in frame.groupby(day):
        prof = _daily_profile(g)
        if prof:
            profiles[d] = prof
    days = list(daily.index)
    prev_of = {d: days[i - 1] for i, d in enumerate(days) if i > 0}

    poc = np.full(len(f), np.nan)
    vah = np.full(len(f), np.nan)
    val = np.full(len(f), np.nan)
    in_lvn = np.zeros(len(f), dtype=bool)
    close_arr = f["close"].to_numpy()
    for d, pos in f.groupby("_day").indices.items():
        prof = profiles.get(prev_of.get(d))
        if not prof:
            continue
        poc[pos] = prof["poc"]
        vah[pos] = prof["vah"]
        val[pos] = prof["val"]
        if len(prof["lvn_lo"]):
            c = close_arr[pos]
            hit = ((c[:, None] >= prof["lvn_lo"][None, :]) & (c[:, None] < prof["lvn_hi"][None, :])).any(axis=1)
            in_lvn[pos] = hit
    f["pd_poc"], f["pd_vah"], f["pd_val"], f["in_lvn"] = poc, vah, val, in_lvn
    return f.drop(columns=["_day"])


def add_flow_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    qv = f["quote_volume"].replace(0.0, np.nan)
    f["delta_ratio"] = 2.0 * (f["taker_buy_quote"] / qv) - 1.0          # +1 all taker-buy, -1 all taker-sell
    f["vol_z"] = (f["volume"] - f["volume"].rolling(288, min_periods=144).mean()) / (
        f["volume"].rolling(288, min_periods=144).std() + 1e-12)
    avg_print = qv / f["trades"].replace(0, np.nan)
    f["avg_print_z"] = (avg_print - avg_print.rolling(288, min_periods=144).mean()) / (
        avg_print.rolling(288, min_periods=144).std() + 1e-12)
    body_pct = (f["close"] - f["open"]).abs() / f["close"]
    f["body_over_atr"] = body_pct / (f["atr_pct"] + 1e-12)
    return f


def _touch(price_series: pd.Series, level: pd.Series, atr_pct: pd.Series, close: pd.Series) -> pd.Series:
    tol = TOUCH_ATR_MULT * atr_pct * close
    return (price_series - level).abs() <= tol


def yush_components(frame: pd.DataFrame, side: str) -> dict:
    f = frame
    close, atr = f["close"], f["atr_pct"]
    va_width = (f["pd_vah"] - f["pd_val"]).abs()

    absorption_core = (f["vol_z"] >= 1.5) & (f["body_over_atr"] <= 0.35)
    big_print = f["avg_print_z"] >= 1.5
    va_mid = (close > f["pd_val"]) & (close < f["pd_vah"]) & ((close - f["pd_poc"]).abs() <= 0.15 * va_width)

    if side == "bottom":
        lvl = _touch(f["low"], f["pdl"], atr, close) | _touch(f["low"], f["pdc"], atr, close)
        asia_lvl = _touch(f["low"], f["onl"], atr, close)
        va_edge = _touch(f["low"], f["pd_val"], atr, close)
        absorption = absorption_core & (f["delta_ratio"] <= -0.05)   # sellers aggressive, price won't fall
        sweep = (f["low"] < f["pdl"]) & (f["close"] > f["pdl"])
    else:
        lvl = _touch(f["high"], f["pdh"], atr, close) | _touch(f["high"], f["pdc"], atr, close)
        asia_lvl = _touch(f["high"], f["onh"], atr, close)
        va_edge = _touch(f["high"], f["pd_vah"], atr, close)
        absorption = absorption_core & (f["delta_ratio"] >= 0.05)    # buyers aggressive, price won't rise
        sweep = (f["high"] > f["pdh"]) & (f["close"] < f["pdh"])

    comps = {
        "Y1_prior_day_level": lvl,
        "Y2_prior_asia_level": asia_lvl,
        "Y3_value_area_edge": va_edge,
        "Y4_value_area_middle": va_mid,
        "Y5_lvn_touch": f["in_lvn"],
        "Y6_absorption_no_move": absorption,
        "Y7_level_sweep_reentry": sweep,
        "Y8_big_print_proxy": big_print,
    }

    # Yush's "minimum two confirmations" rule, as a ladder. Y4 (middle of range) is excluded on
    # purpose -- it is his anti-signal, not a confirmation.
    confirm_cols = ["Y1_prior_day_level", "Y2_prior_asia_level", "Y3_value_area_edge",
                    "Y5_lvn_touch", "Y6_absorption_no_move", "Y8_big_print_proxy"]
    count = sum(comps[c].fillna(False).astype(int) for c in confirm_cols)
    for k in (1, 2, 3):
        comps[f"Y9_confirmations_ge{k}"] = count >= k
    return comps


def run_side(frame: pd.DataFrame, window_mask: np.ndarray, pivots: pd.DataFrame, side: str) -> pd.DataFrame:
    close = frame["close"].to_numpy()
    all_pos = np.flatnonzero(window_mask)
    pivot_pos = frame.index[frame["timestamp"].isin(pivots.loc[pivots["pivot_type"] == side, "timestamp"])].to_numpy()
    rows = []
    for name, mask in yush_components(frame, side).items():
        trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            move = excess_move(trigger_pos, pivot_pos, close, K)
            rows.append({"side": side, "signal": name, "horizon": k_name, **stats,
                         "excess_move_mean_pct": move["mean_pct"]})
    return pd.DataFrame(rows)


def absorption_ablation(frame: pd.DataFrame, window_mask: np.ndarray, pivots: pd.DataFrame,
                        side: str) -> pd.DataFrame:
    """Is Y6's lift just "big volume bar"? Peel the three conjuncts apart, and split VAL vs OOS.

    Also compares against the repo's existing Wyckoff-style long-wick signal shape (volume spike
    + long wick), because that one is ALREADY in the scorecard -- Y6 only matters if it is not
    the same bars under a new name.
    """
    f = frame
    sign = -1.0 if side == "bottom" else 1.0
    wick = (f["low"] if side == "bottom" else f["high"])
    wick_len = ((f[["open", "close"]].min(axis=1) - wick) if side == "bottom"
                else (wick - f[["open", "close"]].max(axis=1))).abs()
    body = (f["close"] - f["open"]).abs()

    variants = {
        "vol_z_only":                 f["vol_z"] >= 1.5,
        "small_body_only":            f["body_over_atr"] <= 0.35,
        "delta_only":                 (sign * f["delta_ratio"]) >= 0.05,
        "vol_z+small_body":           (f["vol_z"] >= 1.5) & (f["body_over_atr"] <= 0.35),
        "Y6_full(vol+body+delta)":    (f["vol_z"] >= 1.5) & (f["body_over_atr"] <= 0.35)
                                      & ((sign * f["delta_ratio"]) >= 0.05),
        "wyckoff_vol+long_wick":      (f["vol_z"] >= 1.5) & (wick_len >= 1.5 * body),
    }

    close = f["close"].to_numpy()
    pivot_pos = f.index[f["timestamp"].isin(pivots.loc[pivots["pivot_type"] == side, "timestamp"])].to_numpy()
    ts = f["timestamp"]
    splits = {
        "VAL+OOS": window_mask,
        "VAL": ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy(),
        "OOS": ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy(),
    }
    rows = []
    for split_name, mask in splits.items():
        all_pos = np.flatnonzero(mask)
        for name, sig in variants.items():
            trigger_pos = np.flatnonzero(sig.fillna(False).to_numpy() & mask)
            stats = event_study(trigger_pos, pivot_pos, all_pos, K_HORIZONS["K12_1h"])
            move = excess_move(trigger_pos, pivot_pos, close, K_HORIZONS["K12_1h"])
            rows.append({"side": side, "split": split_name, "variant": name, **stats,
                         "excess_move_mean_pct": move["mean_pct"]})

    # Overlap between Y6 and the already-scored Wyckoff signal, on the full window.
    y6 = variants["Y6_full(vol+body+delta)"].fillna(False).to_numpy() & window_mask
    wy = variants["wyckoff_vol+long_wick"].fillna(False).to_numpy() & window_mask
    inter = int((y6 & wy).sum())
    print(f"[{side}] Y6 n={int(y6.sum())}, wyckoff n={int(wy.sum())}, overlap={inter} "
          f"({inter / max(int(y6.sum()), 1):.1%} of Y6 bars)")
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_frame()
    frame = compute_indicators(raw).reset_index(drop=True)
    frame = add_session_levels(frame)
    frame = add_flow_features(frame)
    pivots = load_zigzag_pivots()

    ts = frame["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    cov = frame.loc[window_mask, ["pdl", "pd_val", "onl"]].notna().mean()
    print(f"Study window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")
    print(f"Level coverage in-window: prior_day={cov['pdl']:.1%} value_area={cov['pd_val']:.1%} asia={cov['onl']:.1%}")

    rows = pd.concat([run_side(frame, window_mask, pivots, s) for s in ("bottom", "top")], ignore_index=True)

    pd.set_option("display.width", 180)
    cols = ["signal", "n_triggers", "precision", "baseline_rate", "lift", "recall", "excess_move_mean_pct"]
    for side in ("bottom", "top"):
        print(f"\n\n########## {side.upper()} ##########")
        sub = rows[rows["side"] == side]
        for horizon in K_HORIZONS:
            print(f"\n-- {horizon} --")
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    print("\n\n########## ABSORPTION (Y6) ABLATION + VAL/OOS SPLIT, 1h ##########")
    abl = pd.concat([absorption_ablation(frame, window_mask, pivots, s) for s in ("bottom", "top")],
                    ignore_index=True)
    for side in ("bottom", "top"):
        print(f"\n=== {side.upper()} ===")
        print(abl[abl["side"] == side][
            ["split", "variant", "n_triggers", "precision", "baseline_rate", "lift", "excess_move_mean_pct"]
        ].to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_DIR / "yush_component_evidence_table.csv", index=False)
    abl.to_csv(OUT_DIR / "absorption_ablation_table.csv", index=False)
    print(f"\nWrote tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
