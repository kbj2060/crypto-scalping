#!/usr/bin/env python3
"""Follow-up to a discretionary-chart discussion (2026-08-26): user was longing 5m "bottom"
signals against what turned out to be a 5m-local pullback, not a macro downtrend (macro
dual_momentum regime was firmly bullish that day). Open question raised: does the "bottom"
evidence signal's lift actually depend on the macro (1-week) trend direction at the time it
fires, or is that untested?

Conditions the existing bottom/top evidence-signal lift study on macro regime, using the SAME
regime primitive that already gates the live zig075/Odyssey4 entry veto (locked recipe, not
re-derived here): features/engineering.py::FeatureEngineer._dual_momentum (2016-bar = 1wk
lookback, abs_momentum & BTC-relative momentum sign). Regime bucket per bar = which sign
dominated the trailing 1 week: up_frac = rolling(2016).mean(dual_momentum>0), down_frac =
rolling(2016).mean(dual_momentum<0), regime_score = up_frac - down_frac. macro_up if score>0,
macro_down if score<0. This is a majority-vote split (not the rare p90-extreme veto threshold
itself, which barely ever fires -- see 2026-08-26 same-day check, downtrend score peaked at
0.57 over the trailing 32 days vs the 0.9712 locked veto threshold) so both buckets get a
workable sample size.

event_study/excess_move/load_zigzag_pivots reused unmodified from
analyze_eth_confluence_oscillator_bottom_top_evidence_20260814.py. Signals reused unmodified
from that file (adaptive_both) and research_eth_rsi_orthogonal_combo_20260825.py
(orthogonal_combo_live). Same VAL+OOS window as all prior evidence-signal work. Retrospective
evidence study only, NOT a trading algorithm or promotion claim -- same caveat as the base
methodology file.
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

BTC_DATA_PATH = ROOT / "data" / "btc_5m_1year.csv"
WEEK_BARS = 2016  # matches features/engineering.py::_dual_momentum exactly


def add_regime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    btc = pd.read_csv(BTC_DATA_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp", keep="last").rename(columns={"close": "close_btc"})
    out = frame.merge(btc, on="timestamp", how="left")
    assert out["close_btc"].isna().sum() == 0, "BTC close failed to align onto every ETH bar -- timestamp mismatch"

    abs_momentum = (out["close"] / out["close"].shift(WEEK_BARS) - 1).fillna(0)
    btc_momentum = (out["close_btc"] / out["close_btc"].shift(WEEK_BARS) - 1).fillna(0)
    rel_momentum = abs_momentum - btc_momentum
    dual_momentum = np.where((abs_momentum > 0) & (rel_momentum > 0), 1.0,
                      np.where((abs_momentum < 0) & (rel_momentum < 0), -1.0, 0.0))
    dual_momentum = pd.Series(dual_momentum, index=out.index)

    out["dual_momentum"] = dual_momentum
    out["up_frac"] = dual_momentum.gt(0).rolling(WEEK_BARS, min_periods=WEEK_BARS).mean()
    out["down_frac"] = dual_momentum.lt(0).rolling(WEEK_BARS, min_periods=WEEK_BARS).mean()
    out["regime_score"] = out["up_frac"] - out["down_frac"]
    return out


def build_signals(frame: pd.DataFrame, side: str) -> dict:
    if side == "bottom":
        return {
            "orthogonal_combo_live (p_fast<=.10 AND p_slow<=.10 AND delta_z<=-2)":
                (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["delta_z"] <= -2.0),
            "adaptive_both (p_fast<=.10 AND p_slow<=.10)": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10),
        }
    return {
        "orthogonal_combo_live (p_fast>=.90 AND p_slow>=.90 AND delta_z>=2)":
            (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & (frame["delta_z"] >= 2.0),
        "adaptive_both (p_fast>=.90 AND p_slow>=.90)": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90),
    }


REGIME_BUCKETS = {
    "unconditional": None,
    "macro_up (regime_score>0)": lambda f: (f["regime_score"] > 0).to_numpy(),
    "macro_down (regime_score<0)": lambda f: (f["regime_score"] < 0).to_numpy(),
}


def run_side(frame: pd.DataFrame, base_window_mask: np.ndarray, pivots: pd.DataFrame, side: str) -> pd.DataFrame:
    close = frame["close"].to_numpy()
    side_pivots = pivots.loc[pivots["pivot_type"] == side]
    pivot_pos_full = frame.index[frame["timestamp"].isin(side_pivots["timestamp"])].to_numpy()

    rows = []
    for bucket_name, bucket_fn in REGIME_BUCKETS.items():
        window_mask = base_window_mask if bucket_fn is None else (base_window_mask & bucket_fn(frame))
        all_pos = np.flatnonzero(window_mask)
        # pivots restricted to bars that are themselves inside this bucket's window, so recall/
        # baseline_rate are computed within-regime, not diluted by out-of-regime pivots
        pivot_pos = pivot_pos_full[window_mask[pivot_pos_full]] if len(pivot_pos_full) else pivot_pos_full
        for sig_name, mask in build_signals(frame, side).items():
            trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
            for k_name, K in K_HORIZONS.items():
                stats = event_study(trigger_pos, pivot_pos, all_pos, K)
                move = excess_move(trigger_pos, pivot_pos, close, K)
                rows.append({"side": side, "regime": bucket_name, "signal": sig_name, "horizon": k_name,
                             "bucket_bars": int(window_mask.sum()), **stats,
                             "excess_move_mean_pct": move["mean_pct"], "excess_move_median_pct": move["median_pct"]})
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_frame_with_orderflow()
    frame = compute_indicators(raw)
    frame = add_creative_indicators(frame)
    frame = add_regime_columns(frame)
    frame = frame.reset_index(drop=True)
    pivots = load_zigzag_pivots()

    ts = frame["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    n_up = int((window_mask & (frame["regime_score"] > 0).to_numpy()).sum())
    n_down = int((window_mask & (frame["regime_score"] < 0).to_numpy()).sum())
    n_warmup_nan = int(frame.loc[window_mask, "regime_score"].isna().sum())
    print(f"Study window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")
    print(f"Regime split within window: macro_up={n_up} bars ({n_up/window_mask.sum()*100:.1f}%), "
          f"macro_down={n_down} bars ({n_down/window_mask.sum()*100:.1f}%), "
          f"regime_score NaN (warmup/tie)={n_warmup_nan}")

    all_rows = pd.concat([run_side(frame, window_mask, pivots, "bottom"), run_side(frame, window_mask, pivots, "top")], ignore_index=True)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 60)
    for side in ("bottom", "top"):
        print(f"\n=== {side.upper()} evidence, by macro regime ===")
        sub = all_rows[all_rows["side"] == side]
        for horizon in K_HORIZONS:
            print(f"\n-- horizon {horizon} --")
            cols = ["regime", "signal", "bucket_bars", "n_triggers", "precision", "baseline_rate", "lift", "recall", "excess_move_mean_pct"]
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_bottom_signal_macro_regime_conditional_lift_20260826"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out_dir / "evidence_table.csv", index=False)
    print(f"\nWrote full table to {out_dir / 'evidence_table.csv'}")


if __name__ == "__main__":
    main()
