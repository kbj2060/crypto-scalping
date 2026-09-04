#!/usr/bin/env python3
"""New combination research: does nif_retail (리테일 수급, the one model indicator with real,
repeatedly-confirmed direction IC -- see eth_whale_position_vs_retail_flow_direction_ic_20260825)
add anything when used as a CONFIRMATION filter on top of the 9 live evidence signals, using this
repo's own established zigzag-pivot lift methodology (event_study, reused verbatim)? This is the
retail-flow analog of the whale-confirmation combo already tested and REJECTED
(eth_microstructure_1m_history_archive_and_whale_confirmation_rejected_20260823) -- that one used
whale_position_score, which has since been shown to carry no real direction information at all;
nif_retail is a genuinely different, better-motivated candidate since IT does.

Data-overlap constraint (same class of issue as the liquidation-data non-overlap noted in
eth_funding_oscillator_combo_candidate_20260825): nif_retail's only source, microstructure_1m,
starts 2026-05-03 -- ZERO overlap with the evidence-signal master scorecard's VAL+OOS window
(2025-09-01..2026-02-17). No pre-registered gate blocks this combo specifically (unlike the
liquidation 09-15 gate), so this is legitimate new exploratory research on the window that DOES
overlap: 2026-05-03..2026-07-20 (bounded by zigzag pivot label coverage), ~2.5 months, clearly
weaker/shorter than the master scorecard's ~5.5 months -- flagged throughout, not silently ported.

Step 1: correlation screen (nif_retail vs each evidence signal's continuous leg) to avoid testing
already-known-redundant pairs (e.g. nif_retail is highly correlated with taker_buy_ratio, and
taker_delta_z_climax/orthogonal_combo's orderflow leg is built from the same all-trade taker-buy
quantity -- likely redundant by construction, not a new information family).
Step 2: for the least-correlated signals only, test confirmation combos (evidence signal fires AND
nif_retail agrees in direction) against the base signal's own lift, same window, same methodology.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    event_study,
    excess_move,
    load_zigzag_pivots,
)
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

DATA_PATH = ROOT / "data" / "splits" / "year_oos" / "training_features_2026_rebuilt.csv"
BTC_PATH = ROOT / "data" / "splits" / "year_oos" / "btc_features_2026.csv"  # btc_5m_1year.csv ends 2026-02-17, doesn't cover this window
MICRO_DB_PATH = ROOT / "data" / "live" / "microstructure.duckdb"
WIN_START, WIN_END = pd.Timestamp("2026-05-03"), pd.Timestamp("2026-07-20")
RETAIL_Z_WINDOW = 288  # 1 day of 5m bars -- same convention as delta_z/vol_z elsewhere in this lineage


def load_nif_retail_5m() -> pd.DataFrame:
    """nif_retail is native 1-minute; resample to a 5m-bar-aligned series (mean of the trailing
    5 one-minute readings, matching the bar's own close) then rolling-z it the same way every
    other continuous leg in this lineage is z-scored."""
    con = duckdb.connect(str(MICRO_DB_PATH), read_only=True)
    micro = con.execute("SELECT ts, nif_retail FROM microstructure_1m ORDER BY ts").fetchdf()
    con.close()
    micro["ts"] = pd.to_datetime(micro["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)
    micro = micro.set_index("ts").sort_index()
    resampled = micro["nif_retail"].resample("5min", label="right", closed="right").mean()
    out = resampled.reset_index().rename(columns={"ts": "timestamp"})
    mu = out["nif_retail"].rolling(RETAIL_Z_WINDOW, min_periods=RETAIL_Z_WINDOW).mean()
    sd = out["nif_retail"].rolling(RETAIL_Z_WINDOW, min_periods=RETAIL_Z_WINDOW).std()
    out["nif_retail_z"] = (out["nif_retail"] - mu) / sd.replace(0.0, np.nan)
    return out


def build_frame() -> pd.DataFrame:
    raw = pd.read_csv(
        DATA_PATH, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"],
        parse_dates=["timestamp"],
    )
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    btc_raw = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    sig = compute_signals(raw, btc_df=btc_raw)  # funding omitted -- not needed for this combo set; p_fast/p_slow already included
    retail = load_nif_retail_5m()
    sig = sig.merge(retail[["timestamp", "nif_retail_z"]], on="timestamp", how="left")
    return sig


def main() -> None:
    frame = build_frame()
    ts = frame["timestamp"]
    window_mask = ((ts >= WIN_START) & (ts <= WIN_END)).to_numpy()
    print(f"Window: {WIN_START.date()}..{WIN_END.date()}, {int(window_mask.sum())} bars, "
          f"nif_retail coverage in window: {frame.loc[window_mask, 'nif_retail_z'].notna().mean()*100:.1f}%")

    # --- Step 1: correlation screen (which evidence-signal legs is nif_retail redundant with?) ---
    print("\n=== correlation screen: nif_retail_z vs each signal's continuous leg (in-window) ===")
    legs = {
        "p_fast (oscillator, orthogonal_combo leg)": frame["p_fast"],
        "delta_z (taker_delta_z_climax leg)": frame["delta_z"],
        "ret3_z (short_term_return_z leg)": frame["ret3_z"],
        "vol_z (volume_wick_climax leg)": frame["vol_z"],
    }
    sub_mask = window_mask & frame["nif_retail_z"].notna().to_numpy()
    for name, series in legs.items():
        pair = pd.DataFrame({"a": frame["nif_retail_z"], "b": series})[sub_mask].dropna()
        if len(pair) > 200:
            rho = spearmanr(pair["a"], pair["b"]).statistic
            print(f"  {name:<45} n={len(pair):>5}  spearman={rho:+.3f}")

    # --- Step 2: confirmation combos for signals whose leg ISN'T orderflow-family (avoid the
    # already-known taker-flow redundancy) -- price-geometry (liquidity_sweep, fib_extension),
    # cross-asset (smt_divergence), range-position (dalton), pure price-position (orthogonal_combo
    # itself, to see if retail adds beyond its own orderflow leg) ---
    pivots = load_zigzag_pivots()
    close = frame["close"].to_numpy()
    all_pos = np.flatnonzero(window_mask)
    CANDIDATES = ["orthogonal_combo", "liquidity_sweep", "fib_extension_exhaustion", "smt_divergence", "dalton_rule2_balance_edge"]

    print("\n=== base signal lift vs retail-confirmed combo lift (1h horizon) ===")
    K = K_HORIZONS["K12_1h"]
    rows = []
    for name in CANDIDATES:
        for side in ("bottom", "top"):
            base_col = f"{side}_{name}"
            base_mask = frame[base_col].fillna(False).to_numpy() & window_mask
            retail_agrees = (frame["nif_retail_z"] <= -0.5) if side == "bottom" else (frame["nif_retail_z"] >= 0.5)
            combo_mask = base_mask & retail_agrees.fillna(False).to_numpy()

            side_pivots = pivots.loc[pivots["pivot_type"] == side]
            pivot_pos = frame.index[frame["timestamp"].isin(side_pivots["timestamp"])].to_numpy()

            base_stats = event_study(np.flatnonzero(base_mask), pivot_pos, all_pos, K)
            combo_stats = event_study(np.flatnonzero(combo_mask), pivot_pos, all_pos, K)
            rows.append({"signal": name, "side": side,
                         "base_n": base_stats["n_triggers"], "base_lift": base_stats["lift"],
                         "combo_n": combo_stats["n_triggers"], "combo_lift": combo_stats["lift"]})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(res.to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_retail_flow_evidence_signal_combo_20260825"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "combo_table.csv", index=False)
    print(f"\nWrote {out_dir / 'combo_table.csv'}")


if __name__ == "__main__":
    main()
