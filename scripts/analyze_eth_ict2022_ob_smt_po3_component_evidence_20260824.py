#!/usr/bin/env python3
"""Evidence study (NOT a trading algorithm, NOT a promotion claim) for the 3 ICT-2022
components still unmeasured in this repo: Order Blocks, SMT divergence, Power of 3 (Judas
swing). Same harness / windows / zigzag-pivot ground truth as the AMT/VSA/iFVG sibling study
(analyze_eth_amt_vsa_footprint_ifvg_component_evidence_20260815.py) so lifts are directly
comparable to the master scorecard (plain liquidity sweep 3.01x/2.78x is the reference).

Pre-registered design (locked before running):
docs/experiments/eth_ict2022_ob_smt_po3_component_evidence_20260824.md

Timezone: data/eth_5m_1year.csv verified UTC-naive on 2026-08-24 (100% exact close match vs
ETHUSDT-5m-api.csv at identical timestamps; KST hypothesis 0.01%) -- required for the BTC join
(SMT) and the UTC session anchors (Po3).
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

from analyze_eth_amt_vsa_footprint_ifvg_component_evidence_20260815 import (  # noqa: E402
    add_sweep,
    load_frame,
)
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

BTC_PATH = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp" / "eth_ict2022_ob_smt_po3_component_evidence_20260824"
OB_DISPLACEMENT_ATR = 1.0
OB_LIFETIME = 48          # bars, matches the FVG-zone lookback convention
ASIA_MIN_BARS = 48        # require a substantially complete 00-07 UTC range


def add_order_blocks(f: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = f["atr_price"].to_numpy()
    n = len(f)
    c3 = np.full(n, np.nan)
    c3[:-3] = c[3:]

    bull_form = (c < o) & (c3 > h) & ((c3 - c) >= OB_DISPLACEMENT_ATR * atr)
    bear_form = (c > o) & (c3 < l) & ((c - c3) >= OB_DISPLACEMENT_ATR * atr)

    touch_bull = np.zeros(n, dtype=bool)
    touch_bear = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(bull_form):
        zone_lo, zone_hi = l[i], h[i]
        for j in range(i + 4, min(i + 4 + OB_LIFETIME, n)):
            if c[j] < zone_lo:          # body close through the zone -> broken, no fire
                break
            if l[j] <= zone_hi and h[j] >= zone_lo:
                touch_bull[j] = True
    for i in np.flatnonzero(bear_form):
        zone_lo, zone_hi = l[i], h[i]
        for j in range(i + 4, min(i + 4 + OB_LIFETIME, n)):
            if c[j] > zone_hi:
                break
            if l[j] <= zone_hi and h[j] >= zone_lo:
                touch_bear[j] = True

    f["ob_touch_bottom"], f["ob_touch_top"] = touch_bull, touch_bear
    print(f"OB formations: bull={int(bull_form.sum())} bear={int(bear_form.sum())}")
    return f


def add_smt(f: pd.DataFrame) -> pd.DataFrame:
    btc = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    btc = (btc.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
           .rename(columns={"high": "btc_high", "low": "btc_low"}))
    merged = f.merge(btc, on="timestamp", how="left")
    assert len(merged) == len(f), "BTC merge changed row count"
    f["btc_high"], f["btc_low"] = merged["btc_high"].to_numpy(), merged["btc_low"].to_numpy()
    print(f"BTC coverage on ETH grid: {f['btc_low'].notna().mean() * 100:.2f}%")

    btc_swing_low = f["btc_low"].rolling(48, min_periods=48).min().shift(1)
    btc_swing_high = f["btc_high"].rolling(48, min_periods=48).max().shift(1)
    btc_holds_low = (f["btc_low"] > btc_swing_low).fillna(False)
    btc_holds_high = (f["btc_high"] < btc_swing_high).fillna(False)
    eth_break_low = f["low"] < f["swing_low_prior"]
    eth_break_high = f["high"] > f["swing_high_prior"]

    f["smt_raw_bottom"] = eth_break_low & btc_holds_low
    f["smt_raw_top"] = eth_break_high & btc_holds_high
    f["smt_sweep_bottom"] = f["sweep_low"] & btc_holds_low
    f["smt_sweep_top"] = f["sweep_high"] & btc_holds_high
    return f


def add_po3(f: pd.DataFrame) -> pd.DataFrame:
    ts = f["timestamp"]
    day = ts.dt.normalize()
    hour = ts.dt.hour
    asia = hour < 7
    asia_low = f["low"].where(asia).groupby(day).cummin().groupby(day).ffill()
    asia_high = f["high"].where(asia).groupby(day).cummax().groupby(day).ffill()
    asia_cnt = asia.astype(int).groupby(day).cumsum()
    manip = (hour >= 7) & (hour < 14) & (asia_cnt >= ASIA_MIN_BARS)

    f["po3_bottom"] = manip & (f["low"] < asia_low) & (f["close"] > asia_low)
    f["po3_top"] = manip & (f["high"] > asia_high) & (f["close"] < asia_high)
    return f


def components(f: pd.DataFrame, side: str) -> dict[str, pd.Series]:
    if side == "bottom":
        return {"F1_ob_touch": f["ob_touch_bottom"], "F2a_smt_raw": f["smt_raw_bottom"],
                "F2b_smt_sweep_gated": f["smt_sweep_bottom"], "F3_po3_judas": f["po3_bottom"],
                "REF_plain_sweep": f["sweep_low"]}
    return {"F1_ob_touch": f["ob_touch_top"], "F2a_smt_raw": f["smt_raw_top"],
            "F2b_smt_sweep_gated": f["smt_sweep_top"], "F3_po3_judas": f["po3_top"],
            "REF_plain_sweep": f["sweep_high"]}


def run_side(f: pd.DataFrame, mask: np.ndarray, pivots: pd.DataFrame, side: str, window: str) -> pd.DataFrame:
    close = f["close"].to_numpy()
    all_pos = np.flatnonzero(mask)
    pivot_pos = f.index[f["timestamp"].isin(pivots.loc[pivots["pivot_type"] == side, "timestamp"])].to_numpy()
    rows = []
    for name, sig in components(f, side).items():
        trigger_pos = np.flatnonzero(sig.fillna(False).to_numpy() & mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            move = excess_move(trigger_pos, pivot_pos, close, K)
            rows.append({"window": window, "side": side, "signal": name, "horizon": k_name,
                         **stats, "excess_move_mean_pct": move["mean_pct"]})
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_frame()
    f = compute_indicators(raw).reset_index(drop=True)
    f = add_sweep(f)
    f = add_order_blocks(f)
    f = add_smt(f)
    f = add_po3(f)
    pivots = load_zigzag_pivots()

    ts = f["timestamp"]
    masks = {
        "POOLED": (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy(),
        "VAL": ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy(),
        "OOS": ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy(),
    }
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"pooled bars={int(masks['POOLED'].sum())}, pivots={len(pivots)}")
    for side in ("bottom", "top"):
        po3_days = f.loc[f[f"po3_{side}"].fillna(False) & masks["POOLED"], "timestamp"].dt.normalize().nunique()
        print(f"po3_{side}: distinct fire-days in window = {po3_days}")

    # mandatory overlap check vs the nearest already-scored signal (plain sweep)
    print("\nbar-level overlap vs REF_plain_sweep (pooled window):")
    for side in ("bottom", "top"):
        ref = components(f, side)["REF_plain_sweep"].fillna(False).to_numpy() & masks["POOLED"]
        for name in ("F2b_smt_sweep_gated", "F3_po3_judas", "F1_ob_touch", "F2a_smt_raw"):
            sig = components(f, side)[name].fillna(False).to_numpy() & masks["POOLED"]
            inter = (sig & ref).sum()
            print(f"  {side:<7}{name:<22} n={int(sig.sum()):>6}  overlap(sig∧sweep)/sig="
                  f"{inter / sig.sum() * 100 if sig.sum() else float('nan'):.1f}%")

    res = pd.concat([run_side(f, m, pivots, side, w) for w, m in masks.items() for side in ("bottom", "top")],
                    ignore_index=True)

    pd.set_option("display.width", 200)
    cols = ["signal", "n_triggers", "precision", "baseline_rate", "lift", "recall", "excess_move_mean_pct"]
    for side in ("bottom", "top"):
        print(f"\n\n########## {side.upper()} (POOLED) ##########")
        sub = res[(res["side"] == side) & (res["window"] == "POOLED")]
        for horizon in K_HORIZONS:
            print(f"\n-- {horizon} --")
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    print("\n\n########## VAL vs OOS consistency (K12_1h lift) ##########")
    piv = res[res["horizon"] == "K12_1h"].pivot_table(index=["side", "signal"], columns="window",
                                                      values=["lift", "n_triggers"], aggfunc="first")
    print(piv.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "ict2022_ob_smt_po3_evidence_table.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'ict2022_ob_smt_po3_evidence_table.csv'}")


if __name__ == "__main__":
    main()
