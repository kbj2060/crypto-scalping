#!/usr/bin/env python3
"""Combine with the live liquidation-map S/R estimate (scripts/live_liquidation_map_20260824.py::
compute_spliced_levels(), currently deployed -- v1_spliced per eth_liquidation_map_spliced_hybrid_
confirmed_20260826). That function is a pure function of a 24-hourly-bar trailing window + current
price (dashboard/server.py::LIQUIDATION_MAP_LOOKBACK_HOURS=24, LIQUIDATION_MAP_INTERVAL="1h") -- no
live event feed, so it can be causally replayed for any historical t0 using resampled klines this
pilot already fetched. Reuses the real function, not a reimplementation.

Hypothesis: a cascade that breaks a level the liq-map ALSO flags (i.e. real estimated liquidation
density was actually stacked there) behaves differently than one that breaks a level with no liq-map
backing (more likely a thin/noise level -- fakeout-prone).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data" / "research" / "eth_liquidation_cascade_sweep_vs_trend_pilot_20260828"

from scripts.live_liquidation_map_20260824 import compute_spliced_levels  # noqa: E402

LOOKBACK_BARS_1H = 24  # matches dashboard/server.py::LIQUIDATION_MAP_LOOKBACK_HOURS
LEVEL_MATCH_TOL_PCT = 0.01  # 1% -- how close swept_level must be to a liq-map level to count as "confirmed"


def load_events() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "labeled_features_definition_a.csv", parse_dates=["t0"])
    df["genuine_breach"] = (
        ((df["direction"] == "down") & (df["cascade_extreme"] < df["swept_level"]))
        | ((df["direction"] == "up") & (df["cascade_extreme"] > df["swept_level"]))
    )
    sub = df[df["genuine_breach"] & df["label_1h"].isin(["sweep", "continuation"])].copy()
    return sub.sort_values("t0").reset_index(drop=True)


def resample_1h(kl_5m: pd.DataFrame) -> pd.DataFrame:
    d = kl_5m.set_index("timestamp")
    out = d.resample("1h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna().reset_index()


def liq_map_features(events: pd.DataFrame, kl_1h: pd.DataFrame) -> pd.DataFrame:
    kl_1h = kl_1h.sort_values("timestamp").reset_index(drop=True)
    rows = []
    for ev in events.itertuples():
        window = kl_1h[kl_1h["timestamp"] <= ev.t0].tail(LOOKBACK_BARS_1H)
        current_price = float(ev.cascade_extreme)  # price at the cascade itself, not a later close
        res = compute_spliced_levels(window, current_price) if len(window) >= 20 else {"warmed_up": False}
        row = {"event_id": ev.event_id, "liqmap_warmed_up": res.get("warmed_up", False),
               "liqmap_bars_used": res.get("bars_used")}
        if res.get("warmed_up"):
            side_levels = res["support_levels"] if ev.direction == "down" else res["resistance_levels"]
            if side_levels:
                nearest = min(side_levels, key=lambda lv: abs(lv["price"] - ev.swept_level))
                dist_pct = abs(nearest["price"] - ev.swept_level) / ev.swept_level
                row["liq_level_confirmed"] = bool(dist_pct <= LEVEL_MATCH_TOL_PCT)
                row["liq_level_weight"] = nearest["weight_pct"]
                row["liq_level_distance_to_swept_pct"] = dist_pct
                row["n_side_levels"] = len(side_levels)
            else:
                row["liq_level_confirmed"] = False
                row["liq_level_weight"] = 0.0
                row["liq_level_distance_to_swept_pct"] = np.nan
                row["n_side_levels"] = 0
        rows.append(row)
    return events.merge(pd.DataFrame(rows), on="event_id", how="left")


def precision_recall(d: pd.DataFrame, mask: np.ndarray, cls: str):
    pred = np.where(mask, cls, "other")
    actual = d["label_1h"].to_numpy()
    n_pred = int((pred == cls).sum())
    tp = int(((pred == cls) & (actual == cls)).sum())
    n_actual = int((actual == cls).sum())
    prec = tp / n_pred if n_pred else float("nan")
    rec = tp / n_actual if n_actual else float("nan")
    return prec, rec, n_pred


def main() -> None:
    events = load_events()
    kl_5m = pd.read_csv(DATA_DIR / "futures_5m_klines.csv", parse_dates=["timestamp"])
    kl_1h = resample_1h(kl_5m)
    print(f"1h klines: {len(kl_1h)} bars, {kl_1h['timestamp'].min()} -> {kl_1h['timestamp'].max()}")

    full = liq_map_features(events, kl_1h)
    print(f"liqmap warmed_up: {full['liqmap_warmed_up'].sum()}/{len(full)}")
    print(f"liq_level_confirmed rate: {full['liq_level_confirmed'].mean():.1%} "
          f"({full['liq_level_confirmed'].sum()}/{full['liq_level_confirmed'].notna().sum()})")
    print()
    print("liq_level_confirmed vs label_1h:")
    print(pd.crosstab(full["liq_level_confirmed"], full["label_1h"]))
    print()

    micro = pd.read_csv(DATA_DIR / "microstructure_1m.csv", parse_dates=["ts"])
    micro["ts"] = pd.to_datetime(micro["ts"], utc=True)
    micro = micro.sort_values("ts").reset_index(drop=True)
    rows = []
    for ev in events.itertuples():
        win = micro[(micro["ts"] > ev.t0) & (micro["ts"] <= ev.t0 + pd.Timedelta(minutes=15))]
        rows.append({"event_id": ev.event_id, "nif_whale": win["nif_whale"].mean() if len(win) else np.nan})
    full = full.merge(pd.DataFrame(rows), on="event_id", how="left")
    full["nif_whale_rel"] = np.where(full["direction"] == "down", full["nif_whale"], -full["nif_whale"])

    full = full.sort_values("t0").reset_index(drop=True)
    full.to_csv(DATA_DIR / "events_with_liqmap.csv", index=False)
    split = int(len(full) * 0.7)
    dev, hold = full.iloc[:split], full.iloc[split:]
    print(f"dev n={len(dev)}  holdout n={len(hold)}")
    base_dev = dict(dev["label_1h"].value_counts(normalize=True).round(3))
    print(f"dev base rate: {base_dev}\n")

    print("=== standalone: liq_level_confirmed (dev) ===")
    for cls, want_confirmed in [("continuation", True), ("sweep", False)]:
        d = dev.dropna(subset=["liq_level_confirmed"])
        mask = d["liq_level_confirmed"] if want_confirmed else ~d["liq_level_confirmed"]
        p, r, n = precision_recall(d, mask, cls)
        print(f"  liq_level_confirmed={want_confirmed} -> {cls}: dev={p:.1%}(n={n})")

    print("\n=== standalone: liq_level_weight (dev, threshold=dev-median among confirmed) ===")
    d = dev.dropna(subset=["liq_level_weight"])
    med = d.loc[d["liq_level_confirmed"], "liq_level_weight"].median() if d["liq_level_confirmed"].any() else np.nan
    print(f"  dev-median weight among confirmed levels: {med}")

    print("\n=== combined with wick_body_ratio + nif_whale_rel (dev, then holdout once) ===")
    combos = [
        ("wick<0.5 & nif_whale_rel<=0 & liq_level_confirmed -> continuation",
         lambda d: (d['wick_body_ratio'] < 0.5) & (d['nif_whale_rel'] <= 0) & (d['liq_level_confirmed'] == True), 'continuation'),
        ("wick<0.5 & nif_whale_rel<=0 & NOT liq_level_confirmed -> continuation",
         lambda d: (d['wick_body_ratio'] < 0.5) & (d['nif_whale_rel'] <= 0) & (d['liq_level_confirmed'] == False), 'continuation'),
        ("wick>2.0 & liq_level_confirmed -> sweep",
         lambda d: (d['wick_body_ratio'] > 2.0) & (d['liq_level_confirmed'] == True), 'sweep'),
        ("wick>2.0 & NOT liq_level_confirmed -> sweep",
         lambda d: (d['wick_body_ratio'] > 2.0) & (d['liq_level_confirmed'] == False), 'sweep'),
        ("liq_level_confirmed alone -> continuation",
         lambda d: d['liq_level_confirmed'] == True, 'continuation'),
        ("NOT liq_level_confirmed alone -> sweep",
         lambda d: d['liq_level_confirmed'] == False, 'sweep'),
    ]
    for name, fn, cls in combos:
        dv = dev.dropna(subset=['wick_body_ratio', 'nif_whale_rel', 'liq_level_confirmed'])
        hv = hold.dropna(subset=['wick_body_ratio', 'nif_whale_rel', 'liq_level_confirmed'])
        p_dev, _, n_dev = precision_recall(dv, fn(dv), cls)
        p_hold, _, n_hold = precision_recall(hv, fn(hv), cls)
        print(f"{name}\n    dev={p_dev:.1%}(n={n_dev})  HOLDOUT={p_hold:.1%}(n={n_hold})")


if __name__ == "__main__":
    main()
