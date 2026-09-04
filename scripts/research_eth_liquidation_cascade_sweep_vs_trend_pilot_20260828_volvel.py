#!/usr/bin/env python3
"""Two feature refinements to the two surviving axes (wick_body_ratio, nif_whale_rel), per user
request:
  (A) Volume-weighted wick: does volume concentrate AT the cascade's price extreme (1-min sub-bar
      resolution within the 5-min cascade bar) -- a volume climax right at the wick tip, not just a
      long wick, should raise confidence in a sweep/absorption read.
  (B) Whale flow velocity: not just nif_whale's net DIRECTION, but the ARRIVAL RATE of whale-sized
      trades (microstructure_1m.recent_whale_count_5m, trade-based so inherently taker/aggressive
      flow already) right after the cascade vs its pre-cascade baseline -- a burst of aggressive
      whale activity, not just calm net positioning, is the reversal hypothesis being tested.

Same discipline as every other axis in this pilot: chronological dev(70%)/holdout(30%) split
(identical split point to the rest of the pilot), thresholds chosen on dev only, confirmed once on
holdout, never re-adjusted.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "research" / "eth_liquidation_cascade_sweep_vs_trend_pilot_20260828"

WINDOW_START_UTC = pd.Timestamp("2026-07-18 12:00:00", tz="UTC")
FEATURE_WINDOW_MINUTES = 15


def fetch_1m_klines(start_ms: int, end_ms: int) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    out = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get("https://fapi.binance.com/fapi/v1/klines",
                             params={"symbol": "ETHUSDT", "interval": "1m", "startTime": cursor,
                                     "endTime": end_ms, "limit": 1500}, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        out.extend(batch)
        last_open = int(batch[-1][0])
        if last_open <= cursor:
            break
        cursor = last_open + 1
        if len(batch) < 1500:
            break
        time.sleep(0.12)
    df = pd.DataFrame(out, columns=cols)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def load_events() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "labeled_features_definition_a.csv", parse_dates=["t0"])
    df["genuine_breach"] = (
        ((df["direction"] == "down") & (df["cascade_extreme"] < df["swept_level"]))
        | ((df["direction"] == "up") & (df["cascade_extreme"] > df["swept_level"]))
    )
    sub = df[df["genuine_breach"] & df["label_1h"].isin(["sweep", "continuation"])].copy()
    return sub.sort_values("t0").reset_index(drop=True)


def add_volume_weighted_wick(events: pd.DataFrame, kl_1m: pd.DataFrame) -> pd.DataFrame:
    kl_1m = kl_1m.sort_values("timestamp").reset_index(drop=True)
    rows = []
    for ev in events.itertuples():
        # the cascade's OWN 5-min bar spans [t0_5m_open, t0_5m_open + 5min); t0 is the hawkes
        # onset minute, which the pilot's label_events() already snapped to the containing 5m bar
        # via searchsorted -- reproduce that same bar here directly from the 1m data.
        bar_open = ev.t0.floor("5min")
        sub = kl_1m[(kl_1m["timestamp"] >= bar_open) & (kl_1m["timestamp"] < bar_open + pd.Timedelta(minutes=5))]
        if len(sub) < 3:  # need most of the 5 sub-bars present to trust this
            rows.append({"event_id": ev.event_id, "extreme_subbar_volume_share": np.nan})
            continue
        total_vol = sub["volume"].sum()
        if total_vol <= 0:
            rows.append({"event_id": ev.event_id, "extreme_subbar_volume_share": np.nan})
            continue
        if ev.direction == "down":
            extreme_idx = sub["low"].idxmin()
        else:
            extreme_idx = sub["high"].idxmax()
        extreme_vol = sub.loc[extreme_idx, "volume"]
        rows.append({"event_id": ev.event_id, "extreme_subbar_volume_share": float(extreme_vol / total_vol)})
    return events.merge(pd.DataFrame(rows), on="event_id", how="left")


def add_whale_velocity(events: pd.DataFrame, micro: pd.DataFrame) -> pd.DataFrame:
    micro = micro.sort_values("ts").reset_index(drop=True)
    rows = []
    for ev in events.itertuples():
        pre = micro[micro["ts"] <= ev.t0]
        baseline = pre["recent_whale_count_5m"].iloc[-1] if len(pre) else np.nan
        post_win = micro[(micro["ts"] > ev.t0) & (micro["ts"] <= ev.t0 + pd.Timedelta(minutes=FEATURE_WINDOW_MINUTES))]
        post_max = post_win["recent_whale_count_5m"].max() if len(post_win) else np.nan
        ratio = (post_max / max(baseline, 1.0)) if pd.notna(baseline) and pd.notna(post_max) else np.nan
        rows.append({"event_id": ev.event_id, "whale_count_baseline": baseline,
                     "whale_count_post_max": post_max, "whale_velocity_ratio": ratio})
    return events.merge(pd.DataFrame(rows), on="event_id", how="left")


def precision_recall(d: pd.DataFrame, mask, cls: str):
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
    start_ms = int(WINDOW_START_UTC.timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    print("fetching 1m klines (needed for intrabar volume-at-extreme)...")
    kl_1m = fetch_1m_klines(start_ms, end_ms)
    print(f"  {len(kl_1m)} 1m bars, {kl_1m['timestamp'].min()} -> {kl_1m['timestamp'].max()}")
    kl_1m.to_csv(DATA_DIR / "futures_1m_klines.csv", index=False)

    micro = pd.read_csv(DATA_DIR / "microstructure_1m.csv", parse_dates=["ts"])
    micro["ts"] = pd.to_datetime(micro["ts"], utc=True)

    full = add_volume_weighted_wick(events, kl_1m)
    full = add_whale_velocity(full, micro)
    full = full.sort_values("t0").reset_index(drop=True)
    full.to_csv(DATA_DIR / "events_with_volvel.csv", index=False)

    print(f"\nextreme_subbar_volume_share coverage: {full['extreme_subbar_volume_share'].notna().sum()}/{len(full)}")
    print(f"whale_velocity_ratio coverage: {full['whale_velocity_ratio'].notna().sum()}/{len(full)}")

    split = int(len(full) * 0.7)
    dev, hold = full.iloc[:split], full.iloc[split:]
    print(f"\ndev n={len(dev)}  holdout n={len(hold)}")

    print("\n=== extreme_subbar_volume_share by label (dev) ===")
    d = dev.dropna(subset=["extreme_subbar_volume_share"])
    print(d.groupby("label_1h")["extreme_subbar_volume_share"].agg(["mean", "median", "count"]))

    print("\n=== whale_velocity_ratio by label (dev) ===")
    d2 = dev.dropna(subset=["whale_velocity_ratio"])
    print(d2.groupby("label_1h")["whale_velocity_ratio"].agg(["mean", "median", "count"]))

    print("\n=== dev-only threshold screen (natural median split, both directions) ===")
    for col in ["extreme_subbar_volume_share", "whale_velocity_ratio"]:
        d = dev.dropna(subset=[col])
        med = d[col].median()
        for cls, higher_is in [("sweep", True), ("continuation", False)]:
            mask = (d[col] > med) if higher_is else (d[col] <= med)
            p, r, n = precision_recall(d, mask, cls)
            print(f"  {col} {'>' if higher_is else '<='} dev-median({med:.4f}) -> {cls}: dev={p:.1%}(n={n})")

    print("\n=== HOLDOUT confirmation of the two natural hypotheses (median threshold fixed from dev) ===")
    # A's hypothesis: high volume concentration at the extreme -> sweep (climax-and-reverse)
    d = dev.dropna(subset=["extreme_subbar_volume_share"])
    med_a = d["extreme_subbar_volume_share"].median()
    h = hold.dropna(subset=["extreme_subbar_volume_share"])
    p, r, n = precision_recall(h, h["extreme_subbar_volume_share"] > med_a, "sweep")
    print(f"  extreme_subbar_volume_share>{med_a:.4f} -> sweep: HOLDOUT={p:.1%}(n={n}, base={((h['label_1h']=='sweep').mean()):.1%})")

    # B's hypothesis: high whale arrival-rate spike -> sweep (aggressive counter-entry)
    d2 = dev.dropna(subset=["whale_velocity_ratio"])
    med_b = d2["whale_velocity_ratio"].median()
    h2 = hold.dropna(subset=["whale_velocity_ratio"])
    p, r, n = precision_recall(h2, h2["whale_velocity_ratio"] > med_b, "sweep")
    print(f"  whale_velocity_ratio>{med_b:.4f} -> sweep: HOLDOUT={p:.1%}(n={n}, base={((h2['label_1h']=='sweep').mean()):.1%})")

    print("\n=== combined with existing wick_body_ratio>2.0 sweep rule ===")
    dv = dev.dropna(subset=["wick_body_ratio", "extreme_subbar_volume_share", "whale_velocity_ratio"])
    hv = hold.dropna(subset=["wick_body_ratio", "extreme_subbar_volume_share", "whale_velocity_ratio"])
    med_a2 = dv["extreme_subbar_volume_share"].median()
    med_b2 = dv["whale_velocity_ratio"].median()
    for name, cond_dev, cond_hold in [
        ("wick>2.0 alone (baseline, same rows)", dv["wick_body_ratio"] > 2.0, hv["wick_body_ratio"] > 2.0),
        ("wick>2.0 & extreme_vol_share>median", (dv["wick_body_ratio"] > 2.0) & (dv["extreme_subbar_volume_share"] > med_a2),
         (hv["wick_body_ratio"] > 2.0) & (hv["extreme_subbar_volume_share"] > med_a2)),
        ("wick>2.0 & whale_velocity>median", (dv["wick_body_ratio"] > 2.0) & (dv["whale_velocity_ratio"] > med_b2),
         (hv["wick_body_ratio"] > 2.0) & (hv["whale_velocity_ratio"] > med_b2)),
    ]:
        p_dev, _, n_dev = precision_recall(dv, cond_dev, "sweep")
        p_hold, _, n_hold = precision_recall(hv, cond_hold, "sweep")
        print(f"  {name}: dev={p_dev:.1%}(n={n_dev})  HOLDOUT={p_hold:.1%}(n={n_hold})")


if __name__ == "__main__":
    main()
