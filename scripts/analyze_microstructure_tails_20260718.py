"""Tail-conditional forward returns for the strongest microstructure signals (2026-07-18).

Follows analyze_microstructure_edge_20260718.py's causality contract (micro row ts=T usable
from decision T+2min). Questions answered here:
  1. Do extreme quantiles (1%/2%/5% tails) of the contrarian flow signals carry enough bps to
     clear realistic costs (maker ~2bps/side, taker ~4.5bps/side)?
  2. Does a simple equal-weight contrarian composite beat any single feature?
  3. How do conditional returns scale across horizons 1..30m (where does the edge saturate)?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import analyze_microstructure_edge_20260718 as base

HORIZONS = [1, 2, 3, 5, 10, 15, 30]
TAILS = [0.01, 0.02, 0.05, 0.10]


def main() -> None:
    micro = base.add_derived(base.load_micro())
    kl = pd.read_csv(base.KLINES, parse_dates=["timestamp"],
                     usecols=["timestamp", "open", "high", "low", "close"])
    kl = kl[kl["timestamp"] >= micro.index.min() - pd.Timedelta("1h")].set_index("timestamp").sort_index()
    grid = pd.DataFrame(index=kl.index)
    for h in HORIZONS:
        grid[f"fwd_{h}"] = kl["close"].shift(-(h - 1)) / kl["open"] - 1.0
    micro.index = micro.index + pd.Timedelta(minutes=base.AVAIL_SHIFT_MIN)
    df = grid.join(micro, how="inner").dropna(subset=["fwd_30"])

    # Contrarian composite: negative of expanding-window cross-feature z average (no lookahead:
    # z uses only past data via expanding mean/std with 1-day burn-in).
    comps = ["tbr_dev", "nif_whale", "nif_retail", "queue_bias_m15", "tbr_dev_m15", "obi_m15"]
    z = pd.DataFrame(index=df.index)
    for c in comps:
        mu = df[c].expanding(min_periods=1440).mean().shift(1)
        sd = df[c].expanding(min_periods=1440).std().shift(1)
        z[c] = (df[c] - mu) / sd
    df["composite"] = -z.mean(axis=1)  # positive composite => contrarian long signal

    signals = ["composite", "tbr_dev", "nif_whale", "obi_m15", "queue_bias_m15"]
    for sig in signals:
        s = df[sig] if sig == "composite" else -df[sig]  # flip raw contrarian features
        s = s.dropna()
        d = df.loc[s.index]
        print(f"\n=== {sig} (contrarian-long orientation, n={len(s):,}) ===")
        for tail in TAILS:
            lo, hi = s.quantile(tail), s.quantile(1 - tail)
            long_m, short_m = s >= hi, s <= lo
            row = [f"tail={tail:>4.0%} n_long={long_m.sum():>5}"]
            for h in [1, 3, 5, 15, 30]:
                l = d.loc[long_m, f"fwd_{h}"].mean() * 1e4
                sh = d.loc[short_m, f"fwd_{h}"].mean() * 1e4
                row.append(f"h{h:>2}: L{l:+6.2f}/S{sh:+6.2f}")
            print("  " + "  ".join(row))

    # Day-consistency for the composite 2% tail at h=5
    s = df["composite"].dropna()
    d = df.loc[s.index].copy()
    d["sig"] = s
    d["day"] = d.index.date
    daily = []
    for day, g in d.groupby("day"):
        hi, lo = g["sig"].quantile(0.98), g["sig"].quantile(0.02)
        pnl = g.loc[g["sig"] >= hi, "fwd_5"].mean() - g.loc[g["sig"] <= lo, "fwd_5"].mean()
        if np.isfinite(pnl):
            daily.append(pnl * 1e4)
    daily = np.asarray(daily)
    print(f"\ncomposite 2%-tail h=5 daily L-S spread: mean={daily.mean():+.2f}bps "
          f"t={daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))):+.2f} "
          f"pos_days={(daily > 0).mean():.0%} n={len(daily)}")


if __name__ == "__main__":
    main()
