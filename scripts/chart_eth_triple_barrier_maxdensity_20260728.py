#!/usr/bin/env python3
"""RESEARCH ONLY -- v5: shrink the LABEL barrier (not the live execution barrier) and shorten the
vertical window to maximize trade count, per explicit user instruction ("SLTP를 크게 줄이고 시간을
너무 많이 주지 마, 거래를 최대한 늘려서 학습 데이터를 많이 만들어줘").

Density sweep this session (3-month window, extrapolated to full ~638d TRAIN, all via
scripts/chart_eth_triple_barrier_label_ground_truth_20260728.py's label_event, sequential/
non-overlapping):
    live (0.075/0.040, 48h)        ->    326 trades
    0.025/0.013, 8h                 ->  2,371
    0.020/0.011, 6h                 ->  3,422
    0.015/0.008, 4h                 ->  5,655
    0.012/0.0065, 3h                ->  8,084
    0.010/0.0055, 2h                -> 10,774
    0.008/0.0045, 2h                -> 14,913
    0.006/0.0032, 1h                -> 21,569
    0.005/0.0027, 1h                -> 27,079
    0.004/0.0022, 1h                -> 34,140

Picked TP=0.006 (0.6%), SL=0.0032 (0.32%), vertical=12 bars (1h) -- keeps the live ~1.875:1
tp:sl ratio, ~66x the live-barrier trade count (326 -> ~21,569 extrapolated). NOT the most
extreme tested point: round-trip cost in this project's harness is fee+slip ~0.07%/side ~=0.14%
round-trip (train_eval_omega1_2_tabm_diffusion_risk_20260603._load_fee_slip), so a 0.6% TP is
still ~4.3x round-trip cost -- the two smaller configs tested (0.005/0.004 TP) get close enough
to the cost floor that many "wins" would be cost-noise rather than real directional signal, even
though this label is for TRAINING the direction head, not for live execution (live still uses the
real 7.5%/4.0% barrier for position sizing/exits; this label only teaches the direction head "was
there a short-term directional tendency here").

Chart only -- generates no training data file, retrains nothing, touches no live file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path("/home/llewyn/crypto-scalping")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import chart_eth_triple_barrier_label_ground_truth_20260728 as lbl  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

MIN_TP = 0.006
MIN_SL = 0.0032
VERTICAL_BARS = 12  # 1h
WINDOW_START, WINDOW_END = "2025-01-06", "2025-01-20"  # same 2-week slice as v1-v3, for direct comparison
OUT_PNG = ROOT / "tmp/research_20260728/chart_triple_barrier_maxdensity_v5.png"

COLOR_PRICE, COLOR_LONG, COLOR_SHORT = lbl.COLOR_PRICE, lbl.COLOR_LONG, lbl.COLOR_SHORT


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    train_all, _eval_df, _overlay = omega._load_omega_frames()
    train_all["timestamp"] = pd.to_datetime(train_all["timestamp"])
    frame = train_all[(train_all["timestamp"] >= WINDOW_START) & (train_all["timestamp"] <= WINDOW_END)].reset_index(drop=True)
    print(f"train window rows: {len(frame)} [{frame['timestamp'].min()}..{frame['timestamp'].max()}]", flush=True)

    events = []
    t = 0
    last_end = len(frame) - VERTICAL_BARS - 1
    while t < last_end:
        ev = lbl.label_event(frame, t, min_tp=MIN_TP, min_sl=MIN_SL, vertical_bars=VERTICAL_BARS)
        events.append({
            "t": t, "ts": frame["timestamp"].iloc[t], "price": float(frame["close"].iloc[t]),
            "exit_i": ev["exit_i"], "exit_ts": frame["timestamp"].iloc[ev["exit_i"]], "label": ev["label"],
        })
        t = ev["exit_i"] + 1 if ev["label"] != "CASH" else t + 1
    edf = pd.DataFrame(events)
    edf.to_csv(ROOT / "tmp/research_20260728/triple_barrier_maxdensity_sample_v5.csv", index=False)
    print(edf["label"].value_counts().to_string(), flush=True)

    n_long = int((edf["label"] == "LONG").sum())
    n_short = int((edf["label"] == "SHORT").sum())
    n_cash = int((edf["label"] == "CASH").sum())
    n_trades = n_long + n_short

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    ax.plot(frame["timestamp"], frame["close"].astype(float), color=COLOR_PRICE, linewidth=1.0, zorder=1, label="ETH close (TRAIN)")

    long_df = edf[edf["label"] == "LONG"]
    short_df = edf[edf["label"] == "SHORT"]
    # Too many trades to bracket individually (unlike v1-v3) -- plot entry points only, colored
    # by label, with light alpha so density itself is visible as a shaded band along the price line.
    ax.scatter(long_df["ts"], long_df["price"], marker="^", s=10, color=COLOR_LONG, alpha=0.45, zorder=3, linewidth=0, label=f"LONG entry (n={n_long})")
    ax.scatter(short_df["ts"], short_df["price"], marker="v", s=10, color=COLOR_SHORT, alpha=0.45, zorder=3, linewidth=0, label=f"SHORT entry (n={n_short})")

    ax.set_title(f"Triple-barrier LABEL ANSWER KEY v5 (max density) -- TRAIN {WINDOW_START}..{WINDOW_END}\n"
                 f"min_tp={MIN_TP:.4f} ({MIN_TP*100:.2f}%), min_sl={MIN_SL:.4f} ({MIN_SL*100:.2f}%), vertical={VERTICAL_BARS} bars (1h), sequential -- "
                 f"LONG={n_long} SHORT={n_short} CASH={n_cash} (this window: {n_trades} trades / {len(edf)} events, "
                 f"{n_trades/max(len(edf),1):.1%} hit rate)", fontsize=10.5)
    ax.set_ylabel("ETH price (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.15, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, markerscale=2.5)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"saved {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
