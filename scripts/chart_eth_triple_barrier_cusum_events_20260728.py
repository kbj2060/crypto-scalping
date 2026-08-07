#!/usr/bin/env python3
"""RESEARCH ONLY -- v4: CUSUM-filtered event sampling + triple-barrier labeling, the
literature-standard alternative to raising trade density by shrinking the barrier (which the
user explicitly rejected). Reuses v3's label_event/eval_side (live barrier 0.075/0.040,
independent long/short evaluation, 48h vertical) UNCHANGED -- only WHEN a candidate label event
starts changes.

Symmetric CUSUM filter (Lopez de Prado AFML 2.5.2.1, and the 2025 Financial Innovation paper
combining CUSUM-filtered events with triple-barrier labeling for crypto -- see chat sources):
    S_pos[t] = max(0, S_pos[t-1] + r[t])
    S_neg[t] = min(0, S_neg[t-1] + r[t])
    event when S_pos[t] >= h[t] (reset S_pos=0) or S_neg[t] <= -h[t] (reset S_neg=0)
r[t] = simple bar return. h[t] = k * atr_pct[t] (adaptive threshold via this project's own ATR
calc, eval_omega4_1_atr_safety_sltp_20260622._atr_pct, rather than a fresh rolling-std -- reuses
an existing, already-validated volatility measure instead of introducing a new one).

CUSUM only decides WHEN to test the barrier (a moment of already-accumulating momentum); the
label itself is still decided by the SAME independent long/short triple-barrier evaluation as v3
-- CUSUM does not bias which side wins. Sequential/non-overlapping (position-gating): after a
CUSUM-triggered event resolves (LONG/SHORT/CASH) at exit_i, the CUSUM accumulators reset and
scanning resumes from exit_i+1, mirroring "the model can't watch for a new momentum breakout
while already in a trade."

Chart only -- generates no training data file, retrains nothing, touches no live file.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path("/home/llewyn/crypto-scalping")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import chart_eth_triple_barrier_label_ground_truth_20260728 as lbl  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

MIN_TP, MIN_SL, VERTICAL_BARS = lbl.MIN_TP, lbl.MIN_SL, lbl.VERTICAL_BARS
ATR_WINDOW = 192  # matches live's atr_window default
K_THRESHOLD = 3.0  # h[t] = K_THRESHOLD * atr_pct[t]
WINDOW_START, WINDOW_END = lbl.WINDOW_START, lbl.WINDOW_END
OUT_PNG = ROOT / "tmp/research_20260728/chart_triple_barrier_cusum_events_v4.png"

COLOR_PRICE, COLOR_LONG, COLOR_SHORT = lbl.COLOR_PRICE, lbl.COLOR_LONG, lbl.COLOR_SHORT
COLOR_CUSUM_UP, COLOR_CUSUM_DOWN = "#5B8DEF", "#E08E45"


def cusum_events(frame: pd.DataFrame, *, k: float, atr_window: int) -> list[dict]:
    close = frame["close"].astype(float).to_numpy()
    ret = np.zeros(len(close))
    ret[1:] = (close[1:] - close[:-1]) / close[:-1]
    atr_pct = atr_eval._atr_pct(frame, atr_window)

    events = []
    s_pos = s_neg = 0.0
    t = 1
    next_allowed = 1
    last_end = len(frame) - VERTICAL_BARS - 1
    while t < last_end:
        if t < next_allowed:
            t += 1
            continue
        h = k * float(atr_pct[t])
        s_pos = max(0.0, s_pos + ret[t])
        s_neg = min(0.0, s_neg + ret[t])
        direction = None
        if s_pos >= h:
            direction = "up"
            s_pos = 0.0
        elif s_neg <= -h:
            direction = "down"
            s_neg = 0.0
        if direction is not None:
            ev = lbl.label_event(frame, t, min_tp=MIN_TP, min_sl=MIN_SL, vertical_bars=VERTICAL_BARS)
            events.append({
                "t": t, "ts": frame["timestamp"].iloc[t], "price": float(frame["close"].iloc[t]),
                "cusum_direction": direction, "exit_i": ev["exit_i"],
                "exit_ts": frame["timestamp"].iloc[ev["exit_i"]], "label": ev["label"],
            })
            # Position-gating only applies to a REAL trade (LONG/SHORT) -- a CASH/timeout result
            # never opened a position, so there is nothing to wait out; only its own trigger bar
            # is consumed and CUSUM can retrigger immediately on the next bar. (Bug fix: v4's
            # first pass locked out the full 48h vertical window even on CASH outcomes, which
            # saturated total_events at roughly window_length/vertical_bars regardless of k --
            # that is why raising/lowering k barely moved the count in the first test.)
            next_allowed = ev["exit_i"] + 1 if ev["label"] != "CASH" else t + 1
            s_pos = s_neg = 0.0
        t += 1
    return events


def density_check() -> None:
    """Same 3-month comparison window used for the sequential (non-CUSUM) baseline, so the
    density improvement is measured on an apples-to-apples slice."""
    train_all, _eval_df, _overlay = omega._load_omega_frames()
    train_all["timestamp"] = pd.to_datetime(train_all["timestamp"])
    frame = train_all[(train_all["timestamp"] >= "2025-01-01") & (train_all["timestamp"] <= "2025-03-31")].reset_index(drop=True)
    window_days = (frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]).days
    full_train_days = (pd.Timestamp("2025-09-30") - pd.Timestamp("2024-01-01")).days
    for k in (2.0, 3.0, 4.0):
        t0 = time.time()
        evs = cusum_events(frame, k=k, atr_window=ATR_WINDOW)
        labels = [e["label"] for e in evs]
        trades = sum(1 for l in labels if l != "CASH")
        n_long = sum(1 for l in labels if l == "LONG")
        n_short = sum(1 for l in labels if l == "SHORT")
        scale = full_train_days / max(window_days, 1)
        print(f"k={k}: total_events={len(evs)} trades={trades} (LONG={n_long} SHORT={n_short}) "
              f"cash={len(evs)-trades} over {window_days}d -> extrapolated full TRAIN: ~{trades*scale:.0f} trades "
              f"[{time.time()-t0:.1f}s]", flush=True)


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    print("=== density check (3-month window, comparable to the sequential/non-CUSUM baseline) ===", flush=True)
    density_check()

    print(f"\n=== chart (2-week window {WINDOW_START}..{WINDOW_END}, k={K_THRESHOLD}) ===", flush=True)
    train_all, _eval_df, _overlay = omega._load_omega_frames()
    train_all["timestamp"] = pd.to_datetime(train_all["timestamp"])
    frame = train_all[(train_all["timestamp"] >= WINDOW_START) & (train_all["timestamp"] <= WINDOW_END)].reset_index(drop=True)
    events = cusum_events(frame, k=K_THRESHOLD, atr_window=ATR_WINDOW)
    edf = pd.DataFrame(events)
    edf.to_csv(ROOT / "tmp/research_20260728/triple_barrier_cusum_events_sample_v4.csv", index=False)
    print(edf["label"].value_counts().to_string(), flush=True)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    ax.plot(frame["timestamp"], frame["close"].astype(float), color=COLOR_PRICE, linewidth=1.0, zorder=1, label="ETH close (TRAIN)")

    seen = set()
    for _, e in edf.iterrows():
        cusum_color = COLOR_CUSUM_UP if e["cusum_direction"] == "up" else COLOR_CUSUM_DOWN
        ax.scatter([e["ts"]], [e["price"]], marker="d", s=25, color=cusum_color, zorder=3, alpha=0.9,
                   label=f"CUSUM trigger ({e['cusum_direction']})" if f"cusum_{e['cusum_direction']}" not in seen else None)
        seen.add(f"cusum_{e['cusum_direction']}")
        if e["label"] == "CASH":
            continue
        color = COLOR_LONG if e["label"] == "LONG" else COLOR_SHORT
        target_line = e["price"] * (1 + MIN_TP) if e["label"] == "LONG" else e["price"] * (1 - MIN_TP)
        ax.plot([e["ts"], e["exit_ts"]], [e["price"], e["price"]], color="#B0B7BF", linewidth=0.8, linestyle=":", zorder=2)
        ax.plot([e["ts"], e["exit_ts"]], [target_line, target_line], color=color, linewidth=1.2, linestyle="--", alpha=0.6, zorder=2)
        ax.scatter([e["exit_ts"]], [target_line], marker=("o" if e["label"] == "LONG" else "X"), s=60, color=color,
                   zorder=4, edgecolor="white", linewidth=0.6, label=f"{e['label']} label" if e["label"] not in seen else None)
        seen.add(e["label"])

    n_long = int((edf["label"] == "LONG").sum())
    n_short = int((edf["label"] == "SHORT").sum())
    n_cash = int((edf["label"] == "CASH").sum())
    ax.set_title(f"Triple-barrier LABEL ANSWER KEY v4 (CUSUM-filtered event sampling) -- TRAIN {WINDOW_START}..{WINDOW_END}\n"
                 f"h[t]={K_THRESHOLD:g}*atr_pct[t], min_tp={MIN_TP:.3f}, min_sl={MIN_SL:.3f} (live, unchanged), vertical=48h, sequential -- "
                 f"LONG={n_long} SHORT={n_short} CASH={n_cash}", fontsize=11)
    ax.set_ylabel("ETH price (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    ax.grid(True, alpha=0.15, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncols=2)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"saved {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
