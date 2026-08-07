#!/usr/bin/env python3
"""RESEARCH ONLY -- chart the triple-barrier LABEL ANSWER KEY itself (not any model's entry
signals). This is what the direction head's new training target would look like if labels were
built from "which barrier gets touched first" using the live TP/SL (min_tp=0.075, min_sl=0.040)
instead of the current zigzag pivot labels.

v2 (2026-07-28), fixing two problems found in v1's chart:
  1. DIRECTIONAL BIAS from the single symmetric barrier definition. v1 used ONE pair of levels
     (upper=close*(1+tp), lower=close*(1-sl)) and called "upper touched first" LONG, "lower
     touched first" SHORT -- but tp(0.075) != sl(0.040), so in a downtrend the near SL-side level
     gets touched far more easily than the far TP-side level, producing a SHORT-label bias that
     has nothing to do with the model's actual edge, just the barrier asymmetry (observed: 34
     SHORT vs 4 LONG over the v1 sample). Fixed by evaluating LONG and SHORT as two INDEPENDENT
     hypothetical trades, each with its OWN (tp, sl) pair oriented for that side (long: tp=+0.075
     up / sl=-0.040 down; short: tp=-0.075 down / sl=+0.040 up). A bar is labeled LONG only if the
     long trade would have hit ITS OWN tp before its own sl (and the short trade did not also hit
     its tp), SHORT symmetrically, CASH if neither side's tp fires (or both fire, ambiguous -- take
     whichever resolves first in time).
  2. TOO MUCH CASH (51% in v1's 24h-vertical sample). 7.5%/4.0% barriers are wide relative to ETH
     5m volatility (atr_pct p50 ~0.0025, so a 7.5% move is roughly the scale of a full trading
     day's realized range even under diffusive assumptions) -- 24h is often not enough time for
     either side's tp to resolve. Extended the vertical barrier to 576 bars (48h), matching the
     h48qual component's own naming ("h48" = 48h horizon) rather than an arbitrary guess.

v3 (2026-07-28): SEQUENTIAL (non-overlapping) event sampling, replacing v1/v2's fixed 4h stride.
López de Prado's own triple-barrier method does not itself require non-overlapping events -- the
standard treatment tolerates overlap and corrects for it via sample-uniqueness weights at training
time. But THIS project's live system holds at most one open position at a time (every replay in
this investigation gates new entries on pos==0), so a label set built from independently-sampled,
freely-overlapping 48h windows does not match what the deployed model can ever actually act on.
Sequential sampling is the better fit for this specific system, not a paper requirement: walk
forward one bar at a time while flat; a LONG/SHORT label event consumes bars [t, exit_i] (next
candidate starts at exit_i+1, mirroring pos==0 gating); a CASH bar consumes only itself (advance
to t+1, matching "still flat, check the next bar").

Slice is the same 2-week PARENT TRAIN window (2025-01-06..01-20, inside 2024-01-01..2025-09-30).

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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

MIN_TP = 0.075   # live value
MIN_SL = 0.040   # live value
VERTICAL_BARS = 576  # 48h at 5m bars -- matches h48qual's own "h48" horizon naming (was 288/24h in v1)
WINDOW_START, WINDOW_END = "2025-01-06", "2025-01-20"
OUT_PNG = ROOT / "tmp/research_20260728/chart_triple_barrier_label_ground_truth_v3.png"

COLOR_PRICE = "#9AA5B1"
COLOR_LONG = "#2C6FBB"
COLOR_SHORT = "#B5651D"


def eval_side(frame: pd.DataFrame, t: int, side: int, *, min_tp: float, min_sl: float, vertical_bars: int) -> dict:
    """Independent single-side triple barrier: side's OWN tp/sl, oriented for that side.
    LONG: tp=+min_tp (up), sl=-min_sl (down). SHORT: tp=-min_tp (down), sl=+min_sl (up).
    Same conservative same-bar convention as evaluate_exit elsewhere ("assuming the adverse
    touch first"): if both this side's tp and sl would touch on the same bar, call it sl."""
    ref_price = float(frame["close"].iloc[t])
    tp_level = ref_price * (1.0 + min_tp) if side > 0 else ref_price * (1.0 - min_tp)
    sl_level = ref_price * (1.0 - min_sl) if side > 0 else ref_price * (1.0 + min_sl)
    end_i = min(t + vertical_bars, len(frame) - 1)
    for j in range(t + 1, end_i + 1):
        high = float(frame["high"].iloc[j])
        low = float(frame["low"].iloc[j])
        hit_tp = (high >= tp_level) if side > 0 else (low <= tp_level)
        hit_sl = (low <= sl_level) if side > 0 else (high >= sl_level)
        if hit_tp and hit_sl:
            return {"exit_i": j, "outcome": "sl"}
        if hit_sl:
            return {"exit_i": j, "outcome": "sl"}
        if hit_tp:
            return {"exit_i": j, "outcome": "tp", "level": tp_level}
    return {"exit_i": end_i, "outcome": "timeout"}


def label_event(frame: pd.DataFrame, t: int, *, min_tp: float, min_sl: float, vertical_bars: int) -> dict:
    """Independent long/short evaluation -- fixes v1's asymmetric-barrier directional bias.
    label=LONG iff the long trade hits its own tp AND the short trade does not ALSO hit its own
    tp first; symmetric for SHORT; CASH if neither side's tp fires, or if both fire (ambiguous,
    take whichever resolves earlier in time -- ties go to CASH)."""
    long_r = eval_side(frame, t, 1, min_tp=min_tp, min_sl=min_sl, vertical_bars=vertical_bars)
    short_r = eval_side(frame, t, -1, min_tp=min_tp, min_sl=min_sl, vertical_bars=vertical_bars)
    long_wins = long_r["outcome"] == "tp"
    short_wins = short_r["outcome"] == "tp"
    if long_wins and not short_wins:
        return {"exit_i": long_r["exit_i"], "label": "LONG", "reason": "long_tp_only"}
    if short_wins and not long_wins:
        return {"exit_i": short_r["exit_i"], "label": "SHORT", "reason": "short_tp_only"}
    if long_wins and short_wins:
        if long_r["exit_i"] < short_r["exit_i"]:
            return {"exit_i": long_r["exit_i"], "label": "LONG", "reason": "both_hit_long_first"}
        if short_r["exit_i"] < long_r["exit_i"]:
            return {"exit_i": short_r["exit_i"], "label": "SHORT", "reason": "both_hit_short_first"}
        return {"exit_i": long_r["exit_i"], "label": "CASH", "reason": "both_hit_same_bar_tie"}
    return {"exit_i": max(long_r["exit_i"], short_r["exit_i"]), "label": "CASH", "reason": "neither_tp_fired"}


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    train_all, _eval_df, _overlay = omega._load_omega_frames()
    train_all["timestamp"] = pd.to_datetime(train_all["timestamp"])
    frame = train_all[(train_all["timestamp"] >= WINDOW_START) & (train_all["timestamp"] <= WINDOW_END)].reset_index(drop=True)
    print(f"train window rows: {len(frame)} [{frame['timestamp'].min()}..{frame['timestamp'].max()}]", flush=True)

    # Sequential, non-overlapping walk -- mirrors pos==0 gating (at most one open "position" at a
    # time). A LONG/SHORT event consumes bars [t, exit_i]; the next candidate starts at
    # exit_i+1. A CASH bar consumes only itself (still flat -> check the next bar).
    events = []
    t = 0
    last_end = len(frame) - VERTICAL_BARS - 1
    while t < last_end:
        ev = label_event(frame, t, min_tp=MIN_TP, min_sl=MIN_SL, vertical_bars=VERTICAL_BARS)
        events.append({
            "t": t, "ts": frame["timestamp"].iloc[t], "price": float(frame["close"].iloc[t]),
            "exit_i": ev["exit_i"], "exit_ts": frame["timestamp"].iloc[ev["exit_i"]], "label": ev["label"],
        })
        t = ev["exit_i"] + 1 if ev["label"] != "CASH" else t + 1
    edf = pd.DataFrame(events)
    edf.to_csv(ROOT / "tmp/research_20260728/triple_barrier_label_ground_truth_sample_v3.csv", index=False)
    print(edf["label"].value_counts().to_string(), flush=True)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    ax.plot(frame["timestamp"], frame["close"].astype(float), color=COLOR_PRICE, linewidth=1.0, zorder=1, label="ETH close (TRAIN)")

    seen = set()
    for _, e in edf.iterrows():
        if e["label"] == "CASH":
            continue
        color = COLOR_LONG if e["label"] == "LONG" else COLOR_SHORT
        # Each label's OWN tp level (long: +min_tp up, short: -min_tp down) -- matches
        # label_event's independent per-side evaluation, not v1's shared upper/lower pair.
        target_line = e["price"] * (1 + MIN_TP) if e["label"] == "LONG" else e["price"] * (1 - MIN_TP)
        ax.plot([e["ts"], e["exit_ts"]], [e["price"], e["price"]], color="#B0B7BF", linewidth=0.8, linestyle=":", zorder=2)
        ax.plot([e["ts"], e["exit_ts"]], [target_line, target_line], color=color, linewidth=1.2, linestyle="--", alpha=0.6, zorder=2)
        ax.scatter([e["ts"]], [e["price"]], marker="|", s=90, color=color, zorder=4, linewidth=1.8,
                   label=f"{e['label']} label" if e["label"] not in seen else None)
        ax.scatter([e["exit_ts"]], [target_line], marker="o", s=35, color=color, zorder=4, edgecolor="white", linewidth=0.5)
        seen.add(e["label"])

    n_long = int((edf["label"] == "LONG").sum())
    n_short = int((edf["label"] == "SHORT").sum())
    n_cash = int((edf["label"] == "CASH").sum())
    ax.set_title(f"Triple-barrier LABEL ANSWER KEY v3 (sequential, non-overlapping) -- TRAIN {WINDOW_START}..{WINDOW_END}\n"
                 f"min_tp={MIN_TP:.3f}, min_sl={MIN_SL:.3f} (live values), vertical={VERTICAL_BARS} bars (48h), "
                 f"each event starts only after the previous one resolves (pos==0 gating) -- "
                 f"LONG={n_long} SHORT={n_short} CASH={n_cash} (not charted, CASH bars still consume 1 bar each)",
                 fontsize=11)
    ax.set_ylabel("ETH price (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    ax.grid(True, alpha=0.15, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"saved {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
