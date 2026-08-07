#!/usr/bin/env python3
"""RESEARCH ONLY -- chart sample of triple-barrier outcomes on ETH h48qual's ACTUAL OOS entry
signals, before any relabeling/retraining work. User's point: nearly all live exits are SL/TP
(confirmed throughout this investigation -- exit_head fires on ~0-1/25 trades), yet the direction
head is trained on zigzag pivot labels, a different objective than "which barrier gets touched
first under the TP/SL the model will actually be exited by." This script visualizes what a
triple-barrier LABEL would look like for each of h48qual's real OOS entry signals, using the live
TP/SL values (min_tp=0.075, min_sl=0.040) -- the mismatch this chart is meant to make visible.

Entries come from the FROZEN h48qual OOS prediction CSV (same one used throughout this
investigation, research_eth_omega461_exit_sweep_20260721.py's COMPONENTS/load_frame) -- these are
real model signals, not a synthetic sample. For each entry bar, scans forward (causal, using only
that and later already-closed bars, matching this project's fresh-forward convention) up to a
288-bar (24h) vertical barrier, checking the bar's high/low for a TP/SL touch (same intrabar-touch
convention _apply_atr_safety_sltp/evaluate_exit use elsewhere).

Chart only -- does not retrain anything, does not touch trading_bot_modules/, trading_bot.py,
.env. Output: PNG sent to the user, not an interactive artifact (single research chart).
"""
from __future__ import annotations

import sys
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

import research_eth_omega461_exit_sweep_20260721 as base  # noqa: E402

CNAME = "h48qual"
MIN_TP = 0.075   # live value
MIN_SL = 0.040   # live value
VERTICAL_BARS = 288  # 24h at 5m bars -- label-construction cutoff, not a live runtime setting
OUT_PNG = ROOT / "tmp/research_20260728/chart_h48qual_triple_barrier_sample.png"

# Status colors: TP touch = good (teal-green), SL touch = serious (red), vertical timeout =
# neutral (gray) -- redundantly encoded with marker shape too, so outcome is never color-only.
COLOR_TP = "#1B8A5A"
COLOR_SL = "#C0392B"
COLOR_TIMEOUT = "#7F8C8D"
COLOR_PRICE = "#9AA5B1"
COLOR_ENTRY_LONG = "#2C6FBB"
COLOR_ENTRY_SHORT = "#B5651D"


def triple_barrier_outcome(frame: pd.DataFrame, entry_i: int, side: int, *, min_tp: float, min_sl: float,
                            vertical_bars: int) -> dict:
    """Causal forward scan from entry_i+1 (fresh_forward_bar_by_bar). Checks each bar's high/low
    for a TP/SL touch (same intrabar convention as evaluate_exit/_apply_atr_safety_sltp elsewhere
    in this investigation) before falling back to the vertical barrier's close-based outcome."""
    entry_price = float(frame["close"].iloc[entry_i])
    end_i = min(entry_i + vertical_bars, len(frame) - 1)
    for j in range(entry_i + 1, end_i + 1):
        high = float(frame["high"].iloc[j])
        low = float(frame["low"].iloc[j])
        if side > 0:
            move_high = (high - entry_price) / entry_price
            move_low = (low - entry_price) / entry_price
            hit_sl = move_low <= -abs(min_sl)
            hit_tp = move_high >= min_tp
        else:
            move_high = (entry_price - low) / entry_price
            move_low = (entry_price - high) / entry_price
            hit_sl = move_low <= -abs(min_sl)
            hit_tp = move_high >= min_tp
        if hit_sl and hit_tp:
            # Both touched same bar -- conservative assumption (adverse first), matching
            # evaluate_exit's own documented convention ("assuming the adverse touch first").
            return {"exit_i": j, "reason": "stop_loss", "exit_price": None}
        if hit_sl:
            return {"exit_i": j, "reason": "stop_loss", "exit_price": None}
        if hit_tp:
            return {"exit_i": j, "reason": "take_profit", "exit_price": None}
    exit_price = float(frame["close"].iloc[end_i])
    final_move = (exit_price - entry_price) / entry_price if side > 0 else (entry_price - exit_price) / entry_price
    return {"exit_i": end_i, "reason": "vertical", "exit_price": exit_price, "final_move": final_move}


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    cfg = base.COMPONENTS[CNAME]
    frame = base.load_frame(base.OOS_START, base.OOS_END, base_csv=base.BASE_2026, wide24_csv=base.WIDE24_2026)
    pred = base.EXT_PRED_DIR / CNAME / f"oos_predictions_{cfg['q_tag']}.csv"
    src = pd.read_csv(pred)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    keep_ts = set(src["timestamp"])
    frame = frame[frame["timestamp"].isin(keep_ts)].reset_index(drop=True)
    src = src[src["timestamp"].isin(set(frame["timestamp"]))].reset_index(drop=True)
    dec = base.parent._to_decisions(src, oof=False)
    side_arr = pd.to_numeric(dec["side"], errors="raise").to_numpy()

    entries = [i for i in range(len(frame) - 1) if side_arr[i] != 0]
    print(f"found {len(entries)} h48qual OOS entry signals", flush=True)

    trades = []
    for i in entries:
        side = int(side_arr[i])
        entry_i = min(i + 1, len(frame) - 1)  # execution-delay convention: fill on next bar
        outcome = triple_barrier_outcome(frame, entry_i, side, min_tp=MIN_TP, min_sl=MIN_SL, vertical_bars=VERTICAL_BARS)
        trades.append({
            "signal_i": i, "entry_i": entry_i, "side": side,
            "entry_ts": frame["timestamp"].iloc[entry_i], "entry_price": float(frame["close"].iloc[entry_i]),
            "exit_i": outcome["exit_i"], "exit_ts": frame["timestamp"].iloc[outcome["exit_i"]],
            "reason": outcome["reason"],
        })
    tdf = pd.DataFrame(trades)
    tdf.to_csv(ROOT / "tmp/research_20260728/h48qual_oos_triple_barrier_sample.csv", index=False)
    print(tdf["reason"].value_counts().to_string(), flush=True)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    ts = frame["timestamp"]
    close = frame["close"].astype(float)
    ax.plot(ts, close, color=COLOR_PRICE, linewidth=1.0, zorder=1, label="ETH close (OOS)")

    reason_color = {"take_profit": COLOR_TP, "stop_loss": COLOR_SL, "vertical": COLOR_TIMEOUT}
    reason_label = {"take_profit": "TP touched", "stop_loss": "SL touched", "vertical": f"{VERTICAL_BARS}-bar timeout"}
    seen_reasons: set[str] = set()
    seen_sides: set[int] = set()

    for _, t in tdf.iterrows():
        entry_ts, exit_ts = t["entry_ts"], t["exit_ts"]
        entry_px = t["entry_price"]
        tp_line = entry_px * (1 + MIN_TP) if t["side"] > 0 else entry_px * (1 - MIN_TP)
        sl_line = entry_px * (1 - MIN_SL) if t["side"] > 0 else entry_px * (1 + MIN_SL)
        ax.plot([entry_ts, exit_ts], [tp_line, tp_line], color=COLOR_TP, linewidth=1.0, linestyle="--", alpha=0.5, zorder=2)
        ax.plot([entry_ts, exit_ts], [sl_line, sl_line], color=COLOR_SL, linewidth=1.0, linestyle="--", alpha=0.5, zorder=2)
        ax.plot([entry_ts, exit_ts], [entry_px, entry_px], color="#B0B7BF", linewidth=0.8, linestyle=":", zorder=2)

        entry_marker = "^" if t["side"] > 0 else "v"
        entry_color = COLOR_ENTRY_LONG if t["side"] > 0 else COLOR_ENTRY_SHORT
        side_label = ("LONG entry" if t["side"] > 0 else "SHORT entry")
        ax.scatter([entry_ts], [entry_px], marker=entry_marker, s=70, color=entry_color, zorder=4,
                   edgecolor="white", linewidth=0.6, label=side_label if t["side"] not in seen_sides else None)
        seen_sides.add(t["side"])

        exit_marker = {"take_profit": "o", "stop_loss": "X", "vertical": "s"}[t["reason"]]
        ax.scatter([exit_ts], [tp_line if t["reason"] == "take_profit" else (sl_line if t["reason"] == "stop_loss" else close.iloc[t["exit_i"]])],
                   marker=exit_marker, s=60, color=reason_color[t["reason"]], zorder=4, edgecolor="white", linewidth=0.6,
                   label=reason_label[t["reason"]] if t["reason"] not in seen_reasons else None)
        seen_reasons.add(t["reason"])

    ax.set_title(f"ETH h48qual OOS entries (2026-01-01..03-31) under triple-barrier labeling\n"
                 f"min_tp={MIN_TP:.3f}, min_sl={MIN_SL:.3f} (live values), vertical={VERTICAL_BARS} bars (24h) -- "
                 f"{len(tdf)} real model entry signals, dashed lines = barrier levels", fontsize=11)
    ax.set_ylabel("ETH price (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.15, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncols=2)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"saved {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
