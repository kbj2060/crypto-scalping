#!/usr/bin/env python3
"""RESEARCH ONLY -- builds the full-TRAIN-period triple-barrier direction label contract
(v5 max-density config: min_tp=0.006, min_sl=0.0032, vertical=12 bars/1h, independent long/short
evaluation, sequential/non-overlapping), in the same file format the trainer's `_read_labels`
expects (zigzag_action_labels_{year}.csv, columns timestamp + zigzag_action in {0,1,2} =
{CASH,LONG,SHORT}) so it's a drop-in --direction-label-dir replacement for zigzag_action_labels.

Source frame: the recovered 2024+2025 tape from round 19
(scripts/train_eval_omega4_3head_parent72_pinned102_2024tape_20260727.py's frame loader -- same
TRAIN_CSV/overlay swap + the 118-row CryptoMamba year-boundary NaN fix), NOT the default 2025-only
omega.TRAIN_CSV, so this label set covers the actual 2024-01-01..2025-12-31 span a real retrain
would use.

Per-bar label construction (fixes the sparse-event-only gap from the chart scripts): for each
resolved LONG/SHORT event [t, exit_i], every bar in that inclusive range gets that label --
matches how the existing zigzag_action label is dense over trend segments (checked this session:
zigzag_action's train-split class balance is roughly 19,566 CASH / 86,795 LONG / 77,575 SHORT out
of 183,936 rows -- overwhelmingly non-CASH, i.e. forward-filled over segments, not sparse pivot
markers). A CASH event only labels its own single trigger bar t.

Chart scripts this session (chart_eth_triple_barrier_label_ground_truth_20260728.py,
chart_eth_triple_barrier_maxdensity_20260728.py) validated this exact label_event logic and
config on 2-week/3-month samples before this full-period run. Does NOT retrain anything, does
NOT touch trading_bot_modules/, trading_bot.py, .env. Output is a label contract only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/llewyn/crypto-scalping")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import chart_eth_triple_barrier_label_ground_truth_20260728 as lbl  # noqa: E402
import train_eval_omega4_3head_parent72_pinned102_2024tape_20260727 as tape2024  # noqa: E402

MIN_TP = 0.006
MIN_SL = 0.0032
VERTICAL_BARS = 12  # 1h
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_triple_barrier_maxdensity_20260728/label_contracts/triple_barrier_direction_maxdensity_20260728"
ACTION_CASH, ACTION_LONG, ACTION_SHORT = 0, 1, 2


def build_events(frame: pd.DataFrame) -> list[dict]:
    events = []
    t = 0
    last_end = len(frame) - VERTICAL_BARS - 1
    n_total = last_end
    t0 = time.time()
    while t < last_end:
        ev = lbl.label_event(frame, t, min_tp=MIN_TP, min_sl=MIN_SL, vertical_bars=VERTICAL_BARS)
        events.append({"t": t, "exit_i": ev["exit_i"], "label": ev["label"]})
        t = ev["exit_i"] + 1 if ev["label"] != "CASH" else t + 1
        if len(events) % 20000 == 0:
            print(f"  ...{t}/{n_total} bars scanned, {len(events)} events, {time.time()-t0:.0f}s elapsed", flush=True)
    return events


def expand_to_per_bar(frame: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    action = np.full(len(frame), ACTION_CASH, dtype=np.int64)
    for ev in events:
        if ev["label"] == "CASH":
            continue
        code = ACTION_LONG if ev["label"] == "LONG" else ACTION_SHORT
        action[ev["t"]: ev["exit_i"] + 1] = code
    return pd.DataFrame({"timestamp": frame["timestamp"], "zigzag_action": action})


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("stage=load_frame (2024+2025 tape, cmamba-fix applied)", flush=True)
    train_all, eval_df, _overlay = tape2024._load_omega_frames_2024tape()
    train_all["timestamp"] = pd.to_datetime(train_all["timestamp"])
    eval_df["timestamp"] = pd.to_datetime(eval_df["timestamp"])
    print(f"train frame: {len(train_all)} rows [{train_all['timestamp'].min()}..{train_all['timestamp'].max()}]", flush=True)
    print(f"eval (2026 OOS) frame: {len(eval_df)} rows [{eval_df['timestamp'].min()}..{eval_df['timestamp'].max()}]", flush=True)

    all_rows = []
    for tag, frame in (("TRAIN(2024-2025)", train_all), ("EVAL(2026)", eval_df)):
        print(f"stage=build_events [{tag}] (sequential triple-barrier walk)", flush=True)
        events = build_events(frame)
        labels = [e["label"] for e in events]
        n_long = sum(1 for l in labels if l == "LONG")
        n_short = sum(1 for l in labels if l == "SHORT")
        n_cash_events = sum(1 for l in labels if l == "CASH")
        print(f"  events: {len(events)} (LONG={n_long} SHORT={n_short} CASH_events={n_cash_events})", flush=True)

        print(f"stage=expand_to_per_bar [{tag}]", flush=True)
        out = expand_to_per_bar(frame, events)
        counts = out["zigzag_action"].value_counts().to_dict()
        print(f"  per-bar class counts: CASH={counts.get(0,0)} LONG={counts.get(1,0)} SHORT={counts.get(2,0)} "
              f"(total {len(out)})", flush=True)
        all_rows.append(out)

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    print("stage=write_per_year_csv", flush=True)
    combined["year"] = combined["timestamp"].dt.year
    for year, sub in combined.groupby("year"):
        path = OUT_DIR / f"zigzag_action_labels_{int(year)}.csv"
        sub[["timestamp", "zigzag_action"]].to_csv(path, index=False)
        print(f"  wrote {len(sub)} rows -> {path}", flush=True)

    print(f"stage=done out_dir={OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
