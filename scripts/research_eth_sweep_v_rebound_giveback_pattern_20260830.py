#!/usr/bin/env python3
"""User request (2026-08-30, hand-drawn 3-pattern sketch): distinguish, among the currently-
merged V_REBOUND=1 population, "true sustained V" (keeps extending after the fast-window target
is hit) from "spike then plateau/support" (holds above the swept level per the existing confirmed
check, but gives back most of its gain and goes flat instead of continuing) -- pattern 2 (fails
and reverts through the level) is already exactly what label=0/confirmed=False captures, so this
script only needs to further split the EXISTING label=1 population, using data already available
in the 6-bar/30-minute window (no new lookahead needed).

giveback_ratio = (peak_in_window - window_end_close) / (peak_in_window - sweep_extreme)
  0.0   -> ended right at the best point reached (pattern 1, sustained)
  ~1.0  -> gave back essentially the whole move by the end (would mostly overlap label=0 already,
           since giving back everything usually means dropping back through the level)
  in between -> gave back a real chunk but still above the level (pattern 3, plateau/support)

DIAGNOSTIC ONLY -- purely descriptive (histogram + example charts), no label file is written or
changed. Caution from [[giveback_exit_label_uniform_policy_pattern_20260815]]: loose giveback
thresholds on 5m bars tend to just capture noise and fail to generalize across time -- do not
pick a final cutoff from this single script's distribution alone; chart-check + dev/holdout
first, same discipline v1->v4 of this label already went through.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_giveback_pattern_20260830"
LOOKAHEAD_BARS = 6


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_giveback_20260830", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    labels = pd.read_csv(LABEL_CSV)

    ratios = []
    for _, event in labels.iterrows():
        idx = int(event["candidate_index"])
        row = frame.iloc[idx]
        future = frame.iloc[idx + 1: idx + LOOKAHEAD_BARS + 1]
        if event["side"] == "downside":
            sweep_extreme = row["low"]
            peak = future["high"].max()
            end = future["close"].iloc[-1]
        else:
            sweep_extreme = row["high"]
            peak = future["low"].min()
            end = future["close"].iloc[-1]
        total_move = abs(peak - sweep_extreme)
        giveback = (peak - end) if event["side"] == "downside" else (end - peak)
        ratio = float(giveback / total_move) if total_move > 1e-12 else np.nan
        ratios.append(ratio)
    labels["giveback_ratio"] = ratios

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUT_DIR / "events_with_giveback_ratio.csv", index=False)

    v1 = labels[labels["label"] == 1]
    print(f"V_REBOUND=1 events: {len(v1)}")
    print(v1["giveback_ratio"].describe(percentiles=[0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=145)
    ax.hist(v1["giveback_ratio"].clip(-0.2, 1.5), bins=80, color="#2E86AB", edgecolor="white")
    for q, c in ((0.3, "#f2c14e"), (0.5, "#e76f51"), (0.7, "#9d4edd")):
        ax.axvline(v1["giveback_ratio"].quantile(q), color=c, linestyle="--", linewidth=2,
                    label=f"{int(q*100)}th pct = {v1['giveback_ratio'].quantile(q):.2f}")
    ax.set_title(
        f"Existing V_REBOUND=1 events (n={len(v1)}): giveback_ratio distribution\n"
        "0 = ended at the window's best point (pattern 1, sustained V) -- 1 = gave back the whole move by bar 6 (pattern 3 territory)",
        fontsize=17,
    )
    ax.set_xlabel("giveback_ratio = (peak - end) / (peak - sweep_extreme)", fontsize=14)
    ax.set_ylabel("count", fontsize=14)
    ax.legend(fontsize=13)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "giveback_ratio_histogram.png")
    print(f"saved: {OUT_DIR / 'giveback_ratio_histogram.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
