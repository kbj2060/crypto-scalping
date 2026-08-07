#!/usr/bin/env python3
"""RESEARCH ONLY -- summary chart: every trade-density lever tried today (sequential vertical-
barrier extension, CUSUM-filtered event sampling at various k) extrapolated to the full ~638-day
TRAIN period, all at the SAME live TP/SL (min_tp=0.075, min_sl=0.040) -- no barrier-width change
in any of these. Shows the ~320-340 trade ceiling is barrier-width-driven, not a sampling
artifact: nothing tested moves the count meaningfully.

Numbers are copied from this session's actual runs (tmp/research_20260728/check_trade_density_
scaling.py + the wider-vertical follow-up + chart_eth_triple_barrier_cusum_events_20260728.py),
not recomputed here -- this script is presentation only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/llewyn/crypto-scalping")
OUT_PNG = ROOT / "tmp/research_20260728/chart_trade_density_ceiling_summary.png"

# label, extrapolated trades (full ~638d TRAIN), group
DATA = [
    ("24h", 276, "vertical barrier extension"),
    ("48h", 326, "vertical barrier extension"),  # baseline (h48qual's own "h48" naming)
    ("72h", 334, "vertical barrier extension"),
    ("96h", 326, "vertical barrier extension"),
    ("144h", 319, "vertical barrier extension"),
    ("k=4.0", 254, "CUSUM event sampling"),
    ("k=3.0", 268, "CUSUM event sampling"),
    ("k=2.0", 304, "CUSUM event sampling"),
    ("k=1.0", 319, "CUSUM event sampling"),
    ("k=0.5", 341, "CUSUM event sampling"),
]

COLOR_VERTICAL = "#2C6FBB"
COLOR_CUSUM = "#B5651D"
COLOR_BASELINE = "#7F8C8D"

fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
labels = [d[0] for d in DATA]
values = [d[1] for d in DATA]
colors = [COLOR_VERTICAL if d[2] == "vertical barrier extension" else COLOR_CUSUM for d in DATA]
x = range(len(DATA))

bars = ax.bar(x, values, color=colors, width=0.62, zorder=3)
ax.axhline(326, color=COLOR_BASELINE, linewidth=1.2, linestyle="--", zorder=2, alpha=0.8)
ax.text(len(DATA) - 0.4, 326 + 8, "48h sequential baseline (326)", fontsize=8.5, color=COLOR_BASELINE, ha="right")

for i, v in enumerate(values):
    ax.text(i, v + 5, str(v), ha="center", fontsize=9, color="#333")

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Extrapolated trades over full TRAIN (~638 days)")
ax.set_ylim(0, 380)
ax.set_title("Every trade-density lever tried today, same live TP/SL (7.5%/4.0%) throughout\n"
             "Nothing moves the ~320-340 trade ceiling -- it is barrier-WIDTH-driven, not sampling/horizon-driven",
             fontsize=11)
ax.grid(True, axis="y", alpha=0.15, linewidth=0.6, zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLOR_VERTICAL, label="vertical barrier extension (24h -> 144h)"),
    plt.Rectangle((0, 0), 1, 1, color=COLOR_CUSUM, label="CUSUM-filtered event sampling (k=4.0 -> 0.5)"),
]
ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_PNG)
print(f"saved {OUT_PNG}")
