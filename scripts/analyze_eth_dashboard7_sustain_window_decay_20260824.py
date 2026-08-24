#!/usr/bin/env python3
"""Does lift decay across the 1h sustain window (2026-08-24 UX change)? User asked directly:
"couldn't a bar shown as active-only-because-of-sustain be a false positive, does accuracy still
hold." Answered empirically, not by assumption: for each of the 7 dashboard signals, bucket bars
by "how many bars since the ORIGINAL firing" (0 = the exact firing bar the headline lift number
is measured on; 1-11 = bars that are ONLY shown active due to the sustain window, not a fresh
firing) and independently run the SAME event_study/lift computation on each bucket, treating every
bucketed bar as its own trigger for a fresh forward-looking K=12 (1h) window. This directly tests
whether "still shown active" bars carry the same forward-looking predictive value as the original
firing bar, or whether that value decays as the sustain window ages.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    OOS_END,
    event_study,
    load_zigzag_pivots,
)
from analyze_eth_dashboard7_loosened_threshold_lift_20260824 import (  # noqa: E402
    add_creative_cols,
    build_variant,
    components,
    load_frame,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
    compute_indicators,
)

OUT_DIR = ROOT / "tmp" / "eth_dashboard7_sustain_window_decay_20260824"
SUSTAIN_BARS = 12
BUCKETS = [(f"offset{i}", i, i) for i in range(0, 8)] + [("offset8-11", 8, 11)]


def bucket_positions(fire_pos: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """For each firing bar, attribute bars [fire, fire+11] to it (a later firing within that span
    takes over from its own point on -- matches the live rolling-max semantics: at any bar, the
    'age' shown is the age since the MOST RECENT firing, not older ones)."""
    age = np.full(n, -1, dtype=np.int64)  # -1 = not currently active
    fire_sorted = np.sort(fire_pos)
    for f in fire_sorted:
        end = min(f + SUSTAIN_BARS, n)
        ages_here = np.arange(0, end - f)
        # overwrite with THIS firing's age (more recent firing wins, matches rolling-max recency)
        age[f:end] = ages_here
    out = {}
    for label, lo, hi in BUCKETS:
        out[label] = np.flatnonzero((age >= lo) & (age <= hi))
    return out


def main() -> None:
    raw = load_frame()
    f = compute_indicators(raw).reset_index(drop=True)
    f = add_creative_cols(f)
    f = build_variant(f, 48, "")
    pivots = load_zigzag_pivots()

    ts = f["timestamp"]
    mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    all_pos = np.flatnonzero(mask)
    K = K_HORIZONS["K12_1h"]
    n = len(f)

    rows = []
    for side in ("bottom", "top"):
        pivot_pos = f.index[f["timestamp"].isin(pivots.loc[pivots["pivot_type"] == side, "timestamp"])].to_numpy()
        comps = components(f, side, "current")
        for name, sig in comps.items():
            fire_pos = np.flatnonzero(sig.fillna(False).to_numpy() & mask)
            buckets = bucket_positions(fire_pos, n)
            for label, _lo, _hi in BUCKETS:
                bpos = np.intersect1d(buckets[label], all_pos)
                stats = event_study(bpos, pivot_pos, all_pos, K)
                rows.append({"side": side, "signal": name, "bucket": label, **stats})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    for side in ("bottom", "top"):
        print(f"\n########## {side.upper()} -- lift by bars-since-original-firing ##########")
        piv = res[res["side"] == side].pivot_table(index="signal", columns="bucket",
                                                    values=["lift", "n_triggers"], aggfunc="first")
        piv = piv.reorder_levels([1, 0], axis=1)
        col_order = [b[0] for b in BUCKETS]
        piv = piv[[c for c in col_order if c in piv.columns.get_level_values(0)]]
        print(piv.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "sustain_decay_table.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'sustain_decay_table.csv'}")


if __name__ == "__main__":
    main()
