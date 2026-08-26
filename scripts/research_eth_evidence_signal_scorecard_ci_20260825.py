#!/usr/bin/env python3
"""Consolidated accuracy/CI scorecard for all 9 CURRENTLY-LIVE evidence signals
(scripts/live_evidence_signal_dashboard_20260823.py::SIGNAL_ORDER), computed in one pass so every
row uses the exact same formulas (imported verbatim, not re-derived), same data, same window, same
lift methodology (event_study/load_zigzag_pivots, reused unmodified) as the rest of this research
lineage. Adds a Wilson score 95% CI on precision (accuracy) for each signal/side -- not previously
computed anywhere in this repo's evidence-signal work, which reported point-estimate precision/lift
only.

Why re-run instead of pulling numbers from each signal's original (different-day) research script:
several signals' thresholds/definitions were finalized only once ported into the live dashboard
(e.g. smt_divergence's exact swing-prior reuse, dalton's low-vol-regime gate) -- running the SAME
live compute_signals() once, on the SAME canonical window, is the only way to guarantee every row
in this table reflects what's actually deployed right now, not a possibly-earlier-vintage number.

Window: VAL 2025-09-01..2025-12-31 + OOS 2026-01-01..2026-02-17 (this repo's standard evidence-
signal scorecard window). Horizon: 1h (K12_1h), the headline horizon used throughout this lineage;
4h/8h also reported for context.
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
from analyze_eth_creative_reversal_evidence_signals_20260814 import load_frame_with_orderflow  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
)
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_funding_crossasset_combo_signal_20260825 import load_funding_z  # noqa: E402

BTC_PATH = ROOT / "data" / "btc_5m_1year.csv"
Z_95 = 1.959963984540054


def wilson_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion -- more reliable than the normal
    approximation for small n or extreme p, both of which apply to several signals here
    (some have n<100, several have precision <20% or >80%)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> None:
    raw = load_frame_with_orderflow()
    btc_raw = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    funding_df = load_funding_z()

    sig = compute_signals(raw, btc_df=btc_raw, funding_df=funding_df)
    pivots = load_zigzag_pivots()

    ts = sig["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    all_pos = np.flatnonzero(window_mask)
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")

    rows = []
    for name, _desc in SIGNAL_ORDER:
        for side in ("bottom", "top"):
            col = f"{side}_{name}"
            side_pivots = pivots.loc[pivots["pivot_type"] == side]
            pivot_pos = sig.index[sig["timestamp"].isin(side_pivots["timestamp"])].to_numpy()
            trigger_pos = np.flatnonzero(sig[col].fillna(False).to_numpy() & window_mask)
            for k_name, K in K_HORIZONS.items():
                stats = event_study(trigger_pos, pivot_pos, all_pos, K)
                n, prec = stats["n_triggers"], stats["precision"]
                hits = round(prec * n) if n and np.isfinite(prec) else 0
                lo, hi = wilson_ci(hits, n) if n else (float("nan"), float("nan"))
                rows.append({
                    "signal": name, "side": side, "horizon": k_name,
                    "n_triggers": n, "precision": prec, "ci_lo": lo, "ci_hi": hi,
                    "baseline_rate": stats["baseline_rate"], "lift": stats["lift"],
                    "recall": stats["recall"],
                })

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp" / "eth_evidence_signal_scorecard_ci_20260825"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scorecard.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    for horizon in K_HORIZONS:
        print(f"\n=== horizon {horizon} ===")
        sub = df[df["horizon"] == horizon].copy()
        sub["precision_pct"] = (sub["precision"] * 100).round(1)
        sub["ci_lo_pct"] = (sub["ci_lo"] * 100).round(1)
        sub["ci_hi_pct"] = (sub["ci_hi"] * 100).round(1)
        sub["baseline_pct"] = (sub["baseline_rate"] * 100).round(1)
        sub["lift_x"] = sub["lift"].round(2)
        cols = ["signal", "side", "n_triggers", "precision_pct", "ci_lo_pct", "ci_hi_pct", "baseline_pct", "lift_x"]
        print(sub[cols].to_string(index=False))

    print(f"\nWrote {out_dir / 'scorecard.csv'}")


if __name__ == "__main__":
    main()
