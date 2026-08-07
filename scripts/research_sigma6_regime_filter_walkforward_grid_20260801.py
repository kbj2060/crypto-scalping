#!/usr/bin/env python3
"""Fix the validation methodology itself: instead of selecting Sigma6's regime-filter grid on a
single fixed VAL window (the approach that produced 9 VAL winners, 0 of which survived OOS -- see
project-eth-sigma6-1h-timeframe-diversification-failed-20260731.md -- and whose single frozen
winner then only beat baseline on 1/5 rolling windows, the exact one it was picked on -- see
project-sigma6-regime-filter-rolling-window-CONFIRMS-val-overfit-20260801.md), score every grid
point across ALL 5 rolling windows and require it to beat the no-filter baseline on a MAJORITY of
windows before calling it a real candidate. A config that only wins on one out of five windows is
exactly the failure mode already confirmed; this raises the bar to "wins consistently across time"
instead of "wins on the one window we happened to select on".

Same grid as run_sigma6_regime_trend_20260705.py (unmodified), same 5 windows as
research_sigma6_regime_filter_rolling_windows_20260801.py. No held-out OOS touch here -- this is a
pure methodology check: does ANY config in the grid clear a genuine multi-window bar? If none do,
the regime-filter axis is closed under honest methodology, not just for the one previously-picked
config.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402
from run_sigma6_regime_trend_20260705 import load_tape_with_regime, backtest  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260801/sigma6_regime_filter_walkforward_grid"
BASE_KW = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3, fee_mult=1.0)
BASELINE = dict(thr=0.60, lev=3.0, sl=1.5, mode="none", rthr=0.34, stab=0.0)

WINDOWS = [
    ("W1", "2025-07-01", "2025-10-31"),
    ("W2_canonical_VAL", "2025-09-01", "2025-12-31"),
    ("W3", "2025-11-01", "2026-02-28"),
    ("W4_incl_canonical_OOS", "2026-01-01", "2026-04-30"),
    ("W5", "2026-03-01", "2026-06-30"),
]

GRID = list(itertools.product(
    [0.60, 0.70],
    [3.0, 4.0],
    [1.5, 2.5],
    ["trend_agree", "not_chop", "none"],
    [0.34, 0.42, 0.50],
    [0.0, 0.55],
))


def run_cfg(tapes: dict, cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    r = backtest(tapes[cfg["thr"]], leverage=cfg["lev"], sl_atr=cfg["sl"], reg_mode=cfg["mode"],
                 reg_thr=cfg["rthr"], stab_thr=cfg["stab"], start=start, end=end, **BASE_KW)
    return {"pnl": round(r["pnl"], 2), "mdd": round(r["mdd"], 2), "trades": r["trades"]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_tape_with_regime()
    tapes = {thr: v2.apply_quality_threshold(raw, thr) for thr in (0.60, 0.70)}
    windows = [(label, pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta("23h59min59s")) for label, s, e in WINDOWS]

    baseline_by_window = {label: run_cfg(tapes, BASELINE, start, end) for label, start, end in windows}
    print("Baseline (no filter, thr0.6/lev3/sl1.5) per window:")
    for label, r in baseline_by_window.items():
        print(f"  {label}: pnl={r['pnl']}% mdd={r['mdd']}% trades={r['trades']}")

    rows = []
    for thr, lev, sl, mode, rthr, stab in GRID:
        if mode == "none" and (rthr != 0.34 or stab != 0.0):
            continue
        cfg = dict(thr=thr, lev=lev, sl=sl, mode=mode, rthr=rthr, stab=stab)
        wins = 0
        per_window = {}
        for label, start, end in windows:
            r = run_cfg(tapes, cfg, start, end)
            b = baseline_by_window[label]
            beats = r["pnl"] > b["pnl"] and r["mdd"] > b["mdd"]
            wins += int(beats)
            per_window[label] = {**r, "beats_baseline": beats}
        row = {"thr": thr, "lev": lev, "sl": sl, "mode": mode, "rthr": rthr, "stab": stab,
               "windows_beating_baseline": wins, "n_windows": len(windows)}
        for label, d in per_window.items():
            row[f"{label}_pnl"] = d["pnl"]
            row[f"{label}_mdd"] = d["mdd"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("windows_beating_baseline", ascending=False)
    df.to_csv(OUT_DIR / "walkforward_grid_results.csv", index=False)

    print(f"\n{len(df)} grid configs (excluding baseline itself) scored across {len(windows)} windows.\n")
    print("Distribution of windows_beating_baseline (0..5):")
    print(df["windows_beating_baseline"].value_counts().sort_index().to_string())

    majority = df[df["windows_beating_baseline"] >= 3]
    print(f"\n{len(majority)}/{len(df)} configs beat baseline on a MAJORITY (>=3/5) of windows:")
    if not majority.empty:
        print(majority[["thr", "lev", "sl", "mode", "rthr", "stab", "windows_beating_baseline"]].to_string(index=False))
    else:
        print("  (none)")

    unanimous = df[df["windows_beating_baseline"] == 5]
    print(f"\n{len(unanimous)}/{len(df)} configs beat baseline on ALL 5 windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
