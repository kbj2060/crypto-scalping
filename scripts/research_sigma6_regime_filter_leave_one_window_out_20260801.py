#!/usr/bin/env python3
"""Genuine walk-forward check for the Sigma6 regime-filter candidate found in
research_sigma6_regime_filter_walkforward_grid_20260801.py (thr0.6/lev3/sl1.5/not_chop/rthr0.50
beat baseline on 4/5 fixed windows). That result still used the SAME 5 windows for both selection
(scanning the 104-config grid) and success declaration -- a multiple-testing / selection-bias risk,
not a genuine out-of-sample test.

This runs proper leave-one-window-out cross-validation: for each of the 5 windows held out in turn,
select from the grid using ONLY the other 4 windows (config must beat baseline on a majority, i.e.
>=3/4, of the selection windows; among qualifiers, pick the one with the best mean PnL margin over
baseline on the selection windows), then test that selected config on the held-out window it never
saw. A config selected this way that still reliably clears the held-out window across most/all folds
would be real evidence of a generalizing filter; if it doesn't, the earlier 4/5 result was
overfitting to which 5 windows happened to be chosen.
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

OUT_DIR = ROOT / "tmp/research_20260801/sigma6_regime_filter_leave_one_window_out"
BASE_KW = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3, fee_mult=1.0)
BASELINE = dict(thr=0.60, lev=3.0, sl=1.5, mode="none", rthr=0.34, stab=0.0)

WINDOWS = [
    ("W1", "2025-07-01", "2025-10-31"),
    ("W2_canonical_VAL", "2025-09-01", "2025-12-31"),
    ("W3", "2025-11-01", "2026-02-28"),
    ("W4_incl_canonical_OOS", "2026-01-01", "2026-04-30"),
    ("W5", "2026-03-01", "2026-06-30"),
]

GRID = [
    {"thr": thr, "lev": lev, "sl": sl, "mode": mode, "rthr": rthr, "stab": stab}
    for thr, lev, sl, mode, rthr, stab in itertools.product(
        [0.60, 0.70], [3.0, 4.0], [1.5, 2.5],
        ["trend_agree", "not_chop", "none"], [0.34, 0.42, 0.50], [0.0, 0.55],
    )
    if not (mode == "none" and (rthr != 0.34 or stab != 0.0))
]


def run_cfg(tapes: dict, cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    r = backtest(tapes[cfg["thr"]], leverage=cfg["lev"], sl_atr=cfg["sl"], reg_mode=cfg["mode"],
                 reg_thr=cfg["rthr"], stab_thr=cfg["stab"], start=start, end=end, **BASE_KW)
    return {"pnl": round(r["pnl"], 2), "mdd": round(r["mdd"], 2), "trades": r["trades"]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_tape_with_regime()
    tapes = {thr: v2.apply_quality_threshold(raw, thr) for thr in (0.60, 0.70)}
    windows = [(label, pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta("23h59min59s")) for label, s, e in WINDOWS]

    # Precompute baseline and every grid config's result on every window once.
    baseline_res = {label: run_cfg(tapes, BASELINE, start, end) for label, start, end in windows}
    grid_res = {}
    for cfg in GRID:
        key = tuple(sorted(cfg.items()))
        grid_res[key] = {label: run_cfg(tapes, cfg, start, end) for label, start, end in windows}

    fold_rows = []
    for held_idx, (held_label, held_start, held_end) in enumerate(windows):
        selection_labels = [lbl for i, (lbl, _, _) in enumerate(windows) if i != held_idx]

        candidates = []
        for cfg in GRID:
            key = tuple(sorted(cfg.items()))
            wins, margins = 0, []
            for lbl in selection_labels:
                r, b = grid_res[key][lbl], baseline_res[lbl]
                beats = r["pnl"] > b["pnl"] and r["mdd"] > b["mdd"]
                wins += int(beats)
                margins.append(r["pnl"] - b["pnl"])
            if wins >= 3:  # majority of the 4 selection windows
                candidates.append((cfg, wins, sum(margins) / len(margins)))

        if not candidates:
            fold_rows.append({"held_out": held_label, "n_candidates": 0, "selected_cfg": None,
                               "held_out_pnl": None, "held_out_mdd": None, "held_out_beats_baseline": False})
            continue

        candidates.sort(key=lambda t: (-t[1], -t[2]))
        best_cfg, sel_wins, sel_margin = candidates[0]
        key = tuple(sorted(best_cfg.items()))
        held_r = grid_res[key][held_label]
        held_b = baseline_res[held_label]
        beats_held = held_r["pnl"] > held_b["pnl"] and held_r["mdd"] > held_b["mdd"]

        fold_rows.append({
            "held_out": held_label, "n_candidates": len(candidates),
            "selected_cfg": str(best_cfg), "selection_wins_of_4": sel_wins,
            "held_out_baseline_pnl": held_b["pnl"], "held_out_baseline_mdd": held_b["mdd"],
            "held_out_pnl": held_r["pnl"], "held_out_mdd": held_r["mdd"],
            "held_out_beats_baseline": beats_held,
        })

    df = pd.DataFrame(fold_rows)
    df.to_csv(OUT_DIR / "leave_one_window_out_results.csv", index=False)
    print(df.to_string(index=False))

    n_pass = int(df["held_out_beats_baseline"].sum())
    print(f"\n{n_pass}/{len(df)} folds: config selected from the OTHER 4 windows still beats "
          f"baseline on the held-out window it never saw.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
