#!/usr/bin/env python3
"""Does Sigma6's regime filter (bull/bear/chop gate) generalize across time, or did it only win
because it happened to be selected on the canonical VAL window?

Follow-up to project-eth-sigma6-1h-timeframe-diversification-failed-20260731.md (fresh retrain of
Sigma6's regime filter: 9/9 VAL winners collapsed OOS) and
project-eth-1h-patchtst-new-architecture-failed-20260731.md (a THIRD independent architecture
family showing the same VAL-win/OOS-collapse pattern on ETH's canonical VAL/OOS window -- pointing
at "selective long/short ETH models overfit this VAL window's cherry-pickable setups" rather than
architecture choice). Both memories name this exact experiment as the un-run next step: take a
config FROZEN from the canonical-VAL selection (no re-tuning per window -- that would just be more
look-ahead) and replay it, unmodified, across several OTHER rolling ~4-month windows the config was
never selected on. If the regime filter beats the no-filter baseline consistently across windows,
the VAL-collapse pattern is architecture/mechanism-specific and this axis is worth another look. If
it only wins on the window it was picked on, the "selective ETH models overfit VAL" hypothesis is
confirmed for this mechanism too and this line of research should stop.

Data-range note: the Sigma3-1h HGB ensemble tape (this signal's source) only exists from
2025-06-25 onward (see tape_ensemble.parquet) -- it CANNOT be extended back to 2024-01 the way the
naive buy-and-hold/SMA check in project-val-window-favorability-check-20260731.md could (that used
raw closes, not a trained model's signal). So this uses 5 overlapping ~4-month windows spanning the
tape's full available range instead of the originally-referenced 6 windows back to 2024-01.
"""

from __future__ import annotations

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

OUT_DIR = ROOT / "tmp/research_20260801/sigma6_regime_filter_rolling_windows"

# Frozen from project-eth-sigma6-1h-timeframe-diversification-failed-20260731.md's VAL-only
# selection funnel (val_grid.csv row 1) -- NOT re-tuned per window here.
WINNER = dict(thr=0.70, lev=4.0, sl=2.5, mode="not_chop", rthr=0.50, stab=0.55)
BASELINE = dict(thr=0.60, lev=3.0, sl=1.5, mode="none", rthr=0.34, stab=0.0)
BASE_KW = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3, fee_mult=1.0)

WINDOWS = [
    ("W1", "2025-07-01", "2025-10-31"),
    ("W2_canonical_VAL", "2025-09-01", "2025-12-31"),
    ("W3", "2025-11-01", "2026-02-28"),
    ("W4_incl_canonical_OOS", "2026-01-01", "2026-04-30"),
    ("W5", "2026-03-01", "2026-06-30"),
]


def run_cfg(tapes: dict, cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    r = backtest(tapes[cfg["thr"]], leverage=cfg["lev"], sl_atr=cfg["sl"], reg_mode=cfg["mode"],
                 reg_thr=cfg["rthr"], stab_thr=cfg["stab"], start=start, end=end, **BASE_KW)
    return {"pnl": round(r["pnl"], 2), "mdd": round(r["mdd"], 2), "trades": r["trades"], "wr": round(r["wr"], 3)}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_tape_with_regime()
    tapes = {thr: v2.apply_quality_threshold(raw, thr) for thr in (0.60, 0.70)}

    rows = []
    for label, start_s, end_s in WINDOWS:
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s) + pd.Timedelta("23h59min59s")
        base_r = run_cfg(tapes, BASELINE, start, end)
        filt_r = run_cfg(tapes, WINNER, start, end)
        beats_both = filt_r["pnl"] > base_r["pnl"] and filt_r["mdd"] > base_r["mdd"]
        rows.append({
            "window": label, "start": start_s, "end": end_s,
            "baseline_pnl": base_r["pnl"], "baseline_mdd": base_r["mdd"], "baseline_trades": base_r["trades"],
            "filtered_pnl": filt_r["pnl"], "filtered_mdd": filt_r["mdd"], "filtered_trades": filt_r["trades"],
            "filter_beats_baseline_both_axes": beats_both,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rolling_window_results.csv", index=False)
    print(f"Frozen winner config (selected on canonical VAL only): {WINNER}")
    print(f"Frozen baseline config: {BASELINE}\n")
    print(df.to_string(index=False))

    n_win = int(df["filter_beats_baseline_both_axes"].sum())
    print(f"\n{n_win}/{len(df)} windows: regime filter beats no-filter baseline on BOTH pnl and mdd.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
