#!/usr/bin/env python3
"""Evidence study (NOT a trading algorithm, NOT a promotion claim) for the geometric/proportion
-based trader-lore family (Fibonacci retracement/extension, harmonic-pattern B/D-point zones) --
the one concept family confirmed 2026-08-24 to be genuinely untested in this repo, distinct in
mechanism from everything already scored (volume/orderflow-climax, price-position oscillators,
level-touch/sweep). Same harness/windows/zigzag-pivot ground truth as the ICT2022 and AMT/VSA/
iFVG sibling studies so lifts are directly comparable to the master scorecard (plain liquidity
sweep 3.01x/2.78x is the reference; orthogonal_combo 3.51x is the ceiling so far).

Pre-registered design (locked in this docstring before running, per repo convention):

Causal leg construction: for each bar i (i>=48), the trailing 48-bar window [i-48, i-1] (the
SAME window swing_low_prior/swing_high_prior already use) is scanned for the bar-offset of its
low extreme and high extreme via a vectorized sliding-window argmin/argmax. Whichever extreme
occurs LATER in the window is the "active leg's" most recent point:
  - low extreme before high extreme -> "up-leg" (low->high), current price sits somewhere
    relative to that leg; a pullback toward the low is a candidate retracement, a push beyond
    the high is a candidate extension.
  - high extreme before low extreme -> "down-leg" (high->low), mirrored.
This is a genuinely different geometric object from the sweep family (sweep = price BREAKS the
48-bar extreme and closes back; these signals = price touches an INTERIOR or EXTERIOR ratio
level of the established leg's range, never requiring the extreme itself to be broken for G1/G3).

3 signal families x 2 sides = 6 signals, all simple zone-TOUCH (no reject-confirmation
requirement, matching the OB-touch precedent's simplest form so results are comparable):
  G1 fib_golden_pocket   -- classic "D-point"/deep-retracement zone, 61.8%-78.6% retracement of
                            the active leg. Textbook Fibonacci reversal-trade zone.
  G3 fib_shallow_pullback-- classic harmonic "B-point" zone, 38.2%-61.8% retracement. Shallower
                            pullback-continuation entry zone, distinct ratio band from G1.
  G2 fib_extension_exhaustion -- price pushes 127.2%-161.8% beyond the leg's far extreme
                            (measured-move/AB=CD-style extension completion), testing whether
                            that extension zone marks continuation exhaustion (reversal), not
                            confirmation (the C1-C2 continuation-signal family already tested in
                            this repo and universally 0.89-1.06x -- this is a DIFFERENT question,
                            "does the specific fib ratio zone matter", not "does breaking out
                            predict more breakout").

Deliberately EXCLUDED, not silently skipped (repo's "no silent caps" discipline):
  - Elliott Wave: the only objectively causal proxy (wave-3 momentum-extension breakout) is
    mechanistically identical to this repo's already-tested-and-failed continuation/breakout
    family (5 signals, 0.89-1.06x, `eth_broad_evidence_signal_sweep_20260814.md`) -- rerunning it
    under a new name would not test new information. True multi-wave counting requires
    discretionary labeling that cannot be operationalized causally without look-ahead.
  - Gann angles/Square of 9: requires an arbitrary price-time unit-scaling choice with no
    standardized crypto convention; academic validation is essentially absent; any scan would be
    p-hacking the scale parameter itself rather than testing a fixed rule. Not implemented.

Mandatory overlap check vs the nearest already-scored signal (plain liquidity sweep) is run
before the lift table, per the AMT/VSA lesson (confirmation-stacking dilutes, and a "new" signal
that's mostly the same bars as an existing one isn't actually new information).
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

from analyze_eth_amt_vsa_footprint_ifvg_component_evidence_20260815 import (  # noqa: E402
    add_sweep,
    load_frame,
)
from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    OOS_END,
    event_study,
    excess_move,
    load_zigzag_pivots,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
    compute_indicators,
)

OUT_DIR = ROOT / "tmp" / "eth_fibonacci_harmonic_geometric_evidence_20260824"
LEG_WINDOW = 48  # matches swing_low_prior/swing_high_prior's existing convention


def add_leg_direction(f: pd.DataFrame) -> pd.DataFrame:
    low = f["low"].to_numpy()
    high = f["high"].to_numpy()
    n = len(f)
    low_pos = np.full(n, -1, dtype=np.int64)
    high_pos = np.full(n, -1, dtype=np.int64)

    lo_windows = np.lib.stride_tricks.sliding_window_view(low, LEG_WINDOW)   # window j = low[j:j+48]
    hi_windows = np.lib.stride_tricks.sliding_window_view(high, LEG_WINDOW)
    argmin_off = lo_windows.argmin(axis=1)
    argmax_off = hi_windows.argmax(axis=1)
    # bar i's window is raw indices [i-48, i-1] -> window index j = i-48 (valid for i in [48, n-1])
    idx = np.arange(LEG_WINDOW, n)
    j = idx - LEG_WINDOW
    low_pos[idx] = j + argmin_off[j]
    high_pos[idx] = j + argmax_off[j]

    f["leg_up"] = low_pos < high_pos      # low occurred first -> most recent extreme is the high
    f["leg_down"] = high_pos < low_pos    # high occurred first -> most recent extreme is the low
    f.loc[:LEG_WINDOW - 1, ["leg_up", "leg_down"]] = False
    return f


def add_fib_zones(f: pd.DataFrame) -> pd.DataFrame:
    lo, hi = f["swing_low_prior"], f["swing_high_prior"]
    rng = (hi - lo).replace(0.0, np.nan)
    leg_up, leg_down = f["leg_up"], f["leg_down"]

    # G1 golden pocket (61.8-78.6% retracement)
    f["fib_golden_pocket_bottom"] = leg_up & f["low"].between(hi - 0.786 * rng, hi - 0.618 * rng)
    f["fib_golden_pocket_top"] = leg_down & f["high"].between(lo + 0.618 * rng, lo + 0.786 * rng)

    # G3 shallow B-point (38.2-61.8% retracement)
    f["fib_shallow_pullback_bottom"] = leg_up & f["low"].between(hi - 0.618 * rng, hi - 0.382 * rng)
    f["fib_shallow_pullback_top"] = leg_down & f["high"].between(lo + 0.382 * rng, lo + 0.618 * rng)

    # G2 extension exhaustion (127.2-161.8% beyond the far extreme)
    f["fib_extension_exhaustion_top"] = leg_up & f["high"].between(hi + 0.272 * rng, hi + 0.618 * rng)
    f["fib_extension_exhaustion_bottom"] = leg_down & f["low"].between(lo - 0.618 * rng, lo - 0.272 * rng)

    for col in ("fib_golden_pocket_bottom", "fib_golden_pocket_top",
                "fib_shallow_pullback_bottom", "fib_shallow_pullback_top",
                "fib_extension_exhaustion_bottom", "fib_extension_exhaustion_top"):
        f[col] = f[col].fillna(False)
    return f


def components(f: pd.DataFrame, side: str) -> dict[str, pd.Series]:
    if side == "bottom":
        return {"G1_fib_golden_pocket": f["fib_golden_pocket_bottom"],
                "G3_fib_shallow_pullback": f["fib_shallow_pullback_bottom"],
                "G2_fib_extension_exhaustion": f["fib_extension_exhaustion_bottom"],
                "REF_plain_sweep": f["sweep_low"]}
    return {"G1_fib_golden_pocket": f["fib_golden_pocket_top"],
            "G3_fib_shallow_pullback": f["fib_shallow_pullback_top"],
            "G2_fib_extension_exhaustion": f["fib_extension_exhaustion_top"],
            "REF_plain_sweep": f["sweep_high"]}


def run_side(f: pd.DataFrame, mask: np.ndarray, pivots: pd.DataFrame, side: str, window: str) -> pd.DataFrame:
    close = f["close"].to_numpy()
    all_pos = np.flatnonzero(mask)
    pivot_pos = f.index[f["timestamp"].isin(pivots.loc[pivots["pivot_type"] == side, "timestamp"])].to_numpy()
    rows = []
    for name, sig in components(f, side).items():
        trigger_pos = np.flatnonzero(sig.fillna(False).to_numpy() & mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            move = excess_move(trigger_pos, pivot_pos, close, K)
            rows.append({"window": window, "side": side, "signal": name, "horizon": k_name,
                         **stats, "excess_move_mean_pct": move["mean_pct"]})
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_frame()
    f = compute_indicators(raw).reset_index(drop=True)
    f = add_sweep(f)
    f = add_leg_direction(f)
    f = add_fib_zones(f)
    pivots = load_zigzag_pivots()

    ts = f["timestamp"]
    masks = {
        "POOLED": (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy(),
        "VAL": ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy(),
        "OOS": ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy(),
    }
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"pooled bars={int(masks['POOLED'].sum())}, pivots={len(pivots)}")
    print(f"leg_up bars in pooled window: {int((f['leg_up'] & masks['POOLED']).sum())}, "
          f"leg_down bars: {int((f['leg_down'] & masks['POOLED']).sum())}")

    print("\nbar-level overlap vs REF_plain_sweep (pooled window):")
    for side in ("bottom", "top"):
        ref = components(f, side)["REF_plain_sweep"].fillna(False).to_numpy() & masks["POOLED"]
        for name in ("G1_fib_golden_pocket", "G3_fib_shallow_pullback", "G2_fib_extension_exhaustion"):
            sig = components(f, side)[name].fillna(False).to_numpy() & masks["POOLED"]
            inter = (sig & ref).sum()
            print(f"  {side:<7}{name:<28} n={int(sig.sum()):>6}  overlap(sig∧sweep)/sig="
                  f"{inter / sig.sum() * 100 if sig.sum() else float('nan'):.1f}%")

    res = pd.concat([run_side(f, m, pivots, side, w) for w, m in masks.items() for side in ("bottom", "top")],
                    ignore_index=True)

    pd.set_option("display.width", 200)
    cols = ["signal", "n_triggers", "precision", "baseline_rate", "lift", "recall", "excess_move_mean_pct"]
    for side in ("bottom", "top"):
        print(f"\n\n########## {side.upper()} (POOLED) ##########")
        sub = res[(res["side"] == side) & (res["window"] == "POOLED")]
        for horizon in K_HORIZONS:
            print(f"\n-- {horizon} --")
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    print("\n\n########## VAL vs OOS consistency (K12_1h lift) ##########")
    piv = res[res["horizon"] == "K12_1h"].pivot_table(index=["side", "signal"], columns="window",
                                                      values=["lift", "n_triggers"], aggfunc="first")
    print(piv.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "fibonacci_harmonic_evidence_table.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'fibonacci_harmonic_evidence_table.csv'}")


if __name__ == "__main__":
    main()
