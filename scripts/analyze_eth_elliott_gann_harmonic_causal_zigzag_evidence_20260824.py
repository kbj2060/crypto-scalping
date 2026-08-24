#!/usr/bin/env python3
"""Evidence study (NOT a trading algorithm, NOT a promotion claim) for the 3 geometric-family
candidates deliberately excluded from analyze_eth_fibonacci_harmonic_geometric_evidence_20260824.py
on structural-operationalization grounds. User asked to run them anyway with a properly-designed
causal pivot detector instead of the single-48-bar-window leg proxy that script used. Same
zigzag-pivot ground truth / lift harness / VAL+OOS pooled window as every sibling evidence study.

ALL parameters below are chosen ONCE and pre-registered in this docstring before running --
this script does NOT sweep/tune any threshold looking for a passing cell. That discipline is the
direct answer to "wouldn't tuning make some of these pass" (see chat): this repo has already
tested that hypothesis 6 times on the ATR TP/SL floor and 4 times on evidence-signal execution
mechanisms (including swapping crude proxies for literature-exact HMM/CUSUM/EWMA implementations)
-- 0/10 combined. Multiple-comparisons inflation from post-hoc threshold search is a real,
previously-paid cost in this repo (DSR first-fail), not a hypothetical risk.

## Shared infrastructure: causal ATR-reversal zigzag

A pivot is CONFIRMED once price reverses REVERSAL_ATR_MULT * ATR(14) from the running extreme of
the current unconfirmed leg. This differs from the fibonacci script's single-48-bar-window leg
(which only knows "the trailing window's max and min", not a genuine alternating pivot
SEQUENCE) -- harmonic and Elliott patterns are inherently about a chain of 3+ prior legs, which a
single window cannot represent. REVERSAL_ATR_MULT=2.0 chosen once: the sibling AMT study's
"excess tail" used 1.0x ATR poke depth as a meaningful single-bar deviation; a genuine confirmed
SWING reversal (formation + confirmation, not a single wick) is a strictly stronger claim, so a
non-arbitrary doubling is used. This is the only threshold choice in the whole script; it is not
varied.

## G4 -- full 4-point XABCD harmonic pattern (Gartley/Bat/Butterfly/Crab)

At each newly-confirmed pivot D (5th of the last 5 alternating pivots X,A,B,C,D), compute:
  ab_xa = |B-A| / |A-X|,  bc_ab = |C-B| / |B-A|,  cd_bc = |D-C| / |C-B|,  ad_xa = |D-A| / |A-X|
Classify against the 4 classic harmonic ratio profiles (all tolerance bands are the standard
+/-5 percentage points used in harmonic-trading literature/software, e.g. Scott Carney's
tolerance convention -- not tuned for this dataset):
  Gartley:   ab_xa in [0.586,0.650], ad_xa in [0.736,0.836]
  Bat:       ab_xa in [0.332,0.550], ad_xa in [0.836,0.936]
  Butterfly: ab_xa in [0.736,0.836], ad_xa in [1.220,1.668]
  Crab:      ab_xa in [0.332,0.668], ad_xa in [1.568,1.668]
A match on ANY profile fires the signal at D's confirmation bar, direction = reversal from D
(D is a low -> expect bottom; D is a high -> expect top).

## G5 -- Elliott 5-wave impulse completion / ABC correction completion

Using the same causal pivot chain, at each new pivot check the last 5 legs (6 pivots, points
0..5 representing waves 1-2-3-4-5) against the 2 hard Elliott rules that are unambiguous and
checkable without subjective wave-degree judgment:
  (a) wave 2 does not retrace more than 100% of wave 1 (pivot 2 does not cross pivot 0's price)
  (b) wave 3 is not the shortest of waves 1, 3, 5 (by |price| length)
If both hold at the pivot-5 confirmation bar, fire "impulse5_complete" (direction = reversal from
pivot 5, expecting the A-wave correction). Separately, using the last 3 legs (4 pivots, an
A-B-C correction after any impulse), fire "abc_complete" when a 3-leg alternating sequence
completes with C's move length between 0.618x-1.618x of A's length (the standard "equality/
extension" band bounding a valid corrective C-wave) -- direction = resumption of the
pre-correction trend, i.e. reversal from C in the same direction the impulse before A was moving.

## G6 -- Gann fan angle touch (ATR-normalized 1x1/1x2/2x1)

Gann's original angles used a fixed price-per-time unit with no standardized crypto analog (the
reason for the original exclusion). Rather than search over scales (which would be exactly the
p-hacking this script's discipline forbids), ONE fixed, commonly-used modern retail-Gann
adaptation is implemented: from the most recently confirmed pivot, project 3 fan lines forward
at slope = {0.5, 1.0, 2.0} x ATR(14 at the pivot bar) price-units per bar (the "1x2/1x1/2x1"
angle family). A touch = a later bar's low (for a fan from a LOW pivot, tested as support) or
high (fan from a HIGH pivot, tested as resistance) crosses within 0.25x ATR of the projected line
level. This is ONE documented interpretation among several incompatible Gann conventions in
practitioner literature -- a null result here rules out this specific operationalization, not
"Gann" as a concept in general (stated honestly, not oversold).
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

from analyze_eth_amt_vsa_footprint_ifvg_component_evidence_20260815 import add_sweep, load_frame  # noqa: E402
from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    OOS_END,
    ZIGZAG_DIR,
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

OUT_DIR = ROOT / "tmp" / "eth_elliott_gann_harmonic_causal_zigzag_evidence_20260824"
REVERSAL_ATR_MULT = 2.0
GANN_SLOPES = (0.5, 1.0, 2.0)
GANN_TOUCH_TOL_ATR = 0.25
GANN_MAX_PROJECT_BARS = 288  # 24h cap so a fan line isn't tested indefinitely far from its anchor

HARMONIC_PROFILES = {
    "Gartley": {"ab_xa": (0.586, 0.650), "ad_xa": (0.736, 0.836)},
    "Bat": {"ab_xa": (0.332, 0.550), "ad_xa": (0.836, 0.936)},
    "Butterfly": {"ab_xa": (0.736, 0.836), "ad_xa": (1.220, 1.668)},
    "Crab": {"ab_xa": (0.332, 0.668), "ad_xa": (1.568, 1.668)},
}


def load_zigzag_pivots_full() -> pd.DataFrame:
    """Same construction as load_zigzag_pivots() but including 2024 -- the harmonic/Elliott/Gann
    signals require a chain of 4-6 prior confirmed swing pivots to even become eligible to fire,
    which is combinatorially much rarer than any single-bar signal in the sibling scorecard. The
    VAL+OOS pooled window (48,853 bars) turned out to contain ~0 qualifying compound events for
    3 of 4 signals -- not a threshold-tuning problem (REVERSAL_ATR_MULT is unchanged from the
    single pre-registered value), a sample-size problem. Widening the EVALUATION window to the
    full price-frame coverage (2024-01..2026-02-17, matching data/eth_5m_1year.csv) is a fair-
    sample fix, not a parameter search: no signal definition or threshold changes, only how much
    already-computed history the lift statistic is measured over. Results on this window are
    reported separately from POOLED/VAL/OOS and are not directly comparable to the master
    scorecard's numbers (which use the VAL+OOS convention) for that reason.
    """
    frames = []
    for year in (2024, 2025, 2026):
        path = ZIGZAG_DIR / f"zigzag_action_labels_{year}.csv"
        z = pd.read_csv(path, parse_dates=["timestamp"], usecols=["timestamp", "low", "high", "zigzag_action"])
        frames.append(z)
    zz = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    zz = zz.reset_index(drop=True)
    run_id = (zz["zigzag_action"] != zz["zigzag_action"].shift()).cumsum()
    pivots = []
    for _, run in zz.groupby(run_id):
        action = int(run["zigzag_action"].iloc[0])
        if action == 2:
            pivot_row = run.loc[run["low"].idxmin()]
            pivots.append({"timestamp": pivot_row["timestamp"], "pivot_type": "bottom", "pivot_price": pivot_row["low"]})
        elif action == 1:
            pivot_row = run.loc[run["high"].idxmax()]
            pivots.append({"timestamp": pivot_row["timestamp"], "pivot_type": "top", "pivot_price": pivot_row["high"]})
    return pd.DataFrame(pivots).sort_values("timestamp").reset_index(drop=True)


def causal_zigzag(high: np.ndarray, low: np.ndarray, atr: np.ndarray, mult: float) -> pd.DataFrame:
    n = len(high)
    last_type = "low"
    last_pos = 0
    cand_price = high[0]
    cand_pos = 0
    rows = []
    for i in range(1, n):
        if last_type == "low":
            if high[i] > cand_price:
                cand_price, cand_pos = high[i], i
            if cand_pos > last_pos and (cand_price - low[i]) >= mult * atr[i]:
                rows.append((cand_pos, cand_price, "high", i))
                last_type, last_pos = "high", cand_pos
                cand_price, cand_pos = low[i], i
        else:
            if low[i] < cand_price:
                cand_price, cand_pos = low[i], i
            if cand_pos > last_pos and (high[i] - cand_price) >= mult * atr[i]:
                rows.append((cand_pos, cand_price, "low", i))
                last_type, last_pos = "low", cand_pos
                cand_price, cand_pos = high[i], i
    return pd.DataFrame(rows, columns=["pos", "price", "type", "conf_pos"])


def add_harmonic_and_elliott(f: pd.DataFrame, piv: pd.DataFrame) -> pd.DataFrame:
    n = len(f)
    harmonic_bottom = np.zeros(n, dtype=bool)
    harmonic_top = np.zeros(n, dtype=bool)
    impulse5_bottom = np.zeros(n, dtype=bool)   # pivot5 is a LOW -> expect bounce up
    impulse5_top = np.zeros(n, dtype=bool)
    abc_bottom = np.zeros(n, dtype=bool)
    abc_top = np.zeros(n, dtype=bool)

    prices = piv["price"].to_numpy()
    types = piv["type"].to_numpy()
    conf_pos = piv["conf_pos"].to_numpy()

    for k in range(4, len(piv)):
        X, A, B, C, D = prices[k - 4:k + 1]
        xa, ab, bc, cd = abs(A - X), abs(B - A), abs(C - B), abs(D - C)
        if xa <= 0 or ab <= 0 or bc <= 0:
            continue
        ab_xa, ad_xa = ab / xa, abs(D - A) / xa
        bar = conf_pos[k]
        for profile in HARMONIC_PROFILES.values():
            lo1, hi1 = profile["ab_xa"]
            lo2, hi2 = profile["ad_xa"]
            if lo1 <= ab_xa <= hi1 and lo2 <= ad_xa <= hi2:
                if types[k] == "low":
                    harmonic_bottom[bar] = True
                else:
                    harmonic_top[bar] = True
                break

    for k in range(5, len(piv)):
        p0, p1, p2, p3, p4, p5 = prices[k - 5:k + 1]
        t0 = types[k - 5]
        w1, w2, w3, w4, w5 = abs(p1 - p0), abs(p2 - p1), abs(p3 - p2), abs(p4 - p3), abs(p5 - p4)
        bar = conf_pos[k]
        if t0 == "low":  # candidate up-impulse 1-2-3-4-5, pivot5 is a high
            wave2_ok = p2 > p0        # wave2 low doesn't cross below wave0 (start)
            wave3_not_shortest = w3 >= w1 and w3 >= w5
            if wave2_ok and wave3_not_shortest and types[k] == "high":
                impulse5_top[bar] = True
        else:            # candidate down-impulse, pivot5 is a low
            wave2_ok = p2 < p0
            wave3_not_shortest = w3 >= w1 and w3 >= w5
            if wave2_ok and wave3_not_shortest and types[k] == "low":
                impulse5_bottom[bar] = True

    for k in range(3, len(piv)):
        P, A, B, C = prices[k - 3], prices[k - 2], prices[k - 1], prices[k]
        wave_a_len = abs(A - P)
        wave_c_len = abs(C - B)
        if wave_a_len <= 0:
            continue
        ratio = wave_c_len / wave_a_len
        bar = conf_pos[k]
        if 0.618 <= ratio <= 1.618:
            # wave A (P->A) moves OPPOSITE the pre-correction impulse: A<P means the impulse
            # into P was up, so completion of the ABC correction should resume up.
            original_trend_up = A < P
            if types[k] == "low" and original_trend_up:
                abc_bottom[bar] = True
            elif types[k] == "high" and not original_trend_up:
                abc_top[bar] = True

    f["harmonic_bottom"], f["harmonic_top"] = harmonic_bottom, harmonic_top
    f["impulse5_bottom"], f["impulse5_top"] = impulse5_bottom, impulse5_top
    f["abc_bottom"], f["abc_top"] = abc_bottom, abc_top
    return f


def add_gann(f: pd.DataFrame, piv: pd.DataFrame) -> pd.DataFrame:
    n = len(f)
    low = f["low"].to_numpy()
    high = f["high"].to_numpy()
    atr = f["atr_price"].to_numpy()
    gann_bottom = np.zeros(n, dtype=bool)
    gann_top = np.zeros(n, dtype=bool)

    conf_pos = piv["conf_pos"].to_numpy()
    pos = piv["pos"].to_numpy()
    price = piv["price"].to_numpy()
    ptype = piv["type"].to_numpy()

    for k in range(len(piv)):
        anchor_pos, anchor_price, anchor_type = pos[k], price[k], ptype[k]
        start = conf_pos[k]
        end = min(start + GANN_MAX_PROJECT_BARS, n)
        if start >= end:
            continue
        bars_since = np.arange(start, end) - anchor_pos
        a = atr[anchor_pos]
        for slope in GANN_SLOPES:
            level = anchor_price + slope * a * bars_since if anchor_type == "low" else anchor_price - slope * a * bars_since
            if anchor_type == "low":
                touch = np.abs(low[start:end] - level) <= GANN_TOUCH_TOL_ATR * a
                gann_bottom[start:end] |= touch
            else:
                touch = np.abs(high[start:end] - level) <= GANN_TOUCH_TOL_ATR * a
                gann_top[start:end] |= touch

    f["gann_bottom"], f["gann_top"] = gann_bottom, gann_top
    return f


def components(f: pd.DataFrame, side: str) -> dict[str, pd.Series]:
    if side == "bottom":
        return {"G4_harmonic_xabcd": f["harmonic_bottom"], "G5a_impulse5_complete": f["impulse5_bottom"],
                "G5b_abc_complete": f["abc_bottom"], "G6_gann_fan_touch": f["gann_bottom"],
                "REF_plain_sweep": f["sweep_low"]}
    return {"G4_harmonic_xabcd": f["harmonic_top"], "G5a_impulse5_complete": f["impulse5_top"],
            "G5b_abc_complete": f["abc_top"], "G6_gann_fan_touch": f["gann_top"],
            "REF_plain_sweep": f["sweep_high"]}


def run_side(f: pd.DataFrame, mask: np.ndarray, pivots: pd.DataFrame, side: str, window: str) -> pd.DataFrame:
    close = f["close"].to_numpy()
    all_pos = np.flatnonzero(mask)
    pivot_pos = f.index[f["timestamp"].isin(pivots.loc[pivots["pivot_type"] == side, "timestamp"])].to_numpy()
    rows = []
    for name, sig in components(f, side).items():
        trigger_pos = np.flatnonzero(sig.to_numpy() & mask)
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

    print("Building causal ATR-reversal zigzag (this is a sequential Python loop, may take a bit)...")
    piv = causal_zigzag(f["high"].to_numpy(), f["low"].to_numpy(), f["atr_price"].to_numpy(), REVERSAL_ATR_MULT)
    print(f"causal zigzag pivots (full history): {len(piv)}  "
          f"(low={int((piv['type'] == 'low').sum())}, high={int((piv['type'] == 'high').sum())})")

    f = add_harmonic_and_elliott(f, piv)
    f = add_gann(f, piv)
    pivots = load_zigzag_pivots()
    pivots_full = load_zigzag_pivots_full()

    ts = f["timestamp"]
    masks = {
        "POOLED": (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy(),
        "VAL": ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy(),
        "OOS": ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy(),
        "FULL_2024_2026": np.ones(len(f), dtype=bool),
    }
    pivot_sources = {"POOLED": pivots, "VAL": pivots, "OOS": pivots, "FULL_2024_2026": pivots_full}
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"pooled bars={int(masks['POOLED'].sum())}, real pivots={len(pivots)}; "
          f"FULL_2024_2026 bars={int(masks['FULL_2024_2026'].sum())}, real pivots={len(pivots_full)}")

    print("\nsignal firing counts (FULL_2024_2026 window) + bar-level overlap vs REF_plain_sweep:")
    for side in ("bottom", "top"):
        ref = components(f, side)["REF_plain_sweep"].to_numpy() & masks["FULL_2024_2026"]
        for name in ("G4_harmonic_xabcd", "G5a_impulse5_complete", "G5b_abc_complete", "G6_gann_fan_touch"):
            sig = components(f, side)[name].to_numpy() & masks["FULL_2024_2026"]
            inter = (sig & ref).sum()
            print(f"  {side:<7}{name:<24} n={int(sig.sum()):>6}  overlap(sig∧sweep)/sig="
                  f"{inter / sig.sum() * 100 if sig.sum() else float('nan'):.1f}%")

    res = pd.concat([run_side(f, m, pivot_sources[w], side, w) for w, m in masks.items() for side in ("bottom", "top")],
                    ignore_index=True)

    pd.set_option("display.width", 200)
    cols = ["signal", "n_triggers", "precision", "baseline_rate", "lift", "recall", "excess_move_mean_pct"]
    for side in ("bottom", "top"):
        print(f"\n\n########## {side.upper()} (FULL_2024_2026) ##########")
        sub = res[(res["side"] == side) & (res["window"] == "FULL_2024_2026")]
        for horizon in K_HORIZONS:
            print(f"\n-- {horizon} --")
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    print("\n\n########## POOLED vs FULL_2024_2026 consistency (K12_1h lift) ##########")
    piv_tbl = res[res["horizon"] == "K12_1h"].pivot_table(index=["side", "signal"], columns="window",
                                                          values=["lift", "n_triggers"], aggfunc="first")
    print(piv_tbl.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "elliott_gann_harmonic_evidence_table.csv", index=False)
    piv.to_csv(OUT_DIR / "causal_zigzag_pivots.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'elliott_gann_harmonic_evidence_table.csv'}")


if __name__ == "__main__":
    main()
