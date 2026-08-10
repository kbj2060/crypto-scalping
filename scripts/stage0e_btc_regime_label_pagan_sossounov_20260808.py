"""Stage 0e — BRY-BOSCHAN / PAGAN-SOSSOUNOV bull-bear dating as a regime label  (2026-08-08)

Contract: docs/experiments/btc_regime_label_design_bullbearchop_20260808.json (same gates).

WHY THIS IS THE BIGGEST REMAINING GAP.  Bry & Boschan (1971) and Pagan & Sossounov (2003, "A
simple framework for analysing bull and bear markets") are the CANONICAL academic bull/bear dating
algorithms, and this project has never used them.  Our zigzag has exactly one rule — a percentage
reversal from the running extreme.  PS adds four structural constraints, and every one of them
targets the defect Stage 0 measured (median wave amplitude ~1.9*theta against a 2*theta
confirmation tax, so 63-74% of waves are unmonetisable):

  two-sided window extrema   a peak must dominate +/-W bars, not just trigger a reversal
  MIN PHASE duration         peak->trough must last >= P bars, else the phase is removed
  MIN CYCLE duration         peak->peak and trough->trough must last >= C bars
  amplitude censoring        the min-phase rule is WAIVED when the move is large enough

Stage 0 tested `zigzag_dur`, but that was a POST-HOC FILTER (relabel a short wave as chop) which
leaves the greedy pivots where they were.  PS is different in kind: it RE-DERIVES the segmentation
subject to the constraints, MERGING a short phase into its neighbour instead of carving it out.

PARAMETER SCALING, made explicit.  PS's numbers are for monthly equity data (window 8, min phase 4,
min cycle 16 months).  Rather than invent crypto constants, the paper's RATIOS are preserved —
window : min_phase : min_cycle = 2 : 1 : 4 — and min_phase is swept.  Like PS, no smoothing is
applied: large moves are the informative points.

Reported on both accounting bases used earlier in this arc, plus the lag curve, so results are
directly comparable to Stage 0/0b/0c/0d.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from refine_btc_regime_classifier_theta005_20260808 import PANEL_PATH  # noqa: E402
from stage0_btc_regime_label_design_20260808 import (  # noqa: E402
    BEAR, BULL, CHOP, COST, FWD_H, label_family, separation, zigzag_waves,
)
from stage0b_btc_regime_label_design_trendscan_20260808 import gross_net_by_state  # noqa: E402
from stage0c_btc_regime_label_oracle_dp_20260808 import LAG_GRID, oracle_dp, trade  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/btc_regime_label_design_20260808"
MIN_PHASES = [48, 144, 288, 576]          # 4h, 12h, 24h, 48h at 5m
CENSOR_AMPS = [0.02, 0.04]                # waive min-phase when the move clears this
NET_MARGIN = 0.0                          # cost-aware chop gate, same convention as zigzag_net


def ps_pivots(close: np.ndarray, window: int, min_phase: int, min_cycle: int,
              censor_amp: float) -> list[tuple[int, int]]:
    """Bry-Boschan / Pagan-Sossounov turning points. Returns [(index, +1 peak / -1 trough), ...]."""
    s = pd.Series(close)
    hi = s.rolling(2 * window + 1, center=True, min_periods=1).max().to_numpy()
    lo = s.rolling(2 * window + 1, center=True, min_periods=1).min().to_numpy()
    cand: list[tuple[int, int]] = []
    for t in range(len(close)):
        if close[t] >= hi[t]:
            cand.append((t, 1))
        elif close[t] <= lo[t]:
            cand.append((t, -1))
    if not cand:
        return []

    # 1) enforce alternation: among a run of same-kind candidates keep the most extreme
    alt: list[tuple[int, int]] = []
    for t, k in cand:
        if alt and alt[-1][1] == k:
            j, _ = alt[-1]
            better = (close[t] > close[j]) if k == 1 else (close[t] < close[j])
            if better:
                alt[-1] = (t, k)
        else:
            alt.append((t, k))

    # 2) min PHASE: drop a turning point whose phase is too short, unless the move is censored in
    changed = True
    while changed and len(alt) > 2:
        changed = False
        for i in range(1, len(alt)):
            t0, _ = alt[i - 1]
            t1, _ = alt[i]
            amp = abs(close[t1] - close[t0]) / close[t0]
            if (t1 - t0) < min_phase and amp < censor_amp:
                # remove the weaker endpoint, then re-run alternation
                drop = i if abs(close[t1] - close[t0]) > 0 else i
                keep_prev = (close[t1] < close[t0]) if alt[i][1] == 1 else (close[t1] > close[t0])
                alt.pop(i if keep_prev else i - 1)
                merged: list[tuple[int, int]] = []
                for t, k in alt:
                    if merged and merged[-1][1] == k:
                        j, _ = merged[-1]
                        better = (close[t] > close[j]) if k == 1 else (close[t] < close[j])
                        if better:
                            merged[-1] = (t, k)
                    else:
                        merged.append((t, k))
                alt = merged
                changed = True
                break

    # 3) min CYCLE: same-kind turning points must be at least min_cycle apart
    changed = True
    while changed and len(alt) > 3:
        changed = False
        for i in range(2, len(alt)):
            if alt[i][1] == alt[i - 2][1] and (alt[i][0] - alt[i - 2][0]) < min_cycle:
                a, b = alt[i - 2], alt[i]
                weaker = i - 2 if ((close[a[0]] < close[b[0]]) if a[1] == 1 else (close[a[0]] > close[b[0]])) else i
                mid = weaker - 1 if weaker > 0 else 1
                for j in sorted({weaker, mid}, reverse=True):
                    if 0 <= j < len(alt):
                        alt.pop(j)
                changed = True
                break
    return alt


def ps_label(close: np.ndarray, piv: list[tuple[int, int]], net_gate: bool) -> np.ndarray:
    """Trough->peak = bull, peak->trough = bear. With net_gate, phases that cannot cover
    (confirmation-free) trading cost + margin are relabelled chop."""
    out = np.full(len(close), CHOP, dtype=np.int8)
    for (t0, k0), (t1, _) in zip(piv, piv[1:]):
        amp = (close[t1] - close[t0]) / close[t0]
        if net_gate and (abs(amp) - COST) < NET_MARGIN:
            continue
        out[t0:t1 + 1] = BULL if k0 == -1 else BEAR
    return out


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts, close = panel["timestamp"], panel["close"].to_numpy(dtype=np.float64)
    n = len(close)
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    logret[0] = 0.0
    tr = np.flatnonzero((ts <= TRAIN_END).to_numpy())
    oo = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())
    fwd = np.full(n, np.nan)
    fwd[:-FWD_H] = close[FWD_H:] / close[:-FWD_H] - 1.0

    labels: dict[str, np.ndarray] = {}
    for P in MIN_PHASES:
        W, C = 2 * P, 4 * P
        for A in CENSOR_AMPS:
            piv = ps_pivots(close, W, P, C, A)
            labels[f"ps_pure|P{P}|A{A*100:g}"] = ps_label(close, piv, net_gate=False)
            labels[f"ps_net|P{P}|A{A*100:g}"] = ps_label(close, piv, net_gate=True)
            print(json.dumps({f"P{P}|A{A*100:g}": {"window": W, "min_cycle": C,
                                                   "turning_points": len(piv)}}), flush=True)

    zz = label_family(close, {th: zigzag_waves(close, th) for th in (0.005, 0.010, 0.020)})
    labels["zigzag_pure|th0.5 (incumbent)"] = zz["zigzag_pure|th0.5"]
    labels["zigzag_net|th0.5|m0"] = zz["zigzag_net|th0.5|m0"]
    labels["oracle_dp|fee4x (policy ref)"] = oracle_dp(logret, COST * 4)

    rows: dict[str, dict] = {}
    for name, st in labels.items():
        runs = [e - s + 1 for s, e, _ in contiguous_runs(np.where(st == BULL, 2, np.where(st == BEAR, 0, 1)))]
        g = gross_net_by_state(close, st)
        lag = {str(L): trade(np.roll(st, L), logret, oo, COST) for L in LAG_GRID}
        pos_lags = [int(L) for L in lag if lag[L] > 0]
        rec = {"chop_occupancy": round(float((st == CHOP).mean()), 3),
               "median_run_bars": float(np.median(runs)) if runs else None,
               "n_switches": int(np.sum(st[1:] != st[:-1])),
               "G1_move_minus_cost_pct": g,
               "G1_pass": bool(g["chop"] is not None and g["chop"] <= 0
                               and (g["bull"] or -1) > 0 and (g["bear"] or -1) > 0),
               "G2_train_spread": separation(fwd, st, tr)["spread_bull_minus_bear_pct"],
               "G2_oos_spread": separation(fwd, st, oo)["spread_bull_minus_bear_pct"],
               "lag_curve_pct": lag,
               "max_lag_bars_still_positive": max(pos_lags, default=None)}
        rows[name] = rec
        print(f"  {name:26} chop {rec['chop_occupancy']:5.2f} run {str(rec['median_run_bars']):>7} "
              f"sw {rec['n_switches']:6d}  moveB/C/S {g['bull']}/{g['chop']}/{g['bear']}  "
              f"L0 {lag['0']:>10,.0f} L3 {lag['3']:>9,.0f} L5 {lag['5']:>9,.0f} L12 {lag['12']:>9,.0f}  "
              f"G1{'+' if rec['G1_pass'] else '-'}", flush=True)

    out = {"contract": "docs/experiments/btc_regime_label_design_bullbearchop_20260808.json",
           "literature": "Bry & Boschan (1971); Pagan & Sossounov (2003) A Simple Framework for "
                         "Analysing Bull and Bear Markets, J. Applied Econometrics 18(1):23-46",
           "scaling": "PS ratios preserved (window:min_phase:min_cycle = 2:1:4); min_phase swept; "
                      "no smoothing, as in PS",
           "labels": rows}
    (OUT_DIR / "stage0e_pagan_sossounov.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'stage0e_pagan_sossounov.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
