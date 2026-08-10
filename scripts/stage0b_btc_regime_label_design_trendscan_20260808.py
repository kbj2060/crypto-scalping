"""Stage 0b — non-zigzag label logic: TREND-SCAN vs the zigzag families  (2026-08-08)

Contract: docs/experiments/btc_regime_label_design_bullbearchop_20260808.json (same gates).

Stage 0 found that the zigzag label family is structurally taxed: median wave amplitude is
~1.85-1.95x theta at EVERY scale while the causal confirmation tax is exactly 2*theta, so the
median wave is unmonetisable no matter which theta is chosen.  That defect belongs to the GREEDY
REVERSAL rule, not to labelling in general.

Trend-scanning (Lopez de Prado) defines states by a completely different boundary: fit forward OLS
of log(close) on bar index over several candidate horizons, take the horizon with the largest
|t-stat|, label by sign(slope) when |t| clears a significance floor and CASH otherwise.  There is
no reversal threshold, so there is no 2*theta structure at all — a different failure surface.

PRIOR ART, STATED HONESTLY: trend-scan was closed as an ENTRY architecture on 2026-08-04, and its
apparent win there turned out to be a lookahead bug (fixed across 5 files). The label file used
here is post-fix. Scoring it as a REGIME LABEL on the cost battery is a different question, but the
prior closure is on the record and this run does not overturn it.

FAIRNESS FIX vs Stage 0: `net_capturable` there subtracted the zigzag-specific 2*theta tax, which
would be unfair to a family that has no such tax. Everything here is additionally reported on a
COMMON basis — gross move minus trading cost only — so families are comparable, with the
zigzag-specific tax shown separately where it applies.
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
    BEAR, BULL, CHOP, CHOP_OCC, COST, FWD_H, label_family, perm_null, separation, zigzag_waves,
)
from test_statistical_jump_model_regimes_20260808 import contiguous_runs  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/btc_regime_label_design_20260808"
TS_PATH = ROOT / "data/splits/year_oos/btc_5m_trendscan_oracle_labels_20260806.parquet"
T_BARS = [13.55, 19.21, 26.98, 35.98]     # |t| quartiles/p90 of the existing label


def gross_net_by_state(close: np.ndarray, state: np.ndarray) -> dict:
    """Median (|move over the run| - trading cost) by state — COMMON basis, no theta tax."""
    out = {}
    for name, code in (("bull", BULL), ("bear", BEAR), ("chop", CHOP)):
        vals = []
        for s, e, st in contiguous_runs(np.where(state == BULL, 2, np.where(state == BEAR, 0, 1))):
            if {2: BULL, 0: BEAR, 1: CHOP}[st] != code:
                continue
            mv = abs(close[min(e, len(close) - 1)] - close[s]) / close[s]
            vals.append(mv - COST)
        out[name] = round(float(np.median(vals)) * 100, 3) if vals else None
    return out


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts, close = panel["timestamp"], panel["close"].to_numpy(dtype=np.float64)
    n = len(close)

    tsl = pd.read_parquet(TS_PATH)
    tsl["timestamp"] = pd.to_datetime(tsl["timestamp"])
    m = panel[["timestamp"]].merge(tsl, on="timestamp", how="left")
    act = m["trendscan_action"].to_numpy()
    tstat = m["trendscan_tstat"].to_numpy(dtype=np.float64)
    slope = m["trendscan_slope"].to_numpy(dtype=np.float64)
    cover = float(np.isfinite(tstat).mean())

    # recover the action encoding empirically instead of assuming it
    enc = {}
    for a in (0, 1, 2):
        sel = np.isfinite(slope) & (act == a)
        enc[a] = round(float(np.nanmean(np.sign(slope[sel]))), 3) if sel.sum() else None
    print(json.dumps({"trendscan_rows_matched_pct": round(cover * 100, 1),
                      "mean_sign_slope_by_action": enc}), flush=True)
    bull_code = max((a for a in enc if enc[a] is not None), key=lambda a: enc[a])
    bear_code = min((a for a in enc if enc[a] is not None), key=lambda a: enc[a])
    chop_code = [a for a in (0, 1, 2) if a not in (bull_code, bear_code)][0]
    print(json.dumps({"decoded": {"bull": int(bull_code), "bear": int(bear_code),
                                  "chop": int(chop_code)}}), flush=True)

    labels: dict[str, np.ndarray] = {}
    raw = np.full(n, CHOP, dtype=np.int8)
    raw[act == bull_code] = BULL
    raw[act == bear_code] = BEAR
    labels["trendscan_raw"] = raw
    sgn = np.where(slope > 0, BULL, BEAR).astype(np.int8)
    for T in T_BARS:
        s = np.where(np.isfinite(tstat) & (np.abs(tstat) >= T), sgn, CHOP).astype(np.int8)
        labels[f"trendscan_t|T{T:g}"] = s

    # carry the zigzag survivors through on the SAME common basis for comparison
    wave_cache = {th: zigzag_waves(close, th) for th in (0.005, 0.010, 0.020)}
    zz = label_family(close, wave_cache)
    for k in ("zigzag_pure|th0.5", "zigzag_net|th0.5|m0", "zigzag_net|th1|m0",
              "zigzag_net|th2|m0", "zigzag_net|th2|m0.5"):
        labels[k] = zz[k]

    fwd = np.full(n, np.nan)
    fwd[:-FWD_H] = close[FWD_H:] / close[:-FWD_H] - 1.0
    tr = np.flatnonzero((ts <= TRAIN_END).to_numpy())
    oo = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())
    rng = np.random.default_rng(20260808)

    rows: dict[str, dict] = {}
    for name, st in labels.items():
        runs = [e - s + 1 for s, e, _ in contiguous_runs(np.where(st == BULL, 2, np.where(st == BEAR, 0, 1)))]
        med_run = float(np.median(runs)) if runs else float("nan")
        chop_occ = float((st == CHOP).mean())
        g = gross_net_by_state(close, st)
        sep_tr, sep_oo = separation(fwd, st, tr), separation(fwd, st, oo)
        floor = max(4.0, med_run / 4.0)
        rec = {"chop_occupancy": round(chop_occ, 3), "median_run_bars": med_run,
               "G1_common_basis_move_minus_cost_pct": g,
               "G1_pass": bool(g["chop"] is not None and g["chop"] <= 0
                               and (g["bull"] or -1) > 0 and (g["bear"] or -1) > 0),
               "G2_train": sep_tr, "G2_oos": sep_oo,
               "G2_pass": bool(sep_tr["ordering_holds"] and sep_oo["ordering_holds"]),
               "G4_pass": bool(med_run >= floor), "G4_floor_bars": round(floor, 1),
               "G5_pass": bool(CHOP_OCC[0] <= chop_occ <= CHOP_OCC[1])}
        if rec["G2_pass"]:
            rec["G2_perm_null_oos"] = perm_null(fwd, st, oo, rng)
        rec["all_pass"] = bool(rec["G1_pass"] and rec["G2_pass"] and rec["G4_pass"] and rec["G5_pass"])
        rows[name] = rec
        print(f"  {name:26} chop {chop_occ:5.2f}  run {med_run:6.0f}  "
              f"move-cost B/C/S {g['bull']}/{g['chop']}/{g['bear']}  "
              f"fwd tr {sep_tr['spread_bull_minus_bear_pct']} oos {sep_oo['spread_bull_minus_bear_pct']}  "
              f"G1{'+' if rec['G1_pass'] else '-'} G2{'+' if rec['G2_pass'] else '-'}"
              f" G4{'+' if rec['G4_pass'] else '-'} G5{'+' if rec['G5_pass'] else '-'}", flush=True)

    surv = sorted([k for k, v in rows.items() if v["all_pass"]],
                  key=lambda k: -(rows[k]["G2_oos"]["spread_bull_minus_bear_pct"] or -9e9))
    out = {"contract": "docs/experiments/btc_regime_label_design_bullbearchop_20260808.json",
           "note": "common basis = move - trading cost only; the zigzag-specific 2*theta "
                   "confirmation tax is NOT subtracted here so families are comparable",
           "trendscan_prior_art": "closed as an ENTRY architecture 2026-08-04 (apparent win was a "
                                  "lookahead bug, since fixed); this is a different question and "
                                  "does not overturn that closure",
           "labels": rows, "survivors_ranked": surv}
    (OUT_DIR / "stage0b_trendscan.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"survivors": surv}, indent=2), flush=True)
    print(f"wrote {OUT_DIR / 'stage0b_trendscan.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
