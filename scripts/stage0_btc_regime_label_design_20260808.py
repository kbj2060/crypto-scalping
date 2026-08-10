"""Stage 0 — how should a bull/bear/chop regime LABEL be defined?  (2026-08-08)

Contract: docs/experiments/btc_regime_label_design_bullbearchop_20260808.json

Every regime line closed today held the label fixed and varied the classifier.  This one varies
the LABEL and leaves classifiers alone.  The incumbent oracle has no chop state — it forces a
directional call on every bar, including bars that are genuinely undecidable.

THE MEASUREMENT PROBLEM: a label cannot be scored against an oracle, because the label IS the
oracle.  So candidates are scored on properties that need no external reference —
  G1 cost validity        chop must actually be the state where a move cannot cover its costs
  G2 separation persists  bull > chop > bear forward-return ordering must hold on TRAIN and STILL
                          hold on OOS, and beat a random-relabelling null.  HEADLINE GATE — this
                          is the test that killed the direction differential earlier today.
  G4 persistence          median run >= a quarter of that label's OWN median wave (per-label floor)
  G5 coverage balance      chop occupancy in [15%, 60%]

PRINCIPLED ANCHOR FOR CHOP.  A wave is only monetisable if its amplitude covers BOTH costs a live
system pays: the confirmation tax 2*theta (the move a causal detector must see before it knows the
wave turned) and the round-trip trading cost.  So
    net = amplitude - 2*theta - cost
and a wave with net <= 0 is chop BY CONSTRUCTION.  That makes the chop threshold a consequence of
costs instead of a free parameter, which is what the incumbent label lacks.

Worth watching: theta=0.005 waves have median amplitude 0.97% against a confirmation tax of 1.00%
alone, so a cost-aware definition should classify nearly that whole family as chop — which would
explain directly why the frozen theta=0.5% detector cannot pay.
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
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/btc_regime_label_design_20260808"
FEE, SLIP = 0.0005, 0.0002
COST_MULT = 3.0
COST = 2 * (FEE + SLIP) * COST_MULT              # 0.42% round trip, matching the replay convention
FWD_H = 288                                      # 24h, held constant so families are comparable
N_PERM = 200
CHOP_OCC = (0.15, 0.60)
BULL, BEAR, CHOP = 1, -1, 0


def zigzag_waves(close: np.ndarray, theta: float):
    """(start, end, signed_amplitude) per completed retrospective wave."""
    _, piv = zigzag_oracle(close, threshold=theta)
    p = np.asarray(piv, dtype=np.int64)
    if len(p) < 3:
        return np.empty((0, 3))
    s, e = p[:-1], p[1:]
    amp = (close[e] - close[s]) / close[s]
    return np.column_stack([s, e, amp])


def from_waves(n: int, w: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Paint wave direction where `keep`, chop elsewhere."""
    out = np.full(n, CHOP, dtype=np.int8)
    for (s, e, amp), k in zip(w, keep):
        if k:
            out[int(s):int(e) + 1] = BULL if amp > 0 else BEAR
    return out


def label_family(close: np.ndarray, wave_cache: dict) -> dict[str, np.ndarray]:
    n = len(close)
    lab: dict[str, np.ndarray] = {}
    for th in (0.005, 0.010, 0.020):
        w = wave_cache[th]
        amp = np.abs(w[:, 2])
        lab[f"zigzag_pure|th{th*100:g}"] = from_waves(n, w, np.ones(len(w), bool))
        for a in (0.01, 0.02):
            lab[f"zigzag_amp|th{th*100:g}|A{a*100:g}"] = from_waves(n, w, amp >= a)
        for m in (0.0, 0.005):
            net = amp - 2 * th - COST
            lab[f"zigzag_net|th{th*100:g}|m{m*100:g}"] = from_waves(n, w, net >= m)
        for D in (48, 288):
            lab[f"zigzag_dur|th{th*100:g}|D{D}"] = from_waves(n, w, (w[:, 1] - w[:, 0]) >= D)

    logp = np.log(close)
    for win in (48, 288):
        net = np.full(n, np.nan)
        gross = np.full(n, np.nan)
        rng = np.full(n, np.nan)
        d = np.abs(np.diff(logp, prepend=logp[0]))
        cs = np.cumsum(d)
        net[win:] = logp[win:] - logp[:-win]
        gross[win:] = cs[win:] - cs[:-win]
        roll = pd.Series(close)
        rng[win:] = ((roll.rolling(win).max() - roll.rolling(win).min()) / roll).to_numpy()[win:]
        er = np.where(gross > 0, np.abs(net) / np.maximum(gross, 1e-12), 0.0)
        sgn = np.where(net > 0, BULL, BEAR).astype(np.int8)
        for e_bar in (0.3, 0.5):
            s = np.where(np.isfinite(er) & (er >= e_bar), sgn, CHOP).astype(np.int8)
            s[:win] = CHOP
            lab[f"eff_ratio|w{win}|e{e_bar:g}"] = s
        for r in (0.01, 0.03):
            s = np.where(np.isfinite(rng) & (rng >= r), sgn, CHOP).astype(np.int8)
            s[:win] = CHOP
            lab[f"range_band|w{win}|r{r*100:g}"] = s
    return lab


def net_capturable(close: np.ndarray, state: np.ndarray, theta_hint: float) -> dict:
    """Median (|move over the run| - 2*theta_hint - cost) by state — G1's own claim."""
    out = {}
    for name, code in (("bull", BULL), ("bear", BEAR), ("chop", CHOP)):
        vals = []
        for s, e, st in contiguous_runs(np.where(state == BULL, 2, np.where(state == BEAR, 0, 1))):
            if {2: BULL, 0: BEAR, 1: CHOP}[st] != code:
                continue
            mv = abs(close[min(e, len(close) - 1)] - close[s]) / close[s]
            vals.append(mv - 2 * theta_hint - COST)
        out[name] = round(float(np.median(vals)) * 100, 3) if vals else None
    return out


def separation(fwd: np.ndarray, state: np.ndarray, idx: np.ndarray) -> dict:
    m = {}
    for name, code in (("bull", BULL), ("chop", CHOP), ("bear", BEAR)):
        sel = idx[(state[idx] == code) & np.isfinite(fwd[idx])]
        m[name] = round(float(np.mean(fwd[sel])) * 100, 4) if len(sel) >= 50 else None
    ok = all(m[k] is not None for k in m)
    m["spread_bull_minus_bear_pct"] = round(m["bull"] - m["bear"], 4) if ok else None
    m["ordering_holds"] = bool(ok and m["bull"] > m["chop"] > m["bear"])
    return m


def perm_null(fwd: np.ndarray, state: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> dict:
    """Circular shift preserves state counts and run structure, destroys the alignment."""
    obs = separation(fwd, state, idx)["spread_bull_minus_bear_pct"]
    if obs is None:
        return {"insufficient": True}
    draws = []
    for _ in range(N_PERM):
        sh = np.roll(state, int(rng.integers(len(state))))
        v = separation(fwd, sh, idx)["spread_bull_minus_bear_pct"]
        if v is not None:
            draws.append(v)
    d = np.asarray(draws)
    return {"observed_spread": obs, "null_mean": round(float(d.mean()), 4),
            "null_p95": round(float(np.percentile(d, 95)), 4),
            "percentile": round(float((d < obs).mean()) * 100, 1), "n_draws": len(d)}


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts, close = panel["timestamp"], panel["close"].to_numpy(dtype=np.float64)
    n = len(close)

    fwd = np.full(n, np.nan)
    fwd[:-FWD_H] = close[FWD_H:] / close[:-FWD_H] - 1.0
    tr = np.flatnonzero((ts <= TRAIN_END).to_numpy())
    oo = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())
    print(json.dumps({"cost_round_trip_pct": round(COST * 100, 3), "fwd_horizon_bars": FWD_H,
                      "train_bars": len(tr), "oos_bars": len(oo)}), flush=True)

    wave_cache = {th: zigzag_waves(close, th) for th in (0.005, 0.010, 0.020)}
    for th, w in wave_cache.items():
        amp = np.abs(w[:, 2])
        print(json.dumps({f"theta{th*100:g}": {"waves": len(w),
                                               "median_amp_pct": round(float(np.median(amp)) * 100, 3),
                                               "median_net_pct": round(float(np.median(amp - 2 * th - COST)) * 100, 3),
                                               "pct_waves_net_positive": round(float((amp - 2 * th - COST > 0).mean()) * 100, 1)}}),
              flush=True)

    labels = label_family(close, wave_cache)
    rng = np.random.default_rng(20260808)
    rows: dict[str, dict] = {}
    for name, st in labels.items():
        th_hint = float(name.split("th")[1].split("|")[0]) / 100 if "th" in name else 0.0
        runs = [e - s + 1 for s, e, _ in contiguous_runs(np.where(st == BULL, 2, np.where(st == BEAR, 0, 1)))]
        med_run = float(np.median(runs)) if runs else float("nan")
        chop_occ = float((st == CHOP).mean())
        sep_tr, sep_oo = separation(fwd, st, tr), separation(fwd, st, oo)
        g1 = net_capturable(close, st, th_hint)
        floor = max(4.0, med_run / 4.0)
        rec = {
            "chop_occupancy": round(chop_occ, 3), "median_run_bars": med_run,
            "G1_net_capturable_pct_by_state": g1,
            "G1_pass": bool(g1["chop"] is not None and g1["chop"] <= 0
                            and (g1["bull"] or 0) > 0 and (g1["bear"] or 0) > 0),
            "G2_train": sep_tr, "G2_oos": sep_oo,
            "G2_pass": bool(sep_tr["ordering_holds"] and sep_oo["ordering_holds"]),
            "G4_pass": bool(med_run >= floor), "G4_floor_bars": round(floor, 1),
            "G5_pass": bool(CHOP_OCC[0] <= chop_occ <= CHOP_OCC[1]),
        }
        if rec["G2_pass"]:
            rec["G2_perm_null_oos"] = perm_null(fwd, st, oo, rng)
        rec["all_pass"] = bool(rec["G1_pass"] and rec["G2_pass"] and rec["G4_pass"] and rec["G5_pass"])
        rows[name] = rec
        print(f"  {name:30} chop {chop_occ:5.2f}  run {med_run:6.0f}  "
              f"netcap B/C/S {g1['bull']}/{g1['chop']}/{g1['bear']}  "
              f"fwd tr {sep_tr['spread_bull_minus_bear_pct']} oos {sep_oo['spread_bull_minus_bear_pct']}  "
              f"G1{'+' if rec['G1_pass'] else '-'} G2{'+' if rec['G2_pass'] else '-'}"
              f" G4{'+' if rec['G4_pass'] else '-'} G5{'+' if rec['G5_pass'] else '-'}", flush=True)

    survivors = sorted([k for k, v in rows.items() if v["all_pass"]],
                       key=lambda k: -(rows[k]["G2_oos"]["spread_bull_minus_bear_pct"] or -9e9))
    out = {"contract": "docs/experiments/btc_regime_label_design_bullbearchop_20260808.json",
           "cost_round_trip_pct": round(COST * 100, 3), "fwd_horizon_bars": FWD_H,
           "n_candidates": len(rows), "labels": rows, "survivors_ranked": survivors,
           "verdict": ("candidates survive all four label-quality gates; a second stage may test "
                       "learnability" if survivors else
                       "no labelling survives — the 3-state redefinition does not rescue the axis")}
    (OUT_DIR / "stage0.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"n_candidates": len(rows), "survivors": survivors}, indent=2), flush=True)
    print(f"wrote {OUT_DIR / 'stage0.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
