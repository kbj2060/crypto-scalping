"""G4 — BOCPD as the causal counterpart of the Pagan-Sossounov regime label  (2026-08-08)

Contract: docs/experiments/btc_regime_bocpd_detector_g4_20260808.json
Method: Adams & MacKay (2007), Bayesian Online Changepoint Detection.

WHY BOCPD SPECIFICALLY.  PS is retrospective segmentation + a direction per phase.  BOCPD is
CAUSAL segmentation + a direction for the current run — the same object computed online.  And the
piece that makes it more than "another classifier": BOCPD carries an explicit HAZARD FUNCTION over
run length, so PS's defining constraint (a minimum phase duration) can be written directly into the
decoder as

    H(r) = 0        for r < P_min          <- a phase cannot end before P_min bars
    H(r) = h        for r >= P_min

That is exactly the duration prior a single scalar lambda in the jump-penalized decode CANNOT
express.  Today established that the decoder, not the probability model, does the work here, so a
decoder that can represent the label's own constraint is the principled next step.

CONTROL THAT CARRIES THE ARGUMENT: P_min = 0 gives a constant hazard, i.e. a geometric duration
prior — the same family as the incumbent lambda.  If the min-phase hazard does not beat that
control, then encoding the constraint bought nothing and the claim collapses, independently of how
BOCPD scores overall.

Model: Gaussian observations with unknown mean and variance, Normal-Inverse-Gamma conjugate prior,
Student-t posterior predictive.  Run-length posterior truncated for tractability.
Direction: sign of the price move accumulated since the MAP changepoint, with a chop band when the
move has not covered round-trip cost.  Fully causal — every quantity uses bars <= t.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from refine_btc_regime_classifier_theta005_20260808 import MIN_COVERAGE, PANEL_PATH  # noqa: E402
from stage0_btc_regime_label_design_20260808 import BEAR, BULL, CHOP, COST  # noqa: E402
from stage0c_btc_regime_label_oracle_dp_20260808 import trade  # noqa: E402
from stage0e_btc_regime_label_pagan_sossounov_20260808 import ps_label, ps_pivots  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/btc_regime_bocpd_20260808"
OUT_PARQUET = ROOT / "data/research/btc_regime_bocpd_states_20260808.parquet"
PS_P, PS_A = 48, 0.02
R_MAX = 1200
PRUNE = 1e-8
P_MINS = [0, 24, 48, 96]                 # 0 = constant-hazard CONTROL (geometric prior)
EXPECTED_EXTRA = [200, 400, 800]         # 1/h once the min phase is cleared
CHOP_MULTS = [0.0, 1.0, 2.0]             # chop band as a multiple of round-trip cost
CZZ_BASELINES = [0.005, 0.008, 0.012, 0.020, 0.030]


def bocpd_runlength(x: np.ndarray, p_min: int, expected_extra: float) -> np.ndarray:
    """MAP run length per bar. NIG-Gaussian UPM, Student-t predictive, truncated run-length posterior.

    `rvals` carries the ACTUAL run length of every surviving hypothesis. Pruning compacts the
    arrays, so indexing the hazard by array position instead of by run length silently decouples
    H(r) from r — which is exactly the bug the smoke test caught (run length pinned at 2).
    """
    n = len(x)
    mu0, kap0, al0, be0 = 0.0, 1.0, 1.0, float(np.var(x[: min(n, 5000)]) + 1e-12)
    mu = np.array([mu0]); kap = np.array([kap0]); al = np.array([al0]); be = np.array([be0])
    R = np.array([1.0])
    rvals = np.array([0], dtype=np.int64)
    out = np.zeros(n, dtype=np.int32)
    h_base = 1.0 / float(expected_extra)
    for t in range(n):
        scale = np.sqrt(be * (kap + 1.0) / (al * kap))
        pred = stats.t.pdf(x[t], df=2.0 * al, loc=mu, scale=scale)
        H = np.where(rvals < p_min, 0.0, h_base)          # indexed by TRUE run length
        growth = R * pred * (1.0 - H)
        cp = float(np.sum(R * pred * H))

        newR = np.concatenate(([cp], growth))
        new_rvals = np.concatenate(([0], rvals + 1))
        s = newR.sum()
        newR = newR / s if s > 0 else np.concatenate(([1.0], np.zeros(len(growth))))

        mu_n = np.concatenate(([mu0], (kap * mu + x[t]) / (kap + 1.0)))
        be_n = np.concatenate(([be0], be + kap * (x[t] - mu) ** 2 / (2.0 * (kap + 1.0))))
        kap_n = np.concatenate(([kap0], kap + 1.0))
        al_n = np.concatenate(([al0], al + 0.5))

        keep = (newR > PRUNE) & (new_rvals <= R_MAX)
        keep[0] = True
        R, rvals = newR[keep], new_rvals[keep]
        mu, kap, al, be = mu_n[keep], kap_n[keep], al_n[keep], be_n[keep]
        R = R / R.sum()
        out[t] = int(rvals[int(np.argmax(R))])
    return out


def direction_from_runs(close: np.ndarray, runlen: np.ndarray, chop_band: float) -> np.ndarray:
    """Direction = sign of the move since the MAP changepoint; chop while the move is under band."""
    n = len(close)
    st = np.full(n, CHOP, dtype=np.int8)
    for t in range(n):
        s = max(t - int(runlen[t]), 0)
        mv = (close[t] - close[s]) / close[s]
        if abs(mv) >= chop_band:
            st[t] = BULL if mv > 0 else BEAR
    return st


def summ(st: np.ndarray, tgt: np.ndarray, idx: np.ndarray, logret: np.ndarray) -> dict:
    d = np.where(st == BULL, 1, np.where(st == BEAR, -1, 0)).astype(np.int8)
    runs = [e - s + 1 for s, e, _ in contiguous_runs(np.where(st == BULL, 2, np.where(st == BEAR, 0, 1))[idx])]
    m = idx[(d[idx] != 0) & (tgt[idx] != 0)]
    return {"agree": round(float(np.mean(d[m] == tgt[m])) * 100, 1) if len(m) >= 50 else None,
            "coverage_pct": round(float((d[idx] != 0).mean()) * 100, 1),
            "median_run_bars": float(np.median(runs)) if runs else None,
            "n_switches": int(np.sum(st[idx][1:] != st[idx][:-1])),
            "traded_return_pct": trade(st, logret, idx, COST)}


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts, close = panel["timestamp"], panel["close"].to_numpy(dtype=np.float64)
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    logret[0] = 0.0

    piv = ps_pivots(close, 2 * PS_P, PS_P, 4 * PS_P, PS_A)
    ps = ps_label(close, piv, net_gate=True)
    tgt = np.where(ps == BULL, 1, np.where(ps == BEAR, -1, 0)).astype(np.int8)
    runs_ps = [e - s + 1 for s, e, _ in contiguous_runs(np.where(ps == BULL, 2, np.where(ps == BEAR, 0, 1)))]
    run_floor = float(np.median(runs_ps)) / 4.0

    v_idx = np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy())
    o_idx = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())
    windows = {"val_2025Q4": v_idx, "oos_2026Q1": o_idx}
    # standardise returns on TRAIN only
    trm = (ts <= TRAIN_END).to_numpy()
    x = ((logret - logret[trm].mean()) / (logret[trm].std() + 1e-12)).astype(np.float64)
    print(json.dumps({"ps_turning_points": len(piv), "run_floor_bars": round(run_floor, 1),
                      "ps_ceiling_traded": {w: trade(ps, logret, i, COST) for w, i in windows.items()}}),
          flush=True)

    base = {}
    for t in CZZ_BASELINES:
        cz = causal_zigzag(close, threshold=t)
        st = np.where(cz == 1, BULL, np.where(cz == -1, BEAR, CHOP)).astype(np.int8)
        base[f"czz{t*100:g}"] = {w: summ(st, tgt, i, logret) for w, i in windows.items()}
    eb = {k: v for k, v in base.items()
          if v["val_2025Q4"]["coverage_pct"] >= MIN_COVERAGE
          and (v["val_2025Q4"]["median_run_bars"] or 0) >= run_floor}
    best_base = max(eb, key=lambda k: eb[k]["val_2025Q4"]["agree"]) if eb else None
    print(json.dumps({"best_eligible_baseline": best_base,
                      "baseline_oos": None if not best_base else base[best_base]["oos_2026Q1"]}), flush=True)

    rows: dict[str, dict] = {}
    cache: dict[tuple[int, float], np.ndarray] = {}
    for pmin in P_MINS:
        for ex in EXPECTED_EXTRA:
            rl = bocpd_runlength(x, pmin, ex)
            cache[(pmin, ex)] = rl
            for cm in CHOP_MULTS:
                st = direction_from_runs(close, rl, cm * COST)
                key = f"bocpd|Pmin{pmin}|E{ex}|chop{cm:g}x"
                rows[key] = {w: summ(st, tgt, i, logret) for w, i in windows.items()}
                v = rows[key]["val_2025Q4"]
                print(f"  {key:30} VALagree {v['agree']}  cov {v['coverage_pct']}  "
                      f"run {v['median_run_bars']}  traded {v['traded_return_pct']}%", flush=True)

    elig = {k: v for k, v in rows.items()
            if v["val_2025Q4"]["coverage_pct"] >= MIN_COVERAGE
            and (v["val_2025Q4"]["median_run_bars"] or 0) >= run_floor
            and v["val_2025Q4"]["agree"] is not None}
    win = max(elig, key=lambda k: elig[k]["val_2025Q4"]["agree"]) if elig else None
    ctrl = {k: v for k, v in elig.items() if "Pmin0" in k}
    best_ctrl = max(ctrl, key=lambda k: ctrl[k]["val_2025Q4"]["agree"]) if ctrl else None

    out = {"contract": "docs/experiments/btc_regime_bocpd_detector_g4_20260808.json",
           "method": "Adams & MacKay (2007) BOCPD; NIG-Gaussian UPM; hazard encodes the PS minimum-phase constraint",
           "run_floor_bars": round(run_floor, 1), "baselines": base,
           "best_eligible_baseline_on_val": best_base, "cells": rows,
           "val_winner": win, "best_constant_hazard_control": best_ctrl}
    if win is not None:
        o = rows[win]["oos_2026Q1"]
        bb = base[best_base]["oos_2026Q1"]["agree"] if best_base else None
        cc = rows[best_ctrl]["val_2025Q4"]["agree"] if best_ctrl else None
        out["gates"] = {
            "eligible": bool(o["coverage_pct"] >= MIN_COVERAGE and (o["median_run_bars"] or 0) >= run_floor),
            "beats_no_learning_baseline": bool(bb is not None and o["agree"] is not None and o["agree"] > bb),
            "traded_return_positive": bool((o["traded_return_pct"] or -1) > 0),
            "min_phase_hazard_beats_constant_hazard_control_on_val": bool(
                cc is not None and elig[win]["val_2025Q4"]["agree"] > cc and "Pmin0" not in win),
        }
        out["oos_single_read"] = o
        out["adopt"] = bool(all(out["gates"].values()))
        print(json.dumps({"VAL_WINNER": win, "control": best_ctrl, "oos": o,
                          "baseline_oos_agree": bb, "gates": out["gates"],
                          "ADOPT": out["adopt"]}, indent=2), flush=True)
        st = direction_from_runs(close, cache[(int(win.split("Pmin")[1].split("|")[0]),
                                               float(win.split("E")[1].split("|")[0]))],
                                 float(win.split("chop")[1].rstrip("x")) * COST)
        pd.DataFrame({"timestamp": ts, "close": close, "ps_label": ps,
                      "bocpd_state": st}).to_parquet(OUT_PARQUET, index=False)
        print(f"wrote {OUT_PARQUET}", flush=True)
    else:
        out["halted"] = "no eligible cell on VAL"
        print(json.dumps({"halted": out["halted"]}), flush=True)
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
