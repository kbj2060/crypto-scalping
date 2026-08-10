"""Stage 0 — is the theta=0.5% regime duration law non-geometric?  (2026-08-08)

Contract: docs/experiments/btc_regime_hsmm_duration_frontier_20260808.json

WHY THIS RUNS BEFORE ANY HSMM IS WRITTEN.  The frozen detectors decode with a single scalar
lambda, which is exactly a GEOMETRIC (memoryless) duration prior: the cost of switching is the
same whether the current run is 2 bars old or 40.  An HSMM's only structural advantage over that
is an explicit duration distribution.  So if the oracle's wave durations ARE geometric, lambda is
already the right functional form, an HSMM can only re-derive it, and the line should close here
for the price of one cheap measurement.

The test is stated so the geometric case IS the null: regress "does the wave end at this bar" on
log(run length so far).  A geometric law has a constant hazard, i.e. slope beta = 0.  Non-geometric
means beta != 0 — positive beta = the longer a wave has run the more likely it is to end
(anti-persistent, ageing), negative beta = the longer it has run the more likely it is to continue.

Also measured: bull-vs-bear asymmetry, since a single lambda cannot express that either.

TRAIN only for the gate.  VAL is used solely to check that the hazard slope's SIGN reproduces —
a duration dependence that flips sign between windows is unlearnable, and sign flips between
windows are this project's signature failure mode.  No OOS is touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from refine_btc_regime_classifier_theta005_20260808 import PANEL_PATH  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, VAL_START, VAL_END,
)

OUT_DIR = ROOT / "tmp/btc_regime_hsmm_20260808"
THETA = 0.005
SLOPE_P_BAR = 0.01
HAZARD_RATIO_BAR = 1.5


def waves(close: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Durations (bars) and directions of complete theta-waves whose START lies in [lo, hi]."""
    direction, pivots = zigzag_oracle(close, threshold=THETA)
    p = np.asarray(pivots, dtype=np.int64)
    if len(p) < 3:
        return np.array([]), np.array([])
    starts, ends = p[:-1], p[1:]
    keep = (starts >= lo) & (ends <= hi)
    return (ends - starts)[keep], direction[starts[keep]]


def hazard_slope(durations: np.ndarray) -> dict:
    """Logistic regression of 'ends at this bar' on log(run length).  beta == 0 IS the geometric null.

    Expands each wave of length D into D bernoulli trials (survive, survive, ..., end), which is
    the discrete-time hazard likelihood written exactly.
    """
    if len(durations) < 30:
        return {"insufficient": True, "n_waves": int(len(durations))}
    d = np.concatenate([np.arange(1, int(D) + 1) for D in durations])
    y = np.concatenate([np.r_[np.zeros(int(D) - 1), 1.0] for D in durations])
    x = np.log(d)

    def nll(th):
        z = np.clip(th[0] + th[1] * x, -30, 30)
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    fit = optimize.minimize(nll, np.array([-1.0, 0.0]), method="BFGS")
    a, b = fit.x
    # Wald SE from the numeric Hessian inverse
    se = float(np.sqrt(np.abs(fit.hess_inv[1, 1]))) if np.ndim(fit.hess_inv) == 2 else float("nan")
    z = b / se if se and np.isfinite(se) and se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")

    q10, q90 = np.percentile(durations, [10, 90])
    h = lambda dd: 1.0 / (1.0 + np.exp(-(a + b * np.log(max(dd, 1)))))  # noqa: E731
    hr = h(q90) / h(q10) if h(q10) > 0 else float("nan")
    return {"n_waves": int(len(durations)), "n_bar_trials": int(len(d)),
            "intercept": round(float(a), 4), "beta_log_duration": round(float(b), 4),
            "beta_se": round(se, 4) if np.isfinite(se) else None,
            "beta_p": round(p, 6) if np.isfinite(p) else None,
            "median_duration_bars": float(np.median(durations)),
            "q10_q90_bars": [float(q10), float(q90)],
            "hazard_at_q10": round(float(h(q10)), 4), "hazard_at_q90": round(float(h(q90)), 4),
            "hazard_ratio_q90_over_q10": round(float(hr), 3) if np.isfinite(hr) else None}


def geom_vs_nbinom(durations: np.ndarray) -> dict:
    """Likelihood-ratio test: geometric (memoryless) nested in negative binomial (df = 1)."""
    if len(durations) < 30:
        return {"insufficient": True}
    k = durations.astype(np.float64) - 1.0  # failures before the first success

    def nll_geom(th):
        pr = 1.0 / (1.0 + np.exp(-th[0]))
        return -float(np.sum(stats.geom.logpmf(durations, np.clip(pr, 1e-6, 1 - 1e-6))))

    def nll_nb(th):
        r = np.exp(th[0])
        pr = 1.0 / (1.0 + np.exp(-th[1]))
        return -float(np.sum(stats.nbinom.logpmf(k, r, np.clip(pr, 1e-6, 1 - 1e-6))))

    g = optimize.minimize(nll_geom, np.array([0.0]), method="Nelder-Mead")
    n = optimize.minimize(nll_nb, np.array([0.0, 0.0]), method="Nelder-Mead")
    lr = 2.0 * (g.fun - n.fun)
    return {"nll_geometric": round(float(g.fun), 2), "nll_nbinom": round(float(n.fun), 2),
            "lr_stat": round(float(lr), 2), "lr_p": round(float(stats.chi2.sf(max(lr, 0), 1)), 6),
            "nbinom_r": round(float(np.exp(n.x[0])), 3)}


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts, close = panel["timestamp"], panel["close"].to_numpy(dtype=np.float64)

    tr_hi = int(np.flatnonzero((ts <= TRAIN_END).to_numpy())[-1])
    v = np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy())
    windows = {"train": (0, tr_hi), "val": (int(v[0]), int(v[-1]))}

    out: dict = {"contract": "docs/experiments/btc_regime_hsmm_duration_frontier_20260808.json",
                 "theta": THETA, "kind": "no modelling; a property of the TARGET", "windows": {}}
    for wname, (lo, hi) in windows.items():
        dur, side = waves(close, lo, hi)
        blk = {"all": hazard_slope(dur), "geom_vs_nbinom": geom_vs_nbinom(dur)}
        for nm, sel in (("bull", side == 1), ("bear", side == -1)):
            blk[nm] = hazard_slope(dur[sel])
        if (side == 1).sum() > 5 and (side == -1).sum() > 5:
            u, pu = stats.mannwhitneyu(dur[side == 1], dur[side == -1])
            blk["bull_vs_bear_durations"] = {
                "median_bull": float(np.median(dur[side == 1])),
                "median_bear": float(np.median(dur[side == -1])),
                "mannwhitney_p": round(float(pu), 5)}
        out["windows"][wname] = blk
        a = blk["all"]
        print(f"  {wname:6} waves {a.get('n_waves')}  median {a.get('median_duration_bars')}bar  "
              f"beta {a.get('beta_log_duration')} (p {a.get('beta_p')})  "
              f"hazard q10→q90 {a.get('hazard_at_q10')}→{a.get('hazard_at_q90')} "
              f"(ratio {a.get('hazard_ratio_q90_over_q10')})", flush=True)

    tr, va = out["windows"]["train"]["all"], out["windows"]["val"]["all"]
    slope_ok = tr.get("beta_p") is not None and tr["beta_p"] < SLOPE_P_BAR
    hr = tr.get("hazard_ratio_q90_over_q10")
    size_ok = hr is not None and (hr >= HAZARD_RATIO_BAR or hr <= 1.0 / HAZARD_RATIO_BAR)
    sign_ok = (tr.get("beta_log_duration") is not None and va.get("beta_log_duration") is not None
               and np.sign(tr["beta_log_duration"]) == np.sign(va["beta_log_duration"]))
    verdict = {
        "gate": {"slope_significant_train_p<0.01": bool(slope_ok),
                 "hazard_ratio_>=1.5x_either_direction": bool(size_ok),
                 "sign_reproduces_on_val": bool(sign_ok)},
        "pass": bool(slope_ok and size_ok and sign_ok),
        "interpretation": None,
    }
    b = tr.get("beta_log_duration")
    if verdict["pass"]:
        verdict["interpretation"] = (
            "durations are NON-geometric with a stable sign — a single lambda cannot express this, "
            "so an explicit-duration (hazard-form) decode has something real to learn. "
            + ("Positive beta: waves AGE (the longer they run, the likelier they end)."
               if b and b > 0 else
               "Negative beta: waves are SELF-REINFORCING (the longer they run, the likelier they continue)."))
    else:
        verdict["interpretation"] = (
            "the duration law is geometric, too weak to matter, or unstable between windows. "
            "lambda is already the correct functional form; an HSMM could only re-derive it. "
            "CLOSE the line — no model written, no OOS spent.")
    out["verdict"] = verdict
    (OUT_DIR / "stage0_duration_law.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {OUT_DIR / 'stage0_duration_law.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
