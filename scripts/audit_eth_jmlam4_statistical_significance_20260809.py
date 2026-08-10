"""Statistical rigor check on the JM-matched vs baseline router result (n=14 trades each), reusing
this project's own effect_size_report (pipeline/architecture_workbench.py) instead of eyeballing a
win-rate delta. Also runs a binomial test on the win-rate gap and a paired (index-matched, since
both scenarios trade the same ~2-month window at closely matching cadence) block bootstrap on the
total-return difference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.architecture_workbench import effect_size_report  # noqa: E402

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--out-dir", default="tmp/eth_greedy_router_regime_jmlam4_matched_20260809")
_args, _ = _ap.parse_known_args()
OUT_DIR = ROOT / _args.out_dir


def main() -> int:
    base = pd.read_csv(OUT_DIR / "ledger_baseline_wide24.csv")
    jm_candidates = sorted(OUT_DIR.glob("ledger_jmlam4*.csv"))
    if not jm_candidates:
        raise SystemExit(f"no ledger_jmlam4*.csv found in {OUT_DIR}")
    jm = pd.read_csv(jm_candidates[0])
    print(f"using JM ledger: {jm_candidates[0].name}")
    ra = base["trade_return"].to_numpy(dtype=np.float64)
    rb = jm["trade_return"].to_numpy(dtype=np.float64)
    print(f"baseline n={len(ra)} wins={int((ra>0).sum())}  JM n={len(rb)} wins={int((rb>0).sum())}")

    print("\n=== effect_size_report (project's own workbench gate function) ===")
    rep = effect_size_report(ra, rb, label_a="baseline", label_b="jm_matched")
    for k, v in rep.items():
        print(f"  {k}: {v}")

    print("\n=== win-rate binomial test ===")
    # H0: JM's true win prob equals baseline's empirical win prob (0.5, 7/14)
    n, k = len(rb), int((rb > 0).sum())
    p0 = float((ra > 0).mean())
    bt = stats.binomtest(k, n, p0, alternative="greater")
    print(f"  baseline win rate (p0)={p0:.4f}, JM wins={k}/{n}={k/n:.4f}")
    print(f"  binomial test P(X>={k} | n={n}, p={p0:.2f}) one-sided = {bt.pvalue:.4f}")
    bt2 = stats.binomtest(k, n, 0.5, alternative="greater")
    print(f"  binomial test vs fixed p=0.50 (fair coin) one-sided = {bt2.pvalue:.4f}")

    print("\n=== paired (index-matched) bootstrap on total-return gap ===")
    n_common = min(len(ra), len(rb))
    rng = np.random.default_rng(20260809)
    B = 20000
    diffs = np.zeros(B)
    for i in range(B):
        idx = rng.integers(0, n_common, size=n_common)
        diffs[i] = float(np.prod(1.0 + rb[idx]) - np.prod(1.0 + ra[idx]))
    real_diff = float(np.prod(1.0 + rb[:n_common]) - np.prod(1.0 + ra[:n_common]))
    pct_jm_better = float((diffs > 0).mean())
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  real (index-paired) total-return multiplier gap: {real_diff:+.4f}")
    print(f"  bootstrap: JM beats baseline in {pct_jm_better:.4f} of {B} resamples")
    print(f"  95% CI on the gap: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    print("\n=== unpaired (fully independent) bootstrap on total return, for comparison ===")
    diffs_u = np.zeros(B)
    for i in range(B):
        idx_a = rng.integers(0, len(ra), size=len(ra))
        idx_b = rng.integers(0, len(rb), size=len(rb))
        diffs_u[i] = float(np.prod(1.0 + rb[idx_b]) - np.prod(1.0 + ra[idx_a]))
    pct_jm_better_u = float((diffs_u > 0).mean())
    ci_lo_u, ci_hi_u = np.percentile(diffs_u, [2.5, 97.5])
    print(f"  bootstrap: JM beats baseline in {pct_jm_better_u:.4f} of {B} resamples")
    print(f"  95% CI on the gap: [{ci_lo_u:+.4f}, {ci_hi_u:+.4f}]")

    print("\n=== worst-trade sensitivity: drop the single best/worst trade from each side ===")
    ra_sorted = np.sort(ra)
    rb_sorted = np.sort(rb)
    ra_drop_best = ra[ra != ra_sorted[-1]]
    rb_drop_best = rb[rb != rb_sorted[-1]]
    tot_a = float(np.prod(1.0 + ra_drop_best) - 1.0) * 100
    tot_b = float(np.prod(1.0 + rb_drop_best) - 1.0) * 100
    print(f"  drop each side's single BEST trade: baseline total={tot_a:+.2f}%  JM total={tot_b:+.2f}%")
    ra_drop_worst = ra[ra != ra_sorted[0]]
    rb_drop_worst = rb[rb != rb_sorted[0]]
    tot_a2 = float(np.prod(1.0 + ra_drop_worst) - 1.0) * 100
    tot_b2 = float(np.prod(1.0 + rb_drop_worst) - 1.0) * 100
    print(f"  drop each side's single WORST trade: baseline total={tot_a2:+.2f}%  JM total={tot_b2:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
