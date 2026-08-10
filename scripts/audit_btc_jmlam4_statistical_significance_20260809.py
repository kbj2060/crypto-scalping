"""Statistical rigor check on BTC's JM-regime3 swap (q055, live-pinned threshold) vs live HMM
regime3, both at the full-replay sidecar level. Direct BTC mirror of
scripts/audit_eth_jmlam4_statistical_significance_20260809.py -- same battery of tests
(effect_size_report, binomial win-rate test, paired/unpaired bootstrap on total return,
single-best/worst-trade sensitivity), only the ledger source paths differ.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.architecture_workbench import effect_size_report  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--baseline-ledger", default="tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260806_swingtransition/oos_selected_risk_replayed_trade_ledger.csv")
_ap.add_argument("--jm-ledger", default="tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_jmlam4_20260809/oos_selected_risk_replayed_trade_ledger.csv")
_ap.add_argument("--split", default="oos")
_args, _ = _ap.parse_known_args()


def main() -> int:
    base = pd.read_csv(ROOT / _args.baseline_ledger)
    jm = pd.read_csv(ROOT / _args.jm_ledger)
    ra = base["trade_return"].to_numpy(dtype=np.float64)
    rb = jm["trade_return"].to_numpy(dtype=np.float64)
    print(f"split={_args.split}  baseline(HMM) n={len(ra)} wins={int((ra>0).sum())}  JM n={len(rb)} wins={int((rb>0).sum())}")
    print(f"  baseline(HMM) total return: {(np.prod(1.0+ra)-1.0)*100:+.2f}%   JM total return: {(np.prod(1.0+rb)-1.0)*100:+.2f}%")

    print("\n=== effect_size_report (project's own workbench gate function) ===")
    rep = effect_size_report(ra, rb, label_a="hmm_live", label_b="jm_lam4")
    for k, v in rep.items():
        print(f"  {k}: {v}")

    print("\n=== win-rate binomial test ===")
    n, k = len(rb), int((rb > 0).sum())
    p0 = float((ra > 0).mean())
    bt = stats.binomtest(k, n, p0, alternative="greater")
    print(f"  baseline win rate (p0)={p0:.4f}, JM wins={k}/{n}={k/n:.4f}")
    print(f"  binomial test P(X>={k} | n={n}, p={p0:.2f}) one-sided = {bt.pvalue:.4f}")

    print("\n=== paired (index-matched) bootstrap on total-return gap ===")
    n_common = min(len(ra), len(rb))
    rng = np.random.default_rng(20260809)
    B = 20000
    diffs = np.zeros(B)
    for i in range(B):
        idx = rng.integers(0, n_common, size=n_common)
        diffs[i] = float(np.prod(1.0 + rb[idx]) - np.prod(1.0 + ra[idx]))
    pct_jm_better = float((diffs > 0).mean())
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  bootstrap: JM beats baseline in {pct_jm_better:.4f} of {B} resamples")
    print(f"  95% CI on the gap: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    print("\n=== single best/worst trade sensitivity ===")
    ra_sorted, rb_sorted = np.sort(ra), np.sort(rb)
    tot_a = float(np.prod(1.0 + ra[ra != ra_sorted[-1]]) - 1.0) * 100
    tot_b = float(np.prod(1.0 + rb[rb != rb_sorted[-1]]) - 1.0) * 100
    print(f"  drop each side's single BEST trade: HMM total={tot_a:+.2f}%  JM total={tot_b:+.2f}%")
    tot_a2 = float(np.prod(1.0 + ra[ra != ra_sorted[0]]) - 1.0) * 100
    tot_b2 = float(np.prod(1.0 + rb[rb != rb_sorted[0]]) - 1.0) * 100
    print(f"  drop each side's single WORST trade: HMM total={tot_a2:+.2f}%  JM total={tot_b2:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
