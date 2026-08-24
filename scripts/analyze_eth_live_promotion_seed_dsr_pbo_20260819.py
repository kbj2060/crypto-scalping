#!/usr/bin/env python3
"""DSR/PBO-CSCV applied to the ETH live-promotion (Omega4.6.1 dual) seed-robustness check.

Input: window-level with_gate PnL% per seed, from
tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819_summary.json (N=3)
and the N=5 table in docs/experiments/eth_live_promotion_seed_robustness_3seed_20260819.md
(the underlying N=5 report/summary json no longer exists on disk -- only the markdown
table survives, so those 6x5 numbers are hardcoded below with that provenance noted).

Framing note: seed260620 is the ONLY seed that was actually deployed live; the other
seeds were added retrospectively to probe robustness, not as part of a real search-and-
pick process. DSR here answers a related but distinct question: "is 260620's realized
performance distinguishable from the noise floor that a reference class of this many
independently-trained seeds would produce by chance," not "did 260620 win a real search."
"""
import json
from pathlib import Path

import numpy as np

from core.selection_stats import deflated_sharpe_ratio, pbo_cscv, sharpe

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_N3 = ROOT / "tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819_summary.json"
WINDOWS = ["2025q1", "2025q2", "2025q3", "val", "oos_q1", "oos_q2"]

# N=5 table (docs/experiments/eth_live_promotion_seed_robustness_3seed_20260819.md,
# "N=5 확장" section) -- underlying report/summary json for the 2 extra seeds no longer
# exists on disk, only this committed markdown table. seed order matches the doc's columns.
N5_SEEDS = ["260620", "94046540", "524707103", "312069414", "44751167"]
N5_WITH_GATE = {
    "2025q1": [28.54, 49.82, 45.73, 55.49, 163.97],
    "2025q2": [39.99, -33.45, -35.08, 8.89, -6.71],
    "2025q3": [-9.73, -31.24, 4.77, -21.52, -7.14],
    "val": [54.88, 91.44, -16.48, 107.06, 19.26],
    "oos_q1": [28.17, 5.70, 38.41, 28.39, 40.27],
    "oos_q2": [9.85, -4.61, 13.72, 16.24, 55.29],
}


def load_n3_matrix():
    data = json.load(open(SUMMARY_N3))["windows"]
    seed_order = ["seed260620_original", "94046540", "524707103"]
    matrix = np.array(
        [[data[s][w]["with_gate"]["pnl"] for s in seed_order] for w in WINDOWS]
    )
    return matrix, seed_order


def load_n5_matrix():
    matrix = np.array([N5_WITH_GATE[w] for w in WINDOWS])
    return matrix, N5_SEEDS


def run(label, matrix, seed_labels):
    print(f"\n{'=' * 70}\n{label}  (periods={matrix.shape[0]} windows x configs={matrix.shape[1]} seeds)\n{'=' * 70}")
    print(f"seeds: {seed_labels}")
    print(f"returns_matrix (with_gate PnL%, rows=windows {WINDOWS}):")
    for w, row in zip(WINDOWS, matrix):
        print(f"  {w:8s} " + "  ".join(f"{v:8.2f}" for v in row))

    per_seed_sharpe = [sharpe(matrix[:, i]) for i in range(matrix.shape[1])]
    print("\nper-seed Sharpe across the 6 windows (n=6 periods each, treat as very noisy):")
    for s, sr in zip(seed_labels, per_seed_sharpe):
        print(f"  seed {s:12s} sharpe={sr:+.3f}")

    # DSR of the actually-deployed seed (first column = the one actually shipped live)
    deployed_returns = matrix[:, 0]
    dsr = deflated_sharpe_ratio(deployed_returns, np.array(per_seed_sharpe))
    print(f"\nDSR of the actually-deployed seed ({seed_labels[0]}) vs the noise floor implied")
    print(f"by these {matrix.shape[1]} seeds' Sharpe spread:")
    for k, v in dsr.items():
        print(f"  {k:22s} {v}")

    try:
        pbo = pbo_cscv(matrix, n_splits=2)  # max n_splits allowed by 6 periods (need >= n_splits*3)
        print(f"\nPBO-CSCV (n_splits=2, the max this data supports -- only 2 combinatorial")
        print(f"splits exist at this size, so treat this as illustrative, not a real estimate):")
        for k, v in pbo.items():
            print(f"  {k:22s} {v}")
    except ValueError as e:
        print(f"\nPBO-CSCV: could not run -- {e}")

    print("\nfalsification_audit: CANNOT run -- requires n_periods >= 10, this data has "
          f"{matrix.shape[0]} (6 calendar windows). Needs trade-level or bar-level returns, "
          "which no longer exist on disk for this experiment (bundles were cleaned up).")


if __name__ == "__main__":
    m3, s3 = load_n3_matrix()
    run("N=3 (raw JSON, actually on disk)", m3, ["260620", "94046540", "524707103"])

    m5, s5 = load_n5_matrix()
    run("N=5 (from committed markdown table -- underlying JSON no longer on disk)", m5, s5)
