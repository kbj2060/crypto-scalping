"""Selective classification / reject-option check on the entry candidate pool from
eth_rl_entry_gate_oracle_smoketest_20260831.

Literature basis: Chalkidis & Savani, "Trading via Selective Classification"
(arXiv:2110.14914, ICAIF 2021) -- instead of re-ranking all candidates with a new
model (already tried 6x and failed, see docs/eth_rl_autotrading_agent_design_20260831.md
Section 11), test whether an EXISTING confidence signal (the live quality head's own
quality_for_action score, plus the direction head's dir_trade_prob/dir_confidence) can
be turned into a reliable accept/abstain rule via a Selection-with-Guaranteed-Risk (SGR)
style calibration (Geifman & El-Yaniv 2017): pick, on a held-out calibration slice, the
lowest-coverage-cost threshold such that a Clopper-Pearson upper confidence bound on the
true error rate among accepted candidates is <= a target risk, then freeze it and read
VALIDATION exactly once.

Reuses: candidates_{train,validation}_labeled.csv from the prior smoke test (already
has oracle_label, price_move_raw, quality_for_action, dir_* columns -- no new
simulation, no new model). Same internal TRAIN calibration slice boundary as the prior
smoke test (holdout_start_ts/embargo_start_ts from that run's report.json), for
methodological consistency.

fresh_forward_bar_by_bar=true (inherited from the underlying candidate labels),
trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false. No live code touched.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import beta

BASE = "tmp/causal_regen_20260516/eth_rl_entry_gate_oracle_smoketest_20260831"
OUT = "tmp/causal_regen_20260516/eth_rl_entry_selective_classification_20260831"
os.makedirs(OUT, exist_ok=True)

COST_BP = 10.0  # standard taker roundtrip, no fee-discount assumption
HOLDOUT_START = pd.Timestamp("2025-08-10 04:10:00")
EMBARGO_START = pd.Timestamp("2025-08-09 04:10:00")
CONF_SIGNALS = ["quality_for_action", "dir_trade_prob", "dir_confidence"]
TARGET_RISKS = [0.45, 0.40, 0.35]
DELTA = 0.05  # one-sided 95% upper confidence bound on true risk, per Geifman & El-Yaniv

COLS = [
    "timestamp", "oracle_label", "price_move_raw",
    "quality_for_action", "dir_trade_prob", "dir_confidence", "dir_side_edge",
]


def load(split):
    df = pd.read_csv(f"{BASE}/candidates_{split}_labeled.csv", usecols=COLS, parse_dates=["timestamp"])
    df["net_bp"] = df["price_move_raw"] * 10000.0 - COST_BP
    return df


def risk_coverage_curve(df, conf_col):
    d = df.sort_values(conf_col, ascending=False).reset_index(drop=True)
    n = len(d)
    fracs = sorted(set([0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]))
    rows = []
    for f in fracs:
        k = max(1, int(round(n * f)))
        sub = d.iloc[:k]
        rows.append({
            "coverage": k / n, "n_accepted": k, "threshold": float(sub[conf_col].iloc[-1]),
            "risk": float(1 - sub["oracle_label"].mean()), "win_rate": float(sub["oracle_label"].mean()),
            "avg_net_bp": float(sub["net_bp"].mean()), "median_net_bp": float(sub["net_bp"].median()),
        })
    return rows


def sgr_calibrate(cal_df, conf_col, target_risk, delta, min_accept=50):
    d = cal_df.sort_values(conf_col, ascending=False).reset_index(drop=True)
    n = len(d)
    errors = (1 - d["oracle_label"]).values.astype(float)
    cum_err = np.cumsum(errors)
    best_k = None
    for k in range(min_accept, n + 1):
        e = cum_err[k - 1]
        ub = 1.0 if e >= k else float(beta.ppf(1 - delta, e + 1, k - e))
        if ub <= target_risk:
            best_k = k
    if best_k is None:
        return None
    e = cum_err[best_k - 1]
    return {
        "target_risk": target_risk, "delta": delta, "k_accepted_calibration": int(best_k),
        "coverage_calibration": best_k / n, "threshold": float(d[conf_col].iloc[best_k - 1]),
        "empirical_risk_calibration": float(e / best_k),
        "risk_upper_bound_calibration": float(1.0 if e >= best_k else beta.ppf(1 - delta, e + 1, best_k - e)),
    }


def apply_threshold(df, conf_col, threshold):
    sub = df[df[conf_col] >= threshold]
    n = len(sub)
    if n == 0:
        return {"n_accepted": 0, "coverage": 0.0}
    return {
        "n_accepted": int(n), "coverage": n / len(df),
        "risk": float(1 - sub["oracle_label"].mean()), "win_rate": float(sub["oracle_label"].mean()),
        "avg_net_bp": float(sub["net_bp"].mean()), "median_net_bp": float(sub["net_bp"].median()),
        "sum_net_bp": float(sub["net_bp"].sum()),
    }


train = load("train")
val = load("validation")

report = {"cost_bp": COST_BP, "holdout_start_ts": str(HOLDOUT_START), "embargo_start_ts": str(EMBARGO_START)}

# sanity check: replicate the prior smoke test's existing-gate VALIDATION numbers exactly
gate_check = apply_threshold(val, "quality_for_action", 0.50)
report["sanity_check_existing_gate_replication_validation"] = gate_check
report["sanity_check_expected_from_prior_report"] = {
    "n_accepted": 475, "avg_net_bp": 6.016794428402507, "median_net_bp": 9.045575844079954, "win_rate": 0.5473684210526316,
}

fit = train[train["timestamp"] < EMBARGO_START].copy()
holdout = train[train["timestamp"] >= HOLDOUT_START].copy()
report["calibration_holdout_n"] = len(holdout)
report["fit_slice_n_unused_here"] = len(fit)
report["accept_everything_baseline"] = {
    "train_holdout": {"win_rate": float(holdout["oracle_label"].mean()), "avg_net_bp": float(holdout["net_bp"].mean()), "median_net_bp": float(holdout["net_bp"].median())},
    "validation": {"win_rate": float(val["oracle_label"].mean()), "avg_net_bp": float(val["net_bp"].mean()), "median_net_bp": float(val["net_bp"].median())},
}

per_signal = {}
for sig in CONF_SIGNALS:
    entry = {}
    entry["risk_coverage_curve_calibration_holdout"] = risk_coverage_curve(holdout, sig)
    entry["risk_coverage_curve_validation_descriptive_only"] = risk_coverage_curve(val, sig)
    sgr_results = []
    for r in TARGET_RISKS:
        cal = sgr_calibrate(holdout, sig, r, DELTA)
        if cal is None:
            sgr_results.append({"target_risk": r, "status": "no_threshold_meets_bound_even_at_min_accept"})
            continue
        val_applied = apply_threshold(val, sig, cal["threshold"])
        sgr_results.append({"calibration": cal, "validation_single_read": val_applied})
    entry["sgr_calibrated_thresholds"] = sgr_results
    per_signal[sig] = entry

report["per_confidence_signal"] = per_signal

with open(f"{OUT}/report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print(json.dumps(report["sanity_check_existing_gate_replication_validation"], indent=2))
print(json.dumps(report["sanity_check_expected_from_prior_report"], indent=2))
print("---")
for sig in CONF_SIGNALS:
    print(f"=== {sig} ===")
    print("calibration-holdout risk-coverage (coverage, risk, avg_bp, median_bp):")
    for row in per_signal[sig]["risk_coverage_curve_calibration_holdout"]:
        print(f"  cov={row['coverage']:.3f} n={row['n_accepted']:>5} risk={row['risk']:.3f} avg_bp={row['avg_net_bp']:+7.2f} med_bp={row['median_net_bp']:+7.2f}")
    print("SGR-calibrated thresholds -> VALIDATION single read:")
    for res in per_signal[sig]["sgr_calibrated_thresholds"]:
        print(" ", json.dumps(res, default=str))
    print()

print(f"Wrote {OUT}/report.json")
