"""Full derivation-trail verification for the 16 features selected for the zig075 Step-C feature
set. For each feature, checks:
  1. price-trend contamination: Spearman corr(feature, close) over TRAIN
  2. data health: NaN rate, degenerate-value check (constant / near-zero variance)
  3. mutual redundancy: full 16x16 Spearman correlation matrix (flag |r|>0.7 pairs)
  4. provenance: Step-B AUC screen pass/fail + AUC, per-knockoff-run pass/fail (fdr 0.10/0.20),
     mRMR appearance -- pulled from the already-saved result JSONs from this session.

Outputs: tmp/eth_zig075_oracle_label_check_20260811/final16_verification.json (+ printed report).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TECH_PANEL_PATH = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"
REGIME3_PATHS = [
    ROOT / f"data/ensemble/supervised/eth_regime3_current_hmm_jmredesign_20260810_{yr}_maskedname.csv"
    for yr in ("2024", "2025", "2026")
]
CHECK_DIR = ROOT / "tmp/eth_zig075_oracle_label_check_20260811"
OUT_PATH = CHECK_DIR / "final16_verification.json"

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-06-30")
CONTAMINATION_FLAG = 0.20
REDUNDANCY_FLAG = 0.70

FINAL16 = [
    "regime3_current_sensitive_wide24_chop_prob", "rsi", "ofti", "btc_lead_eth_follow_gap_3",
    "btc_volume_impulse_z", "log_return", "btc_ret_3", "smart_money_flow", "cvp_poc_dist",
    "oi_change_rate", "cvp_regime", "funding_roc_288", "ou_halflife", "vwap_dist_24",
    "funding_roc_48", "breakout_strength",
]

KNOCKOFF_RUNS = {
    "decon_n200": "knockoff_mrmr_result.json",
    "raw_n200": "knockoff_mrmr_result_raw_control.json",
    "decon_n1000": "knockoff_mrmr_result_ntrees1000.json",
}


def load_data() -> pd.DataFrame:
    tech = pd.read_csv(TECH_PANEL_PATH, low_memory=False)
    tech["timestamp"] = pd.to_datetime(tech["timestamp"])
    regime3 = pd.concat([pd.read_csv(p) for p in REGIME3_PATHS], ignore_index=True)
    regime3["timestamp"] = pd.to_datetime(regime3["timestamp"])
    df = tech.merge(regime3, on="timestamp", how="inner")
    df = df[(df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)].reset_index(drop=True)
    return df


def main() -> int:
    df = load_data()
    close = df["close"].to_numpy(dtype=np.float64)
    x = df[FINAL16].replace([np.inf, -np.inf], np.nan)

    # -- 1/2: contamination + data health --
    health = {}
    for f in FINAL16:
        v = x[f].to_numpy(dtype=np.float64)
        nan_rate = float(np.isnan(v).mean())
        finite = v[np.isfinite(v)]
        corr_close = float(pd.Series(finite).corr(pd.Series(close[np.isfinite(v)]), method="spearman")) if len(finite) > 10 else float("nan")
        health[f] = {
            "nan_rate": round(nan_rate, 4),
            "std": round(float(np.nanstd(v)), 6),
            "degenerate": bool(np.nanstd(v) < 1e-10),
            "corr_with_close": round(corr_close, 4),
            "contaminated": bool(abs(corr_close) >= CONTAMINATION_FLAG),
        }

    # -- 3: mutual redundancy --
    corr16 = x.corr(method="spearman")
    redundant_pairs = []
    for i, a in enumerate(FINAL16):
        for b in FINAL16[i + 1:]:
            r = float(corr16.loc[a, b])
            if abs(r) >= REDUNDANCY_FLAG:
                redundant_pairs.append({"a": a, "b": b, "rho": round(r, 4)})

    # -- 4: provenance from saved results --
    step_b = json.loads((CHECK_DIR / "oracle_feature_analysis.json").read_text())
    step_b_by_feat = {r["feature"]: r for r in step_b["per_feature"]}

    knockoff_by_run = {}
    for tag, fname in KNOCKOFF_RUNS.items():
        p = CHECK_DIR / fname
        if p.exists():
            knockoff_by_run[tag] = json.loads(p.read_text())

    provenance = {}
    for f in FINAL16:
        rec = {}
        sb = step_b_by_feat.get(f)
        rec["step_b_passed"] = bool(sb["passed"]) if sb else False
        rec["step_b_dir_auc_train"] = round(sb["dir_auc_train"], 4) if sb else None
        rec["step_b_dir_auc_val"] = round(sb["dir_auc_val"], 4) if sb else None
        for tag, data in knockoff_by_run.items():
            gate = data.get("knockoff_gate_results", {})
            dir_01 = f in gate.get("direction", {}).get("fdr_0.1", [])
            dir_02 = f in gate.get("direction", {}).get("fdr_0.2", [])
            trade_01 = f in gate.get("tradeability", {}).get("fdr_0.1", [])
            trade_02 = f in gate.get("tradeability", {}).get("fdr_0.2", [])
            in_mrmr = f in data.get("mrmr_top_k", [])
            in_final = f in data.get("final_after_dedup", [])
            rec[f"knockoff_{tag}"] = {
                "direction_fdr0.1": dir_01, "direction_fdr0.2": dir_02,
                "tradeability_fdr0.1": trade_01, "tradeability_fdr0.2": trade_02,
                "mrmr_top20": in_mrmr, "final_after_dedup": in_final,
            }
        rec["h48qual_reference"] = f in [
            "cvp_regime", "funding_roc_288", "ou_halflife", "vwap_dist_24",
            "funding_roc_48", "breakout_strength", "regime3_current_sensitive_wide24_chop_prob",
        ]
        n_sources = sum([
            rec["step_b_passed"],
            any(knockoff_by_run.get(t, {}).get("final_after_dedup") and f in knockoff_by_run[t]["final_after_dedup"] for t in KNOCKOFF_RUNS),
            rec["h48qual_reference"],
        ])
        rec["n_independent_sources"] = n_sources
        provenance[f] = rec

    # -- report --
    print("=== contamination / health ===")
    for f in FINAL16:
        h = health[f]
        flag = " <-- CONTAMINATION FLAG" if h["contaminated"] else ""
        deg = " <-- DEGENERATE" if h["degenerate"] else ""
        print(f"  {f}: corr(close)={h['corr_with_close']:+.3f} nan_rate={h['nan_rate']:.3f}{flag}{deg}")

    print("\n=== redundant pairs among the 16 (|rho|>=0.70) ===")
    if redundant_pairs:
        for p in redundant_pairs:
            print(f"  {p['a']} <-> {p['b']}: rho={p['rho']}")
    else:
        print("  none")

    print("\n=== provenance summary ===")
    for f in FINAL16:
        p = provenance[f]
        print(f"  {f}: step_b={p['step_b_passed']} h48qual_ref={p['h48qual_reference']} "
              f"n_independent_sources={p['n_independent_sources']}")

    OUT_PATH.write_text(json.dumps({
        "health": health, "redundant_pairs": redundant_pairs, "provenance": provenance,
    }, indent=2))
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
