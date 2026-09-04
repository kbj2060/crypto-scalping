#!/usr/bin/env python3
"""Cross-check: on the exact 122 events where the liquidation cascade indicator co-occurs with a
real V_REBOUND sweep (research_eth_sweep_v_rebound_liquidation_cascade_diagnostic_20260829.py's
output), does the ALREADY-FINALIZED TabPFN model (Tier0+rsi, same TRAIN<2025-09-01 as always --
not retrained, not re-decided) already get these right, or does the cascade rule catch something
TabPFN misses? User's own framing: "does the cascade indicator add anything beyond what TabPFN
already knows on those same events".

DIAGNOSTIC ONLY, same caveat as the prior two scripts in this lineage: these 122 events sit in
2026-07-18..08-29, overlapping the already-"spent" reserved holdout tail. No retraining, no new
promotion decision -- this only runs the existing finalized model's inference on a fixed slice
and compares it against a fixed external rule, both scored against the same real V_REBOUND label.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
TIER0_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
RSI_SOURCES = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
MATCHED_EVENTS = ROOT / "data/research/eth_liquidation_cascade_sweep_vs_trend_pilot_20260828/vrebound_matched_events.csv"
OUT_DIR = ROOT / "data/research/eth_liquidation_cascade_sweep_vs_trend_pilot_20260828"

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
SEEDS = [20260829, 141592, 271828, 577215]

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]


def main() -> int:
    tier0 = pd.read_csv(TIER0_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    frames = []
    for path in RSI_SOURCES:
        f = pd.read_csv(path, usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    rsi = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    df = tier0.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)

    train = df.loc[df["timestamp"] < VAL_START]
    print(f"train n={len(train)}")

    matched = pd.read_csv(MATCHED_EVENTS, parse_dates=["bar_timestamp"])
    matched["bar_timestamp"] = pd.to_datetime(matched["bar_timestamp"], utc=True)
    target = matched.merge(df[["timestamp"] + FEATURES], left_on="bar_timestamp", right_on="timestamp", how="left")
    missing = target[FEATURES].isna().any(axis=1).sum()
    print(f"matched events: {len(matched)}, with complete Tier0+rsi features: {len(target) - missing}")
    target = target.dropna(subset=FEATURES).reset_index(drop=True)

    proba_by_seed = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURES], train["label"].to_numpy())
        proba = clf.predict_proba(target[FEATURES])[:, 1]
        proba_by_seed.append(proba)
        print(f"  seed={seed} done")

    target["tabpfn_proba"] = np.mean(proba_by_seed, axis=0)
    target["tabpfn_pred"] = (target["tabpfn_proba"] >= 0.5).astype(int)
    target["tabpfn_correct"] = (target["tabpfn_pred"] == target["label"]).astype(int)

    overall_acc = float(target["tabpfn_correct"].mean())
    print(f"\nTabPFN accuracy on all {len(target)} matched events: {overall_acc:.4f}")

    def subset_report(mask: pd.Series, rule_name: str, rule_predicts_label: int, rule_precision: float, rule_n: int) -> dict:
        sub = target.loc[mask]
        tabpfn_acc_on_subset = float(sub["tabpfn_correct"].mean()) if len(sub) else None
        tabpfn_pred_matches_rule = float((sub["tabpfn_pred"] == rule_predicts_label).mean()) if len(sub) else None
        # of the cases the RULE got right, did tabpfn also get them right? (redundancy check)
        rule_correct_mask = sub["label"] == rule_predicts_label
        tabpfn_acc_where_rule_right = (
            float(sub.loc[rule_correct_mask, "tabpfn_correct"].mean()) if rule_correct_mask.any() else None
        )
        # of the cases the RULE got WRONG, did tabpfn get them right instead?
        rule_wrong_mask = ~rule_correct_mask
        tabpfn_acc_where_rule_wrong = (
            float(sub.loc[rule_wrong_mask, "tabpfn_correct"].mean()) if rule_wrong_mask.any() else None
        )
        return {
            "rule": rule_name, "n": int(len(sub)),
            "rule_precision_(reported_earlier)": rule_precision, "rule_n_(reported_earlier)": rule_n,
            "tabpfn_accuracy_on_this_same_subset": tabpfn_acc_on_subset,
            "tabpfn_predicts_same_label_as_rule_pct": tabpfn_pred_matches_rule,
            "n_rule_correct": int(rule_correct_mask.sum()),
            "tabpfn_accuracy_on_rule-correct_cases": tabpfn_acc_where_rule_right,
            "n_rule_wrong": int(rule_wrong_mask.sum()),
            "tabpfn_accuracy_on_rule-wrong_cases": tabpfn_acc_where_rule_wrong,
        }

    sustain_mask = (target["wick_body_ratio"] < 0.5) & (target["nif_whale_rel"] <= 0)
    switching_mask = target["wick_body_ratio"] > 2.0

    result = {
        "n_matched_events_with_features": int(len(target)),
        "tabpfn_overall_accuracy_on_matched_events": overall_acc,
        "sustain_rule_crosscheck": subset_report(sustain_mask, "sustain (wick<0.5 & nif_whale_rel<=0 -> predicts label=0)", 0, 0.8125, 16),
        "switching_rule_crosscheck": subset_report(switching_mask, "switching (wick>2.0 -> predicts label=1)", 1, 0.5510204081632653, 49),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    (OUT_DIR / "tabpfn_vs_cascade_crosscheck_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    target.to_csv(OUT_DIR / "tabpfn_vs_cascade_crosscheck_events.csv", index=False)
    print("TABPFN_CASCADE_CROSSCHECK_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
