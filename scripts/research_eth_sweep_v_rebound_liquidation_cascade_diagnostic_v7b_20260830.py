#!/usr/bin/env python3
"""Re-run of research_eth_sweep_v_rebound_liquidation_cascade_diagnostic_20260829.py's cross-check
("does the liquidation-cascade switching/sustain rule tell TabPFN anything it doesn't already
know?"), now against v7b instead of the old v4 model that first cross-check used. User request
(2026-08-30): "청산 스위치 지속 지표와 지금 이 유동성 스윕 v자 반등 지표를 다시 성능 비교해줘".

Original (v4) finding: sustain-call (wick<0.5 & nif_whale_rel<=0) was 100% redundant with v4's own
predictions; switching-call (wick>2.0) was a genuine blind spot -- v4's own accuracy on that subset
fell to 48.6% (worse than random) while the rule caught real reversals v4 missed. v7b's label
redesign improved general AUC by +0.07-0.10 over v4 but changed nothing about the FEATURE set
(still Tier0+rsi) -- the open question is whether that label-only improvement also closed the
feature-driven blind spot, or left it exactly where it was.

Reuses the ORIGINAL matched population verbatim (data/research/eth_liquidation_cascade_sweep_vs_
trend_pilot_20260828/vrebound_matched_events.csv, 122 rows -- the causal hawkes-cascade replay +
genuine_breach filter + same-bar-and-side join to a real V_REBOUND sweep event) instead of
re-running the hawkes replay, since that population is a property of the cascade/sweep timestamps
alone and does not depend on which V_REBOUND label version is used to score it afterward.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "data/research/eth_liquidation_cascade_sweep_vs_trend_pilot_20260828"
MATCHED_CSV = PILOT_DIR / "vrebound_matched_events.csv"
LABEL_DIR = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829"
VREBOUND_LABELS = LABEL_DIR / "eth_5m_sweep_v_rebound_labels.csv"
TIER0_FULL_CSV = LABEL_DIR / "eth_5m_sweep_v_rebound_features_tier0.csv"
TIER0_V7B_CSV = LABEL_DIR / "eth_5m_sweep_v_rebound_features_tier0_v7b_20260830.csv"
TRAIN_CONTEXT_CSV = LABEL_DIR / "tabpfn_train_context_frozen_v7b_20260830.csv"

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]
SEEDS = [20260829, 141592, 271828, 577215]


def load_rsi() -> pd.DataFrame:
    frames = []
    for y in ("2024", "2025", "2026_rebuilt"):
        f = pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{y}.csv", usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")


def main() -> int:
    matched = pd.read_csv(MATCHED_CSV)
    matched["bar_timestamp"] = pd.to_datetime(matched["bar_timestamp"], utc=True)
    print(f"original matched population (cascade x V_REBOUND sweep, same bar+side): {len(matched)}")

    vrebound_idx = pd.read_csv(VREBOUND_LABELS, usecols=["candidate_index", "timestamp", "side"])
    vrebound_idx["timestamp"] = pd.to_datetime(vrebound_idx["timestamp"], utc=True)
    matched = matched.merge(
        vrebound_idx, left_on=["bar_timestamp", "side"], right_on=["timestamp", "side"], how="inner"
    )
    print(f"resolved candidate_index for: {len(matched)}")

    tier0 = pd.read_csv(TIER0_FULL_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    rsi = load_rsi()
    tier0 = tier0.drop(columns=["label"]).merge(rsi, on="timestamp", how="left")

    scored = matched.merge(tier0, on=["candidate_index", "side"], how="left", suffixes=("", "_tier0"))
    scored = scored.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"feature-complete (Tier0+rsi available): {len(scored)}  "
          f"(original v4 cross-analysis used 88 of these same events)")

    v7b_labels = pd.read_csv(TIER0_V7B_CSV, usecols=["candidate_index", "side", "label"])
    v7b_labels = v7b_labels.rename(columns={"label": "label_v7b"})
    scored = scored.merge(v7b_labels, on=["candidate_index", "side"], how="left")
    n_defined = int(scored["label_v7b"].notna().sum())
    print(f"of these, v7b has a DEFINED (non-excluded) ground truth for: {n_defined}/{len(scored)} "
          f"({n_defined / len(scored):.1%}) -- the rest fall in v7b's excluded fuzzy middle")

    train = pd.read_csv(TRAIN_CONTEXT_CSV)
    probas = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURES], train["label"].to_numpy())
        probas.append(clf.predict_proba(scored[FEATURES])[:, 1])
        print(f"  seed={seed} done")
    scored["v7b_proba"] = np.mean(probas, axis=0)
    scored["v7b_call"] = (scored["v7b_proba"] >= 0.5).astype(int)

    defined = scored.dropna(subset=["label_v7b"]).copy()
    defined["label_v7b"] = defined["label_v7b"].astype(int)
    overall_acc = float((defined["v7b_call"] == defined["label_v7b"]).mean())
    print(f"\nv7b model's OWN accuracy on this matched (cascade-cooccurring) subpopulation: "
          f"{overall_acc:.1%} (n={len(defined)})")

    def report_rule(name: str, mask: pd.Series, predicts_label: int) -> None:
        d = defined[mask]
        n = len(d)
        if n == 0:
            print(f"\n{name}: n=0 (no eligible events)")
            return
        precision = float((d["label_v7b"] == predicts_label).mean())
        naive = float((defined["label_v7b"] == predicts_label).mean())
        agree_with_v7b_call = float((d["v7b_call"] == predicts_label).mean())
        v7b_acc_on_subset = float((d["v7b_call"] == d["label_v7b"]).mean())
        print(f"\n{name} (n={n}):")
        print(f"  rule precision (vs v7b ground truth): {precision:.1%}  "
              f"(naive base rate for this label in the subpopulation: {naive:.1%})")
        print(f"  rule's call vs v7b model's OWN call -- agreement rate: {agree_with_v7b_call:.1%}")
        print(f"  v7b model's OWN accuracy restricted to this subset: {v7b_acc_on_subset:.1%}")

    sustain_mask = (defined["wick_body_ratio"] < 0.5) & (defined["nif_whale_rel"] <= 0)
    report_rule("지속콜 (wick<0.5 & nif_whale_rel<=0, predicts label=0/지지횡보)", sustain_mask, 0)

    switch_mask = defined["wick_body_ratio"] > 2.0
    report_rule("스위칭콜 (wick>2.0, predicts label=1/V자반등)", switch_mask, 1)

    scored.to_csv(PILOT_DIR / "vrebound_v7b_diagnostic_scored.csv", index=False)
    print(f"\nsaved: {PILOT_DIR / 'vrebound_v7b_diagnostic_scored.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
