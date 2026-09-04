#!/usr/bin/env python3
"""One-off: measure real single-inference TabPFN latency on the server GPU before designing the
live dashboard signal's cache TTL -- flagged as an open question in eth_liquidity_sweep_v_rebound_
feature_plan_20260829.md ("실제 1회 추론 지연시간을 아직 서버에서 측정 안 해봤습니다")."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
TIER0_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
RSI_SOURCES = [ROOT / f"data/splits/year_oos/training_features_{y}.csv" for y in ("2024", "2025", "2026_rebuilt")]

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]


def main() -> int:
    tier0 = pd.read_csv(TIER0_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    frames = []
    for p in RSI_SOURCES:
        f = pd.read_csv(p, usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    rsi = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    df = tier0.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    train = df.loc[df["timestamp"] < pd.Timestamp("2025-09-01", tz="UTC")]
    print("train n=", len(train))

    t0 = time.time()
    clf = TabPFNClassifier(device="cuda", random_state=20260829)
    clf.fit(train[FEATURES], train["label"].to_numpy())
    t1 = time.time()
    print(f"fit time: {t1 - t0:.2f}s")

    for n_test in (1, 5, 48):
        t2 = time.time()
        clf.predict_proba(train[FEATURES].iloc[:n_test])
        t3 = time.time()
        print(f"predict_proba(n={n_test}): {t3 - t2:.2f}s")

    t4 = time.time()
    clf2 = TabPFNClassifier(device="cuda", random_state=20260829)
    clf2.fit(train[FEATURES], train["label"].to_numpy())
    clf2.predict_proba(train[FEATURES].iloc[:1])
    t5 = time.time()
    print(f"full cold fit+predict(n=1) cycle: {t5 - t4:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
