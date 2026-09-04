#!/usr/bin/env python3
"""One-off latency check: how long does a single-seed TabPFN fit+predict take with the 6000-row
frozen live-serving context (tabpfn_train_context_frozen_multitrigger_v1_20260831.csv), scoring a
single new row -- must comfortably fit inside the 60s dashboard cache window alongside klines
fetch + feature computation. Not part of the live script itself."""
import time
from pathlib import Path

import pandas as pd
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/tabpfn_train_context_frozen_multitrigger_v1_20260831.csv"
FEATURES = ["is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864", "range_width_pct",
            "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z", "p_fast", "p_slow", "ret3_z",
            "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio", "upper_wick_ratio",
            "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "rsi"]

train = pd.read_csv(CSV)
print(f"train n={len(train)}")
t0 = time.time()
clf = TabPFNClassifier(device="cuda", random_state=20260829)
clf.fit(train[FEATURES], train["label"].to_numpy())
t1 = time.time()
print(f"fit: {t1-t0:.2f}s")
proba = clf.predict_proba(train[FEATURES].iloc[:1])
t2 = time.time()
print(f"predict(1 row): {t2-t1:.2f}s, proba={proba}")
print(f"total: {t2-t0:.2f}s")
