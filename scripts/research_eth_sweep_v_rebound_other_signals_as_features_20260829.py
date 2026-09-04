#!/usr/bin/env python3
"""Corrected re-test of "would combining other evidence signals help" after the user's valid
pushback: this repo's OWN prior finding is NOT a blanket "stacking always dilutes" -- it's
"combining REDUNDANT signals (sharing the same underlying leg, e.g. sweep_flow_combo/smt_flow_
combo sharing taker-delta with taker_delta_z_climax) dilutes, but combining ORTHOGONAL signals
CAN amplify" (wick_body_ratio+obi: holdout precision 62.5%->83.3%; wick_body_ratio+nif_whale_rel:
->88.9%, see eth_liquidation_cascade_sweep_vs_trend_pilot_20260828.md). The pooling pilot just run
(dalton_rule2 events mixed INTO the liquidity_sweep TRAINING POPULATION) tested a fundamentally
DIFFERENT operation than that successful pattern -- population-pooling across two different
trigger types, not feature-level fusion for ONE trigger type. This script tests the CORRECT
analogy to the successful pattern instead: keep the liquidity_sweep event population UNCHANGED
(same 14,259 events, same TRAIN/VAL/OOS, no context-size confound this time), and add "were the
OTHER 7 dashboard evidence signals ALSO firing (same side) at this exact bar" as 7 new binary
features on top of Tier0+rsi.

Reuses live_evidence_signal_dashboard_20260823.py::compute_signals() VERBATIM (the actual live
8-signal computation, not reimplemented) to get bottom_{name}/top_{name} for all 8 signals across
the full history, then looks up each existing V_REBOUND event's own timestamp.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TIER0_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
DASHBOARD_SCRIPT = ROOT / "scripts/live_evidence_signal_dashboard_20260823.py"
ETH_SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_SOURCE = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
RSI_SOURCES = [ROOT / f"data/splits/year_oos/training_features_{y}.csv" for y in ("2024", "2025", "2026_rebuilt")]

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
LABEL_WINDOW = pd.Timedelta(minutes=30)
SEEDS = [20260829, 141592, 271828, 577215]

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
OTHER_SIGNALS = [
    "orthogonal_combo", "volume_wick_climax", "short_term_return_z", "taker_delta_z_climax",
    "smt_divergence", "fib_extension_exhaustion", "dalton_rule2_balance_edge",
]  # all 8 minus liquidity_sweep itself (always True for our own event population by construction)


def load_dashboard():
    spec = importlib.util.spec_from_file_location("dashboard_signals_20260829", DASHBOARD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_klines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def main() -> int:
    dash = load_dashboard()
    eth = load_klines(ETH_SOURCE)
    btc = load_klines(BTC_SOURCE)
    print(f"computing all 8 signals via compute_signals() (verbatim reuse) on {len(eth)} bars...")
    signals = dash.compute_signals(eth, btc_df=btc)  # no funding_df -- orthogonal_combo bottom degrades gracefully

    tier0 = pd.read_csv(TIER0_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)

    other_cols = []
    for name in OTHER_SIGNALS:
        for side_label, col in (("bottom", f"bottom_{name}"), ("top", f"top_{name}")):
            other_cols.append(col)
    sig_lookup = signals[["timestamp"] + other_cols].copy()

    df = tier0.merge(sig_lookup, on="timestamp", how="left")
    is_down = df["is_downside"] == 1
    for name in OTHER_SIGNALS:
        feat_col = f"other_{name}_same_side"
        df[feat_col] = np.where(is_down, df[f"bottom_{name}"], df[f"top_{name}"]).astype(float)
    OTHER_FEATURES = [f"other_{name}_same_side" for name in OTHER_SIGNALS]

    frames = []
    for p in RSI_SOURCES:
        f = pd.read_csv(p, usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    rsi = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    df = df.merge(rsi, on="timestamp", how="left")

    variants = {
        "tier0+rsi (baseline)": TIER0 + ["rsi"],
        "tier0+rsi+other7signals": TIER0 + ["rsi"] + OTHER_FEATURES,
    }

    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    for name, feature_cols in variants.items():
        sub = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
        s_ts = sub["timestamp"]
        s_we = s_ts + LABEL_WINDOW
        train = sub.loc[(s_ts < VAL_START) & (s_we < VAL_START)]
        val = sub.loc[(s_ts >= VAL_START) & (s_ts <= VAL_END) & (s_we < OOS_START)]
        oos = sub.loc[(s_ts >= OOS_START) & (s_ts <= OOS_END)]
        print(f"\n=== {name} ({len(feature_cols)} features, train n={len(train)}, val n={len(val)}, oos n={len(oos)}) ===")
        seed_rows = []
        for seed in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=seed)
            clf.fit(train[feature_cols], train["label"].to_numpy())
            row = {"seed": seed}
            for split_name, split in (("val", val), ("oos", oos)):
                proba = clf.predict_proba(split[feature_cols])[:, 1]
                row[f"{split_name}_auc"] = round(float(roc_auc_score(split["label"], proba)), 4)
            seed_rows.append(row)
            print(f"  seed={seed}: val_auc={row['val_auc']:.4f} oos_auc={row['oos_auc']:.4f}")
        table = pd.DataFrame(seed_rows)
        print(f"  -> VAL {table['val_auc'].mean():.4f}+/-{table['val_auc'].std(ddof=1):.4f}  "
              f"OOS {table['oos_auc'].mean():.4f}+/-{table['oos_auc'].std(ddof=1):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
