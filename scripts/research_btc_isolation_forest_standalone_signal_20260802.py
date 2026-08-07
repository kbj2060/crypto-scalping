#!/usr/bin/env python3
"""Standalone signal-quality check for the newly trained BTC Isolation Forest anomaly detector
(data/ensemble/unsupervised/btc/isolation_forest_btc.pkl, trained on
data/splits/year_oos/btc_features_2026.csv via ensemble/unsupervised/train_isolation_forest.py --
first-ever BTC-trained Isolation Forest; prior artifact at data/ensemble/unsupervised/isolation_forest.pkl
is ETH-only and untouched by this run -- confirmed via git status, no diff).

An Isolation Forest anomaly score (here: -model.decision_function(x), higher = more anomalous) has no
direction -- it cannot generate its own long/short trades. This script mirrors the VAE/GMM sibling
scripts' methodology exactly (scripts/research_btc_vae_anomaly_standalone_signal_20260802.py,
scripts/research_btc_gmm_volatility_standalone_signal_20260802.py): test whether the score correlates
with forward N-bar realized volatility or forward N-bar |return|, per-year AND on the model's own
genuinely-held-out val split, watching specifically for the VAE's failure pattern (correlation looks
fine pooled/other-years but SIGN-FLIPS on the model's own held-out split).

Score is computed causally: at each bar t, the score uses ONLY that bar's own already-computed feature
columns (mean/std/model are frozen from training, no future information enters the bar-t score).
Forward vol/move targets look FORWARD from t (label leakage direction only, never backward) -- this is
fine because we are only testing correlation "does score predict what happens next", never using the
label to build the score itself.

Data coverage: data/splits/year_oos/btc_features_2024_2026.csv (2024-01-01 .. 2026-08-01), so 2024 and
2025 are genuinely out-of-sample for the model (trained only on 2026 Jan-Aug data); within 2026, the
first ~80% (train_ratio, train_isolation_forest.py default) is in-sample/memorized and the last ~20% is
the model's own held-out val split -- reported separately from 2024/2025.

DIAGNOSTIC per CLAUDE.md Fresh-Forward Rule: vectorized pooled/per-year correlation replay, not a
bar-by-bar live walk-forward. Does not touch trading_bot.py or any live wiring.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT_PKL = ROOT / "data/ensemble/unsupervised/btc/isolation_forest_btc.pkl"
DATA_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"
TRAIN_ONLY_PATH = ROOT / "data/splits/year_oos/btc_features_2026.csv"  # to recover train/val cut
OUT_DIR = ROOT / "tmp/research_20260802/btc_isolation_forest_signal_check"
TRAIN_RATIO = 0.8  # must match train_isolation_forest.py default used for this artifact

FORWARD_HORIZONS = {"h6_30m": 6, "h12_1h": 12, "h48_4h": 48, "h288_1d": 288}


def load_model():
    with open(ARTIFACT_PKL, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["feature_cols"], payload["mean"], payload["std"]


def score_frame(df: pd.DataFrame, model, feature_cols, mean, std) -> np.ndarray:
    x_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    x_raw = np.nan_to_num(x_raw, nan=0.0)
    x = (x_raw - mean) / std
    return -model.decision_function(x)  # higher = more anomalous, matches training script convention


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 30:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(x, y)[0, 1])
    if abs(r) >= 1.0 or n <= 2:
        t = float("nan")
    else:
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return r, float(t), n


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, feature_cols, mean, std = load_model()
    with open(ARTIFACT_PKL.with_suffix(".json"), "r") as f:
        meta = json.load(f)["meta"]
    print(f"Loaded {ARTIFACT_PKL}: n_features={len(feature_cols)} feature_cols={feature_cols} "
          f"anomaly_ratio={meta['anomaly_ratio']:.4f} best_params={meta['best_params']}")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing feature columns in combined frame: {missing}")

    df["if_score"] = score_frame(df, model, feature_cols, mean, std)
    df["year"] = df["timestamp"].dt.year

    # mark 2026 train/val split boundary exactly as train_isolation_forest.py would have computed it
    n2026_total = int((pd.read_csv(TRAIN_ONLY_PATH, usecols=["timestamp"]).shape[0]))
    n2026_train = max(10, int(n2026_total * TRAIN_RATIO))
    df_2026 = df[df["year"] == 2026].reset_index(drop=True)
    train_cut_ts = df_2026["timestamp"].iloc[min(n2026_train, len(df_2026) - 1)]
    df["split_2026"] = np.where(
        df["year"] != 2026, "n/a",
        np.where(df["timestamp"] < train_cut_ts, "2026_train_insample", "2026_val_heldout"),
    )
    print(f"2026 train/val cut at {train_cut_ts} (n_train={n2026_train}/{n2026_total})")

    close = df["close"].to_numpy(np.float64)
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))

    from numpy.lib.stride_tricks import sliding_window_view

    n = len(df)
    for label, h in FORWARD_HORIZONS.items():
        fwd_vol = np.full(n, np.nan)
        fwd_absret = np.full(n, np.nan)
        future = log_ret[1:]
        if len(future) >= h:
            windows = sliding_window_view(future, h)
            n_valid = windows.shape[0]
            fwd_ret_valid = windows.sum(axis=1)
            fwd_vol_valid = windows.std(axis=1) * np.sqrt(h)
            fwd_absret_valid = np.abs(fwd_ret_valid)
            fwd_vol[:n_valid] = fwd_vol_valid
            fwd_absret[:n_valid] = fwd_absret_valid
        df[f"fwd_vol_{label}"] = fwd_vol
        df[f"fwd_absret_{label}"] = fwd_absret

    groups = {
        "pooled_all": df,
        "2024": df[df["year"] == 2024],
        "2025": df[df["year"] == 2025],
        "2026_train_insample": df[df["split_2026"] == "2026_train_insample"],
        "2026_val_heldout": df[df["split_2026"] == "2026_val_heldout"],
    }

    rows = []
    for gname, gdf in groups.items():
        for label in FORWARD_HORIZONS:
            for target_col in (f"fwd_vol_{label}", f"fwd_absret_{label}"):
                r, t, n = pearson(gdf["if_score"].to_numpy(), gdf[target_col].to_numpy())
                rows.append({"group": gname, "horizon": label, "target": target_col, "pearson_r": r, "t_stat": t, "n": n})

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(OUT_DIR / "standalone_correlation_table.csv", index=False)

    print("\n########## Pearson correlation: if_score vs forward vol / |forward return| ##########")
    for label in FORWARD_HORIZONS:
        print(f"\n--- horizon {label} ---")
        sub = corr_df[corr_df["horizon"] == label]
        for _, r in sub.iterrows():
            print(f"  {r['group']:<22} {r['target']:<20} r={r['pearson_r']:+.4f} t={r['t_stat']:+.2f} n={int(r['n'])}")

    # sign-flip check: does the held-out split disagree in sign with pooled/2024/2025?
    print("\n########## Sign-stability check (fwd_vol_h12_1h) ##########")
    flip_rows = corr_df[corr_df["target"] == "fwd_vol_h12_1h"]
    signs = {row["group"]: np.sign(row["pearson_r"]) for _, row in flip_rows.iterrows()}
    print(signs)
    heldout_sign = signs.get("2026_val_heldout", float("nan"))
    other_signs = [v for k, v in signs.items() if k != "2026_val_heldout" and not np.isnan(v)]
    sign_flip = bool(other_signs) and any(heldout_sign != s for s in other_signs)
    print(f"held-out sign={heldout_sign}, other-group signs={other_signs}, SIGN_FLIP_DETECTED={sign_flip}")

    # decile check on pooled data for the most economically relevant horizon (1h)
    print("\n########## Decile check: mean fwd_vol_h12_1h by if_score decile (pooled) ##########")
    dsub = df.dropna(subset=["if_score", "fwd_vol_h12_1h"]).copy()
    dsub["decile"] = pd.qcut(dsub["if_score"], 10, labels=False, duplicates="drop")
    decile_stats = dsub.groupby("decile")["fwd_vol_h12_1h"].agg(["mean", "count"]).reset_index()
    decile_stats.to_csv(OUT_DIR / "decile_fwd_vol_h12_1h.csv", index=False)
    print(decile_stats.to_string(index=False))
    monotonic = decile_stats["mean"].is_monotonic_increasing
    print(f"monotonic increasing across deciles: {monotonic}")

    summary = {
        "artifact": str(ARTIFACT_PKL),
        "meta": meta,
        "n_rows_total": len(df),
        "train_cut_2026": str(train_cut_ts),
        "decile_monotonic_fwd_vol_1h": bool(monotonic),
        "sign_flip_detected_h12_1h": sign_flip,
        "group_signs_h12_1h": {k: (None if np.isnan(v) else float(v)) for k, v in signs.items()},
    }
    with open(OUT_DIR / "standalone_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # cache the causal score series (timestamp, if_score) for reuse by the entry-filter backtest
    df[["timestamp", "if_score", "year", "split_2026"]].to_csv(OUT_DIR / "if_score_series_full.csv", index=False)

    print(f"\nWrote {OUT_DIR / 'standalone_correlation_table.csv'}, "
          f"{OUT_DIR / 'decile_fwd_vol_h12_1h.csv'}, {OUT_DIR / 'standalone_summary.json'}, "
          f"{OUT_DIR / 'if_score_series_full.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
