#!/usr/bin/env python3
"""Standalone signal-quality check for the newly trained BTC VAE anomaly detector
(data/ensemble/unsupervised/btc/vae_anomaly_btc.pkl, trained on
data/splits/year_oos/btc_features_2026.csv via ensemble/unsupervised/train_vae_anomaly.py --
first-ever BTC-trained VAE, prior artifact at data/ensemble/unsupervised/vae_anomaly.pkl is ETH-only).

A VAE reconstruction-error anomaly score has no direction -- it cannot generate its own long/short
trades. This script tests whether the score correlates with anything predictively useful: forward
N-bar realized volatility and forward N-bar absolute return (proxy for "big move coming, in either
direction"). Methodology mirrors scripts/research_btc_cryptomamba_standalone_h6_signal_20260802.py's
per-year stability discipline (pooled correlation alone is not trusted; must hold up per-year and
not be a single-year fluke).

Score is computed causally: at each bar t, the VAE reconstruction error uses ONLY that bar's own
already-computed feature columns (mean/std/model weights are all frozen from training, no future
information enters the bar-t score). Forward vol/move targets look FORWARD from t (label leakage
direction only, never backward) -- this is fine because we are only testing correlation "does score
predict what happens next", never using the label to build the score itself.

Data coverage: data/splits/year_oos/btc_features_2024_2026.csv (2024-01-01 .. 2026-08-01), so 2024
and 2025 are genuinely out-of-sample for the model (trained only on 2026 Jan-Aug data); within 2026,
the first ~85% (train_ratio) is in-sample/memorized and the last ~15% is the model's own held-out
val split -- both are reported separately from 2024/2025 to avoid conflating in-sample fit quality
with genuine generalization.

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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.unsupervised.train_vae_anomaly import VAE  # noqa: E402

ARTIFACT_PKL = ROOT / "data/ensemble/unsupervised/btc/vae_anomaly_btc.pkl"
DATA_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"
TRAIN_ONLY_PATH = ROOT / "data/splits/year_oos/btc_features_2026.csv"  # to recover train/val cut
OUT_DIR = ROOT / "tmp/research_20260802/btc_vae_anomaly_signal_check"
TRAIN_RATIO = 0.85  # must match train_vae_anomaly.py default used for this artifact

FORWARD_HORIZONS = {"h6_30m": 6, "h12_1h": 12, "h48_4h": 48, "h288_1d": 288}


def load_model():
    with open(ARTIFACT_PKL, "rb") as f:
        payload = pickle.load(f)
    feature_cols = payload["feature_cols"]
    mean = payload["mean"]
    std = payload["std"]
    meta = payload["meta"]
    model = VAE(input_dim=len(feature_cols), latent_dim=int(meta["latent_dim"]), hidden_dim=int(meta["hidden_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, feature_cols, mean, std, payload["threshold"], meta


def score_frame(df: pd.DataFrame, model, feature_cols, mean, std) -> np.ndarray:
    x_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    x_raw = np.nan_to_num(x_raw, nan=0.0)
    x = (x_raw - mean) / std
    with torch.no_grad():
        xt = torch.from_numpy(x.astype(np.float32))
        recon, _, _ = model(xt)
        err = torch.mean((recon - xt) ** 2, dim=1).numpy()
    return err


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 30:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(x, y)[0, 1])
    # t-stat for Pearson r
    if abs(r) >= 1.0 or n <= 2:
        t = float("nan")
    else:
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return r, float(t), n


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, feature_cols, mean, std, threshold, meta = load_model()
    print(f"Loaded {ARTIFACT_PKL}: latent_dim={meta['latent_dim']} hidden_dim={meta['hidden_dim']} "
          f"val_anomaly_ratio={meta['val_anomaly_ratio']:.4f} threshold={threshold:.6f} "
          f"n_features={len(feature_cols)}")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing feature columns in combined frame: {missing}")

    df["vae_score"] = score_frame(df, model, feature_cols, mean, std)
    df["year"] = df["timestamp"].dt.year

    # mark 2026 train/val split boundary exactly as train_vae_anomaly.py would have computed it
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
        future = log_ret[1:]  # future[i] == log_ret[i+1], so windows[i] == log_ret[i+1:i+1+h]
        if len(future) >= h:
            windows = sliding_window_view(future, h)  # shape (n-1-h+1, h), row i -> log_ret[i+1:i+1+h]
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
                r, t, n = pearson(gdf["vae_score"].to_numpy(), gdf[target_col].to_numpy())
                rows.append({"group": gname, "horizon": label, "target": target_col, "pearson_r": r, "t_stat": t, "n": n})

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(OUT_DIR / "standalone_correlation_table.csv", index=False)

    print("\n########## Pearson correlation: vae_score vs forward vol / |forward return| ##########")
    for label in FORWARD_HORIZONS:
        print(f"\n--- horizon {label} ---")
        sub = corr_df[corr_df["horizon"] == label]
        for _, r in sub.iterrows():
            print(f"  {r['group']:<22} {r['target']:<20} r={r['pearson_r']:+.4f} t={r['t_stat']:+.2f} n={int(r['n'])}")

    # decile check on pooled data for the most economically relevant horizon (1h)
    print("\n########## Decile check: mean fwd_vol_h12_1h by vae_score decile (pooled) ##########")
    dsub = df.dropna(subset=["vae_score", "fwd_vol_h12_1h"]).copy()
    dsub["decile"] = pd.qcut(dsub["vae_score"], 10, labels=False, duplicates="drop")
    decile_stats = dsub.groupby("decile")["fwd_vol_h12_1h"].agg(["mean", "count"]).reset_index()
    decile_stats.to_csv(OUT_DIR / "decile_fwd_vol_h12_1h.csv", index=False)
    print(decile_stats.to_string(index=False))
    monotonic = decile_stats["mean"].is_monotonic_increasing
    print(f"monotonic increasing across deciles: {monotonic}")

    summary = {
        "artifact": str(ARTIFACT_PKL),
        "meta": {k: v for k, v in meta.items() if k != "best_params"},
        "n_rows_total": len(df),
        "train_cut_2026": str(train_cut_ts),
        "decile_monotonic_fwd_vol_1h": bool(monotonic),
    }
    with open(OUT_DIR / "standalone_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # cache the causal score series (timestamp, vae_score) for reuse by the entry-filter backtest
    df[["timestamp", "vae_score", "year", "split_2026"]].to_csv(OUT_DIR / "vae_score_series_full.csv", index=False)

    print(f"\nWrote {OUT_DIR / 'standalone_correlation_table.csv'}, "
          f"{OUT_DIR / 'decile_fwd_vol_h12_1h.csv'}, {OUT_DIR / 'standalone_summary.json'}, "
          f"{OUT_DIR / 'vae_score_series_full.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
