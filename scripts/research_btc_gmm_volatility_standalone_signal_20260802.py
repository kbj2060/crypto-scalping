#!/usr/bin/env python3
"""Standalone signal-quality check for the newly trained BTC GMM volatility-regime model
(data/ensemble/unsupervised/btc/gmm_volatility_btc.pkl, trained on
data/splits/year_oos/btc_features_2026.csv via ensemble/unsupervised/train_gmm_volatility.py --
first-ever BTC-trained GMM regime model, prior artifact at data/ensemble/unsupervised/gmm_volatility.pkl
is ETH-only). Direct follow-up to the just-closed BTC VAE anomaly investigation
(scripts/research_btc_vae_anomaly_standalone_signal_20260802.py) -- same methodology, adapted for a
discrete regime label instead of a continuous reconstruction-error score.

Pitfall avoided (confirmed by reading code before running): train_gmm_volatility.py's shared
training-results cache is keyed only on os.path.dirname(save_path) with hash validation disabled
(ensemble/optuna_helper.py load_reusable_results always reuses if the file exists, regardless of
data/config hash) -- so --save-path was pointed at an ISOLATED asset-specific directory
(data/ensemble/unsupervised/btc/gmm_volatility_btc.pkl) from the start, never touching the ETH
artifact's directory or its gmm_volatility_training_results.json.

rl_path optionality: confirmed by reading ensemble/supervised/common.py::load_feature_frame -- it
merges rl_path only `if os.path.exists(rl_path)`. Training run used a deliberately nonexistent
--rl-path to skip the merge (BTC has no rl_base file, and pointing at ETH's rl_base_2024.csv caused
an inner join with a disjoint date range and 0 rows -- caught and fixed before the real training run).

The GMM outputs a per-bar cluster label (0..5) plus a cluster_rank_map that orders clusters by mean
z-scored value of the first (highest-variance) volatility feature, i.e. cluster_rank 0 = lowest
average volatility, cluster_rank 5 = highest. This script tests whether cluster_rank (and the raw
posterior probability of the assigned cluster, `gmm_confidence`) says anything about forward N-bar
realized volatility, forward N-bar absolute return, and regime persistence (how long a regime lasts
once entered) -- checked per-year and on the model's own held-out val split, mirroring the VAE
script's stability discipline (a pooled correlation alone is not trusted).

Score is computed causally: at each bar t, cluster assignment uses ONLY that bar's own already-
computed feature columns (mean/std/model params frozen from training). Forward vol/move targets look
FORWARD from t only for measuring correlation, never for building the assignment itself.

Data coverage: data/splits/year_oos/btc_features_2024_2026.csv (2024-01-01 .. 2026-08-01). 2024/2025
are genuinely out-of-sample for the model (trained only on 2026 Jan-Aug data, train_ratio=0.8); within
2026 the first ~80% is in-sample/memorized (Optuna val split during tuning used the same cut) and the
last ~20% is the model's own held-out split -- reported separately from 2024/2025.

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

ARTIFACT_PKL = ROOT / "data/ensemble/unsupervised/btc/gmm_volatility_btc.pkl"
DATA_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"
TRAIN_ONLY_PATH = ROOT / "data/splits/year_oos/btc_features_2026.csv"  # to recover train/val cut
OUT_DIR = ROOT / "tmp/research_20260802/btc_gmm_volatility_signal_check"
TRAIN_RATIO = 0.8  # must match train_gmm_volatility.py default used for this artifact (unmodified)

FORWARD_HORIZONS = {"h6_30m": 6, "h12_1h": 12, "h48_4h": 48, "h288_1d": 288}


def load_model():
    with open(ARTIFACT_PKL, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["feature_cols"], payload["mean"], payload["std"], payload["cluster_rank_map"]


def score_frame(df: pd.DataFrame, model, feature_cols, mean, std, cluster_rank_map):
    x_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    x_raw = np.nan_to_num(x_raw, nan=0.0)
    x = (x_raw - mean) / std
    labels = model.predict(x)
    probs = model.predict_proba(x)
    rank_map = {int(k): int(v) for k, v in cluster_rank_map.items()}
    cluster_rank = np.array([rank_map[int(lbl)] for lbl in labels])
    confidence = probs[np.arange(len(labels)), labels]
    return labels, cluster_rank, confidence


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


def regime_persistence(labels: np.ndarray) -> dict:
    """Mean run-length (in bars) of consecutive identical regime labels."""
    if len(labels) == 0:
        return {"mean_run_len": float("nan"), "n_runs": 0}
    change = np.where(np.diff(labels) != 0)[0]
    boundaries = np.concatenate(([0], change + 1, [len(labels)]))
    run_lens = np.diff(boundaries)
    return {"mean_run_len": float(np.mean(run_lens)), "n_runs": int(len(run_lens))}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, feature_cols, mean, std, cluster_rank_map = load_model()
    n_components = model.n_components
    print(f"Loaded {ARTIFACT_PKL}: n_components={n_components} n_features={len(feature_cols)} "
          f"cluster_rank_map={cluster_rank_map}")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing feature columns in combined frame: {missing}")

    labels, cluster_rank, confidence = score_frame(df, model, feature_cols, mean, std, cluster_rank_map)
    df["gmm_label"] = labels
    df["gmm_cluster_rank"] = cluster_rank
    df["gmm_confidence"] = confidence
    df["year"] = df["timestamp"].dt.year

    # mark 2026 train/val split boundary exactly as train_gmm_volatility.py would have computed it
    n2026_total = int(pd.read_csv(TRAIN_ONLY_PATH, usecols=["timestamp"]).shape[0])
    n2026_train = max(10, int(n2026_total * TRAIN_RATIO))
    n2026_train = min(n2026_train, n2026_total - 1)
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

    fwd_ret_h12 = np.full(n, np.nan)
    h12 = FORWARD_HORIZONS["h12_1h"]
    if len(future) >= h12:
        windows12 = sliding_window_view(future, h12)
        fwd_ret_h12[: windows12.shape[0]] = windows12.sum(axis=1)
    df["fwd_ret_h12_1h"] = fwd_ret_h12

    groups = {
        "pooled_all": df,
        "2024": df[df["year"] == 2024],
        "2025": df[df["year"] == 2025],
        "2026_train_insample": df[df["split_2026"] == "2026_train_insample"],
        "2026_val_heldout": df[df["split_2026"] == "2026_val_heldout"],
    }

    # (1) correlation of cluster_rank / confidence with forward vol & |forward return|
    rows = []
    for gname, gdf in groups.items():
        for label in FORWARD_HORIZONS:
            for target_col in (f"fwd_vol_{label}", f"fwd_absret_{label}"):
                for score_col in ("gmm_cluster_rank", "gmm_confidence"):
                    r, t, n_ = pearson(gdf[score_col].to_numpy(dtype=float), gdf[target_col].to_numpy())
                    rows.append({"group": gname, "horizon": label, "target": target_col,
                                 "score": score_col, "pearson_r": r, "t_stat": t, "n": n_})
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(OUT_DIR / "standalone_correlation_table.csv", index=False)

    print("\n########## Pearson correlation: gmm_cluster_rank / gmm_confidence vs forward vol / |ret| ##########")
    for label in FORWARD_HORIZONS:
        print(f"\n--- horizon {label} ---")
        sub = corr_df[(corr_df["horizon"] == label) & (corr_df["target"] == f"fwd_vol_{label}")]
        for _, r in sub.iterrows():
            print(f"  {r['group']:<22} {r['score']:<18} r={r['pearson_r']:+.4f} t={r['t_stat']:+.2f} n={int(r['n'])}")

    # (2) mean realized forward vol by cluster_rank (does rank actually order volatility out-of-sample?)
    print("\n########## Mean fwd_vol_h12_1h by gmm_cluster_rank, per group ##########")
    rank_rows = []
    for gname, gdf in groups.items():
        gsub = gdf.dropna(subset=["fwd_vol_h12_1h"])
        stats = gsub.groupby("gmm_cluster_rank")["fwd_vol_h12_1h"].agg(["mean", "count"]).reset_index()
        stats["group"] = gname
        rank_rows.append(stats)
        monotonic = stats.sort_values("gmm_cluster_rank")["mean"].is_monotonic_increasing
        print(f"[{gname}] monotonic increasing rank->vol: {monotonic}")
        print(stats.to_string(index=False))
    rank_df = pd.concat(rank_rows, ignore_index=True)
    rank_df.to_csv(OUT_DIR / "fwd_vol_by_cluster_rank.csv", index=False)

    # (3) regime persistence per year + held-out split
    print("\n########## Regime persistence (run-length in bars) ##########")
    persist_rows = []
    for gname, gdf in groups.items():
        p = regime_persistence(gdf["gmm_label"].to_numpy())
        p["group"] = gname
        persist_rows.append(p)
        print(f"  {gname:<22} mean_run_len={p['mean_run_len']:.2f} bars  n_runs={p['n_runs']}")
    persist_df = pd.DataFrame(persist_rows)
    persist_df.to_csv(OUT_DIR / "regime_persistence.csv", index=False)

    # (4) forward return SIGN by cluster label (does any specific regime predict directional forward return?)
    print("\n########## Mean fwd_ret (signed, h12_1h) by gmm_label, per group ##########")
    signed_rows = []
    for gname, gdf in groups.items():
        gsub = gdf.dropna(subset=["fwd_ret_h12_1h"])
        stats = gsub.groupby("gmm_label")["fwd_ret_h12_1h"].agg(["mean", "count"]).reset_index()
        stats["group"] = gname
        signed_rows.append(stats)
    signed_df = pd.concat(signed_rows, ignore_index=True)
    signed_df.to_csv(OUT_DIR / "signed_fwd_ret_by_label.csv", index=False)
    for gname in groups:
        sub = signed_df[signed_df["group"] == gname].sort_values("gmm_label")
        print(f"[{gname}]")
        print(sub.to_string(index=False))

    summary = {
        "artifact": str(ARTIFACT_PKL),
        "n_components": int(n_components),
        "cluster_rank_map": {str(k): int(v) for k, v in cluster_rank_map.items()},
        "n_rows_total": len(df),
        "train_cut_2026": str(train_cut_ts),
    }
    with open(OUT_DIR / "standalone_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # cache the causal score series for reuse by the entry-filter backtest
    df[["timestamp", "gmm_label", "gmm_cluster_rank", "gmm_confidence", "year", "split_2026"]].to_csv(
        OUT_DIR / "gmm_score_series_full.csv", index=False
    )

    print(f"\nWrote outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
