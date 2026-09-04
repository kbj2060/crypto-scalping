#!/usr/bin/env python3
"""TabPFN meta-labeling for BTC liquidity_sweep, using round 2's grid-screen winner label
definition (touch_giveback_sustained, H=20, K=2.0, giveback<=0.20) -- see
docs/experiments/btc_5m_liquidity_sweep_gridscreen_featureanalysis_20260901.md for the full
grid-screen writeup this label was selected from (round 1: touch-based MFE grid, HIT definition
fixed at touch_mfe; round 2: added HIT_TYPE as a 3rd grid axis after a user pushback on assuming
one fixed HIT definition -- touch_giveback_sustained at H=20/K=2.0 won TRAIN+VAL joint lift by a
clear margin over the other 3 HIT_TYPEs, TRAIN joint 1.371 / VAL joint 1.333 vs 1.06-1.11 for the
rest -- see that doc's HIT_TYPE leaderboard table). Formula cross-checked against the actual
round-2 script (scripts/research_btc_liquidity_sweep_gridscreen_hittype_20260901.py::giveback_hit)
before writing this, not just the prose doc.

This script ports this project's established TabPFN methodology (multi-seed panel + hand-rolled
permutation importance + reserved-holdout single exposure), mirroring
scripts/research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py's evaluate()/
run_tabpfn_panel()/compute_permutation_importance() verbatim (same function bodies, same 4 SEEDS).

Unlike that reference script, this one does NOT recompute Tier0 indicators from raw klines --
BTC's Tier0 CSV (data/labels/btc_5m_evidence_signal_candidates_20260901/
btc_5m_evidence_signal_candidates_tier0.csv) already carries nearly the entire feature set as
precomputed columns (round 1/2's grid screens already used them this way). Only 4 columns are
missing and built fresh here, copying the exact formulas from the reference script's
build_indicator_frame: nyse_open_flag, er_24, realized_vol_ratio, and atr_pct (recomputed as
atr/close from Tier0's own `atr` column -- the CSV already has a differently-derived `atr_pct`
column of its own, which is intentionally overwritten here per the task's explicit formula, to
keep the feature consistent with the `atr` column the label definition itself uses).

Label definition (round 2 winner, reused exactly, not re-searched):
  FAST_WINDOW = H = 20, FULL_WINDOW = 2*H = 40, K = 2.0, giveback ceiling = 0.20.
  entry=close[i], atr=atr[i] (Tier0's own absolute-price ATR14 column, not atr_pct).
  bottom: fast_move = close[i+1:i+21].max()-entry; peak = high[i+1:i+41].max();
          end_price = close[i+40]; denom = peak-entry; giveback = (peak-end_price)/denom
  top:    fast_move = entry-close[i+1:i+21].min(); peak = low[i+1:i+41].min();
          end_price = close[i+40]; denom = entry-peak; giveback = (end_price-peak)/denom
  hit = 1 if fast_move/atr >= K and giveback <= 0.20 else 0
  (denom is provably > 0 whenever fast_mult>=K -- peak is a max/min over a strict superset window
  using high/low, which always dominates a max/min over the fast window using close -- so the
  giveback division is only ever NaN-guarded for rows that already fail the fast_mult condition
  and can't become hit=1 regardless; matches round 2's own documented reasoning.)

Cluster-dedup (this project's established practice for metalabel training, added here -- round
2's grid screen explicitly skipped it, "GAP ... 이번 스크립트에서 아예 다루지 않았다"): collapse
same-side fires within GAP=6 bars into one cluster, keep only the bar with the largest same-bar
sweep-penetration depth ((sweep_level_low[i]-low[i]) for bottom, (high[i]-sweep_level_high[i]) for
top) -- causal, uses only the fire bar's own OHLC/sweep-level columns, never future price. Ported
from the reference script's cluster_dedup (same greedy gap-based clustering), generalized to a
single "pick max extremity" mode since penetration depth is a magnitude for both sides (unlike
delta_z's signed most-negative/most-positive split there).

Splits: this repo's Fresh-Forward default (CLAUDE.md): TRAIN <2025-09-01, VAL 2025-09-01..
2026-01-01, OOS 2026-01-01..2026-04-01, HOLDOUT >=2026-04-01 (first exposure for this signal --
rounds 1-2 never touched it; single-touch evaluation here, only after VAL/OOS are already done
informing nothing further).

This is label-separability + TabPFN generalization evaluation (fixed-window MFE/giveback label vs
TRAIN-fit classifier predict_proba on held-out time splits), not a bar-by-bar TP/SL trade-ledger
walk-forward -- same disclosure precedent as round 1/2's grid-screen reports for this signal (see
their fresh_forward_bar_by_bar=false note). No trade ledger, no saved exit timestamps, no future
rows joined into any row's own features -- only the label's own hit/giveback computation looks
forward from its fire bar, same as every other TabPFN metalabel script in this repo.

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) under a repo-wide flock
(.tabpfn_gpu.lock) since only one 8GB GPU is shared across concurrently-running evidence-signal
research sessions this day.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_DIR = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901"

H = 20                # FAST_WINDOW, round 2 grid-screen winner
FULL_WINDOW = 2 * H    # 40
K = 2.0
GIVEBACK_CEILING = 0.20
CLUSTER_GAP = 6        # bars; collapse same-side fires within this gap into one cluster

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")

SEEDS = [20260829, 141592, 271828, 577215]  # same 4 seeds as every ETH reference script

FEATURE_COLUMNS = [
    "is_bottom", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    "rsi", "atr", "range_width_pct",
]


def log(msg: str) -> None:
    print(f"[btc_liquidity_sweep_tabpfn] {msg}", flush=True)


def load_tier0() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all(), "gap/dup in Tier0 csv"
    return df


def add_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """The 4 features build_indicator_frame() computes in the ETH reference script that are NOT
    already usable columns in the BTC Tier0 CSV. Formulas copied verbatim from
    research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::build_indicator_frame."""
    close = df["close"]

    atr_pct_recomputed = df["atr"] / close.clip(lower=1e-12)
    corr = df["atr_pct"].corr(atr_pct_recomputed)
    log(f"atr_pct recompute (atr/close) vs Tier0 CSV's own atr_pct column: corr={corr:.4f} "
        f"(recompute used per task spec -- overwrites the CSV's version)")
    df["atr_pct"] = atr_pct_recomputed

    tmin = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    is_weekday = df["timestamp"].dt.dayofweek < 5
    df["nyse_open_flag"] = (is_weekday & (tmin >= 12 * 60 + 30) & (tmin <= 14 * 60 + 30)).astype(int)

    net_change_24 = close - close.shift(24)
    diff_abs = close.diff().abs()
    df["er_24"] = (net_change_24.abs() / (diff_abs.rolling(24, min_periods=4).sum() + 1e-12)).fillna(0.0)

    log_ret = np.log(close / close.shift(1))
    df["realized_vol_ratio"] = log_ret.rolling(12, min_periods=12).std() / log_ret.rolling(288, min_periods=288).std()
    return df


def cluster_dedup(idx: np.ndarray, extremity_at_idx: np.ndarray, gap: int = CLUSTER_GAP) -> np.ndarray:
    """Collapse same-side fires within `gap` bars of each other into one cluster, keep only the
    bar with the LARGEST extremity per cluster (extremity here is always a positive-is-more-
    extreme magnitude -- sweep penetration depth -- unlike the ETH reference's signed delta_z, so
    this always picks argmax, no most_negative/most_positive split needed). Causal: extremity is
    computed from the fire bar's own OHLC/sweep-level columns only, never future price."""
    order = np.argsort(idx)
    idx_sorted = idx[order]
    ext_sorted = extremity_at_idx[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    tmp = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "ext": ext_sorted})
    keep = tmp.loc[tmp.groupby("cluster")["ext"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def build_fires_and_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    sweep_level_low = df["sweep_level_low"].to_numpy()
    sweep_level_high = df["sweep_level_high"].to_numpy()
    n = len(df)

    dedup_counts: dict = {}
    rows = []
    for side, trig_col in [("bottom", "bottom_liquidity_sweep"), ("top", "top_liquidity_sweep")]:
        idx = np.flatnonzero(df[trig_col].fillna(False).to_numpy())
        idx = idx[idx < n - FULL_WINDOW]
        n_before = len(idx)

        if side == "bottom":
            extremity = sweep_level_low[idx] - low[idx]
        else:
            extremity = high[idx] - sweep_level_high[idx]
        idx = cluster_dedup(idx, extremity)
        n_after = len(idx)
        dedup_counts[side] = {"before": int(n_before), "after": int(n_after)}
        log(f"  {side}: {n_before} raw fires -> {n_after} after cluster-dedup (gap={CLUSTER_GAP})")

        entry = close[idx]
        atr_i = atr[idx]
        if side == "bottom":
            fast_move = np.array([close[i + 1:i + H + 1].max() for i in idx]) - entry
            peak = np.array([high[i + 1:i + FULL_WINDOW + 1].max() for i in idx])
            end_price = close[idx + FULL_WINDOW]
            denom = peak - entry
            denom_safe = np.where(denom > 1e-12, denom, np.nan)
            giveback = (peak - end_price) / denom_safe
        else:
            fast_move = entry - np.array([close[i + 1:i + H + 1].min() for i in idx])
            peak = np.array([low[i + 1:i + FULL_WINDOW + 1].min() for i in idx])
            end_price = close[idx + FULL_WINDOW]
            denom = entry - peak
            denom_safe = np.where(denom > 1e-12, denom, np.nan)
            giveback = (end_price - peak) / denom_safe

        fast_mult = fast_move / atr_i
        hit = (fast_mult >= K) & (giveback <= GIVEBACK_CEILING)

        feat_rows = df.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "hit": hit.astype(float), "fast_mult": fast_mult, "giveback": giveback,
            "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)

    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return fires, dedup_counts


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    from tabpfn import TabPFNClassifier
    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[feature_cols], train["hit"].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[feature_cols])[:, 1]
        r = evaluate(proba, eval_df["hit"].to_numpy().astype(int))
        r["seed"] = seed
        seed_rows.append(r)
        log(f"  [{tag}] seed={seed}: auc={r['auc']:.4f} acc={r['accuracy']:.4f} "
            f"bal_acc={r['balanced_accuracy']:.4f} (naive={r['naive_majority_accuracy']:.4f})")
    table = pd.DataFrame(seed_rows)
    return {
        "n_train": int(len(train)), "n_eval": int(len(eval_df)),
        "auc_mean": round(float(table["auc"].mean()), 4), "auc_std": round(float(table["auc"].std(ddof=1)), 4),
        "accuracy_mean": round(float(table["accuracy"].mean()), 4),
        "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
        "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"],
        "per_seed": seed_rows,
    }


def compute_permutation_importance(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str],
                                    seed: int = SEEDS[0], n_repeats: int = 5) -> dict:
    """Single-seed, hand-rolled permutation importance (AUC-scored) on the VAL set -- model-
    agnostic (TabPFN has no native .feature_importances_), ported verbatim from the reference
    script's compute_permutation_importance."""
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(device="cuda", random_state=seed)
    clf.fit(train[feature_cols], train["hit"].to_numpy().astype(int))
    y = eval_df["hit"].to_numpy().astype(int)
    X = eval_df[feature_cols].to_numpy()
    baseline_auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])

    rng = np.random.default_rng(seed)
    rows = []
    for j, feat in enumerate(feature_cols):
        shuffled_aucs = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            shuffled_aucs.append(roc_auc_score(y, clf.predict_proba(X_perm)[:, 1]))
        importance = baseline_auc - np.mean(shuffled_aucs)
        rows.append({"feature": feat, "importance_mean": round(float(importance), 5),
                     "importance_std": round(float(np.std(shuffled_aucs, ddof=1)), 5)})
    rows.sort(key=lambda r: -r["importance_mean"])
    return {"baseline_auc": round(float(baseline_auc), 4), "n_repeats": n_repeats, "seed": seed, "importances": rows}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading BTC Tier0 csv...")
    df = load_tier0()
    log(f"{len(df)} bars loaded, range {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}")

    log("adding missing features (atr_pct recompute, nyse_open_flag, er_24, realized_vol_ratio)...")
    df = add_missing_features(df)

    log("building liquidity_sweep fires (touch_giveback_sustained, H=20/K=2.0/giveback<=0.20) + cluster-dedup...")
    fires, dedup_counts = build_fires_and_features(df)
    n_before_dropna = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before_dropna} usable fires after dropna "
        f"(bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, "
        f"HOLDOUT(>={HOLDOUT_START.date()}) n={len(holdout)}")
    for split_name, split_df in [("TRAIN", train), ("VAL", val), ("OOS", oos), ("HOLDOUT", holdout)]:
        if len(split_df) == 0:
            continue
        log(f"  {split_name} hit rate: overall={split_df['hit'].mean():.4f} "
            f"bottom={split_df.loc[split_df['side']=='bottom','hit'].mean():.4f} "
            f"top={split_df.loc[split_df['side']=='top','hit'].mean():.4f}")

    fires.to_csv(OUT_DIR / "btc_5m_liquidity_sweep_metalabel_features.csv", index=False)

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}  "
        f"acc {val_result['accuracy_mean']:.4f}  bal_acc {val_result['balanced_accuracy_mean']:.4f}")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}  "
        f"acc {oos_result['accuracy_mean']:.4f}  bal_acc {oos_result['balanced_accuracy_mean']:.4f}")

    log("=== RESERVED HOLDOUT evaluation (2026-04-01~latest, single-touch, TRAIN-fit, 4 seeds) ===")
    holdout_result = run_tabpfn_panel(train, holdout, FEATURE_COLUMNS, "HOLDOUT") if len(holdout) >= 30 else {"note": "too few holdout fires"}
    if "auc_mean" in holdout_result:
        log(f"HOLDOUT -> AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f}  "
            f"acc {holdout_result['accuracy_mean']:.4f}  bal_acc {holdout_result['balanced_accuracy_mean']:.4f}")

    log("=== permutation feature importance (VAL, single seed, AUC-scored, 5 repeats) ===")
    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    log(f"baseline VAL AUC (single seed {perm_importance['seed']}): {perm_importance['baseline_auc']:.4f}")
    for row in perm_importance["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")

    report = {
        "signal": "liquidity_sweep",
        "asset": "BTCUSDT",
        "label_definition": {
            "hit_type": "touch_giveback_sustained",
            "fast_window_h": H, "full_window": FULL_WINDOW, "k": K,
            "giveback_ceiling": GIVEBACK_CEILING,
            "source": "round 2 grid-screen winner, reused exactly, not re-searched -- see "
                      "docs/experiments/btc_5m_liquidity_sweep_gridscreen_featureanalysis_20260901.md",
        },
        "cluster_dedup": {"gap_bars": CLUSTER_GAP, "counts": dedup_counts},
        "feature_columns": FEATURE_COLUMNS,
        "n_fires_total": int(len(fires)),
        "split_sizes": {"train": int(len(train)), "val": int(len(val)), "oos": int(len(oos)), "holdout": int(len(holdout))},
        "val": val_result,
        "oos": oos_result,
        "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
        "fresh_forward_bar_by_bar": False,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "fresh_forward_note": (
            "This is label-separability + TabPFN generalization evaluation (fixed-window MFE/"
            "giveback label vs TRAIN-fit classifier predict_proba on held-out time splits), not a "
            "bar-by-bar TP/SL trade-ledger walk-forward -- same scope/disclosure precedent as "
            "round 1/2's grid-screen reports for this same signal. No trade ledger, no saved exit "
            "timestamps, no future rows joined into any row's own features (features are read "
            "from the fire bar i itself; only the label's hit/giveback computation looks forward, "
            "exactly like every other TabPFN metalabel script in this repo)."
        ),
    }
    out_path = OUT_DIR / "liquidity_sweep_tabpfn_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
