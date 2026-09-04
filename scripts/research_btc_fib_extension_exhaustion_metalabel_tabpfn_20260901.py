#!/usr/bin/env python3
"""TabPFN meta-labeling, final round, for BTC's `fib_extension_exhaustion` evidence signal -- this
project's weakest BTC signal across two prior grid-screening rounds. See
docs/experiments/btc_5m_fib_extension_exhaustion_gridscreen_featureanalysis_20260901.md for the
full history:
  - Round 1 (touch-based MFE only): VAL AUC 0.57-0.60, TRAIN joint lift 1.34x, ~600 candidates/side.
  - Round 2 (HIT_TYPE itself grid-searched, 4 families x 5 horizons x 5 K's x 2 sides = 200 cells):
    the apparent global-best cell (touch_giveback_sustained, H=10, K=1.5, TRAIN lift 2.08-2.27x)
    COMPLETELY COLLAPSED on VAL -- bottom lift 1.154 (Wilson CI contains the baseline), top lift
    EXACTLY 1.000 (hit rate identical to the random-bar baseline to 4 decimal places) -- a textbook
    train-only artifact of a 200-cell multiple-comparison search.

This script does NOT re-search anything. It uses round 2's most sample-robust, VAL-surviving grid
point instead of the (VAL-refuted) global-best cell: `close_at_h`, H=10, K=2.75 (TRAIN joint lift
1.696x, n_hits 68/78 bottom/top) -- a strict, order-blind hit definition (no touch-then-credit, no
giveback persistence condition riding on a thin complex-condition tail), the least likely of round
2's top candidates to be a multiple-comparison artifact.

Label (this round's chosen point, taken as given, NOT re-searched): entry=close[i], atr=atr[i]
(Tier0's own absolute-price ATR14 column -- same convention both grid-screen rounds used):
    bottom: hit = 1 if close[i+10] >= entry + 2.75*atr[i] else 0
    top:    hit = 1 if close[i+10] <= entry - 2.75*atr[i] else 0

Cluster-dedup (SIMPLIFICATION, stated explicitly): fib_extension_exhaustion is a zone-touch trigger
(48-bar leg direction + 127.2-161.8% fib-extension zone touch), not a continuous extremity z-score
like taker_delta_z_climax's delta_z -- so there is no natural "most extreme" bar to anchor a
cluster on the way research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py's cluster_dedup
does. Simplest defensible rule instead: collapse same-side fires within GAP=6 bars into one
cluster, keep the LAST bar per cluster (most recent information). Causal (uses only fire
timestamps, never future price).

Features: 19 of the 24 final features are Tier0-canonical bar-wide columns, read AS-IS from the
BTC Tier0 CSV (NOT recomputed): atr_percentile_864, hour_utc, weekday, p_fast, p_slow, ret3_z,
vwap_dev_z, cvd_roll_roc_48, vol_z, lower_wick_ratio, upper_wick_ratio, bb_pctb, adx14, pdi, ndi,
bb_width_pctile, rsi, delta_z, range_width_pct. The other 5 are added here: is_bottom (trivial side
indicator) plus 4 ported VERBATIM (formula-for-formula) from research_eth_taker_delta_climax_
metalabel_tabpfn_20260829.py::build_indicator_frame -- atr_pct, nyse_open_flag, er_24,
realized_vol_ratio. NOTE: the Tier0 CSV has a column literally named `atr_pct`, but it is NOT
reused -- it comes from an earlier, non-Tier0-canonical indicator stage (compute_indicators) and is
NOT part of Tier0's own `bar_wide_features` list (see build_btc_5m_evidence_signal_candidates_
tier0_20260901.py / build_report.json). The task's own formula is atr_pct = atr/close using Tier0's
canonical `atr` column (the same absolute-price ATR14 the hit label itself is defined on) -- that
column is simply never loaded here (see load_tier0's usecols), so there is no name collision at
runtime. All 4 ported formulas are computed on the FULL continuous Tier0 frame (one row per 5m bar,
2024-01-01..2026-08-20) before being indexed down to candidate rows only -- exactly mirroring
build_indicator_frame's own full-series-then-index pattern (all causal: .rolling()/.ewm()/.diff()/
.shift(positive) or same-bar OHLC only).

Model/eval infrastructure -- run_tabpfn_panel(), evaluate(), compute_permutation_importance() --
is ported VERBATIM from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py, this
project's canonical reusable TabPFN scaffolding (TabPFNClassifier device="cuda", 4-seed panels,
hand-rolled AUC-scored permutation importance since TabPFN has no native .feature_importances_).
fib_extension_exhaustion has no dedicated ETH TabPFN reference script beyond this shared
infrastructure (it was historically ETH's "experimental/7th" signal, never TabPFN-trained there).

Splits (Fresh-Forward, this repo's default / CLAUDE.md): TRAIN <2025-09-01, VAL 2025-09-01..
2026-01-01, OOS 2026-01-01..2026-04-01, HOLDOUT >=2026-04-01 -- FIRST TOUCH for this signal (round
1 and round 2 both truncated the loaded frame at HOLDOUT_START and never read it at all).

SMALL-SAMPLE HANDLING (judgment call, not covered by the ported infrastructure): this is this
project's thinnest BTC signal (928/1009 raw lifetime fires before any filtering). Any eval split
with <20 candidates, or with only one class present (AUC undefined), is still given a TabPFN panel
where possible, but flagged low_confidence=true / skipped with a reason in the report -- per this
round's explicit instruction not to present a noisy small-sample AUC as equivalent-confidence to
this project's larger-sample signals. Permutation importance on VAL only runs if VAL has >=20
candidates AND both classes present; otherwise it is skipped with a note.

Runs on the GPU server under a system-wide flock (single shared 8GB GPU) -- NOT run locally (no
GPU on the dev machine). Research-only: does not touch dashboard/, trading_bot.py, or any
live-serving code.
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

DATA_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
REPORT_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/fib_extension_exhaustion_tabpfn_report.json"
FEATURES_CSV_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/fib_extension_exhaustion_tabpfn_features.csv"

# ---- chosen label point (round 2's most sample-robust VAL-surviving cell -- NOT re-searched) ----
HORIZON = 10        # close_at_h: hit evaluated at exactly bar i+HORIZON's close
K_MULT = 2.75        # ATR multiple
CLUSTER_GAP = 6      # bars; simplification for a trigger with no natural extremity ranking

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")

SEEDS = [20260829, 141592, 271828, 577215]  # identical to the canonical ETH TabPFN script
MIN_CONFIDENT_N = 20  # below this, AUC is flagged low_confidence rather than a firm conclusion

FEATURE_COLUMNS = [
    "is_bottom", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi",
    "bb_width_pctile", "er_24", "realized_vol_ratio", "rsi", "delta_z", "range_width_pct",
]

TIER0_NATIVE_FEATURES = [c for c in FEATURE_COLUMNS if c not in
                          ("is_bottom", "atr_pct", "nyse_open_flag", "er_24", "realized_vol_ratio")]


def log(msg: str) -> None:
    print(f"[fib_extension_exhaustion_tabpfn] {msg}", flush=True)


def load_tier0() -> pd.DataFrame:
    usecols = (["timestamp", "close", "atr",
                "bottom_fib_extension_exhaustion", "top_fib_extension_exhaustion"]
               + TIER0_NATIVE_FEATURES)
    df = pd.read_csv(DATA_PATH, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all(), "gap/dup in Tier0 CSV"
    return df


def augment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds atr_pct/nyse_open_flag/er_24/realized_vol_ratio, ported VERBATIM from
    research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::build_indicator_frame. Computed
    on the full continuous frame before candidate-row indexing (see module docstring)."""
    close = df["close"]
    df["atr_pct"] = df["atr"] / close.clip(lower=1e-12)

    tmin = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    is_weekday = df["timestamp"].dt.dayofweek < 5
    df["nyse_open_flag"] = (is_weekday & (tmin >= 12 * 60 + 30) & (tmin <= 14 * 60 + 30)).astype(int)

    net_change_24 = close - close.shift(24)
    diff_abs = close.diff().abs()
    df["er_24"] = (net_change_24.abs() / (diff_abs.rolling(24, min_periods=4).sum() + 1e-12)).fillna(0.0)

    log_ret = np.log(close / close.shift(1))
    df["realized_vol_ratio"] = log_ret.rolling(12, min_periods=12).std() / log_ret.rolling(288, min_periods=288).std()

    return df


def cluster_dedup_last(idx: np.ndarray, gap: int) -> np.ndarray:
    """Collapse same-side fires within `gap` bars of each other into one cluster, keep only the
    LAST (most recent) bar per cluster. SIMPLIFICATION vs the canonical taker_delta_climax script's
    cluster_dedup (which keeps the most-extreme-delta_z bar): fib_extension_exhaustion has no
    natural single continuous metric to rank cluster members by, so "most recent" is the simplest
    defensible rule. Causal -- uses only fire-bar positions, never future price."""
    idx_sorted = np.sort(idx)
    if len(idx_sorted) == 0:
        return idx_sorted
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    keep = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id}).groupby("cluster")["idx"].max()
    return np.sort(keep.to_numpy())


def build_fires_and_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    n = len(df)
    elig = ~np.isnan(atr) & (atr > 0) & ~np.isnan(close)

    rows = []
    dedup_log: dict = {}
    for side, col in [("bottom", "bottom_fib_extension_exhaustion"), ("top", "top_fib_extension_exhaustion")]:
        raw_idx = np.flatnonzero(df[col].fillna(False).to_numpy() & elig)
        raw_idx = raw_idx[raw_idx + HORIZON < n]
        n_before = len(raw_idx)
        idx = cluster_dedup_last(raw_idx, CLUSTER_GAP)
        n_after = len(idx)
        log(f"  {side}: {n_before} eligible raw fires -> {n_after} after cluster-dedup (GAP={CLUSTER_GAP})")
        dedup_log[side] = {"n_before_dedup": int(n_before), "n_after_dedup": int(n_after)}

        entry = close[idx]
        a = atr[idx]
        target = close[idx + HORIZON]
        if side == "bottom":
            hit = (target >= entry + K_MULT * a).astype(float)
        else:
            hit = (target <= entry - K_MULT * a).astype(float)

        feat_rows = df.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "hit": hit, "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)

    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return fires, dedup_log


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    """Ported VERBATIM from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py."""
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    """Ported VERBATIM from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py."""
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
    """Ported VERBATIM from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py."""
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


def safe_run_panel(train: pd.DataFrame, eval_df: pd.DataFrame, tag: str) -> dict:
    """Wraps run_tabpfn_panel (kept verbatim) with the small-sample / single-class guarding this
    round's task explicitly asked for -- NOT part of the ported infrastructure, added here as this
    signal's own judgment call given its unusually thin sample."""
    n_eval = len(eval_df)
    n_hits = int(eval_df["hit"].sum()) if n_eval else 0
    if n_eval == 0:
        return {"n_train": int(len(train)), "n_eval": 0, "n_hits_eval": 0,
                "skipped": True, "low_confidence": True, "reason": "no candidates in this split"}
    if eval_df["hit"].nunique() < 2:
        return {"n_train": int(len(train)), "n_eval": int(n_eval), "n_hits_eval": n_hits,
                "skipped": True, "low_confidence": True,
                "reason": f"only one class present in {tag} (n={n_eval}, hits={n_hits}) -- AUC undefined"}
    result = run_tabpfn_panel(train, eval_df, FEATURE_COLUMNS, tag)
    result["n_hits_eval"] = n_hits
    result["low_confidence"] = bool(n_eval < MIN_CONFIDENT_N)
    if result["low_confidence"]:
        log(f"  [{tag}] LOW-CONFIDENCE FLAG: n_eval={n_eval} < {MIN_CONFIDENT_N} -- treat AUC as indicative-only")
    return result


def main() -> int:
    log("loading BTC Tier0 candidate CSV (fib_extension_exhaustion-relevant columns only)...")
    df = load_tier0()
    log(f"{len(df)} rows loaded, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    log("augmenting with atr_pct / nyse_open_flag / er_24 / realized_vol_ratio (ported verbatim)...")
    df = augment_features(df)

    log(f"raw trigger fires (whole loaded frame, pre-eligibility): "
        f"bottom={int(df['bottom_fib_extension_exhaustion'].sum())} top={int(df['top_fib_extension_exhaustion'].sum())}")

    log(f"building fires + labels (close_at_h, HORIZON={HORIZON}, K={K_MULT}) + cluster-dedup (GAP={CLUSTER_GAP})...")
    fires, dedup_log = build_fires_and_features(df)

    n_before_dropna = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before_dropna} usable fires after dropna "
        f"(bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    fires.to_csv(FEATURES_CSV_PATH, index=False)
    log(f"features CSV written -> {FEATURES_CSV_PATH}")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)} (hits={int(train['hit'].sum())}), "
        f"VAL n={len(val)} (hits={int(val['hit'].sum())}), "
        f"OOS n={len(oos)} (hits={int(oos['hit'].sum())}), "
        f"HOLDOUT(>={HOLDOUT_START.date()}, FIRST TOUCH) n={len(holdout)} (hits={int(holdout['hit'].sum())})")
    assert train["hit"].nunique() == 2, "TRAIN has only one class -- cannot fit TabPFN"

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = safe_run_panel(train, val, "VAL")
    if "auc_mean" in val_result:
        log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f} "
            f"(low_confidence={val_result['low_confidence']}, n_eval={val_result['n_eval']})")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = safe_run_panel(train, oos, "OOS")
    if "auc_mean" in oos_result:
        log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f} "
            f"(low_confidence={oos_result['low_confidence']}, n_eval={oos_result['n_eval']})")

    log("=== RESERVED HOLDOUT evaluation (2026-04-01~latest, FIRST TOUCH, TRAIN-fit, 4 seeds) ===")
    holdout_result = safe_run_panel(train, holdout, "HOLDOUT")
    if "auc_mean" in holdout_result:
        log(f"HOLDOUT -> AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f} "
            f"(low_confidence={holdout_result['low_confidence']}, n_eval={holdout_result['n_eval']})")

    log("=== permutation feature importance (VAL, single seed, AUC-scored, 5 repeats) ===")
    if len(val) >= MIN_CONFIDENT_N and val["hit"].nunique() >= 2:
        perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
        log(f"baseline VAL AUC (single seed {perm_importance['seed']}): {perm_importance['baseline_auc']:.4f}")
        for row in perm_importance["importances"][:10]:
            log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")
    else:
        perm_importance = {"skipped": True,
                            "reason": f"VAL n={len(val)} (<{MIN_CONFIDENT_N}) or single-class -- "
                                      "permutation importance unreliable/undefined at this sample size"}
        log(f"SKIPPED: {perm_importance['reason']}")

    report = {
        "signal": "fib_extension_exhaustion",
        "asset": "BTC",
        "status": "final_tabpfn_read_after_two_gridscreen_rounds",
        "chosen_point": {
            "hit_type": "close_at_h", "horizon_bars": HORIZON, "k_atr_mult": K_MULT,
            "source": ("round 2 grid screen's most sample-robust VAL-surviving point (TRAIN joint "
                       "lift 1.696x, n_hits 68/78 bottom/top) -- NOT the round-2 global-best cell "
                       "(touch_giveback_sustained, H=10, K=1.5), which collapsed on VAL (top lift "
                       "exactly 1.000). See docs/experiments/"
                       "btc_5m_fib_extension_exhaustion_gridscreen_featureanalysis_20260901.md"),
        },
        "cluster_dedup": {
            "gap_bars": CLUSTER_GAP,
            "rule": ("keep LAST bar per cluster -- SIMPLIFICATION, this trigger has no natural "
                     "continuous extremity metric (unlike delta_z for taker_delta_climax) to rank "
                     "cluster members by"),
            "by_side": dedup_log,
        },
        "feature_columns": FEATURE_COLUMNS,
        "n_candidates_total": int(len(fires)),
        "splits": {
            "train": {"n": int(len(train)), "n_hits": int(train["hit"].sum()),
                       "hit_rate": round(float(train["hit"].mean()), 4)},
            "val": {"n": int(len(val)), "n_hits": int(val["hit"].sum()),
                     "hit_rate": round(float(val["hit"].mean()), 4) if len(val) else None},
            "oos": {"n": int(len(oos)), "n_hits": int(oos["hit"].sum()),
                     "hit_rate": round(float(oos["hit"].mean()), 4) if len(oos) else None},
            "holdout": {"n": int(len(holdout)), "n_hits": int(holdout["hit"].sum()),
                        "hit_rate": round(float(holdout["hit"].mean()), 4) if len(holdout) else None},
        },
        "holdout_single_touch": True,
        "holdout_note": "first time this signal's HOLDOUT (>=2026-04-01) has been read by any script -- round 1 and round 2 both truncated the loaded frame before this date.",
        "val": val_result,
        "oos": oos_result,
        "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
        "min_confident_n": MIN_CONFIDENT_N,
        "fresh_forward_bar_by_bar": False,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "fresh_forward_note": ("N/A -- this is a label-separability TabPFN AUC screen (candidate "
                                "hit/no-hit classification), not a bar-by-bar TP/SL backtest. Same "
                                "convention round 1/round 2's reports used for this signal."),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
