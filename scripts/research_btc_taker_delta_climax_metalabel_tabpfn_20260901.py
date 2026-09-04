#!/usr/bin/env python3
"""Meta-labeling for BTC taker_delta_z_climax -- final TabPFN run using round 2's grid-screen
winner label definition, porting this project's established methodology VERBATIM from
scripts/research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py (this same signal's own ETH
TabPFN script -- the closest possible template, per user instruction).

Provenance:
  - round 1 (scripts/research_btc_taker_delta_climax_gridscreen_20260901.py): HORIZON x K 2D grid,
    touch-based MFE hit definition fixed. Found short horizons beat long ones (opposite of ETH).
  - round 2 (scripts/research_btc_taker_delta_climax_gridscreen_hittype_20260901.py, writeup in
    docs/experiments/btc_5m_taker_delta_climax_gridscreen_featureanalysis_20260901.md): added
    HIT_TYPE as a 3rd axis (4 hit_types x 6 horizons x 5 K = 120 cells). The thick-sample-family
    winner (mechanical global argmax `touch_giveback_sustained` was explicitly flagged as too
    thin/unstable -- TRAIN 109-136 hits, VAL/OOS sign instability observed during development --
    and NOT recommended for adoption) is `close_at_h` @ H=6, K=2.0 (TRAIN lift 1.290x/1.252x,
    VAL 1.305x/1.043x, both directionally consistent, thick samples: TRAIN 574/516 hits).
    This script adopts that winner AS GIVEN (no re-search here).

Label (close_at_h, H=6, K=2.0 -- round 2's winner, more conservative than touch_mfe: only the
bar-6 CLOSE counts, no intrabar-touch-then-reverted credit):
  entry = close[i], atr = atr[i] (Tier0's raw price-unit ATR at the fire bar)
  bottom: hit = 1 if close[i+6] >= entry + 2.0*atr else 0
  top:    hit = 1 if close[i+6] <= entry - 2.0*atr else 0

Data: Tier0 CSV already carries almost the full ETH canonical 23-feature set (built and validated
across rounds 1-2), so unlike the ETH script (which had to run compute_indicators /
add_creative_indicators / add_broad_indicators / compute_signals from raw klines), this script
only adds the handful of genuinely missing columns: is_bottom (trivial, per-fire), atr_pct
(=atr/close -- Tier0 happens to already carry a matching atr_pct column, verified numerically
identical to atr/close from row 13 onward; recomputed here anyway from the `atr` column to
guarantee exact formula parity with the ETH reference regardless of that coincidence),
nyse_open_flag, er_24, realized_vol_ratio -- all 4 formulas ported verbatim from
research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::build_indicator_frame. Final feature
list = ETH's exact 23 plus this Tier0's own `range_width_pct` (informative in round 1/2 analysis),
24 total.

cluster_dedup() is ported VERBATIM (identical code) from the ETH reference script. This is the
FIRST time dedup is applied to this BTC signal -- round 1/2 grid screening explicitly left it
unapplied ("클러스터앵커링(dedup) -- 라운드1과 동일하게 미적용", round 2 doc, "다음 단계" section).

run_tabpfn_panel(), evaluate(), compute_permutation_importance() are ported VERBATIM (identical
code, only FEATURE_COLUMNS/SEEDS/log() references resolve to this module's own copies) from the
same ETH reference script.

Adaptation notes (data-format only, not methodology changes):
  - BTC Tier0 CSV timestamps are tz-aware UTC (`+00:00` suffix); ETH's klines CSV is tz-naive
    (implicitly UTC). Stripped to tz-naive immediately after load so all boundary comparisons
    (VAL_START/OOS_START/HOLDOUT_START, defined as naive Timestamps) behave identically to the ETH
    script -- purely a dtype-normalization step, not a change in what "UTC hour" means anywhere
    (nyse_open_flag, hour_utc, etc. are unaffected).
  - Label threshold uses raw `atr` (price units), matching the user-specified close_at_h formula
    verbatim (`close[i+H] >= entry + K*atr`) -- NOT atr_pct, unlike ETH's label (which compares
    dimensionless MFE ratios against K*atr_pct). atr_pct is still computed and used purely as an
    input FEATURE, never for the label threshold. This mirrors round 1/2's own documented
    convention (`docs/experiments/.../gridscreen_featureanalysis_20260901.md`: "atr 컬럼(=raw
    가격단위 ATR, atr_pct 아님) ... 그대로 사용").

Splits: this repo's Fresh-Forward default (CLAUDE.md) -- TRAIN < 2025-09-01, VAL 2025-09-01..
2026-01-01, OOS 2026-01-01..2026-04-01, HOLDOUT 2026-04-01..latest (single-touch, first exposure
this round). Identical boundaries to the ETH reference script (VAL_START/OOS_START/HOLDOUT_START
are the same 3 dates).

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) under a system-wide flock (single
8GB GPU shared across concurrently running signal-research agents this session) -- see
scripts/ops/handoff.sh push/launch before executing remotely.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

TIER0_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_DIR = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901"
REPORT_PATH = OUT_DIR / "taker_delta_climax_tabpfn_report.json"
FEATURES_CSV_PATH = OUT_DIR / "btc_5m_taker_delta_climax_metalabel_features.csv"

START = pd.Timestamp("2024-01-01")
HORIZON = 6  # round 2 winner: close_at_h, H=6 (bar-6 CLOSE only, no intrabar touch credit)
ATR_HIT_MULT = 2.0  # round 2 winner K=2.0
CLUSTER_GAP_MERGE = 3  # identical constant to the ETH reference's v4 dedup convention

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SEEDS = [20260829, 141592, 271828, 577215]  # identical seed list to the ETH reference script

FEATURE_COLUMNS = [
    # structural / signal-intensity (analog of ETH's sweep-derived group)
    "is_bottom", "delta_z", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    # evidence-signal family, RAW (not side-normalized)
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    # trend/volatility context
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    "rsi",
    # BTC Tier0's own addition (round 1/2 found it informative), not in ETH's 23
    "range_width_pct",
]


def log(msg: str) -> None:
    print(f"[btc_taker_delta_climax_tabpfn] {msg}", flush=True)


def load_tier0() -> pd.DataFrame:
    df = pd.read_csv(TIER0_PATH, parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # tz-aware UTC -> naive UTC, see docstring
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all(), "gap/dup in BTC Tier0 rows"
    return df


def add_missing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the handful of columns not already in Tier0. Formulas ported verbatim from
    research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::build_indicator_frame
    (is_bottom is added later, per-fire, not per-bar)."""
    close = frame["close"]

    frame["atr_pct"] = frame["atr"] / close.clip(lower=1e-12)

    tmin = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    is_weekday = frame["timestamp"].dt.dayofweek < 5
    frame["nyse_open_flag"] = (is_weekday & (tmin >= 12 * 60 + 30) & (tmin <= 14 * 60 + 30)).astype(int)

    net_change_24 = close - close.shift(24)
    diff_abs = close.diff().abs()
    frame["er_24"] = (net_change_24.abs() / (diff_abs.rolling(24, min_periods=4).sum() + 1e-12)).fillna(0.0)

    log_ret = np.log(close / close.shift(1))
    frame["realized_vol_ratio"] = log_ret.rolling(12, min_periods=12).std() / log_ret.rolling(288, min_periods=288).std()

    return frame


def cluster_dedup(idx: np.ndarray, delta_z_at_idx: np.ndarray, most_negative: bool) -> np.ndarray:
    """VERBATIM port from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py.
    Collapse consecutive same-side fires (gap<=CLUSTER_GAP_MERGE bars) into one cluster, keep only
    the bar with the most extreme delta_z per cluster. Causal -- uses only delta_z, never future
    price."""
    order = np.argsort(idx)
    idx_sorted = idx[order]
    dz_sorted = delta_z_at_idx[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > CLUSTER_GAP_MERGE:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "dz": dz_sorted})
    keep = df.loc[df.groupby("cluster")["dz"].idxmin()] if most_negative else df.loc[df.groupby("cluster")["dz"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def build_fires_and_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    close = frame["close"].to_numpy()
    atr = frame["atr"].to_numpy()
    n = len(frame)
    delta_z_all = frame["delta_z"].to_numpy()
    ts_all = frame["timestamp"].to_numpy()
    rows = []
    dedup_stats = {}
    for side, col in [("bottom", "bottom_taker_delta_z_climax"), ("top", "top_taker_delta_z_climax")]:
        idx = np.flatnonzero(frame[col].fillna(False).to_numpy())
        idx = idx[(idx < n - HORIZON) & (ts_all[idx] >= np.datetime64(START))]
        idx_before_dedup = len(idx)
        idx = cluster_dedup(idx, delta_z_all[idx], most_negative=(side == "bottom"))
        log(f"  {side}: {idx_before_dedup} raw fires -> {len(idx)} after cluster-anchor dedup")
        dedup_stats[side] = {"raw": int(idx_before_dedup), "deduped": int(len(idx))}

        entry = close[idx]
        entry_atr = atr[idx]
        close_h = close[idx + HORIZON]
        # close_at_h (round 2 winner): H-bar-ahead CLOSE only, no intrabar touch credit
        if side == "bottom":
            hit = (close_h >= entry + ATR_HIT_MULT * entry_atr).astype(float)
        else:
            hit = (close_h <= entry - ATR_HIT_MULT * entry_atr).astype(float)

        feat_rows = frame.iloc[idx]
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
    return fires, dedup_stats


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
    agnostic (TabPFN has no native .feature_importances_), hand-rolled rather than sklearn's
    permutation_importance to avoid that helper's fitted-estimator/wrapper-class edge cases on a
    non-sklearn-native classifier. VERBATIM port from the ETH reference script."""
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

    log("loading BTC Tier0 CSV...")
    frame = load_tier0()
    log(f"{len(frame)} bars loaded")

    log("adding missing features (atr_pct/nyse_open_flag/er_24/realized_vol_ratio)...")
    frame = add_missing_features(frame)

    log("building taker_delta_z_climax fires + close_at_h(H=6,K=2.0) labels + cluster dedup...")
    fires, dedup_stats = build_fires_and_features(frame)
    n_before = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before} usable fires after dropna "
        f"(bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    fire_hit_rate = float(fires["hit"].mean())
    log(f"pooled hit rate (deduped fires): {fire_hit_rate:.4f}")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, HOLDOUT(>={HOLDOUT_START.date()}) n={len(holdout)}")

    fires.to_csv(FEATURES_CSV_PATH, index=False)
    log(f"features CSV saved -> {FEATURES_CSV_PATH}")

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
        "signal": "taker_delta_z_climax",
        "asset": "BTCUSDT",
        "label_definition": "close_at_h (round 2 grid-screen winner, thick-sample family), H=6, K=2.0, cluster-anchor deduped",
        "provenance": (
            "Round 1: scripts/research_btc_taker_delta_climax_gridscreen_20260901.py (HORIZON x K "
            "grid, touch-based MFE hit fixed). Round 2: "
            "scripts/research_btc_taker_delta_climax_gridscreen_hittype_20260901.py, writeup "
            "docs/experiments/btc_5m_taker_delta_climax_gridscreen_featureanalysis_20260901.md "
            "(added HIT_TYPE as a 3rd grid axis; close_at_h @ H=6,K=2.0 chosen as the defensible "
            "thick-sample winner -- the mechanical global argmax touch_giveback_sustained was "
            "explicitly flagged there as too thin/unstable for adoption). This script: first "
            "TabPFN run for this label, methodology ported verbatim from "
            "scripts/research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py (this same "
            "signal's own ETH TabPFN script), including cluster_dedup (applied here for the FIRST "
            "time on this BTC signal -- rounds 1-2 explicitly left it unapplied)."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "n_fires_total": int(len(fires)),
        "dedup_stats": dedup_stats,
        "fired_signal_hit_rate_pooled": fire_hit_rate,
        "val": val_result,
        "oos": oos_result,
        "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
        "eth_own_result_for_comparison": {
            "note": "same signal (taker_delta_z_climax), ETH, v4 label (touch-based MFE H=24 "
                    "2h window, K=2.0, cluster-deduped) -- different label definition, comparison "
                    "is directional/methodological only, not apples-to-apples on the label itself",
            "val_auc": 0.622, "oos_auc": 0.608, "holdout_auc": 0.650,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
