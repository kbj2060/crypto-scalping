#!/usr/bin/env python3
"""dalton_rule2_balance_edge v1 (HORIZON=30/GAP=12/K=1.90, VAL/OOS/HOLDOUT AUC 0.598/0.605/0.576
with Tier0's 23 features) -- does the richer "Omega" engineered feature pool help, especially given
dalton's OWN defining variables (distance to its 48-bar range edge, its own 288-bar ATR percentile)
are entirely ABSENT from Tier0? Unlike volume_wick_climax (where the Omega pool's conceptual
relevance to that signal was speculative), several Omega columns are directly on-topic here:
`distance_to_day_high_low_pct`, `compression_score`/`compression_release_up/down`,
`bb_width_pct_rank_288`, `atr_pct_rank_288`, `range_contraction_breakout_dir` -- all describe
range-position/compression, exactly what dalton's trigger is about. User request (2026-08-30):
"피쳐를 좀 더 추가하는건 어때? 150개나 있는데 150개 모두 중요도 분석을 해줘" -- add the Omega
features AND run permutation importance across the full combined set (not just an AUC comparison).

Same feature source, same 7 excluded price/time-trend-contaminated raw columns, and the same 2
name-collision suffixes (`rsi`, `realized_vol_ratio` -> `_omega142`) as
research_eth_volume_wick_climax_metalabel_omega_features_20260830.py -- see that script's docstring
for the contamination-check rationale (spearmanr vs price/time-ordinal, |corr|>=0.5 threshold from
feedback_raw_feature_price_trend_contamination), not re-run here since the excluded-column list is
data-independent (it's about the RAW COLUMN's own trend properties, not this signal's fires).

Does NOT change dalton's label at all (same fires, same 'hit' column, same TRAIN/VAL/OOS/HOLDOUT
split as v1) -- purely a feature-set comparison + a full permutation-importance report, same
methodology as the volume_wick_climax precedent (3 configs: tier0_23 / omega_134 / combined_157,
identical matched+dropna row set for all 3).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
FIRES_CSV = ROOT / "data/labels/eth_5m_dalton_rule2_balance_edge_metalabel_20260830/eth_5m_dalton_rule2_balance_edge_metalabel_features.csv"
OMEGA_CSVS = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
REPORT_DIR = ROOT / "tmp/eth_dalton_rule2_balance_edge_metalabel_tabpfn_20260830"

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SEEDS = [20260829, 141592, 271828, 577215]

TIER0_FEATURES = [
    "is_bottom", "delta_z", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    "rsi",
]
EXCLUDED_CONTAMINATED = [
    "open", "high", "low", "close",
    "sum_open_interest_value", "sum_toptrader_long_short_ratio", "last_funding_rate",
]
COLLISION_SUFFIX = "_omega142"
COLLIDING_NAMES = {"realized_vol_ratio", "rsi"}


def log(msg: str) -> None:
    print(f"[dalton_omega_features] {msg}", flush=True)


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[feature_cols], train["hit"].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[feature_cols])[:, 1]
        r = evaluate(proba, eval_df["hit"].to_numpy().astype(int))
        r["seed"] = seed
        seed_rows.append(r)
        log(f"  [{tag}] seed={seed}: auc={r['auc']:.4f} acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f}")
    table = pd.DataFrame(seed_rows)
    return {
        "n_train": int(len(train)), "n_eval": int(len(eval_df)),
        "auc_mean": round(float(table["auc"].mean()), 4), "auc_std": round(float(table["auc"].std(ddof=1)), 4),
        "accuracy_mean": round(float(table["accuracy"].mean()), 4),
        "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
        "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"],
        "per_seed": seed_rows,
    }


def compute_permutation_importance_full(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str],
                                         seed: int = SEEDS[0], n_repeats: int = 5) -> dict:
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
        log(f"  [{len(rows)}/{len(feature_cols)}] {feat:<32s} importance={importance:+.5f}")
    rows.sort(key=lambda r: -r["importance_mean"])
    return {"baseline_auc": round(float(baseline_auc), 4), "n_repeats": n_repeats, "seed": seed, "importances": rows}


def main() -> int:
    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"])
    log(f"loaded {len(fires)} v1 fires (hit/side/tier0-features already fixed, unchanged)")

    omega = pd.concat([pd.read_csv(f, parse_dates=["timestamp"]) for f in OMEGA_CSVS], ignore_index=True)
    omega = omega.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    omega = omega.drop(columns=EXCLUDED_CONTAMINATED)
    omega_feature_cols_raw = [c for c in omega.columns if c != "timestamp"]
    log(f"omega frame: {len(omega)} rows ({omega['timestamp'].min()}~{omega['timestamp'].max()}), "
        f"{len(omega_feature_cols_raw)} features after excluding {len(EXCLUDED_CONTAMINATED)} contaminated raw columns")

    rename_map = {c: c + COLLISION_SUFFIX for c in omega_feature_cols_raw if c in COLLIDING_NAMES}
    omega = omega.rename(columns=rename_map)
    omega_feature_cols = [rename_map.get(c, c) for c in omega_feature_cols_raw]
    log(f"collision-suffixed: {list(rename_map.items())}")

    merged = fires.merge(omega, on="timestamp", how="inner", suffixes=("", "_dup"))
    log(f"merged (inner join on timestamp): {len(merged)}/{len(fires)} fires matched")

    all_needed = list(dict.fromkeys(TIER0_FEATURES + omega_feature_cols))
    n_before_dropna = len(merged)
    merged = merged.dropna(subset=all_needed + ["hit"]).reset_index(drop=True)
    log(f"after dropna on union of all features: {len(merged)}/{n_before_dropna}")

    ts = merged["timestamp"]
    train = merged.loc[ts < VAL_START].reset_index(drop=True)
    val = merged.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = merged.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = merged.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, HOLDOUT n={len(holdout)} "
        f"(all 3 configs below share this EXACT row set)")

    configs = {
        "tier0_23": TIER0_FEATURES,
        "omega_134": omega_feature_cols,
        "combined_157": list(dict.fromkeys(TIER0_FEATURES + omega_feature_cols)),
    }

    results = {}
    for label, feats in configs.items():
        log(f"=== {label} ({len(feats)} features) ===")
        results[label] = {
            "feature_columns": feats,
            "val": run_panel(train, val, feats, f"{label}/VAL"),
            "oos": run_panel(train, oos, feats, f"{label}/OOS"),
            "holdout": run_panel(train, holdout, feats, f"{label}/HOLDOUT"),
        }

    log("=== FULL permutation importance across all 157 features (combined_157, VAL, single seed, 5 repeats) ===")
    perm_full = compute_permutation_importance_full(train, val, configs["combined_157"])

    out = {
        "note": "feature-set comparison + full permutation importance -- label/split unchanged from v1, "
                "all 3 AUC configs share the identical matched+dropna row set",
        "excluded_contaminated_columns": EXCLUDED_CONTAMINATED,
        "collision_renames": rename_map,
        "n_fires_matched": int(len(merged)), "n_fires_total_v1": int(len(fires)),
        "results": results,
        "permutation_importance_combined_157_val": perm_full,
    }
    out_path = REPORT_DIR / "omega_feature_comparison_report.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log(f"saved -> {out_path}")

    log("")
    log("=== SUMMARY (VAL / OOS / HOLDOUT AUC) ===")
    for label in results:
        r = results[label]
        log(f"  {label}: VAL={r['val']['auc_mean']:.4f}  OOS={r['oos']['auc_mean']:.4f}  HOLDOUT={r['holdout']['auc_mean']:.4f}")
    log("")
    log("=== TOP 20 permutation importance (combined_157) ===")
    for row in perm_full["importances"][:20]:
        log(f"  {row['feature']:<32s} importance={row['importance_mean']:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
