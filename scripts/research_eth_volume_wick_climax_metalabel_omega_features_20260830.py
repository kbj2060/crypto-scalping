#!/usr/bin/env python3
"""volume_wick_climax v1 (HORIZON=2h/GAP=3/K=1.90, VAL/OOS/HOLDOUT AUC 0.612/0.563/0.565 with the
Tier0 23-feature set) -- does the much richer "Omega" engineered feature pool reveal predictive
information Tier0 is missing? User request (2026-08-30): "오메가 모델에 썼던 150여개 피쳐들 이걸로도
테스트해줘". This does NOT change the label at all (same fires, same 'hit' column, same
TRAIN/VAL/OOS/HOLDOUT split as v1) -- purely a feature-set swap/augmentation, evaluated the same way
the group ablation was (full VAL/OOS/HOLDOUT panel per config, no config selected using HOLDOUT
feedback).

Feature source: `data/splits/year_oos/training_features_{2024,2025,2026_rebuilt}.csv` -- the
already-materialized, pre-merged feature frame the Omega model family (FeatureEngineer.process() +
raw OI/funding/toptrader/BTC) is trained from (142 cols incl. timestamp, on the same regular 5m ETH
grid as Tier0). Coverage 2024-01-01~2026-08-19 -- 99.9%/100%/100%/96.7% of TRAIN/VAL/OOS/HOLDOUT
fires have a matching row (HOLDOUT loses 16/486 fires from 2026-08-20~08-27, past this file's
cutoff); all 3 feature-set configs below are evaluated on the IDENTICAL matched+dropna row set, so
the AUC comparison is apples-to-apples with each other (not directly with v1's original 0.612/
0.563/0.565, which used the full unmatched fire set -- Tier0 is re-run here on the same matched
subset as its own fair baseline).

Excluded 7 of the 142 raw columns for PRICE/TIME-TREND CONTAMINATION
(feedback_raw_feature_price_trend_contamination -- spearmanr(feature,close) / spearmanr(feature,
time-ordinal) over the full 2024-2026 combined frame, disqualifying threshold |corr|>=0.5, checked
locally before this script was written, not re-checked here):
  open/high/low/close: corr(price)=1.000 (trivial self-correlation)
  sum_open_interest_value: corr(time)=0.617 (OI notional has grown over 2+ years)
  sum_toptrader_long_short_ratio: corr(price)=0.629
  last_funding_rate: corr(time)=-0.559 (funding regime drift)
Kept despite being "raw levels" because the SAME check found them clean (|corr|<0.5 both ways):
volume, quote_volume, trades, taker_buy_base, taker_buy_quote, count_long_short_ratio, close_btc,
volume_btc, quote_volume_btc -- category (raw vs. engineered) alone doesn't predict contamination,
per that memory's own explicit warning; verified each individually rather than excluding by category.
134 features remain (141 non-timestamp columns - 7 excluded).

2 exact name collisions with Tier0's 23 features (realized_vol_ratio, rsi -- independently computed
by each pipeline) are suffixed `_omega142` in the combined config to avoid silent column overwrite.

3 configs, identical to the ablation script's reporting style: tier0_23 (baseline, re-run on the
matched subset), omega_134 (Omega features only, Tier0 fully replaced), combined_155
(tier0_23 + omega_134, deduped/suffixed).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIRES_CSV = ROOT / "data/labels/eth_5m_volume_wick_climax_metalabel_20260830/eth_5m_volume_wick_climax_metalabel_features.csv"
OMEGA_CSVS = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
REPORT_DIR = ROOT / "tmp/eth_volume_wick_climax_metalabel_tabpfn_20260830"

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
    print(f"[vwc_omega_features] {msg}", flush=True)


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    from tabpfn import TabPFNClassifier
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

    out = {
        "note": "feature-set comparison only -- label/split unchanged from v1, all 3 configs share "
                "the identical matched+dropna row set (fair comparison), HOLDOUT here is 470 rows "
                "(96.7% of v1's original 486, capped by Omega feature source's 2026-08-19 cutoff)",
        "excluded_contaminated_columns": EXCLUDED_CONTAMINATED,
        "collision_renames": rename_map,
        "n_fires_matched": int(len(merged)), "n_fires_total_v1": int(len(fires)),
        "results": results,
    }
    out_path = REPORT_DIR / "omega_feature_comparison_report.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log(f"saved -> {out_path}")

    log("")
    log("=== SUMMARY (VAL / OOS / HOLDOUT AUC) ===")
    for label in results:
        r = results[label]
        log(f"  {label}: VAL={r['val']['auc_mean']:.4f}  OOS={r['oos']['auc_mean']:.4f}  HOLDOUT={r['holdout']['auc_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
