#!/usr/bin/env python3
"""Sigma3 1h nested purged walk-forward, 163-feature (38 base + 125 secondary) dataset.

Fixes a methodological problem found in train_sigma3_1h_purged_walkforward_secondary_20260801.py:
that script grid-searched 27 configs directly against VAL (2025-09..12) using the final full-pool
model, which found 16/27 "passing" configs -- but the SAME champion config's 7-fold walk-forward
result was still unstable (3/7 cost1 positive, 3/7 cost3 positive, worst fold right before VAL).
That contradiction is the classic signature of overfitting a wide config grid to a single holdout
window, especially with 163 features giving the model much more capacity to happen to fit VAL's
specific pattern (same failure mode as this project's Sigma6 regime-filter: passed VAL, 0/9
survived OOS).

Nested design (agreed with user): use the 7 folds as the INNER validation loop to select a config,
never touching VAL for selection. Only the config(s) that generalize across folds are then checked
against VAL exactly once (a real holdout, no grid search there). OOS stays untouched regardless.

  1. For each of 7 folds: retrain on that fold's purged training data (163 features), predict a
     tape for the test window, apply ALL 27 pre-registered configs to that one tape (cheap -- no
     retraining per config), record cost1/cost3 pnl/mdd/trades for each (fold, config) pair.
  2. Aggregate per config across all 7 folds: fraction of folds cost1>0, fraction cost3>0, mean
     cost3 pnl, worst-fold cost3 mdd. Select config(s) passing a fold-level bar (>=5/7 folds
     cost3>0, worst-fold mdd >= -20%) -- selection uses ONLY fold data, VAL is never read here.
  3. Train the final model on the full pool (2024-01..2025-08, purged 48h before VAL), generate
     one VAL tape, and score ONLY the fold-selected config(s) on VAL (single look, not a sweep).
  4. OOS (2026-01-01..03-31) is not read anywhere in this script.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

DATA_PATH = ROOT / "tmp/causal_regen_20260516/sigma3_1h_trendscan_continuous_secondary_20260801/sigma3_1h_continuous_secondary.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_nested_walkforward_secondary_20260801"

NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
HORIZON = pd.Timedelta(hours=48)

POOL_START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")

FOLDS = [
    ("2024-06-30 23:59:59", "2024-07-01", "2024-08-31 23:59:59"),
    ("2024-08-31 23:59:59", "2024-09-01", "2024-10-31 23:59:59"),
    ("2024-10-31 23:59:59", "2024-11-01", "2024-12-31 23:59:59"),
    ("2024-12-31 23:59:59", "2025-01-01", "2025-02-28 23:59:59"),
    ("2025-02-28 23:59:59", "2025-03-01", "2025-04-30 23:59:59"),
    ("2025-04-30 23:59:59", "2025-05-01", "2025-06-30 23:59:59"),
    ("2025-06-30 23:59:59", "2025-07-01", "2025-08-31 23:59:59"),
]

THRESHOLDS = [0.50, 0.60, 0.70]
PERSISTS = [0, 2, 4]
TPSL = [(1.5, 1.0), (2.0, 0.9), (2.5, 1.2)]
COOLDOWN = 3
MAX_HOLD = 48
CONFIGS = list(itertools.product(THRESHOLDS, PERSISTS, TPSL))


def load_pool() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def fit_hgb(train: pd.DataFrame, feat_cols: list[str], seed: int) -> HistGradientBoostingClassifier:
    Xtr = train[feat_cols].to_numpy(dtype=np.float64)
    ytr = train["ts_action"].to_numpy(dtype=np.int64)
    w = np.clip(np.abs(train["ts_t_value"].to_numpy(dtype=np.float64)), 0.5, 12.0)
    clf = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.03, max_iter=400, max_depth=4,
        l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=80,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
        random_state=int(seed), class_weight="balanced",
    )
    clf.fit(Xtr, ytr, sample_weight=w)
    return clf


def predict_tape(clf: HistGradientBoostingClassifier, frame: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    proba = clf.predict_proba(frame[feat_cols].to_numpy(dtype=np.float64))
    cls = list(clf.classes_)
    col_for = {c: i for i, c in enumerate(cls)}
    p_cash = proba[:, col_for[0]] if 0 in col_for else np.zeros(len(frame))
    p_long = proba[:, col_for[1]] if 1 in col_for else np.zeros(len(frame))
    p_short = proba[:, col_for[2]] if 2 in col_for else np.zeros(len(frame))
    probs = np.column_stack([p_cash, p_long, p_short])
    dir_action = probs.argmax(axis=1)
    qual = np.where(dir_action > 0, probs[np.arange(len(frame)), dir_action], probs[:, 0])
    DEFAULT_THR = 0.45
    final_action = np.where((dir_action != 0) & (qual >= DEFAULT_THR), dir_action, 0)
    side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))
    return pd.DataFrame({
        "i": np.arange(len(frame)), "timestamp": frame["timestamp"].to_numpy(),
        "open": frame["open"].astype(float).to_numpy(), "high": frame["high"].astype(float).to_numpy(),
        "low": frame["low"].astype(float).to_numpy(), "close": frame["close"].astype(float).to_numpy(),
        "atr_pct": frame["atr_pct"].astype(float).to_numpy(),
        "primary_action": final_action, "primary_side": side,
        "primary_route_confidence": 1.0, "primary_route_margin": 1.0,
        "primary_dir_p_cash": p_cash, "primary_dir_p_long": p_long, "primary_dir_p_short": p_short,
        "primary_quality_p_cash": p_cash, "primary_quality_p_long": p_long, "primary_quality_p_short": p_short,
        "primary_quality_score": np.where(final_action != 0, qual, 0.0),
        "primary_confidence": probs.max(axis=1),
        "fallback_action": 0, "fallback_side": 0,
        "fallback_route_confidence": 0.0, "fallback_route_margin": 0.0,
        "fallback_dir_p_cash": 1.0, "fallback_dir_p_long": 0.0, "fallback_dir_p_short": 0.0,
        "fallback_quality_p_cash": 1.0, "fallback_quality_p_long": 0.0, "fallback_quality_p_short": 0.0,
        "fallback_confidence": 0.0,
    })


def score_all_configs(tape: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    rows = []
    for thr, persist, (tp, sl) in CONFIGS:
        cfg = v2.VariantConfig(
            name=f"qt{thr}_p{persist}_tp{tp}_sl{sl}", tp_mode="atr_scaled", tp_atr_mult=tp, sl_atr_mult=sl,
            sizing_mode="fixed", fixed_margin=0.30, fixed_leverage=2.0, cooldown_bars=COOLDOWN,
            quality_threshold=thr, persistence_bars=persist, max_hold_bars=MAX_HOLD, use_fallback=False,
        )
        tape_thr = v2.apply_quality_threshold(tape, thr)
        r = v2.cost_stress(tape_thr, cfg, start=start, end=end)
        rows.append({
            "quality_threshold": thr, "persistence_bars": persist, "tp_mult": tp, "sl_mult": sl,
            "cost1_pnl": r["cost1"]["pnl"], "cost1_mdd": r["cost1"]["mdd"], "cost1_trades": r["cost1"]["trades"],
            "cost3_pnl": r["cost3"]["pnl"], "cost3_mdd": r["cost3"]["mdd"],
        })
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pool()
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    print(f"features: {len(feat_cols)}", flush=True)

    # ---- Step 1: per-fold, all-27-config scoring (inner loop, VAL never read here) ----
    all_rows = []
    for fold_i, (train_end, test_start, test_end) in enumerate(FOLDS, 1):
        purge_cutoff = pd.Timestamp(train_end) - HORIZON
        train = df[(df["timestamp"] >= POOL_START) & (df["timestamp"] <= purge_cutoff)]
        clf = fit_hgb(train, feat_cols, seed=270705)
        test_frame = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)].reset_index(drop=True)
        tape = predict_tape(clf, test_frame, feat_cols)
        cfg_rows = score_all_configs(tape, pd.Timestamp(test_start), pd.Timestamp(test_end))
        for row in cfg_rows:
            row["fold"] = fold_i
            all_rows.append(row)
        print(f"fold {fold_i} done ({test_start}..{test_end})", flush=True)

    fold_df = pd.DataFrame(all_rows)
    fold_df.to_csv(OUT_DIR / "fold_config_matrix.csv", index=False)

    # ---- Step 2: aggregate per config across folds, select using fold evidence only ----
    agg = fold_df.groupby(["quality_threshold", "persistence_bars", "tp_mult", "sl_mult"]).agg(
        n_folds=("fold", "count"),
        cost1_pos_folds=("cost1_pnl", lambda s: int((s > 0).sum())),
        cost3_pos_folds=("cost3_pnl", lambda s: int((s > 0).sum())),
        mean_cost1_pnl=("cost1_pnl", "mean"), mean_cost3_pnl=("cost3_pnl", "mean"),
        worst_cost1_mdd=("cost1_mdd", "min"), worst_cost3_mdd=("cost3_mdd", "min"),
        mean_trades=("cost1_trades", "mean"),
    ).reset_index()
    agg["selected"] = (
        (agg["cost3_pos_folds"] >= 5) & (agg["cost1_pos_folds"] >= 5)
        & (agg["worst_cost1_mdd"] >= -20.0) & (agg["worst_cost3_mdd"] >= -20.0)
    )
    agg = agg.sort_values(["selected", "mean_cost3_pnl"], ascending=[False, False])
    agg.to_csv(OUT_DIR / "fold_aggregate_ranking.csv", index=False)
    n_selected = int(agg["selected"].sum())
    print(f"\nFold-based selection (>=5/7 folds cost1>0 AND cost3>0, worst-fold MDD>=-20% both tiers): "
          f"{n_selected}/27 configs pass", flush=True)
    print(agg.head(10).to_string(index=False), flush=True)

    # ---- Step 3: train final full-pool model, score ONLY the fold-selected config(s) on VAL ----
    val_purge_cutoff = VAL_START - HORIZON
    final_train = df[(df["timestamp"] >= POOL_START) & (df["timestamp"] <= val_purge_cutoff)]
    val_frame = df[(df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)].reset_index(drop=True)

    val_results = []
    if n_selected > 0:
        selected_cfgs = agg[agg["selected"]]
        seed_tapes = {}
        for seed_name, seed in (("seedA", 270705), ("seedB", 314159)):
            clf = fit_hgb(final_train, feat_cols, seed=seed)
            seed_tapes[seed_name] = predict_tape(clf, val_frame, feat_cols)
        for _, row in selected_cfgs.iterrows():
            thr, persist, tp, sl = row["quality_threshold"], row["persistence_bars"], row["tp_mult"], row["sl_mult"]
            cfg = v2.VariantConfig(
                name=f"qt{thr}_p{persist}_tp{tp}_sl{sl}", tp_mode="atr_scaled", tp_atr_mult=tp, sl_atr_mult=sl,
                sizing_mode="fixed", fixed_margin=0.30, fixed_leverage=2.0, cooldown_bars=COOLDOWN,
                quality_threshold=thr, persistence_bars=persist, max_hold_bars=MAX_HOLD, use_fallback=False,
            )
            entry = {"quality_threshold": thr, "persistence_bars": persist, "tp_mult": tp, "sl_mult": sl}
            for seed_name in ("seedA", "seedB"):
                tape_thr = v2.apply_quality_threshold(seed_tapes[seed_name], thr)
                r = v2.cost_stress(tape_thr, cfg, start=VAL_START, end=VAL_END)
                entry[f"cost1_pnl_{seed_name}"] = r["cost1"]["pnl"]
                entry[f"cost1_mdd_{seed_name}"] = r["cost1"]["mdd"]
                entry[f"cost3_pnl_{seed_name}"] = r["cost3"]["pnl"]
                entry[f"cost3_mdd_{seed_name}"] = r["cost3"]["mdd"]
                entry[f"trades_{seed_name}"] = r["cost1"]["trades"]
            val_results.append(entry)
        val_df = pd.DataFrame(val_results)
        val_df.to_csv(OUT_DIR / "val_holdout_result.csv", index=False)
        print(f"\nVAL holdout (single look, {n_selected} fold-selected config(s), never used for selection):", flush=True)
        print(val_df.to_string(index=False), flush=True)
    else:
        print("\nNo config passed the fold-based selection bar -- VAL is not evaluated (nothing to test).", flush=True)

    summary = {
        "pool": [str(POOL_START), str(val_purge_cutoff)], "val": [str(VAL_START), str(VAL_END)],
        "oos_touched": False, "n_configs_selected_by_folds": n_selected,
        "val_results": val_results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
