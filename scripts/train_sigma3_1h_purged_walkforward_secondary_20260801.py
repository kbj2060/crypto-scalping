#!/usr/bin/env python3
"""Sigma3 1h purged walk-forward retrain, using the continuous (no year-boundary cold-start)
dataset from build_1h_trendscan_dataset_continuous_20260801.py.

Per user decision (2026-08-01 session): 2024 data CAN be used as real training rows (not just
rolling-feature warmup) because Sigma3-1h's 38 features are all self-computed rolling/shift
stats with no pre-fit upstream component (unlike e.g. HMM regime models that are calibrated on
a fixed 2024 window) -- there is no leakage reason to exclude 2024. Separately, folding lets us
directly observe the fold-to-fold instability previously only inferred indirectly
(docs: Sigma6 fresh-window fragility, 2024+2025H1 retrain flipped ~4% of labels and the OOS
sign). Purging follows this project's existing convention
(scripts/walkforward_scalp_1m_weighted_purged_20260716.py): drop training rows whose
label's forward window (here up to 48h, trend-scanning's max window) extends past the fold's
train/test boundary.

Design (agreed with user):
  - Train pool: 2024-01-01 .. 2025-08-31 (used for both the internal folds AND the final model).
  - Internal folds: expanding-window, 2-month test blocks, HORIZON=48h purge before each fold's
    train_end. Reports per-fold cost1/cost3 pnl/mdd/trades at the frozen champion config
    (qt0.7/p0/tp1.5/sl1.0) to directly show retrain stability across time, not just at one split.
  - Final model: 2 seeds trained on the full pool (purged 48h before VAL_START), producing a
    decision tape over VAL only. Gate-swept over the same pre-registered 27-config grid as the
    original contract (scripts/replay_sigma3_1h_gates_20260705.py) so results are comparable.
  - VAL = 2025-09-01..12-31 (holdout, never in any fold's train set).
  - OOS = 2026-01-01..03-31 is NOT read anywhere in this script (kept untouched for one-shot use
    later, per this project's fresh-forward discipline).
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
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_purged_walkforward_secondary_20260801"

NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
HORIZON = pd.Timedelta(hours=48)  # max trend-scan window

POOL_START = pd.Timestamp("2024-01-01")
POOL_END = pd.Timestamp("2025-08-31 23:59:59")
VAL_START = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")
TAPE_CONTEXT_START = VAL_START - pd.Timedelta(days=5)  # a few days of pre-VAL context for cooldown/persistence state, no different from the original contract's TAPE_START convention

FOLDS = [
    # (train_end, test_start, test_end)
    ("2024-06-30 23:59:59", "2024-07-01", "2024-08-31 23:59:59"),
    ("2024-08-31 23:59:59", "2024-09-01", "2024-10-31 23:59:59"),
    ("2024-10-31 23:59:59", "2024-11-01", "2024-12-31 23:59:59"),
    ("2024-12-31 23:59:59", "2025-01-01", "2025-02-28 23:59:59"),
    ("2025-02-28 23:59:59", "2025-03-01", "2025-04-30 23:59:59"),
    ("2025-04-30 23:59:59", "2025-05-01", "2025-06-30 23:59:59"),
    ("2025-06-30 23:59:59", "2025-07-01", "2025-08-31 23:59:59"),
]

CHAMPION_CFG = dict(quality_threshold=0.70, persistence_bars=0, tp_atr_mult=1.5, sl_atr_mult=1.0)

THRESHOLDS = [0.50, 0.60, 0.70]
PERSISTS = [0, 2, 4]
TPSL = [(1.5, 1.0), (2.0, 0.9), (2.5, 1.2)]
COOLDOWN = 3
MAX_HOLD = 48


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


def run_fold_gate(tape: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    cfg = v2.VariantConfig(
        name="champion", tp_mode="atr_scaled", tp_atr_mult=CHAMPION_CFG["tp_atr_mult"],
        sl_atr_mult=CHAMPION_CFG["sl_atr_mult"], sizing_mode="fixed", fixed_margin=0.30,
        fixed_leverage=2.0, cooldown_bars=COOLDOWN, quality_threshold=CHAMPION_CFG["quality_threshold"],
        persistence_bars=CHAMPION_CFG["persistence_bars"], max_hold_bars=MAX_HOLD, use_fallback=False,
    )
    tape_thr = v2.apply_quality_threshold(tape, CHAMPION_CFG["quality_threshold"])
    r = v2.cost_stress(tape_thr, cfg, start=start, end=end)
    return {
        "cost1_pnl": r["cost1"]["pnl"], "cost1_mdd": r["cost1"]["mdd"], "cost1_trades": r["cost1"]["trades"],
        "cost1_wr": r["cost1"]["wr"], "cost3_pnl": r["cost3"]["pnl"], "cost3_mdd": r["cost3"]["mdd"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pool()
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    print(f"features: {len(feat_cols)}, pool rows: {len(df[(df['timestamp'] >= POOL_START) & (df['timestamp'] <= POOL_END)])}", flush=True)

    # ---- Part 1: purged walk-forward folds over the train pool ----
    fold_results = []
    for i, (train_end, test_start, test_end) in enumerate(FOLDS, 1):
        purge_cutoff = pd.Timestamp(train_end) - HORIZON
        train = df[(df["timestamp"] >= POOL_START) & (df["timestamp"] <= purge_cutoff)]
        n_purged = len(df[(df["timestamp"] > purge_cutoff) & (df["timestamp"] <= train_end)])
        clf = fit_hgb(train, feat_cols, seed=270705)
        test_ctx_start = pd.Timestamp(test_start) - pd.Timedelta(days=5)
        test_frame = df[(df["timestamp"] >= test_ctx_start) & (df["timestamp"] <= test_end)].reset_index(drop=True)
        tape = predict_tape(clf, test_frame, feat_cols)
        gate = run_fold_gate(tape, pd.Timestamp(test_start), pd.Timestamp(test_end))
        row = {
            "fold": i, "train_rows": len(train), "purge_cutoff": str(purge_cutoff), "n_purged": n_purged,
            "test_start": test_start, "test_end": test_end, "iters": int(clf.n_iter_), **gate,
        }
        fold_results.append(row)
        print(json.dumps(row), flush=True)

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(OUT_DIR / "fold_results.csv", index=False)
    n_pos_cost1 = int((fold_df["cost1_pnl"] > 0).sum())
    n_pos_cost3 = int((fold_df["cost3_pnl"] > 0).sum())
    print(f"\nFold stability at champion config (qt0.7/p0/tp1.5/sl1.0): "
          f"cost1 positive {n_pos_cost1}/{len(fold_df)}, cost3 positive {n_pos_cost3}/{len(fold_df)}", flush=True)

    # ---- Part 2: final model, full pool -> VAL holdout, 2-seed gate sweep ----
    val_purge_cutoff = VAL_START - HORIZON
    final_train = df[(df["timestamp"] >= POOL_START) & (df["timestamp"] <= val_purge_cutoff)]
    print(f"\nFinal model train rows (pool, purged before VAL): {len(final_train)}", flush=True)

    val_frame = df[(df["timestamp"] >= TAPE_CONTEXT_START) & (df["timestamp"] <= VAL_END)].reset_index(drop=True)
    seed_gate_rows = []
    tapes_by_seed = {}
    for seed_name, seed in (("seedA", 270705), ("seedB", 314159)):
        clf = fit_hgb(final_train, feat_cols, seed=seed)
        tape = predict_tape(clf, val_frame, feat_cols)
        tapes_by_seed[seed_name] = tape
        tape.to_parquet(OUT_DIR / f"final_tape_{seed_name}.parquet", index=False)

    a_rows, b_rows = [], []
    for thr, persist, (tp, sl) in itertools.product(THRESHOLDS, PERSISTS, TPSL):
        cfg = v2.VariantConfig(
            name=f"qt{thr}_p{persist}_tp{tp}_sl{sl}", tp_mode="atr_scaled", tp_atr_mult=tp, sl_atr_mult=sl,
            sizing_mode="fixed", fixed_margin=0.30, fixed_leverage=2.0, cooldown_bars=COOLDOWN,
            quality_threshold=thr, persistence_bars=persist, max_hold_bars=MAX_HOLD, use_fallback=False,
        )
        for seed_name, rows in (("seedA", a_rows), ("seedB", b_rows)):
            tape_thr = v2.apply_quality_threshold(tapes_by_seed[seed_name], thr)
            r = v2.cost_stress(tape_thr, cfg, start=VAL_START, end=VAL_END)
            rows.append({
                "quality_threshold": thr, "persistence_bars": persist, "tp_mult": tp, "sl_mult": sl,
                "cost1_pnl": r["cost1"]["pnl"], "cost1_mdd": r["cost1"]["mdd"], "cost1_trades": r["cost1"]["trades"],
                "cost1_wr": r["cost1"]["wr"], "cost3_pnl": r["cost3"]["pnl"], "cost3_mdd": r["cost3"]["mdd"],
                "months": len(r["cost1"]["trades_by_month"]),
            })

    a_df = pd.DataFrame(a_rows)
    b_df = pd.DataFrame(b_rows)
    m = a_df.merge(b_df, on=["quality_threshold", "persistence_bars", "tp_mult", "sl_mult"], suffixes=("_A", "_B"))
    m["gate_pass_A"] = (
        (m["cost1_pnl_A"] > 0) & (m["cost3_pnl_A"] > 0)
        & (m["cost1_mdd_A"] >= -20.0) & (m["cost3_mdd_A"] >= -20.0)
        & (m["cost1_trades_A"] >= 40) & (m["months_A"] >= 3)
    )
    m["joint_pass"] = m["gate_pass_A"] & (m["cost1_pnl_B"] > 0)
    m.to_csv(OUT_DIR / "val_gate_ranking.csv", index=False)
    passing = m[m["joint_pass"]].sort_values("cost3_pnl_A", ascending=False)
    print(f"\nVAL (2025-09..12, holdout, purged-48h): seedA gate_pass {int(m['gate_pass_A'].sum())}/27, "
          f"joint (seedB cost1>0) {len(passing)}/27", flush=True)
    if len(passing):
        print(passing[["quality_threshold", "persistence_bars", "tp_mult", "sl_mult",
                        "cost1_pnl_A", "cost3_pnl_A", "cost1_mdd_A", "cost3_mdd_A", "cost1_trades_A"]].to_string(index=False), flush=True)
    else:
        print("top 8 seedA by cost3_pnl (none jointly pass):", flush=True)
        print(a_df.sort_values("cost3_pnl", ascending=False).head(8).to_string(index=False), flush=True)

    summary = {
        "pool": [str(POOL_START), str(POOL_END)], "val": [str(VAL_START), str(VAL_END)],
        "oos_touched": False,
        "fold_stability": {"cost1_positive": f"{n_pos_cost1}/{len(fold_df)}", "cost3_positive": f"{n_pos_cost3}/{len(fold_df)}"},
        "val_seedA_gate_pass": int(m["gate_pass_A"].sum()), "val_joint_pass": int(len(passing)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
