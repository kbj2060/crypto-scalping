#!/usr/bin/env python3
"""Sigma3 1h seed-ensemble candidate: average predicted probabilities across N seeds instead of
trusting a single frozen seed, following the original contract's own precedent (2-seed sweep 0/27
-> 5-seed averaged ensemble 1/27) -- this session's 8-seed test showed the same champion config
swings cost3 by +-25pp across raw seeds, so a single frozen checkpoint is likely a lucky draw, not
a robust signal. Averaging reduces variance from near-coin-flip individual entries without
requiring a new "stable retrain" that today's experiments showed doesn't exist.

Uses the continuous (cold-start-fixed) 38-feature dataset -- NOT the 125-secondary-feature set,
since that axis was separately closed today (nested fold nested_walkforward_secondary: 0/27).

Train: 2024-01-01..2025-06-30 (purged 48h before the earliest eval window), same as the original
contract's train window. Eval on the SAME two windows already used (and effectively "spent") by
this morning's joint-portfolio research for the single-seed frozen tape, so this is a fair
apples-to-apples re-check, not a fresh one-shot burn:
  VAL: 2025-10-01..12-31 (matches replay_eth_sigma3_1h_omega461 correlation/joint-portfolio work)
  OOS: 2026-01-01..03-31
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

DATA_PATH = ROOT / "tmp/causal_regen_20260516/sigma3_1h_trendscan_continuous_20260801/sigma3_1h_continuous.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_seed_ensemble_20260801"

NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
HORIZON = pd.Timedelta(hours=48)

POOL_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2025-06-30 23:59:59")
TAPE_START = pd.Timestamp("2025-09-25")
VAL_START = pd.Timestamp("2025-10-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-03-31 23:59:59")

SEEDS = [270705, 314159, 27, 1000, 42, 8675309, 2026, 555]  # same 8 as this session's variance check

THRESHOLDS = [0.50, 0.60, 0.70]
PERSISTS = [0, 2, 4]
TPSL = [(1.5, 1.0), (2.0, 0.9), (2.5, 1.2)]
COOLDOWN = 3
MAX_HOLD = 48
CHAMPION = dict(quality_threshold=0.70, persistence_bars=0, tp_atr_mult=1.5, sl_atr_mult=1.0)


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


def seed_probs(clf: HistGradientBoostingClassifier, frame: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    proba = clf.predict_proba(frame[feat_cols].to_numpy(dtype=np.float64))
    cls = list(clf.classes_)
    col_for = {c: i for i, c in enumerate(cls)}
    p_cash = proba[:, col_for[0]] if 0 in col_for else np.zeros(len(frame))
    p_long = proba[:, col_for[1]] if 1 in col_for else np.zeros(len(frame))
    p_short = proba[:, col_for[2]] if 2 in col_for else np.zeros(len(frame))
    return np.column_stack([p_cash, p_long, p_short])


def build_tape(probs: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    dir_action = probs.argmax(axis=1)
    qual = np.where(dir_action > 0, probs[np.arange(len(frame)), dir_action], probs[:, 0])
    DEFAULT_THR = 0.45
    final_action = np.where((dir_action != 0) & (qual >= DEFAULT_THR), dir_action, 0)
    side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))
    p_cash, p_long, p_short = probs[:, 0], probs[:, 1], probs[:, 2]
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


def eval_grid(tape: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for thr, persist, (tp, sl) in itertools.product(THRESHOLDS, PERSISTS, TPSL):
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
            "cost3_pnl": r["cost3"]["pnl"], "cost3_mdd": r["cost3"]["mdd"], "months": len(r["cost1"]["trades_by_month"]),
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pool()
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    purge_cutoff = TRAIN_END - HORIZON
    train = df[(df["timestamp"] >= POOL_START) & (df["timestamp"] <= purge_cutoff)]
    tape_frame = df[(df["timestamp"] >= TAPE_START) & (df["timestamp"] <= OOS_END)].reset_index(drop=True)
    print(f"features: {len(feat_cols)}, train rows: {len(train)}, tape rows: {len(tape_frame)}", flush=True)

    probs_by_seed = []
    for seed in SEEDS:
        clf = fit_hgb(train, feat_cols, seed=seed)
        p = seed_probs(clf, tape_frame, feat_cols)
        probs_by_seed.append(p)
        print(f"seed {seed} trained (iters={clf.n_iter_})", flush=True)

    all_probs = np.stack(probs_by_seed, axis=0)  # (n_seeds, n_rows, 3)
    ens_probs = all_probs.mean(axis=0)
    ens_tape = build_tape(ens_probs, tape_frame)
    ens_tape.to_parquet(OUT_DIR / "ensemble_tape.parquet", index=False)

    val_grid = eval_grid(ens_tape, VAL_START, VAL_END)
    val_grid["gate_pass"] = (
        (val_grid["cost1_pnl"] > 0) & (val_grid["cost3_pnl"] > 0)
        & (val_grid["cost1_mdd"] >= -20.0) & (val_grid["cost3_mdd"] >= -20.0)
        & (val_grid["cost1_trades"] >= 30) & (val_grid["months"] >= 2)
    )
    val_grid.to_csv(OUT_DIR / "val_grid.csv", index=False)
    print(f"\nEnsemble ({len(SEEDS)} seeds) VAL (2025-10..12) gate_pass: {int(val_grid['gate_pass'].sum())}/27", flush=True)

    champ_val = val_grid[(val_grid["quality_threshold"] == CHAMPION["quality_threshold"])
                          & (val_grid["persistence_bars"] == CHAMPION["persistence_bars"])
                          & (val_grid["tp_mult"] == CHAMPION["tp_atr_mult"])
                          & (val_grid["sl_mult"] == CHAMPION["sl_atr_mult"])]
    print("champion config VAL:", flush=True)
    print(champ_val.to_string(index=False), flush=True)

    # OOS: score only the VAL-passing configs (still not a blind full grid-search on OOS)
    passing = val_grid[val_grid["gate_pass"]]
    oos_rows = []
    if len(passing):
        oos_full = eval_grid(ens_tape, OOS_START, OOS_END)
        for _, row in passing.iterrows():
            match = oos_full[(oos_full["quality_threshold"] == row["quality_threshold"])
                              & (oos_full["persistence_bars"] == row["persistence_bars"])
                              & (oos_full["tp_mult"] == row["tp_mult"]) & (oos_full["sl_mult"] == row["sl_mult"])]
            oos_rows.append(match.iloc[0].to_dict())
        oos_df = pd.DataFrame(oos_rows)
        oos_df.to_csv(OUT_DIR / "oos_for_val_passers.csv", index=False)
        print(f"\nOOS (2026-01..03) for the {len(passing)} VAL-passing config(s):", flush=True)
        print(oos_df.to_string(index=False), flush=True)
    else:
        print("\nNo config passed VAL -- OOS not evaluated for any config.", flush=True)

    # also report the always-relevant champion config's OOS number for direct comparison to the
    # single-seed frozen tape's known OOS result (+24.34%/-17.95%, per this morning's research)
    champ_oos_grid = eval_grid(ens_tape, OOS_START, OOS_END)
    champ_oos = champ_oos_grid[(champ_oos_grid["quality_threshold"] == CHAMPION["quality_threshold"])
                                & (champ_oos_grid["persistence_bars"] == CHAMPION["persistence_bars"])
                                & (champ_oos_grid["tp_mult"] == CHAMPION["tp_atr_mult"])
                                & (champ_oos_grid["sl_mult"] == CHAMPION["sl_atr_mult"])]
    print("\nchampion config OOS (for direct comparison to single-seed frozen tape's +24.34%/-17.95%):", flush=True)
    print(champ_oos.to_string(index=False), flush=True)

    summary = {
        "seeds": SEEDS, "train": [str(POOL_START), str(purge_cutoff)],
        "val": [str(VAL_START), str(VAL_END)], "oos": [str(OOS_START), str(OOS_END)],
        "val_gate_pass": int(val_grid["gate_pass"].sum()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
