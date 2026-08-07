#!/usr/bin/env python3
"""Sigma3 1h, ORIGINAL single-split methodology (not the fold approach), multi-seed variance check.

User question: the original contract (docs/model_contracts/sigma3_1h_trendscan_20260705_contract.md)
reported "2-seed sweep: 0/27 joint pass, severe seed instability (seedB passed several configs
cleanly, seedA negative at the same configs)" then moved to a "5-seed ensemble... 1/27 gate pass"
-- i.e. individual seeds did NOT agree, only an averaged ensemble stabilized enough to pass once.
This script re-verifies that directly and with more seeds (8), using the original single train/val
split (not the 7-fold walk-forward), on the continuous (cold-start-bug-fixed) 38-feature dataset --
same base features as the original contract, just with the year-boundary rolling-feature bug fixed.

Split matches the original contract exactly: train 2024-01-01..2025-06-30 (purged 48h before VAL,
same HORIZON convention used elsewhere this session -- the original contract did not purge, this
is a minor, more-correct addition), VAL 2025-07-01..2025-12-31 (6 months, matching the original
grid's design). OOS is not read here.
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
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_multiseed_original_split_20260801"

NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
HORIZON = pd.Timedelta(hours=48)

POOL_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2025-06-30 23:59:59")
VAL_START = pd.Timestamp("2025-07-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")

SEEDS = [270705, 314159, 27, 1000, 42, 8675309, 2026, 555]

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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pool()
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    purge_cutoff = TRAIN_END - HORIZON
    train = df[(df["timestamp"] >= POOL_START) & (df["timestamp"] <= purge_cutoff)]
    val_frame = df[(df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)].reset_index(drop=True)
    print(f"features: {len(feat_cols)}, train rows: {len(train)} (purged {purge_cutoff}), "
          f"val rows: {len(val_frame)} ({VAL_START.date()}..{VAL_END.date()})", flush=True)

    all_rows = []
    champion_rows = []
    for seed in SEEDS:
        clf = fit_hgb(train, feat_cols, seed=seed)
        tape = predict_tape(clf, val_frame, feat_cols)
        for thr, persist, (tp, sl) in itertools.product(THRESHOLDS, PERSISTS, TPSL):
            cfg = v2.VariantConfig(
                name=f"qt{thr}_p{persist}_tp{tp}_sl{sl}", tp_mode="atr_scaled", tp_atr_mult=tp, sl_atr_mult=sl,
                sizing_mode="fixed", fixed_margin=0.30, fixed_leverage=2.0, cooldown_bars=COOLDOWN,
                quality_threshold=thr, persistence_bars=persist, max_hold_bars=MAX_HOLD, use_fallback=False,
            )
            tape_thr = v2.apply_quality_threshold(tape, thr)
            r = v2.cost_stress(tape_thr, cfg, start=VAL_START, end=VAL_END)
            row = {
                "seed": seed, "quality_threshold": thr, "persistence_bars": persist, "tp_mult": tp, "sl_mult": sl,
                "cost1_pnl": r["cost1"]["pnl"], "cost1_mdd": r["cost1"]["mdd"], "cost1_trades": r["cost1"]["trades"],
                "cost3_pnl": r["cost3"]["pnl"], "cost3_mdd": r["cost3"]["mdd"], "months": len(r["cost1"]["trades_by_month"]),
            }
            row["gate_pass"] = (
                row["cost1_pnl"] > 0 and row["cost3_pnl"] > 0
                and row["cost1_mdd"] >= -20.0 and row["cost3_mdd"] >= -20.0
                and row["cost1_trades"] >= 40 and row["months"] >= 5
            )
            all_rows.append(row)
            if thr == CHAMPION["quality_threshold"] and persist == CHAMPION["persistence_bars"] \
               and tp == CHAMPION["tp_atr_mult"] and sl == CHAMPION["sl_atr_mult"]:
                champion_rows.append(row)
        n_pass = sum(1 for r in all_rows if r["seed"] == seed and r["gate_pass"])
        print(f"seed {seed}: iters={clf.n_iter_}, gate_pass {n_pass}/27, "
              f"champion cost1={champion_rows[-1]['cost1_pnl']:.2f} cost3={champion_rows[-1]['cost3_pnl']:.2f}", flush=True)

    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(OUT_DIR / "multiseed_grid.csv", index=False)
    champ_df = pd.DataFrame(champion_rows)
    champ_df.to_csv(OUT_DIR / "multiseed_champion_config.csv", index=False)

    print(f"\n=== Champion config (qt0.7/p0/tp1.5/sl1.0) across {len(SEEDS)} seeds ===", flush=True)
    print(champ_df[["seed", "cost1_pnl", "cost1_mdd", "cost3_pnl", "cost3_mdd", "cost1_trades", "gate_pass"]].to_string(index=False), flush=True)
    print(f"\ncost1 positive: {int((champ_df['cost1_pnl']>0).sum())}/{len(SEEDS)}, "
          f"cost3 positive: {int((champ_df['cost3_pnl']>0).sum())}/{len(SEEDS)}, "
          f"gate_pass: {int(champ_df['gate_pass'].sum())}/{len(SEEDS)}", flush=True)
    print(f"cost1_pnl std: {champ_df['cost1_pnl'].std():.2f}, cost3_pnl std: {champ_df['cost3_pnl'].std():.2f}", flush=True)

    per_seed_pass = full_df.groupby("seed")["gate_pass"].sum()
    print(f"\nper-seed gate_pass count out of 27 configs: {per_seed_pass.to_dict()}", flush=True)

    summary = {
        "split": {"train": [str(POOL_START), str(purge_cutoff)], "val": [str(VAL_START), str(VAL_END)]},
        "seeds": SEEDS,
        "champion_config_cost1_positive": int((champ_df["cost1_pnl"] > 0).sum()),
        "champion_config_cost3_positive": int((champ_df["cost3_pnl"] > 0).sum()),
        "champion_config_gate_pass": int(champ_df["gate_pass"].sum()),
        "per_seed_27config_gate_pass": per_seed_pass.to_dict(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
