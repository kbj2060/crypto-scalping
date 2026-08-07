"""SOL port of train_sigma3_1h_ensemble_20260705.py. Identical recipe (5 fixed-iteration HGB
seeds, averaged class probabilities -> deterministic ensemble tape), pointed at SOL's 1h
trend-scanning dataset (build_1h_trendscan_dataset_sol_20260715.py output)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_trendscan_sol_20260715"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_hgb_sol_20260715"

TRAIN_END = pd.Timestamp("2025-06-30 23:59:59")
TAPE_START = pd.Timestamp("2025-06-25")
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
SEEDS = [270705, 270710, 270715, 270720, 270725]


def load_all() -> pd.DataFrame:
    frames = [pd.read_parquet(DATA_DIR / f"sigma3_1h_sol_{y}.parquet") for y in (2024, 2025, 2026)]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    train_mask = df["timestamp"] <= TRAIN_END
    Xtr = df.loc[train_mask, feat_cols].to_numpy(dtype=np.float64)
    ytr = df.loc[train_mask, "ts_action"].to_numpy(dtype=np.int64)
    w = np.clip(np.abs(df.loc[train_mask, "ts_t_value"].to_numpy(dtype=np.float64)), 0.5, 12.0)
    Xall = df[feat_cols].to_numpy(dtype=np.float64)

    proba_sum = np.zeros((len(df), 3), dtype=np.float64)
    for s in SEEDS:
        clf = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.03, max_iter=250, max_depth=4,
            l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=80,
            early_stopping=False, random_state=int(s), class_weight="balanced",
        )
        clf.fit(Xtr, ytr, sample_weight=w)
        pr = clf.predict_proba(Xall)
        colmap = {c: i for i, c in enumerate(list(clf.classes_))}
        out = np.zeros((len(df), 3))
        for k in (0, 1, 2):
            if k in colmap:
                out[:, k] = pr[:, colmap[k]]
        proba_sum += out
        print(f"seed {s} done", flush=True)
    proba = proba_sum / len(SEEDS)

    tape_mask = (df["timestamp"] >= TAPE_START).to_numpy()
    sub = df.loc[tape_mask].reset_index(drop=True)
    pc, pl, ps = proba[tape_mask, 0], proba[tape_mask, 1], proba[tape_mask, 2]
    P = np.column_stack([pc, pl, ps])
    dir_action = P.argmax(axis=1)
    qual = np.where(dir_action > 0, P[np.arange(len(sub)), dir_action], P[:, 0])
    final_action = np.where((dir_action != 0) & (qual >= 0.45), dir_action, 0)
    side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))

    tape = pd.DataFrame({
        "i": np.arange(len(sub)), "timestamp": sub["timestamp"],
        "open": sub["open"].astype(float), "high": sub["high"].astype(float),
        "low": sub["low"].astype(float), "close": sub["close"].astype(float),
        "jump_flag": 0.0, "evt_tail_flag": 0.0, "jump_z": 0.0, "atr_pct": sub["atr_pct"].astype(float),
        "primary_action": final_action, "primary_side": side, "primary_expert": "sigma3ens_sol",
        "primary_route_confidence": 1.0, "primary_route_margin": 1.0,
        "primary_dir_p_cash": pc, "primary_dir_p_long": pl, "primary_dir_p_short": ps,
        "primary_quality_p_cash": pc, "primary_quality_p_long": pl, "primary_quality_p_short": ps,
        "primary_quality_score": np.where(final_action != 0, qual, 0.0), "primary_confidence": P.max(axis=1),
        "fallback_action": 0, "fallback_side": 0, "fallback_expert": "none",
        "fallback_route_confidence": 0.0, "fallback_route_margin": 0.0,
        "fallback_dir_p_cash": 1.0, "fallback_dir_p_long": 0.0, "fallback_dir_p_short": 0.0,
        "fallback_quality_p_cash": 1.0, "fallback_quality_p_long": 0.0, "fallback_quality_p_short": 0.0,
        "fallback_quality_score": 0.0, "fallback_confidence": 0.0,
    })
    out_path = OUT_DIR / "tape_ensemble_sol.parquet"
    tape.to_parquet(out_path, index=False)
    print(f"wrote ensemble tape {len(tape)} rows -> {out_path}", flush=True)
    print(f"nonzero pct: {(tape['primary_side'] != 0).mean():.3f}", flush=True)
    (OUT_DIR / "report_ensemble.json").write_text(json.dumps({"seeds": SEEDS, "n_features": len(feat_cols)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
