#!/usr/bin/env python3
"""Sigma7: 5-minute decision cadence + 1h/6h context + Regime3 not-chop filter + let-winners-run
trailing-stop trend-following. Combines the user's 5m requirement with the two things that
rescued 1h trend-following out-of-sample in Sigma6 (regime filter + trailing stop), which the
earlier 5m attempt (Sigma4) lacked.

Trains a 5-seed HGB ensemble on the 5m MTF2 features (2024-01..2025-06), emits a 5m tape, merges
the 5m-native Regime3 wide24 (bull/bear/chop) + CryptoMamba stability columns, and runs the
Sigma6 backtest (reused verbatim) with 5m-appropriate barriers. Validation 2025-07..12; OOS
2026-03..06 reported with the caveat that that window is now heavily peeked (real test = 2026-07+).
"""

from __future__ import annotations

import itertools
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
import run_sigma6_regime_trend_20260705 as s6  # noqa: E402 (reuse backtest + PFX)

DATA_DIR = ROOT / "tmp/causal_regen_20260516/sigma7_5m_mtf2_20260705"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma7_5m_regime_trend_20260705"
REG_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
CM_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601"
TRAIN_END = pd.Timestamp("2025-06-30 23:59:59")
TAPE_START = pd.Timestamp("2025-06-20")
VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-03-02"), pd.Timestamp("2026-06-30 23:59:59")
SEEDS = [270705, 270710, 270715, 270720, 270725]
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
PFX = s6.PFX


def load_all():
    fr = [pd.read_parquet(DATA_DIR / f"sigma7_5m_{y}.parquet") for y in (2024, 2025, 2026)]
    df = pd.concat(fr, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def merge_regime(tape):
    reg = pd.concat([
        pd.read_csv(REG_DIR / "training_features_2025_regime3_current_sensitive_hmm_wide24.csv", parse_dates=["timestamp"]),
        pd.read_csv(REG_DIR / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv", parse_dates=["timestamp"]),
    ], ignore_index=True).sort_values("timestamp")
    keep = ["timestamp", f"{PFX}bull_prob", f"{PFX}bear_prob", f"{PFX}chop_prob"]
    tape = pd.merge_asof(tape.sort_values("timestamp"), reg[keep], on="timestamp", direction="backward")
    cm = pd.concat([
        pd.read_csv(CM_DIR / "training_features_2025_regime3_cryptomamba_h6_sidecar_20260601.csv", parse_dates=["timestamp"]),
        pd.read_csv(CM_DIR / "training_features_2026_rebuilt_regime3_cryptomamba_h6_sidecar_20260601.csv", parse_dates=["timestamp"]),
    ], ignore_index=True).sort_values("timestamp")
    tape = pd.merge_asof(tape, cm[["timestamp", "regime3_cmamba_h6_sidecar_stability_score"]], on="timestamp", direction="backward")
    return tape.sort_values("i").reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    print(f"features: {len(feat_cols)}", flush=True)
    tr = (df["timestamp"] <= TRAIN_END).to_numpy()
    Xtr, ytr = df.loc[tr, feat_cols].to_numpy(np.float64), df.loc[tr, "ts_action"].to_numpy(np.int64)
    w = np.clip(np.abs(df.loc[tr, "ts_t_value"].to_numpy(np.float64)), 0.5, 12.0)
    Xall = df[feat_cols].to_numpy(np.float64)
    psum = np.zeros((len(df), 3))
    for s in SEEDS:
        clf = HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.03, max_iter=250, max_depth=4,
                                             l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=120,
                                             early_stopping=False, random_state=s, class_weight="balanced")
        clf.fit(Xtr, ytr, sample_weight=w)
        pr = clf.predict_proba(Xall); cm = {c: i for i, c in enumerate(list(clf.classes_))}
        for k in (0, 1, 2):
            if k in cm:
                psum[:, k] += pr[:, cm[k]]
        print(f"seed {s} done", flush=True)
    proba = psum / len(SEEDS)
    tm = (df["timestamp"] >= TAPE_START).to_numpy()
    sub = df.loc[tm].reset_index(drop=True)
    pc, pl, ps = proba[tm, 0], proba[tm, 1], proba[tm, 2]
    P = np.column_stack([pc, pl, ps]); da = P.argmax(1)
    qual = np.where(da > 0, P[np.arange(len(sub)), da], P[:, 0])
    fa = np.where((da != 0) & (qual >= 0.45), da, 0)
    side = np.where(fa == 1, 1, np.where(fa == 2, -1, 0))
    tape = pd.DataFrame({"i": np.arange(len(sub)), "timestamp": sub["timestamp"],
        "open": sub["open"].astype(float), "high": sub["high"].astype(float), "low": sub["low"].astype(float),
        "close": sub["close"].astype(float), "atr_pct": sub["atr_pct"].astype(float),
        "primary_action": fa, "primary_side": side,
        "primary_dir_p_cash": pc, "primary_dir_p_long": pl, "primary_dir_p_short": ps,
        "primary_quality_p_cash": pc, "primary_quality_p_long": pl, "primary_quality_p_short": ps})
    tape = merge_regime(tape)
    tape.to_parquet(OUT_DIR / "tape_regime.parquet", index=False)
    print(f"tape {len(tape)} rows, atr median {tape['atr_pct'].median():.5f}", flush=True)

    tapes = {thr: v2.apply_quality_threshold(tape, thr) for thr in (0.60, 0.70)}
    base = dict(margin=0.30, min_profit_atr=5.0, cooldown=12, reg_mode="not_chop", reg_thr=0.42, stab_thr=0.55)
    grid = itertools.product([0.60, 0.70], [3.0, 4.0], [8.0, 12.0, 20.0], [4.0, 8.0], [576, 1152])
    rows = []
    for thr, lev, trail, sl, mh in grid:
        r1 = s6.backtest(tapes[thr], leverage=lev, trail_atr=trail, sl_atr=sl, max_hold=mh, fee_mult=1.0, start=VAL_START, end=VAL_END, **base)
        rows.append({"thr": thr, "lev": lev, "trail": trail, "sl": sl, "mh": mh,
                     "c1": round(r1["pnl"], 1), "mdd": round(r1["mdd"], 1), "tr": r1["trades"],
                     "wr": round(r1["wr"], 3), "mo": len(r1["by_month"]),
                     "minmo": round(min(r1["by_month"].values()) * 100, 1) if r1["by_month"] else 0})
    d = pd.DataFrame(rows).sort_values("c1", ascending=False)
    d.to_csv(OUT_DIR / "val_frontier.csv", index=False)
    print("=== VAL 2025-07..12, Sigma7 (5m + 1h/6h + regime + trailing), top 18 by cost1 ===", flush=True)
    print(d.head(18).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
