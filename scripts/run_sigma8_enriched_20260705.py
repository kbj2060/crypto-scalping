#!/usr/bin/env python3
"""Sigma8: retrain the Sigma3-1h ensemble on the ENRICHED 43-feature set (38 base + 5 audit-winner
features) and run the Sigma6 regime-trend backtest, comparing head-to-head against the original
Sigma6 (38-feature) result. Evaluated primarily on VALIDATION (2025-07..12) improvement; OOS
2026-03..06 reported for continuity but treated as heavily-peeked (not clean confirmation).
"""

from __future__ import annotations

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
import run_sigma6_regime_trend_20260705 as s6  # noqa: E402

DATA_DIR = ROOT / "tmp/causal_regen_20260516/sigma8_1h_enriched_20260705"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma8_enriched_20260705"
REG_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
CM_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601"
TRAIN_END = pd.Timestamp("2025-06-30 23:59:59")
TAPE_START = pd.Timestamp("2025-06-25")
SEEDS = [270705, 270710, 270715, 270720, 270725]
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
PFX = s6.PFX


def merge_regime(tape):
    reg = pd.concat([pd.read_csv(REG_DIR / f"training_features_{y}_regime3_current_sensitive_hmm_wide24.csv", parse_dates=["timestamp"]) for y in ("2025", "2026_rebuilt")], ignore_index=True).sort_values("timestamp")
    keep = ["timestamp", f"{PFX}bull_prob", f"{PFX}bear_prob", f"{PFX}chop_prob"]
    tape = pd.merge_asof(tape.sort_values("timestamp"), reg[keep], on="timestamp", direction="backward")
    cm = pd.concat([pd.read_csv(CM_DIR / f"training_features_{y}_regime3_cryptomamba_h6_sidecar_20260601.csv", parse_dates=["timestamp"]) for y in ("2025", "2026_rebuilt")], ignore_index=True).sort_values("timestamp")
    tape = pd.merge_asof(tape, cm[["timestamp", "regime3_cmamba_h6_sidecar_stability_score"]], on="timestamp", direction="backward")
    return tape.sort_values("i").reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_parquet(DATA_DIR / f"sigma8_1h_{y}.parquet") for y in (2024, 2025, 2026)], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    print(f"features: {len(feat_cols)} (enriched)", flush=True)
    tr = (df["timestamp"] <= TRAIN_END).to_numpy()
    Xtr, ytr = df.loc[tr, feat_cols].to_numpy(np.float64), df.loc[tr, "ts_action"].to_numpy(np.int64)
    w = np.clip(np.abs(df.loc[tr, "ts_t_value"].to_numpy(np.float64)), 0.5, 12.0)
    Xall = df[feat_cols].to_numpy(np.float64)
    psum = np.zeros((len(df), 3))
    for s in SEEDS:
        clf = HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.03, max_iter=250, max_depth=4,
                                             l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=80,
                                             early_stopping=False, random_state=s, class_weight="balanced")
        clf.fit(Xtr, ytr, sample_weight=w)
        pr = clf.predict_proba(Xall); cm = {c: i for i, c in enumerate(list(clf.classes_))}
        for k in (0, 1, 2):
            if k in cm:
                psum[:, k] += pr[:, cm[k]]
    proba = psum / len(SEEDS)
    tm = (df["timestamp"] >= TAPE_START).to_numpy()
    sub = df.loc[tm].reset_index(drop=True)
    pc, pl, ps = proba[tm, 0], proba[tm, 1], proba[tm, 2]
    P = np.column_stack([pc, pl, ps]); da = P.argmax(1)
    qual = np.where(da > 0, P[np.arange(len(sub)), da], P[:, 0])
    fa = np.where((da != 0) & (qual >= 0.45), da, 0); side = np.where(fa == 1, 1, np.where(fa == 2, -1, 0))
    tape = pd.DataFrame({"i": np.arange(len(sub)), "timestamp": sub["timestamp"],
        "open": sub["open"].astype(float), "high": sub["high"].astype(float), "low": sub["low"].astype(float), "close": sub["close"].astype(float),
        "atr_pct": sub["atr_pct"].astype(float), "primary_action": fa, "primary_side": side,
        "primary_dir_p_cash": pc, "primary_dir_p_long": pl, "primary_dir_p_short": ps,
        "primary_quality_p_cash": pc, "primary_quality_p_long": pl, "primary_quality_p_short": ps})
    tape = merge_regime(tape)
    tape.to_parquet(OUT_DIR / "tape_regime.parquet", index=False)

    tapes = {thr: v2.apply_quality_threshold(tape, thr) for thr in (0.60, 0.70)}
    base = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3, reg_mode="not_chop", reg_thr=0.42, stab_thr=0.55)
    print("\n=== Sigma8 (enriched 43-feat) vs Sigma6 (38-feat) at the Sigma6 winning configs ===", flush=True)
    for name, c in {"lev3 not_chop+stab": dict(thr=0.70, leverage=3.0, sl_atr=2.5),
                     "lev4 not_chop+stab": dict(thr=0.70, leverage=4.0, sl_atr=2.5)}.items():
        thr = c.pop("thr"); tpx = tapes[thr]
        rv = s6.backtest(tpx, fee_mult=1.0, start=s6.VAL_START, end=s6.VAL_END, **c, **base)
        ro = s6.backtest(tpx, fee_mult=1.0, start=s6.OOS_START, end=s6.OOS_END, **c, **base)
        print(f"{name}:", flush=True)
        print(f"  VAL c1={rv['pnl']:.1f}% mdd={rv['mdd']:.1f}% tr={rv['trades']} wr={rv['wr']:.3f}", flush=True)
        print(f"  OOS c1={ro['pnl']:.1f}% mdd={ro['mdd']:.1f}% tr={ro['trades']} wr={ro['wr']:.3f} (peeked, continuity only)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
