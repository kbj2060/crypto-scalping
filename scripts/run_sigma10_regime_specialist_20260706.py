#!/usr/bin/env python3
"""Sigma10: REGIME-SPECIALIST expert, one of the 4 architecture directions proposed when asked
"is Sigma6 architecturally optimal?" (multi-asset diversification = Sigma9, failed; regime-specific
experts = this; learned sizing = Sigma11; order-book microstructure = blocked, insufficient
history).

Sigma6 trains ONE generalist HGB ensemble on ALL bars (chop+trend mixed) then applies a post-hoc
"not_chop" entry filter at inference. The specialist hypothesis: training on trend-regime rows
ONLY (dropping chop-regime rows from the training set, not just gating trades at inference) lets
the trees fit trend-regime statistics without dilution from chop-regime noise, which might sharpen
the signal specifically where Sigma6 actually trades.

Same 1h trend-scanning dataset, same 5-seed HGB hyperparameters as Sigma3, same not_chop entry
filter at inference (both need it: the specialist is still just a direction classifier, not a
trade/no-trade decider). Regime3 current bull/bear/chop merged causally (merge_asof backward) onto
the 2024-2026 training frame BEFORE the train/inference split -- this is a training-time row
filter using only past-available regime probs, no lookahead.
"""

from __future__ import annotations

import argparse
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

DATA_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_trendscan_20260705"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma10_regime_specialist_20260706"
REG_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
TRAIN_END = pd.Timestamp("2025-06-30 23:59:59")
TAPE_START = pd.Timestamp("2025-06-25")
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
SEEDS = [270705, 270710, 270715, 270720, 270725]
PFX = s6.PFX
CHOP_TRAIN_THR = 0.42  # same threshold as Sigma6's winning not_chop entry filter, overridable via --chop-thr


def load_all() -> pd.DataFrame:
    frames = [pd.read_parquet(DATA_DIR / f"sigma3_1h_{y}.parquet") for y in (2024, 2025, 2026)]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def merge_regime_train(df: pd.DataFrame) -> pd.DataFrame:
    reg = pd.concat([
        pd.read_csv(REG_DIR / f"training_features_{y}_regime3_current_sensitive_hmm_wide24.csv", parse_dates=["timestamp"])
        for y in ("2024", "2025", "2026_rebuilt")
    ], ignore_index=True).sort_values("timestamp")
    keep = ["timestamp", f"{PFX}bull_prob", f"{PFX}bear_prob", f"{PFX}chop_prob"]
    return pd.merge_asof(df.sort_values("timestamp"), reg[keep], on="timestamp", direction="backward")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chop-thr", type=float, default=CHOP_TRAIN_THR)
    args = ap.parse_args()
    chop_thr = args.chop_thr

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    df = merge_regime_train(df)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE and not c.startswith(PFX)]
    print(f"features: {len(feat_cols)}  chop_train_thr={chop_thr}", flush=True)

    train_mask = (df["timestamp"] <= TRAIN_END).to_numpy()
    spec_mask = train_mask & (df[f"{PFX}chop_prob"].to_numpy(np.float64) < chop_thr)
    print(f"generalist train rows: {train_mask.sum()}, specialist (non-chop only) train rows: {spec_mask.sum()} "
          f"({spec_mask.sum() / max(train_mask.sum(), 1):.1%} kept)", flush=True)

    Xtr = df.loc[spec_mask, feat_cols].to_numpy(dtype=np.float64)
    ytr = df.loc[spec_mask, "ts_action"].to_numpy(dtype=np.int64)
    w = np.clip(np.abs(df.loc[spec_mask, "ts_t_value"].to_numpy(dtype=np.float64)), 0.5, 12.0)
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
        "atr_pct": sub["atr_pct"].astype(float),
        "primary_action": final_action, "primary_side": side,
        "primary_dir_p_cash": pc, "primary_dir_p_long": pl, "primary_dir_p_short": ps,
        "primary_quality_p_cash": pc, "primary_quality_p_long": pl, "primary_quality_p_short": ps,
        "fallback_dir_p_cash": 1.0, "fallback_dir_p_long": 0.0, "fallback_dir_p_short": 0.0,
        "fallback_quality_p_cash": 1.0, "fallback_quality_p_long": 0.0, "fallback_quality_p_short": 0.0,
        "fallback_side": 0, "primary_route_margin": 1.0,
    })
    tape.to_parquet(OUT_DIR / "tape_specialist.parquet", index=False)
    print(f"nonzero pct: {(tape['primary_side'] != 0).mean():.3f}", flush=True)

    # merge CryptoMamba stability + bull/bear/chop for backtest (same as Sigma6/Sigma8 convention)
    cm_dir = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601"
    cm = pd.concat([pd.read_csv(cm_dir / f"training_features_{y}_regime3_cryptomamba_h6_sidecar_20260601.csv", parse_dates=["timestamp"]) for y in ("2025", "2026_rebuilt")], ignore_index=True).sort_values("timestamp")
    reg = pd.concat([pd.read_csv(REG_DIR / f"training_features_{y}_regime3_current_sensitive_hmm_wide24.csv", parse_dates=["timestamp"]) for y in ("2025", "2026_rebuilt")], ignore_index=True).sort_values("timestamp")
    keep = ["timestamp", f"{PFX}bull_prob", f"{PFX}bear_prob", f"{PFX}chop_prob"]
    tape_bt = pd.merge_asof(tape.sort_values("timestamp"), reg[keep], on="timestamp", direction="backward")
    tape_bt = pd.merge_asof(tape_bt, cm[["timestamp", "regime3_cmamba_h6_sidecar_stability_score"]], on="timestamp", direction="backward")
    tape_bt = tape_bt.sort_values("i").reset_index(drop=True)

    tapes = {thr: v2.apply_quality_threshold(tape_bt, thr) for thr in (0.60, 0.70)}
    base = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3, reg_mode="not_chop", stab_thr=0.55)
    print("\n=== Sigma10 regime-specialist vs Sigma6 generalist, VAL 2025-07..12 ===", flush=True)
    for name, cfg in {"lev3 (thr0.70, rthr0.50)": dict(thr=0.70, leverage=3.0, sl_atr=2.5, reg_thr=0.50),
                       "lev4 (thr0.70, rthr0.42)": dict(thr=0.70, leverage=4.0, sl_atr=2.5, reg_thr=0.42)}.items():
        c = dict(cfg); thr = c.pop("thr"); tpx = tapes[thr]
        rv = s6.backtest(tpx, fee_mult=1.0, start=s6.VAL_START, end=s6.VAL_END, **c, **base)
        print(f"{name}: VAL c1={rv['pnl']:.1f}% mdd={rv['mdd']:.1f}% tr={rv['trades']} wr={rv['wr']:.3f}", flush=True)
    print("\nSigma6 generalist baseline for reference: lev3 +34.3% (mdd -14.2%) / lev4 +71.1% (mdd -15.9%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
