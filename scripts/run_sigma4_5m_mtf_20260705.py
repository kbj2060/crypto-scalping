#!/usr/bin/env python3
"""Sigma4: train the 5m-MTF HGB ensemble and sweep the pre-registered grid on validation
(2025-07..12), gating on COST1 (per user: cost3 is too strict, report it as context only).
5m decision cadence, 1h context as reference features.

If a config passes the (relaxed, cost1) gate, freeze a robust one and one-shot 2026-03-02..06-30.
Caveat: that window was scored once for Sigma3-1h; this is its 2nd use -> degraded evidential
value. Sigma4 is a different model family (5m MTF), but note the overlap honestly.
"""

from __future__ import annotations

import argparse
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

DATA_DIR = ROOT / "tmp/causal_regen_20260516/sigma4_5m_mtf_20260705"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma4_5m_mtf_20260705"
TRAIN_END = pd.Timestamp("2025-06-30 23:59:59")
TAPE_START = pd.Timestamp("2025-06-24")
VAL_START = pd.Timestamp("2025-07-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-03-02")
OOS_END = pd.Timestamp("2026-06-30 23:59:59")
SEEDS = [270705, 270710, 270715, 270720, 270725]
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}

THRESHOLDS = [0.55, 0.65, 0.75]
PERSISTS = [0, 3, 6]
TPSL = [(2.5, 1.5), (3.5, 1.5), (5.0, 2.0)]
COOLDOWNS = [6, 12]
MAX_HOLD = 288


def load_all() -> pd.DataFrame:
    fr = [pd.read_parquet(DATA_DIR / f"sigma4_5m_{y}.parquet") for y in (2024, 2025, 2026)]
    df = pd.concat(fr, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def build_tape(df, proba, mask):
    sub = df.loc[mask].reset_index(drop=True)
    pc, pl, ps = proba[mask, 0], proba[mask, 1], proba[mask, 2]
    P = np.column_stack([pc, pl, ps])
    da = P.argmax(axis=1)
    qual = np.where(da > 0, P[np.arange(len(sub)), da], P[:, 0])
    fa = np.where((da != 0) & (qual >= 0.45), da, 0)
    side = np.where(fa == 1, 1, np.where(fa == 2, -1, 0))
    return pd.DataFrame({
        "i": np.arange(len(sub)), "timestamp": sub["timestamp"],
        "open": sub["open"].astype(float), "high": sub["high"].astype(float),
        "low": sub["low"].astype(float), "close": sub["close"].astype(float),
        "jump_flag": 0.0, "evt_tail_flag": 0.0, "jump_z": 0.0, "atr_pct": sub["atr_pct"].astype(float),
        "primary_action": fa, "primary_side": side, "primary_expert": "sigma4",
        "primary_route_confidence": 1.0, "primary_route_margin": 1.0,
        "primary_dir_p_cash": pc, "primary_dir_p_long": pl, "primary_dir_p_short": ps,
        "primary_quality_p_cash": pc, "primary_quality_p_long": pl, "primary_quality_p_short": ps,
        "primary_quality_score": np.where(fa != 0, qual, 0.0), "primary_confidence": P.max(axis=1),
        "fallback_action": 0, "fallback_side": 0, "fallback_expert": "none",
        "fallback_route_confidence": 0.0, "fallback_route_margin": 0.0,
        "fallback_dir_p_cash": 1.0, "fallback_dir_p_long": 0.0, "fallback_dir_p_short": 0.0,
        "fallback_quality_p_cash": 1.0, "fallback_quality_p_long": 0.0, "fallback_quality_p_short": 0.0,
        "fallback_quality_score": 0.0, "fallback_confidence": 0.0,
    })


def gate_cost1(r: dict) -> bool:
    c1, c3 = r["cost1"], r["cost3"]
    # RELAXED per user: gate on cost1 only; cost3 reported but not required positive.
    return c1["pnl"] > 0 and c1["mdd"] >= -20.0 and c3["mdd"] >= -25.0 and c1["trades"] >= 60 and len(c1["trades_by_month"]) >= 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-shot", action="store_true", help="also score the frozen config on 2026-03-02..06-30")
    ap.add_argument("--freeze", default="", help="thr,per,tp,sl,cd to freeze for one-shot")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    print(f"features: {len(feat_cols)}", flush=True)
    train_mask = (df["timestamp"] <= TRAIN_END).to_numpy()
    Xtr = df.loc[train_mask, feat_cols].to_numpy(dtype=np.float64)
    ytr = df.loc[train_mask, "ts_action"].to_numpy(dtype=np.int64)
    w = np.clip(np.abs(df.loc[train_mask, "ts_t_value"].to_numpy(dtype=np.float64)), 0.5, 12.0)
    Xall = df[feat_cols].to_numpy(dtype=np.float64)

    proba_sum = np.zeros((len(df), 3))
    for s in SEEDS:
        clf = HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.03, max_iter=250,
                                             max_depth=4, l2_regularization=1.0, max_leaf_nodes=31,
                                             min_samples_leaf=120, early_stopping=False,
                                             random_state=int(s), class_weight="balanced")
        clf.fit(Xtr, ytr, sample_weight=w)
        pr = clf.predict_proba(Xall)
        cm = {c: i for i, c in enumerate(list(clf.classes_))}
        for k in (0, 1, 2):
            if k in cm:
                proba_sum[:, k] += pr[:, cm[k]]
        print(f"seed {s} done", flush=True)
    proba = proba_sum / len(SEEDS)

    tape_mask = (df["timestamp"] >= TAPE_START).to_numpy()
    tape = build_tape(df, proba, tape_mask)
    tape.to_parquet(OUT_DIR / "tape_ensemble.parquet", index=False)
    print(f"tape rows {len(tape)}, atr_pct median {tape['atr_pct'].median():.5f}, nonzero {(tape['primary_side']!=0).mean():.3f}", flush=True)

    tapes = {thr: v2.apply_quality_threshold(tape, thr) for thr in THRESHOLDS}
    rows = []
    for thr, per, (tp, sl), cd in itertools.product(THRESHOLDS, PERSISTS, TPSL, COOLDOWNS):
        cfg = v2.VariantConfig(name=f"s4_qt{thr}_p{per}_tp{tp}_sl{sl}_cd{cd}", tp_mode="atr_scaled",
                               tp_atr_mult=tp, sl_atr_mult=sl, sizing_mode="fixed", fixed_margin=0.30,
                               fixed_leverage=2.0, cooldown_bars=cd, quality_threshold=thr,
                               persistence_bars=per, max_hold_bars=MAX_HOLD, use_fallback=False)
        r = v2.cost_stress(tapes[thr], cfg, start=VAL_START, end=VAL_END)
        rows.append({"thr": thr, "per": per, "tp": tp, "sl": sl, "cd": cd,
                     "c1": round(r["cost1"]["pnl"], 2), "c1mdd": round(r["cost1"]["mdd"], 2),
                     "c1tr": r["cost1"]["trades"], "c1wr": round(r["cost1"]["wr"], 3),
                     "c3": round(r["cost3"]["pnl"], 2), "c3mdd": round(r["cost3"]["mdd"], 2),
                     "mo": len(r["cost1"]["trades_by_month"]), "pass": gate_cost1(r)})
    rdf = pd.DataFrame(rows).sort_values(["pass", "c1"], ascending=[False, False])
    rdf.to_csv(OUT_DIR / "gate_ranking.csv", index=False)
    print(f"\ngate_pass (cost1): {int(rdf['pass'].sum())}/{len(rdf)}", flush=True)
    print(rdf.head(16).to_string(index=False), flush=True)

    if args.one_shot and args.freeze:
        thr, per, tp, sl, cd = args.freeze.split(",")
        cfg = v2.VariantConfig(name="frozen", tp_mode="atr_scaled", tp_atr_mult=float(tp), sl_atr_mult=float(sl),
                               sizing_mode="fixed", fixed_margin=0.30, fixed_leverage=2.0, cooldown_bars=int(cd),
                               quality_threshold=float(thr), persistence_bars=int(per), max_hold_bars=MAX_HOLD, use_fallback=False)
        tp_qt = v2.apply_quality_threshold(tape, float(thr))
        r = v2.cost_stress(tp_qt, cfg, start=OOS_START, end=OOS_END)
        print("\n=== ONE-SHOT 2026-03-02..06-30 (2nd use, degraded evidential value) ===", flush=True)
        for tag in ("cost1", "cost3"):
            x = r[tag]
            print(f"{tag}: pnl={x['pnl']:.2f}% mdd={x['mdd']:.2f}% trades={x['trades']} wr={x['wr']:.3f} months={len(x['trades_by_month'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
