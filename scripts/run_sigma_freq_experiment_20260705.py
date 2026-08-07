#!/usr/bin/env python3
"""Parametrized frequency experiment: given a bar frequency (2h, 4h, ...), rebuild the
trend-scanning dataset, train a 5-seed HGB ensemble on 2024-01..2025-06, emit a decision tape,
and run the SAME pre-registered gate grid on validation (2025-07..12). Reuses Sigma3's proven
recipe (1h version already validated at OOS cost1 +7.34%/cost3 -3.88%); pushing frequency lower
should widen the cost3 margin (bigger per-trade moves vs fixed costs).

Barriers are ATR-multiples so the same grid transfers across frequencies (ATR auto-scales with
bar size). max_hold scales to hold a ~2-day cap regardless of frequency.

Validation only. One-shot on 2026-03-02..06-30 is done separately, once, by the caller, ONLY if
a config passes here -- and with the explicit caveat that that window was already scored once for
Sigma3-1h (2nd use = degraded evidential value; a fresh 2026-07+ window is preferable when it
accumulates).
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

from build_1h_trendscan_dataset_20260705 import compute_features, SRC_FILES  # noqa: E402
from build_trend_scanning_action_labels_20260531 import _trend_scan_fast  # noqa: E402
import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

TRAIN_END = pd.Timestamp("2025-06-30 23:59:59")
TAPE_START = pd.Timestamp("2025-06-25")
VAL_START = pd.Timestamp("2025-07-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")
SEEDS = [270705, 270710, 270715, 270720, 270725]
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
THRESHOLDS = [0.50, 0.60, 0.70]
PERSISTS = [0, 2, 4]
TPSL = [(1.5, 1.0), (2.0, 0.9), (2.5, 1.2)]


def resample(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    f = f.set_index("timestamp").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "quote_volume": "sum", "taker_buy_base": "sum", "close_btc": "last"}
    agg = {k: v for k, v in agg.items() if k in f.columns}
    last_cols = [c for c in ("last_funding_rate", "sum_open_interest_value",
                             "sum_toptrader_long_short_ratio", "count_long_short_ratio", "volume_btc")
                 if c in f.columns]
    r = f.resample(freq, label="left", closed="left").agg(agg)
    for c in last_cols:
        r[c] = f[c].resample(freq, label="left", closed="left").last()
    return r.dropna(subset=["open", "high", "low", "close"]).reset_index()


def build_year(year: int, freq: str, windows: np.ndarray, thr: float) -> pd.DataFrame:
    src = pd.read_csv(SRC_FILES[year], low_memory=False)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    r = resample(src, freq)
    feats = compute_features(r)
    logc = np.log(np.maximum(feats["close"].to_numpy(dtype=np.float64), 1e-12))
    t_vals, opt_l, betas = _trend_scan_fast(logc, windows)
    labels = np.zeros(len(feats), dtype=np.int64)
    labels[(np.abs(t_vals) >= thr) & (betas > 0)] = 1
    labels[(np.abs(t_vals) >= thr) & (betas < 0)] = 2
    feats["ts_action"] = labels
    feats["ts_t_value"] = t_vals.astype(np.float32)
    feats["ts_opt_L"] = opt_l.astype(np.int16)
    return feats


def passes_gates_6mo(result: dict) -> bool:
    c1, c3 = result["cost1"], result["cost3"]
    return (c1["pnl"] > 0 and c3["pnl"] > 0 and c1["mdd"] >= -20.0 and c3["mdd"] >= -20.0
            and c1["trades"] >= 40 and len(c1["trades_by_month"]) >= 5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", required=True, help="pandas offset, e.g. 2h or 4h")
    ap.add_argument("--windows", required=True, help="trend-scan forward windows in bars, comma-sep")
    ap.add_argument("--ts-threshold", type=float, default=2.5)
    ap.add_argument("--max-hold", type=int, required=True, help="max hold bars (~2 days)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = ROOT / "tmp/causal_regen_20260516" / f"sigma_freq_{args.tag}_20260705"
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = np.array(sorted({int(w) for w in args.windows.split(",")}), dtype=np.int32)

    frames = [build_year(y, args.freq, windows, float(args.ts_threshold)) for y in (2024, 2025, 2026)]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    counts = np.bincount(df["ts_action"].to_numpy(), minlength=3).tolist()
    print(f"[{args.tag}] freq={args.freq} rows={len(df)} features={len(feat_cols)} labelCLS={counts}", flush=True)

    train_mask = df["timestamp"] <= TRAIN_END
    Xtr = df.loc[train_mask, feat_cols].to_numpy(dtype=np.float64)
    ytr = df.loc[train_mask, "ts_action"].to_numpy(dtype=np.int64)
    w = np.clip(np.abs(df.loc[train_mask, "ts_t_value"].to_numpy(dtype=np.float64)), 0.5, 12.0)
    Xall = df[feat_cols].to_numpy(dtype=np.float64)
    print(f"train rows: {len(Xtr)}", flush=True)

    proba_sum = np.zeros((len(df), 3))
    for s in SEEDS:
        clf = HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.03, max_iter=250,
                                             max_depth=4, l2_regularization=1.0, max_leaf_nodes=31,
                                             min_samples_leaf=80, early_stopping=False,
                                             random_state=int(s), class_weight="balanced")
        clf.fit(Xtr, ytr, sample_weight=w)
        pr = clf.predict_proba(Xall)
        cm = {c: i for i, c in enumerate(list(clf.classes_))}
        o = np.zeros((len(df), 3))
        for k in (0, 1, 2):
            if k in cm:
                o[:, k] = pr[:, cm[k]]
        proba_sum += o
    proba = proba_sum / len(SEEDS)

    tape_mask = (df["timestamp"] >= TAPE_START).to_numpy()
    sub = df.loc[tape_mask].reset_index(drop=True)
    pc, pl, ps = proba[tape_mask, 0], proba[tape_mask, 1], proba[tape_mask, 2]
    P = np.column_stack([pc, pl, ps])
    da = P.argmax(axis=1)
    qual = np.where(da > 0, P[np.arange(len(sub)), da], P[:, 0])
    fa = np.where((da != 0) & (qual >= 0.45), da, 0)
    side = np.where(fa == 1, 1, np.where(fa == 2, -1, 0))
    tape = pd.DataFrame({
        "i": np.arange(len(sub)), "timestamp": sub["timestamp"],
        "open": sub["open"].astype(float), "high": sub["high"].astype(float),
        "low": sub["low"].astype(float), "close": sub["close"].astype(float),
        "jump_flag": 0.0, "evt_tail_flag": 0.0, "jump_z": 0.0, "atr_pct": sub["atr_pct"].astype(float),
        "primary_action": fa, "primary_side": side, "primary_expert": args.tag,
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
    tape.to_parquet(out_dir / "tape_ensemble.parquet", index=False)
    print(f"tape rows {len(tape)}, atr_pct median {tape['atr_pct'].median():.5f}, nonzero {(tape['primary_side']!=0).mean():.3f}", flush=True)

    tapes = {thr: v2.apply_quality_threshold(tape, thr) for thr in THRESHOLDS}
    rows = []
    for thr, per, (tp, sl) in itertools.product(THRESHOLDS, PERSISTS, TPSL):
        cfg = v2.VariantConfig(name=f"{args.tag}_qt{thr}_p{per}_tp{tp}_sl{sl}", tp_mode="atr_scaled",
                               tp_atr_mult=tp, sl_atr_mult=sl, sizing_mode="fixed", fixed_margin=0.30,
                               fixed_leverage=2.0, cooldown_bars=3, quality_threshold=thr,
                               persistence_bars=per, max_hold_bars=int(args.max_hold), use_fallback=False)
        r = v2.cost_stress(tapes[thr], cfg, start=VAL_START, end=VAL_END)
        rows.append({"thr": thr, "per": per, "tp": tp, "sl": sl,
                     "c1": round(r["cost1"]["pnl"], 2), "c1mdd": round(r["cost1"]["mdd"], 2),
                     "c1tr": r["cost1"]["trades"], "c1wr": round(r["cost1"]["wr"], 3),
                     "c3": round(r["cost3"]["pnl"], 2), "c3mdd": round(r["cost3"]["mdd"], 2),
                     "mo": len(r["cost1"]["trades_by_month"]), "pass": passes_gates_6mo(r)})
    rdf = pd.DataFrame(rows).sort_values(["pass", "c3"], ascending=[False, False])
    rdf.to_csv(out_dir / "gate_ranking.csv", index=False)
    print(f"\n[{args.tag}] gate_pass: {int(rdf['pass'].sum())}/27", flush=True)
    print(rdf.head(12).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
