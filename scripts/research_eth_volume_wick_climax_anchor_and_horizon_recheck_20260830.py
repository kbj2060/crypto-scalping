#!/usr/bin/env python3
"""volume_wick_climax v1 re-check, prompted by user pushback: the pre-DL rule-based scorecard
(2026-08-25) showed this signal solidly mid-pack (1h lift 2.50-2.94x, beating dalton_rule2's
1.60x), yet the TabPFN meta-label came out weakest of the 3 signals processed (VAL/OOS/HOLDOUT
0.612/0.563/0.565). User's hypothesis: something in the label/feature construction is actually
wrong, not just "this signal is inherently harder" -- investigate rather than accept.

2 concrete, testable hypotheses that are STRUCTURALLY UNIQUE to this signal (taker/short_term_
return_z don't have them, since their fire condition is a single signed variable's threshold, not
an AND of two different variables):

1. ATR self-inclusion contamination differing by signal (checked locally first, BEFORE writing this
   script, via a plain numpy comparison of fire-bar TR / 14-bar-ATR-including-itself across the 3
   signals): volume_wick_climax's fires have LOWER self-contribution (median 12.2%) than
   taker(14.8%)/short_term_return_z(12.6%), not higher -- this hypothesis is REFUTED by the data,
   not investigated further here.

2. Cluster-anchor criterion mismatch: v1 anchors same-side clusters (gap<=3 bars) on vol_z alone
   (loudest volume bar). Checked locally: in 52-56% of multi-fire clusters, the vol_z-loudest bar is
   a DIFFERENT bar than the one with the most extreme wick_ratio -- i.e. picking "loudest volume"
   often does NOT pick "cleanest wick shape". Since the fire condition itself is an AND of both
   variables, anchoring on only one of them is a plausible, previously-untested design choice that
   could be diluting the label. This script tests 3 anchor criteria head-to-head: vol_z-only
   (v1, current), wick_ratio-only, and a combined vol_z*wick_ratio intensity score.

Also re-checks HORIZON robustness: the original 9-combo grid (6/12/24 x gap 3/6/12) found HORIZON=24
best and HORIZON=12 WORST -- a non-monotonic pattern across only 3 sparse points, worth confirming
isn't a lucky VAL draw with more points nearby (16/20/30/36/48 added, gap=3 fixed, vol_z anchor
fixed since that's the currently-adopted criterion).

Both checks are single-seed VAL+OOS AUC screening only (TRAIN-fit, HOLDOUT untouched) -- purely
diagnostic, no design change is finalized in this script; if either reveals something, a follow-up
full 4-seed rebuild would happen only after reviewing results.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from live_evidence_signal_dashboard_20260823 import compute_signals
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
    FEATURE_COLUMNS, build_indicator_frame, load_klines,
)

START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SEED = 20260829
GAP = 3


def log(msg: str) -> None:
    print(f"[vwc_recheck] {msg}", flush=True)


def cluster_dedup_generic(idx: np.ndarray, score_at_idx: np.ndarray, gap: int) -> np.ndarray:
    """Same mechanism as v1's cluster_dedup_by_vol_z, generalized to any unsigned 'higher=more
    extreme' score array (vol_z, wick_ratio, or their product)."""
    order = np.argsort(idx)
    idx_sorted, s_sorted = idx[order], score_at_idx[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "s": s_sorted})
    keep = df.loc[df.groupby("cluster")["s"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def build_fires(klines, indicator_frame, sig, gap: int, horizon: int, anchor: str, K: float | None) -> pd.DataFrame:
    high = sig["high"].to_numpy(); low = sig["low"].to_numpy(); close = sig["close"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    vol_z_all = indicator_frame["vol_z"].to_numpy()
    lower_wick_all = indicator_frame["lower_wick_ratio"].to_numpy()
    upper_wick_all = indicator_frame["upper_wick_ratio"].to_numpy()
    ts = sig["timestamp"].to_numpy()
    n = len(sig)
    rows = []
    for side, col, wick_all in [("bottom", "bottom_volume_wick_climax", lower_wick_all),
                                 ("top", "top_volume_wick_climax", upper_wick_all)]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - horizon) & (ts[idx] >= np.datetime64(START))]
        if anchor == "vol_z":
            score = vol_z_all[idx]
        elif anchor == "wick_ratio":
            score = wick_all[idx]
        elif anchor == "combined":
            score = vol_z_all[idx] * wick_all[idx]
        else:
            raise ValueError(anchor)
        idx = cluster_dedup_generic(idx, score, gap)
        entry = close[idx]; a = atr_pct[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        feat_rows = indicator_frame.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "entry": entry, "atr_pct": a, "pred_dir_ret": pred_dir_ret,
            "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    fires = fires.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    if K is None:
        pred, a = fires["pred_dir_ret"].to_numpy(), fires["atr_pct"].to_numpy()
        best_k, best_diff = None, np.inf
        for k in np.round(np.arange(0.30, 3.01, 0.05), 2):
            diff = abs(float((pred >= k * a).mean()) - 0.5)
            if diff < best_diff:
                best_diff, best_k = diff, float(k)
        K = best_k
    fires["hit"] = (fires["pred_dir_ret"] >= K * fires["atr_pct"]).astype(float)
    fires.attrs["K"] = K
    return fires


def screen(fires: pd.DataFrame, tag: str) -> dict:
    from tabpfn import TabPFNClassifier
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    clf = TabPFNClassifier(device="cuda", random_state=SEED)
    clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
    val_auc = roc_auc_score(val["hit"].to_numpy().astype(int), clf.predict_proba(val[FEATURE_COLUMNS])[:, 1])
    oos_auc = roc_auc_score(oos["hit"].to_numpy().astype(int), clf.predict_proba(oos[FEATURE_COLUMNS])[:, 1])
    row = {"tag": tag, "K": round(fires.attrs["K"], 2), "n_fires": len(fires), "n_train": len(train),
           "n_val": len(val), "n_oos": len(oos), "hit_rate": round(float(fires["hit"].mean()), 4),
           "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4)}
    log(f"  {tag:<28s} K={row['K']:.2f} n={row['n_fires']:>5d} hit_rate={row['hit_rate']:.3f} "
        f"VAL_AUC={row['val_auc']:.4f} OOS_AUC={row['oos_auc']:.4f}")
    return row


def main() -> int:
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)

    log("=== check 1: cluster anchor criterion (HORIZON=24, GAP=3 fixed) ===")
    anchor_rows = []
    for anchor in ["vol_z", "wick_ratio", "combined"]:
        fires = build_fires(klines, indicator_frame, sig, GAP, 24, anchor, K=None)
        anchor_rows.append(screen(fires, f"anchor={anchor}"))
    pd.DataFrame(anchor_rows).to_csv("/tmp/vwc_recheck_anchor.csv", index=False)

    log("")
    log("=== check 2: finer HORIZON grid (anchor=vol_z/v1, GAP=3 fixed) ===")
    horizon_rows = []
    for h in [8, 12, 16, 20, 24, 30, 36, 48]:
        fires = build_fires(klines, indicator_frame, sig, GAP, h, "vol_z", K=None)
        horizon_rows.append(screen(fires, f"horizon={h}"))
    pd.DataFrame(horizon_rows).to_csv("/tmp/vwc_recheck_horizon.csv", index=False)

    log("")
    log("=== SUMMARY: anchor check ===")
    for r in anchor_rows:
        log(f"  {r['tag']:<28s} VAL={r['val_auc']:.4f}  OOS={r['oos_auc']:.4f}")
    log("=== SUMMARY: horizon check ===")
    for r in horizon_rows:
        log(f"  {r['tag']:<28s} VAL={r['val_auc']:.4f}  OOS={r['oos_auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
