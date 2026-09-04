#!/usr/bin/env python3
"""HORIZON x CLUSTER_GAP grid screen for liquidity_sweep "top/down" metalabel (Homer signal #2
redo, standard touch-based-MFE template). Fast GBM proxy (not TabPFN -- a cheap screening pass,
matching dalton_rule2_balance_edge/volume_wick_climax's own "screen many combos cheaply, confirm
winner with TabPFN" 2-stage practice) over HORIZON in {8,12,16,20,24,30,36,48} x CLUSTER_GAP in
{3,6,12}, K fixed at 1.5 (phase1 diagnostic: research_eth_liquidity_sweep_topdown_metalabel_
phase1_20260830.py showed K~1.5 gives roughly-balanced hit rates in the HORIZON=12-24 range).
Selection rule per docs/homer/README.md 5.5: min(VAL,OOS) AUC, not a single split's max (the
volume_wick_climax lesson -- a 3-point/VAL-only pick chose an overfit HORIZON there).

Cluster-anchor keeps, per same-side cluster (gap <= CLUSTER_GAP_MERGE bars), the fire bar with the
DEEPEST sweep penetration (swept level minus wick extreme) -- the causal, definition-intrinsic
"how extreme was this sweep" metric, analogous to taker_delta_z_climax's own delta_z-magnitude
cluster anchor (never the price OUTCOME -- non-circular).
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SWEEP_LOOKBACK = 48

K = 1.5
HORIZON_GRID = [8, 12, 16, 20, 24, 30, 36, 48]
GAP_GRID = [3, 6, 12]
GBM_SEED = 20260830


def log(msg: str) -> None:
    print(f"[liq_sweep_topdown_gridscreen] {msg}", flush=True)


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all(), "gap/dup in klines"
    return df


def cluster_dedup_by_penetration(idx: np.ndarray, penetration: np.ndarray, gap: int) -> np.ndarray:
    """Collapse consecutive same-side fires (gap<=`gap` bars) into one cluster, keep only the bar
    with the DEEPEST penetration per cluster. `idx` must already be sorted ascending."""
    cluster_id = np.zeros(len(idx), dtype=int)
    cid = 0
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx, "cluster": cluster_id, "pen": penetration})
    keep = df.loc[df.groupby("cluster")["pen"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def build_fires(klines: pd.DataFrame, ind: pd.DataFrame, sig: pd.DataFrame,
                 horizon: int, gap: int) -> pd.DataFrame:
    high = sig["high"].to_numpy()
    low = sig["low"].to_numpy()
    close = sig["close"].to_numpy()
    atr_pct = ind["atr_pct"].to_numpy()
    ts = sig["timestamp"].to_numpy()
    n = len(sig)

    swing_low_prior = low_s = pd.Series(low).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1).to_numpy()
    swing_high_prior = pd.Series(high).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1).to_numpy()

    rows = []
    for side, col in [("bottom", "bottom_liquidity_sweep"), ("top", "top_liquidity_sweep")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - horizon) & (ts[idx] >= np.datetime64(START))]
        idx = np.sort(idx)
        if side == "bottom":
            penetration = swing_low_prior[idx] - low[idx]
        else:
            penetration = high[idx] - swing_high_prior[idx]
        idx = cluster_dedup_by_penetration(idx, penetration, gap)

        entry = close[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        hit = (pred_dir_ret >= K * atr_pct[idx]).astype(float)

        feat_rows = ind.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "hit": hit, "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)
    return pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    log("loading klines + building indicator frame + signals (once, shared across grid)...")
    klines = load_klines()
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    ind = build_indicator_frame(klines)
    assert len(sig) == len(ind) and (sig["timestamp"].to_numpy() == ind["timestamp"].to_numpy()).all()
    log(f"{len(klines)} bars ready")

    results = []
    for horizon in HORIZON_GRID:
        for gap in GAP_GRID:
            fires = build_fires(klines, ind, sig, horizon, gap)
            n_before = len(fires)
            fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
            ts = fires["timestamp"]
            train = fires.loc[ts < VAL_START].reset_index(drop=True)
            val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
            oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)

            clf = HistGradientBoostingClassifier(random_state=GBM_SEED)
            clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
            val_auc = roc_auc_score(val["hit"].to_numpy().astype(int), clf.predict_proba(val[FEATURE_COLUMNS])[:, 1])
            oos_auc = roc_auc_score(oos["hit"].to_numpy().astype(int), clf.predict_proba(oos[FEATURE_COLUMNS])[:, 1])
            gap_metric = abs(val_auc - oos_auc)
            min_auc = min(val_auc, oos_auc)
            hit_rate = float(fires["hit"].mean())
            row = {
                "horizon": horizon, "gap": gap, "n_fires": n_before, "n_usable": len(fires),
                "n_train": len(train), "n_val": len(val), "n_oos": len(oos),
                "hit_rate": round(hit_rate, 4),
                "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4),
                "val_oos_gap": round(float(gap_metric), 4), "min_val_oos": round(float(min_auc), 4),
            }
            results.append(row)
            log(f"  H={horizon:>3d} GAP={gap:>2d}: n_fires={n_before:>5d}->{len(fires):>5d} "
                f"hit_rate={hit_rate:.3f} VAL={val_auc:.4f} OOS={oos_auc:.4f} "
                f"|gap|={gap_metric:.4f} min={min_auc:.4f}")

    table = pd.DataFrame(results).sort_values("min_val_oos", ascending=False)
    log("\n=== TOP 8 by min(VAL,OOS) AUC ===")
    for _, r in table.head(8).iterrows():
        log(f"  H={int(r['horizon']):>3d} GAP={int(r['gap']):>2d}: VAL={r['val_auc']:.4f} OOS={r['oos_auc']:.4f} "
            f"min={r['min_val_oos']:.4f} |gap|={r['val_oos_gap']:.4f} hit_rate={r['hit_rate']:.3f} n={int(r['n_usable'])}")

    out_dir = ROOT / "tmp/eth_liquidity_sweep_topdown_metalabel_20260830"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "gridscreen_gbm_results.csv", index=False)
    log(f"\nfull grid saved -> {out_dir / 'gridscreen_gbm_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
