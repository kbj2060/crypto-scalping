#!/usr/bin/env python3
"""K (ATR multiple) sweep at the winning HORIZON=30/CLUSTER_GAP=12 config (picked by the TabPFN
confirmation pass) -- checks whether a different K gives a more balanced hit rate (this project's
convention: "K는 스윕 후 균형분포(50/50 근접)로 선택") without sacrificing AUC. GBM proxy (fast,
local, no server) -- final K still gets a TabPFN re-check once chosen, matching this project's
own 2-stage screen-then-confirm practice.
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
from research_eth_liquidity_sweep_topdown_metalabel_gridscreen_20260830 import (  # noqa: E402
    cluster_dedup_by_penetration,
    load_klines,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SWEEP_LOOKBACK = 48
HORIZON = 30
GAP = 12
K_GRID = [4.0, 4.5, 5.0, 5.5, 6.0]
GBM_SEED = 20260830


def log(msg: str) -> None:
    print(f"[liq_sweep_ksweep] {msg}", flush=True)


def main() -> int:
    klines = load_klines()
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    ind = build_indicator_frame(klines)
    assert len(sig) == len(ind) and (sig["timestamp"].to_numpy() == ind["timestamp"].to_numpy()).all()

    high = sig["high"].to_numpy(); low = sig["low"].to_numpy(); close = sig["close"].to_numpy()
    atr_pct = ind["atr_pct"].to_numpy(); ts = sig["timestamp"].to_numpy(); n = len(sig)
    swing_low_prior = pd.Series(low).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1).to_numpy()
    swing_high_prior = pd.Series(high).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1).to_numpy()

    # cluster-anchor once (independent of K), reuse pred_dir_ret/atr across all K
    anchored = {}
    for side, col in [("bottom", "bottom_liquidity_sweep"), ("top", "top_liquidity_sweep")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - HORIZON) & (ts[idx] >= np.datetime64(START))]
        idx = np.sort(idx)
        penetration = (swing_low_prior[idx] - low[idx]) if side == "bottom" else (high[idx] - swing_high_prior[idx])
        idx = cluster_dedup_by_penetration(idx, penetration, GAP)
        entry = close[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + HORIZON + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + HORIZON + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        anchored[side] = {"idx": idx, "pred_dir_ret": pred_dir_ret, "atr": atr_pct[idx]}

    for K in K_GRID:
        rows = []
        for side in ("bottom", "top"):
            a = anchored[side]
            hit = (a["pred_dir_ret"] >= K * a["atr"]).astype(float)
            feat_rows = ind.iloc[a["idx"]]
            out = pd.DataFrame({"timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
                                 "hit": hit, "is_bottom": 1 if side == "bottom" else 0})
            for c in FEATURE_COLUMNS:
                if c != "is_bottom":
                    out[c] = feat_rows[c].to_numpy()
            rows.append(out)
        fires = pd.concat(rows, ignore_index=True).dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
        tsf = fires["timestamp"]
        train = fires.loc[tsf < VAL_START]
        val = fires.loc[(tsf >= VAL_START) & (tsf < OOS_START)]
        oos = fires.loc[(tsf >= OOS_START) & (tsf < HOLDOUT_START)]

        clf = HistGradientBoostingClassifier(random_state=GBM_SEED)
        clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
        val_auc = roc_auc_score(val["hit"].to_numpy().astype(int), clf.predict_proba(val[FEATURE_COLUMNS])[:, 1])
        oos_auc = roc_auc_score(oos["hit"].to_numpy().astype(int), clf.predict_proba(oos[FEATURE_COLUMNS])[:, 1])
        log(f"K={K}: hit_rate={fires['hit'].mean():.3f} (bottom={fires.loc[fires.side=='bottom','hit'].mean():.3f} "
            f"top={fires.loc[fires.side=='top','hit'].mean():.3f}) VAL={val_auc:.4f} OOS={oos_auc:.4f} "
            f"min={min(val_auc,oos_auc):.4f} n_train={len(train)}(pos={int(train['hit'].sum())}) "
            f"n_val={len(val)}(pos={int(val['hit'].sum())}) n_oos={len(oos)}(pos={int(oos['hit'].sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
