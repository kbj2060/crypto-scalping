#!/usr/bin/env python3
"""HORIZON x CLUSTER_GAP grid screen for the trend-continuation EXTEND/REVERT pure-direction
label. research_eth_trend_continuation_head_phase1_20260831.py only screened HORIZON at
GAP=12 fixed (5 points: 3/6/12/24/48) and never varied GAP -- sparser than this project's usual
practice (research_eth_kalman_demarker_gridscreen_20260831.py's 8-point HORIZON_GRID x 3-point
GAP_GRID joint screen, itself following research_eth_liquidity_sweep_topdown_metalabel_
gridscreen_20260830.py). This fills that gap: fast GBM proxy (not TabPFN), selection by
min(VAL,OOS) AUC (the volume_wick_climax lesson: never pick by a single split's max).

Difference from the kalman/demarker screen: K is NOT held fixed across the grid. For THIS
label K isn't an independently-chosen ATR-multiple threshold picked from phase1 diagnostics --
it IS defined as the median continuation/ATR ratio of whichever (H, GAP) candidate pool is
currently in play (docs/homer/README.md common-lesson #7: reclustering changes the population,
so a held-fixed K silently drifts off its 50/50 target -- the taker v4->v5 bug). So K is
recomputed fresh per cell, same as phase1 already did per-horizon -- this screen just adds the
GAP dimension on top.
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

from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

ETH_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_PATH = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

HORIZON_GRID = [8, 12, 16, 20, 24, 30, 36, 48]   # matches this project's usual density (kalman/demarker)
GAP_GRID = [3, 6, 12]                             # matches kalman/demarker's grid
GBM_SEED = 20260831


def log(msg: str) -> None:
    print(f"[trend_continuation_gridscreen] {msg}", flush=True)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df[df["timestamp"] >= START - pd.Timedelta(days=10)].reset_index(drop=True)


def forward_extremes(close: np.ndarray, high: np.ndarray, low: np.ndarray, h: int):
    fh = pd.Series(high).shift(-1).rolling(h, min_periods=h).max().shift(-(h - 1)).to_numpy()
    fl = pd.Series(low).shift(-1).rolling(h, min_periods=h).min().shift(-(h - 1)).to_numpy()
    return (fh - close) / close, (close - fl) / close


def main() -> int:
    eth, btc = load(ETH_PATH), load(BTC_PATH)
    sig = compute_signals(eth, btc, None)
    sig = sig[sig["timestamp"] >= START].reset_index(drop=True)
    feats = build_indicator_frame(eth)
    feats = feats[feats["timestamp"] >= START].reset_index(drop=True)
    assert len(feats) == len(sig) and (feats["timestamp"].to_numpy() == sig["timestamp"].to_numpy()).all()

    ts = sig["timestamp"]
    close = sig["close"].to_numpy(); high = sig["high"].to_numpy(); low = sig["low"].to_numpy()
    atr_pct = feats["atr_pct"].to_numpy()
    tall = ts.to_numpy()

    names = [n for n, _ in SIGNAL_ORDER]
    bot = np.zeros(len(sig), bool); top = np.zeros(len(sig), bool)
    for n in names:
        if f"bottom_{n}" in sig: bot |= sig[f"bottom_{n}"].to_numpy()
        if f"top_{n}" in sig:    top |= sig[f"top_{n}"].to_numpy()
    log(f"union bottom fires={bot.sum()}  top fires={top.sum()}")

    feat_cols = [c for c in FEATURE_COLUMNS if c != "is_bottom"]
    results = []
    for h in HORIZON_GRID:
        up, dn = forward_extremes(close, high, low, h)
        for gap in GAP_GRID:
            rows = []
            for side, m in (("bottom", bot), ("top", top)):
                last = -10**9
                for i in np.flatnonzero(m):
                    if i - last < gap:
                        continue
                    last = i
                    rows.append((i, side == "bottom"))
            ev = pd.DataFrame(rows, columns=["i", "is_bottom"]).sort_values("i").reset_index(drop=True)
            iu = ev["i"].to_numpy(); isb = ev["is_bottom"].to_numpy()
            cont = np.where(isb, dn[iu], up[iu]); rev = np.where(isb, up[iu], dn[iu])
            ok = ~np.isnan(cont) & ~np.isnan(rev) & (atr_pct[iu] > 0)
            if ok.sum() < 200:
                log(f"H={h:>3d} GAP={gap:>2d}: skipped (n={int(ok.sum())} too small)")
                continue
            k50 = float(np.median(cont[ok] / atr_pct[iu][ok]))
            y_ext = ok & (cont >= k50 * atr_pct[iu]); y_rev = ok & (rev >= k50 * atr_pct[iu])
            pure = y_ext ^ y_rev

            X = feats.iloc[iu][feat_cols].copy()
            X["is_bottom"] = isb.astype(int)
            y = y_ext.astype(int)
            t = tall[iu]
            tr = pure & (t < VAL_START)
            va = pure & (t >= VAL_START) & (t < OOS_START)
            oo = pure & (t >= OOS_START) & (t < HOLDOUT_START)
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2 or len(np.unique(y[oo])) < 2:
                log(f"H={h:>3d} GAP={gap:>2d}: skipped (degenerate class split)")
                continue
            clf = HistGradientBoostingClassifier(random_state=GBM_SEED)
            clf.fit(X[tr], y[tr])
            val_auc = roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1])
            oos_auc = roc_auc_score(y[oo], clf.predict_proba(X[oo])[:, 1])
            row = {
                "horizon": h, "gap": gap, "k50": round(k50, 3),
                "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_oos": int(oo.sum()),
                "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4),
                "val_oos_gap": round(abs(val_auc - oos_auc), 4),
                "min_val_oos": round(min(val_auc, oos_auc), 4),
            }
            results.append(row)
            log(f"H={h:>3d} GAP={gap:>2d}: K={k50:.2f} n_tr={row['n_train']:>5d} n_va={row['n_val']:>4d} "
                f"n_oo={row['n_oos']:>4d}  VAL={val_auc:.4f} OOS={oos_auc:.4f} min={row['min_val_oos']:.4f}")

    df = pd.DataFrame(results)
    out_dir = ROOT / "tmp/eth_trend_continuation_20260831"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "horizon_gap_gridscreen.csv", index=False)

    pd.set_option("display.width", 200)
    df_sorted = df.sort_values("min_val_oos", ascending=False)
    log("\n=== TOP 10 by min(VAL,OOS) AUC ===")
    print(df_sorted.head(10).to_string(index=False))
    log("\n=== BOTTOM 10 by min(VAL,OOS) AUC ===")
    print(df_sorted.tail(10).to_string(index=False))
    log(f"\nfull grid ({len(df)} cells) saved -> {out_dir / 'horizon_gap_gridscreen.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
