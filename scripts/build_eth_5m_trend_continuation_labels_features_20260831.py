#!/usr/bin/env python3
"""Build the trend-continuation EXTEND/REVERT pure-direction label + Tier0 23 features as a
single CSV for the TabPFN cheap_gate, matching the exact construction already visually verified
(render_eth_trend_continuation_extend_revert_label_20examples_20260831.py, user-approved chart)
and grid-screened (research_eth_trend_continuation_horizon_gap_gridscreen_20260831.py, GBM proxy
flat 0.4872-0.5327 across 24 H x GAP cells -- this is the confirmatory TabPFN check before fully
closing the axis, matching research_eth_breakout_continuation_tabpfn_cheap_gate_20260831.py's
"weak GBM/raw-lift doesn't prove TabPFN fails" precedent).

Label (unchanged from the visual-verification chart): union of the 8 live evidence-signal fires
(live_evidence_signal_dashboard_20260823.SIGNAL_ORDER), cluster-anchored GAP=12 bars, H=24 bars
(120min) forward. K = the median continuation/ATR ratio at this population (recomputed from
data, not hardcoded). EXTEND=1 if the continuation-direction excursion crosses K*ATR first,
REVERT=0 if the opposite-direction excursion does; events where neither/both cross are dropped.
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

from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

ETH_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_PATH = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_trend_continuation_20260831"
START = pd.Timestamp("2024-01-01")
H, GAP = 24, 12


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df[df["timestamp"] >= START - pd.Timedelta(days=10)].reset_index(drop=True)


def forward_extremes(close, high, low, h):
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

    names = [n for n, _ in SIGNAL_ORDER]
    bot = np.zeros(len(sig), bool); top = np.zeros(len(sig), bool)
    for n in names:
        if f"bottom_{n}" in sig: bot |= sig[f"bottom_{n}"].to_numpy()
        if f"top_{n}" in sig:    top |= sig[f"top_{n}"].to_numpy()

    rows = []
    for side, m in (("bottom", bot), ("top", top)):
        last = -10**9
        for i in np.flatnonzero(m):
            if i - last < GAP:
                continue
            last = i
            rows.append((i, side == "bottom"))
    ev = pd.DataFrame(rows, columns=["i", "is_bottom"]).sort_values("i").reset_index(drop=True)

    up, dn = forward_extremes(close, high, low, H)
    iu = ev["i"].to_numpy(); isb = ev["is_bottom"].to_numpy()
    cont = np.where(isb, dn[iu], up[iu]); rev = np.where(isb, up[iu], dn[iu])
    ok = ~np.isnan(cont) & ~np.isnan(rev) & (atr_pct[iu] > 0)
    k50 = float(np.median(cont[ok] / atr_pct[iu][ok]))
    y_ext = ok & (cont >= k50 * atr_pct[iu]); y_rev = ok & (rev >= k50 * atr_pct[iu])
    pure = y_ext ^ y_rev

    feat_cols = [c for c in FEATURE_COLUMNS if c != "is_bottom"]
    out = feats.iloc[iu[pure]][["timestamp"] + feat_cols].reset_index(drop=True)
    out["is_bottom"] = isb[pure].astype(int)
    out["side"] = np.where(isb[pure], "bottom", "top")
    out["outcome"] = np.where(y_ext[pure], "EXTEND", "REVERT")
    out["label"] = y_ext[pure].astype(int)
    out["k50"] = k50

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "eth_5m_trend_continuation_features_tier0.csv"
    out.to_csv(out_path, index=False)
    print(f"H={H} GAP={GAP} K={k50:.3f}")
    print(f"n={len(out)}  EXTEND={out['label'].sum()} ({out['label'].mean()*100:.1f}%)  "
          f"REVERT={(1-out['label']).sum()}")
    print(out.groupby(["side", "outcome"]).size().to_string())
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
