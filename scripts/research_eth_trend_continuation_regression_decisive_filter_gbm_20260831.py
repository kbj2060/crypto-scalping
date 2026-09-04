#!/usr/bin/env python3
"""GBM regression cheap-check for trend-continuation, per user's new framing (2026-08-31):
instead of binary classification on EXTEND/REVERT, regress a continuous signed target and
restrict TRAINING to only "decisive" outcomes (drop the more-ambiguous/sideways half by outcome
magnitude), rather than the classification axis's XOR pure-direction filter.

Target: net = (cont - rev) / atr_pct, using the same H=24-bar continuation/reversal excursions
(ATR units) as the rest of this axis (research_eth_trend_continuation_head_phase1_20260831.py's
forward_extremes()). net>0 = continuation dominated, net<0 = reversal dominated, |net| near 0 =
the two directions moved similarly (choppy/ambiguous -- "sideways" in the user's framing).

Decisiveness filter (TRAIN only): keep events with |net| >= the population median |net| (top 50%
most decisive outcomes) -- directly implements "train only on clear up/down trend, give the
model zero sideways/ambiguous examples" without hand-picking an arbitrary cutoff.

VAL/OOS are scored BOTH on the same decisive-only subset (apples-to-apples with train) AND on
the FULL population including the "sideways" half (what the model actually faces at inference,
since you can't know net's sign/magnitude in advance to pre-filter live events) -- this project
has a direct precedent for why that second number is the one that matters
(orthogonal_combo's "kept-only" headline AUC was later found inflated vs its full-population re-
evaluation, docs/homer/README.md).
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
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
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
H, GAP = 24, 12
GBM_SEED = 20260831


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

    ts = sig["timestamp"]; tall = ts.to_numpy()
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
    iu = ev["i"].to_numpy(); isb = ev["is_bottom"].to_numpy()

    up, dn = forward_extremes(close, high, low, H)
    cont = np.where(isb, dn[iu], up[iu]); rev = np.where(isb, up[iu], dn[iu])
    ok = ~np.isnan(cont) & ~np.isnan(rev) & (atr_pct[iu] > 0)
    net = np.where(ok, (cont - rev) / np.where(atr_pct[iu] > 0, atr_pct[iu], np.nan), np.nan)

    med_abs = float(np.nanmedian(np.abs(net[ok])))
    decisive = ok & (np.abs(net) >= med_abs)
    print(f"population: total events={len(iu)}, ok(non-nan)={int(ok.sum())}, "
          f"median|net|={med_abs:.3f}, decisive(kept)={int(decisive.sum())} "
          f"({100*decisive.sum()/ok.sum():.1f}% of ok)")

    feat_cols = [c for c in FEATURE_COLUMNS if c != "is_bottom"]
    X = feats.iloc[iu][feat_cols].copy()
    X["is_bottom"] = isb.astype(int)
    t = tall[iu]

    tr = decisive & (t < VAL_START)
    va_dec = decisive & (t >= VAL_START) & (t < OOS_START)
    oo_dec = decisive & (t >= OOS_START) & (t < HOLDOUT_START)
    print(f"train (decisive only) n={int(tr.sum())}  val(decisive) n={int(va_dec.sum())}  "
          f"oos(decisive) n={int(oo_dec.sum())}")

    reg = HistGradientBoostingRegressor(random_state=GBM_SEED)
    reg.fit(X[tr], net[tr])

    def report(tag: str, mask: np.ndarray) -> None:
        pred = reg.predict(X[mask])
        actual = net[mask]
        corr, _ = spearmanr(pred, actual)
        sign_acc = float((np.sign(pred) == np.sign(actual)).mean())
        y_bin = (actual > 0).astype(int)
        auc = roc_auc_score(y_bin, pred) if len(np.unique(y_bin)) == 2 else float("nan")
        print(f"  {tag:32s} n={int(mask.sum()):5d}  spearman={corr:+.4f}  sign_acc={sign_acc:.4f}  "
              f"auc(sign-as-label,net_hat-as-score)={auc:.4f}")

    print("\n=== evaluated on DECISIVE-ONLY subset (same distribution as train) ===")
    report("VAL (decisive)", va_dec)
    report("OOS (decisive)", oo_dec)

    print("\n=== evaluated on FULL population incl. sideways (what inference actually faces) ===")
    va_full = ok & (t >= VAL_START) & (t < OOS_START)
    oo_full = ok & (t >= OOS_START) & (t < HOLDOUT_START)
    report("VAL (full, incl. sideways)", va_full)
    report("OOS (full, incl. sideways)", oo_full)

    print("\n=== for reference: classification AUC already measured on the SAME population "
          "(H=24/GAP=12, pure-direction XOR filter) ===")
    print("  GBM proxy: VAL=0.5188 OOS=0.5164 | TabPFN: VAL=0.5033 OOS=0.5263 (beats_naive=False)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
