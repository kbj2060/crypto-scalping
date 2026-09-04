#!/usr/bin/env python3
"""Phase1 label-design diagnostics for a NEW Homer head: "추세 지속"(trend continuation).

Motivation (2026-08-31, user): all 8 evidence signals in SIGNAL_ORDER are extreme/exhaustion
detectors, so a sustained trend fires them all on the fade side -- verified live on 2026-08-31
KST 08:15-08:50 (ETH 2461->2401, -2.4%): 21 bottom fires vs 2 top fires, while the same 24h
window was a balanced 61 bottom / 62 top. The reversal meta-label answers "will fading this
work"; nothing in this repo answers "will this move EXTEND". A first pass over data/eth_5m_1year
.csv found the extend outcome is not just present but the LARGER class at bottom fires
(extend-only 37.7% vs revert-only 35.3%, H=24/K=2.0 touch), and it is entirely unmodeled.

Why this is NOT the closed direction axis (registry: eth_weekly_tsmom_bias_cheap_gate_20260817,
eth_direction_timescale_resample_screen_20260817, eth_moderntcn_nhits_direction_backbones_
20260818, btc_1h_native_swing_entry, eth_tier123_combined_direction_model_20260824): every one of
those predicted direction UNCONDITIONALLY at every bar. Here the event supplies the direction
(a bottom fire means the recent move was down; the continuation trade is short) and the model
only answers a binary "does it extend K*ATR further" on ~5% of bars that were pre-selected by an
already-built trigger. Same population where DeMarker's reversal head reached HOLDOUT AUC 0.7464,
i.e. these bars are demonstrably separable.

Follows docs/homer/README.md "재사용 방법론 템플릿" §2 checklist (items 1-4; item 5/persistence
deliberately skipped per the template's own v5/V_REBOUND warning; item 6 = 20-example visual
verification is a separate follow-up script requiring user sign-off before phase 2).

Adds two checks that are specific to this head and are its real go/no-go:
  A) raw lift of the continuation outcome at fire bars vs all bars -- expected near 1.0x, since
     unlike the reversal rules the trigger makes no unconditional continuation claim; reported
     honestly rather than framed away.
  B) revert/extend joint table + a fast GBM proxy trained on BOTH labels off the same Tier0 bank,
     to test whether p_extend is merely 1 - p_revert (if so this head adds nothing new and should
     be dropped before phase 2).

Source: binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv (2023-12-31..2026-08-28, gap-free), the
template's canonical klines-only label-design source. BTC klines from the sibling BTCUSDT file so
smt_divergence participates in the union (it cannot fire without them).

Splits (template §6 / CLAUDE.md Fresh-Forward): TRAIN <2025-09-01 / VAL 2025-09..12 /
OOS 2026-01..03. HOLDOUT (2026-04..08) is NOT touched by this script.
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
HOLDOUT_START = pd.Timestamp("2026-04-01")   # never read here

HORIZONS = [3, 6, 12, 24, 48]                # 15m / 30m / 1h / 2h / 4h (template item 1)
LAG_WINDOW = 24                              # +-2h, template item 3's standard window
DRAFT_H, DRAFT_K, DRAFT_GAP = 24, 2.0, 12    # only for the checks that need one fixed cell
NAMES = [n for n, _ in SIGNAL_ORDER]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df[df["timestamp"] >= START - pd.Timedelta(days=10)].reset_index(drop=True)


def forward_extremes(close: np.ndarray, high: np.ndarray, low: np.ndarray, h: int):
    """MFE up / MFE down over bars [i+1, i+h], intrabar high/low (the template's touch basis)."""
    fh = pd.Series(high).shift(-1).rolling(h, min_periods=h).max().shift(-(h - 1)).to_numpy()
    fl = pd.Series(low).shift(-1).rolling(h, min_periods=h).min().shift(-(h - 1)).to_numpy()
    return (fh - close) / close, (close - fl) / close


def main() -> int:
    eth, btc = load(ETH_PATH), load(BTC_PATH)
    sig = compute_signals(eth, btc, None)   # funding omitted -> orthogonal_combo bottom leg
    sig = sig[sig["timestamp"] >= START].reset_index(drop=True)  # degrades to its pre-08-27 form
    feats = build_indicator_frame(eth)
    feats = feats[feats["timestamp"] >= START].reset_index(drop=True)
    assert len(feats) == len(sig) and (feats["timestamp"].to_numpy() == sig["timestamp"].to_numpy()).all()

    ts = sig["timestamp"]
    close = sig["close"].to_numpy(); high = sig["high"].to_numpy(); low = sig["low"].to_numpy()
    atr_pct = feats["atr_pct"].to_numpy()

    bot = np.zeros(len(sig), bool); top = np.zeros(len(sig), bool)
    for n in NAMES:
        if f"bottom_{n}" in sig: bot |= sig[f"bottom_{n}"].to_numpy()
        if f"top_{n}" in sig:    top |= sig[f"top_{n}"].to_numpy()
    both_side = bot & top
    print(f"bars={len(sig)}  {ts.iloc[0].date()}..{ts.iloc[-1].date()}")
    print(f"union bottom fires={bot.sum()} ({100*bot.mean():.1f}%)  top fires={top.sum()} "
          f"({100*top.mean():.1f}%)  both-side same bar={both_side.sum()}\n")

    # ---- item 1: horizon sensitivity of the CONTINUATION hit rate + K that balances 50/50 ----
    print("=== [1] CONTINUATION hit-rate vs horizon (K=2.0 fixed), and the K giving ~50/50 ===")
    print(f"{'H(bars)':>8}{'min':>6}{'side':>8}{'n':>8}{'cont_hit%@K2.0':>16}{'K@50/50':>10}"
          f"{'revert_hit%@K2.0':>18}")
    for h in HORIZONS:
        up, dn = forward_extremes(close, high, low, h)
        for side, m in (("bottom", bot), ("top", top)):
            cont = dn if side == "bottom" else up      # continuation = with the move
            rev = up if side == "bottom" else dn
            v = m & ~np.isnan(cont) & ~np.isnan(atr_pct) & (atr_pct > 0)
            ratio = cont[v] / atr_pct[v]
            k50 = float(np.median(ratio))
            print(f"{h:>8}{h*5:>6}{side:>8}{v.sum():>8}"
                  f"{100*(cont[v] >= 2.0*atr_pct[v]).mean():>16.1f}{k50:>10.2f}"
                  f"{100*(rev[v] >= 2.0*atr_pct[v]).mean():>18.1f}")

    # ---- item 2: magnitude distribution of the continuation excursion (in ATR units) ----
    up, dn = forward_extremes(close, high, low, DRAFT_H)
    print(f"\n=== [2] continuation MFE / ATR distribution at H={DRAFT_H} ===")
    print(f"{'side':>8}{'p10':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p90':>8}")
    for side, m in (("bottom", bot), ("top", top)):
        cont = dn if side == "bottom" else up
        v = m & ~np.isnan(cont) & (atr_pct > 0)
        r = cont[v] / atr_pct[v]
        print(f"{side:>8}" + "".join(f"{np.percentile(r, q):>8.2f}" for q in (10, 25, 50, 75, 90)))

    # ---- item 3: fire bar vs the TREND-direction extreme inside +-2h ----
    print(f"\n=== [3] fire bar vs trend-direction extreme in a +-{LAG_WINDOW}-bar window ===")
    for side, m in (("bottom", bot), ("top", top)):
        idx = np.flatnonzero(m)
        idx = idx[(idx >= LAG_WINDOW) & (idx < len(sig) - LAG_WINDOW)]
        offs = np.empty(len(idx), int)
        for j, i in enumerate(idx):
            w = slice(i - LAG_WINDOW, i + LAG_WINDOW + 1)
            offs[j] = (int(np.argmin(low[w])) if side == "bottom"
                       else int(np.argmax(high[w]))) - LAG_WINDOW
        print(f"  {side}: n={len(idx)}  median offset={np.median(offs):+.0f} bars "
              f"({np.median(offs)*5:+.0f} min)  BEFORE={100*(offs<0).mean():.1f}%  "
              f"AT={100*(offs==0).mean():.1f}%  AFTER={100*(offs>0).mean():.1f}%")

    # ---- item 4: consecutive-fire clustering ----
    print("\n=== [4] consecutive same-side fire gap distribution ===")
    for side, m in (("bottom", bot), ("top", top)):
        g = np.diff(np.flatnonzero(m))
        print(f"  {side}: n_gaps={len(g)}  <=3bars={100*(g<=3).mean():.1f}%  "
              f"<=6={100*(g<=6).mean():.1f}%  <=12={100*(g<=12).mean():.1f}%  median={np.median(g):.0f}")

    # ---- (A) raw lift vs all bars ----
    print("\n=== [A] raw lift of the continuation outcome: fire bars vs ALL bars (K=2.0) ===")
    print(f"{'H':>4}{'side':>8}{'P(cont|fire)':>15}{'P(cont|any bar)':>18}{'lift':>8}")
    for h in HORIZONS:
        up_h, dn_h = forward_extremes(close, high, low, h)
        for side, m in (("bottom", bot), ("top", top)):
            cont = dn_h if side == "bottom" else up_h
            base_v = ~np.isnan(cont) & (atr_pct > 0)
            base = (cont[base_v] >= 2.0*atr_pct[base_v]).mean()
            v = m & base_v
            pf = (cont[v] >= 2.0*atr_pct[v]).mean()
            print(f"{h:>4}{side:>8}{100*pf:>14.1f}%{100*base:>17.1f}%{pf/base:>8.2f}x")

    # ---- (B) revert/extend joint + GBM proxy on both labels ----
    print(f"\n=== [B] revert vs extend joint table (H={DRAFT_H}, K={DRAFT_K}) ===")
    print(f"{'side':>8}{'n':>8}{'revert only%':>14}{'extend only%':>14}{'both%':>8}{'neither%':>10}")
    for side, m in (("bottom", bot), ("top", top)):
        cont = dn if side == "bottom" else up
        rev = up if side == "bottom" else dn
        v = m & ~np.isnan(cont) & (atr_pct > 0)
        r, e = rev[v] >= DRAFT_K*atr_pct[v], cont[v] >= DRAFT_K*atr_pct[v]
        print(f"{side:>8}{v.sum():>8}{100*(r&~e).mean():>14.1f}{100*(e&~r).mean():>14.1f}"
              f"{100*(r&e).mean():>8.1f}{100*(~r&~e).mean():>10.1f}")

    # cluster-anchored, side-tagged event table for the GBM proxy -- swept across horizons,
    # because [A] shows the continuation lift lives at 15-30min and is gone by 1h. K is
    # recalibrated per horizon to the ~50/50 point (template §3) so AUCs stay comparable.
    rows = []
    for side, m in (("bottom", bot), ("top", top)):
        last = -10**9
        for i in np.flatnonzero(m):
            if i - last < DRAFT_GAP:
                continue
            last = i
            rows.append((i, side == "bottom"))
    ev = pd.DataFrame(rows, columns=["i", "is_bottom"]).sort_values("i").reset_index(drop=True)
    iu = ev["i"].to_numpy(); isb = ev["is_bottom"].to_numpy()
    Xall = feats.iloc[iu][[c for c in FEATURE_COLUMNS if c != "is_bottom"]].copy()
    Xall["is_bottom"] = isb.astype(int)
    tall = ts.iloc[iu].to_numpy()

    print(f"\n=== [B] fast GBM proxy (screening only, NOT TabPFN) across horizons, "
          f"Tier0 23 features, cluster-anchored GAP={DRAFT_GAP} ===")
    print(f"{'H':>4}{'min':>5}{'K50':>6}{'n':>7}{'EXT VAL':>9}{'EXT OOS':>9}{'REV VAL':>9}"
          f"{'REV OOS':>9}{'corr':>7}{'REVasEXT VAL':>14}{'REVasEXT OOS':>14}")
    for h in HORIZONS:
        up_h, dn_h = forward_extremes(close, high, low, h)
        cont = np.where(isb, dn_h[iu], up_h[iu]); rev = np.where(isb, up_h[iu], dn_h[iu])
        ok = ~np.isnan(cont) & ~np.isnan(rev) & (atr_pct[iu] > 0)
        k50 = float(np.median(cont[ok] / atr_pct[iu][ok]))
        y_ext = (cont >= k50 * atr_pct[iu]).astype(int)
        y_rev = (rev >= k50 * atr_pct[iu]).astype(int)
        tr = ok & (tall < VAL_START)
        va = ok & (tall >= VAL_START) & (tall < OOS_START)
        oo = ok & (tall >= OOS_START) & (tall < HOLDOUT_START)
        out = {}
        for tag, y in (("EXT", y_ext), ("REV", y_rev)):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                 max_leaf_nodes=15, l2_regularization=1.0,
                                                 early_stopping=True, random_state=0)
            clf.fit(Xall[tr], y[tr])
            out[tag] = (clf.predict_proba(Xall[va])[:, 1], clf.predict_proba(Xall[oo])[:, 1])
        ev_v, ev_o = out["EXT"]; rv_v, rv_o = out["REV"]
        print(f"{h:>4}{h*5:>5}{k50:>6.2f}{int(ok.sum()):>7}"
              f"{roc_auc_score(y_ext[va], ev_v):>9.4f}{roc_auc_score(y_ext[oo], ev_o):>9.4f}"
              f"{roc_auc_score(y_rev[va], rv_v):>9.4f}{roc_auc_score(y_rev[oo], rv_o):>9.4f}"
              f"{np.corrcoef(ev_o, rv_o)[0,1]:>+7.2f}"
              f"{roc_auc_score(y_ext[va], rv_v):>14.4f}{roc_auc_score(y_ext[oo], rv_o):>14.4f}")
    print("  REVasEXT = the EXISTING reversal head's proba used to predict EXTEND. If it matches "
          "the EXT column,\n  a dedicated continuation head adds nothing beyond 'a big move is "
          "coming' (volatility), and should be dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
