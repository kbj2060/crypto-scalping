#!/usr/bin/env python3
"""Decisive follow-up to research_eth_trend_continuation_regime_conditional_20260831.py's Part C,
which found the continuation trade is robust at a physically-floored trail (ARM0.5/Trail0.2,
~5.2bp) ONLY in the "bear" GBM3 regime bucket (VAL +2.18bp/OOS +9.13bp), weak-to-negative in
"bull" (VAL -1.27/OOS +1.06), and dead in "chop" (both negative).

Before treating "bear regime" as a genuine conditioning edge, this must be split by which side is
actually driving it. A bear-regime bucket contains BOTH:
  - bottom fires traded SHORT (drift-aligned: the regime model calls it "bear" partly BECAUSE
    price has been falling, so shorting more of an already-falling market is close to tautological
    momentum, not new information)
  - top fires traded LONG (a genuine counter-regime bet: betting a bounce continues even though
    the model currently says "bear")
If only the short leg is positive, "bear regime helps continuation" is barely more than restating
the quarterly finding in eth_trend_continuation_at_evidence_signal_fires_20260831.md §3.2 (down
quarters averaged +2.53bp) at finer (per-bar GBM3 state) granularity, which is worth reporting
honestly as "not new" rather than as a fresh discovered edge. If BOTH legs are positive, that is
a materially stronger, non-circular finding.

Also required before any promotion-flavored claim: monthly disaggregation of the bear-only cell
(this repo's standing trap -- eth_bottom_flush_fade_strategy_v1_20260824 died on "all profit in
one month") and a cost-stress sweep on that specific cell.

Reuses build_regime_frame()/build_indicator_frame() etc. verbatim from the parent script.
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

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame  # noqa: E402
from research_eth_trend_continuation_regime_conditional_20260831 import (  # noqa: E402
    ARM, COST, FLOOR_ARM, FLOOR_TRAIL, GAP, H, HOLDOUT_START, LEV, MARGIN, OOS_START, SL, START,
    TRAIL, VAL_START, build_regime_frame, load_kl,
)

RANDOM_SEED = 20260831


def main() -> int:
    regime = build_regime_frame()
    eth, btc = load_kl("ETHUSDT"), load_kl("BTCUSDT")
    sig = compute_signals(eth, btc, None)
    sig = sig.loc[sig["timestamp"] >= START].reset_index(drop=True)
    kl = eth.loc[eth["timestamp"] >= START].reset_index(drop=True)
    ind = build_indicator_frame(eth)
    ind = ind.loc[ind["timestamp"] >= START].reset_index(drop=True)
    merged = kl[["timestamp"]].merge(regime, on="timestamp", how="left")
    regime_label = merged["regime_label"].to_numpy()

    names = [n for n, _ in SIGNAL_ORDER]
    bot = np.zeros(len(sig), bool); top = np.zeros(len(sig), bool)
    for n in names:
        bot |= sig[f"bottom_{n}"].to_numpy(); top |= sig[f"top_{n}"].to_numpy()
    rows = []
    for side, m in (("bottom", bot), ("top", top)):
        last = -10**9
        for i in np.flatnonzero(m):
            if i - last < GAP:
                continue
            last = i; rows.append((i, side))
    ev = pd.DataFrame(rows, columns=["pos", "side"]).sort_values("pos").reset_index(drop=True)

    ts = kl["timestamp"]; o = kl["open"].to_numpy(); hi = kl["high"].to_numpy()
    lo = kl["low"].to_numpy(); c = kl["close"].to_numpy()
    atr_pct = ind["atr_pct"].to_numpy()

    def econ(dec, sc, arm, trail, w):
        s, e = (VAL_START, OOS_START) if w == "val" else (OOS_START, HOLDOUT_START)
        el = set(np.flatnonzero(purged_decision_mask(ts, start=s, end=e, horizon_bars=H)).tolist())
        m = np.array([d in el for d in dec])
        if m.sum() == 0:
            return None
        a = atr_pct[dec][m]
        r = simulate_single_position(timestamps=ts, open_px=o, high=hi, low=lo, close=c,
            decision_indices=dec[m], scores=sc[m], tp_moves=np.full(int(m.sum()), 999.0),
            sl_moves=SL * a, upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=H,
            margin_fraction=MARGIN, leverage=LEV, roundtrip_cost_rate=COST,
            arm_moves=arm * a, trail_moves=trail * a)
        return r.ledger

    print("=== [1] side x regime economics, FLOORED cell (ARM=0.5/Trail=0.2, ~5.2bp) ===")
    print(f"{'fire_side':<10}{'traded_as':<10}{'regime':<8}{'split':>6}{'n':>6}{'bp':>9}{'win%':>7}")
    for fire_side, traded_as, sc_val in (("bottom", "SHORT", -1.0), ("top", "LONG", 1.0)):
        side_mask = (ev["side"] == fire_side).to_numpy()
        for rname in ("bull", "bear", "chop"):
            rmask = regime_label == rname
            keep = side_mask & rmask[ev["pos"].to_numpy()]
            d = ev["pos"].to_numpy()[keep]
            sc = np.full(len(d), sc_val)
            for w in ("val", "oos"):
                L = econ(d, sc, FLOOR_ARM, FLOOR_TRAIL, w)
                if L is None or len(L) < 20:
                    print(f"{fire_side:<10}{traded_as:<10}{rname:<8}{w:>6}{0 if L is None else len(L):>6}   too few")
                    continue
                print(f"{fire_side:<10}{traded_as:<10}{rname:<8}{w:>6}{len(L):>6}"
                      f"{L['trade_return'].mean()*1e4:>+9.2f}{(L['price_move']>0).mean():>7.1%}")

    print("\n=== [2] monthly disaggregation -- BEAR regime only, FLOORED cell, both sides pooled ===")
    bear_mask = regime_label == "bear"
    keep = bear_mask[ev["pos"].to_numpy()]
    d = ev["pos"].to_numpy()[keep]
    sc = np.where(ev["side"].to_numpy()[keep] == "bottom", -1.0, 1.0)
    ledgers = []
    for w in ("val", "oos"):
        L = econ(d, sc, FLOOR_ARM, FLOOR_TRAIL, w)
        if L is not None:
            ledgers.append(L)
    led = pd.concat(ledgers, ignore_index=True)
    mtab = led.assign(m=led["entry_timestamp"].dt.to_period("M")).groupby("m")["trade_return"].agg(["size", "mean"])
    mtab["bp"] = mtab["mean"] * 1e4
    print(mtab[["size", "bp"]].round(2).to_string())
    print(f"months positive: {(mtab['bp'] > 0).sum()}/{len(mtab)}   total n={len(led)}   "
          f"overall mean={led['trade_return'].mean()*1e4:+.2f}bp")

    print("\n=== [3] cost stress -- BEAR regime, FLOORED cell (price_move re-priced at each cost) ===")
    notional = MARGIN * LEV
    print(f"{'split':>6}{'n':>6}" + "".join(f"{f'{x}bp':>9}" for x in (10, 13, 16, 20)))
    for w in ("val", "oos"):
        L = econ(d, sc, FLOOR_ARM, FLOOR_TRAIL, w)
        if L is None:
            continue
        line = f"{w:>6}{len(L):>6}"
        for cost in (0.0010, 0.0013, 0.0016, 0.0020):
            bp = (L["price_move"].to_numpy() * notional - cost * notional).mean() * 1e4
            line += f"{bp:>9.2f}"
        print(line)

    print("\n=== [4] regime label stability at fire bars -- was this actually 'bear' 1h earlier "
          "(causal plausibility, not just the fire-bar snapshot)? ===")
    pos_of = pd.Series(np.arange(len(kl)), index=ts.to_numpy())
    d_all = ev["pos"].to_numpy()
    lookback_1h_label = np.full(len(d_all), None, dtype=object)
    valid = d_all >= 12
    lookback_1h_label[valid] = regime_label[d_all[valid] - 12]
    same = (lookback_1h_label == regime_label[d_all])
    print(f"  fire-bar regime label unchanged from 1h earlier: {100*np.nanmean(same.astype(float)):.1f}% "
          f"of {valid.sum()} fires with a valid lookback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
