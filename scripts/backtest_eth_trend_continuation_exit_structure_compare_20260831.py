#!/usr/bin/env python3
"""Exit-structure comparison for the trend-continuation trade -- prompted by the user's
2026-08-31 observation that the confirmed cell's holding times look too short (winners exit in
2-7 bars / 10-35 min) and that waiting could have captured more.

The confirmed cell (SL3.5/ARM0.5/Trail0.1) sits at the TIGHT edge of the original grid on both
knobs (Trail 0.1>0.2>0.3>0.5 monotonically, ARM 0.5 best), so "loosen the trail" is already
measured and rejected INSIDE the trailing family. What was never tested is leaving the winner
uncut at all. Three families over the identical population:

  A) trailing      -- the incumbent, plus a tighter extension of the boundary (Trail 0.05, ARM 0.25)
  B) fixed TP:SL   -- take profit at TP x ATR, stop at SL x ATR, no trailing
  C) time exit     -- stop only, no TP: hold the full horizon unless stopped ("just wait")

plus a hindsight diagnostic: how much favorable excursion was still available AFTER each trade's
actual exit, out to 2h/4h/8h -- the direct measurement of "money left on the table".

VAL 2025-09..12 + OOS 2026-01..03. HOLDOUT (2026-04..08) NOT touched.
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

START = pd.Timestamp("2024-01-01")
VAL_START, OOS_START, HOLDOUT_START = (pd.Timestamp(x) for x in ("2025-09-01", "2026-01-01", "2026-04-01"))
GAP = 12
MARGIN, LEV, COST = 0.30, 3.0, 0.001
OUT_DIR = ROOT / "tmp/eth_trend_continuation_exit_structure_20260831"


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"binance_data/klines/{name}/{name}-5m-api.csv", parse_dates=["timestamp"])
    return df.loc[df["timestamp"] >= START - pd.Timedelta(days=10)].reset_index(drop=True)


def main() -> int:
    eth, btc = load("ETHUSDT"), load("BTCUSDT")
    sig = compute_signals(eth, btc, None)
    sig = sig.loc[sig["timestamp"] >= START].reset_index(drop=True)
    kl = eth.loc[eth["timestamp"] >= START].reset_index(drop=True)
    ind = build_indicator_frame(eth)
    ind = ind.loc[ind["timestamp"] >= START].reset_index(drop=True)

    bot = np.zeros(len(sig), bool); top = np.zeros(len(sig), bool)
    for n, _ in SIGNAL_ORDER:
        bot |= sig[f"bottom_{n}"].to_numpy(); top |= sig[f"top_{n}"].to_numpy()
    rows = []
    for side, m in (("bottom", bot), ("top", top)):
        last = -10**9
        for i in np.flatnonzero(m):
            if i - last < GAP:
                continue
            last = i; rows.append((i, side))
    ev = pd.DataFrame(rows, columns=["pos", "side"]).sort_values("pos").reset_index(drop=True)

    ts = kl["timestamp"]
    o, hi, lo, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    atr_pct = ind["atr_pct"].to_numpy()
    dec = ev["pos"].to_numpy(np.int64)
    scores = np.where(ev["side"].to_numpy() == "bottom", -1.0, 1.0)

    def run(horizon, sl, *, tp=None, arm=None, trail=None):
        out, led_all = {}, []
        for wname, (s, e) in (("val", (VAL_START, OOS_START)), ("oos", (OOS_START, HOLDOUT_START))):
            el = set(np.flatnonzero(purged_decision_mask(ts, start=s, end=e, horizon_bars=horizon)).tolist())
            m = np.array([d in el for d in dec])
            a = atr_pct[dec][m]
            kw = {"arm_moves": arm * a, "trail_moves": trail * a} if arm is not None else {}
            r = simulate_single_position(
                timestamps=ts, open_px=o, high=hi, low=lo, close=c, decision_indices=dec[m],
                scores=scores[m], tp_moves=(tp * a if tp is not None else np.full(int(m.sum()), 999.0)),
                sl_moves=sl * a, upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=horizon,
                margin_fraction=MARGIN, leverage=LEV, roundtrip_cost_rate=COST, **kw)
            L = r.ledger
            out[wname] = (len(L), L["trade_return"].mean() * 1e4, (L["price_move"] > 0).mean(),
                          L["bars_held"].mean())
            led_all.append(L)
        return out, pd.concat(led_all, ignore_index=True)

    print(f"population: {len(ev)} cluster-anchored candidates (GAP={GAP})\n")
    hdr = (f"{'structure':<32}{'H':>4}{'VALn':>6}{'VAL bp':>9}{'VALwin':>8}{'VALbars':>9}"
           f"{'OOSn':>6}{'OOS bp':>9}{'OOSwin':>8}{'OOSbars':>9}")
    results = []

    def show(label, horizon, **kw):
        r, led = run(horizon, **kw)
        v, oo = r["val"], r["oos"]
        print(f"{label:<32}{horizon:>4}{v[0]:>6}{v[1]:>+9.2f}{v[2]:>8.1%}{v[3]:>9.1f}"
              f"{oo[0]:>6}{oo[1]:>+9.2f}{oo[2]:>8.1%}{oo[3]:>9.1f}")
        results.append({"structure": label, "horizon": horizon, "val_bp": v[1], "oos_bp": oo[1],
                        "val_bars": round(v[3], 1), "oos_bars": round(oo[3], 1),
                        "min_bp": min(v[1], oo[1])})
        return led

    print("=== A) TRAILING (incumbent + tighter boundary extension) ===")
    print(hdr)
    led_incumbent = show("trail SL3.5/ARM0.5/Tr0.1", 24, sl=3.5, arm=0.5, trail=0.1)
    for arm, tr in ((0.25, 0.1), (0.5, 0.05), (0.25, 0.05)):
        show(f"trail SL3.5/ARM{arm}/Tr{tr}", 24, sl=3.5, arm=arm, trail=tr)
    for h in (48, 96):
        show("trail SL3.5/ARM0.5/Tr0.1", h, sl=3.5, arm=0.5, trail=0.1)

    print("\n=== B) FIXED TP:SL (no trailing -- the winner is not cut early) ===")
    print(hdr)
    for h in (24, 48, 96):
        for tp in (1.0, 2.0, 3.0, 4.0):
            show(f"fixed TP{tp}/SL3.5", h, sl=3.5, tp=tp)

    print("\n=== C) TIME EXIT ONLY (no TP -- 'just wait' to the horizon) ===")
    print(hdr)
    for h in (24, 48, 96, 144):
        for sl in (3.5, 6.0):
            show(f"time-exit SL{sl}", h, sl=sl)

    tab = pd.DataFrame(results).sort_values("min_bp", ascending=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT_DIR / "exit_structure_comparison.csv", index=False)
    print("\n=== ranked by min(VAL,OOS) bp ===")
    print(tab.head(10).round(2).to_string(index=False))

    print("\n=== D) hindsight -- favorable excursion still available AFTER the actual exit ===")
    pos = pd.Series(np.arange(len(kl)), index=ts.to_numpy())
    L = led_incumbent
    xi = L["exit_timestamp"].map(pos).to_numpy()
    fi = L["decision_timestamp"].map(pos).to_numpy()
    short = L["score"].to_numpy() < 0
    a = atr_pct[fi]
    print(f"{'window after fire':<24}{'median extra (xATR)':>21}{'p75':>8}{'share>0.5xATR':>16}")
    for extra_h in (24, 48, 96):
        vals = []
        for k in range(len(L)):
            end = min(fi[k] + extra_h, len(kl) - 1)
            if xi[k] >= end:
                vals.append(0.0); continue
            seg = slice(xi[k] + 1, end + 1)
            px = lo[seg].min() if short[k] else hi[seg].max()
            exit_px = c[xi[k]]
            move = (exit_px - px) / exit_px if short[k] else (px - exit_px) / exit_px
            vals.append(max(move, 0.0) / a[k])
        v = np.array(vals)
        print(f"{f'fire+{extra_h} bars ({extra_h*5}min)':<24}{np.median(v):>21.2f}{np.percentile(v,75):>8.2f}"
              f"{(v>0.5).mean():>16.1%}")
    print("  (hindsight only -- an ex-ante rule that captures it must show up in A/B/C above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
