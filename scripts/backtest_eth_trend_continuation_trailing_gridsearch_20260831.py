#!/usr/bin/env python3
"""ATR trailing-stop cost gate for the TREND-CONTINUATION trade at evidence-signal fire bars.

The mirror of every prior Homer economics gate: identical engine (core.causal_futures_backtest.
simulate_single_position), identical constants (MARGIN_FRACTION=0.30/LEVERAGE=3.0/
ROUNDTRIP_COST_RATE=0.001=10bp), identical SL x ARM x Trail grid (96 combos), identical
"trade every cluster-anchored candidate unconditionally" convention -- the ONLY change is that
`scores` is FLIPPED: a bottom fire is traded SHORT (with the move) instead of LONG (fading it).

Why this and not a model: research_eth_trend_continuation_head_phase1_20260831.py established
(a) the raw continuation tilt is real but lives only at 15-30min (raw lift 1.56x at H=3, 1.22x at
H=6, gone by H=12), and (b) a learned continuation head is dead -- conditional on exactly one of
{extend, revert} happening, Tier0's pure-direction AUC is 0.49-0.52 across 5 horizons (n~11k per
cell, VAL and OOS), i.e. the features carry zero directional information at these bars. So the
only remaining question is whether the UNCONDITIONAL raw tilt clears standard costs.

Population: union of the 8 SIGNAL_ORDER signals' bottom/top fires (what the user actually watches
-- during a trend the whole chip row lights up on one side), cluster-anchored at GAP=12.

RANDOM-ENTRY BASELINE included, per this lineage's own lesson (orthogonal_combo/fib_extension:
a trailing stop with a low ARM produces 82-93% win rates on random entries, so win rate alone
proves nothing -- only the bp margin over the random baseline counts).

VAL 2025-09..12 + OOS 2026-01..03 only. HOLDOUT (2026-04..08) NOT touched.
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

OUT_DIR = ROOT / "tmp/eth_trend_continuation_trailing_gridsearch_20260831"
START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.001
SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]
HORIZONS = [3, 6, 12, 24, 48]        # 15m..4h -- does the tilt flip sign at longer horizons?
CLUSTER_GAP = 12
RANDOM_SEED = 20260831


def log(m: str) -> None:
    print(f"[trend_continuation_gate] {m}", flush=True)


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"binance_data/klines/{name}/{name}-5m-api.csv", parse_dates=["timestamp"])
    return df.loc[df["timestamp"] >= START - pd.Timedelta(days=10)].reset_index(drop=True)


def run_grid(tag: str, ts_full, open_px, high, low, close, dec, scores, atr, horizon):
    eligible = {
        "val": set(np.flatnonzero(purged_decision_mask(ts_full, start=VAL_START, end=OOS_START,
                                                       horizon_bars=horizon)).tolist()),
        "oos": set(np.flatnonzero(purged_decision_mask(ts_full, start=OOS_START, end=HOLDOUT_START,
                                                       horizon_bars=horizon)).tolist()),
    }
    masks = {w: np.array([d in s for d in dec]) for w, s in eligible.items()}
    log(f"  [{tag}] eligible decisions: val={masks['val'].sum()} oos={masks['oos'].sum()}")
    tp_ph = np.full(len(dec), 999.0)
    rows = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for trail in TRAIL_GRID:
                row = {"sl": sl, "arm": arm, "trail": trail}
                ok = True
                for w, m in masks.items():
                    r = simulate_single_position(
                        timestamps=ts_full, open_px=open_px, high=high, low=low, close=close,
                        decision_indices=dec[m], scores=scores[m], tp_moves=tp_ph[m],
                        sl_moves=sl * atr[m], upper_threshold=1.0, lower_threshold=-1.0,
                        horizon_bars=horizon, margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
                        roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
                        arm_moves=arm * atr[m], trail_moves=trail * atr[m])
                    n = int(len(r.ledger))
                    bp = float(r.ledger["trade_return"].mean() * 1e4) if n else float("nan")
                    row[f"{w}_n"], row[f"{w}_avg_bp"] = n, round(bp, 3)
                    row[f"{w}_win_rate"] = round(float((r.ledger["price_move"] > 0).mean()), 4) if n else float("nan")
                    if not (n > 0 and bp > 0):
                        ok = False
                row["both_positive"] = ok
                rows.append(row)
    t = pd.DataFrame(rows)
    t["min_bp"] = t[["val_avg_bp", "oos_avg_bp"]].min(axis=1)
    return t.sort_values("min_bp", ascending=False).reset_index(drop=True)


def main() -> int:
    log("loading klines + signals...")
    eth, btc = load("ETHUSDT"), load("BTCUSDT")
    sig = compute_signals(eth, btc, None)
    keep = sig["timestamp"] >= START
    sig = sig.loc[keep].reset_index(drop=True)
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
            if i - last < CLUSTER_GAP:
                continue
            last = i
            rows.append((i, side))
    ev = pd.DataFrame(rows, columns=["pos", "side"]).sort_values("pos").reset_index(drop=True)
    log(f"cluster-anchored candidates (GAP={CLUSTER_GAP}): {len(ev)} "
        f"(bottom={(ev.side=='bottom').sum()}, top={(ev.side=='top').sum()})")

    ts_full = kl["timestamp"]
    open_px, high, low, close = (kl[c].to_numpy() for c in ("open", "high", "low", "close"))
    atr_pct = ind["atr_pct"].to_numpy()
    dec = ev["pos"].to_numpy(np.int64)
    # FLIPPED vs every prior gate: bottom fire -> SHORT (ride the down move), top fire -> LONG
    scores_cont = np.where(ev["side"].to_numpy() == "bottom", -1.0, 1.0)
    scores_fade = -scores_cont
    atr = atr_pct[dec]

    rng = np.random.default_rng(RANDOM_SEED)
    valid = np.flatnonzero(~np.isnan(atr_pct) & (atr_pct > 0))
    valid = valid[(valid > 900) & (valid < len(kl) - 60)]
    rnd_dec = np.sort(rng.choice(valid, size=len(dec), replace=False))
    rnd_scores = rng.choice([-1.0, 1.0], size=len(rnd_dec))
    rnd_atr = atr_pct[rnd_dec]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for h in HORIZONS:
        log(f"\n=== HORIZON = {h} bars ({h*5} min) ===")
        cont = run_grid(f"H{h} CONTINUATION", ts_full, open_px, high, low, close, dec, scores_cont, atr, h)
        rnd = run_grid(f"H{h} RANDOM", ts_full, open_px, high, low, close, rnd_dec, rnd_scores, rnd_atr, h)
        fade = run_grid(f"H{h} FADE(ref)", ts_full, open_px, high, low, close, dec, scores_fade, atr, h)
        for nm, t in (("continuation", cont), ("random", rnd), ("fade", fade)):
            t.to_csv(OUT_DIR / f"h{h}_{nm}.csv", index=False)
        log(f"  CONTINUATION: {int(cont.both_positive.sum())}/{len(cont)} combos VAL+OOS both-positive")
        log(f"  RANDOM     : {int(rnd.both_positive.sum())}/{len(rnd)} combos VAL+OOS both-positive")
        log(f"  FADE (ref) : {int(fade.both_positive.sum())}/{len(fade)} combos VAL+OOS both-positive")
        for nm, t in (("CONTINUATION", cont), ("RANDOM", rnd), ("FADE(ref)", fade)):
            r = t.iloc[0]
            log(f"    best {nm:<13} SL={r.sl} ARM={r.arm} Trail={r.trail}: "
                f"VAL={r.val_avg_bp:+.2f}bp(win={r.val_win_rate:.1%},n={int(r.val_n)}) "
                f"OOS={r.oos_avg_bp:+.2f}bp(win={r.oos_win_rate:.1%},n={int(r.oos_n)})")
        # same-cell comparison: continuation at the random baseline's best cell and vice versa
        rb = rnd.iloc[0]
        cell = cont[(cont.sl == rb.sl) & (cont.arm == rb.arm) & (cont.trail == rb.trail)].iloc[0]
        log(f"    continuation AT the random-best cell: VAL={cell.val_avg_bp:+.2f}bp "
            f"OOS={cell.oos_avg_bp:+.2f}bp  (random there: VAL={rb.val_avg_bp:+.2f} OOS={rb.oos_avg_bp:+.2f})")
    log(f"\ngrids -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
