#!/usr/bin/env python3
"""SL-race diagnostic for liquidity_sweep top/down metalabel (final config: H=30/GAP=12/K=4.0) --
Grinold's Law style check ("for fires the label calls a WIN, at what SL width would a naive fixed
stop have been hit BEFORE the favorable K*ATR target was reached") that every other Homer signal's
trailing-stop cost-gate design started from (docs/homer/README.md, taker_delta_z_climax/
short_term_return_z/liquidity_sweep-V_REBOUND all ran this before grid-searching SL/ARM/Trail).
Independent per-signal measurement -- "자동승계 금지" (taker's SL-race curve does not transfer).

Optimistic vs pessimistic intrabar ordering (feedback_intrabar_ordering_optimistic_pessimistic_
bracket_20260830.md): within a single bar that both touches the adverse SL level AND the
favorable target, we don't know which happened first from OHLC alone. Optimistic = favorable
touch resolves first (best case for the trader); pessimistic = adverse SL resolves first (worst
case). Both are reported -- this project's own precedent found the narrower/pessimistic case is
the one that matters for a real go/no-go decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

FIRES_CSV = ROOT / "data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830/eth_5m_liquidity_sweep_topdown_metalabel_features_H30_GAP12_K4.0.csv"
KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
HORIZON = 30
TARGET_K = 4.0  # the label's own hit threshold -- SL-race checks whether SL fires before THIS target
SL_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]


def log(msg: str) -> None:
    print(f"[liq_sweep_slrace] {msg}", flush=True)


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def sl_race_for_winners(fires: pd.DataFrame, klines: pd.DataFrame, sl_mult: float, pessimistic: bool) -> float:
    """Fraction of hit==1 fires where the adverse move (SL_mult*atr_pct against entry) is touched
    at-or-before the favorable target (TARGET_K*atr_pct) within the HORIZON window. Bar-by-bar,
    checking each bar's adverse extreme against SL and favorable extreme against target in the
    same forward scan; ties within a single bar resolved by `pessimistic` (SL wins ties) vs
    optimistic (target wins ties)."""
    high = klines["high"].to_numpy(); low = klines["low"].to_numpy(); close = klines["close"].to_numpy()
    winners = fires[fires["hit"] == 1.0]
    stopped_first = 0
    total = 0
    for _, r in winners.iterrows():
        i = int(r["pos"]); side = r["side"]; atr = r["atr_pct"]
        entry = close[i]
        sl_level_dist = sl_mult * atr * entry
        target_dist = TARGET_K * atr * entry
        total += 1
        hit_sl = False
        hit_target = False
        for b in range(i + 1, min(i + HORIZON + 1, len(klines))):
            if side == "bottom":
                adverse = entry - low[b]     # long: adverse = price falling
                favorable = high[b] - entry  # long: favorable = price rising
            else:
                adverse = high[b] - entry    # short: adverse = price rising
                favorable = entry - low[b]   # short: favorable = price falling
            bar_hits_sl = adverse >= sl_level_dist
            bar_hits_target = favorable >= target_dist
            if bar_hits_sl and bar_hits_target:
                if pessimistic:
                    hit_sl = True
                else:
                    hit_target = True
                break
            if bar_hits_sl:
                hit_sl = True
                break
            if bar_hits_target:
                hit_target = True
                break
        if hit_sl and not hit_target:
            stopped_first += 1
    return stopped_first / total if total else float("nan")


def main() -> int:
    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"])
    klines = load_klines()
    log(f"{len(fires)} fires loaded, {int((fires['hit']==1.0).sum())} winners (hit=1) to SL-race test")

    log("\nSL width (xATR) -> fraction of WINNING fires stopped out before reaching the K=4.0xATR target:")
    for sl in SL_GRID:
        pess = sl_race_for_winners(fires, klines, sl, pessimistic=True)
        opt = sl_race_for_winners(fires, klines, sl, pessimistic=False)
        log(f"  SL={sl:>4.2f}x: pessimistic={pess:.3f}  optimistic={opt:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
