"""Paired replay gate for MicroExec v1.5 maker-placement (docs/micro_scalp_1m_design_20260718.md).

Arms, evaluated on the same dense intent grid (every valid minute x both sides) as the v1
timing replay:
  baseline : taker at open of the intent minute D -> all-in cost = taker fee (4.5 bps), price 0.
  naive    : join top-of-book immediately, reprice every minute, taker at deadline. No signal.
  adaptive : v1.5 choose_placement() — contrarian composite modulates patience (join / rest
             deeper by deep_frac*15m-range / cross on urgent+momentum), veto minutes suspend
             the order, taker at deadline.

Conservative fill rule (the anti-optimism lesson from the invalidated 1m line): a resting buy
at price L fills in minute m only if low[m] < L strictly (trade-through; touching never
fills), and always fills AT L. The join line is open*(1 - 0.08bps) ≈ best bid minus one tick
(book is ~1 tick wide: spread median 0.053 bps in orderbook_decision_snapshots), so this
under-counts real join-bid fills. Unfilled intents pay the full chase: taker at open[D+K].

all_in_cost_bps = side*(P_fill - open[D])/open[D]*1e4 + fee.  improvement = 4.5 - cost.
Significance: daily-block t. Causality identical to the v1 replay (score usable from ts+2min;
mom3/range15 built from bars closed by the decision minute).
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from trading_bot_modules.micro_exec_overlay import (  # noqa: E402
    PlacementConfig, prepare_overlay_frame,
)
import analyze_microstructure_edge_20260718 as base  # noqa: E402
from replay_micro_exec_overlay_20260718 import daily_t  # noqa: E402

OUT_JSON = os.path.join(_ROOT, "data", "ensemble", "reports",
                        "micro_exec_placement_replay_20260718.json")
DEADLINES = [5, 10, 15]


def simulate_arm(*, idx, k, side, open_np, low_np, high_np, off_bps, taker_now, suspend):
    """Vectorized minute-by-minute working-order sim over intents idx with deadline k.

    off_bps[m]: per-minute limit offset (NaN = no order that minute). taker_now[m]: cross the
    spread immediately this minute. suspend[m]: order inactive (veto). Returns (cost_bps,
    fill_j, was_maker) aligned to idx.
    """
    n = len(idx)
    filled = np.zeros(n, dtype=bool)
    fill_px = np.full(n, np.nan)
    fill_j = np.full(n, k, dtype=np.int64)
    was_maker = np.zeros(n, dtype=bool)
    for j in range(k):
        m = idx + j
        act = ~filled & ~suspend[m]
        if not act.any():
            continue
        tk = act & taker_now[m] & np.isfinite(open_np[m])
        filled[tk] = True
        fill_px[tk] = open_np[idx[tk] + j]
        fill_j[tk] = j
        lim = act & ~tk & np.isfinite(off_bps[m]) & np.isfinite(open_np[m])
        L = open_np[m] * (1.0 - side * off_bps[m] / 1e4)
        thru = np.where(side > 0, low_np[m] < L, high_np[m] > L)
        hit = lim & np.nan_to_num(thru, nan=False)
        filled[hit] = True
        fill_px[hit] = L[hit]
        fill_j[hit] = j
        was_maker[hit] = True
    forced = ~filled
    fill_px[forced] = open_np[idx[forced] + k]
    base_px = open_np[idx]
    price_bps = side * (fill_px - base_px) / base_px * 1e4
    fee = np.where(was_maker, PlacementConfig.maker_fee_bps, PlacementConfig.taker_fee_bps)
    return price_bps + fee, fill_j, was_maker


def main() -> None:
    pcfg = PlacementConfig()
    micro = base.load_micro()
    ov = prepare_overlay_frame(micro)
    kl = pd.read_csv(base.KLINES, parse_dates=["timestamp"],
                     usecols=["timestamp", "open", "high", "low"]).set_index("timestamp").sort_index()
    t0 = max(ov.index.min(), kl.index.min())
    t1 = min(ov.index.max(), kl.index.max())
    grid = pd.date_range(t0, t1, freq="1min")
    open_s = kl["open"].reindex(grid)
    open_np = open_s.to_numpy()
    low_np = kl["low"].reindex(grid).to_numpy()
    high_np = kl["high"].reindex(grid).to_numpy()
    score = ov["score"].reindex(grid).to_numpy()
    veto = ov["veto"].reindex(grid).astype(object).fillna(False).to_numpy().astype(bool)
    valid_px = np.isfinite(open_np)

    # Causal state series: mom3 uses opens (observable at m); range15 uses bars closed by m.
    mom3 = (open_s / open_s.shift(3) - 1.0).to_numpy() * 1e4
    hi15 = kl["high"].rolling(15, min_periods=8).max().reindex(grid).shift(1).to_numpy()
    lo15 = kl["low"].rolling(15, min_periods=8).min().reindex(grid).shift(1).to_numpy()
    range15 = (hi15 - lo15) / open_np * 1e4
    print(f"grid {len(grid):,} min  {t0} -> {t1}  score_cov={np.isfinite(score).mean():.1%}")

    results = []
    max_k = max(DEADLINES)
    for side_name, side in [("long", 1.0), ("short", -1.0)]:
        s_side = side * np.nan_to_num(score, nan=0.0)
        arms = {}
        no_suspend = np.zeros(len(grid), dtype=bool)
        never_taker = np.zeros(len(grid), dtype=bool)
        join_off = np.where(valid_px, pcfg.join_offset_bps, np.nan)
        arms["naive_join"] = (join_off, never_taker, no_suspend)
        urgent = s_side >= pcfg.urgent_z
        patient = s_side <= pcfg.patient_z
        adapt_off = pcfg.join_offset_bps + np.where(
            patient, np.nan_to_num(pcfg.deep_frac * range15, nan=0.0), 0.0)
        adapt_off = np.where(valid_px, adapt_off, np.nan)
        adapt_taker = urgent & (side * np.nan_to_num(mom3, nan=0.0) >= pcfg.momentum_taker_bps)
        arms["adaptive"] = (adapt_off, adapt_taker, veto)
        arms["adaptive_join_only"] = (join_off, adapt_taker, veto)

        for k in DEADLINES:
            n = len(grid) - k
            ok = valid_px[:n] & np.isfinite(score[:n]) & valid_px[k:k + n]
            idx = np.nonzero(ok)[0]
            for arm, (off, tknow, susp) in arms.items():
                cost, fj, mk = simulate_arm(idx=idx, k=k, side=side, open_np=open_np,
                                            low_np=low_np, high_np=high_np, off_bps=off,
                                            taker_now=tknow, suspend=susp)
                imp = pcfg.taker_fee_bps - cost
                mean_d, t_d, ndays = daily_t(pd.Series(imp, index=grid[idx]))
                row = {"side": side_name, "arm": arm, "deadline_min": k,
                       "n_intents": int(len(idx)),
                       "improve_mean_bps": round(float(imp.mean()), 3),
                       "improve_daily_t": round(t_d, 2), "n_days": ndays,
                       "maker_fill_rate": round(float(mk.mean()), 3),
                       "forced_deadline_rate": round(float((fj == k).mean()), 3),
                       "mean_fill_delay_min": round(float(fj.mean()), 2),
                       "p05_improve_bps": round(float(np.percentile(imp, 5)), 2),
                       "p95_improve_bps": round(float(np.percentile(imp, 95)), 2)}
                results.append(row)
                print(f"{side_name:5s} {arm:18s} K={k:<2} n={len(idx):,} "
                      f"improve={row['improve_mean_bps']:+.2f}bps t={t_d:+.2f} "
                      f"maker_fill={row['maker_fill_rate']:.0%} "
                      f"forced={row['forced_deadline_rate']:.0%} delay={row['mean_fill_delay_min']:.1f}m")

    report = {"generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
              "window_utc": [str(t0), str(t1)],
              "placement_config": {kk: getattr(pcfg, kk) for kk in
                                   ("maker_fee_bps", "taker_fee_bps", "urgent_z", "patient_z",
                                    "momentum_taker_bps", "deep_frac", "join_offset_bps")},
              "fill_rule": "strict trade-through of open*(1 -/+ off/1e4); fill at limit; "
                           "touch never fills; unfilled pays taker at open[D+K]",
              "baseline": "taker at open[D], 4.5bps fee",
              "fresh_forward_bar_by_bar": True,
              "trade_ledgers_used_as_input": False,
              "future_rows_used_for_entry": False,
              "results": results}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
