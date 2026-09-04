#!/usr/bin/env python3
"""Isolated single-variable A/B, 2026-08-26 user follow-up (2nd of two candle-color proposals --
see research_eth_liquidation_map_candle_color_direction_isolated_ab_20260826.py for the position-
ratio version): use the candle's OWN high/low, chosen by candle color, as its hypothetical entry
price instead of a fixed close or (high+low)/2. User-confirmed mapping (AskUserQuestion,
2026-08-26): up-candle (close>open) -> high, down-candle (close<open) -> low -- "the direction
that formed the candle had its actual execution concentrated at that extreme." Doji (close==open,
9 bars in the full window) falls back to close.

Reuses research_eth_liquidation_map_entry_price_isolated_ab_20260826.py's
compute_raw_bins_entry_price() / identity check / snapshot generator unmodified -- same variable
(entry-price basis) as that script's v1_mid, just a different (candle-color-conditioned) price
formula instead of a constant (high+low)/2, so this stays isolated from the direction-split axis
(v1_taker/v1_blend/v1_toptrader/v1_candle_color) tested separately.

Pre-registered gate: adopt only if OOS pairWR AND magnitude beat v1_live on resistance without
making support worse; otherwise REJECTED.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_v2_cohort_ab_backtest_20260825 as v2ab
import scripts.research_eth_liquidation_map_v2_phase0_data_audit_20260825 as audit
import scripts.research_eth_liquidation_map_v1_direction_isolated_ab_20260826 as v1dir
import scripts.research_eth_liquidation_map_entry_price_isolated_ab_20260826 as epdir

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_candle_color_price_isolated_ab_20260826.json"

VARIANTS = ("v1_live", "v1_color_price")


def main() -> None:
    px1h = v2ab.load_hourly_with_taker()
    m, clean = audit.load_metrics()
    df, join_stats = audit.hourly_join(m, px1h)
    print(f"join: {join_stats}", flush=True)
    df = df.rename(columns={"sum_open_interest": "oi"})

    epdir._identity_check(df)

    close = df["close"].to_numpy(dtype="float64")
    open_ = df["open"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    is_up = close > open_
    is_down = close < open_
    color_price = close.copy()
    color_price[is_up] = high[is_up]
    color_price[is_down] = low[is_down]
    print(f"candle color: up={int(is_up.sum())} down={int(is_down.sum())} "
          f"doji={int((~is_up & ~is_down).sum())} / {len(df)}", flush=True)
    diff_pct = np.abs(color_price - close) / close * 100.0
    print(f"|color_price-close|/close%%: mean={diff_pct.mean():.4f} median={np.median(diff_pct):.4f} "
          f"p95={np.percentile(diff_pct, 95):.4f} max={diff_pct.max():.4f}", flush=True)

    entry_price = {"v1_live": close, "v1_color_price": color_price}

    n = len(df)
    split_i = int(n * v1dir.TRAIN_FRACTION)
    eval_idxs = base.asof_indices(n, v1dir.WARMUP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    print(f"bars={n} split at {df['timestamp'].iloc[split_i]} eval_points={len(eval_idxs)} "
          f"(train={sum(1 for i in eval_idxs if i < split_i)}, oos={sum(1 for i in eval_idxs if i >= split_i)})",
          flush=True)

    all_snaps: dict[str, list[dict]] = {}
    for var in VARIANTS:
        t = time.time()
        all_snaps[var] = epdir.snapshots_v1_entry_price(df, eval_idxs, entry_price[var])
        print(f"{var} snapshots: {len(all_snaps[var])} ({time.time()-t:.0f}s)", flush=True)

    results = []
    for k, (name, snaps) in enumerate(all_snaps.items()):
        for split, sel in (("TRAIN", [s for s in snaps if s["t0"] < split_i]),
                           ("OOS", [s for s in snaps if s["t0"] >= split_i])):
            results.append(v1dir.summarize(name, split, sel, df, seed_off=k * 10 + (0 if split == "TRAIN" else 1)))
            print(f"evaluated {name}/{split} (n={len(sel)})", flush=True)

    print(f"\n{'variant':15s} {'split':6s} {'side':11s} {'buf%':5s} {'pairWR':7s} {'holdR':7s} {'holdP':7s} "
          f"{'mag24 diff':11s} {'mag72 diff':11s} {'nTouch':6s}")
    for r_ in results:
        for side in ("support", "resistance"):
            d = r_["eval"][side]
            for buf in ("0.005", "0.001"):
                row = d["by_buffer"][buf]
                mag24 = d["magnitude"]["24"]["mean_diff_pct"]
                mag72 = d["magnitude"]["72"]["mean_diff_pct"]
                print(f"{r_['variant']:15s} {r_['split']:6s} {side:11s} {float(buf)*100:4.1f} "
                      f"{str(row['paired']['winrate'])[:6]:7s} {str(row['real']['hold_rate'])[:6]:7s} "
                      f"{str(row['placebo']['hold_rate'])[:6]:7s} "
                      f"{('None' if mag24 is None else f'{mag24:+.3f}'):11s} "
                      f"{('None' if mag72 is None else f'{mag72:+.3f}'):11s} "
                      f"{row['real']['n_touched']:6d}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "join_stats": join_stats, "n_bars": n, "split_bar": split_i,
        "split_ts": str(df["timestamp"].iloc[split_i]), "warmup_hours": v1dir.WARMUP_HOURS,
        "lookback_hours_live": v1dir.LOOKBACK_HOURS_LIVE,
        "candle_color_counts": {"up": int(is_up.sum()), "down": int(is_down.sum()),
                                 "doji": int((~is_up & ~is_down).sum())},
        "color_price_vs_close_diff_pct": {"mean": float(diff_pct.mean()), "median": float(np.median(diff_pct)),
                                           "p95": float(np.percentile(diff_pct, 95)), "max": float(diff_pct.max())},
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
