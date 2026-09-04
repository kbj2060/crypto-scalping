#!/usr/bin/env python3
"""Isolated single-variable A/B, 2026-08-26 user follow-up: derive the long/short weight split
from the candle's OWN color (close vs open) instead of an external long/short-ratio feed. Every
direction source tried so far (taker buy-share, count_long_short_ratio, sum_toptrader_long_short_
ratio) needed an external data join; this one needs nothing beyond the OHLCV already in every
candle -- "an up-candle (close>open) leans long, a down-candle leans short" is the simplest
possible reading of the user's "양봉/음봉에 따라 포지션 비율을 다르게" proposal.

Mapping (parameter-free -- reuses the SAME LONG_SHARE_CLIP=(0.1,0.9) bound every other direction
variant already used, rather than inventing a new tunable tilt magnitude): long_share = 0.9 on an
up-candle, 0.1 on a down-candle, 0.5 on the (rare) exact-doji tie. This is a hard binary tilt
pinned to the project's existing clip bounds, not a continuous function of candle-body size --
keeping it in the same "no free parameters" spirit as v2b/v2c/v1_toptrader.

Reuses v1_direction_isolated_ab_20260826's compute_raw_bins_directional() / identity check /
snapshot generator / evaluate harness unmodified. Same pre-registered gate: adopt only if OOS
pairWR AND magnitude beat v1_live on resistance without making support worse; otherwise REJECTED.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_v2_20260825 as v2
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_v2_cohort_ab_backtest_20260825 as v2ab
import scripts.research_eth_liquidation_map_v2_phase0_data_audit_20260825 as audit
import scripts.research_eth_liquidation_map_v1_direction_isolated_ab_20260826 as v1dir

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_candle_color_direction_isolated_ab_20260826.json"

VARIANTS = ("v1_live", "v1_candle_color")


def main() -> None:
    px1h = v2ab.load_hourly_with_taker()
    m, clean = audit.load_metrics()
    df, join_stats = audit.hourly_join(m, px1h)
    print(f"join: {join_stats}", flush=True)
    df = df.rename(columns={"sum_open_interest": "oi"})

    v1dir._identity_check(df)

    is_up = (df["close"] > df["open"]).to_numpy()
    is_down = (df["close"] < df["open"]).to_numpy()
    candle_color_share = np.full(len(df), 0.5)
    candle_color_share[is_up] = v2.LONG_SHARE_CLIP[1]
    candle_color_share[is_down] = v2.LONG_SHARE_CLIP[0]
    print(f"candle color: up={int(is_up.sum())} down={int(is_down.sum())} "
          f"doji={int((~is_up & ~is_down).sum())} / {len(df)}", flush=True)

    long_share = {"v1_live": np.full(len(df), 0.5), "v1_candle_color": candle_color_share}

    n = len(df)
    split_i = int(n * v1dir.TRAIN_FRACTION)
    eval_idxs = base.asof_indices(n, v1dir.WARMUP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    print(f"bars={n} split at {df['timestamp'].iloc[split_i]} eval_points={len(eval_idxs)} "
          f"(train={sum(1 for i in eval_idxs if i < split_i)}, oos={sum(1 for i in eval_idxs if i >= split_i)})",
          flush=True)

    all_snaps: dict[str, list[dict]] = {}
    for var in VARIANTS:
        t = time.time()
        all_snaps[var] = v1dir.snapshots_v1_directional(df, eval_idxs, long_share[var])
        print(f"{var} snapshots: {len(all_snaps[var])} ({time.time()-t:.0f}s)", flush=True)

    results = []
    for k, (name, snaps) in enumerate(all_snaps.items()):
        for split, sel in (("TRAIN", [s for s in snaps if s["t0"] < split_i]),
                           ("OOS", [s for s in snaps if s["t0"] >= split_i])):
            results.append(v1dir.summarize(name, split, sel, df, seed_off=k * 10 + (0 if split == "TRAIN" else 1)))
            print(f"evaluated {name}/{split} (n={len(sel)})", flush=True)

    print(f"\n{'variant':16s} {'split':6s} {'side':11s} {'buf%':5s} {'pairWR':7s} {'holdR':7s} {'holdP':7s} "
          f"{'mag24 diff':11s} {'mag72 diff':11s} {'nTouch':6s}")
    for r_ in results:
        for side in ("support", "resistance"):
            d = r_["eval"][side]
            for buf in ("0.005", "0.001"):
                row = d["by_buffer"][buf]
                mag24 = d["magnitude"]["24"]["mean_diff_pct"]
                mag72 = d["magnitude"]["72"]["mean_diff_pct"]
                print(f"{r_['variant']:16s} {r_['split']:6s} {side:11s} {float(buf)*100:4.1f} "
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
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
