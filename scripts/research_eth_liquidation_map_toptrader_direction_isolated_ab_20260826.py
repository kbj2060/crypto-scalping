#!/usr/bin/env python3
"""Follow-up to research_eth_liquidation_map_v1_direction_isolated_ab_20260826.py: adds a 4th
direction-data source, sum_toptrader_long_short_ratio (Binance's topLongShortPositionRatio --
POSITION-SIZE weighted, unlike the account-COUNT-based count_long_short_ratio the v1_taker/v1_blend
variants used), per 2026-08-26 user follow-up. User's stated reasoning: count_long_short_ratio
mixes "who" (account count, retail-dominated) with dOI's "how much" (notional) -- a whale's $10M
short and a retail $100 long move the account ratio identically but not OI. topLongShortPositionRatio
sums actual position size for Binance's "top trader" cohort instead of counting accounts, fixing
that specific mismatch (at the cost of a different caveat: coverage is limited to whichever traders
Binance classifies as "top", not the whole market).

Coverage check (2026-08-26, ad hoc): sum_toptrader_long_short_ratio is 99.98% non-null, zero <=0
rows, full 2024-01..2026-08 span in data/TOTAL_ETHUSDT_metrics_2024_2026.csv -- same archive
v1_taker/v1_blend already used, just a column audit.hourly_join() doesn't select by default. Range
0.93..6.16 (mean 2.34, median 2.26) -- top traders run persistently net-long over this whole window,
unlike the roughly-balanced count-based ratio, so this variant's long_share sits well away from 0.5
on most bars (a genuinely different-shaped signal, not just a noisier version of the same one).

Reuses v1_direction_isolated_ab_20260826's compute_raw_bins_directional() / identity check /
snapshot generator / evaluate harness unmodified (imported, not copied) -- only addition here is the
extra archive join + one more long_share source. Same pre-registered gate as that script: adopt a
direction variant only if OOS pairWR AND magnitude beat v1_live on resistance (the side that keeps
regressing across every direction-data variant tried so far) without making support worse; anything
weaker is REJECTED and v1_live (as currently deployed) stays.
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
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_toptrader_direction_isolated_ab_20260826.json"

VARIANTS = ("v1_live", "v1_taker", "v1_blend", "v1_toptrader")


def main() -> None:
    px1h = v2ab.load_hourly_with_taker()
    m, clean = audit.load_metrics()
    df, join_stats = audit.hourly_join(m, px1h)
    print(f"join: {join_stats}", flush=True)
    df = df.rename(columns={"sum_open_interest": "oi"})
    r = df["count_long_short_ratio"].ffill()
    df["long_account_frac"] = (r / (1.0 + r)).fillna(0.5)

    # Extra join: sum_toptrader_long_short_ratio, same end-label convention as audit.hourly_join()
    # ("Kline bar T (open-label, [T,T+1h)) <- OI/LS/taker-ratio snapshot at end-label T+1h").
    snap_top = m.set_index("create_time")["sum_toptrader_long_short_ratio"]
    end_label = df["timestamp"] + pd.Timedelta(hours=1)
    df["sum_toptrader_long_short_ratio"] = snap_top.reindex(end_label).to_numpy()
    n_missing_top = int(df["sum_toptrader_long_short_ratio"].isna().sum())
    df["sum_toptrader_long_short_ratio"] = df["sum_toptrader_long_short_ratio"].ffill().bfill()
    print(f"toptrader ratio joined, missing (ffilled): {n_missing_top}/{len(df)}", flush=True)

    v1dir._identity_check(df)

    arrs = v2.prepare_cohort_arrays(df)
    top_r = df["sum_toptrader_long_short_ratio"].to_numpy(dtype="float64")
    top_long_frac = np.clip(top_r / (1.0 + top_r), *v2.LONG_SHARE_CLIP)
    long_share = {
        "v1_live": np.full(len(df), 0.5),
        "v1_taker": arrs["long_share"]["v2b"],
        "v1_blend": arrs["long_share"]["v2c"],
        "v1_toptrader": top_long_frac,
    }
    print(f"v1_toptrader long_share: mean={top_long_frac.mean():.3f} "
          f"min={top_long_frac.min():.3f} max={top_long_frac.max():.3f}", flush=True)

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

    print(f"\n{'variant':12s} {'split':6s} {'side':11s} {'buf%':5s} {'pairWR':7s} {'holdR':7s} {'holdP':7s} "
          f"{'mag24 diff':11s} {'mag72 diff':11s} {'nTouch':6s}")
    for r_ in results:
        for side in ("support", "resistance"):
            d = r_["eval"][side]
            for buf in ("0.005", "0.001"):
                row = d["by_buffer"][buf]
                mag24 = d["magnitude"]["24"]["mean_diff_pct"]
                mag72 = d["magnitude"]["72"]["mean_diff_pct"]
                print(f"{r_['variant']:12s} {r_['split']:6s} {side:11s} {float(buf)*100:4.1f} "
                      f"{str(row['paired']['winrate'])[:6]:7s} {str(row['real']['hold_rate'])[:6]:7s} "
                      f"{str(row['placebo']['hold_rate'])[:6]:7s} "
                      f"{('None' if mag24 is None else f'{mag24:+.3f}'):11s} "
                      f"{('None' if mag72 is None else f'{mag72:+.3f}'):11s} "
                      f"{row['real']['n_touched']:6d}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "join_stats": join_stats, "toptrader_missing_ffilled": n_missing_top, "n_bars": n,
        "split_bar": split_i, "split_ts": str(df["timestamp"].iloc[split_i]),
        "warmup_hours": v1dir.WARMUP_HOURS, "lookback_hours_live": v1dir.LOOKBACK_HOURS_LIVE,
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
