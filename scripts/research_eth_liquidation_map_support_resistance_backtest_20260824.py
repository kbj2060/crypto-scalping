#!/usr/bin/env python3
"""Fresh-forward backtest of the Snapshot-tab liquidation map's support/resistance levels
(scripts/live_liquidation_map_20260824.py), 2026-08-24 -- user asked "are these levels actually
right, test within the data we have" after the panel shipped. This was NEVER validated before
shipping (it's a discretionary reading aid, not a promoted model -- see
docs/experiments/eth_candidate_liquidation_heatmap_magnet_signal_scoping_20260822.md), so this is
the first real look at whether the estimation method carries any information at all.

2026-08-24 follow-up: the first run used a 0.1% close-through buffer before counting a touched
level as "broken". User asked what tolerance was used and whether it was too tight -- "if it passes
through by about +-0.5% and then supports/resists, count that as a success; only a long pass-through
is a real loss." BUFFER_PCTS now runs BOTH 0.001 and 0.005 side by side per config so the two are
directly comparable, using the IDENTICAL real levels and IDENTICAL placebo draws for both buffer
widths (build_episodes() runs once per config; score_episodes() is called once per buffer width on
the same episode list) -- otherwise a second independent RNG draw would confound "buffer width
changed the result" with "the random placebo draw was just different this time".

=== Data ===
2026-08-24 2nd follow-up: user asked "are you actually using the maximum data we have?" -- the
answer had been no on both edges. Extended to the true local+reachable maximum:
  1. data/eth_5m_2021_2023_archive.csv (2021-12-01 00:00 -> 2023-12-31 23:55 UTC) prepended --
     overlaps data/eth_5m_1year.csv by ~9h at the boundary (archive runs to 23:55 on 12-31, the
     "1year" file starts at 15:00 the same day); rows before the "1year" file's first timestamp are
     kept from the archive, the rest come from "1year" -- zero gaps/dupes verified in both files
     independently before concatenating.
  2. A live Binance klines fetch (1h interval, paginated) fills 2026-02-17 15:00 UTC -> now, since
     no local file covers that stretch. This is the one place this script touches the network --
     unlike the first run's "no network calls" choice, "maximum data" cannot be answered honestly
     from local files alone once today's date is 6 months past the newest local bar. Means this
     script's most recent ~6 months of coverage, and therefore its as-of/episode count, GROWS on
     every future re-run -- not perfectly reproducible at the edge, unlike the fixed 2021-2026-02
     core. The still-forming current bar is dropped.
Combined range is therefore ~2021-12-01 -> as-of-run-time, roughly 4.7 years vs the first run's 2.15
-- nearly the practical maximum: this repo's own OI/LSR archives
(reference_clean_data_locations_20260823) also bottom out at 2021-12, and no earlier local OHLCV
was found. OHLCV is not one of the columns flagged unaudited in reference_clean_data_locations_20260823
(that flag is about the derived "metrics"/feature files this script does not touch; these are plain
Binance kline dumps + a direct Binance kline fetch). Note: 2022 includes the LUNA/FTX collapse --
a materially different vol/liquidation regime from 2024-2026 (the exact concern that got
[[eth_wide24_history_extension_2022_comparison_20260823]] rejected for a regime-conditional model)
-- this backtest makes no regime claim so it is not rejected on the same grounds, but results ARE
now pooled across regimes and any old-vs-new split should be read with that in mind.

=== Design ===
At each as-of point t0 (causal: only bars up to and including t0 are visible), call the PRODUCTION
compute_liquidation_levels() on the trailing LOOKBACK_HOURS window to get real support/resistance
levels, exactly as the live dashboard does it. For each real level, also generate a matched
PLACEBO level on the same side (support/resistance) at a distance_pct resampled from the pooled
empirical distribution of real distances for that side+config (with replacement) -- same "how far
from price do these levels typically sit" profile, but the specific price is NOT density-weighted.
This isolates whether the liquidation-density price SELECTION carries information, versus the
trivial "any level a plausible distance away sees some reaction" effect that would inflate a raw
hold-rate number with no comparison point.

Touch (within FORWARD_HOURS, using intrabar high/low -- a wick reaching the level counts, matching
this repo's established barrier convention: intrabar for "did price get there", close-based for
"did it actually reverse" -- see CLAUDE.md's Position-Feature Train/Inference Parity Contract for
the same intrabar-vs-close split applied to TP/SL barriers) then, among touched levels, HOLD vs
BREAK based on the first subsequent CLOSE beyond the level (with BUFFER_PCT slack -- see the
2026-08-24 follow-up note above for why this is now swept, not fixed) within FOLLOWTHROUGH_HOURS of
the touch bar -- this is exactly the "근거리 풀 스윕=진입타이밍, 종가이탈=무효화/반전" (near sweep
= entry timing, close-through = invalidation/reversal) framing the user used to describe how they
read Coinglass's heatmap in the first place.

As-of points are spaced FORWARD_HOURS apart (non-overlapping evaluation windows) to avoid the worst
double-counting of the same forward price move as independent confirmations -- lookback WINDOWS
still overlap heavily between adjacent episodes (a real limitation of only having ~2.15 years for a
lookback that itself eats up to 45 days), so episode counts are reported alongside results and must
not be read as fully independent samples. No p-values are reported for the same reason; the
headline comparison is a paired per-episode sign count (real hold-rate vs placebo hold-rate,
episodes where both sides had >=1 touch), which is simple, robust, and does not pretend to more
precision than a ~2-year single-asset single-timeframe sample supports.

MONKEYPATCHING NOTE: compute_liquidation_levels() reads its tunable parameters (LEVERAGE_TIERS,
RECENCY_HALFLIFE_HOURS, MAX_LEVELS_PER_SIDE, MIN_LEVEL_SHARE, BIN_WIDTH_PCT) as module globals, not
function arguments -- this script monkeypatches those globals on the imported module before each
config's run rather than editing the shipped module (which is live in production and already
tested; adding a parameters argument nobody else needs would be scope creep on that file for a
one-off research question). Safe here because this is a standalone process separate from the
dashboard server.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import scripts.live_liquidation_map_20260824 as liqmap

ROOT = Path(__file__).resolve().parents[1]
PRICE_CSV = ROOT / "data" / "eth_5m_1year.csv"
ARCHIVE_CSV = ROOT / "data" / "eth_5m_2021_2023_archive.csv"
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_sr_backtest_20260824.json"
GAP_FETCH_SYMBOL = "ETHUSDT"

FORWARD_HOURS = 72          # touch-evaluation window after the as-of point
FOLLOWTHROUGH_HOURS = 24    # window after a touch to check for a close-through (break)
BUFFER_PCTS = (0.001, 0.005)  # 0.1% (original) vs 0.5% (user-requested, 2026-08-24 follow-up)
N_PLACEBO_DRAWS_SEED = 20260824  # fixed seed, reproducible

PRIMARY_LOOKBACK_DAYS = (1, 5, 7, 15, 30, 45, 90)   # 1/7/30/90 match Coinglass's own lookback
                                                      # presets (2026-08-24 3rd follow-up); 5/15/45
                                                      # kept from earlier rounds for continuity
SECONDARY_HALFLIFE_HOURS = (60.0, 720.0)             # vs default 240.0 -- aggressive/mild decay
SECONDARY_LEVERAGE_SETS = {
    "low_lev_10_20_25": (10, 20, 25),               # wider, low-leverage-implied levels
    "high_lev_50_75_100": (50, 75, 100),             # tighter, high-leverage-implied levels
}
DEFAULT_HALFLIFE = liqmap.RECENCY_HALFLIFE_HOURS
DEFAULT_LEVERAGE = liqmap.LEVERAGE_TIERS


def _load_local_5m() -> pd.DataFrame:
    """Archive (2021-12-01 -> 2023-12-31 23:55) + main file (2023-12-31 15:00 -> 2026-02-17),
    overlap resolved by preferring the main file for any timestamp it also covers."""
    archive = pd.read_csv(ARCHIVE_CSV)
    archive["timestamp"] = pd.to_datetime(archive["open_time"], unit="ms", utc=True)
    main = pd.read_csv(PRICE_CSV, parse_dates=["timestamp"])
    main["timestamp"] = main["timestamp"].dt.tz_localize("UTC")
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    combined = pd.concat([archive[cols], main[cols]], ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return combined


def _resample_1h(df_5m: pd.DataFrame) -> pd.DataFrame:
    d = df_5m.set_index("timestamp")
    return (
        d.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )


def _fetch_binance_1h_gap(start_after: pd.Timestamp) -> pd.DataFrame:
    """Paginates /fapi/v1/klines (1h) from just after start_after through the last fully-closed
    hour. Network failures propagate (caller decides whether to proceed on local-only data)."""
    start_ms = int((start_after + pd.Timedelta(hours=1)).timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    rows = []
    while start_ms < now_ms:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": GAP_FETCH_SYMBOL, "interval": "1h", "startTime": start_ms, "limit": 1500},
            timeout=15,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open_ms = int(batch[-1][0])
        if last_open_ms <= start_ms:  # safety: no forward progress, avoid an infinite loop
            break
        start_ms = last_open_ms + 3600_000
        if len(batch) < 1500:
            break
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close_time"] = df["close_time"].astype("int64")
    df = df[df["close_time"] < now_ms]  # drop the still-forming current bar
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)


def load_hourly() -> pd.DataFrame:
    local_1h = _resample_1h(_load_local_5m())
    try:
        gap = _fetch_binance_1h_gap(local_1h["timestamp"].iloc[-1])
        print(f"fetched {len(gap)} gap bars from Binance: "
              f"{gap['timestamp'].iloc[0] if len(gap) else '-'} -> {gap['timestamp'].iloc[-1] if len(gap) else '-'}", flush=True)
    except requests.RequestException as e:
        print(f"gap fetch failed ({e}) -- proceeding on local data only (up to {local_1h['timestamp'].iloc[-1]})", flush=True)
        gap = pd.DataFrame(columns=local_1h.columns)
    combined = pd.concat([local_1h, gap], ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    gaps = combined["timestamp"].diff().dropna()
    n_gaps = int((gaps > pd.Timedelta(hours=1)).sum())
    if n_gaps:
        # asof_indices()/evaluate_forward() window by ROW COUNT (72 rows for FORWARD_HOURS=72),
        # not by elapsed wall-clock time -- a real gap inside a lookback/forward window would
        # silently make that window span more real time than its hour-count implies. Not fatal for
        # a handful of short exchange-maintenance-style gaps, but large gaps would need excluding
        # the affected as-of range, which this script does NOT currently do -- flagged, not fixed.
        print(f"WARNING: {n_gaps} gap(s) >1h found in combined series (max {gaps.max()}) -- "
              f"row-count-based windowing means any as-of/forward window spanning a gap is skewed. "
              f"Inspect before trusting results near these gaps.", flush=True)
    return combined


def asof_indices(n: int, lookback_hours: int, forward_hours: int, followthrough_hours: int) -> list[int]:
    start = lookback_hours
    end = n - forward_hours - followthrough_hours - 1
    if end <= start:
        return []
    return list(range(start, end, forward_hours))


def evaluate_forward(df: pd.DataFrame, asof_idx: int, level_price: float, side: str, buffer_pct: float) -> str:
    """Returns 'not_touched' | 'hold' | 'break'. side: 'support' (below) or 'resistance' (above).
    Touch itself has zero tolerance (any wick contact counts, the most lenient possible touch
    definition) -- buffer_pct only widens how far a subsequent CLOSE must travel past the level
    before it counts as a genuine break, not a normal overshoot-and-reverse."""
    n = len(df)
    fwd_end = min(n, asof_idx + 1 + FORWARD_HOURS)
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    touch_i = None
    for i in range(asof_idx + 1, fwd_end):
        if side == "support" and lows[i] <= level_price:
            touch_i = i
            break
        if side == "resistance" and highs[i] >= level_price:
            touch_i = i
            break
    if touch_i is None:
        return "not_touched"
    ft_end = min(n, touch_i + 1 + FOLLOWTHROUGH_HOURS)
    if side == "support":
        broke = bool(np.any(closes[touch_i:ft_end] < level_price * (1 - buffer_pct)))
    else:
        broke = bool(np.any(closes[touch_i:ft_end] > level_price * (1 + buffer_pct)))
    return "break" if broke else "hold"


def build_episodes(df: pd.DataFrame, lookback_days: int, halflife_hours: float, leverage_tiers: tuple,
                    rng: np.random.Generator) -> list[dict]:
    """Runs the PRODUCTION compute_liquidation_levels() once per as-of point and draws the matched
    placebo prices -- independent of buffer_pct, so this is computed exactly once per (lookback,
    halflife, leverage) config and then scored at every buffer width against the identical draws."""
    lookback_hours = lookback_days * 24
    liqmap.RECENCY_HALFLIFE_HOURS = halflife_hours
    liqmap.LEVERAGE_TIERS = leverage_tiers

    idxs = asof_indices(len(df), lookback_hours, FORWARD_HOURS, FOLLOWTHROUGH_HOURS)
    episodes = []
    real_distances = {"support": [], "resistance": []}

    for t0 in idxs:
        window = df.iloc[t0 - lookback_hours: t0 + 1]
        current_price = float(window["close"].iloc[-1])
        result = liqmap.compute_liquidation_levels(window, current_price)
        if not result["warmed_up"]:
            continue
        sup = result["support_levels"]
        res = result["resistance_levels"]
        if not sup and not res:
            continue
        real_distances["support"].extend(lv["distance_pct"] for lv in sup)
        real_distances["resistance"].extend(lv["distance_pct"] for lv in res)
        episodes.append({"t0": t0, "current_price": current_price, "support": sup, "resistance": res})

    if not episodes:
        return []

    dist_pool = {
        side: np.array(vals) if vals else np.array([2.0, -2.0])
        for side, vals in real_distances.items()
    }
    for ep in episodes:
        for side in ("support", "resistance"):
            levels = ep[side]
            if not levels:
                ep[f"placebo_{side}"] = []
                continue
            placebo_dists = rng.choice(dist_pool[side], size=len(levels), replace=True)
            ep[f"placebo_{side}"] = [ep["current_price"] * (1 + d / 100.0) for d in placebo_dists]
    return episodes


def score_episodes(df: pd.DataFrame, episodes: list[dict], buffer_pct: float) -> dict:
    sides_out = {}
    for side in ("support", "resistance"):
        real_rows = []      # (episode_i, outcome)
        placebo_rows = []
        for ei, ep in enumerate(episodes):
            for lv in ep[side]:
                real_rows.append((ei, evaluate_forward(df, ep["t0"], lv["price"], side, buffer_pct)))
            for pp in ep[f"placebo_{side}"]:
                placebo_rows.append((ei, evaluate_forward(df, ep["t0"], pp, side, buffer_pct)))

        def agg(rows):
            outcomes = [o for _, o in rows]
            n = len(outcomes)
            touched = [o for o in outcomes if o != "not_touched"]
            n_touched = len(touched)
            n_hold = sum(1 for o in touched if o == "hold")
            return {
                "n_levels": n,
                "touch_rate": n_touched / n if n else None,
                "n_touched": n_touched,
                "hold_rate": n_hold / n_touched if n_touched else None,
            }

        real_agg = agg(real_rows)
        placebo_agg = agg(placebo_rows)

        # Paired-by-episode sign comparison: per episode, hold-rate among that episode's touched
        # real levels vs touched placebo levels, only where both sides have >=1 touch.
        by_ep_real, by_ep_placebo = {}, {}
        for ei, o in real_rows:
            if o == "not_touched":
                continue
            by_ep_real.setdefault(ei, []).append(1 if o == "hold" else 0)
        for ei, o in placebo_rows:
            if o == "not_touched":
                continue
            by_ep_placebo.setdefault(ei, []).append(1 if o == "hold" else 0)
        paired_diffs = []
        for ei in set(by_ep_real) & set(by_ep_placebo):
            r = np.mean(by_ep_real[ei])
            p = np.mean(by_ep_placebo[ei])
            paired_diffs.append(r - p)
        n_paired = len(paired_diffs)
        n_favor_real = sum(1 for d in paired_diffs if d > 0)
        n_favor_placebo = sum(1 for d in paired_diffs if d < 0)
        n_tie = n_paired - n_favor_real - n_favor_placebo

        sides_out[side] = {
            "real": real_agg,
            "placebo": placebo_agg,
            "paired": {
                "n_paired_episodes": n_paired,
                "mean_paired_diff": float(np.mean(paired_diffs)) if paired_diffs else None,
                "n_favor_real": n_favor_real,
                "n_favor_placebo": n_favor_placebo,
                "n_tie": n_tie,
            },
        }
    return sides_out


def run_config(df: pd.DataFrame, lookback_days: int, halflife_hours: float, leverage_tiers: tuple,
               label: str, rng: np.random.Generator) -> list[dict]:
    episodes = build_episodes(df, lookback_days, halflife_hours, leverage_tiers, rng)
    out = []
    for buffer_pct in BUFFER_PCTS:
        out.append({
            "label": label, "lookback_days": lookback_days, "halflife_hours": halflife_hours,
            "leverage_tiers": list(leverage_tiers), "buffer_pct": buffer_pct,
            "n_episodes": len(episodes),
            "sides": score_episodes(df, episodes, buffer_pct) if episodes else {},
        })
    return out


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "


def main() -> None:
    df = load_hourly()
    print(f"hourly bars: {len(df)}  range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}", flush=True)
    rng = np.random.default_rng(N_PLACEBO_DRAWS_SEED)

    configs = []
    for d in PRIMARY_LOOKBACK_DAYS:
        configs.append((d, DEFAULT_HALFLIFE, DEFAULT_LEVERAGE, f"lookback={d}d,halflife=default,lev=default"))
    for d in PRIMARY_LOOKBACK_DAYS:
        for hl in SECONDARY_HALFLIFE_HOURS:
            configs.append((d, hl, DEFAULT_LEVERAGE, f"lookback={d}d,halflife={hl}h,lev=default"))
        for lev_name, lev in SECONDARY_LEVERAGE_SETS.items():
            configs.append((d, DEFAULT_HALFLIFE, lev, f"lookback={d}d,halflife=default,lev={lev_name}"))

    results = []
    for d, hl, lev, label in configs:
        for r in run_config(df, d, hl, lev, label, rng):
            results.append(r)
            s = r["sides"].get("support", {})
            rr = r["sides"].get("resistance", {})
            print(
                f"{label:55s} buf={r['buffer_pct']*100:.1f}% n_ep={r['n_episodes']:4d} | "
                f"sup touch={fmt(s.get('real',{}).get('touch_rate'))} hold real={fmt(s.get('real',{}).get('hold_rate'))} "
                f"placebo={fmt(s.get('placebo',{}).get('hold_rate'))} pairedN={s.get('paired',{}).get('n_paired_episodes','-')} "
                f"favor_real={s.get('paired',{}).get('n_favor_real','-')} favor_placebo={s.get('paired',{}).get('n_favor_placebo','-')} | "
                f"res hold real={fmt(rr.get('real',{}).get('hold_rate'))} placebo={fmt(rr.get('placebo',{}).get('hold_rate'))} "
                f"favor_real={rr.get('paired',{}).get('n_favor_real','-')} favor_placebo={rr.get('paired',{}).get('n_favor_placebo','-')}",
                flush=True,
            )

    liqmap.RECENCY_HALFLIFE_HOURS = DEFAULT_HALFLIFE
    liqmap.LEVERAGE_TIERS = DEFAULT_LEVERAGE

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
