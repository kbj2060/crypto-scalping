#!/usr/bin/env python3
"""Phase 1 A/B for the liquidation-map v2 (OI cohort) design -- docs/experiments/
eth_liquidation_map_v2_oi_cohort_direction_design_20260825.md section 5.

Compares, on the identical eval grid and with the identical touch/hold/placebo/magnitude harness
(ed.evaluate, unmodified):
  v1  -- production event-driven estimate (ed.simulate's state machine, same code path as live)
  v2a -- OI-cohort weighting, symmetric sides
  v2b -- + direction split by taker share (clip 0.1..0.9, no free parameters)
  v2c -- + 50:50 blend with global long/short account fraction (fixed blend, no sweep)

Pre-registered gate (design doc 5): adopt the highest ladder rung with OOS >= v1 and > placebo;
if none, v2 is REJECTED and v1 stays. TRAIN(first 80% of bars)/OOS(last 20%) split by snapshot t0,
identical convention to research_eth_liquidation_map_event_driven_min_floor_sweep_20260825.py --
that sweep's TRAIN-promising/OOS-rejected outcome is the expectation-setting precedent here.

Data: hourly klines (local 5m CSVs + Binance gap fetch, WITH taker_buy_base kept) joined to the
integrity-audited metrics archive per the Phase 0 conventions (kline bar T <- OI snapshot at
T+1h end-label; OI<=0 ffilled; |dOI|<=volume clamped inside prepare_cohort_arrays). Effective
window = metrics overlap 2024-01 .. 2026-08 (~2.6y), eval starts after a 90d cohort warmup.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.live_liquidation_map_v2_20260825 as v2
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed
import scripts.research_eth_liquidation_map_v2_phase0_data_audit_20260825 as audit

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_v2_cohort_ab_backtest_20260825.json"

WARMUP_HOURS = 90 * 24   # cohort accumulation before first eval (Phase 0: unattributed initial
                         # mass < 0.9% by then); also >= v1's own 168h bootstrap
TRAIN_FRACTION = 0.8
SEED = 20260825


def load_hourly_with_taker() -> pd.DataFrame:
    """base.load_hourly() but keeping taker_buy_base -- v2b/c need it, and the gap-fetched region
    (2026-02-17..) covers most of the OOS split. Same sources, same forming-bar drop."""
    archive = pd.read_csv(base.ARCHIVE_CSV)
    archive["timestamp"] = pd.to_datetime(archive["open_time"], unit="ms", utc=True)
    main = pd.read_csv(base.PRICE_CSV, parse_dates=["timestamp"])
    main["timestamp"] = main["timestamp"].dt.tz_localize("UTC")
    cols = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]
    px5 = (pd.concat([archive[cols], main[cols]], ignore_index=True)
           .sort_values("timestamp").drop_duplicates("timestamp", keep="last"))
    local = (px5.set_index("timestamp").resample("1h")
             .agg({"open": "first", "high": "max", "low": "min", "close": "last",
                   "volume": "sum", "taker_buy_base": "sum"})
             .dropna().reset_index())

    start_ms = int((local["timestamp"].iloc[-1] + pd.Timedelta(hours=1)).timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    rows = []
    while start_ms < now_ms:
        resp = requests.get("https://fapi.binance.com/fapi/v1/klines",
                            params={"symbol": "ETHUSDT", "interval": "1h",
                                    "startTime": start_ms, "limit": 1500}, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        if last_open <= start_ms:
            break
        start_ms = last_open + 3600_000
        if len(batch) < 1500:
            break
    gap = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
                                      "close_time", "qv", "tr", "taker_buy_base", "tq", "ig"])
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        gap[c] = gap[c].astype("float64")
    gap["timestamp"] = pd.to_datetime(gap["open_time"].astype("int64"), unit="ms", utc=True)
    gap = gap[gap["close_time"].astype("int64") < now_ms][cols]
    out = (pd.concat([local, gap], ignore_index=True)
           .sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    print(f"hourly bars with taker: {len(out)} ({out['timestamp'].iloc[0]} .. {out['timestamp'].iloc[-1]})",
          flush=True)
    return out


def simulate_v1(df: pd.DataFrame, eval_idxs: set[int]) -> list[dict]:
    """ed.simulate() verbatim except snapshots are taken at the INJECTED eval grid so all variants
    share identical as-of points (ed hardcodes its own asof_indices start)."""
    n = len(df)
    close = df["close"].to_numpy()
    support_reset_idx = resistance_reset_idx = 0

    def regenerate(side: str, reset_idx: int, i: int) -> list[dict]:
        start = max(reset_idx, i - ed.MAX_LOOKBACK_HOURS)
        start = min(start, max(0, i - ed.MIN_FLOOR_HOURS))
        raw = liqmap.compute_raw_bins(df.iloc[start:i + 1], float(close[i]))
        if raw is None:
            return []
        bins, bin_width, _, _ = raw
        key = "support_levels" if side == "support" else "resistance_levels"
        return liqmap.levels_from_bins(bins, bin_width, float(close[i]))[key]

    support_levels = regenerate("support", 0, ed.BOOTSTRAP_HOURS)
    resistance_levels = regenerate("resistance", 0, ed.BOOTSTRAP_HOURS)
    snapshots = []
    for i in range(ed.BOOTSTRAP_HOURS + 1, n):
        price = close[i]
        broke_s = any(price < lv["price"] * (1 - ed.BREAK_TOLERANCE_PCT) for lv in support_levels)
        broke_r = any(price > lv["price"] * (1 + ed.BREAK_TOLERANCE_PCT) for lv in resistance_levels)
        drift_s = bool(support_levels) and \
            (price - max(lv["price"] for lv in support_levels)) / price > ed.DRIFT_TOLERANCE_PCT
        drift_r = bool(resistance_levels) and \
            (min(lv["price"] for lv in resistance_levels) - price) / price > ed.DRIFT_TOLERANCE_PCT
        if broke_s or drift_s:
            support_levels = regenerate("support", support_reset_idx, i)
            support_reset_idx = i
        if broke_r or drift_r:
            resistance_levels = regenerate("resistance", resistance_reset_idx, i)
            resistance_reset_idx = i
        if i in eval_idxs:
            snapshots.append({"t0": i, "current_price": float(price),
                              "support_levels": support_levels, "resistance_levels": resistance_levels})
    return snapshots


def snapshots_v2(df: pd.DataFrame, eval_idxs: list[int], variant: str) -> list[dict]:
    arrs = v2.prepare_cohort_arrays(df)
    close = df["close"].to_numpy()
    out = []
    for i in eval_idxs:
        payload = v2.compute_cohort_levels(arrs, i, variant)
        if not payload.get("warmed_up"):
            continue
        out.append({"t0": i, "current_price": float(close[i]),
                    "support_levels": payload["support_levels"],
                    "resistance_levels": payload["resistance_levels"],
                    "long_usd_total": payload["long_usd_total"],
                    "short_usd_total": payload["short_usd_total"]})
    return out


def summarize(name: str, split: str, snaps: list[dict], df: pd.DataFrame, seed_off: int) -> dict:
    rng = np.random.default_rng(SEED + seed_off)
    ev = ed.evaluate(df, snaps, rng)
    n_lv = [len(s["support_levels"]) + len(s["resistance_levels"]) for s in snaps]
    return {"variant": name, "split": split, "n_snapshots": len(snaps),
            "avg_levels_per_snapshot": round(float(np.mean(n_lv)), 2) if n_lv else 0.0,
            "eval": ev}


def main() -> None:
    px1h = load_hourly_with_taker()
    m, clean = audit.load_metrics()
    df, join_stats = audit.hourly_join(m, px1h)
    print(f"join: {join_stats}", flush=True)
    df = df.rename(columns={"sum_open_interest": "oi"})
    r = df["count_long_short_ratio"].ffill()
    df["long_account_frac"] = (r / (1.0 + r)).fillna(0.5)

    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    eval_idxs = base.asof_indices(n, WARMUP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    print(f"bars={n} split at {df['timestamp'].iloc[split_i]} eval_points={len(eval_idxs)} "
          f"(train={sum(1 for i in eval_idxs if i < split_i)}, oos={sum(1 for i in eval_idxs if i >= split_i)})",
          flush=True)

    all_snaps: dict[str, list[dict]] = {}
    t = time.time()
    all_snaps["v1"] = simulate_v1(df, set(eval_idxs))
    print(f"v1 snapshots: {len(all_snaps['v1'])} ({time.time()-t:.0f}s)", flush=True)
    for var in v2.VARIANTS:
        t = time.time()
        all_snaps[var] = snapshots_v2(df, eval_idxs, var)
        print(f"{var} snapshots: {len(all_snaps[var])} ({time.time()-t:.0f}s)", flush=True)

    results = []
    for k, (name, snaps) in enumerate(all_snaps.items()):
        for split, sel in (("TRAIN", [s for s in snaps if s["t0"] < split_i]),
                           ("OOS", [s for s in snaps if s["t0"] >= split_i])):
            results.append(summarize(name, split, sel, df, seed_off=k * 10 + (0 if split == "TRAIN" else 1)))
            print(f"evaluated {name}/{split} (n={len(sel)})", flush=True)

    print(f"\n{'variant':8s} {'split':6s} {'side':11s} {'buf%':5s} {'pairWR':7s} {'holdR':7s} {'holdP':7s} "
          f"{'mag24 diff':11s} {'mag72 diff':11s} {'nTouch':6s}")
    for r_ in results:
        for side in ("support", "resistance"):
            d = r_["eval"][side]
            for buf in ("0.005", "0.001"):
                row = d["by_buffer"][buf]
                mag24 = d["magnitude"]["24"]["mean_diff_pct"]
                mag72 = d["magnitude"]["72"]["mean_diff_pct"]
                print(f"{r_['variant']:8s} {r_['split']:6s} {side:11s} {float(buf)*100:4.1f} "
                      f"{str(row['paired']['winrate'])[:6]:7s} {str(row['real']['hold_rate'])[:6]:7s} "
                      f"{str(row['placebo']['hold_rate'])[:6]:7s} "
                      f"{('None' if mag24 is None else f'{mag24:+.3f}'):11s} "
                      f"{('None' if mag72 is None else f'{mag72:+.3f}'):11s} "
                      f"{row['real']['n_touched']:6d}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "join_stats": join_stats, "zero_clean": clean, "n_bars": n,
        "split_bar": split_i, "split_ts": str(df["timestamp"].iloc[split_i]),
        "warmup_hours": WARMUP_HOURS, "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
