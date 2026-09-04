#!/usr/bin/env python3
"""Do the 491 non-sweep V자반등 events (tmp/eth_v_rebound_sweep_gate_recall_check_20260831/
events.csv, group=no_sweep) temporally coincide with any of this dashboard's other 5 evidence
signals firing nearby? If most do, "reuse existing evidence signals as additional V자반등
triggers" is directly actionable via code reuse. If not, a new trigger needs to be designed from
scratch for the ones that don't overlap.

Reuses (imports, does not reimplement) compute_signals() from
scripts/live_evidence_signal_dashboard_20260823.py -- the single canonical, pre-TabPFN, pure-
pandas home for all 8 raw evidence-signal boolean triggers (verified by an independent Explore
pass against each signal's original source script before writing this -- byte-for-byte match).
liquidity_sweep is excluded (it's the trigger already fully accounted for in the sweep/no_sweep
split); volume_wick_climax and dalton_rule2_balance_edge are excluded (removed from the dashboard
2026-08-31, weak/failed signals per eth_volume_wick_climax_metalabel_v1_weak_signal_20260830 and
eth_dalton_rule2_balance_edge_metalabel_v1_20260830).

Coincidence window: +-3 bars (+-15min) around each V자반등 event's pivot bar, matching this
project's own empirically-derived SUSTAIN_BARS=4 cutoff (scripts/live_evidence_signal_dashboard_
20260823.py's 2026-08-24 sustain-window note: lift stays clearly above baseline through offset 3
for all 14 signal x side series, several cross to at-or-below baseline at offset 4) -- not a
freshly guessed number.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path("/home/kbj20/crypto-scalping")
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

EVENTS_CSV = ROOT / "tmp/eth_v_rebound_sweep_gate_recall_check_20260831/events.csv"
ETH_LOCAL_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_LOCAL_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp/eth_v_rebound_sweep_gate_recall_check_20260831"

WINDOW_DAYS = 90
LOOKBACK_MARGIN_DAYS = 5  # covers orthogonal_combo's 864-bar (~3day) warmup with margin
COINCIDENCE_BARS = 3  # +-15min around the V-rebound event's pivot bar

SIGNALS = ["taker_delta_z_climax", "short_term_return_z", "orthogonal_combo",
           "smt_divergence", "fib_extension_exhaustion"]


def fetch_klines_since(symbol: str, start_ms: int) -> pd.DataFrame:
    frames = []
    cursor = start_ms
    now_ms = int(time.time() * 1000)
    while cursor < now_ms:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": symbol, "interval": "5m", "startTime": cursor, "limit": 1500},
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            break
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv",
                "trades", "taker_buy_base", "tq", "ignore"]
        df = pd.DataFrame(raw, columns=cols)
        frames.append(df)
        last_open = int(df.iloc[-1]["open_time"])
        if last_open <= cursor or len(raw) < 1500:
            cursor = last_open + 1
            break
        cursor = last_open + 1
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    out = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        out[c] = out[c].astype(float)
    out["timestamp"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    out = out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    now_ms2 = int(time.time() * 1000)
    if len(out) and int(out.iloc[-1]["close_time"]) >= now_ms2:
        out = out.iloc[:-1]
    return out[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]].reset_index(drop=True)


def load_combined(symbol: str, local_csv: Path, lookback_start: pd.Timestamp) -> pd.DataFrame:
    local = pd.read_csv(local_csv, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
    local = local[local["timestamp"] >= lookback_start]
    gap_start_ms = (int(local["timestamp"].iloc[-1].timestamp() * 1000) + 1
                    if len(local) else int(lookback_start.timestamp() * 1000))
    live = fetch_klines_since(symbol, gap_start_ms)
    combined = pd.concat([local, live], ignore_index=True)
    combined = combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return combined


def main() -> None:
    t0 = time.time()
    now_utc = pd.Timestamp.now(tz="UTC")
    analysis_start = now_utc - pd.Timedelta(days=WINDOW_DAYS)
    lookback_start = analysis_start - pd.Timedelta(days=LOOKBACK_MARGIN_DAYS)

    eth = load_combined("ETHUSDT", ETH_LOCAL_CSV, lookback_start)
    btc = load_combined("BTCUSDT", BTC_LOCAL_CSV, lookback_start)
    print(f"ETH {len(eth)}bars {eth['timestamp'].iloc[0]} ~ {eth['timestamp'].iloc[-1]}")
    print(f"BTC {len(btc)}bars {btc['timestamp'].iloc[0]} ~ {btc['timestamp'].iloc[-1]}")

    sig = compute_signals(eth, btc_df=btc, funding_df=None)
    sig = sig.reset_index(drop=True)
    n = len(sig)
    ts_to_idx = {ts: i for i, ts in enumerate(sig["timestamp"])}

    window_mask = sig["timestamp"] >= analysis_start
    base_rates = {}
    for name in SIGNALS:
        b = sig.loc[window_mask, f"bottom_{name}"]
        t = sig.loc[window_mask, f"top_{name}"]
        base_rates[name] = {"bottom": float(b.mean()), "top": float(t.mean())}

    events = pd.read_csv(EVENTS_CSV)
    events = events[events["outcome"] == "V자반등"].copy()

    def coincidence_row(row) -> dict:
        ts = pd.Timestamp(row["timestamp_utc"])
        idx = ts_to_idx.get(ts)
        is_down = row["direction"] == "downside"
        col_prefix = "bottom" if is_down else "top"
        result = {"group": row["group"], "direction": row["direction"]}
        if idx is None:
            for name in SIGNALS:
                result[name] = None
            result["any"] = None
            return result
        lo, hi = max(0, idx - COINCIDENCE_BARS), min(n, idx + COINCIDENCE_BARS + 1)
        any_fired = False
        for name in SIGNALS:
            fired = bool(sig[f"{col_prefix}_{name}"].iloc[lo:hi].any())
            result[name] = fired
            any_fired = any_fired or fired
        result["any"] = any_fired
        return result

    rows = [coincidence_row(r) for _, r in events.iterrows()]
    cdf = pd.DataFrame(rows)
    matched = cdf[cdf[SIGNALS[0]].notna()]  # drop events whose timestamp didn't align (should be ~0)
    print(f"\n이벤트 {len(events)}건 중 시그널프레임과 정렬된 것: {len(matched)}건 "
          f"(정렬안된 {len(events)-len(matched)}건은 웜업/경계 근처로 추정, 통계에서 제외)\n")

    for group in ("no_sweep", "sweep"):
        sub = matched[matched["group"] == group]
        if not len(sub):
            continue
        n_down = (sub["direction"] == "downside").sum()
        n_up = (sub["direction"] == "upside").sum()
        print(f"### group={group} (n={len(sub)}, downside={n_down}/upside={n_up}) ###")
        for name in SIGNALS:
            rate = sub[name].mean()
            # direction-weighted base rate for fair comparison
            base = (n_down * base_rates[name]["bottom"] + n_up * base_rates[name]["top"]) / len(sub)
            print(f"  {name:26s}: 겹침 {rate:5.1%}   (기저율 {base:5.1%}, {rate/base if base>0 else float('nan'):.1f}x)")
        print(f"  {'(5개 중 최소1개)':26s}: 겹침 {sub['any'].mean():5.1%}")
        print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cdf.to_csv(OUT_DIR / "evidence_signal_coincidence.csv", index=False)
    print(f"산출물: {OUT_DIR}/evidence_signal_coincidence.csv")
    print(f"실행시간: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
