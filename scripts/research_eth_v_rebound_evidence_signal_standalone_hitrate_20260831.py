#!/usr/bin/env python3
"""Apply the EXACT SAME v7b outcome/answer-label formula used to score liquidity_sweep events
(scripts/research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py::realized_outcome,
imported not reimplemented) directly to each of the OTHER 5 evidence signals' own raw firings --
i.e. "if taker_delta_z_climax fires, what fraction of ITS OWN firings become a clean V자반등?",
directly comparable to liquidity_sweep's own 16.4% (204/1247).

This is DIFFERENT from research_eth_v_rebound_evidence_signal_correlation_20260831.py, which
checked temporal coincidence between these signals and an INDEPENDENTLY-confirmed set of local-
extreme V자반등 events. This script instead scores each signal's firings as its OWN standalone
candidate pool, exactly the way sweep events were scored in Part 1 of the recall-gap check.

Same 90-day window, same data loading, same compute_signals() reuse as the sibling scripts.
"""
from __future__ import annotations

import importlib.util
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

RECALL_SCRIPT = ROOT / "scripts/research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py"
_spec = importlib.util.spec_from_file_location("recall_check_90d_20260831", RECALL_SCRIPT)
_recall = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_recall)
realized_outcome = _recall.realized_outcome  # reused verbatim, not reimplemented

ETH_LOCAL_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_LOCAL_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp/eth_v_rebound_sweep_gate_recall_check_20260831"

WINDOW_DAYS = 90
LOOKBACK_MARGIN_DAYS = 5

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

    # realized_outcome() needs an 'atr' column computed the identical way the recall-gap script
    # did (pre-sweep ATR, from add_causal_columns) -- compute_signals() doesn't expose that name,
    # so build it the same way the sibling script does and merge onto the compute_signals() frame.
    impl = _recall.load_impl()
    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())

    sig = compute_signals(eth, btc_df=btc, funding_df=None).reset_index(drop=True)
    sig["atr"] = causal["atr"].to_numpy()
    n = len(sig)
    window_mask = (sig["timestamp"] >= analysis_start).to_numpy()

    # sweep, for a like-for-like reference row at the top of the table
    level_low, level_high = causal["sweep_level_low"], causal["sweep_level_high"]
    low, high, close = sig["low"], sig["high"], sig["close"]
    is_down_sweep = (level_low.notna() & (low < level_low) & (close > level_low)).to_numpy()
    is_up_sweep = (level_high.notna() & (high > level_high) & (close < level_high)).to_numpy()

    def score_signal(fire_bottom: np.ndarray, fire_top: np.ndarray) -> dict:
        buckets = {"V자반등": 0, "지지/횡보": 0, "애매(제외권)": 0}
        n_fired, n_scored = 0, 0
        for is_down, fire in ((True, fire_bottom), (False, fire_top)):
            for i in range(n):
                if not (window_mask[i] and fire[i]):
                    continue
                n_fired += 1
                o = realized_outcome(sig, i, is_down)
                if o is None or o["partial_window"]:
                    continue
                n_scored += 1
                buckets[o["outcome"]] += 1
        return {"n_fired": n_fired, "n_scored": n_scored, **buckets}

    print(f"분석구간: {analysis_start.date()} ~ {now_utc.date()} ({WINDOW_DAYS}일)\n")
    print(f"{'신호':28s} {'발동':>6s} {'채점':>6s} {'V자반등':>10s} {'지지/횡보':>10s} {'애매':>10s}")

    ref = score_signal(is_down_sweep, is_up_sweep)
    v_rate = ref["V자반등"] / ref["n_scored"] * 100 if ref["n_scored"] else float("nan")
    print(f"{'liquidity_sweep(참고,기존)':28s} {ref['n_fired']:6d} {ref['n_scored']:6d} "
          f"{ref['V자반등']:4d}({v_rate:4.1f}%) "
          f"{ref['지지/횡보']:4d}({ref['지지/횡보']/ref['n_scored']*100:4.1f}%) "
          f"{ref['애매(제외권)']:4d}({ref['애매(제외권)']/ref['n_scored']*100:4.1f}%)")

    results = {"liquidity_sweep": ref}
    for name in SIGNALS:
        fire_bottom = sig[f"bottom_{name}"].to_numpy()
        fire_top = sig[f"top_{name}"].to_numpy()
        r = score_signal(fire_bottom, fire_top)
        results[name] = r
        if r["n_scored"]:
            v_rate = r["V자반등"] / r["n_scored"] * 100
            print(f"{name:28s} {r['n_fired']:6d} {r['n_scored']:6d} "
                  f"{r['V자반등']:4d}({v_rate:4.1f}%) "
                  f"{r['지지/횡보']:4d}({r['지지/횡보']/r['n_scored']*100:4.1f}%) "
                  f"{r['애매(제외권)']:4d}({r['애매(제외권)']/r['n_scored']*100:4.1f}%)")
        else:
            print(f"{name:28s} {r['n_fired']:6d} {r['n_scored']:6d}  (채점가능표본없음)")

    import json
    (OUT_DIR / "standalone_hitrate.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n산출물: {OUT_DIR}/standalone_hitrate.json")
    print(f"실행시간: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
