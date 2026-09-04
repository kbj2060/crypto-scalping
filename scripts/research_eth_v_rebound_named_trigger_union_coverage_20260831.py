#!/usr/bin/env python3
"""Final consolidation: dedup across (sweep + 5 named evidence signals) to get the TRUE union
count of V자반등 events they jointly cover (not the naive sum, which double-counts bars where
multiple signals co-fire), then compare against the local-extreme method's 491 to see how much
of the "true addressable total" is reachable using ONLY existing named signals as triggers vs
needing the fully generic local-extreme approach on top.

Reuses realized_outcome() (imported, not reimplemented) and compute_signals() exactly as the
sibling scripts in this investigation thread do.
"""
from __future__ import annotations

import importlib.util
import json
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
_spec = importlib.util.spec_from_file_location("recall_check_90d_20260831b", RECALL_SCRIPT)
_recall = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_recall)
realized_outcome = _recall.realized_outcome

ETH_LOCAL_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_LOCAL_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp/eth_v_rebound_sweep_gate_recall_check_20260831"
EVENTS_CSV = OUT_DIR / "events.csv"  # local-extreme method's 695 (204 sweep + 491 no_sweep)

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
    impl = _recall.load_impl()
    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())

    sig = compute_signals(eth, btc_df=btc, funding_df=None).reset_index(drop=True)
    sig["atr"] = causal["atr"].to_numpy()
    n = len(sig)
    window_mask = (sig["timestamp"] >= analysis_start).to_numpy()
    ts_to_idx = {ts: i for i, ts in enumerate(sig["timestamp"])}

    level_low, level_high = causal["sweep_level_low"], causal["sweep_level_high"]
    low, high, close = sig["low"], sig["high"], sig["close"]
    is_down_sweep = (level_low.notna() & (low < level_low) & (close > level_low)).to_numpy()
    is_up_sweep = (level_high.notna() & (high > level_high) & (close < level_high)).to_numpy()

    fire_down = is_down_sweep.copy()
    fire_up = is_up_sweep.copy()
    for name in SIGNALS:
        fire_down |= sig[f"bottom_{name}"].to_numpy()
        fire_up |= sig[f"top_{name}"].to_numpy()

    named_v_rebound = set()  # (idx, is_down) pairs, deduped union of sweep + 5 signals
    for is_down, fire in ((True, fire_down), (False, fire_up)):
        for i in range(n):
            if not (window_mask[i] and fire[i]):
                continue
            o = realized_outcome(sig, i, is_down)
            if o is not None and not o["partial_window"] and o["outcome"] == "V자반등":
                named_v_rebound.add((i, is_down))

    events = pd.read_csv(EVENTS_CSV)
    events = events[events["outcome"] == "V자반등"].copy()
    local_extreme_v_rebound = set()
    for _, row in events.iterrows():
        idx = ts_to_idx.get(pd.Timestamp(row["timestamp_utc"]))
        if idx is None:
            continue
        local_extreme_v_rebound.add((idx, row["direction"] == "downside"))

    union_all = named_v_rebound | local_extreme_v_rebound
    only_named = named_v_rebound - local_extreme_v_rebound
    only_local = local_extreme_v_rebound - named_v_rebound
    both = named_v_rebound & local_extreme_v_rebound

    print(f"분석구간: {analysis_start.date()} ~ {now_utc.date()} ({WINDOW_DAYS}일)\n")
    print(f"sweep+5개신호 합집합(중복제거) V자반등: {len(named_v_rebound)}건")
    print(f"로컬극값 방식 V자반등(sweep포함, 참고): {len(local_extreme_v_rebound)}건")
    print(f"두 방식 전체 합집합(진짜 총합 추정치): {len(union_all)}건")
    print(f"  - 두 방식 다 잡음(교집합): {len(both)}건")
    print(f"  - named신호만 잡고 로컬극값은 놓침: {len(only_named)}건")
    print(f"  - 로컬극값만 잡고 named신호는 놓침: {len(only_local)}건")
    print(f"\nsweep+5개신호만으로 전체 추정치의 {len(named_v_rebound)/len(union_all)*100:.1f}% 커버")
    print(f"(로컬극값 방식이 여기에 추가로 더하는 몫: {len(only_local)/len(union_all)*100:.1f}%)")

    (OUT_DIR / "named_trigger_union_coverage.json").write_text(json.dumps({
        "named_v_rebound": len(named_v_rebound),
        "local_extreme_v_rebound": len(local_extreme_v_rebound),
        "union_all": len(union_all),
        "both": len(both),
        "only_named": len(only_named),
        "only_local": len(only_local),
        "named_coverage_pct": round(len(named_v_rebound) / len(union_all) * 100, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n실행시간: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
