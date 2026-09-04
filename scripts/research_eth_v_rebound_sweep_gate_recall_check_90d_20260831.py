#!/usr/bin/env python3
"""90-day extension of research_eth_v_rebound_sweep_gate_recall_check_20260831.py: quantify the
'genuine V-shaped reversal happened but no liquidity_sweep triggered' gap over a recent multi-
month window instead of one morning. Same v7b outcome formula, same sweep definition, reused not
reimplemented (see that script's docstring for full formula provenance).

Data: local binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv (fast, covers through 2026-08-28) +
one live API fetch to fill the ~2-3 day gap to now, concatenated. Window: last 90 days before now
(with a 5-day lookback margin prepended purely for indicator warmup, not scored).

Reports AGGREGATE stats (this window has hundreds-thousands of events, not a handful) + saves the
full per-event table to tmp/ for follow-up visual spot-checking.

2026-08-31 result (2026-06-01 -> 2026-08-30, 90 days): 1,247 sweep events (204 V자반등/314
지지횡보/729 애매), 2,281 non-sweep local-extreme candidates (491 met V자반등 criteria at a
HIGHER rate, 21.5% vs sweep's 16.4%). Of all 695 genuine V자반등-quality moves, 70.6% (491) had
NO sweep trigger -- structurally invisible to the current live signal. Quality is statistically
identical between groups (median fast_move_atr_mult 2.59x vs 2.58x, giveback 0.120 vs 0.117) --
the missed ones are not weaker signals, purely a gating-design gap. See memory:
eth_v_rebound_sweep_gated_recall_gap_20260831 for full writeup and recommended next steps.
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

IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
LOCAL_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp/eth_v_rebound_sweep_gate_recall_check_20260831"

FAST_BARS = 6
FULL_BARS = 12
ATR_MULT = 1.5
CHOP_MULT = 1.0
T_SUSTAIN = 0.20
WINDOW_DAYS = 90
LOOKBACK_MARGIN_DAYS = 5
W = 6  # local-extreme detection window (+-30min), same scale as FAST_BARS


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_audit90d_20260831", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_live_klines_trimmed(limit=1500) -> pd.DataFrame:
    resp = requests.get(
        "https://fapi.binance.com/fapi/v1/klines",
        params={"symbol": "ETHUSDT", "interval": "5m", "limit": limit}, timeout=20,
    )
    resp.raise_for_status()
    raw = resp.json()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "trades",
            "taker_buy_base", "tq", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    now_ms = int(time.time() * 1000)
    if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
        df = df.iloc[:-1]
    return df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]].reset_index(drop=True)


def load_combined(lookback_start: pd.Timestamp) -> pd.DataFrame:
    local = pd.read_csv(LOCAL_CSV, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
    local = local[local["timestamp"] >= lookback_start]
    live = fetch_live_klines_trimmed(1500)
    combined = pd.concat([local, live], ignore_index=True)
    combined = combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return combined


def build_frame(kl: pd.DataFrame, impl) -> pd.DataFrame:
    causal = impl.add_causal_columns(kl[["timestamp", "open", "high", "low", "close"]].copy())
    frame = kl.copy()
    frame["sweep_level_low"] = causal["sweep_level_low"]
    frame["sweep_level_high"] = causal["sweep_level_high"]
    frame["atr"] = causal["atr"]
    return frame.reset_index(drop=True)


def realized_outcome(frame: pd.DataFrame, idx: int, is_down: bool) -> dict | None:
    n = len(frame)
    pre_atr = frame["atr"].iloc[idx - 1]
    if not np.isfinite(pre_atr) or pre_atr <= 0:
        return None
    extreme = frame["low"].iloc[idx] if is_down else frame["high"].iloc[idx]
    fast_slice = frame.iloc[idx + 1: idx + FAST_BARS + 1]
    full_slice = frame.iloc[idx + 1: idx + FULL_BARS + 1]
    if fast_slice.empty:
        return None
    partial = (idx + FULL_BARS) > (n - 1)

    if is_down:
        fast_move = fast_slice["close"].max() - extreme
        peak = full_slice["high"].max() if not full_slice.empty else fast_slice["high"].max()
    else:
        fast_move = extreme - fast_slice["close"].min()
        peak = full_slice["low"].min() if not full_slice.empty else fast_slice["low"].min()

    fast_mult = float(fast_move / pre_atr)
    end_price = float(full_slice["close"].iloc[-1]) if not full_slice.empty else float(fast_slice["close"].iloc[-1])
    denom = (peak - extreme) if is_down else (extreme - peak)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        giveback = np.nan
    else:
        giveback = ((peak - end_price) / denom) if is_down else ((end_price - peak) / denom)

    if fast_mult >= ATR_MULT and np.isfinite(giveback) and giveback <= T_SUSTAIN:
        label = "V자반등"
    elif fast_mult < CHOP_MULT:
        label = "지지/횡보"
    else:
        label = "애매(제외권)"

    return {
        "fast_move_atr_mult": round(fast_mult, 3),
        "giveback_ratio": None if not np.isfinite(giveback) else round(float(giveback), 3),
        "outcome": label,
        "partial_window": bool(partial),
    }


def main() -> None:
    t0 = time.time()
    impl = load_impl()
    now_utc = pd.Timestamp.now(tz="UTC")
    analysis_start = now_utc - pd.Timedelta(days=WINDOW_DAYS)
    lookback_start = analysis_start - pd.Timedelta(days=LOOKBACK_MARGIN_DAYS)

    kl = load_combined(lookback_start)
    frame = build_frame(kl, impl)
    print(f"로드: {len(frame)}개 5분봉, {frame['timestamp'].iloc[0]} ~ {frame['timestamp'].iloc[-1]} UTC "
          f"(분석구간 {analysis_start.date()} ~ {now_utc.date()}, 웜업여유 {LOOKBACK_MARGIN_DAYS}일 별도)")

    window = frame[frame["timestamp"] >= analysis_start]
    n_days = (now_utc - analysis_start).total_seconds() / 86400

    low, high, close = frame["low"], frame["high"], frame["close"]
    level_low, level_high = frame["sweep_level_low"], frame["sweep_level_high"]
    is_down_sweep = level_low.notna() & (low < level_low) & (close > level_low)
    is_up_sweep = level_high.notna() & (high > level_high) & (close < level_high)

    # --- Part 1: sweep-triggered events ---
    sweep_idx = [i for i in window.index if bool(is_down_sweep.iloc[i]) or bool(is_up_sweep.iloc[i])]
    part1 = []
    for idx in sweep_idx:
        is_down = bool(is_down_sweep.iloc[idx])
        o = realized_outcome(frame, idx, is_down)
        if o is None or o["partial_window"]:
            continue
        part1.append({"idx": idx, "timestamp_utc": frame["timestamp"].iloc[idx].isoformat(),
                       "group": "sweep", "direction": "downside" if is_down else "upside", **o})

    # --- Part 2: non-sweep local-extreme candidates ---
    lo, hi = frame["low"].to_numpy(), frame["high"].to_numpy()
    n = len(frame)
    part2 = []
    n_candidates = 0
    for i in window.index:
        if i < W or i + W >= n:
            continue
        if bool(is_down_sweep.iloc[i]) or bool(is_up_sweep.iloc[i]):
            continue
        window_lo, window_hi = lo[i - W:i + W + 1], hi[i - W:i + W + 1]
        for is_local, is_down in ((lo[i] == window_lo.min(), True), (hi[i] == window_hi.max(), False)):
            if not is_local:
                continue
            n_candidates += 1
            o = realized_outcome(frame, i, is_down)
            if o is None or o["partial_window"]:
                continue
            if o["outcome"] == "V자반등":
                part2.append({"idx": i, "timestamp_utc": frame["timestamp"].iloc[i].isoformat(),
                               "group": "no_sweep", "direction": "downside" if is_down else "upside", **o})

    p1_df = pd.DataFrame(part1)
    p2_df = pd.DataFrame(part2)

    def bucket_counts(df, col="outcome"):
        return df[col].value_counts().to_dict() if len(df) else {}

    sweep_counts = bucket_counts(p1_df)
    sweep_v = sweep_counts.get("V자반등", 0)
    sweep_total = len(p1_df)
    no_sweep_v = len(p2_df)

    print(f"\n=== 분석구간: 최근 {WINDOW_DAYS}일 ({n_days:.1f}일 실측), sweep이벤트 {sweep_total}건, "
          f"비sweep 로컬극값후보 {n_candidates}건 평가 ===\n")

    print("### Part 1: sweep 트리거 이벤트 결과분포 ###")
    for k in ("V자반등", "지지/횡보", "애매(제외권)"):
        n_k = sweep_counts.get(k, 0)
        pct = n_k / sweep_total * 100 if sweep_total else 0
        print(f"  {k:12s}: {n_k:5d}건 ({pct:5.1f}%)  -- {n_k/n_days:.2f}건/일")

    print(f"\n### Part 2: 비sweep 로컬극값 중 V자반등 기준 충족 ###")
    print(f"  V자반등(스윕없음): {no_sweep_v:5d}건  -- {no_sweep_v/n_days:.2f}건/일  "
          f"(평가한 {n_candidates}개 후보 중 {no_sweep_v/n_candidates*100:.1f}%)")

    total_v = sweep_v + no_sweep_v
    gap_pct = no_sweep_v / total_v * 100 if total_v else float("nan")
    print(f"\n### 핵심 비교 ###")
    print(f"  전체 '진짜 V자반등' 중 스윕 있었던 것: {sweep_v}건 ({100-gap_pct:.1f}%)")
    print(f"  전체 '진짜 V자반등' 중 스윕 없었던 것(현재 신호가 구조적으로 놓치는 몫): "
          f"{no_sweep_v}건 ({gap_pct:.1f}%)")

    if sweep_v and no_sweep_v:
        sweep_v_rows = p1_df[p1_df["outcome"] == "V자반등"]
        print(f"\n  품질 비교(fast_move_atr_mult / giveback_ratio, 중앙값):")
        print(f"    스윕있음: fast={sweep_v_rows['fast_move_atr_mult'].median():.2f}x  "
              f"giveback={sweep_v_rows['giveback_ratio'].median():.3f}")
        print(f"    스윕없음: fast={p2_df['fast_move_atr_mult'].median():.2f}x  "
              f"giveback={p2_df['giveback_ratio'].median():.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_events = pd.concat([p1_df, p2_df], ignore_index=True) if (len(p1_df) or len(p2_df)) else pd.DataFrame()
    all_events.to_csv(OUT_DIR / "events.csv", index=False)
    report = {
        "window_days": WINDOW_DAYS,
        "analysis_start_utc": str(analysis_start),
        "now_utc": str(now_utc),
        "n_bars_days_actual": round(n_days, 2),
        "sweep_total": sweep_total,
        "sweep_bucket_counts": sweep_counts,
        "no_sweep_candidates_evaluated": n_candidates,
        "no_sweep_v_rebound": no_sweep_v,
        "gap_pct_of_all_v_rebounds": None if total_v == 0 else round(gap_pct, 2),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n산출물: {OUT_DIR}/report.json, {OUT_DIR}/events.csv")
    print(f"실행시간: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
