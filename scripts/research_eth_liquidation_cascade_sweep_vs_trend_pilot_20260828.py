#!/usr/bin/env python3
"""Pilot: does OI trajectory / spot-futures CVD divergence / candle wick-body shape / order-book
absorption distinguish a "sweep-then-reverse" liquidation cascade from a "continuation" one, over
the past ~3 days of ETH data? Design doc:
docs/experiments/eth_liquidation_cascade_sweep_vs_trend_pilot_design_20260828.md

Descriptive/pilot event-study, NOT a promotion-grade or Fresh-Forward test (CLAUDE.md's
Fresh-Forward rule requires a full pre-registered walk-forward split; this is a 3-day retrospective
event study with an expected small N). Never wire into trading_bot.py or use as promotion evidence.

Runs entirely on dev against the frozen snapshot pulled from the server
(data/research/eth_liquidation_cascade_sweep_vs_trend_pilot_20260828/*.csv, extracted by
research_eth_liquidation_cascade_sweep_vs_trend_pilot_20260828_extract.py) plus freshly-fetched
public Binance klines (no server/internal-network dependency for those).
"""
from __future__ import annotations

import json
import time as time_module
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "research" / "eth_liquidation_cascade_sweep_vs_trend_pilot_20260828"
OUT_DIR = DATA_DIR  # reuse same dir for outputs

WINDOW_START_UTC = pd.Timestamp("2026-07-18 12:00:00", tz="UTC")  # tail_risk_1m valid-epoch start (see extract script docstring)

SWEEP_LOOKBACK_BARS = 48   # 4h at 5m -- reused verbatim from live_evidence_signal_dashboard_20260823.py
BUF_PCT = 0.005            # 0.5% -- reused verbatim from the liquidation-map S/R backtest convention
HORIZONS_BARS = {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48}
PRIMARY_HORIZON = "1h"
FEATURE_WINDOW_BARS = 3    # 15min post-t0, matches the shortest horizon (documented overlap, see report)


# ── klines fetch (public REST, no server needed) ────────────────────────────────────────────────

def fetch_klines(base_url: str, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    out = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(base_url, params={"symbol": symbol, "interval": "5m", "startTime": cursor,
                                                "endTime": end_ms, "limit": 1000}, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        out.extend(batch)
        last_open = int(batch[-1][0])
        if last_open <= cursor:
            break
        cursor = last_open + 1
        if len(batch) < 1000:
            break
        time_module.sleep(0.15)
    df = pd.DataFrame(out, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["taker_buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    return df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_ratio"]]


# ── Definition A: causal replay of the real TailRiskInterceptor hawkes trigger ──────────────────

def replay_definition_a(tail_risk: pd.DataFrame) -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(ROOT))
    import tail_risk_interceptor as tri

    interceptor = tri.TailRiskInterceptor(symbol="ethusdt")
    events = []
    for row in tail_risk.itertuples():
        ts_epoch = row.ts.timestamp()
        prev_active = interceptor._hawkes_active
        prev_crisis = interceptor._crisis_type
        with patch.object(tri.time, "time", return_value=ts_epoch):
            interceptor._update_hawkes_state(float(row.long_usd_1m), float(row.short_usd_1m))
        is_onset = interceptor._hawkes_active and ((not prev_active) or (interceptor._crisis_type != prev_crisis))
        if is_onset:
            events.append({
                "event_id": f"A_{row.ts.isoformat()}",
                "definition": "A_hawkes",
                "t0": row.ts,
                "crisis_type": interceptor._crisis_type,
                "z_peak": interceptor._peak_liq_intensity,
                "peak_usd": float(row.long_usd_1m) if interceptor._crisis_type == "LONG_CRISIS" else float(row.short_usd_1m),
            })
        interceptor._history_long.append(float(row.long_usd_1m))
        interceptor._history_short.append(float(row.short_usd_1m))
        interceptor._recalculate_stats()
    return pd.DataFrame(events)


def load_definition_b(l2_events: pd.DataFrame) -> pd.DataFrame:
    df = l2_events.copy()
    df["t0"] = pd.to_datetime(df["triggered_at_kst"], utc=True)
    # price_move_pct_60s sign tells direction: negative move = price fell = LONG_CRISIS-equivalent
    df["crisis_type"] = np.where(df["price_move_pct_60s"] < 0, "LONG_CRISIS", "SHORT_CRISIS")
    df["event_id"] = "B_" + df["event_id"]
    df["definition"] = "B_l2anomaly"
    df["z_peak"] = df["liq_z"]
    df["peak_usd"] = df["liq_burst_usd_60s"]
    return df[["event_id", "definition", "t0", "crisis_type", "z_peak", "peak_usd",
               "liquidity_withdrawal_matched"]]


# ── Triple-barrier labeling ──────────────────────────────────────────────────────────────────────

def label_events(events: pd.DataFrame, kl: pd.DataFrame) -> pd.DataFrame:
    kl = kl.reset_index(drop=True)
    ts = kl["timestamp"]
    max_idx = len(kl) - 1
    rows = []
    for ev in events.itertuples():
        idx_arr = ts.searchsorted(ev.t0, side="right") - 1
        if idx_arr < SWEEP_LOOKBACK_BARS or idx_arr < 0 or idx_arr > max_idx:
            continue
        t0_idx = int(idx_arr)
        direction = "down" if ev.crisis_type == "LONG_CRISIS" else "up"

        pre = kl.iloc[t0_idx - SWEEP_LOOKBACK_BARS: t0_idx]
        swept_level = pre["low"].min() if direction == "down" else pre["high"].max()
        cascade_extreme = kl["low"].iloc[t0_idx] if direction == "down" else kl["high"].iloc[t0_idx]

        max_h = HORIZONS_BARS[max(HORIZONS_BARS, key=HORIZONS_BARS.get)]
        fwd_end = min(t0_idx + max_h, max_idx)
        censored_at = fwd_end - t0_idx  # bars actually available forward

        resolution, bars_to_resolution = None, None
        for j in range(t0_idx + 1, fwd_end + 1):
            bar = kl.iloc[j]
            n = j - t0_idx
            if direction == "down":
                continuation_hit = bar["low"] <= cascade_extreme * (1 - BUF_PCT)
                reversal_hit = bar["close"] >= swept_level * (1 + BUF_PCT)
            else:
                continuation_hit = bar["high"] >= cascade_extreme * (1 + BUF_PCT)
                reversal_hit = bar["close"] <= swept_level * (1 - BUF_PCT)
            if continuation_hit:  # intrabar precheck first, matches CLAUDE.md barrier convention
                resolution, bars_to_resolution = "continuation", n
                break
            if reversal_hit:
                resolution, bars_to_resolution = "sweep", n
                break

        rec = {
            "event_id": ev.event_id, "definition": ev.definition, "t0": ev.t0,
            "crisis_type": ev.crisis_type, "direction": direction, "z_peak": ev.z_peak,
            "peak_usd": ev.peak_usd, "t0_idx": t0_idx, "swept_level": swept_level,
            "cascade_extreme": cascade_extreme, "censored_bars_available": censored_at,
        }
        for h_name, h_bars in HORIZONS_BARS.items():
            if censored_at < h_bars:
                rec[f"label_{h_name}"] = "censored"
            elif resolution is not None and bars_to_resolution <= h_bars:
                rec[f"label_{h_name}"] = resolution
            else:
                rec[f"label_{h_name}"] = "ambiguous"
        rows.append(rec)
    return pd.DataFrame(rows)


# ── Feature extraction (causal: only [t0, t0+FEATURE_WINDOW_BARS] or t0 bar itself) ─────────────

def _nearest_at_or_before(df: pd.DataFrame, ts_col: str, target: pd.Timestamp):
    sub = df[df[ts_col] <= target]
    return sub.iloc[-1] if len(sub) else None


def _nearest_at_or_after(df: pd.DataFrame, ts_col: str, target: pd.Timestamp):
    sub = df[df[ts_col] >= target]
    return sub.iloc[0] if len(sub) else None


def extract_features(labeled: pd.DataFrame, fut_kl: pd.DataFrame, spot_kl: pd.DataFrame,
                      oi_df: pd.DataFrame, ob_df: pd.DataFrame, micro_df: pd.DataFrame) -> pd.DataFrame:
    fut_kl = fut_kl.reset_index(drop=True)
    feats = []
    for ev in labeled.itertuples():
        t0 = ev.t0
        t0_idx = ev.t0_idx
        direction = ev.direction
        window_end_idx = min(t0_idx + FEATURE_WINDOW_BARS, len(fut_kl) - 1)
        t_feat_end = fut_kl["timestamp"].iloc[window_end_idx]

        # ③ candle shape (t0 bar itself -- fully determined at bar close, no forward leakage)
        bar = fut_kl.iloc[t0_idx]
        body = abs(bar["close"] - bar["open"])
        body = max(body, 1e-9)
        if direction == "down":
            wick_in_direction = min(bar["open"], bar["close"]) - bar["low"]
        else:
            wick_in_direction = bar["high"] - max(bar["open"], bar["close"])
        wick_body_ratio = wick_in_direction / body

        # ① OI trajectory (post-t0, [t0, t0+15m])
        oi_start = _nearest_at_or_before(oi_df, "ts", t0)
        oi_end = _nearest_at_or_after(oi_df, "ts", t_feat_end)
        if oi_start is not None and oi_end is not None and oi_start["sum_open_interest"] > 0:
            oi_pct_change = (oi_end["sum_open_interest"] - oi_start["sum_open_interest"]) / oi_start["sum_open_interest"]
            ls_shift = oi_end["global_ls_ratio"] - oi_start["global_ls_ratio"]
            same_dir_expansion = bool((oi_pct_change > 0) and (
                (direction == "down" and ls_shift < 0) or (direction == "up" and ls_shift > 0)))
        else:
            oi_pct_change, ls_shift, same_dir_expansion = None, None, None

        # ② spot-futures CVD divergence (post-t0, [t0, t0+15m], 5m-kline taker-buy-ratio proxy)
        fut_win = fut_kl.iloc[t0_idx + 1: window_end_idx + 1]
        spot_win = spot_kl[(spot_kl["timestamp"] > t0) & (spot_kl["timestamp"] <= t_feat_end)]
        fut_ratio = fut_win["taker_buy_ratio"].mean() if len(fut_win) else None
        spot_ratio = spot_win["taker_buy_ratio"].mean() if len(spot_win) else None
        if fut_ratio is not None and spot_ratio is not None and not (np.isnan(fut_ratio) or np.isnan(spot_ratio)):
            cvd_divergence = abs((fut_ratio - 0.5) - (spot_ratio - 0.5))
            cvd_sign_agree = bool(np.sign(fut_ratio - 0.5) == np.sign(spot_ratio - 0.5))
        else:
            cvd_divergence, cvd_sign_agree = None, None

        # ④ order-book absorption (3 sources)
        ob_pre = _nearest_at_or_before(ob_df, "recorded_at_kst", t0)
        ob_post = _nearest_at_or_after(ob_df, "recorded_at_kst", t_feat_end)
        if ob_pre is not None and ob_post is not None:
            pre_notional = ob_pre["bid_notional_10"] + ob_pre["ask_notional_10"]
            post_notional = ob_post["bid_notional_10"] + ob_post["ask_notional_10"]
            book_notional_pct_change = (post_notional - pre_notional) / pre_notional if pre_notional > 0 else None
        else:
            book_notional_pct_change = None

        micro_win = micro_df[(micro_df["ts"] >= t0) & (micro_df["ts"] <= t_feat_end)]
        shadow_queue_collapse_max = micro_win["shadow_queue_collapse"].max() if len(micro_win) else None
        shadow_absorption_score_mean = micro_win["shadow_absorption_score"].mean() if len(micro_win) else None

        feats.append({
            "event_id": ev.event_id,
            "wick_body_ratio": wick_body_ratio,
            "oi_pct_change_15m": oi_pct_change,
            "ls_ratio_shift_15m": ls_shift,
            "oi_same_dir_expansion": same_dir_expansion,
            "fut_taker_ratio_15m": fut_ratio,
            "spot_taker_ratio_15m": spot_ratio,
            "cvd_divergence_15m": cvd_divergence,
            "cvd_sign_agree_15m": cvd_sign_agree,
            "book_notional10_pct_change": book_notional_pct_change,
            "shadow_queue_collapse_max_15m": shadow_queue_collapse_max,
            "shadow_absorption_score_mean_15m": shadow_absorption_score_mean,
        })
    return labeled.merge(pd.DataFrame(feats), on="event_id", how="left")


def main() -> None:
    tail_risk = pd.read_csv(DATA_DIR / "tail_risk_1m.csv", parse_dates=["ts"])
    tail_risk["ts"] = pd.to_datetime(tail_risk["ts"], utc=True)
    n_before = len(tail_risk)
    tail_risk = tail_risk[(tail_risk["valid_liq_stream"] == True) & (tail_risk["ws_stale"] != True)]  # noqa: E712
    tail_risk = tail_risk.sort_values("ts").reset_index(drop=True)
    print(f"tail_risk_1m: {n_before} rows -> {len(tail_risk)} after valid_liq_stream/ws_stale filter")

    oi_df = pd.read_csv(DATA_DIR / "oi_lsratio_5m.csv", parse_dates=["ts"])
    oi_df["ts"] = pd.to_datetime(oi_df["ts"], utc=True)
    oi_df = oi_df.sort_values("ts").reset_index(drop=True)

    ob_df = pd.read_csv(DATA_DIR / "orderbook_decision_snapshots.csv", parse_dates=["recorded_at_kst"])
    ob_df["recorded_at_kst"] = pd.to_datetime(ob_df["recorded_at_kst"], utc=True)
    ob_df = ob_df.sort_values("recorded_at_kst").reset_index(drop=True)

    micro_df = pd.read_csv(DATA_DIR / "microstructure_1m.csv", parse_dates=["ts"])
    micro_df["ts"] = pd.to_datetime(micro_df["ts"], utc=True)
    micro_df = micro_df.sort_values("ts").reset_index(drop=True)

    l2_events = pd.read_csv(DATA_DIR / "l2_anomaly_events.csv")

    start_ms = int(WINDOW_START_UTC.timestamp() * 1000)
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    print("fetching futures 5m klines...")
    fut_kl = fetch_klines("https://fapi.binance.com/fapi/v1/klines", "ETHUSDT", start_ms, end_ms)
    print(f"  {len(fut_kl)} futures bars, {fut_kl['timestamp'].min()} -> {fut_kl['timestamp'].max()}")
    print("fetching spot 5m klines...")
    spot_kl = fetch_klines("https://api.binance.com/api/v3/klines", "ETHUSDT", start_ms, end_ms)
    print(f"  {len(spot_kl)} spot bars, {spot_kl['timestamp'].min()} -> {spot_kl['timestamp'].max()}")
    fut_kl.to_csv(OUT_DIR / "futures_5m_klines.csv", index=False)
    spot_kl.to_csv(OUT_DIR / "spot_5m_klines.csv", index=False)

    print("replaying Definition A (hawkes cascade, causal)...")
    events_a = replay_definition_a(tail_risk)
    print(f"  {len(events_a)} onset events")
    events_a.to_csv(OUT_DIR / "events_definition_a.csv", index=False)

    events_b = load_definition_b(l2_events)
    print(f"Definition B: {len(events_b)} events")

    # cross-reference
    if len(events_a) and len(events_b):
        a_ts = pd.to_datetime(events_a["t0"], utc=True)
        overlap = 0
        for t0b in pd.to_datetime(events_b["t0"], utc=True):
            if ((a_ts - t0b).abs() <= pd.Timedelta(seconds=300)).any():
                overlap += 1
        print(f"cross-reference: {overlap}/{len(events_b)} Definition-B events fall within +/-5min of a Definition-A onset")

    print("labeling (triple-barrier)...")
    labeled_a = label_events(events_a, fut_kl)
    labeled_b = label_events(events_b.rename(columns={}), fut_kl)
    print(f"  A: {len(labeled_a)} labeled (of {len(events_a)}), B: {len(labeled_b)} labeled (of {len(events_b)})")

    print("extracting features...")
    full_a = extract_features(labeled_a, fut_kl, spot_kl, oi_df, ob_df, micro_df)
    full_b = extract_features(labeled_b, fut_kl, spot_kl, oi_df, ob_df, micro_df)
    if len(full_b) and "liquidity_withdrawal_matched" in events_b.columns:
        full_b = full_b.merge(events_b[["event_id", "liquidity_withdrawal_matched"]], on="event_id", how="left")

    full_a.to_csv(OUT_DIR / "labeled_features_definition_a.csv", index=False)
    full_b.to_csv(OUT_DIR / "labeled_features_definition_b.csv", index=False)
    print(f"saved: {OUT_DIR / 'labeled_features_definition_a.csv'}")
    print(f"saved: {OUT_DIR / 'labeled_features_definition_b.csv'}")

    print("\n=== label distribution, Definition A, primary horizon (1h) ===")
    print(full_a[f"label_{PRIMARY_HORIZON}"].value_counts())
    print("\n=== label distribution, Definition B, primary horizon (1h) ===")
    if len(full_b):
        print(full_b[f"label_{PRIMARY_HORIZON}"].value_counts())


if __name__ == "__main__":
    main()
