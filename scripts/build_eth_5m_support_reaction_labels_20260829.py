#!/usr/bin/env python3
"""Build support-reaction labels from aligned ETHUSDT 5m futures, spot, and OI data.

This follows the supplied classify_support_reaction() contract for a lower support:
  0 = SWEEP_V_BOUNCE
  1 = SUPPORT_BREAKOUT

Only bars that touch the prior 48-bar support are candidates. All other bars and all failures
of the dead-cat filter are excluded. Future bars are used only for outcome labeling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2024-01-01", tz="UTC")
LABELS = {"SWEEP_V_BOUNCE": 0, "SUPPORT_BREAKOUT": 1}
BAR_MINUTES = 5
SUPPORT_LOOKBACK_BARS = 48
LOOKAHEAD_BARS = 12
INITIAL_FLOW_BARS = 3
OI_SWEEP_THRESHOLD = -0.0015
OI_BREAKOUT_THRESHOLD = 0.002
SWEEP_WICK_THRESHOLD = 0.4
SWEEP_HOLD_RATE = 0.66
BREAKOUT_HOLD_RATE = 0.75
MATCH_TOLERANCE_BARS = 2
ATR_N = 14
SUPPORT_PROXIMITY_ATR = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--futures", type=Path, default=ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
    parser.add_argument("--spot", type=Path, default=ROOT / "binance_data/klines_spot/ETHUSDT/ETHUSDT-5m-spot.csv")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "binance_data/metrics")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/labels/eth_5m_support_reaction_20260829")
    return parser.parse_args()


def load_klines(path: Path, prefix: str) -> pd.DataFrame:
    columns = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    frame = frame.loc[frame["timestamp"] >= START].copy()
    frame[f"{prefix}_delta"] = 2.0 * frame["taker_buy_base"] - frame["volume"]
    return frame[["timestamp", "open", "high", "low", "close", f"{prefix}_delta"]].rename(
        columns={column: f"{prefix}_{column}" for column in ["open", "high", "low", "close"]}
    )


def load_oi(metrics_dir: Path) -> pd.DataFrame:
    paths = sorted(metrics_dir.glob("ETHUSDT-metrics-*.zip"))
    if not paths:
        raise RuntimeError(f"no ETHUSDT metrics zip files under {metrics_dir}")
    parts = []
    for path in paths:
        part = pd.read_csv(path, compression="zip", usecols=["create_time", "sum_open_interest"], low_memory=False)
        part["timestamp"] = pd.to_datetime(part["create_time"], utc=True, errors="coerce")
        part["oi"] = pd.to_numeric(part["sum_open_interest"], errors="coerce")
        parts.append(part[["timestamp", "oi"]])
    return pd.concat(parts, ignore_index=True).dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def load_aligned(options: argparse.Namespace) -> pd.DataFrame:
    futures = load_klines(options.futures, "fut")
    spot = load_klines(options.spot, "spot")
    oi = load_oi(options.metrics_dir)
    frame = futures.merge(spot, on="timestamp", how="inner").merge(oi, on="timestamp", how="inner")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    current_bar_start = pd.Timestamp.now(tz="UTC").floor("5min")
    frame = frame.loc[frame["timestamp"] < current_bar_start].reset_index(drop=True)
    frame["support_level"] = frame["fut_low"].shift(1).rolling(SUPPORT_LOOKBACK_BARS, min_periods=SUPPORT_LOOKBACK_BARS).min()
    candle_range = (frame["fut_high"] - frame["fut_low"]).clip(lower=1e-12)
    frame["lower_wick_ratio"] = (frame[["fut_open", "fut_close"]].min(axis=1) - frame["fut_low"]) / candle_range
    previous_close = frame["fut_close"].shift(1)
    true_range = pd.concat([
        frame["fut_high"] - frame["fut_low"],
        (frame["fut_high"] - previous_close).abs(),
        (frame["fut_low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    frame["atr"] = true_range.rolling(ATR_N, min_periods=ATR_N).mean()
    return frame


def contiguous(timestamps: np.ndarray, index: int) -> bool:
    end = index + LOOKAHEAD_BARS
    if end >= len(timestamps):
        return False
    expected = timestamps[index] + pd.to_timedelta(np.arange(LOOKAHEAD_BARS + 1) * BAR_MINUTES, unit="min")
    return bool(np.array_equal(timestamps[index:end + 1], expected))


def candidate(frame: pd.DataFrame, timestamps: np.ndarray, index: int) -> list[dict]:
    if index < SUPPORT_LOOKBACK_BARS or index + LOOKAHEAD_BARS >= len(frame) or not contiguous(timestamps, index):
        return []
    row = frame.iloc[index]
    support = row["support_level"]
    oi_open = row["oi"]
    atr = row["atr"]
    if not np.isfinite(support) or not np.isfinite(atr) or not np.isfinite(oi_open) or oi_open <= 0:
        return []

    # Candidate must be near support. The supplied function assumes P_support is already selected;
    # this explicit proximity gate prevents ordinary, distant candles becoming breakout labels.
    if row["fut_low"] > support + SUPPORT_PROXIMITY_ATR * atr:
        return []
    future = frame.iloc[index + 1:index + LOOKAHEAD_BARS + 1]
    oi_change = (frame.iloc[index + INITIAL_FLOW_BARS]["oi"] - oi_open) / oi_open
    cum_spot_delta = float(frame.iloc[index:index + INITIAL_FLOW_BARS + 1]["spot_delta"].sum())
    cum_fut_delta = float(frame.iloc[index:index + INITIAL_FLOW_BARS + 1]["fut_delta"].sum())
    hold_rate = float((future["fut_close"] > support).sum() / LOOKAHEAD_BARS)
    is_t0_close_above = bool(row["fut_close"] >= support)
    is_pierced = bool(row["fut_low"] < support)
    lower_wick_ratio = float(row["lower_wick_ratio"])

    # Dead-cat bounce filter from the supplied contract.
    if cum_spot_delta < 0 and cum_fut_delta > 0:
        return []

    is_sweep_oi = oi_change < OI_SWEEP_THRESHOLD
    is_sweep_price = is_pierced and lower_wick_ratio >= SWEEP_WICK_THRESHOLD
    sweep_v_bounce = is_sweep_oi and is_sweep_price and hold_rate >= SWEEP_HOLD_RATE

    is_breakout_oi = oi_change > OI_BREAKOUT_THRESHOLD
    is_strong_spot = cum_spot_delta > 0
    is_breakout_price = (not is_pierced) and is_t0_close_above and lower_wick_ratio < SWEEP_WICK_THRESHOLD
    support_breakout = is_breakout_oi and is_strong_spot and is_breakout_price and hold_rate >= BREAKOUT_HOLD_RATE

    rows = []
    for event_type, matched in (("SWEEP_V_BOUNCE", sweep_v_bounce), ("SUPPORT_BREAKOUT", support_breakout)):
        if matched:
            rows.append({
                "candidate_index": index,
                "timestamp": pd.Timestamp(timestamps[index]).isoformat(),
                "side": "lower_support",
                "event_type": event_type,
                "label": LABELS[event_type],
                "support_level": float(support),
                "atr": float(atr),
                "oi_delta_pct_15m": float(oi_change),
                "cum_spot_delta_15m": cum_spot_delta,
                "cum_fut_delta_15m": cum_fut_delta,
                "hold_rate_60m": hold_rate,
                "is_pierced": is_pierced,
                "lower_wick_ratio": lower_wick_ratio,
                "dead_cat_filtered": False,
            })
    return rows


def generate(frame: pd.DataFrame) -> list[dict]:
    timestamps = frame["timestamp"].to_numpy()
    rows = []
    for index in range(SUPPORT_LOOKBACK_BARS, len(frame) - LOOKAHEAD_BARS):
        rows.extend(candidate(frame, timestamps, index))
    return rows


def resolve(candidates: list[dict], frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not candidates:
        return pd.DataFrame(), pd.DataFrame()
    candidates = sorted(candidates, key=lambda row: (row["candidate_index"], row["event_type"]))
    clusters: list[list[dict]] = []
    for row in candidates:
        if not clusters or row["candidate_index"] - clusters[-1][-1]["candidate_index"] > MATCH_TOLERANCE_BARS:
            clusters.append([row])
        else:
            clusters[-1].append(row)
    labels, overlaps = [], []
    timestamps = frame["timestamp"]
    for event_id, cluster in enumerate(clusters, 1):
        types = sorted({row["event_type"] for row in cluster})
        anchor = min(cluster, key=lambda row: row["candidate_index"])
        start = timestamps.iloc[max(0, anchor["candidate_index"] - MATCH_TOLERANCE_BARS)].isoformat()
        end = timestamps.iloc[min(len(frame) - 1, anchor["candidate_index"] + MATCH_TOLERANCE_BARS)].isoformat()
        if len(types) != 1:
            for row in cluster:
                overlaps.append({**row, "event_id": event_id, "overlap_types": ",".join(types), "match_window_start": start, "match_window_end": end})
        else:
            labels.append({**anchor, "event_id": event_id, "match_window_start": start, "match_window_end": end, "overlap_excluded": False})
    return pd.DataFrame(labels), pd.DataFrame(overlaps)


def main() -> int:
    options = parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_aligned(options)
    candidates = generate(frame)
    labels, overlaps = resolve(candidates, frame)
    label_path = options.output_dir / "eth_5m_support_reaction_labels.csv"
    overlap_path = options.output_dir / "eth_5m_support_reaction_overlaps.csv"
    report_path = options.output_dir / "report.json"
    labels.to_csv(label_path, index=False)
    overlaps.to_csv(overlap_path, index=False)
    report = {
        "label_contract": LABELS,
        "source_period": {"start": str(frame["timestamp"].min()), "end": str(frame["timestamp"].max()), "aligned_5m_bars": int(len(frame))},
        "candidate_rows": len(candidates),
        "label_rows": len(labels),
        "overlap_rows": len(overlaps),
        "label_counts": {name: int((labels["event_type"] == name).sum()) if not labels.empty else 0 for name in LABELS},
        "overlap_counts": {name: int((overlaps["event_type"] == name).sum()) if not overlaps.empty else 0 for name in LABELS},
        "parameters": {
            "bar_minutes": BAR_MINUTES,
            "support_lookback_bars": SUPPORT_LOOKBACK_BARS,
            "initial_flow_minutes": INITIAL_FLOW_BARS * BAR_MINUTES,
            "hold_minutes": LOOKAHEAD_BARS * BAR_MINUTES,
            "oi_sweep_threshold_lt": OI_SWEEP_THRESHOLD,
            "oi_breakout_threshold_gt": OI_BREAKOUT_THRESHOLD,
            "wick_threshold": SWEEP_WICK_THRESHOLD,
            "sweep_hold_rate_gte": SWEEP_HOLD_RATE,
            "breakout_hold_rate_gte": BREAKOUT_HOLD_RATE,
            "support_proximity_atr_lte": SUPPORT_PROXIMITY_ATR,
            "match_tolerance_minutes": MATCH_TOLERANCE_BARS * BAR_MINUTES,
        },
        "dead_cat_filter": "cum_spot_delta_15m < 0 and cum_fut_delta_15m > 0 => NONE",
        "alignment_policy": "inner join futures/spot/OI on UTC 5m timestamps; missing inputs excluded",
        "future_features_used_for_labels": True,
        "output_labels": str(label_path),
        "output_overlaps": str(overlap_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
