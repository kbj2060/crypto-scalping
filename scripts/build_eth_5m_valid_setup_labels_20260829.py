#!/usr/bin/env python3
"""Build high-confidence ETHUSDT 5m SWEEP/BREAKOUT labels from aligned data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2024-01-01", tz="UTC")
LABELS = {"SWEEP": 0, "BREAKOUT": 1}
BAR_MINUTES = 5
LEVEL_LOOKBACK_BARS = 288
LOOKAHEAD_BARS = 12
MATCH_TOLERANCE_BARS = 2
BASIS_THRESHOLD = 0.5
SPOT_REVERSE_RATIO_MAX = 0.5
BREAKOUT_DELTA_OI_PCT_MIN = 0.002


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--futures", type=Path, default=ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
    parser.add_argument("--spot", type=Path, default=ROOT / "binance_data/klines_spot/ETHUSDT/ETHUSDT-5m-spot.csv")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "binance_data/metrics")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/labels/eth_5m_valid_setup_tuned_20260829")
    return parser.parse_args()


def load_klines(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"], low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
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
    frame = pd.concat(parts, ignore_index=True).dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame.loc[frame["timestamp"] >= START].copy()


def load_aligned(args: argparse.Namespace) -> pd.DataFrame:
    futures = load_klines(args.futures, "fut")
    spot = load_klines(args.spot, "spot")
    oi = load_oi(args.metrics_dir)
    frame = futures.merge(spot, on="timestamp", how="inner").merge(oi, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    current_bar_start = pd.Timestamp.now(tz="UTC").floor("5min")
    frame = frame.loc[frame["timestamp"] < current_bar_start].reset_index(drop=True)
    frame["level_high"] = frame["fut_high"].shift(1).rolling(LEVEL_LOOKBACK_BARS, min_periods=LEVEL_LOOKBACK_BARS).max()
    frame["level_low"] = frame["fut_low"].shift(1).rolling(LEVEL_LOOKBACK_BARS, min_periods=LEVEL_LOOKBACK_BARS).min()
    frame["basis"] = (frame["fut_close"] - frame["spot_close"]) / frame["spot_close"]
    frame["upper_wick_ratio"] = (frame["fut_high"] - frame[["fut_open", "fut_close"]].max(axis=1)) / (frame["fut_high"] - frame["fut_low"]).clip(lower=1e-12)
    frame["lower_wick_ratio"] = (frame[["fut_open", "fut_close"]].min(axis=1) - frame["fut_low"]) / (frame["fut_high"] - frame["fut_low"]).clip(lower=1e-12)
    return frame


def contiguous(timestamps: np.ndarray, index: int) -> bool:
    end = index + LOOKAHEAD_BARS
    if end >= len(timestamps):
        return False
    expected = timestamps[index] + pd.to_timedelta(np.arange(LOOKAHEAD_BARS + 1) * BAR_MINUTES, unit="min")
    return bool(np.array_equal(timestamps[index:end + 1], expected))


def candidate(c: dict[str, np.ndarray], ts: np.ndarray, index: int, side: str) -> list[dict]:
    level = c["level_high"][index] if side == "up" else c["level_low"][index]
    if not np.isfinite(level) or not contiguous(ts, index):
        return []
    previous_high, previous_low = c["fut_high"][index - 1], c["fut_low"][index - 1]
    high, low, close = c["fut_high"][index], c["fut_low"][index], c["fut_close"][index]
    future_close = c["fut_close"][index + 1:index + LOOKAHEAD_BARS + 1]
    future_low = c["fut_low"][index + 1:index + LOOKAHEAD_BARS + 1]
    future_high = c["fut_high"][index + 1:index + LOOKAHEAD_BARS + 1]
    oi_open = c["oi"][index]
    if oi_open <= 0:
        return []
    delta_oi_pct = (c["oi"][index + 3] - oi_open) / oi_open
    cum_spot_delta = c["spot_delta"][index:index + 4].sum()
    cum_fut_delta = c["fut_delta"][index:index + 4].sum()
    avg_basis = c["basis"][index + 1:index + 4].mean()
    if not np.isfinite(delta_oi_pct) or not np.isfinite(avg_basis):
        return []

    if side == "up":
        first_cross = previous_high <= level and high > level
        is_close_outside = close > level
        wick_ratio = c["upper_wick_ratio"][index]
        acceptance_rate = float((future_close > level).sum() / LOOKAHEAD_BARS)
        t12_status = bool(future_close[-1] > level)
        spot_reverse_ratio = max(0.0, -cum_spot_delta) / max(abs(cum_fut_delta), 1e-12)
        spot_divergence = cum_spot_delta < 0 and spot_reverse_ratio > SPOT_REVERSE_RATIO_MAX
        oi_lacking = is_close_outside and delta_oi_pct < BREAKOUT_DELTA_OI_PCT_MIN
        sweep = delta_oi_pct < -0.0015 and wick_ratio > 0.5 and acceptance_rate <= 0.35
        spot_supports_breakout = cum_spot_delta >= 0 or spot_reverse_ratio <= SPOT_REVERSE_RATIO_MAX
        breakout = is_close_outside and delta_oi_pct > BREAKOUT_DELTA_OI_PCT_MIN and spot_supports_breakout and acceptance_rate >= 0.66
    else:
        first_cross = previous_low >= level and low < level
        is_close_outside = close < level
        wick_ratio = c["lower_wick_ratio"][index]
        acceptance_rate = float((future_close < level).sum() / LOOKAHEAD_BARS)
        t12_status = bool(future_close[-1] < level)
        spot_reverse_ratio = max(0.0, cum_spot_delta) / max(abs(cum_fut_delta), 1e-12)
        spot_divergence = cum_spot_delta > 0 and spot_reverse_ratio > SPOT_REVERSE_RATIO_MAX
        oi_lacking = is_close_outside and delta_oi_pct < BREAKOUT_DELTA_OI_PCT_MIN
        sweep = delta_oi_pct < -0.0015 and wick_ratio > 0.5 and acceptance_rate <= 0.35
        spot_supports_breakout = cum_spot_delta <= 0 or spot_reverse_ratio <= SPOT_REVERSE_RATIO_MAX
        breakout = is_close_outside and delta_oi_pct > BREAKOUT_DELTA_OI_PCT_MIN and spot_supports_breakout and acceptance_rate >= 0.66

    if not first_cross or spot_divergence or oi_lacking or abs(avg_basis) > BASIS_THRESHOLD:
        return []
    if sweep:
        event_type, variant = "SWEEP", "LIQUIDITY_SWEEP"
    elif breakout:
        event_type, variant = "BREAKOUT", "TREND_BREAKOUT"
    else:
        return []
    return [{
        "candidate_index": index,
        "timestamp": pd.Timestamp(ts[index]).isoformat(),
        "side": side,
        "event_type": event_type,
        "label": LABELS[event_type],
        "rule_variant": variant,
        "level_price": float(level),
        "delta_oi_pct": float(delta_oi_pct),
        "cum_spot_delta": float(cum_spot_delta),
        "cum_fut_delta": float(cum_fut_delta),
        "spot_reverse_ratio": float(spot_reverse_ratio),
        "avg_basis": float(avg_basis),
        "wick_ratio": float(wick_ratio),
        "acceptance_rate": acceptance_rate,
        "t12_status": t12_status,
    }]


def generate(frame: pd.DataFrame) -> list[dict]:
    columns = {column: frame[column].to_numpy(dtype=float, copy=False) for column in [
        "fut_high", "fut_low", "fut_close", "fut_delta", "spot_delta", "oi", "basis",
        "level_high", "level_low", "upper_wick_ratio", "lower_wick_ratio",
    ]}
    ts = frame["timestamp"].to_numpy()
    out: list[dict] = []
    for index in range(LEVEL_LOOKBACK_BARS, len(frame) - LOOKAHEAD_BARS):
        out.extend(candidate(columns, ts, index, "up"))
        out.extend(candidate(columns, ts, index, "down"))
    return out


def resolve(candidates: list[dict], frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not candidates:
        return pd.DataFrame(), pd.DataFrame()
    candidates = sorted(candidates, key=lambda row: (row["candidate_index"], row["side"], row["event_type"]))
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
        sides = sorted({row["side"] for row in cluster})
        anchor = min(cluster, key=lambda row: row["candidate_index"])
        start = timestamps.iloc[max(0, anchor["candidate_index"] - MATCH_TOLERANCE_BARS)].isoformat()
        end = timestamps.iloc[min(len(frame) - 1, anchor["candidate_index"] + MATCH_TOLERANCE_BARS)].isoformat()
        if len(types) != 1 or len(sides) != 1:
            overlaps.extend({**row, "event_id": event_id, "overlap_types": ",".join(types), "overlap_sides": ",".join(sides), "match_window_start": start, "match_window_end": end} for row in cluster)
        else:
            labels.append({**anchor, "event_id": event_id, "match_window_start": start, "match_window_end": end, "overlap_excluded": False})
    return pd.DataFrame(labels), pd.DataFrame(overlaps)


def main() -> int:
    options = args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_aligned(options)
    candidates = generate(frame)
    labels, overlaps = resolve(candidates, frame)
    label_path = options.output_dir / "eth_5m_valid_setup_labels.csv"
    overlap_path = options.output_dir / "eth_5m_valid_setup_overlaps.csv"
    report_path = options.output_dir / "report.json"
    labels.to_csv(label_path, index=False)
    overlaps.to_csv(overlap_path, index=False)
    counts = {name: int((labels["event_type"] == name).sum()) if not labels.empty else 0 for name in LABELS}
    overlap_counts = {name: int((overlaps["event_type"] == name).sum()) if not overlaps.empty else 0 for name in LABELS}
    report = {
        "label_contract": LABELS,
        "fakeout_trap_removed": True,
        "source_period": {"start": str(frame["timestamp"].min()), "end": str(frame["timestamp"].max()), "aligned_5m_bars": int(len(frame))},
        "candidate_rows": len(candidates), "label_rows": len(labels), "overlap_rows": len(overlaps),
        "label_counts": counts, "overlap_counts": overlap_counts,
        "parameters": {"bar_minutes": 5, "lookahead_minutes": 60, "match_tolerance_minutes": 10, "level_lookback_bars": 288, "basis_threshold_raw_ratio": BASIS_THRESHOLD, "oi_lacking_delta_oi_pct_lt": BREAKOUT_DELTA_OI_PCT_MIN, "sweep_delta_oi_pct_lt": -0.0015, "breakout_delta_oi_pct_gt": BREAKOUT_DELTA_OI_PCT_MIN, "breakout_acceptance_rate_gte": 0.66, "sweep_acceptance_rate_lte": 0.35, "spot_reverse_ratio_max": SPOT_REVERSE_RATIO_MAX},
        "input_features": ["futures_5m_ohlcv", "spot_5m_taker_delta", "futures_5m_taker_delta", "open_interest_5m", "spot_perp_basis"],
        "alignment_policy": "inner join on UTC 5m timestamps; rows missing any required input are excluded",
        "overlap_policy": "different event types or directions within +/-10m are excluded from labels and written separately",
        "future_features_used_for_labels": False,
        "output_labels": str(label_path), "output_overlaps": str(overlap_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
