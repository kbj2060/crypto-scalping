#!/usr/bin/env python3
"""Build causal ETHUSDT 5m structure labels using the finalized 0/1 rules.

label 0 = liquidity_sweep (including rejection)
label 1 = trend_breakout (including retest)

Fakeout/trap and all other noise are intentionally excluded.  The level uses only the prior
24 hours of closed 5m bars.  A candidate is the first crossing of that level; the following
12 closed bars are used only to determine the label outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data/labels/eth_5m_structure_events_20260829"
START = pd.Timestamp("2024-01-01", tz="UTC")

LABELS = {"liquidity_sweep": 0, "trend_breakout": 1}
BAR_MINUTES = 5
MATCH_TOLERANCE_BARS = 2
LEVEL_LOOKBACK_BARS = 288
LOOKAHEAD_BARS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_5m(path: Path) -> pd.DataFrame:
    required = ["timestamp", "open", "high", "low", "close"]
    frame = pd.read_csv(path, usecols=required, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=required)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    frame = frame.loc[frame["timestamp"] >= START].reset_index(drop=True)
    current_bar_start = pd.Timestamp.now(tz="UTC").floor("5min")
    frame = frame.loc[frame["timestamp"] < current_bar_start].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("5m source has no rows from 2024-01-01 onward")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError("5m source contains non-positive OHLC")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise RuntimeError("5m source has high below open/close")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise RuntimeError("5m source has low above open/close")
    return frame


def add_causal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["level_high"] = out["high"].shift(1).rolling(LEVEL_LOOKBACK_BARS, min_periods=LEVEL_LOOKBACK_BARS).max()
    out["level_low"] = out["low"].shift(1).rolling(LEVEL_LOOKBACK_BARS, min_periods=LEVEL_LOOKBACK_BARS).min()
    candle_range = (out["high"] - out["low"]).clip(lower=1e-12)
    out["upper_wick_ratio"] = (out["high"] - out[["open", "close"]].max(axis=1)) / candle_range
    out["lower_wick_ratio"] = (out[["open", "close"]].min(axis=1) - out["low"]) / candle_range
    out["bar_index"] = np.arange(len(out), dtype=np.int64)
    return out


def _future_is_contiguous(timestamps: np.ndarray, index: int) -> bool:
    end_index = index + LOOKAHEAD_BARS
    if end_index >= len(timestamps):
        return False
    expected = timestamps[index] + pd.to_timedelta(
        np.arange(LOOKAHEAD_BARS + 1) * BAR_MINUTES, unit="min"
    )
    return bool(np.array_equal(timestamps[index:end_index + 1], expected))


def _candidate(columns: dict[str, np.ndarray], timestamps: np.ndarray, index: int, side: str) -> list[dict]:
    level = columns["level_high"][index] if side == "up" else columns["level_low"][index]
    if not np.isfinite(level) or not _future_is_contiguous(timestamps, index):
        return []

    previous_high = columns["high"][index - 1]
    previous_low = columns["low"][index - 1]
    previous_close = columns["close"][index - 1]
    high = columns["high"][index]
    low = columns["low"][index]
    close = columns["close"][index]
    future_high = columns["high"][index + 1:index + LOOKAHEAD_BARS + 1]
    future_low = columns["low"][index + 1:index + LOOKAHEAD_BARS + 1]
    future_close = columns["close"][index + 1:index + LOOKAHEAD_BARS + 1]

    if side == "up":
        first_cross = previous_high <= level and high > level
        is_t0_close_above = close > level
        wick_ratio = columns["upper_wick_ratio"][index]
        acceptance_rate = float((future_close > level).sum() / LOOKAHEAD_BARS)
        t12_status = bool(future_close[-1] > level)
        sweep_variant = (
            (not is_t0_close_above) and wick_ratio > 0.60 and acceptance_rate <= 0.15
        )
        rejection_variant = is_t0_close_above and acceptance_rate <= 0.25 and not t12_status
        trend_variant = is_t0_close_above and wick_ratio < 0.40 and acceptance_rate >= 0.75
        retest_variant = (
            acceptance_rate >= 0.50
            and t12_status
            and future_low.min() > level * 0.995
        )
    else:
        first_cross = previous_low >= level and low < level
        is_t0_close_above = close < level
        wick_ratio = columns["lower_wick_ratio"][index]
        acceptance_rate = float((future_close < level).sum() / LOOKAHEAD_BARS)
        t12_status = bool(future_close[-1] < level)
        sweep_variant = (
            (not is_t0_close_above) and wick_ratio > 0.60 and acceptance_rate <= 0.15
        )
        rejection_variant = is_t0_close_above and acceptance_rate <= 0.25 and not t12_status
        trend_variant = is_t0_close_above and wick_ratio < 0.40 and acceptance_rate >= 0.75
        retest_variant = (
            acceptance_rate >= 0.50
            and t12_status
            and future_high.max() < level * 1.005
        )

    if not first_cross:
        return []
    if sweep_variant:
        event_type, rule_variant = "liquidity_sweep", "LIQUIDITY_SWEEP"
    elif rejection_variant:
        event_type, rule_variant = "liquidity_sweep", "LIQUIDITY_SWEEP_REJECTION"
    elif trend_variant:
        event_type, rule_variant = "trend_breakout", "TREND_BREAKOUT"
    elif retest_variant:
        event_type, rule_variant = "trend_breakout", "TREND_BREAKOUT_WITH_RETEST"
    else:
        return []

    return [{
        "candidate_index": index,
        "timestamp": pd.Timestamp(timestamps[index]).isoformat(),
        "side": side,
        "event_type": event_type,
        "label": LABELS[event_type],
        "rule_variant": rule_variant,
        "level_price": float(level),
        "wick_ratio": float(wick_ratio),
        "acceptance_rate": acceptance_rate,
        "t12_status": t12_status,
    }]


def generate_candidates(frame: pd.DataFrame) -> list[dict]:
    columns = {
        column: frame[column].to_numpy(dtype=float, copy=False)
        for column in [
            "high", "low", "close", "level_high", "level_low",
            "upper_wick_ratio", "lower_wick_ratio",
        ]
    }
    timestamps = frame["timestamp"].to_numpy()
    candidates: list[dict] = []
    for index in range(LEVEL_LOOKBACK_BARS, len(frame) - LOOKAHEAD_BARS):
        candidates.extend(_candidate(columns, timestamps, index, "up"))
        candidates.extend(_candidate(columns, timestamps, index, "down"))
    return candidates


def resolve_candidates(candidates: list[dict], frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not candidates:
        return pd.DataFrame(), pd.DataFrame()
    candidates = sorted(candidates, key=lambda row: (row["candidate_index"], row["side"], row["event_type"]))
    clusters: list[list[dict]] = []
    for candidate in candidates:
        if not clusters or candidate["candidate_index"] - clusters[-1][-1]["candidate_index"] > MATCH_TOLERANCE_BARS:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)

    labels: list[dict] = []
    overlaps: list[dict] = []
    timestamps = frame["timestamp"]
    for event_id, cluster in enumerate(clusters, start=1):
        types = sorted({row["event_type"] for row in cluster})
        sides = sorted({row["side"] for row in cluster})
        anchor = min(cluster, key=lambda row: row["candidate_index"])
        window_start = timestamps.iloc[max(0, anchor["candidate_index"] - MATCH_TOLERANCE_BARS)].isoformat()
        window_end = timestamps.iloc[min(len(frame) - 1, anchor["candidate_index"] + MATCH_TOLERANCE_BARS)].isoformat()
        if len(types) != 1 or len(sides) != 1:
            for row in cluster:
                overlaps.append({
                    **row,
                    "event_id": event_id,
                    "overlap_types": ",".join(types),
                    "overlap_sides": ",".join(sides),
                    "match_window_start": window_start,
                    "match_window_end": window_end,
                })
        else:
            labels.append({
                **anchor,
                "event_id": event_id,
                "match_window_start": window_start,
                "match_window_end": window_end,
                "overlap_excluded": False,
            })
    return pd.DataFrame(labels), pd.DataFrame(overlaps)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = add_causal_columns(load_5m(args.source))
    candidates = generate_candidates(frame)
    labels, overlaps = resolve_candidates(candidates, frame)
    label_path = args.output_dir / "eth_5m_structure_labels.csv"
    overlap_path = args.output_dir / "eth_5m_structure_overlaps.csv"
    report_path = args.output_dir / "report.json"
    labels.to_csv(label_path, index=False)
    overlaps.to_csv(overlap_path, index=False)

    label_counts = {name: int((labels["event_type"] == name).sum()) if not labels.empty else 0 for name in LABELS}
    overlap_counts = {name: int((overlaps["event_type"] == name).sum()) if not overlaps.empty else 0 for name in LABELS}
    report = {
        "label_contract": LABELS,
        "fakeout_trap_removed": True,
        "source": str(args.source),
        "start": str(frame["timestamp"].min()),
        "end": str(frame["timestamp"].max()),
        "bars": int(len(frame)),
        "candidate_rows": int(len(candidates)),
        "label_rows": int(len(labels)),
        "overlap_rows": int(len(overlaps)),
        "label_counts": label_counts,
        "overlap_counts": overlap_counts,
        "parameters": {
            "bar_minutes": BAR_MINUTES,
            "match_tolerance_minutes": MATCH_TOLERANCE_BARS * BAR_MINUTES,
            "level_lookback_bars": LEVEL_LOOKBACK_BARS,
            "lookahead_minutes": LOOKAHEAD_BARS * BAR_MINUTES,
            "sweep_wick_ratio_gt": 0.60,
            "sweep_acceptance_rate_lte": 0.15,
            "rejection_acceptance_rate_lte": 0.25,
            "trend_wick_ratio_lt": 0.40,
            "trend_acceptance_rate_gte": 0.75,
            "retest_acceptance_rate_gte": 0.50,
            "retest_level_tolerance": 0.005,
        },
        "level_policy": "prior 24h rolling high/low from closed 5m bars; first level crossing is t0",
        "unclassified_noise_written": False,
        "future_features_used_for_labels": False,
        "subminute_or_1m_data_used": False,
        "output_labels": str(label_path),
        "output_overlaps": str(overlap_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
