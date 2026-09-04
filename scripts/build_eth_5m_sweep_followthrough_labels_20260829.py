#!/usr/bin/env python3
"""Build labels for the two follow-through paths after the dashboard liquidity_sweep signal.

The event definition is copied from the live evidence dashboard:
  - downside sweep: low < prior 48-bar low and close > that low
  - upside sweep: high > prior 48-bar high and close < that high

Only the post-sweep path is labeled:
  0 = SWEEP_BREAKOUT_SUPPORT: the reclaimed level is accepted for the next hour
  1 = SWEEP_V_REBOUND: a fast reversal reaches one ATR within the next 15 minutes

All other sweep events are excluded. Future bars are used only for the outcome label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2024-01-01", tz="UTC")
LABELS = {"SWEEP_BREAKOUT_SUPPORT": 0, "SWEEP_V_REBOUND": 1}
BAR_MINUTES = 5
SWEEP_LOOKBACK_BARS = 48
LOOKAHEAD_BARS = 12
V_REBOUND_BARS = 3
ATR_N = 14
V_REBOUND_ATR_MULT = 1.0
SUPPORT_ACCEPTANCE_RATE = 0.66
SUPPORT_MAX_RETEST_ATR = 0.25
MATCH_TOLERANCE_BARS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/labels/eth_5m_sweep_followthrough_20260829",
    )
    return parser.parse_args()


def load_5m(path: Path) -> pd.DataFrame:
    columns = ["timestamp", "open", "high", "low", "close"]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna()
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .loc[lambda value: value["timestamp"] >= START]
        .reset_index(drop=True)
    )
    current_bar_start = pd.Timestamp.now(tz="UTC").floor("5min")
    frame = frame.loc[frame["timestamp"] < current_bar_start].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("5m source has no closed rows from 2024-01-01 onward")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise RuntimeError("5m source has high below open/close")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise RuntimeError("5m source has low above open/close")
    return frame


def add_causal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["sweep_level_high"] = out["high"].rolling(
        SWEEP_LOOKBACK_BARS, min_periods=SWEEP_LOOKBACK_BARS
    ).max().shift(1)
    out["sweep_level_low"] = out["low"].rolling(
        SWEEP_LOOKBACK_BARS, min_periods=SWEEP_LOOKBACK_BARS
    ).min().shift(1)
    previous_close = out["close"].shift(1)
    true_range = pd.concat(
        [out["high"] - out["low"],
         (out["high"] - previous_close).abs(),
         (out["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr"] = true_range.rolling(ATR_N, min_periods=ATR_N).mean()
    return out


def contiguous(timestamps: np.ndarray, start: int, end: int) -> bool:
    if end >= len(timestamps):
        return False
    expected = timestamps[start] + pd.to_timedelta(
        np.arange(end - start + 1) * BAR_MINUTES, unit="min"
    )
    return bool(np.array_equal(timestamps[start:end + 1], expected))


def candidate(frame: pd.DataFrame, timestamps: np.ndarray, index: int, side: str) -> dict | None:
    if index < SWEEP_LOOKBACK_BARS or index + LOOKAHEAD_BARS >= len(frame):
        return None
    if not contiguous(timestamps, index, index + LOOKAHEAD_BARS):
        return None

    row = frame.iloc[index]
    level = row["sweep_level_low"] if side == "downside" else row["sweep_level_high"]
    atr = row["atr"]
    if not np.isfinite(level) or not np.isfinite(atr) or atr <= 0:
        return None

    is_sweep = (
        row["low"] < level and row["close"] > level
        if side == "downside"
        else row["high"] > level and row["close"] < level
    )
    if not is_sweep:
        return None

    future = frame.iloc[index + 1:index + LOOKAHEAD_BARS + 1]
    future_close = future["close"].to_numpy(dtype=float)
    first_three = future.iloc[:V_REBOUND_BARS]
    if side == "downside":
        rebound_move = float(first_three["high"].max() - row["close"])
        acceptance_rate = float((future_close > level).sum() / LOOKAHEAD_BARS)
        support_holds = float(future["low"].min()) > level - SUPPORT_MAX_RETEST_ATR * atr
        final_on_reclaimed_side = bool(future_close[-1] > level)
    else:
        rebound_move = float(row["close"] - first_three["low"].min())
        acceptance_rate = float((future_close < level).sum() / LOOKAHEAD_BARS)
        support_holds = float(future["high"].max()) < level + SUPPORT_MAX_RETEST_ATR * atr
        final_on_reclaimed_side = bool(future_close[-1] < level)

    v_rebound = rebound_move >= V_REBOUND_ATR_MULT * atr
    breakout_support = (
        acceptance_rate >= SUPPORT_ACCEPTANCE_RATE
        and final_on_reclaimed_side
        and support_holds
    )
    if v_rebound:
        event_type = "SWEEP_V_REBOUND"
    elif breakout_support:
        event_type = "SWEEP_BREAKOUT_SUPPORT"
    else:
        return None

    return {
        "candidate_index": index,
        "timestamp": pd.Timestamp(timestamps[index]).isoformat(),
        "side": side,
        "event_type": event_type,
        "label": LABELS[event_type],
        "sweep_level": float(level),
        "atr": float(atr),
        "rebound_move_first_15m": rebound_move,
        "rebound_atr_multiple": rebound_move / float(atr),
        "acceptance_rate_next_60m": acceptance_rate,
        "final_on_reclaimed_side": final_on_reclaimed_side,
        "support_holds_within_0p25_atr": support_holds,
    }


def generate_candidates(frame: pd.DataFrame) -> list[dict]:
    timestamps = frame["timestamp"].to_numpy()
    candidates: list[dict] = []
    for index in range(SWEEP_LOOKBACK_BARS, len(frame) - LOOKAHEAD_BARS):
        for side in ("downside", "upside"):
            row = candidate(frame, timestamps, index, side)
            if row is not None:
                candidates.append(row)
    return candidates


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
    options = parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    frame = add_causal_columns(load_5m(options.source))
    candidates = generate_candidates(frame)
    labels, overlaps = resolve(candidates, frame)
    label_path = options.output_dir / "eth_5m_sweep_followthrough_labels.csv"
    overlap_path = options.output_dir / "eth_5m_sweep_followthrough_overlaps.csv"
    report_path = options.output_dir / "report.json"
    labels.to_csv(label_path, index=False)
    overlaps.to_csv(overlap_path, index=False)
    report = {
        "label_contract": LABELS,
        "source_period": {
            "start": str(frame["timestamp"].min()),
            "end": str(frame["timestamp"].max()),
            "closed_5m_bars": int(len(frame)),
        },
        "candidate_rows": len(candidates),
        "label_rows": len(labels),
        "overlap_rows": len(overlaps),
        "label_counts": {
            name: int((labels["event_type"] == name).sum()) if not labels.empty else 0
            for name in LABELS
        },
        "overlap_counts": {
            name: int((overlaps["event_type"] == name).sum()) if not overlaps.empty else 0
            for name in LABELS
        },
        "parameters": {
            "bar_minutes": BAR_MINUTES,
            "sweep_lookback_bars": SWEEP_LOOKBACK_BARS,
            "lookahead_minutes": LOOKAHEAD_BARS * BAR_MINUTES,
            "v_rebound_minutes": V_REBOUND_BARS * BAR_MINUTES,
            "v_rebound_atr_multiple": V_REBOUND_ATR_MULT,
            "support_acceptance_rate_gte": SUPPORT_ACCEPTANCE_RATE,
            "support_max_retest_atr": SUPPORT_MAX_RETEST_ATR,
            "match_tolerance_minutes": MATCH_TOLERANCE_BARS * BAR_MINUTES,
        },
        "sweep_definition": "dashboard liquidity_sweep: low/high pokes beyond prior 48-bar swing and close reclaims the level",
        "future_features_used_for_labels": True,
        "input_features": ["futures_5m_ohlcv"],
        "output_labels": str(label_path),
        "output_overlaps": str(overlap_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
