#!/usr/bin/env python3
"""돌파 지속 (breakout continuation) label + Tier0 features, for the TabPFN cheap_gate step
(raw-lift precheck showed ~1.0x lift -- same as sweep's own raw-lift history, which didn't stop
V자반등 from becoming this project's best classifier once TabPFN discriminated within the
triggered pool -- see research_eth_breakout_continuation_giveback_check_20260831.py for the
precheck + its own docstring for the full reasoning chain).

Trigger: close-confirmed breakout of the causal 48-bar swing level (cluster-anchored, first bar of
each consecutive run only -- see research_eth_breakout_continuation_audit_20260831.py for why).
Label (v7b giveback formula, reused verbatim, direction-mirrored for continuation not reversal --
see research_eth_breakout_continuation_giveback_check_20260831.py::continuation_outcome):
  지속(1): fast_move_atr_mult>=1.5 (30min, close-based, entry=breakout bar's own close) AND
    giveback_ratio<=0.20 (60min full window).
  정체(0): fast_move_atr_mult<1.0.
  else: excluded (애매), same exclude-middle convention as v7b.
ATR uses atr[idx-1] (pre-trigger, not self-inclusive -- the audited bug fix).

Tier0 23 features (22 + rsi) reused verbatim from build_eth_5m_sweep_v_rebound_features_tier0_
20260829.py::build_indicator_frame -- the same feature bank every signal in this project's
lineage uses. is_downside/sweep_penetration_atr/flow_aligned_delta_z computed generically per
candidate direction, same pattern as build_eth_5m_v_rebound_multitrigger_features_tier0_
20260831.py used for the 9-trigger V자반등 label.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIER0_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_v_rebound_features_tier0_20260829.py"
OUT_DIR = ROOT / "data/labels/eth_5m_breakout_continuation_20260831"

FAST_BARS = 6
FULL_BARS = 12
ATR_MULT = 1.5
T_SUSTAIN = 0.20


def first_bar_of_each_run(idx: np.ndarray) -> np.ndarray:
    if len(idx) == 0:
        return idx
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    return idx[starts]


def continuation_outcome(close: np.ndarray, high: np.ndarray, low: np.ndarray, atr: np.ndarray,
                          idx: int, is_up: bool, n: int) -> dict | None:
    if idx - 1 < 0 or idx + FULL_BARS >= n:
        return None
    pre_atr = atr[idx - 1]
    if not np.isfinite(pre_atr) or pre_atr <= 0:
        return None
    entry = close[idx]
    fast_close = close[idx + 1: idx + FAST_BARS + 1]
    full_high = high[idx + 1: idx + FULL_BARS + 1]
    full_low = low[idx + 1: idx + FULL_BARS + 1]
    full_close_end = close[idx + FULL_BARS]

    if is_up:
        fast_move = fast_close.max() - entry
        peak = full_high.max()
        denom = peak - entry
        giveback = (peak - full_close_end) / denom if denom > 1e-12 else np.nan
    else:
        fast_move = entry - fast_close.min()
        peak = full_low.min()
        denom = entry - peak
        giveback = (full_close_end - peak) / denom if denom > 1e-12 else np.nan

    fast_mult = fast_move / pre_atr
    if fast_mult >= ATR_MULT and np.isfinite(giveback) and giveback <= T_SUSTAIN:
        outcome = "지속"
    elif fast_mult < 1.0:
        outcome = "정체"
    else:
        outcome = "애매"
    return {"fast_move_atr_mult": float(fast_mult),
            "giveback_ratio": float(giveback) if np.isfinite(giveback) else None,
            "outcome": outcome}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _spec = importlib.util.spec_from_file_location("tier0_features_20260829_breakout", TIER0_SCRIPT)
    _tier0 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tier0)

    sweep_impl = _tier0.load_sweep_impl()
    indicator_frame = _tier0.build_indicator_frame(sweep_impl)  # >=2024-01-01, Tier0 22 cols + timestamp/OHLC
    sweep_frame = sweep_impl.add_causal_columns(sweep_impl.load_5m(_tier0.SOURCE))
    assert len(indicator_frame) == len(sweep_frame)
    assert (indicator_frame["timestamp"].to_numpy() == sweep_frame["timestamp"].to_numpy()).all()

    frame = indicator_frame.copy()
    frame["sweep_level_low"] = sweep_frame["sweep_level_low"]
    frame["sweep_level_high"] = sweep_frame["sweep_level_high"]
    frame["atr"] = sweep_frame["atr"]
    frame["atr_percentile_864"] = frame["atr"].rolling(864, min_periods=864).rank(pct=True)
    frame["range_width_pct"] = (frame["sweep_level_high"] - frame["sweep_level_low"]) / frame["close"]
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.weekday

    def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        return 100 - 100 / (1 + rs)

    frame["rsi"] = rsi_wilder(frame["close"])

    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    level_high = frame["sweep_level_high"].to_numpy()
    level_low = frame["sweep_level_low"].to_numpy()
    atr = frame["atr"].to_numpy()
    delta_z = frame["delta_z"].to_numpy()
    n = len(frame)

    is_breakout_up = np.isfinite(level_high) & (close > level_high)
    is_breakout_down = np.isfinite(level_low) & (close < level_low)
    up_idx = first_bar_of_each_run(np.flatnonzero(is_breakout_up))
    down_idx = first_bar_of_each_run(np.flatnonzero(is_breakout_down))
    print(f"cluster-anchored triggers: up={len(up_idx)} down={len(down_idx)}")

    rows = []
    for idx_arr, is_up in ((up_idx, True), (down_idx, False)):
        for i in idx_arr:
            o = continuation_outcome(close, high, low, atr, int(i), is_up, n)
            if o is None:
                continue
            level = level_high[i] if is_up else level_low[i]
            penetration = (close[i] - level) if is_up else (level - close[i])
            rows.append({
                "idx": int(i), "timestamp": frame["timestamp"].iloc[i], "direction": "up" if is_up else "down",
                "is_downside": int(not is_up),  # "is_downside" name kept for schema parity w/ V자반등 Tier0
                "sweep_penetration_atr": penetration / atr[i - 1] if atr[i - 1] > 0 else np.nan,
                "flow_aligned_delta_z": delta_z[i] if is_up else -delta_z[i],
                **o,
            })
    labels = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    other_cols = ["atr", "atr_percentile_864", "range_width_pct", "hour_utc", "weekday", "delta_z",
                  "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
                  "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi",
                  "bb_width_pctile", "rsi"]
    merged = labels.merge(frame[["timestamp"] + other_cols], on="timestamp", how="left")

    FEATURE_COLUMNS = ["is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
                        "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
                        "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
                        "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi",
                        "bb_width_pctile", "rsi"]
    merged = merged.dropna(subset=FEATURE_COLUMNS)

    out_path = OUT_DIR / "eth_5m_breakout_continuation_features_tier0.csv"
    merged.to_csv(out_path, index=False)

    outcome_counts = merged["outcome"].value_counts().to_dict()
    report = {
        "total_candidates": int(len(merged)), "outcome_counts": outcome_counts,
        "outcome_rate": {k: round(v / len(merged), 4) for k, v in outcome_counts.items()},
        "feature_columns": FEATURE_COLUMNS, "output": str(out_path),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
