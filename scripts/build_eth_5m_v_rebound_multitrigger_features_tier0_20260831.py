#!/usr/bin/env python3
"""Tier0 features (22 + rsi) for the new 7-trigger V자반등 label -- SAME feature set/formulas as
v7b's own build_eth_5m_sweep_v_rebound_features_tier0_20260829.py, reused verbatim (imports
build_indicator_frame() from that script rather than reimplementing), just joined onto the wider
multitrigger candidate pool instead of sweep-only. sweep_penetration_atr/range_width_pct are still
well-defined for non-sweep candidates (they describe "how extended is this bar vs. the recent
48-bar range", independent of which trigger brought the bar into the pool) -- unchanged formula.

Joined by TIMESTAMP (not positional index) since the multitrigger label's row indices were
computed against a full-history frame (2023-12-31 onward, no date floor), while build_indicator_
frame() applies the canonical START=2024-01-01 floor -- an inner merge on timestamp naturally
drops the ~9hr of pre-2024-01-01 candidates without needing a separate filter, matching the
project's established convention automatically.

Run with the quant_ai conda env (torch is a transitive dependency of compute_indicators):
  /home/kbj20/anaconda3/envs/quant_ai/bin/python scripts/build_eth_5m_v_rebound_multitrigger_features_tier0_20260831.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TIER0_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_v_rebound_features_tier0_20260829.py"
_spec = importlib.util.spec_from_file_location("tier0_features_20260829", TIER0_SCRIPT)
_tier0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tier0)

SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
LABEL_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_labels.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831"

FEATURE_COLUMNS = _tier0.FEATURE_COLUMNS + ["rsi"]  # matches v7b's TIER0 + rsi exactly


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_impl = _tier0.load_sweep_impl()

    indicator_frame = _tier0.build_indicator_frame(sweep_impl)
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
    frame["rsi"] = rsi_wilder(frame["close"])

    labels = pd.read_csv(LABEL_CSV)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    n_before_2024 = int((labels["timestamp"] < pd.Timestamp("2024-01-01", tz="UTC")).sum())

    merged = labels.merge(frame, on="timestamp", how="inner", suffixes=("", "_ind"))
    dropped = len(labels) - len(merged)
    print(f"라벨 {len(labels)}건 -> 피처merge후 {len(merged)}건 (드롭 {dropped}건, "
          f"그중 2024-01-01 이전 {n_before_2024}건 -- START 컨벤션과 정합)")

    is_down = (merged["direction"] == "downside").to_numpy()
    result = merged[["idx", "timestamp", "direction", "triggers", "n_triggers",
                      "fast_move_atr_mult", "giveback_ratio", "outcome"]].copy()
    result["is_downside"] = is_down.astype(np.int8)

    level = np.where(is_down, merged["sweep_level_low"], merged["sweep_level_high"]).astype(float)
    atr = merged["atr"].to_numpy(dtype=float)
    penetration = np.where(is_down, level - merged["low"].to_numpy(), merged["high"].to_numpy() - level)
    result["sweep_penetration_atr"] = penetration / atr

    result["atr"] = atr
    result["atr_percentile_864"] = merged["atr_percentile_864"].to_numpy()
    result["range_width_pct"] = merged["range_width_pct"].to_numpy()
    result["hour_utc"] = merged["hour_utc"].to_numpy()
    result["weekday"] = merged["weekday"].to_numpy()

    delta_z = merged["delta_z"].to_numpy(dtype=float)
    result["delta_z"] = delta_z
    result["flow_aligned_delta_z"] = np.where(is_down, delta_z, -delta_z)

    for col in ["p_fast", "p_slow", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
                "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi",
                "bb_width_pctile", "ret3_z", "rsi"]:
        result[col] = merged[col].to_numpy()

    out_path = OUT_DIR / "eth_5m_v_rebound_multitrigger_features_tier0.csv"
    result.to_csv(out_path, index=False)

    nan_counts = result[FEATURE_COLUMNS].isna().sum()
    rows_any_nan = int(result[FEATURE_COLUMNS].isna().any(axis=1).sum())
    outcome_counts = result["outcome"].value_counts().to_dict()
    report = {
        "rows": int(len(result)),
        "rows_dropped_at_merge": dropped,
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "outcome_counts": outcome_counts,
        "outcome_rate": {k: round(v / len(result), 4) for k, v in outcome_counts.items()},
        "nan_counts": {k: int(v) for k, v in nan_counts.items() if v > 0},
        "rows_any_nan": rows_any_nan,
        "output": str(out_path),
    }
    (OUT_DIR / "features_tier0_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
