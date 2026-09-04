#!/usr/bin/env python3
"""Tier0 features for the hybrid-anchor V_REBOUND label (build_eth_5m_liquidity_sweep_hybrid_
anchor_labels_20260829.py) -- identical feature set/formulas as build_eth_5m_sweep_v_rebound_
features_tier0_20260829.py (same 22 columns), just joined against the new hybrid-anchor event
population instead of the original 48-bar-only one. sweep_level_low/high/atr come from
compute_hybrid_levels() (imported, not recomputed) instead of add_causal_columns().

Needs torch (transitive import inside compute_indicators) -- run with the quant_ai conda env.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators  # noqa: E402
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402

SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_hybrid_anchor_v_rebound_20260829/eth_5m_sweep_hybrid_anchor_v_rebound_labels.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_sweep_hybrid_anchor_v_rebound_20260829"
HYBRID_SCRIPT = ROOT / "scripts/build_eth_5m_liquidity_sweep_hybrid_anchor_labels_20260829.py"

FEATURE_COLUMNS = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]


def load_hybrid_module():
    spec = importlib.util.spec_from_file_location("hybrid_impl_features_20260829", HYBRID_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_indicator_frame(hybrid, sweep_impl) -> pd.DataFrame:
    raw = pd.read_csv(SOURCE, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = (
        raw.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        .loc[lambda d: d["timestamp"] >= sweep_impl.START].reset_index(drop=True)
    )
    current_bar_start = pd.Timestamp.now(tz="UTC").floor("5min")
    raw = raw.loc[raw["timestamp"] < current_bar_start].reset_index(drop=True)

    frame = compute_indicators(raw)
    frame = add_creative_indicators(frame)
    frame = add_broad_indicators(frame)

    ret3 = frame["close"] / frame["close"].shift(3) - 1.0
    ret3_mean = ret3.rolling(288, min_periods=288).mean()
    ret3_std = ret3.rolling(288, min_periods=288).std()
    frame["ret3_z"] = (ret3 - ret3_mean) / ret3_std.replace(0.0, np.nan)
    return frame


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hybrid = load_hybrid_module()
    sweep_impl = hybrid.load_sweep_impl()

    indicator_frame = build_indicator_frame(hybrid, sweep_impl)
    raw_frame = sweep_impl.load_5m(SOURCE)
    assert len(indicator_frame) == len(raw_frame), "row count mismatch"
    assert (indicator_frame["timestamp"].to_numpy() == raw_frame["timestamp"].to_numpy()).all(), "timestamp mismatch"

    levels = hybrid.compute_hybrid_levels(raw_frame, sweep_impl)
    frame = indicator_frame.copy()
    frame["sweep_level_low"] = np.nan  # placeholder columns, filled per-side below via labels join
    frame["sweep_level_high"] = np.nan
    frame["atr"] = levels["atr"]
    frame["atr_percentile_864"] = frame["atr"].rolling(864, min_periods=864).rank(pct=True)
    frame["range_width_pct"] = np.nan  # filled below (depends on which side's level applies per row)
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.weekday
    hybrid_low_series = pd.Series(levels["hybrid_low"])
    hybrid_high_series = pd.Series(levels["hybrid_high"])

    labels = pd.read_csv(LABEL_CSV)
    features = frame.iloc[labels["candidate_index"].to_numpy()].reset_index(drop=True)
    label_ts = pd.to_datetime(labels["timestamp"], utc=True)
    assert (features["timestamp"].to_numpy() == label_ts.to_numpy()).all(), "candidate_index misalignment"

    result = labels[["candidate_index", "timestamp", "side", "label"]].copy()
    result["is_downside"] = (labels["side"] == "downside").astype(np.int8)
    is_down = result["is_downside"].to_numpy(dtype=bool)
    idx = labels["candidate_index"].to_numpy()

    level = np.where(is_down, hybrid_low_series.iloc[idx].to_numpy(), hybrid_high_series.iloc[idx].to_numpy())
    atr_vals = features["atr"].to_numpy(dtype=float)
    penetration = np.where(is_down, level - features["low"].to_numpy(), features["high"].to_numpy() - level)
    result["sweep_penetration_atr"] = penetration / atr_vals
    result["atr"] = atr_vals
    result["atr_percentile_864"] = features["atr_percentile_864"].to_numpy()

    sweep_level_high_at_idx = hybrid_high_series.iloc[idx].to_numpy()
    sweep_level_low_at_idx = hybrid_low_series.iloc[idx].to_numpy()
    result["range_width_pct"] = (sweep_level_high_at_idx - sweep_level_low_at_idx) / features["close"].to_numpy()
    result["hour_utc"] = features["hour_utc"].to_numpy()
    result["weekday"] = features["weekday"].to_numpy()

    delta_z = features["delta_z"].to_numpy(dtype=float)
    result["delta_z"] = delta_z
    result["flow_aligned_delta_z"] = np.where(is_down, delta_z, -delta_z)

    for col in ["p_fast", "p_slow", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
                "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi",
                "bb_width_pctile", "ret3_z"]:
        result[col] = features[col].to_numpy()

    out_path = OUT_DIR / "eth_5m_sweep_hybrid_anchor_features_tier0.csv"
    result.to_csv(out_path, index=False)

    nan_counts = result[FEATURE_COLUMNS].isna().sum()
    report = {
        "rows": int(len(result)),
        "feature_columns": FEATURE_COLUMNS,
        "label_rate": float(result["label"].mean()),
        "nan_counts": {k: int(v) for k, v in nan_counts.items() if v > 0},
        "rows_any_nan": int(result[FEATURE_COLUMNS].isna().any(axis=1).sum()),
        "output": str(out_path),
    }
    (OUT_DIR / "features_tier0_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
