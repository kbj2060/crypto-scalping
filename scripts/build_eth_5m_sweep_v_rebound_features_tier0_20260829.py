#!/usr/bin/env python3
"""Tier 0 input features for the liquidity_sweep -> V_REBOUND model
(docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md).

Every feature here is causal (computable at the sweep bar's own close, no lookahead) and comes
from ETH's own OHLCV+taker_buy_base -- the same source file as the label, zero extra joins,
zero coverage gaps. Regime features (GBM2/GBM3/wide24 HMM/variance-ratio) were investigated and
excluded -- see the feature plan doc's "레짐 계열 최종 결론" section (GBM3 is ~94% in-sample vs
this label's period; wide24/variance-ratio showed near-zero correlation with the label, r<0.02).

Formulas are reused, not reimplemented, from the evidence-signal research lineage:
  - p_fast/p_slow/adx14/pdi/ndi   <- compute_indicators (backtest_eth_slowk_williamsr_persistence_
                                      confluence_20260814.py)
  - delta_z/cvd_roll_roc_48/vwap_dev_z/vol_z/wick ratios
                                  <- add_creative_indicators (analyze_eth_creative_reversal_
                                      evidence_signals_20260814.py)
  - bb_pctb/bb_width_pctile       <- add_broad_indicators (analyze_eth_broad_evidence_signal_
                                      sweep_20260814.py)
  - ret3_z                       <- inlined verbatim (2-line expression, not exposed as a
                                      standalone function anywhere in that lineage either --
                                      same choice build_eth_evidence_signal_context_features_
                                      20260814.py made for the same reason)
  - sweep_level_low/high, atr, side
                                  <- add_causal_columns (build_eth_5m_sweep_followthrough_v2_
                                      labels_20260829.py, same function the label itself uses)

This chain needs torch (a transitive import inside compute_indicators' dependencies) -- run with
the quant_ai conda env, not the base anaconda env used by this project's other pandas-only scripts:
  /home/kbj20/anaconda3/envs/quant_ai/bin/python3 scripts/build_eth_5m_sweep_v_rebound_features_tier0_20260829.py
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
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829"
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"

FEATURE_COLUMNS = [
    # sweep-derived (new, self-contained)
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    # order flow
    "delta_z", "flow_aligned_delta_z",
    # evidence-signal family (continuous, reused verbatim)
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    # trend/volatility context
    "adx14", "pdi", "ndi", "bb_width_pctile",
]


def load_sweep_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_features_20260829", SWEEP_IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_indicator_frame(sweep_impl) -> pd.DataFrame:
    # Same filtering as sweep_impl.load_5m (>=2024-01-01, drop the still-open bar), replicated
    # rather than called directly since load_5m only reads the 5 OHLC columns -- this needs
    # volume/taker_buy_base too. Row-count/timestamp-equality asserted against sweep_frame below.
    raw = pd.read_csv(
        SOURCE, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]
    )
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

    # ret3_z: inlined verbatim (analyze_eth_deep_evidence_signal_sweep_round2_20260814.py::
    # add_short_term_and_patterns), not importable as a standalone function.
    ret3 = frame["close"] / frame["close"].shift(3) - 1.0
    ret3_mean = ret3.rolling(288, min_periods=288).mean()
    ret3_std = ret3.rolling(288, min_periods=288).std()
    frame["ret3_z"] = (ret3 - ret3_mean) / ret3_std.replace(0.0, np.nan)
    return frame


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_impl = load_sweep_impl()

    indicator_frame = build_indicator_frame(sweep_impl)
    sweep_frame = sweep_impl.add_causal_columns(sweep_impl.load_5m(SOURCE))
    assert len(indicator_frame) == len(sweep_frame), "row count mismatch between the two indicator builds"
    assert (indicator_frame["timestamp"].to_numpy() == sweep_frame["timestamp"].to_numpy()).all(), (
        "timestamp misalignment between the two indicator builds"
    )

    frame = indicator_frame.copy()
    frame["sweep_level_low"] = sweep_frame["sweep_level_low"]
    frame["sweep_level_high"] = sweep_frame["sweep_level_high"]
    frame["atr"] = sweep_frame["atr"]
    frame["atr_percentile_864"] = frame["atr"].rolling(864, min_periods=864).rank(pct=True)
    frame["range_width_pct"] = (frame["sweep_level_high"] - frame["sweep_level_low"]) / frame["close"]
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.weekday

    labels = pd.read_csv(LABEL_CSV)
    features = frame.iloc[labels["candidate_index"].to_numpy()].reset_index(drop=True)
    label_ts = pd.to_datetime(labels["timestamp"], utc=True)
    assert (features["timestamp"].to_numpy() == label_ts.to_numpy()).all(), (
        "candidate_index positions do not line up with the label file's own timestamps"
    )

    result = labels[["candidate_index", "timestamp", "side", "label"]].copy()
    result["is_downside"] = (labels["side"] == "downside").astype(np.int8)
    is_down = result["is_downside"].to_numpy(dtype=bool)

    level = np.where(is_down, features["sweep_level_low"], features["sweep_level_high"]).astype(float)
    atr = features["atr"].to_numpy(dtype=float)
    penetration = np.where(is_down, level - features["low"].to_numpy(), features["high"].to_numpy() - level)
    result["sweep_penetration_atr"] = penetration / atr

    result["atr"] = atr
    result["atr_percentile_864"] = features["atr_percentile_864"].to_numpy()
    result["range_width_pct"] = features["range_width_pct"].to_numpy()
    result["hour_utc"] = features["hour_utc"].to_numpy()
    result["weekday"] = features["weekday"].to_numpy()

    delta_z = features["delta_z"].to_numpy(dtype=float)
    result["delta_z"] = delta_z
    result["flow_aligned_delta_z"] = np.where(is_down, delta_z, -delta_z)

    for col in ["p_fast", "p_slow", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
                "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi",
                "bb_width_pctile", "ret3_z"]:
        result[col] = features[col].to_numpy()

    out_path = OUT_DIR / "eth_5m_sweep_v_rebound_features_tier0.csv"
    result.to_csv(out_path, index=False)

    nan_counts = result[FEATURE_COLUMNS].isna().sum()
    report = {
        "rows": int(len(result)),
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "label_rate": float(result["label"].mean()),
        "nan_counts": {k: int(v) for k, v in nan_counts.items() if v > 0},
        "rows_any_nan": int(result[FEATURE_COLUMNS].isna().any(axis=1).sum()),
        "excluded_regime_features": "GBM2/GBM3/wide24 HMM/variance-ratio -- see feature plan doc",
        "output": str(out_path),
    }
    (OUT_DIR / "features_tier0_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(result[FEATURE_COLUMNS].describe().T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
