#!/usr/bin/env python3
"""Correlation screen: Omega4.6.1's 102 base_cols (ETH h48qual/zig075 bundles) against the
V_REBOUND label, for columns not already individually tested in the Tier0/Tier1 rounds
(docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md).

Already tested and REJECTED, excluded here to avoid re-litigating a closed question:
  last_funding_rate, funding_z_score, funding_abs, oi_change_rate, btc_corr_60,
  regime3_current_sensitive_wide24_{bull,bear,chop}_prob/confidence/entropy/margin
Raw OHLCV levels excluded (uninformative as absolute levels, already captured via Tier 0's
derived transforms of the same source): open, high, low, close, volume, quote_volume, trades,
taker_buy_base, taker_buy_quote, sum_open_interest_value, close_btc, volume_btc, quote_volume_btc.

Reports BOTH plain Pearson correlation AND decile-spread (max-min V_REBOUND rate across 10
quantile bins) -- Pearson alone missed p_fast's real hump-shaped relationship in the Tier 0
round (corr 0.0006 vs permutation importance 0.094), so decile-spread is included as a cheap
catch-all for non-linear/non-monotonic relationships this early screening step could otherwise miss.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAINING_FEATURES = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"

ALREADY_TESTED_OR_RAW = {
    "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value",
    "close_btc", "volume_btc", "quote_volume_btc",
    "last_funding_rate", "funding_z_score", "funding_abs", "oi_change_rate", "btc_corr_60",
    "regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob", "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy", "regime3_current_sensitive_wide24_margin",
}

OMEGA_BASE_COLS_ETH = [
    "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
    "sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio", "last_funding_rate",
    "close_btc", "volume_btc", "quote_volume_btc", "whale_retail_ratio", "whale_conviction", "smart_money_flow",
    "squeeze_power", "oi_change_rate", "net_taker_ratio", "taker_acceleration", "trade_intensity", "big_trade_ratio",
    "log_return", "volatility_z", "rsi", "macd_hist", "bb_width", "bb_width_z", "hma_slope", "wick_ratio",
    "garman_klass_vol", "realized_vol_ratio", "mtf_trend_1h", "mtf_trend_4h", "rogers_satchell_vol", "parkinson_vol",
    "amihud_illiquidity_z", "btc_corr_60", "eth_btc_ratio_change", "fvg_dist", "chop_index", "hour_sin", "hour_cos",
    "minute_sin", "minute_cos", "session_europe", "session_us", "is_hour_open", "cvp_poc_dist", "cvp_vah_val_width",
    "cvp_cluster_position", "cvp_volume_imbalance", "cvp_regime", "turtle_signal", "dual_momentum", "mean_reversion_z",
    "breakout_strength", "volume_profile_signal", "fibonacci_level", "funding_roc_12", "funding_roc_48",
    "funding_roc_288", "funding_z_score", "long_squeeze_risk", "short_squeeze_risk", "funding_price_divergence",
    "hurst_48", "hurst_288", "regime_trending", "ofi_acceleration", "kalman_velocity", "realized_skewness", "ofti",
    "kel", "mta_funding", "svps", "funding_abs", "funding_pressure", "garch_vol_z", "ou_funding_z", "ou_halflife",
    "jump_flag", "jump_z", "evt_tail_flag", "evt_excess_z", "sig_volume_confirm", "sig_liquidity_trap",
    "sig_trend_health", "regime_persistence", "cross_scale_curvature", "liquidity_vacuum", "crowding_pressure",
    "execution_quality",
    "regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob", "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy", "regime3_current_sensitive_wide24_margin",
]
CANDIDATE_COLS = [c for c in OMEGA_BASE_COLS_ETH if c not in ALREADY_TESTED_OR_RAW]


def decile_spread(values: pd.Series, label: pd.Series) -> float:
    valid = values.notna()
    if valid.sum() < 100:
        return float("nan")
    try:
        deciles = pd.qcut(values[valid], 10, duplicates="drop")
    except ValueError:
        return float("nan")
    rates = label[valid].groupby(deciles, observed=True).mean()
    return float(rates.max() - rates.min()) if len(rates) > 1 else float("nan")


def main() -> int:
    frames = []
    for path in TRAINING_FEATURES:
        df = pd.read_csv(path, usecols=["timestamp"] + CANDIDATE_COLS, low_memory=False)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames.append(df)
    features = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")

    labels = pd.read_csv(LABEL_CSV)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    merged = labels.merge(features, on="timestamp", how="left")
    print(f"label events: {len(labels)}  matched: {merged[CANDIDATE_COLS[0]].notna().sum()} "
          f"({100 * merged[CANDIDATE_COLS[0]].notna().mean():.1f}%)")

    rows = []
    for col in CANDIDATE_COLS:
        series = pd.to_numeric(merged[col], errors="coerce")
        n_valid = series.notna().sum()
        if n_valid < 100:
            rows.append({"feature": col, "n_valid": int(n_valid), "pearson_corr": float("nan"), "decile_spread": float("nan")})
            continue
        corr = series.corr(merged["label"])
        spread = decile_spread(series, merged["label"])
        rows.append({"feature": col, "n_valid": int(n_valid), "pearson_corr": corr, "decile_spread": spread})

    table = pd.DataFrame(rows)
    table["abs_corr"] = table["pearson_corr"].abs()
    print(f"\n{len(CANDIDATE_COLS)} candidate features screened (excluding {len(ALREADY_TESTED_OR_RAW)} already-tested/raw columns)\n")
    print("=== sorted by |Pearson corr| ===")
    print(table.sort_values("abs_corr", ascending=False)[["feature", "n_valid", "pearson_corr", "decile_spread"]].to_string(index=False))
    print("\n=== sorted by decile_spread (catches non-linear/hump relationships Pearson can miss) ===")
    print(table.sort_values("decile_spread", ascending=False)[["feature", "n_valid", "pearson_corr", "decile_spread"]].to_string(index=False))

    out_path = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/omega461_feature_correlation_screen.csv"
    table.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
