"""Category split of the causalfix_final 114-col BTC 5m panel, for the new deep-feature
(CNN/Transformer) architecture line. Groups the 113 feature columns (all cols except
`timestamp`) into named categories so downstream encoders can treat categories as distinct
input groups (e.g. per-category embedding before a category-axis CNN).

This is a fresh standalone line -- does not reuse or extend the closed causalfix_final
quality-classifier lineage (docs/btc_panel_crossasset_architecture_design_20260804.md) or the
closed JEPA deep-feature encoder (ensemble/deep_features/tabular_jepa_encoder.py); only the
raw panel and OHLC are shared inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
MANIFEST_OUT = ROOT / "docs/model_contracts/btc_feature_categories_20260806.json"

CATEGORY_MAP: dict[str, list[str]] = {
    "ohlcv_raw": [
        "open", "high", "low", "close", "volume", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote",
    ],
    "derivatives_funding": [
        "sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio",
        "last_funding_rate", "funding_roc_288", "funding_abs", "funding_pressure",
        "long_squeeze_risk", "short_squeeze_risk", "funding_price_divergence", "oi_up_price_down",
    ],
    "orderflow_microstructure": [
        "whale_retail_ratio", "whale_conviction", "smart_money_flow", "squeeze_power",
        "taker_acceleration", "trade_intensity", "big_trade_ratio", "ofi_acceleration", "ofti",
        "cvd_12", "cvd_48", "cvd_288", "cvd_slope_12", "price_cvd_divergence", "cvd_breakout_z",
        "cvp_poc_dist", "cvp_vah_val_width", "cvp_cluster_position", "cvp_volume_imbalance",
        "cvp_regime", "amihud_illiquidity_z",
    ],
    "volatility": [
        "volatility_z", "bb_width", "bb_width_z", "bb_width_pct_rank_288", "garman_klass_vol",
        "realized_vol_ratio", "atr_pct_rank_288", "garch_vol_z", "ou_halflife",
    ],
    "momentum_trend": [
        "log_return", "rsi", "macd_hist", "hma_slope", "dual_momentum", "mean_reversion_z",
        "turtle_signal", "mtf_trend_1h", "mtf_trend_4h", "compression_release_up",
        "range_contraction_breakout_dir",
    ],
    "cross_asset_eth": [
        "close_btc", "volume_btc", "quote_volume_btc", "btc_ret_1", "btc_ret_3", "btc_ret_12",
        "eth_btc_ret_spread_12", "eth_btc_ret_spread_48", "eth_btc_beta_residual_z",
        "btc_lead_eth_follow_gap_3", "btc_volume_impulse_z", "btc_eth_volume_rank_spread",
    ],
    "regime": [
        "chop_index", "hurst_48", "hurst_288", "regime_persistence", "cross_scale_curvature",
        "liquidity_vacuum", "crowding_pressure", "execution_quality",
        "regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob",
        "regime3_current_sensitive_wide24_chop_prob", "regime3_current_sensitive_wide24_confidence",
    ],
    "structural_price_location": [
        "wick_ratio", "upper_wick_z", "lower_wick_z", "fvg_dist", "volume_profile_signal",
        "fibonacci_level", "vwap_dist_24", "vwap_dist_96", "vwap_dist_288",
        "distance_to_day_high_low_pct", "realized_skewness", "kel", "svps",
        "sig_volume_confirm", "sig_trend_health",
    ],
    "time_session": ["hour_sin", "hour_cos", "session_us"],
    "mtf_1h_sidecar": [
        "mtf1h_ts_t_value", "mtf1h_ts_opt_L", "mtf1h_rsi_14", "mtf1h_rvol_6", "mtf1h_rvol_12",
        "mtf1h_rvol_24", "mtf1h_rvol_48", "mtf1h_atr_pct", "mtf1h_bb_width", "mtf1h_bb_pos",
    ],
}

# Fixed category order -- encoders that build a (n_categories, ...) tensor must use this order.
CATEGORY_ORDER: list[str] = list(CATEGORY_MAP.keys())


def feature_columns() -> list[str]:
    cols: list[str] = []
    for cat in CATEGORY_ORDER:
        cols.extend(CATEGORY_MAP[cat])
    return cols


def category_of(column: str) -> str:
    for cat, cols in CATEGORY_MAP.items():
        if column in cols:
            return cat
    raise KeyError(column)


def _validate_against_panel(panel_cols: list[str]) -> None:
    mapped = feature_columns()
    if len(mapped) != len(set(mapped)):
        dupes = {c for c in mapped if mapped.count(c) > 1}
        raise RuntimeError(f"CATEGORY_MAP has duplicate columns: {sorted(dupes)}")
    panel_feature_cols = [c for c in panel_cols if c != "timestamp"]
    missing_from_map = sorted(set(panel_feature_cols) - set(mapped))
    extra_in_map = sorted(set(mapped) - set(panel_feature_cols))
    if missing_from_map or extra_in_map:
        raise RuntimeError(
            "CATEGORY_MAP out of sync with causalfix_final panel columns: "
            f"missing_from_map={missing_from_map} extra_in_map={extra_in_map}"
        )


def main() -> int:
    panel_cols = list(pd.read_parquet(PANEL_PATH).columns)
    _validate_against_panel(panel_cols)

    manifest = {
        "source_panel": str(PANEL_PATH.relative_to(ROOT)),
        "n_categories": len(CATEGORY_ORDER),
        "n_feature_columns": len(feature_columns()),
        "category_order": CATEGORY_ORDER,
        "category_sizes": {cat: len(cols) for cat, cols in CATEGORY_MAP.items()},
        "category_map": CATEGORY_MAP,
    }
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
