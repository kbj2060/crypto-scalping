from __future__ import annotations

import json
import os
from typing import Iterable

import pandas as pd

FORBIDDEN_ACTIVE_REGIME_PREFIXES = (
    "clean_regime_2024_unsup_v4_",
    "clean_regime4_2024_unsup_v1_",
)


def _is_forbidden_active_regime_col(col: str) -> bool:
    return str(col).startswith(FORBIDDEN_ACTIVE_REGIME_PREFIXES)

STATE_PRED: list[str] = []
STATE_CONF: list[str] = []
STATE_REGIME = ["regime_chop", "regime_whipsaw", "regime_bull", "regime_bear", "regime_normal"]
STATE_ELITE = [
    "sig_ai_squeeze",
    "sig_whale",
    "sig_oi_divergence",
    "sig_volume_confirm",
    "sig_liquidity_trap",
    "sig_trend_health",
]
STATE_ALPHA = [
    "mtf_trend_1h",
    "mtf_trend_4h",
    "smart_money_flow",
    "taker_acceleration",
    "rogers_satchell_vol",
    "amihud_illiquidity_z",
]
STATE_HIGH_ORDER = [
    "regime_persistence",
    "cross_scale_curvature",
    "liquidity_vacuum",
    "crowding_pressure",
    "execution_quality",
]
STATE_SYNTH = ["ofti", "kel"]
STATE_DIRECTION_ALPHA = [
    "cvd_12",
    "cvd_48",
    "cvd_288",
    "cvd_slope_12",
    "cvd_slope_48",
    "price_cvd_divergence",
    "cvd_breakout_z",
    "btc_ret_1",
    "btc_ret_3",
    "btc_ret_6",
    "btc_ret_12",
    "btc_ret_z_48",
    "eth_btc_ret_spread_12",
    "eth_btc_ret_spread_48",
    "eth_btc_beta_residual_z",
    "btc_lead_eth_follow_gap_3",
    "btc_breakout_eth_lag_dir",
    "btc_volume_impulse_z",
    "btc_eth_volume_rank_spread",
    "btc_impulse_x_eth_beta",
    "bb_width_pct_rank_288",
    "atr_pct_rank_288",
    "compression_score",
    "compression_release_up",
    "compression_release_down",
    "range_contraction_breakout_dir",
    "vwap_dist_24",
    "vwap_dist_96",
    "vwap_dist_288",
    "anchored_vwap_session_dist",
    "vwap_reclaim_flag",
    "vwap_reject_flag",
    "distance_to_day_high_low_pct",
    "funding_oi_divergence",
    "funding_flip_signal",
    "oi_up_price_down",
    "oi_up_price_up",
    "crowded_long_unwind_risk",
    "crowded_short_squeeze_risk",
    "upper_wick_z",
    "lower_wick_z",
    "sweep_prev_high_reclaim",
    "sweep_prev_low_reclaim",
    "failed_breakout_up",
    "failed_breakout_down",
    "cvd_slope_48_x_trend_prob",
    "funding_oi_divergence_x_instability_prob",
    "vwap_reclaim_x_chop_prob",
]

# PatchTST 추론에서 close 외 exog 계산 시 필요한 선행 컬럼들.
# 런타임/학습 프루닝에서 빠지면 funding_pressure KeyError가 발생할 수 있다.
NF_RUNTIME_REQUIRED_COLS = [
    "whale_conviction",
    "funding_roc_12",
    "funding_roc_48",
    "funding_roc_288",
    "funding_abs",
    "funding_pressure",
    "squeeze_power",
    "cvp_vah_val_width",
]

# RL/라이브 공통으로 반드시 유지할 베이스 마켓 컬럼
MARKET_BASE_COLS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "close_btc",
    "volume_btc",
    "quote_volume_btc",
    "sum_open_interest_value",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "last_funding_rate",
    # HMM / 리스크 / 알파 핵심
    "log_return",
    "garch_vol_z",
    "oi_change_rate",
    "jump_z",
    "evt_excess_z",
    "jump_flag",
    "evt_tail_flag",
    "amihud_illiquidity_z",
    "rogers_satchell_vol",
]

# elite_builder.row_to_market_row가 직접 참조하는 원시 컬럼(런타임 필수)
ELITE_BUILDER_REQUIRED_COLS = [
    "whale_retail_ratio",
    "whale_conviction",
    "smart_money_flow",
    "last_funding_rate",
    "net_taker_ratio",
    "taker_acceleration",
    "rsi",
    "wick_ratio",
    "log_return",
    "hurst_48",
    "hurst_288",
    "ofi_acceleration",
    "trade_intensity",
    "funding_price_divergence",
    "short_squeeze_risk",
    "long_squeeze_risk",
    "oi_change_rate",
    "big_trade_ratio",
    "funding_roc_12",
    "funding_roc_288",
    "cvp_cluster_position",
    "fibonacci_level",
    "count_long_short_ratio",
    "btc_corr_60",
    "eth_btc_ratio_change",
    "session_us",
    "session_europe",
    "session_japan",
    "session_europe_open",
    "session_us_open",
    "session_japan_open",
    "hour_cos",
    "cvp_poc_dist",
    "cvp_volume_imbalance",
    "fvg_dist",
    "breakout_strength",
    "squeeze_power",
    "garman_klass_vol",
    "funding_z_score",
    "volatility_z",
    "garch_vol_z",
    "ou_funding_z",
    "ou_halflife",
    "jump_flag",
    "jump_z",
    "evt_tail_flag",
    "evt_excess_z",
]

# Base technical-indicator columns (rsi, macd_hist, session flags, cvd_*, btc_ret_*,
# etc.) that are shared inputs across multiple still-active model contracts (regime3
# current/cmamba/risk, Omega4.6.1/4.6.2, entry-price/quantile-style scoring). These
# small JSON files are column-name manifests only (not model weights or predictions);
# they are the source of truth for "what base features are actually consumed
# somewhere" so active-path pruning does not silently drop a column another live
# model still needs.
_ACTIVE_MODEL_FEATURE_COL_MANIFESTS = [
    "data/ensemble/supervised/trend_xgb.json",
    "data/ensemble/supervised/multi_target_lgbm.json",
    "data/ensemble/supervised/quantile_forest.json",
    "data/ensemble/supervised/entry_price_model.json",
]


def load_active_model_feature_keep(project_root: str | None = None) -> set[str]:
    """Union of feature_cols across still-active model contract manifests."""
    root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keep: set[str] = set()
    for rel_path in _ACTIVE_MODEL_FEATURE_COL_MANIFESTS:
        path = os.path.join(root, rel_path)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            cols = payload.get("feature_cols", [])
            if isinstance(cols, list):
                keep.update(str(c) for c in cols)
        except Exception:
            continue
    return keep


def build_rl_feature_keep(include_entry_price: bool = False) -> set[str]:
    cols = set(MARKET_BASE_COLS)
    cols.update(ELITE_BUILDER_REQUIRED_COLS)
    cols.update(STATE_PRED)
    cols.update(STATE_CONF)
    cols.update(STATE_ELITE)
    cols.update(STATE_ALPHA)
    cols.update(STATE_HIGH_ORDER)
    cols.update(STATE_REGIME)
    cols.update(STATE_SYNTH)
    cols.update(STATE_DIRECTION_ALPHA)
    cols.update(NF_RUNTIME_REQUIRED_COLS)
    return cols


def build_active_feature_keep(
    *,
    include_entry_price: bool = False,
    project_root: str | None = None,
) -> set[str]:
    """RL + 현재 활성 모델 계약 기준 실제 사용 피처 집합."""
    keep = build_rl_feature_keep(include_entry_price=include_entry_price)
    keep.update(load_active_model_feature_keep(project_root=project_root))
    keep = {c for c in keep if not _is_forbidden_active_regime_col(c)}
    return keep


def prune_to_feature_keep(
    df: pd.DataFrame,
    *,
    include_entry_price: bool = False,
    extra_keep: Iterable[str] | None = None,
) -> pd.DataFrame:
    keep = build_rl_feature_keep(include_entry_price=include_entry_price)
    if extra_keep is not None:
        keep.update([str(c) for c in extra_keep])
    keep = {c for c in keep if not _is_forbidden_active_regime_col(c)}
    cols = [c for c in df.columns if c in keep]
    if not cols:
        return df.copy()
    return df[cols].copy()


def prune_to_active_feature_keep(
    df: pd.DataFrame,
    *,
    include_entry_price: bool = False,
    extra_keep: Iterable[str] | None = None,
) -> pd.DataFrame:
    keep = build_active_feature_keep(
        include_entry_price=include_entry_price,
    )
    if extra_keep is not None:
        keep.update([str(c) for c in extra_keep])
    keep = {c for c in keep if not _is_forbidden_active_regime_col(c)}
    cols = [c for c in df.columns if c in keep]
    if not cols:
        return df.copy()
    return df[cols].copy()
