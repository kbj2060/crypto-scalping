from __future__ import annotations

import json
import os
from typing import Iterable

import pandas as pd
from .registry import get_m7_columns

STATE_PRED = ["pred_patchtst", "pred_chronos", "pred_tide"]
STATE_CONF = ["conf_patchtst", "conf_chronos", "conf_tide"]
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
STATE_SYNTH = ["ofti", "kel"]

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
    "count_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "btc_corr_60",
    "eth_btc_ratio_change",
    "session_us",
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

# RL keep-set에서 유지해야 하는 m7 파생 컬럼(SSOT: features.registry)
M7_STATE_COLS = sorted(get_m7_columns("rl_keep", include_entry_price=True))


def build_rl_feature_keep(include_entry_price: bool = False) -> set[str]:
    cols = set(MARKET_BASE_COLS)
    cols.update(ELITE_BUILDER_REQUIRED_COLS)
    cols.update(STATE_PRED)
    cols.update(STATE_CONF)
    cols.update(STATE_ELITE)
    cols.update(STATE_ALPHA)
    cols.update(STATE_REGIME)
    cols.update(STATE_SYNTH)
    cols.update(get_m7_columns("rl_keep", include_entry_price=include_entry_price))
    return cols


def load_m7_model_feature_keep(project_root: str | None = None) -> set[str]:
    """현재 아티팩트가 실제로 요구하는 M7 feature_cols 집합."""
    root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_paths = [
        os.path.join(root, "data", "ensemble", "supervised", "trend_xgb.json"),
        os.path.join(root, "data", "ensemble", "supervised", "multi_target_lgbm.json"),
        os.path.join(root, "data", "ensemble", "supervised", "quantile_forest.json"),
        os.path.join(root, "data", "ensemble", "supervised", "entry_price_model.json"),
        os.path.join(root, "data", "ensemble", "unsupervised", "gmm_volatility.json"),
        os.path.join(root, "data", "ensemble", "unsupervised", "hdbscan_regime.json"),
        os.path.join(root, "data", "ensemble", "unsupervised", "isolation_forest.json"),
        os.path.join(root, "data", "ensemble", "unsupervised", "vae_anomaly.json"),
    ]
    keep: set[str] = set()
    for path in meta_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            cols = payload.get("feature_cols", [])
            if isinstance(cols, list):
                keep.update([str(c) for c in cols])
        except Exception:
            continue
    return keep


def build_active_feature_keep(
    *,
    include_entry_price: bool = False,
    include_m7_artifacts: bool = True,
    project_root: str | None = None,
) -> set[str]:
    """RL + (선택)현재 M7 아티팩트 기준 실제 사용 피처 집합."""
    keep = build_rl_feature_keep(include_entry_price=include_entry_price)
    if include_m7_artifacts:
        keep.update(load_m7_model_feature_keep(project_root=project_root))
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
    cols = [c for c in df.columns if c in keep]
    if not cols:
        return df.copy()
    return df[cols].copy()


def prune_to_active_feature_keep(
    df: pd.DataFrame,
    *,
    include_entry_price: bool = False,
    include_m7_artifacts: bool = True,
    extra_keep: Iterable[str] | None = None,
    project_root: str | None = None,
) -> pd.DataFrame:
    keep = build_active_feature_keep(
        include_entry_price=include_entry_price,
        include_m7_artifacts=include_m7_artifacts,
        project_root=project_root,
    )
    if extra_keep is not None:
        keep.update([str(c) for c in extra_keep])
    cols = [c for c in df.columns if c in keep]
    if not cols:
        return df.copy()
    return df[cols].copy()
