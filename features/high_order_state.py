from __future__ import annotations

import numpy as np
import pandas as pd


HIGH_ORDER_STATE_COLS = [
    "regime_persistence",
    "cross_scale_curvature",
    "liquidity_vacuum",
    "crowding_pressure",
    "execution_quality",
]


def _to_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _safe_zscore(series: pd.Series, window: int, min_periods: int = 20) -> pd.Series:
    roll_mean = series.rolling(window=window, min_periods=min_periods).mean()
    roll_std = series.rolling(window=window, min_periods=min_periods).std()
    return ((series - roll_mean) / roll_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalize_signed(series: pd.Series, scale: float = 1.0) -> pd.Series:
    safe_scale = max(float(scale), 1e-8)
    return pd.Series(np.tanh(series.astype(float) / safe_scale), index=series.index)


def add_high_order_state_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    close = _to_numeric(out, "close")
    log_return = _to_numeric(out, "log_return")
    volume = _to_numeric(out, "volume")
    trades = _to_numeric(out, "trades")
    quote_volume = _to_numeric(out, "quote_volume")
    mtf_1h = _to_numeric(out, "mtf_trend_1h")
    mtf_4h = _to_numeric(out, "mtf_trend_4h")
    hma_slope = _to_numeric(out, "hma_slope")
    chop_index = _to_numeric(out, "chop_index")
    breakout_strength = _to_numeric(out, "breakout_strength")
    amihud = _to_numeric(out, "amihud_illiquidity_z")
    funding_z = _to_numeric(out, "funding_z_score")
    funding_div = _to_numeric(out, "funding_price_divergence")
    oi_change = _to_numeric(out, "oi_change_rate")
    whale_ratio = _to_numeric(out, "whale_retail_ratio")
    whale_conviction = _to_numeric(out, "whale_conviction")
    long_squeeze = _to_numeric(out, "long_squeeze_risk")
    short_squeeze = _to_numeric(out, "short_squeeze_risk")
    cvp_poc_dist = _to_numeric(out, "cvp_poc_dist")
    cvp_cluster_position = _to_numeric(out, "cvp_cluster_position")
    cvp_vol_imb = _to_numeric(out, "cvp_volume_imbalance")
    net_taker = _to_numeric(out, "net_taker_ratio")
    trade_intensity = _to_numeric(out, "trade_intensity")
    wick_ratio = _to_numeric(out, "wick_ratio")
    bb_width_z = _to_numeric(out, "bb_width_z")
    smart_money_flow = _to_numeric(out, "smart_money_flow")
    vwap_dist = _to_numeric(out, "vwap_dist")

    # 1) Regime persistence: how long the current market mode has lasted.
    trend_dir = np.sign((mtf_1h + mtf_4h + hma_slope).fillna(0.0))
    break_dir = np.sign(breakout_strength.fillna(0.0))
    chop_flag = (chop_index > 61.8).astype(float)
    regime_code = pd.Series(np.where(chop_flag > 0.5, 0.0, np.where(break_dir != 0.0, break_dir, trend_dir)), index=out.index)
    regime_change = regime_code.ne(regime_code.shift(1)).cumsum()
    streak = regime_code.groupby(regime_change).cumcount() + 1
    persistence_core = np.log1p(streak.astype(float)) / np.log(49.0)
    persistence_strength = 0.55 * (mtf_1h.abs() + mtf_4h.abs()) + 0.45 * breakout_strength.abs()
    out["regime_persistence"] = _normalize_signed(np.sign(regime_code) * persistence_core * (1.0 + persistence_strength), scale=1.8)

    # 2) Cross-scale curvature: signed second derivative across 5m -> 1h -> 4h trend stack.
    curvature = (mtf_4h - 2.0 * mtf_1h + hma_slope)
    alignment = np.sign(mtf_4h) * np.sign(mtf_1h)
    out["cross_scale_curvature"] = _normalize_signed(curvature * (1.0 + 0.35 * alignment), scale=0.75)

    # 3) Liquidity vacuum: price impulse with weak trade/volume backing.
    impulse = log_return.abs().rolling(window=3, min_periods=1).sum()
    vol_softness = -0.45 * _safe_zscore(volume, 96) - 0.35 * _safe_zscore(trades, 96) - 0.20 * _safe_zscore(quote_volume, 96)
    vacuum_raw = 0.65 * _safe_zscore(impulse, 96) + 0.55 * amihud + 0.30 * wick_ratio + vol_softness
    out["liquidity_vacuum"] = _normalize_signed(vacuum_raw, scale=1.6)

    # 4) Crowding pressure: funding + OI + crowding/squeeze measures.
    crowd_raw = (
        0.35 * funding_z
        + 0.25 * _safe_zscore(oi_change, 96)
        + 0.25 * _safe_zscore(whale_ratio, 96)
        + 0.15 * _safe_zscore(whale_conviction, 96)
        + 0.20 * long_squeeze
        - 0.20 * short_squeeze
        + 0.20 * funding_div
    )
    out["crowding_pressure"] = _normalize_signed(crowd_raw, scale=1.7)

    # 5) Execution quality: anchored, liquid, non-toxic entry environment.
    anchor_quality = -0.55 * cvp_poc_dist.abs() - 0.20 * (cvp_cluster_position - 0.5).abs() + 0.20 * cvp_vol_imb.abs()
    flow_quality = 0.25 * smart_money_flow - 0.25 * net_taker.abs() + 0.20 * _safe_zscore(trade_intensity, 96)
    fric_quality = -0.35 * out["liquidity_vacuum"] - 0.20 * bb_width_z.abs() + 0.15 * wick_ratio - 0.15 * vwap_dist.abs()
    out["execution_quality"] = _normalize_signed(anchor_quality + flow_quality + fric_quality, scale=1.4)

    for col in HIGH_ORDER_STATE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out
