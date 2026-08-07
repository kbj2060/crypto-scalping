"""Build ETH TRAIN_CSV/EVAL_CSV variants for the session-open-feature A/B test.

The current data/splits/year_oos/trade_candidates_*.csv-derived files
(tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/) have
accreted ~75 extra columns since the live zig075 bundle was actually trained
(2026-06-29) -- m7/regime4/clean-funding overlay columns that were not part of
its true feature contract. The live bundle's own base_cols (extracted from
true_3head_tabm_bundle.pt) is the ground-truth 102-column list: 96 plain
FeatureEngineer columns + 6 regime3_current_sensitive_wide24_* overlay columns
(the latter get re-attached separately by the parent trainer's
_overlay_required() from REGIME3_CURRENT_2025/2026, so they're excluded here
to avoid duplicate-column collisions).

This script produces two column-restricted CSV pairs:
  - baseline: exactly the live 96 non-regime3 base columns (+ timestamp) --
    the true production feature set, unchanged.
  - candidate: the same 96 columns + the 4 new session-open columns
    (session_japan, session_europe_open, session_us_open, session_japan_open;
    session_europe/session_us are ALREADY in the live 102, confirmed via
    bundle inspection 2026-07-30).

Both draw from data/splits/session_open_dummy_eth_20260730/ (which already has
the 4 new columns joined on) so only column-subsetting happens here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data/splits/session_open_dummy_eth_20260730"
OUT_DIR = ROOT / "data/splits/session_open_dummy_eth_20260730"

SRC_FILES = {
    2025: SRC_DIR / "trade_candidates_2025_alpha6_current_tail111_exact.csv",
    2026: SRC_DIR / "trade_candidates_2026_alpha6_current_tail111_exact.csv",
}

# Live zig075 bundle's 102 base_cols, minus the 6 regime3_current_* columns
# (those are re-attached by the parent trainer's own overlay step).
LIVE_96_BASE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'sum_open_interest_value', 'sum_toptrader_long_short_ratio',
    'count_long_short_ratio', 'last_funding_rate', 'close_btc', 'volume_btc', 'quote_volume_btc',
    'whale_retail_ratio', 'whale_conviction', 'smart_money_flow', 'squeeze_power', 'oi_change_rate',
    'net_taker_ratio', 'taker_acceleration', 'trade_intensity', 'big_trade_ratio', 'log_return',
    'volatility_z', 'rsi', 'macd_hist', 'bb_width', 'bb_width_z', 'hma_slope', 'wick_ratio',
    'garman_klass_vol', 'realized_vol_ratio', 'mtf_trend_1h', 'mtf_trend_4h', 'rogers_satchell_vol',
    'parkinson_vol', 'amihud_illiquidity_z', 'btc_corr_60', 'eth_btc_ratio_change', 'fvg_dist',
    'chop_index', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'session_europe', 'session_us',
    'is_hour_open', 'cvp_poc_dist', 'cvp_vah_val_width', 'cvp_cluster_position', 'cvp_volume_imbalance',
    'cvp_regime', 'turtle_signal', 'dual_momentum', 'mean_reversion_z', 'breakout_strength',
    'volume_profile_signal', 'fibonacci_level', 'funding_roc_12', 'funding_roc_48', 'funding_roc_288',
    'funding_z_score', 'long_squeeze_risk', 'short_squeeze_risk', 'funding_price_divergence',
    'hurst_48', 'hurst_288', 'regime_trending', 'ofi_acceleration', 'kalman_velocity',
    'realized_skewness', 'ofti', 'kel', 'mta_funding', 'svps', 'funding_abs', 'funding_pressure',
    'garch_vol_z', 'ou_funding_z', 'ou_halflife', 'jump_flag', 'jump_z', 'evt_tail_flag',
    'evt_excess_z', 'sig_volume_confirm', 'sig_liquidity_trap', 'sig_trend_health',
    'regime_persistence', 'cross_scale_curvature', 'liquidity_vacuum', 'crowding_pressure',
    'execution_quality',
]
NEW_SESSION_COLS = ["session_japan", "session_europe_open", "session_us_open", "session_japan_open"]

assert len(LIVE_96_BASE_COLS) == 96, len(LIVE_96_BASE_COLS)


def _subset(year: int, cols: list[str], out_name: str) -> None:
    header = pd.read_csv(SRC_FILES[year], nrows=0).columns
    avail = [c for c in cols if c in header]
    df = pd.read_csv(SRC_FILES[year], low_memory=False, usecols=["timestamp"] + avail)
    df = df[["timestamp"] + avail]
    out_path = OUT_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"{year} -> {out_path}: {len(df)} rows, {len(df.columns)} cols", flush=True)


def main() -> int:
    # 7 of the original 96 columns no longer exist in the current feature
    # pipeline output (features/engineering.py drifted since the live bundle
    # was trained 2026-06-29): fibonacci_level, funding_roc_12, funding_roc_48,
    # funding_z_score, short_squeeze_risk, hurst_288, regime_persistence.
    # Both baseline and candidate use the same available 89-column base so the
    # comparison stays apples-to-apples; only the candidate adds the 4 new
    # session-open columns.
    available_base = None
    for year in (2025, 2026):
        header = pd.read_csv(SRC_FILES[year], nrows=0).columns
        avail = [c for c in LIVE_96_BASE_COLS if c in header]
        available_base = avail if available_base is None else [c for c in available_base if c in avail]
    missing = [c for c in LIVE_96_BASE_COLS if c not in available_base]
    print(f"using {len(available_base)}/{len(LIVE_96_BASE_COLS)} live base cols (missing: {missing})", flush=True)

    for year in (2025, 2026):
        _subset(year, available_base, f"baseline96_trade_candidates_{year}.csv")
        _subset(year, available_base + NEW_SESSION_COLS, f"candidate100_trade_candidates_{year}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
