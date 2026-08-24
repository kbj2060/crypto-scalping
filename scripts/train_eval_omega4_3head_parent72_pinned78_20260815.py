"""RESEARCH ONLY -- fork of scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py that
pins base_cols to a DEDUPLICATED 78-column subset of the live 102-column contract instead of the
live 102 verbatim.

Why. `docs/experiments/eth_omega461_live_102feature_redundancy_audit_20260815.md` found 14
connected-components (Spearman |corr| > 0.9) covering 38/102 live base_cols -- including exact
duplicates (r=1.0000: smart_money_flow==oi_change_rate, funding_z_score==ou_funding_z). This
script drops 24 of those 38 (one representative per cluster kept, chosen deterministically as
whichever cluster member appears FIRST in the live bundle's base_cols order -- decided before any
training ran), leaving 78 columns, to test whether removing near-duplicate features changes the
h48qual/zig075 direction_head no-skill wall documented in
`docs/experiments/eth_omega461_zig075_direction_head_skill_formal_nseed_20260815.md` (102-feature
result: zig075 10/10 losses vs always_short across 5 genuinely random seeds).

Everything else is unchanged from the pinned102 wrapper: same two monkeypatches (train-side 2025
column repair for the 6 REPAIR_COLS that survive dedup -- fibonacci_level was itself dropped by
dedup, so it is no longer needed and is not repaired here; _numeric_feature_cols now returns the
fixed REDUCED_78_COLS list instead of a bundle's base_cols) and the same underlying trainer
(architecture, labels, hyperparameters, epochs, split) via
scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py. Does NOT touch
trading_bot_modules/, trading_bot.py, runtime_config.py, .env, or any live checkpoint.

Usage (identical CLI to pinned102, minus --pin-component which pinned102 needs to pick which live
bundle's base_cols to copy -- this script has only one fixed 78-column list, shared by both
h48qual and zig075 since the source 102 was already identical & shared between them):
  python scripts/train_eval_omega4_3head_parent72_pinned78_20260815.py \\
      --epochs 2 --quality-mode same_as_direction \\
      --direction-label-dir tmp/.../zigzag_action_labels_20260531 \\
      --quality-thresholds 0.55,...,0.95 --max-exit-samples 30000 --max-train-rows 0 \\
      --exit-label-mode entry_label_terminal_giveback \\
      --out-suffix pinned78_zig075_dedup_seed<SEED> --device cpu --seed <SEED>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

omega = parent_script.omega

# Deterministic dedup of the live 102-column contract: for each of the 14 |corr|>0.9 clusters
# found in the 20260815 redundancy audit, keep only the member that appears FIRST in the live
# bundle's base_cols order (index printed from true_3head_tabm_bundle.pt["base_cols"], both
# h48qual and zig075 bundles share the identical 102-column list/order). 24 features dropped,
# 78 kept. Full before/after list is reproduced in
# docs/experiments/eth_omega461_dedup78feature_nseed_skill_retest_20260815.md.
REDUCED_78_COLS = [
    "open", "volume", "sum_toptrader_long_short_ratio", "count_long_short_ratio",
    "last_funding_rate", "close_btc", "volume_btc", "whale_retail_ratio", "whale_conviction",
    "smart_money_flow", "net_taker_ratio", "taker_acceleration", "trade_intensity",
    "big_trade_ratio", "log_return", "volatility_z", "rsi", "macd_hist", "bb_width",
    "bb_width_z", "hma_slope", "wick_ratio", "garman_klass_vol", "realized_vol_ratio",
    "mtf_trend_1h", "amihud_illiquidity_z", "btc_corr_60", "eth_btc_ratio_change", "fvg_dist",
    "chop_index", "hour_sin", "hour_cos", "minute_sin", "minute_cos", "session_europe",
    "session_us", "is_hour_open", "cvp_poc_dist", "cvp_vah_val_width", "cvp_cluster_position",
    "cvp_volume_imbalance", "cvp_regime", "turtle_signal", "dual_momentum", "mean_reversion_z",
    "funding_roc_12", "funding_roc_48", "funding_roc_288", "funding_z_score",
    "short_squeeze_risk", "funding_price_divergence", "hurst_48", "hurst_288",
    "regime_trending", "ofi_acceleration", "realized_skewness", "ofti", "kel", "mta_funding",
    "svps", "funding_pressure", "garch_vol_z", "ou_halflife", "jump_flag", "evt_tail_flag",
    "sig_volume_confirm", "sig_liquidity_trap", "sig_trend_health", "regime_persistence",
    "cross_scale_curvature", "liquidity_vacuum", "crowding_pressure", "execution_quality",
    "regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob", "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
]
assert len(REDUCED_78_COLS) == 78, len(REDUCED_78_COLS)
assert len(set(REDUCED_78_COLS)) == 78, "duplicate column in REDUCED_78_COLS"

# Present in the live 102-col contract but dropped from today's 2025 candidate CSV (same repair
# set as pinned102, minus fibonacci_level -- fibonacci_level was itself dropped by dedup above,
# so no longer part of the input contract and does not need repairing).
REPAIR_COLS = ["funding_roc_12", "funding_roc_48", "funding_z_score",
               "short_squeeze_risk", "hurst_288", "regime_persistence"]
REPAIR_SOURCE = ROOT / "data/splits/year_oos/training_features_2025.csv"

_orig_load_frames = omega._load_omega_frames


def _repair_train_columns(frame: pd.DataFrame) -> pd.DataFrame:
    need = [c for c in REPAIR_COLS if c not in frame.columns]
    if not need:
        return frame
    src = pd.read_csv(REPAIR_SOURCE, usecols=["timestamp", *need], low_memory=False)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    src = src.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last")
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"])
    joined = src.set_index("timestamp").reindex(ts)
    for c in need:
        vals = pd.to_numeric(joined[c], errors="coerce").to_numpy()
        if pd.isna(vals).any():
            raise RuntimeError(f"pinned78 repair: {c} has {int(pd.isna(vals).sum())} missing values after join")
        out[c] = vals
    print(f"[pinned78] repaired {len(need)} train columns from {REPAIR_SOURCE.name}: {need}", flush=True)
    return out


def _patched_load_frames():
    train_all, eval_df, overlay_report = _orig_load_frames()
    return _repair_train_columns(train_all), eval_df, overlay_report


def _patched_numeric_feature_cols(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    missing_train = [c for c in REDUCED_78_COLS if c not in train_df.columns]
    missing_eval = [c for c in REDUCED_78_COLS if c not in eval_df.columns]
    if missing_train or missing_eval:
        raise RuntimeError(
            f"pinned78: reduced base_cols unavailable (train missing {missing_train}, eval missing {missing_eval})")
    return list(REDUCED_78_COLS)


def _install_pin() -> None:
    print(f"[pinned78] pinning base_cols to deduplicated 78-column set (dropped 24/102 near-duplicates)", flush=True)
    omega._load_omega_frames = _patched_load_frames
    omega._numeric_feature_cols = _patched_numeric_feature_cols


def main() -> int:
    _install_pin()
    return parent_script.main()


if __name__ == "__main__":
    raise SystemExit(main())
