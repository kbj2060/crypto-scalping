# Unified Pipeline Design

## Goal

Reduce the current multi-script training flow into three user-facing steps:

1. Build shared features and year-based datasets
2. Train M7 ensemble and augment RL data with `m7_*`
3. Train DSAC and load the same schema in `trading_bot.py`

This design does not delete the legacy scripts yet. Instead, it adds a thin
orchestration layer that reuses them while we gradually move duplicated logic
into shared modules.

## Current Pain Points

- Feature computation is duplicated across `FeatureEngineer`,
  `scripts/generate_training_data.py`, `scripts/augment_rl_training_with_model7.py`,
  and `trading_bot.py`.
- The user has to understand multiple intermediate CSV files.
- The split between supervised training and RL training is valid, but the
  boundaries are exposed too directly to the operator.
- Runtime and training feature paths are conceptually aligned but physically
  scattered.

## Target User-Facing Flow

### Step 1. Build Dataset

Entrypoint: `pipeline/build_rl_dataset.py`

Responsibilities:

- Optionally generate `rl_training_data_full.csv`
- Split feature and RL base data by year
- Save:
  - `training_features_<sup_year>.csv`
  - `training_features_<rl_year>.csv`
  - `rl_base_<sup_year>.csv`
  - `rl_base_<rl_year>.csv`
- Train ensemble models on the supervised year
- Augment RL year data with `SevenModelEnsemble`
- Produce final:
  - `rl_training_<rl_year>_m7.csv`

### Step 2. Train RL

Entrypoint: `pipeline/run_train.py`

Responsibilities:

- Call `pipeline/build_rl_dataset.py`
- Train one RL agent family:
  - `iqn`
  - `sac`
  - `dsac`

### Step 3. Live Trading

Entrypoint: `trading_bot.py`

Responsibilities:

- Use `FeatureEngineer` as the primary runtime feature engine
- Use the same M7 inference path as training augmentation
- Load DSAC checkpoint trained from `rl_training_<year>_m7.csv`
- Apply execution overlays after DSAC output

## Migration Rules

### Rule 1. One shared feature engine

The long-term source of truth should be `features/engineering.py`.

Anything currently recalculated in:

- `scripts/generate_training_data.py`
- `scripts/augment_rl_training_with_model7.py`
- `trading_bot.py`

should move into reusable feature-layer helpers over time.

### Rule 2. One RL dataset builder

`pipeline/build_rl_dataset.py` becomes the official orchestration layer for all
pre-DSAC work. Legacy scripts can remain as implementation details until their
logic is absorbed into shared modules.

### Rule 3. One final RL dataset contract

The final handoff into RL training should always be:

- `rl_training_<year>_m7.csv`

This file is the contract between data prep and RL training.

## Files Added In This Phase

- `pipeline/__init__.py`
- `pipeline/build_rl_dataset.py`
- `pipeline/run_train.py`
- `docs/unified_pipeline_design.md`

## What This Phase Does Not Yet Change

- It does not remove legacy scripts
- It does not fully deduplicate feature computation
- It does not change `trading_bot.py` runtime internals yet

This phase is intentionally a low-risk facade layer so the team can begin using
the simpler entrypoints immediately, then migrate internals behind them.

## Final Feature Contract

This section defines the intended long-term feature contract for the unified
pipeline. The goal is to keep one shared feature engine, use M7 as a compact
summary layer, and keep the DSAC input state smaller and more stable than the
current raw feature spread.

### 1. Shared Base Features

These are the reusable raw/runtime features that should be produced by the
shared feature engine and then selectively consumed by supervised M7 models and
runtime augmentation.

| Group | Keep |
|---|---|
| Price / returns | `open`, `high`, `low`, `close`, `log_return` |
| Volume / execution | `volume`, `quote_volume`, `trades`, `taker_buy_base`, `taker_buy_quote`, `trade_intensity`, `big_trade_ratio`, `net_taker_ratio`, `taker_acceleration`, `ofi_acceleration` |
| OI / funding raw | `sum_open_interest_value`, `last_funding_rate`, `funding_price_divergence` |
| BTC-relative | `close_btc`, `btc_corr_60`, `eth_btc_ratio_change` |
| Volatility raw | `garman_klass_vol`, `rogers_satchell_vol`, `parkinson_vol`, `bb_width_z`, `realized_skewness`, `amihud_illiquidity_z` |
| Trend / structure | `rsi`, `macd_hist`, `mtf_trend_1h`, `mtf_trend_4h`, `turtle_signal`, `squeeze_power`, `chop_index`, `wick_ratio`, `fvg_dist` |
| CVP / profile | `cvp_poc_dist`, `cvp_cluster_position`, `cvp_volume_imbalance`, `cvp_regime`, `volume_profile_signal` |
| Flow / whale | `smart_money_flow`, `whale_retail_ratio`, `whale_conviction`, `long_squeeze_risk` |
| High-order state | `regime_persistence`, `cross_scale_curvature`, `liquidity_vacuum`, `crowding_pressure`, `execution_quality` |

### 2. M7 Supervised Inputs

M7 should be treated as a summary layer for DSAC, not as a standalone trading
engine. The supervised models should therefore focus on robust structure,
distribution, and execution-quality signals.

| Model | Preferred Inputs |
|---|---|
| `trend_xgb` | `mtf_trend_1h`, `mtf_trend_4h`, `rsi`, `macd_hist`, `turtle_signal`, `btc_corr_60`, `bb_width_z`, `realized_skewness`, `squeeze_power`, `funding_price_divergence`, `smart_money_flow`, `net_taker_ratio`, `cvp_cluster_position`, `regime_persistence`, `cross_scale_curvature`, `sig_volume_confirm`, `sig_trend_health` |
| `multi_target_lgbm` | `btc_corr_60`, `whale_retail_ratio`, `mtf_trend_1h`, `mtf_trend_4h`, `bb_width_z`, `realized_vol_ratio`, `squeeze_power`, `turtle_signal`, `funding_price_divergence`, `smart_money_flow`, `crowding_pressure`, `execution_quality`, `trade_intensity`, `long_squeeze_risk` |
| `quantile_forest` | `rsi`, `mtf_trend_1h`, `mtf_trend_4h`, `bb_width_z`, `garman_klass_vol`, `rogers_satchell_vol`, `amihud_illiquidity_z`, `btc_corr_60`, `smart_money_flow`, `taker_acceleration`, `liquidity_vacuum`, `cvp_regime`, `cvp_poc_dist`, `whale_retail_ratio`, `fvg_dist` |
| `entry_price_model` | `cvp_poc_dist`, `cvp_cluster_position`, `cvp_volume_imbalance`, `bb_width_z`, `amihud_illiquidity_z`, `execution_quality`, `smart_money_flow`, `net_taker_ratio`, `trade_intensity`, `fvg_dist`, `wick_ratio`, `mtf_trend_1h` |

### 3. Removed M7 Unsupervised Layer

The active M7 generation and required-column contract no longer includes the
`gmm_volatility`, `isolation_forest`, or `vae_anomaly` model/meta keys. It also
does not generate or require `m7_gmm_*`, `m7_iso_*`, `m7_vae_*`, `m7_gate_block`,
`m7_size`, or `m7_hdb_*` columns.

Historical artifacts that still contain those columns are diagnostic-only until
retrained/rescored under the current active contract.

### 4. M7 Outputs Passed Forward

The RL handoff contract should not expose the full upstream feature sprawl.
Instead, M7 should emit compact outputs that summarize directional bias,
uncertainty, horizon, and risk.

| Group | Keep |
|---|---|
| Action context | `m7_action`, `m7_confidence` |
| Quantiles / uncertainty | `m7_q10`, `m7_q50`, `m7_q90`, `m7_qwidth`, `m7_expected_ret`, `m7_tail_risk`, `m7_composite_score` |
| Execution | `m7_entry_long_offset`, `m7_entry_short_offset`, `m7_tp_offset`, `m7_sl_offset`, `m7_target_hold` |

### 5. DSAC Compact State

DSAC should not absorb the entire raw training feature matrix. It should see a
small, stable compact state built from position state, core trend/risk context,
and M7 summaries.

| Group | Keep |
|---|---|
| Position state | `current_position`, `unrealized_pnl_norm`, `time_in_trade_norm`, `drawdown_norm`, `margin_usage` |
| Core market state | `mtf_trend_1h_norm`, `mtf_trend_4h_norm`, `rogers_satchell_vol_norm`, `micro_vol5_norm`, `spread_norm`, `amihud_norm`, `smart_money_flow_norm` |

### 6. Deletion / Cleanup Priorities

The following features or outputs should be removed, disabled, or treated as
cleanup priorities before new RL conclusions are trusted:

| Priority | Candidate | Reason |
|---|---|---|
| High | `pred_ttm`, `pred_timesfm`, `conf_ttm`, `conf_timesfm` | Suspected duplication with `patchtst` family |
| High | `meta_primary_std` | Currently constant / fallback-like |
| High | `hdbscan_regime` | Low contribution and easy to disable |
| High | `funding_roc_48`, `funding_roc_12`, `funding_z_score` | Repeatedly harmful in M7-only ablation tests |
| Medium | `session_us`, `session_europe`, `is_hour_open` | Repeated low-importance time/session dummies |
| Medium | `regime_trending`, `mta_funding`, `dual_momentum` | Weak contribution candidates |
| Validate | `garch_vol_z` | Suspected dead/degenerate feature |

### 7. Implementation Note

When this contract is enforced in code, `pipeline/build_rl_dataset.py` should
grow support for a feature keep/drop manifest rather than hardcoding these rules
inside multiple scripts. The matching machine-readable manifest for this design
is stored in:

- `docs/feature_contract_manifest.json`
