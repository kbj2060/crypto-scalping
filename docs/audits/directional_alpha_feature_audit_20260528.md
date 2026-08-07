# Directional Alpha Feature Audit - 2026-05-28

## Scope

This report audits the 48 newly added direction-oriented features in `features/engineering.py`, including BTC lead-lag features.

Artifacts:

- Feature scores: `tmp/causal_regen_20260516/directional_alpha_feature_audit_20260528/directional_alpha_feature_scores.csv`
- Redundancy pairs: `tmp/causal_regen_20260516/directional_alpha_feature_audit_20260528/directional_alpha_redundancy_abs090.csv`
- Summary: `tmp/causal_regen_20260516/directional_alpha_feature_audit_20260528/summary.json`

## Verdict Counts

| Verdict | Count |
|---|---:|
| `SECONDARY_CONTEXT` | 29 |
| `LOW_SIGNAL_SECONDARY` | 15 |
| `KEEP_RISK_CONTEXT` | 4 |

## Top Return-Tendency Features

| Feature | Verdict | Ret IC | Vol IC | PSI | Reason |
|---|---|---:|---:|---:|---|
| `compression_score` | `KEEP_RISK_CONTEXT` | 0.072 | 0.205 | 0.002 | stronger risk/volatility utility than direction |
| `atr_pct_rank_288` | `KEEP_RISK_CONTEXT` | 0.069 | 0.212 | 0.008 | stronger risk/volatility utility than direction |
| `bb_width_pct_rank_288` | `KEEP_RISK_CONTEXT` | 0.063 | 0.184 | 0.003 | stronger risk/volatility utility than direction |
| `vwap_dist_96` | `SECONDARY_CONTEXT` | 0.062 | 0.085 | 0.046 | moderate context; require ablation before promotion |
| `cvd_288` | `SECONDARY_CONTEXT` | 0.060 | 0.111 | 0.090 | moderate context; require ablation before promotion |
| `eth_btc_ret_spread_12` | `SECONDARY_CONTEXT` | 0.058 | 0.042 | 0.094 | moderate context; require ablation before promotion |
| `crowded_short_squeeze_risk` | `SECONDARY_CONTEXT` | 0.056 | 0.035 | 0.000 | moderate context; require ablation before promotion |
| `btc_lead_eth_follow_gap_3` | `SECONDARY_CONTEXT` | 0.052 | 0.032 | 0.104 | moderate context; require ablation before promotion |
| `anchored_vwap_session_dist` | `SECONDARY_CONTEXT` | 0.050 | 0.094 | 0.052 | moderate context; require ablation before promotion |
| `btc_volume_impulse_z` | `KEEP_RISK_CONTEXT` | 0.050 | 0.193 | 0.000 | stronger risk/volatility utility than direction |
| `range_contraction_breakout_dir` | `SECONDARY_CONTEXT` | 0.049 | 0.036 | 0.000 | moderate context; require ablation before promotion |
| `distance_to_day_high_low_pct` | `SECONDARY_CONTEXT` | 0.049 | 0.106 | 0.030 | moderate context; require ablation before promotion |
| `vwap_dist_24` | `SECONDARY_CONTEXT` | 0.047 | 0.064 | 0.026 | moderate context; require ablation before promotion |
| `price_cvd_divergence` | `SECONDARY_CONTEXT` | 0.042 | 0.068 | 0.049 | moderate context; require ablation before promotion |
| `funding_oi_divergence` | `SECONDARY_CONTEXT` | 0.041 | 0.033 | 0.027 | moderate context; require ablation before promotion |

## Top Risk/Volatility-Tendency Features

| Feature | Verdict | Ret IC | Vol IC | PSI | Reason |
|---|---|---:|---:|---:|---|
| `atr_pct_rank_288` | `KEEP_RISK_CONTEXT` | 0.069 | 0.212 | 0.008 | stronger risk/volatility utility than direction |
| `compression_score` | `KEEP_RISK_CONTEXT` | 0.072 | 0.205 | 0.002 | stronger risk/volatility utility than direction |
| `btc_volume_impulse_z` | `KEEP_RISK_CONTEXT` | 0.050 | 0.193 | 0.000 | stronger risk/volatility utility than direction |
| `bb_width_pct_rank_288` | `KEEP_RISK_CONTEXT` | 0.063 | 0.184 | 0.003 | stronger risk/volatility utility than direction |
| `vwap_dist_288` | `SECONDARY_CONTEXT` | 0.029 | 0.119 | 0.049 | moderate context; require ablation before promotion |
| `cvd_288` | `SECONDARY_CONTEXT` | 0.060 | 0.111 | 0.090 | moderate context; require ablation before promotion |
| `distance_to_day_high_low_pct` | `SECONDARY_CONTEXT` | 0.049 | 0.106 | 0.030 | moderate context; require ablation before promotion |
| `anchored_vwap_session_dist` | `SECONDARY_CONTEXT` | 0.050 | 0.094 | 0.052 | moderate context; require ablation before promotion |
| `vwap_dist_96` | `SECONDARY_CONTEXT` | 0.062 | 0.085 | 0.046 | moderate context; require ablation before promotion |
| `eth_btc_ret_spread_48` | `SECONDARY_CONTEXT` | 0.037 | 0.082 | 0.138 | moderate context; require ablation before promotion |
| `price_cvd_divergence` | `SECONDARY_CONTEXT` | 0.042 | 0.068 | 0.049 | moderate context; require ablation before promotion |
| `compression_release_up` | `SECONDARY_CONTEXT` | 0.032 | 0.067 | 0.001 | moderate context; require ablation before promotion |
| `vwap_dist_24` | `SECONDARY_CONTEXT` | 0.047 | 0.064 | 0.026 | moderate context; require ablation before promotion |
| `btc_ret_12` | `SECONDARY_CONTEXT` | 0.027 | 0.059 | 0.030 | moderate context; require ablation before promotion |
| `sweep_prev_low_reclaim` | `SECONDARY_CONTEXT` | 0.013 | 0.057 | 0.000 | moderate context; require ablation before promotion |

## Per-Feature Table

| Feature | Verdict | Ret IC | Vol IC | PSI | Reason |
|---|---|---:|---:|---:|---|
| `anchored_vwap_session_dist` | `SECONDARY_CONTEXT` | 0.050 | 0.094 | 0.052 | moderate context; require ablation before promotion |
| `atr_pct_rank_288` | `KEEP_RISK_CONTEXT` | 0.069 | 0.212 | 0.008 | stronger risk/volatility utility than direction |
| `bb_width_pct_rank_288` | `KEEP_RISK_CONTEXT` | 0.063 | 0.184 | 0.003 | stronger risk/volatility utility than direction |
| `btc_breakout_eth_lag_dir` | `LOW_SIGNAL_SECONDARY` | 0.023 | 0.021 | 0.003 | low standalone OOS tendency |
| `btc_eth_volume_rank_spread` | `LOW_SIGNAL_SECONDARY` | 0.007 | 0.013 | 0.059 | low standalone OOS tendency |
| `btc_impulse_x_eth_beta` | `SECONDARY_CONTEXT` | 0.038 | 0.042 | 0.022 | moderate context; require ablation before promotion |
| `btc_lead_eth_follow_gap_3` | `SECONDARY_CONTEXT` | 0.052 | 0.032 | 0.104 | moderate context; require ablation before promotion |
| `btc_ret_1` | `LOW_SIGNAL_SECONDARY` | 0.027 | 0.029 | 0.026 | low standalone OOS tendency |
| `btc_ret_12` | `SECONDARY_CONTEXT` | 0.027 | 0.059 | 0.030 | moderate context; require ablation before promotion |
| `btc_ret_3` | `SECONDARY_CONTEXT` | 0.038 | 0.042 | 0.028 | moderate context; require ablation before promotion |
| `btc_ret_6` | `SECONDARY_CONTEXT` | 0.032 | 0.041 | 0.026 | moderate context; require ablation before promotion |
| `btc_ret_z_48` | `LOW_SIGNAL_SECONDARY` | 0.026 | 0.023 | 0.002 | low standalone OOS tendency |
| `btc_volume_impulse_z` | `KEEP_RISK_CONTEXT` | 0.050 | 0.193 | 0.000 | stronger risk/volatility utility than direction |
| `compression_release_down` | `SECONDARY_CONTEXT` | 0.030 | 0.020 | 0.000 | moderate context; require ablation before promotion |
| `compression_release_up` | `SECONDARY_CONTEXT` | 0.032 | 0.067 | 0.001 | moderate context; require ablation before promotion |
| `compression_score` | `KEEP_RISK_CONTEXT` | 0.072 | 0.205 | 0.002 | stronger risk/volatility utility than direction |
| `crowded_long_unwind_risk` | `SECONDARY_CONTEXT` | 0.034 | 0.018 | 0.003 | moderate context; require ablation before promotion |
| `crowded_short_squeeze_risk` | `SECONDARY_CONTEXT` | 0.056 | 0.035 | 0.000 | moderate context; require ablation before promotion |
| `cvd_12` | `SECONDARY_CONTEXT` | 0.036 | 0.041 | 0.007 | moderate context; require ablation before promotion |
| `cvd_288` | `SECONDARY_CONTEXT` | 0.060 | 0.111 | 0.090 | moderate context; require ablation before promotion |
| `cvd_48` | `SECONDARY_CONTEXT` | 0.027 | 0.049 | 0.034 | moderate context; require ablation before promotion |
| `cvd_breakout_z` | `SECONDARY_CONTEXT` | 0.025 | 0.051 | 0.009 | moderate context; require ablation before promotion |
| `cvd_slope_12` | `LOW_SIGNAL_SECONDARY` | 0.013 | 0.014 | 0.003 | low standalone OOS tendency |
| `cvd_slope_48` | `SECONDARY_CONTEXT` | 0.022 | 0.032 | 0.002 | moderate context; require ablation before promotion |
| `cvd_slope_48_x_trend_prob` | `SECONDARY_CONTEXT` | 0.025 | 0.034 | 0.001 | moderate context; require ablation before promotion |
| `distance_to_day_high_low_pct` | `SECONDARY_CONTEXT` | 0.049 | 0.106 | 0.030 | moderate context; require ablation before promotion |
| `eth_btc_beta_residual_z` | `LOW_SIGNAL_SECONDARY` | 0.024 | 0.009 | 0.000 | low standalone OOS tendency |
| `eth_btc_ret_spread_12` | `SECONDARY_CONTEXT` | 0.058 | 0.042 | 0.094 | moderate context; require ablation before promotion |
| `eth_btc_ret_spread_48` | `SECONDARY_CONTEXT` | 0.037 | 0.082 | 0.138 | moderate context; require ablation before promotion |
| `failed_breakout_down` | `SECONDARY_CONTEXT` | 0.013 | 0.057 | 0.000 | moderate context; require ablation before promotion |
| `failed_breakout_up` | `LOW_SIGNAL_SECONDARY` | 0.020 | 0.015 | 0.000 | low standalone OOS tendency |
| `funding_flip_signal` | `LOW_SIGNAL_SECONDARY` | 0.029 | 0.012 | 0.000 | low standalone OOS tendency |
| `funding_oi_divergence` | `SECONDARY_CONTEXT` | 0.041 | 0.033 | 0.027 | moderate context; require ablation before promotion |
| `funding_oi_divergence_x_instability_prob` | `SECONDARY_CONTEXT` | 0.040 | 0.031 | 0.024 | moderate context; require ablation before promotion |
| `lower_wick_z` | `LOW_SIGNAL_SECONDARY` | 0.010 | 0.018 | 0.001 | low standalone OOS tendency |
| `oi_up_price_down` | `LOW_SIGNAL_SECONDARY` | 0.013 | 0.022 | 0.003 | low standalone OOS tendency |
| `oi_up_price_up` | `SECONDARY_CONTEXT` | 0.036 | 0.018 | 0.006 | moderate context; require ablation before promotion |
| `price_cvd_divergence` | `SECONDARY_CONTEXT` | 0.042 | 0.068 | 0.049 | moderate context; require ablation before promotion |
| `range_contraction_breakout_dir` | `SECONDARY_CONTEXT` | 0.049 | 0.036 | 0.000 | moderate context; require ablation before promotion |
| `sweep_prev_high_reclaim` | `LOW_SIGNAL_SECONDARY` | 0.020 | 0.015 | 0.000 | low standalone OOS tendency |
| `sweep_prev_low_reclaim` | `SECONDARY_CONTEXT` | 0.013 | 0.057 | 0.000 | moderate context; require ablation before promotion |
| `upper_wick_z` | `LOW_SIGNAL_SECONDARY` | 0.004 | 0.009 | 0.002 | low standalone OOS tendency |
| `vwap_dist_24` | `SECONDARY_CONTEXT` | 0.047 | 0.064 | 0.026 | moderate context; require ablation before promotion |
| `vwap_dist_288` | `SECONDARY_CONTEXT` | 0.029 | 0.119 | 0.049 | moderate context; require ablation before promotion |
| `vwap_dist_96` | `SECONDARY_CONTEXT` | 0.062 | 0.085 | 0.046 | moderate context; require ablation before promotion |
| `vwap_reclaim_flag` | `LOW_SIGNAL_SECONDARY` | 0.007 | 0.010 | 0.000 | low standalone OOS tendency |
| `vwap_reclaim_x_chop_prob` | `LOW_SIGNAL_SECONDARY` | 0.007 | 0.011 | 0.000 | low standalone OOS tendency |
| `vwap_reject_flag` | `LOW_SIGNAL_SECONDARY` | 0.011 | 0.005 | 0.000 | low standalone OOS tendency |

## Secondary Feature Retraining Implication

M7, AI, and regime-derived outputs do not automatically use these new features. If the goal is to let those second-order artifacts learn from the new direction block, their 2024-only training artifacts must be regenerated, then 2025/2026 sidecars rescored under the same causal split policy. Existing live artifacts remain valid but are blind to this new feature block.

## Source-Required Direction Features

The current offline active frame supports BTC lead-lag through `close_btc`, `volume_btc`, and `quote_volume_btc`. True orderbook imbalance, liquidation map/cluster distance, cross-exchange premium/basis, side-specific OI, and on-chain exchange-flow features are not present in the offline training frame. They must not be added as zero-filled active inputs; add them only after the historical source is persisted and the feature contract can fail fast on missing columns.
