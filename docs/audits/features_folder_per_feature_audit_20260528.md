# Features Folder Per-Feature Audit - 2026-05-28

## Scope

This report expands the feature-folder audit to a feature-by-feature verdict table. It uses the active generated frames for statistics and `features/*.py` for source-code inventory. It does not promote broad feature expansion by itself; every recommendation is role-specific.

Known-bug regime prefixes are now forbidden for active/live use regardless of the historical verdict rows below: `clean_regime_2024_unsup_v4_*` and `clean_regime4_2024_unsup_v1_*`.

Funding-family source issues were remediated after `docs/audits/last_funding_rate_source_audit_20260528.md`; active split CSVs now use ETHUSDT-only backward-asof funding.

Artifacts:

- Per-feature verdict CSV: `tmp/causal_regen_20260516/features_folder_per_feature_audit_20260528/per_feature_verdict.csv`
- Family verdict counts: `tmp/causal_regen_20260516/features_folder_per_feature_audit_20260528/family_verdict_counts.csv`
- Source inventory: `tmp/causal_regen_20260516/features_folder_code_inventory_20260528/`
- Base score table: `tmp/causal_regen_20260516/all_feature_usage_20260528/feature_usage.csv`

## Verdict Legend

- `KEEP_ROLE_SPECIFIC`: useful, but only in its proper layer such as risk sizing, exit, execution, or regime meta.
- `KEEP_ENTRY_CONTEXT`: weak directional/context utility; can modulate entry but should not own direction alone.
- `SECONDARY_CONTEXT`: usable as a supporting context feature, not a priority input.
- `LOW_SIGNAL_SECONDARY`: weak standalone signal; keep out of compact contracts unless an ablation proves value.
- `DEDUP_DROP`: redundant with another feature; use the listed representative.
- `MONITOR_OR_VETO_ONLY`: drift or distribution risk is too high for direct model input.
- `DROP_RAW_LEVEL`: raw price-level feature; use transformed distances/returns instead.
- `BUG_RISK_REGENERATE`: suspicious generation/statistics; regenerate or replace before active/live use.

## Summary Counts

| Verdict | Count |
|---|---:|
| `SECONDARY_CONTEXT` | 69 |
| `KEEP_ROLE_SPECIFIC` | 66 |
| `LOW_SIGNAL_SECONDARY` | 21 |
| `KEEP_ENTRY_CONTEXT` | 19 |
| `DEDUP_DROP` | 15 |
| `MONITOR_OR_VETO_ONLY` | 11 |
| `DROP_RAW_LEVEL` | 9 |
| `BUG_RISK_REGENERATE` | 1 |

## Family Verdict Counts

| Family | Verdict | Count |
|---|---|---:|
| `ai` | `SECONDARY_CONTEXT` | 10 |
| `ai` | `KEEP_ROLE_SPECIFIC` | 4 |
| `ai` | `LOW_SIGNAL_SECONDARY` | 1 |
| `calendar` | `SECONDARY_CONTEXT` | 4 |
| `calendar` | `LOW_SIGNAL_SECONDARY` | 1 |
| `funding` | `KEEP_ROLE_SPECIFIC` | 4 |
| `funding` | `KEEP_ENTRY_CONTEXT` | 2 |
| `m7` | `KEEP_ROLE_SPECIFIC` | 15 |
| `m7` | `SECONDARY_CONTEXT` | 11 |
| `m7` | `LOW_SIGNAL_SECONDARY` | 8 |
| `m7` | `DROP_RAW_LEVEL` | 4 |
| `m7` | `DEDUP_DROP` | 2 |
| `microstructure` | `KEEP_ROLE_SPECIFIC` | 10 |
| `microstructure` | `KEEP_ENTRY_CONTEXT` | 3 |
| `microstructure` | `DEDUP_DROP` | 1 |
| `microstructure` | `MONITOR_OR_VETO_ONLY` | 1 |
| `open_interest` | `KEEP_ENTRY_CONTEXT` | 1 |
| `open_interest` | `MONITOR_OR_VETO_ONLY` | 1 |
| `other` | `SECONDARY_CONTEXT` | 19 |
| `other` | `LOW_SIGNAL_SECONDARY` | 7 |
| `other` | `DROP_RAW_LEVEL` | 5 |
| `other` | `MONITOR_OR_VETO_ONLY` | 4 |
| `patchtst` | `DEDUP_DROP` | 2 |
| `patchtst` | `KEEP_ROLE_SPECIFIC` | 1 |
| `patchtst` | `LOW_SIGNAL_SECONDARY` | 1 |
| `regime_legacy` | `SECONDARY_CONTEXT` | 20 |
| `regime_legacy` | `DEDUP_DROP` | 1 |
| `regime_legacy` | `LOW_SIGNAL_SECONDARY` | 1 |
| `regime_legacy` | `MONITOR_OR_VETO_ONLY` | 1 |
| `regime_pred` | `KEEP_ROLE_SPECIFIC` | 8 |
| `regime_pred` | `DEDUP_DROP` | 3 |
| `regime_pred` | `KEEP_ENTRY_CONTEXT` | 1 |
| `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | 12 |
| `regime_sticky_v2` | `DEDUP_DROP` | 3 |
| `regime_sticky_v2` | `MONITOR_OR_VETO_ONLY` | 3 |
| `regime_sticky_v2` | `KEEP_ENTRY_CONTEXT` | 2 |
| `teacher` | `SECONDARY_CONTEXT` | 5 |
| `teacher` | `DEDUP_DROP` | 1 |
| `teacher` | `KEEP_ROLE_SPECIFIC` | 1 |
| `technical` | `KEEP_ENTRY_CONTEXT` | 10 |
| `technical` | `MONITOR_OR_VETO_ONLY` | 1 |
| `ts_model` | `KEEP_ROLE_SPECIFIC` | 3 |
| `ts_model` | `DEDUP_DROP` | 2 |
| `ts_model` | `LOW_SIGNAL_SECONDARY` | 2 |
| `volatility` | `KEEP_ROLE_SPECIFIC` | 8 |
| `volatility` | `BUG_RISK_REGENERATE` | 1 |

## Per-Feature Verdict Table

| Feature | Family | Verdict | Tendency | Layer | Ret IC | Vol IC | PSI | Reason |
|---|---|---|---|---|---:|---:|---:|---|
| `ai_adverse_risk` | `ai` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.020 | 0.341 | 0.040 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `ai_anchor_overheat` | `ai` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.057 | 0.078 | 0.042 | usable but not priority; prefer compact role-specific tests |
| `ai_anchor_revert_prob` | `ai` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.013 | 0.037 | 0.014 | usable but not priority; prefer compact role-specific tests |
| `ai_anchor_trend_escape_prob` | `ai` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.013 | 0.037 | 0.014 | usable but not priority; prefer compact role-specific tests |
| `ai_dir_edge` | `ai` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.039 | 0.055 | 0.005 | usable but not priority; prefer compact role-specific tests |
| `ai_dir_entropy` | `ai` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.031 | 0.059 | 0.004 | usable but not priority; prefer compact role-specific tests |
| `ai_dir_p_down` | `ai` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.039 | 0.055 | 0.005 | usable but not priority; prefer compact role-specific tests |
| `ai_dir_p_flat` | `ai` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.030 | 0.070 | 0.005 | usable but not priority; prefer compact role-specific tests |
| `ai_dir_p_up` | `ai` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.039 | 0.055 | 0.005 | usable but not priority; prefer compact role-specific tests |
| `ai_flow_exhaustion` | `ai` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.009 | 0.063 | 0.002 | usable but not priority; prefer compact role-specific tests |
| `ai_flow_flip_prob` | `ai` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.024 | 0.065 | 0.009 | usable but not priority; prefer compact role-specific tests |
| `ai_flow_pressure` | `ai` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_sizing_or_exit` | 0.015 | 0.082 | 0.012 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `ai_flow_slope` | `ai` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.024 | 0.011 | 0.004 | low standalone OOS tendency; only as secondary if model proves value |
| `ai_reward_risk` | `ai` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_sizing_or_exit` | 0.052 | 0.102 | 0.006 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `ai_vol_regime_pct` | `ai` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_sizing_or_exit` | 0.038 | 0.120 | 0.014 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `hour_cos` | `calendar` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.010 | 0.172 | 0.000 | usable but not priority; prefer compact role-specific tests |
| `hour_sin` | `calendar` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.066 | 0.130 | 0.000 | usable but not priority; prefer compact role-specific tests |
| `is_hour_open` | `calendar` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.005 | 0.017 | 0.000 | low standalone OOS tendency; only as secondary if model proves value |
| `session_europe` | `calendar` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.013 | 0.215 | 0.000 | usable but not priority; prefer compact role-specific tests |
| `session_us` | `calendar` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.086 | 0.204 | 0.000 | usable but not priority; prefer compact role-specific tests |
| `funding_abs` | `funding` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `execution_risk_sizing` | 0.025 | 0.081 | 0.415 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `funding_pressure` | `funding` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `execution_risk_sizing` | 0.078 | 0.199 | 0.256 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `funding_price_divergence` | `funding` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `execution_risk_sizing` | 0.040 | 0.071 | 0.003 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `funding_roc_288` | `funding` | `KEEP_ENTRY_CONTEXT` | `direction_context` | `entry_context` | 0.080 | 0.027 | 0.275 | weak directional/context edge; do not use as hard owner alone |
| `last_funding_rate` | `funding` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `execution_risk_sizing` | 0.149 | 0.178 | 0.470 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `mta_funding` | `funding` | `KEEP_ENTRY_CONTEXT` | `mixed_context` | `entry_context` | 0.033 | 0.050 | 0.016 | weak directional/context edge; do not use as hard owner alone |
| `m7_action` | `m7` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_sizing_or_exit` | 0.027 | 0.084 | 0.001 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_composite_score` | `m7` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.018 | 0.017 | 0.000 | low standalone OOS tendency; only as secondary if model proves value |
| `m7_confidence` | `m7` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.010 | 0.027 | 0.003 | low standalone OOS tendency; only as secondary if model proves value |
| `m7_entry_long_offset` | `m7` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.013 | 0.448 | 0.115 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_entry_long_price` | `m7` | `DROP_RAW_LEVEL` | `risk_vol` | `monitor_or_veto_only` | 0.059 | 0.233 | 5.084 | raw price-level or raw M7 price output; use offset/return/distance instead |
| `m7_entry_short_offset` | `m7` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.015 | 0.461 | 0.127 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_entry_short_price` | `m7` | `DROP_RAW_LEVEL` | `risk_vol` | `monitor_or_veto_only` | 0.058 | 0.230 | 5.159 | raw price-level or raw M7 price output; use offset/return/distance instead |
| `m7_expected_ret` | `m7` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.018 | 0.017 | 0.000 | low standalone OOS tendency; only as secondary if model proves value |
| `m7_gate_block` | `m7` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.019 | 0.072 | 0.000 | usable but not priority; prefer compact role-specific tests |
| `m7_gmm_cluster` | `m7` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.066 | 0.154 | 0.006 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_gmm_conf` | `m7` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.069 | 0.135 | 0.007 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_gmm_vol_rank` | `m7` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.069 | 0.165 | 0.013 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_hold_pred` | `m7` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_sizing_or_exit` | 0.016 | 0.089 | 0.021 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_iso_anom` | `m7` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_sizing_or_exit` | 0.039 | 0.110 | 0.000 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_iso_pred` | `m7` | `DEDUP_DROP` | `mixed_context` | `risk_sizing_or_exit` | 0.039 | 0.110 | 0.000 | near/exact duplicate; representative=m7_iso_anom |
| `m7_iso_score` | `m7` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.027 | 0.077 | 0.003 | usable but not priority; prefer compact role-specific tests |
| `m7_mtl_dn` | `m7` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.029 | 0.025 | 0.002 | low standalone OOS tendency; only as secondary if model proves value |
| `m7_mtl_up` | `m7` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.029 | 0.030 | 0.002 | low standalone OOS tendency; only as secondary if model proves value |
| `m7_q10` | `m7` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.053 | 0.214 | 0.022 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_q50` | `m7` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.069 | 0.073 | 0.022 | usable but not priority; prefer compact role-specific tests |
| `m7_q90` | `m7` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.063 | 0.263 | 0.051 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_quality_pred` | `m7` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.019 | 0.415 | 0.082 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_quant_dn` | `m7` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.076 | 0.071 | 0.013 | usable but not priority; prefer compact role-specific tests |
| `m7_quant_up` | `m7` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.077 | 0.075 | 0.011 | usable but not priority; prefer compact role-specific tests |
| `m7_qwidth` | `m7` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.024 | 0.343 | 0.030 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_size` | `m7` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.026 | 0.024 | 0.002 | low standalone OOS tendency; only as secondary if model proves value |
| `m7_sl_offset` | `m7` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.031 | 0.014 | 0.031 | usable but not priority; prefer compact role-specific tests |
| `m7_sl_price` | `m7` | `DROP_RAW_LEVEL` | `risk_vol` | `monitor_or_veto_only` | 0.055 | 0.232 | 5.070 | raw price-level or raw M7 price output; use offset/return/distance instead |
| `m7_tail_risk` | `m7` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.023 | 0.010 | 0.000 | low standalone OOS tendency; only as secondary if model proves value |
| `m7_target_hold` | `m7` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.017 | 0.014 | 0.000 | low standalone OOS tendency; only as secondary if model proves value |
| `m7_target_quality` | `m7` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.021 | 0.400 | 0.095 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `m7_tp_offset` | `m7` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.047 | 0.021 | 0.018 | usable but not priority; prefer compact role-specific tests |
| `m7_tp_price` | `m7` | `DROP_RAW_LEVEL` | `risk_vol` | `monitor_or_veto_only` | 0.052 | 0.231 | 5.014 | raw price-level or raw M7 price output; use offset/return/distance instead |
| `m7_trend_xgb_dn` | `m7` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.031 | 0.026 | 0.004 | usable but not priority; prefer compact role-specific tests |
| `m7_trend_xgb_up` | `m7` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.032 | 0.031 | 0.004 | usable but not priority; prefer compact role-specific tests |
| `m7_vae_anom` | `m7` | `DEDUP_DROP` | `mixed_context` | `secondary_context` | 0.019 | 0.072 | 0.000 | near/exact duplicate; representative=m7_gate_block |
| `m7_vae_error` | `m7` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.068 | 0.139 | 0.006 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `amihud_illiquidity_z` | `microstructure` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `execution_risk_sizing` | 0.012 | 0.072 | 0.007 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `cvp_volume_imbalance` | `microstructure` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `execution_risk_sizing` | 0.042 | 0.117 | 0.024 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `liquidity_vacuum` | `microstructure` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `execution_risk_sizing` | 0.018 | 0.089 | 0.005 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `net_taker_ratio` | `microstructure` | `KEEP_ENTRY_CONTEXT` | `weak` | `entry_context` | 0.024 | 0.017 | 0.006 | weak directional/context edge; do not use as hard owner alone |
| `quote_volume` | `microstructure` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `execution_risk_sizing` | 0.022 | 0.362 | 0.105 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `quote_volume_btc` | `microstructure` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `execution_risk_sizing` | 0.043 | 0.357 | 0.098 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `sig_liquidity_trap` | `microstructure` | `KEEP_ENTRY_CONTEXT` | `weak` | `entry_context` | 0.018 | 0.010 | 0.001 | weak directional/context edge; do not use as hard owner alone |
| `smart_money_flow` | `microstructure` | `DEDUP_DROP` | `weak` | `entry_context` | 0.025 | 0.016 | 0.031 | near/exact duplicate; representative=oi_change_rate |
| `taker_acceleration` | `microstructure` | `KEEP_ENTRY_CONTEXT` | `weak` | `entry_context` | 0.019 | 0.022 | 0.005 | weak directional/context edge; do not use as hard owner alone |
| `taker_buy_base` | `microstructure` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `execution_risk_sizing` | 0.029 | 0.388 | 0.074 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `taker_buy_quote` | `microstructure` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `execution_risk_sizing` | 0.025 | 0.356 | 0.094 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `trade_intensity` | `microstructure` | `MONITOR_OR_VETO_ONLY` | `mixed_context` | `monitor_or_veto_only` | 0.027 | 0.053 | 1.376 | high OOS drift PSI=1.376; avoid direct active input |
| `volume` | `microstructure` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `execution_risk_sizing` | 0.027 | 0.395 | 0.085 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `volume_btc` | `microstructure` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `execution_risk_sizing` | 0.045 | 0.376 | 0.013 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `volume_profile_signal` | `microstructure` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `execution_risk_sizing` | 0.023 | 0.119 | 0.064 | use in execution_risk_sizing; stronger risk/vol utility than direction |
| `oi_change_rate` | `open_interest` | `KEEP_ENTRY_CONTEXT` | `weak` | `entry_context` | 0.025 | 0.016 | 0.031 | weak directional/context edge; do not use as hard owner alone |
| `sum_open_interest_value` | `open_interest` | `MONITOR_OR_VETO_ONLY` | `risk_vol` | `monitor_or_veto_only` | 0.052 | 0.199 | 5.377 | high OOS drift PSI=5.377; avoid direct active input |
| `big_trade_ratio` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.018 | 0.037 | 0.006 | usable but not priority; prefer compact role-specific tests |
| `breakout_strength` | `other` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.049 | 0.122 | 0.056 | usable but not priority; prefer compact role-specific tests |
| `btc_corr_60` | `other` | `MONITOR_OR_VETO_ONLY` | `risk_vol` | `monitor_or_veto_only` | 0.013 | 0.255 | 0.928 | high OOS drift PSI=0.928; avoid direct active input |
| `close` | `other` | `DROP_RAW_LEVEL` | `risk_vol` | `monitor_or_veto_only` | 0.058 | 0.232 | 5.143 | raw price-level or raw M7 price output; use offset/return/distance instead |
| `close_btc` | `other` | `DROP_RAW_LEVEL` | `risk_vol` | `monitor_or_veto_only` | 0.056 | 0.237 | 7.947 | raw price-level or raw M7 price output; use offset/return/distance instead |
| `count_long_short_ratio` | `other` | `MONITOR_OR_VETO_ONLY` | `mixed_context` | `monitor_or_veto_only` | 0.055 | 0.107 | 2.780 | high OOS drift PSI=2.780; avoid direct active input |
| `crowding_pressure` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.082 | 0.072 | 0.032 | usable but not priority; prefer compact role-specific tests |
| `cvp_cluster_position` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.037 | 0.030 | 0.005 | usable but not priority; prefer compact role-specific tests |
| `cvp_poc_dist` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.059 | 0.059 | 0.034 | usable but not priority; prefer compact role-specific tests |
| `cvp_regime` | `other` | `SECONDARY_CONTEXT` | `direction_context` | `secondary_context` | 0.076 | 0.063 | 0.023 | usable but not priority; prefer compact role-specific tests |
| `eth_btc_ratio_change` | `other` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.030 | 0.021 | 0.087 | low standalone OOS tendency; only as secondary if model proves value |
| `evt_excess_z` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.016 | 0.099 | 0.000 | usable but not priority; prefer compact role-specific tests |
| `evt_tail_flag` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.016 | 0.099 | 0.000 | usable but not priority; prefer compact role-specific tests |
| `execution_quality` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.011 | 0.058 | 0.002 | usable but not priority; prefer compact role-specific tests |
| `fvg_dist` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.029 | 0.075 | 0.014 | usable but not priority; prefer compact role-specific tests |
| `high` | `other` | `DROP_RAW_LEVEL` | `risk_vol` | `monitor_or_veto_only` | 0.058 | 0.230 | 5.193 | raw price-level or raw M7 price output; use offset/return/distance instead |
| `hma_slope` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.038 | 0.045 | 0.018 | usable but not priority; prefer compact role-specific tests |
| `jump_flag` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.023 | 0.109 | 0.000 | usable but not priority; prefer compact role-specific tests |
| `jump_z` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.040 | 0.031 | 0.009 | usable but not priority; prefer compact role-specific tests |
| `kalman_velocity` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.041 | 0.057 | 0.021 | usable but not priority; prefer compact role-specific tests |
| `kel` | `other` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.015 | 0.010 | 0.001 | low standalone OOS tendency; only as secondary if model proves value |
| `low` | `other` | `DROP_RAW_LEVEL` | `risk_vol` | `monitor_or_veto_only` | 0.059 | 0.233 | 5.066 | raw price-level or raw M7 price output; use offset/return/distance instead |
| `minute_cos` | `other` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.007 | 0.027 | 0.116 | low standalone OOS tendency; only as secondary if model proves value |
| `minute_sin` | `other` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.004 | 0.009 | 0.000 | low standalone OOS tendency; only as secondary if model proves value |
| `ofi_acceleration` | `other` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.019 | 0.016 | 0.007 | low standalone OOS tendency; only as secondary if model proves value |
| `ofti` | `other` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.012 | 0.024 | 0.178 | low standalone OOS tendency; only as secondary if model proves value |
| `open` | `other` | `DROP_RAW_LEVEL` | `risk_vol` | `monitor_or_veto_only` | 0.058 | 0.231 | 5.143 | raw price-level or raw M7 price output; use offset/return/distance instead |
| `sum_toptrader_long_short_ratio` | `other` | `MONITOR_OR_VETO_ONLY` | `risk_vol` | `monitor_or_veto_only` | 0.011 | 0.250 | 7.906 | high OOS drift PSI=7.906; avoid direct active input |
| `svps` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.031 | 0.068 | 0.040 | usable but not priority; prefer compact role-specific tests |
| `tp_sl_action_score` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.009 | 0.094 | 0.195 | usable but not priority; prefer compact role-specific tests |
| `trades` | `other` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.025 | 0.442 | 0.172 | usable but not priority; prefer compact role-specific tests |
| `turtle_signal` | `other` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.027 | 0.155 | 0.047 | usable but not priority; prefer compact role-specific tests |
| `whale_conviction` | `other` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.017 | 0.026 | 0.254 | low standalone OOS tendency; only as secondary if model proves value |
| `whale_retail_ratio` | `other` | `MONITOR_OR_VETO_ONLY` | `risk_vol` | `monitor_or_veto_only` | 0.025 | 0.203 | 1.792 | high OOS drift PSI=1.792; avoid direct active input |
| `wick_ratio` | `other` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.005 | 0.038 | 0.001 | usable but not priority; prefer compact role-specific tests |
| `conf_patchtst` | `patchtst` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_sizing_or_exit` | 0.014 | 0.111 | 0.004 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `patchtst_median` | `patchtst` | `DEDUP_DROP` | `mixed_context` | `secondary_context` | 0.039 | 0.055 | 0.005 | near/exact duplicate; representative=ai_dir_edge |
| `patchtst_regime_sim` | `patchtst` | `DEDUP_DROP` | `mixed_context` | `secondary_context` | 0.031 | 0.059 | 0.004 | near/exact duplicate; representative=ai_dir_entropy |
| `pred_patchtst` | `patchtst` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.023 | 0.025 | 0.000 | low standalone OOS tendency; only as secondary if model proves value |
| `clean_regime_2024_unsup_v4_bear_prob` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.049 | 0.056 | 0.001 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_bull_prob` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.048 | 0.082 | 0.012 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_chop_prob` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.033 | 0.014 | 0.163 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_cluster` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.030 | 0.119 | 0.001 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_cluster_confidence` | `regime_legacy` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.017 | 0.024 | 0.001 | low standalone OOS tendency; only as secondary if model proves value |
| `clean_regime_2024_unsup_v4_cluster_prob_0` | `regime_legacy` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.055 | 0.128 | 0.007 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_cluster_prob_1` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.028 | 0.089 | 0.001 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_cluster_prob_2` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.033 | 0.022 | 0.007 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_cluster_prob_3` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.037 | 0.099 | 0.001 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_cluster_prob_4` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.030 | 0.101 | 0.001 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_confidence` | `regime_legacy` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.047 | 0.203 | 0.153 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_entropy` | `regime_legacy` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.034 | 0.165 | 0.020 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_factor_crowding` | `regime_legacy` | `DEDUP_DROP` | `mixed_context` | `secondary_context` | 0.015 | 0.084 | 0.206 | near/exact duplicate; representative=clean_regime4_state24_sticky090_v2_factor_crowding |
| `clean_regime_2024_unsup_v4_factor_flow` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.025 | 0.039 | 0.005 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_factor_liquidity` | `regime_legacy` | `MONITOR_OR_VETO_ONLY` | `weak` | `monitor_or_veto_only` | 0.011 | 0.014 | 1.136 | high OOS drift PSI=1.136; avoid direct active input |
| `clean_regime_2024_unsup_v4_factor_trend` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.047 | 0.076 | 0.017 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_factor_vol` | `regime_legacy` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.056 | 0.159 | 0.008 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_normal_prob` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.039 | 0.116 | 0.391 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_risk_off_prob` | `regime_legacy` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.039 | 0.192 | 0.367 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_state_code` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.015 | 0.065 | 0.003 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_transition_risk` | `regime_legacy` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.020 | 0.194 | 0.044 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_trend_bias` | `regime_legacy` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.049 | 0.068 | 0.006 | usable but not priority; prefer compact role-specific tests |
| `clean_regime_2024_unsup_v4_whipsaw_prob` | `regime_legacy` | `SECONDARY_CONTEXT` | `risk_vol` | `secondary_context` | 0.044 | 0.203 | 0.162 | usable but not priority; prefer compact role-specific tests |
| `regime4_pred_bear_prob` | `regime_pred` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_meta_layer` | 0.031 | 0.129 | 0.027 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `regime4_pred_bull_prob` | `regime_pred` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_meta_layer` | 0.038 | 0.073 | 0.036 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `regime4_pred_chop_prob` | `regime_pred` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_meta_layer` | 0.067 | 0.283 | 0.072 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `regime4_pred_confidence` | `regime_pred` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_meta_layer` | 0.030 | 0.052 | 0.046 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `regime4_pred_directional_bias` | `regime_pred` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_meta_layer` | 0.032 | 0.095 | 0.035 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `regime4_pred_entropy` | `regime_pred` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_meta_layer` | 0.028 | 0.099 | 0.047 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `regime4_pred_instability_prob` | `regime_pred` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_meta_layer` | 0.069 | 0.142 | 0.030 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `regime4_pred_margin` | `regime_pred` | `KEEP_ENTRY_CONTEXT` | `mixed_context` | `entry_meta_context` | 0.031 | 0.025 | 0.043 | weak directional/context edge; do not use as hard owner alone |
| `regime4_pred_micro_prob` | `regime_pred` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_meta_layer` | 0.039 | 0.149 | 0.115 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `regime4_pred_range_prob` | `regime_pred` | `DEDUP_DROP` | `risk_vol` | `risk_meta_layer` | 0.067 | 0.283 | 0.072 | near/exact duplicate; representative=regime4_pred_chop_prob |
| `regime4_pred_trend_prob` | `regime_pred` | `DEDUP_DROP` | `risk_vol` | `risk_meta_layer` | 0.039 | 0.149 | 0.115 | near/exact duplicate; representative=regime4_pred_micro_prob |
| `regime4_pred_whipsaw_prob` | `regime_pred` | `DEDUP_DROP` | `risk_vol` | `risk_meta_layer` | 0.069 | 0.142 | 0.030 | near/exact duplicate; representative=regime4_pred_instability_prob |
| `clean_regime4_state24_sticky090_v2_bear_prob` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_meta_layer` | 0.025 | 0.106 | 0.014 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_bull_prob` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_meta_layer` | 0.019 | 0.057 | 0.023 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_chop_prob` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_meta_layer` | 0.038 | 0.259 | 0.022 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_confidence` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_meta_layer` | 0.025 | 0.154 | 0.007 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_directional_bias` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_meta_layer` | 0.027 | 0.093 | 0.022 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_entropy` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_meta_layer` | 0.015 | 0.180 | 0.007 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_factor_crowding` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_meta_layer` | 0.015 | 0.084 | 0.206 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_factor_flow` | `regime_sticky_v2` | `KEEP_ENTRY_CONTEXT` | `weak` | `entry_meta_context` | 0.024 | 0.014 | 0.007 | weak directional/context edge; do not use as hard owner alone |
| `clean_regime4_state24_sticky090_v2_factor_liquidity` | `regime_sticky_v2` | `MONITOR_OR_VETO_ONLY` | `weak` | `monitor_or_veto_only` | 0.009 | 0.021 | 1.198 | high OOS drift PSI=1.198; avoid direct active input |
| `clean_regime4_state24_sticky090_v2_factor_trend` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_meta_layer` | 0.055 | 0.098 | 0.023 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_factor_vol` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_meta_layer` | 0.050 | 0.078 | 0.003 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_instability_prob` | `regime_sticky_v2` | `KEEP_ENTRY_CONTEXT` | `mixed_context` | `entry_meta_context` | 0.057 | 0.038 | 0.010 | weak directional/context edge; do not use as hard owner alone |
| `clean_regime4_state24_sticky090_v2_margin` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_meta_layer` | 0.031 | 0.132 | 0.012 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_micro_prob` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_meta_layer` | 0.019 | 0.209 | 0.015 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_range_prob` | `regime_sticky_v2` | `DEDUP_DROP` | `risk_vol` | `risk_meta_layer` | 0.038 | 0.259 | 0.022 | near/exact duplicate; representative=clean_regime4_state24_sticky090_v2_chop_prob |
| `clean_regime4_state24_sticky090_v2_risk_off_prob` | `regime_sticky_v2` | `MONITOR_OR_VETO_ONLY` | `mixed_context` | `monitor_or_veto_only` | 0.032 | 0.065 | 0.621 | high OOS drift PSI=0.621; avoid direct active input |
| `clean_regime4_state24_sticky090_v2_transition_risk` | `regime_sticky_v2` | `MONITOR_OR_VETO_ONLY` | `mixed_context` | `monitor_or_veto_only` | 0.032 | 0.065 | 0.621 | high OOS drift PSI=0.621; avoid direct active input |
| `clean_regime4_state24_sticky090_v2_trend_bias` | `regime_sticky_v2` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_meta_layer` | 0.055 | 0.082 | 0.006 | use in risk_meta_layer; stronger risk/vol utility than direction |
| `clean_regime4_state24_sticky090_v2_trend_prob` | `regime_sticky_v2` | `DEDUP_DROP` | `risk_vol` | `risk_meta_layer` | 0.019 | 0.209 | 0.015 | near/exact duplicate; representative=clean_regime4_state24_sticky090_v2_micro_prob |
| `clean_regime4_state24_sticky090_v2_whipsaw_prob` | `regime_sticky_v2` | `DEDUP_DROP` | `mixed_context` | `entry_meta_context` | 0.057 | 0.038 | 0.010 | near/exact duplicate; representative=clean_regime4_state24_sticky090_v2_instability_prob |
| `teacher_long_edge` | `teacher` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.037 | 0.015 | 0.003 | usable but not priority; prefer compact role-specific tests |
| `teacher_quantile_skew` | `teacher` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.016 | 0.059 | 0.011 | usable but not priority; prefer compact role-specific tests |
| `teacher_short_edge` | `teacher` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.036 | 0.032 | 0.003 | usable but not priority; prefer compact role-specific tests |
| `teacher_side_disagreement` | `teacher` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.009 | 0.034 | 0.000 | usable but not priority; prefer compact role-specific tests |
| `teacher_side_margin` | `teacher` | `SECONDARY_CONTEXT` | `mixed_context` | `secondary_context` | 0.036 | 0.023 | 0.003 | usable but not priority; prefer compact role-specific tests |
| `teacher_tail_warning` | `teacher` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.025 | 0.350 | 0.033 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `teacher_uncertainty` | `teacher` | `DEDUP_DROP` | `risk_vol` | `risk_sizing_or_exit` | 0.024 | 0.343 | 0.030 | near/exact duplicate; representative=m7_qwidth |
| `chop_index` | `technical` | `KEEP_ENTRY_CONTEXT` | `weak` | `entry_context` | 0.023 | 0.026 | 0.008 | weak directional/context edge; do not use as hard owner alone |
| `dual_momentum` | `technical` | `KEEP_ENTRY_CONTEXT` | `risk_vol` | `entry_context` | 0.040 | 0.129 | 0.021 | weak directional/context edge; do not use as hard owner alone |
| `log_return` | `technical` | `KEEP_ENTRY_CONTEXT` | `mixed_context` | `entry_context` | 0.039 | 0.032 | 0.027 | weak directional/context edge; do not use as hard owner alone |
| `long_squeeze_risk` | `technical` | `KEEP_ENTRY_CONTEXT` | `mixed_context` | `entry_context` | 0.141 | 0.124 | 0.276 | weak directional/context edge; do not use as hard owner alone |
| `macd_hist` | `technical` | `KEEP_ENTRY_CONTEXT` | `mixed_context` | `entry_context` | 0.037 | 0.029 | 0.028 | weak directional/context edge; do not use as hard owner alone |
| `mean_reversion_z` | `technical` | `KEEP_ENTRY_CONTEXT` | `risk_vol` | `entry_context` | 0.022 | 0.139 | 0.031 | weak directional/context edge; do not use as hard owner alone |
| `mtf_trend_1h` | `technical` | `KEEP_ENTRY_CONTEXT` | `mixed_context` | `entry_context` | 0.055 | 0.067 | 0.025 | weak directional/context edge; do not use as hard owner alone |
| `mtf_trend_4h` | `technical` | `KEEP_ENTRY_CONTEXT` | `mixed_context` | `entry_context` | 0.050 | 0.092 | 0.033 | weak directional/context edge; do not use as hard owner alone |
| `regime_trending` | `technical` | `KEEP_ENTRY_CONTEXT` | `mixed_context` | `entry_context` | 0.015 | 0.037 | 0.000 | weak directional/context edge; do not use as hard owner alone |
| `rsi` | `technical` | `KEEP_ENTRY_CONTEXT` | `mixed_context` | `entry_context` | 0.052 | 0.091 | 0.005 | weak directional/context edge; do not use as hard owner alone |
| `squeeze_power` | `technical` | `MONITOR_OR_VETO_ONLY` | `mixed_context` | `monitor_or_veto_only` | 0.127 | 0.185 | 0.857 | high OOS drift PSI=0.857; avoid direct active input |
| `dlinear_smf_ema` | `ts_model` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_sizing_or_exit` | 0.010 | 0.102 | 0.026 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `dlinear_smf_slope` | `ts_model` | `DEDUP_DROP` | `weak` | `secondary_context` | 0.024 | 0.011 | 0.004 | near/exact duplicate; representative=ai_flow_slope |
| `tide_vol_raw` | `ts_model` | `DEDUP_DROP` | `risk_vol` | `risk_sizing_or_exit` | 0.020 | 0.341 | 0.040 | near/exact duplicate; representative=ai_adverse_risk |
| `tide_vol_zscore` | `ts_model` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.038 | 0.120 | 0.014 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `timesnet_cycle_cos` | `ts_model` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.020 | 0.024 | 0.003 | low standalone OOS tendency; only as secondary if model proves value |
| `timesnet_cycle_delta` | `ts_model` | `LOW_SIGNAL_SECONDARY` | `weak` | `secondary_context` | 0.022 | 0.024 | 0.007 | low standalone OOS tendency; only as secondary if model proves value |
| `timesnet_cycle_sin` | `ts_model` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_sizing_or_exit` | 0.035 | 0.116 | 0.037 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `bb_width` | `volatility` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.012 | 0.410 | 0.097 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `bb_width_z` | `volatility` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.025 | 0.152 | 0.005 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `garch_vol_z` | `volatility` | `BUG_RISK_REGENERATE` | `weak` | `monitor_or_veto_only` | nan | nan | 12.434 | extreme PSI/invalid tendency; regenerate or replace before active use |
| `garman_klass_vol` | `volatility` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.014 | 0.449 | 0.116 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `parkinson_vol` | `volatility` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.012 | 0.450 | 0.126 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `realized_skewness` | `volatility` | `KEEP_ROLE_SPECIFIC` | `mixed_context` | `risk_sizing_or_exit` | 0.008 | 0.042 | 0.033 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `realized_vol_ratio` | `volatility` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.051 | 0.210 | 0.054 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `rogers_satchell_vol` | `volatility` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.017 | 0.447 | 0.113 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |
| `volatility_z` | `volatility` | `KEEP_ROLE_SPECIFIC` | `risk_vol` | `risk_sizing_or_exit` | 0.073 | 0.205 | 0.019 | use in risk_sizing_or_exit; stronger risk/vol utility than direction |

## Actionable Contract Changes

1. Keep compact Alpha7.1/01965-style inputs as the benchmark. Do not add all features directly to parent/deep direction models.
2. Build separate contracts for entry context, risk/exit, execution context, and regime meta overlay.
3. Enforce direct-input exclusion for `DROP_RAW_LEVEL` and `BUG_RISK_REGENERATE` groups in new active/live contracts.
4. For `DEDUP_DROP`, use one representative per duplicate cluster unless a specific model ablation proves redundant views improve OOS results.
5. Normalize and re-audit high-drift features before promotion.
