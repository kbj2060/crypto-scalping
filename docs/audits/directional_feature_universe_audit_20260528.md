# Directional Feature Universe Audit - 2026-05-28

## Scope

This report re-audits all direction-like candidates available in the active/offline frame plus the 48 engineered `STATE_DIRECTION_ALPHA` features.

Known-bug regime prefixes are now forbidden for active/live use regardless of the historical family scores below: `clean_regime_2024_unsup_v4_*` and `clean_regime4_2024_unsup_v1_*`.

Artifacts:
- Feature scores: `tmp/causal_regen_20260516/directional_feature_universe_audit_20260528/directional_feature_universe_scores.csv`
- Family summary: `tmp/causal_regen_20260516/directional_feature_universe_audit_20260528/directional_feature_family_summary.csv`
- Family probe AUC: `tmp/causal_regen_20260516/directional_feature_universe_audit_20260528/directional_feature_family_probe_auc.csv`
- Summary: `tmp/causal_regen_20260516/directional_feature_universe_audit_20260528/summary.json`

## Verdict Counts

| Verdict | Count |
|---|---:|
| `SECONDARY_CONTEXT` | 65 |
| `SECONDARY_ENTRY_CONTEXT` | 42 |
| `KEEP_RISK_CONTEXT` | 33 |
| `LOW_SIGNAL_SECONDARY` | 25 |
| `KEEP_ENTRY_CONTEXT` | 24 |
| `DEDUP_DROP_USE_REPRESENTATIVE` | 15 |
| `MONITOR_OR_VETO_ONLY` | 11 |
| `DROP_RAW_LEVEL` | 5 |

## Best Direction-Tendency Candidates

| Feature | Family | Verdict | Ret IC | Vol IC | PSI | Prior | Reason |
|---|---|---|---:|---:|---:|---|---|
| `last_funding_rate` | `funding` | `KEEP_ENTRY_CONTEXT` | 0.149 | 0.178 | 0.470 | `KEEP_ROLE_SPECIFIC` | best current direction-context candidate; ablate in entry/meta layer |
| `long_squeeze_risk` | `technical` | `KEEP_ENTRY_CONTEXT` | 0.141 | 0.124 | 0.276 | `KEEP_ENTRY_CONTEXT` | best current direction-context candidate; ablate in entry/meta layer |
| `squeeze_power` | `technical` | `MONITOR_OR_VETO_ONLY` | 0.127 | 0.185 | 0.857 | `MONITOR_OR_VETO_ONLY` | prior monitor/veto feature with OOS drift PSI=0.857 |
| `crowding_pressure` | `other` | `KEEP_ENTRY_CONTEXT` | 0.082 | 0.072 | 0.032 | `SECONDARY_CONTEXT` | best current direction-context candidate; ablate in entry/meta layer |
| `funding_roc_288` | `funding` | `SECONDARY_ENTRY_CONTEXT` | 0.080 | 0.027 | 0.275 | `KEEP_ENTRY_CONTEXT` | moderate direction tendency; use only inside compact ablations |
| `funding_pressure` | `funding` | `KEEP_ENTRY_CONTEXT` | 0.078 | 0.199 | 0.256 | `KEEP_ROLE_SPECIFIC` | best current direction-context candidate; ablate in entry/meta layer |
| `m7_quant_up` | `m7` | `KEEP_ENTRY_CONTEXT` | 0.077 | 0.075 | 0.011 | `SECONDARY_CONTEXT` | best current direction-context candidate; ablate in entry/meta layer |
| `m7_quant_dn` | `m7` | `KEEP_ENTRY_CONTEXT` | 0.076 | 0.071 | 0.013 | `SECONDARY_CONTEXT` | best current direction-context candidate; ablate in entry/meta layer |
| `cvp_regime` | `other` | `SECONDARY_ENTRY_CONTEXT` | 0.076 | 0.063 | 0.023 | `SECONDARY_CONTEXT` | moderate direction tendency; use only inside compact ablations |
| `compression_score` | `other` | `KEEP_ENTRY_CONTEXT` | 0.072 | 0.205 | 0.002 | `` | best current direction-context candidate; ablate in entry/meta layer |
| `atr_pct_rank_288` | `volatility` | `KEEP_ENTRY_CONTEXT` | 0.069 | 0.212 | 0.008 | `` | best current direction-context candidate; ablate in entry/meta layer |
| `regime4_pred_whipsaw_prob` | `regime_pred` | `DEDUP_DROP_USE_REPRESENTATIVE` | 0.069 | 0.142 | 0.030 | `DEDUP_DROP` | prior audit marked as duplicate; use representative feature instead |
| `regime4_pred_instability_prob` | `regime_pred` | `KEEP_ENTRY_CONTEXT` | 0.069 | 0.142 | 0.030 | `KEEP_ROLE_SPECIFIC` | best current direction-context candidate; ablate in entry/meta layer |
| `m7_q50` | `m7` | `KEEP_ENTRY_CONTEXT` | 0.069 | 0.073 | 0.022 | `SECONDARY_CONTEXT` | best current direction-context candidate; ablate in entry/meta layer |
| `m7_gmm_conf` | `m7` | `KEEP_ENTRY_CONTEXT` | 0.069 | 0.135 | 0.007 | `KEEP_ROLE_SPECIFIC` | best current direction-context candidate; ablate in entry/meta layer |
| `m7_gmm_vol_rank` | `m7` | `KEEP_ENTRY_CONTEXT` | 0.069 | 0.165 | 0.013 | `KEEP_ROLE_SPECIFIC` | best current direction-context candidate; ablate in entry/meta layer |
| `m7_vae_error` | `m7` | `KEEP_ENTRY_CONTEXT` | 0.068 | 0.139 | 0.006 | `KEEP_ROLE_SPECIFIC` | best current direction-context candidate; ablate in entry/meta layer |
| `regime4_pred_chop_prob` | `regime_pred` | `KEEP_ENTRY_CONTEXT` | 0.067 | 0.283 | 0.072 | `KEEP_ROLE_SPECIFIC` | best current direction-context candidate; ablate in entry/meta layer |
| `regime4_pred_range_prob` | `regime_pred` | `DEDUP_DROP_USE_REPRESENTATIVE` | 0.067 | 0.283 | 0.072 | `DEDUP_DROP` | prior audit marked as duplicate; use representative feature instead |
| `m7_gmm_cluster` | `m7` | `KEEP_ENTRY_CONTEXT` | 0.066 | 0.154 | 0.006 | `KEEP_ROLE_SPECIFIC` | best current direction-context candidate; ablate in entry/meta layer |
| `bb_width_pct_rank_288` | `volatility` | `KEEP_RISK_CONTEXT` | 0.063 | 0.184 | 0.003 | `` | stronger risk/volatility utility than pure direction |
| `m7_q90` | `m7` | `KEEP_ENTRY_CONTEXT` | 0.063 | 0.263 | 0.051 | `KEEP_ROLE_SPECIFIC` | best current direction-context candidate; ablate in entry/meta layer |
| `vwap_dist_96` | `other` | `KEEP_ENTRY_CONTEXT` | 0.062 | 0.085 | 0.046 | `` | best current direction-context candidate; ablate in entry/meta layer |
| `cvd_288` | `other` | `KEEP_ENTRY_CONTEXT` | 0.060 | 0.111 | 0.090 | `` | best current direction-context candidate; ablate in entry/meta layer |
| `m7_entry_long_price` | `m7` | `DROP_RAW_LEVEL` | 0.059 | 0.233 | 5.084 | `DROP_RAW_LEVEL` | prior audit excludes raw level/price-like features from direct active input |
| `m7_entry_short_price` | `m7` | `DROP_RAW_LEVEL` | 0.058 | 0.230 | 5.159 | `DROP_RAW_LEVEL` | prior audit excludes raw level/price-like features from direct active input |
| `eth_btc_ret_spread_12` | `other` | `KEEP_ENTRY_CONTEXT` | 0.058 | 0.042 | 0.094 | `` | best current direction-context candidate; ablate in entry/meta layer |
| `ai_anchor_overheat` | `ai` | `KEEP_ENTRY_CONTEXT` | 0.057 | 0.078 | 0.042 | `SECONDARY_CONTEXT` | best current direction-context candidate; ablate in entry/meta layer |
| `clean_regime4_state24_sticky090_v2_instability_prob` | `regime_sticky_v2` | `KEEP_ENTRY_CONTEXT` | 0.057 | 0.038 | 0.010 | `KEEP_ENTRY_CONTEXT` | best current direction-context candidate; ablate in entry/meta layer |
| `clean_regime4_state24_sticky090_v2_whipsaw_prob` | `regime_sticky_v2` | `DEDUP_DROP_USE_REPRESENTATIVE` | 0.057 | 0.038 | 0.010 | `DEDUP_DROP` | prior audit marked as duplicate; use representative feature instead |

## Best Risk/Volatility Context

| Feature | Family | Verdict | Ret IC | Vol IC | PSI | Reason |
|---|---|---|---:|---:|---:|---|
| `m7_entry_short_offset` | `m7` | `KEEP_RISK_CONTEXT` | 0.015 | 0.461 | 0.127 | stronger risk/volatility utility than pure direction |
| `m7_entry_long_offset` | `m7` | `KEEP_RISK_CONTEXT` | 0.013 | 0.448 | 0.115 | stronger risk/volatility utility than pure direction |
| `m7_quality_pred` | `m7` | `KEEP_RISK_CONTEXT` | 0.019 | 0.415 | 0.082 | stronger risk/volatility utility than pure direction |
| `m7_target_quality` | `m7` | `KEEP_RISK_CONTEXT` | 0.021 | 0.400 | 0.095 | stronger risk/volatility utility than pure direction |
| `volume` | `microstructure` | `KEEP_RISK_CONTEXT` | 0.027 | 0.395 | 0.085 | stronger risk/volatility utility than pure direction |
| `taker_buy_base` | `microstructure` | `KEEP_RISK_CONTEXT` | 0.029 | 0.388 | 0.074 | stronger risk/volatility utility than pure direction |
| `volume_btc` | `microstructure` | `KEEP_RISK_CONTEXT` | 0.045 | 0.376 | 0.013 | stronger risk/volatility utility than pure direction |
| `quote_volume` | `microstructure` | `KEEP_RISK_CONTEXT` | 0.022 | 0.362 | 0.105 | stronger risk/volatility utility than pure direction |
| `quote_volume_btc` | `microstructure` | `KEEP_RISK_CONTEXT` | 0.043 | 0.357 | 0.098 | stronger risk/volatility utility than pure direction |
| `taker_buy_quote` | `microstructure` | `KEEP_RISK_CONTEXT` | 0.025 | 0.356 | 0.094 | stronger risk/volatility utility than pure direction |
| `teacher_tail_warning` | `teacher` | `KEEP_RISK_CONTEXT` | 0.025 | 0.350 | 0.033 | stronger risk/volatility utility than pure direction |
| `m7_qwidth` | `m7` | `KEEP_RISK_CONTEXT` | 0.024 | 0.343 | 0.030 | stronger risk/volatility utility than pure direction |
| `teacher_uncertainty` | `teacher` | `DEDUP_DROP_USE_REPRESENTATIVE` | 0.024 | 0.343 | 0.030 | prior audit marked as duplicate; use representative feature instead |
| `tide_vol_raw` | `ts_model` | `DEDUP_DROP_USE_REPRESENTATIVE` | 0.020 | 0.341 | 0.040 | prior audit marked as duplicate; use representative feature instead |
| `ai_adverse_risk` | `ai` | `KEEP_RISK_CONTEXT` | 0.020 | 0.341 | 0.040 | stronger risk/volatility utility than pure direction |
| `regime4_pred_chop_prob` | `regime_pred` | `KEEP_ENTRY_CONTEXT` | 0.067 | 0.283 | 0.072 | best current direction-context candidate; ablate in entry/meta layer |
| `regime4_pred_range_prob` | `regime_pred` | `DEDUP_DROP_USE_REPRESENTATIVE` | 0.067 | 0.283 | 0.072 | prior audit marked as duplicate; use representative feature instead |
| `m7_q90` | `m7` | `KEEP_ENTRY_CONTEXT` | 0.063 | 0.263 | 0.051 | best current direction-context candidate; ablate in entry/meta layer |
| `clean_regime4_state24_sticky090_v2_chop_prob` | `regime_sticky_v2` | `KEEP_RISK_CONTEXT` | 0.038 | 0.259 | 0.022 | stronger risk/volatility utility than pure direction |
| `clean_regime4_state24_sticky090_v2_range_prob` | `regime_sticky_v2` | `DEDUP_DROP_USE_REPRESENTATIVE` | 0.038 | 0.259 | 0.022 | prior audit marked as duplicate; use representative feature instead |

## Family Summary

| Family | N | Best Ret Feature | Best Ret IC | Best Vol Feature | Best Vol IC | Median PSI |
|---|---:|---|---:|---|---:|---:|
| `funding` | 9 | `last_funding_rate` | 0.149 | `funding_pressure` | 0.199 | 0.027 |
| `technical` | 14 | `long_squeeze_risk` | 0.141 | `squeeze_power` | 0.185 | 0.023 |
| `other` | 45 | `crowding_pressure` | 0.082 | `btc_corr_60` | 0.255 | 0.026 |
| `m7` | 40 | `m7_quant_up` | 0.077 | `m7_entry_short_offset` | 0.461 | 0.007 |
| `volatility` | 2 | `atr_pct_rank_288` | 0.069 | `atr_pct_rank_288` | 0.212 | 0.005 |
| `regime_pred` | 12 | `regime4_pred_whipsaw_prob` | 0.069 | `regime4_pred_range_prob` | 0.283 | 0.044 |
| `ai` | 15 | `ai_anchor_overheat` | 0.057 | `ai_adverse_risk` | 0.341 | 0.006 |
| `regime_sticky_v2` | 20 | `clean_regime4_state24_sticky090_v2_whipsaw_prob` | 0.057 | `clean_regime4_state24_sticky090_v2_range_prob` | 0.259 | 0.015 |
| `regime_legacy` | 23 | `clean_regime_2024_unsup_v4_factor_vol` | 0.056 | `clean_regime_2024_unsup_v4_whipsaw_prob` | 0.203 | 0.008 |
| `open_interest` | 4 | `sum_open_interest_value` | 0.052 | `sum_open_interest_value` | 0.199 | 0.018 |
| `calendar` | 1 | `anchored_vwap_session_dist` | 0.050 | `anchored_vwap_session_dist` | 0.094 | 0.052 |
| `microstructure` | 17 | `btc_volume_impulse_z` | 0.050 | `volume` | 0.395 | 0.031 |
| `patchtst` | 4 | `patchtst_median` | 0.039 | `conf_patchtst` | 0.111 | 0.004 |
| `ts_model` | 7 | `tide_vol_zscore` | 0.038 | `tide_vol_raw` | 0.341 | 0.014 |
| `teacher` | 7 | `teacher_long_edge` | 0.037 | `teacher_tail_warning` | 0.350 | 0.003 |

## Notes

- Available offline whale/OI features are proxy-level: `whale_conviction`, `whale_retail_ratio`, aggregate OI, top-trader ratios, taker ratios, and engineered funding/OI interactions.
- True liquidation flow, orderbook imbalance, side-specific OI, and on-chain exchange-flow features are still source-required and are not scored here because they are absent from the active/offline training frame.
- Direction IC is a cheap screen, not a trade-level proof. Candidates marked `KEEP_ENTRY_CONTEXT` or `SECONDARY_ENTRY_CONTEXT` still need layer-specific ablation.
