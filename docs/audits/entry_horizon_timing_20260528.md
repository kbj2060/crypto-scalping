# Entry Horizon Timing Audit - 2026-05-28

Horizons: `12`, `24`, `36`, `48`, `64`, `96` bars.

Known-bug regime prefixes are now forbidden for active/live use regardless of the historical ranking rows below: `clean_regime_2024_unsup_v4_*` and `clean_regime4_2024_unsup_v1_*`.

## Artifacts

- Feature scores: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/entry_horizon_timing_20260528/entry_horizon_feature_scores.csv`
- Family summary: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/entry_horizon_timing_20260528/entry_horizon_family_summary.csv`
- Top by horizon: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/entry_horizon_timing_20260528/entry_horizon_top_by_horizon.csv`

## Top Clean Features By Horizon

| Horizon | Feature | Family | IC OOS | Direction AUC OOS | Stable Sign | PSI |
|---:|---|---|---:|---:|---:|---:|
| 12 | `last_funding_rate` | `funding` | 0.093 | 0.532 | True | 0.470 |
| 12 | `long_squeeze_risk` | `technical` | 0.085 | 0.529 | True | 0.276 |
| 12 | `m7_quant_dn` | `m7` | -0.064 | 0.537 | True | 0.013 |
| 12 | `m7_quant_up` | `m7` | 0.063 | 0.537 | True | 0.011 |
| 12 | `vwap_dist_96` | `other` | -0.062 | 0.545 | True | 0.046 |
| 12 | `m7_q50` | `m7` | 0.060 | 0.534 | True | 0.022 |
| 12 | `funding_roc_288` | `funding` | 0.058 | 0.516 | True | 0.275 |
| 12 | `eth_btc_ret_spread_12` | `other` | -0.058 | 0.533 | True | 0.094 |
| 12 | `m7_q90` | `m7` | 0.050 | 0.527 | True | 0.051 |
| 12 | `anchored_vwap_session_dist` | `calendar` | -0.050 | 0.538 | True | 0.052 |
| 24 | `last_funding_rate` | `funding` | 0.130 | 0.557 | True | 0.470 |
| 24 | `long_squeeze_risk` | `technical` | 0.112 | 0.548 | True | 0.276 |
| 24 | `funding_roc_288` | `funding` | 0.080 | 0.535 | True | 0.275 |
| 24 | `m7_quant_up` | `m7` | 0.077 | 0.545 | True | 0.011 |
| 24 | `m7_quant_dn` | `m7` | -0.076 | 0.544 | True | 0.013 |
| 24 | `m7_q50` | `m7` | 0.069 | 0.540 | True | 0.022 |
| 24 | `funding_pressure` | `funding` | 0.062 | 0.529 | True | 0.256 |
| 24 | `m7_q90` | `m7` | 0.060 | 0.531 | True | 0.051 |
| 24 | `crowding_pressure` | `other` | 0.060 | 0.519 | True | 0.032 |
| 24 | `clean_regime4_state24_sticky090_v2_instability_prob` | `regime_sticky_v2` | 0.057 | 0.534 | True | 0.010 |
| 36 | `last_funding_rate` | `funding` | 0.140 | 0.557 | True | 0.470 |
| 36 | `long_squeeze_risk` | `technical` | 0.127 | 0.552 | True | 0.276 |
| 36 | `funding_roc_288` | `funding` | 0.083 | 0.535 | True | 0.275 |
| 36 | `crowding_pressure` | `other` | 0.077 | 0.520 | False | 0.032 |
| 36 | `clean_regime4_state24_sticky090_v2_instability_prob` | `regime_sticky_v2` | 0.073 | 0.545 | True | 0.010 |
| 36 | `regime4_pred_instability_prob` | `regime_pred` | 0.072 | 0.530 | True | 0.030 |
| 36 | `m7_quant_up` | `m7` | 0.069 | 0.531 | True | 0.011 |
| 36 | `m7_quant_dn` | `m7` | -0.066 | 0.529 | True | 0.013 |
| 36 | `funding_pressure` | `funding` | 0.062 | 0.530 | True | 0.256 |
| 36 | `m7_q90` | `m7` | 0.062 | 0.525 | True | 0.051 |
| 48 | `last_funding_rate` | `funding` | 0.149 | 0.562 | True | 0.470 |
| 48 | `long_squeeze_risk` | `technical` | 0.141 | 0.560 | True | 0.276 |
| 48 | `crowding_pressure` | `other` | 0.082 | 0.523 | False | 0.032 |
| 48 | `funding_roc_288` | `funding` | 0.080 | 0.532 | False | 0.275 |
| 48 | `funding_pressure` | `funding` | 0.078 | 0.540 | True | 0.256 |
| 48 | `cvp_regime` | `other` | -0.076 | 0.551 | False | 0.023 |
| 48 | `m7_quant_up` | `m7` | 0.072 | 0.529 | True | 0.011 |
| 48 | `compression_score` | `other` | -0.072 | 0.532 | True | 0.002 |
| 48 | `m7_quant_dn` | `m7` | -0.070 | 0.528 | True | 0.013 |
| 48 | `atr_pct_rank_288` | `volatility` | 0.069 | 0.532 | True | 0.008 |
| 64 | `last_funding_rate` | `funding` | 0.154 | 0.563 | True | 0.470 |
| 64 | `long_squeeze_risk` | `technical` | 0.141 | 0.554 | True | 0.276 |
| 64 | `funding_pressure` | `funding` | 0.100 | 0.552 | True | 0.256 |
| 64 | `crowding_pressure` | `other` | 0.087 | 0.522 | False | 0.032 |
| 64 | `cvp_regime` | `other` | -0.080 | 0.547 | True | 0.023 |
| 64 | `regime4_pred_chop_prob` | `regime_pred` | -0.079 | 0.531 | True | 0.072 |
| 64 | `atr_pct_rank_288` | `volatility` | 0.075 | 0.531 | True | 0.008 |
| 64 | `m7_vae_error` | `m7` | 0.075 | 0.534 | True | 0.006 |
| 64 | `compression_score` | `other` | -0.073 | 0.530 | True | 0.002 |
| 64 | `funding_roc_288` | `funding` | 0.073 | 0.523 | False | 0.275 |
| 96 | `last_funding_rate` | `funding` | 0.163 | 0.578 | True | 0.470 |
| 96 | `long_squeeze_risk` | `technical` | 0.144 | 0.562 | True | 0.276 |
| 96 | `funding_pressure` | `funding` | 0.135 | 0.575 | True | 0.256 |
| 96 | `crowding_pressure` | `other` | 0.092 | 0.523 | False | 0.032 |
| 96 | `regime4_pred_chop_prob` | `regime_pred` | -0.083 | 0.522 | True | 0.072 |
| 96 | `cvp_regime` | `other` | -0.082 | 0.542 | True | 0.023 |
| 96 | `cvd_288` | `other` | 0.076 | 0.536 | True | 0.090 |
| 96 | `btc_volume_impulse_z` | `microstructure` | 0.074 | 0.524 | True | 0.000 |
| 96 | `mta_funding` | `funding` | -0.068 | 0.531 | False | 0.016 |
| 96 | `regime4_pred_instability_prob` | `regime_pred` | 0.064 | 0.521 | True | 0.030 |

## Family Winners

| Family | Best Feature | Best Horizon | Best Abs IC |
|---|---|---:|---:|
| `funding` | `last_funding_rate` | 96 | 0.163 |
| `technical` | `long_squeeze_risk` | 96 | 0.144 |
| `other` | `crowding_pressure` | 96 | 0.092 |
| `regime_pred` | `regime4_pred_chop_prob` | 96 | 0.083 |
| `m7` | `m7_quant_up` | 24 | 0.077 |
| `volatility` | `atr_pct_rank_288` | 64 | 0.075 |
| `microstructure` | `btc_volume_impulse_z` | 96 | 0.074 |
| `regime_sticky_v2` | `clean_regime4_state24_sticky090_v2_instability_prob` | 36 | 0.073 |
| `regime_legacy` | `clean_regime_2024_unsup_v4_cluster_prob_0` | 64 | 0.062 |
| `ai` | `ai_anchor_overheat` | 64 | 0.058 |
| `calendar` | `anchored_vwap_session_dist` | 12 | 0.050 |
| `ts_model` | `dlinear_smf_ema` | 96 | 0.045 |
| `teacher` | `teacher_tail_warning` | 36 | 0.028 |
| `patchtst` | `conf_patchtst` | 96 | 0.026 |
| `open_interest` | `oi_up_price_up` | 12 | 0.022 |
