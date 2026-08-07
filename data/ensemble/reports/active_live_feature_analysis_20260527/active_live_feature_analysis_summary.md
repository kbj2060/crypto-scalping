# Active/Live Feature Utility Analysis

- features: 73
- rows 2025: 105,064
- rows 2026: 16,897
- high corr edges abs>=0.90: 59
- high corr clusters abs>=0.95: 13

## Top Future Predictive Features
| feature | family | future_score_0_100 | best_horizon | best_ic_2026 | best_auc_directional_2026 | psi_2026_vs_2025 | utility_bucket |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m7_quant_up | m7 | 95.6958 | 24 | 0.0762262 | 0.542725 | 0.00696468 | core_future |
| m7_quant_dn | m7 | 95.6958 | 24 | -0.0762262 | 0.542724 | 0.00696468 | core_future |
| m7_expected_ret | m7 | 94.7492 | 24 | 0.079339 | 0.543603 | 0.0216589 | core_future |
| m7_q50 | m7 | 94.7492 | 24 | 0.079339 | 0.543603 | 0.0216589 | core_future |
| ai_reward_risk | ai | 94.3967 | 3 | 0.0650348 | 0.539661 | 0.00459839 | core_future |
| ai_dir_edge | ai | 92.3823 | 3 | -0.0514904 | 0.535487 | 0.00390936 | core_future |
| ai_dir_p_down | ai | 92.3823 | 3 | 0.0514904 | 0.535487 | 0.00390936 | core_future |
| ai_dir_p_up | ai | 92.3823 | 3 | -0.0514904 | 0.535487 | 0.00390936 | core_future |
| patchtst_median | nf_direct | 92.3823 | 3 | -0.0514904 | 0.535487 | 0.00390936 | core_future |
| ai_anchor_overheat | ai | 90.9183 | 3 | -0.0556908 | 0.537547 | 0.0314056 | core_future |
| m7_tp_offset | m7 | 80.2081 | 24 | 0.0465417 | 0.522447 | 0.0361893 | core_future |
| m7_q90 | m7 | 80.208 | 24 | 0.0465415 | 0.522447 | 0.0361893 | core_future |
| timesnet_cycle_delta | nf_direct | 79.3431 | 3 | -0.0419113 | 0.524335 | 0.00686379 | core_future |
| teacher_long_edge | teacher | 75.5907 | 6 | -0.0384929 | 0.522279 | 0.00254658 | core_future |
| teacher_side_margin | teacher | 75.5301 | 6 | -0.038612 | 0.521983 | 0.00280606 | core_future |
| teacher_short_edge | teacher | 75.2398 | 6 | 0.0384883 | 0.521577 | 0.00250342 | core_future |
| timesnet_cycle_sin | nf_direct | 73.1243 | 3 | -0.0355838 | 0.526482 | 0.0292965 | core_future |
| m7_trend_xgb_dn | m7 | 71.6471 | 6 | 0.0355059 | 0.519876 | 0.00343958 | core_future |
| m7_trend_xgb_up | m7 | 71.0105 | 6 | -0.0347437 | 0.519923 | 0.00310309 | core_future |
| ai_dir_entropy | ai | 66.6868 | 6 | -0.0307317 | 0.518561 | 0.0037845 | core_future |

## Top Current Context Features
| feature | family | current_score_0_100 | best_current_target | best_current_corr_2026 | psi_2026_vs_2025 | utility_bucket |
| --- | --- | --- | --- | --- | --- | --- |
| m7_action | m7 | 100 | current_ret_1 | 0.720465 | 0 | useful_future |
| pred_patchtst | nf_direct | 100 | net_taker_ratio | 0.670193 | 0 | current_context |
| m7_mtl_dn | m7 | 99.868 | current_ret_1 | -0.906753 | 0.00132147 | useful_future |
| m7_mtl_up | m7 | 99.8468 | current_ret_1 | 0.906395 | 0.00153397 | useful_future |
| teacher_short_edge | teacher | 99.7503 | current_ret_1 | -0.889588 | 0.00250342 | core_future |
| teacher_long_edge | teacher | 99.746 | current_ret_1 | 0.890751 | 0.00254658 | core_future |
| teacher_side_margin | teacher | 99.7202 | current_ret_1 | 0.893651 | 0.00280606 | core_future |
| m7_trend_xgb_up | m7 | 99.6907 | current_ret_1 | 0.905374 | 0.00310309 | core_future |
| m7_trend_xgb_dn | m7 | 99.6572 | current_ret_1 | -0.906298 | 0.00343958 | core_future |
| m7_composite_score | m7 | 99.6461 | current_ret_1 | 0.838869 | 0.0035515 | weak_or_redundant |
| patchtst_median | nf_direct | 99.6106 | clean_regime_2024_unsup_v4_trend_bias | 0.675266 | 0.00390936 | core_future |
| ai_dir_p_up | ai | 99.6106 | clean_regime_2024_unsup_v4_trend_bias | 0.675266 | 0.00390936 | core_future |
| ai_dir_p_down | ai | 99.6106 | clean_regime_2024_unsup_v4_trend_bias | -0.675266 | 0.00390936 | core_future |
| ai_dir_edge | ai | 99.6106 | clean_regime_2024_unsup_v4_trend_bias | 0.675266 | 0.00390936 | core_future |
| ai_reward_risk | ai | 99.5423 | clean_regime_2024_unsup_v4_trend_bias | -0.62992 | 0.00459839 | core_future |
| tide_vol_zscore | nf_direct | 98.8749 | volatility_z | 0.584908 | 0.0113789 | useful_future |
| ai_vol_regime_pct | ai | 98.8663 | volatility_z | 0.583936 | 0.011467 | useful_future |
| m7_confidence | m7 | 98.1663 | current_abs_ret_1 | 0.541313 | 0.00259084 | current_context |
| timesnet_cycle_sin | nf_direct | 97.1537 | clean_regime_2024_unsup_v4_trend_bias | 0.560887 | 0.0292965 | core_future |
| m7_qwidth | m7 | 97.032 | clean_regime_2024_unsup_v4_transition_risk | 0.657161 | 0.0305878 | current_context |