# Omega 4.6.2 v5 Roll6 Hold-Compressed Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701`
- Reference: `omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `REDTEAM_FAIL`
- Research pass: `False`
- Full live pass: `False`

## Selected Candidate

- Exposure spec: `lf0.400_sf1.200_cap5.00`
- Max roll hold: `6.0h`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 541.1938% | 316.1422% | 5.6124h | 4.9759h | 7.0000h | 6.0000h |
| oos | 172.9123% | 140.6726% | 6.0401h | 5.1296h | 7.0000h | 6.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- `selected_gate_pass`: {'observed': False}
- `hold_compression_contract`: {'reference': {'exposure_spec': 'lf1.000_sf1.200_cap5.00', 'exposure_long_factor': 1.0, 'exposure_short_factor': 1.2, 'exposure_cap_notional': 5.0, 'roll7_max_hours': 7.0, 'roll7_long_tp_move': 0.02, 'roll7_long_sl_move': 0.03, 'roll7_short_tp_move': 0.025, 'roll7_short_sl_move': 0.0385, 'validation_pnl': 541.1937526159629, 'validation_mdd': -22.193263325111733, 'validation_trades': 201, 'validation_wr': 0.5422885572139303, 'validation_avg_hold_hours': 5.612354892205638, 'validation_max_hold_hours': 7.0, 'validation_hold_over_24h_count': 0, 'validation_max_leverage': 5.0, 'validation_avg_notional': 2.9844840397977914, 'validation_max_notional': 4.498695031300278, 'validation_max_margin_fraction': 0.8997390062600555, 'validation_skipped': 30, 'validation_overlap_count': 0, 'validation_accounting_error_max_abs': 0.0, 'validation_notional_contract_error_max_abs': 4.440892098500626e-16, 'validation_long_trades': 48, 'validation_short_trades': 153, 'validation_reason_counts': '{"roll7_bracket_sl": 4, "roll7_bracket_tp": 49, "roll7_final": 15, "roll7_time_exit": 133}', 'oos_pnl': 172.9123364407632, 'oos_mdd': -23.794341256711714, 'oos_trades': 108, 'oos_wr': 0.5185185185185185, 'oos_avg_hold_hours': 6.040123456790123, 'oos_max_hold_hours': 7.0, 'oos_hold_over_24h_count': 0, 'oos_max_leverage': 5.0, 'oos_avg_notional': 3.1723597703022497, 'oos_max_notional': 4.9272, 'oos_max_margin_fraction': 0.98544, 'oos_skipped': 19, 'oos_overlap_count': 0, 'oos_accounting_error_max_abs': 0.0, 'oos_notional_contract_error_max_abs': 2.220446049250313e-16, 'oos_long_trades': 12, 'oos_short_trades': 96, 'oos_reason_counts': '{"roll7_bracket_sl": 5, "roll7_bracket_tp": 19, "roll7_final": 8, "roll7_time_exit": 76}', 'validation_roll7_hold_compressed_gate_pass': False, 'oos_safety_gate_pass': False, 'research_roll7_hold_compressed_gate_pass': False}, 'candidate': {'validation': {'pnl': 316.1422149576749, 'mdd': -24.658059202904358, 'trades': 225, 'wr': 0.5466666666666666, 'avg_hold_hours': 4.975925925925925, 'max_hold_hours': 6.0, 'hold_over_24h_count': 0, 'max_leverage': 5.0, 'avg_notional': 2.6810878357420895, 'max_notional': 4.498695031300278, 'max_margin_fraction': 0.8997390062600555, 'skipped': 35, 'overlap_count': 0, 'accounting_error_max_abs': 3.7470027081099033e-16, 'notional_contract_error_max_abs': 8.881784197001252e-16, 'long_trades': 54, 'short_trades': 171, 'reason_counts': {'roll7_bracket_sl': 10, 'roll7_bracket_tp': 49, 'roll7_final': 17, 'roll7_time_exit': 149}}, 'oos': {'pnl': 140.6726151829274, 'mdd': -30.302343851352752, 'trades': 126, 'wr': 0.5396825396825397, 'avg_hold_hours': 5.129629629629628, 'max_hold_hours': 6.0, 'hold_over_24h_count': 0, 'max_leverage': 5.0, 'avg_notional': 3.009544188984038, 'max_notional': 4.9272, 'max_margin_fraction': 0.98544, 'skipped': 23, 'overlap_count': 0, 'accounting_error_max_abs': 4.579669976578771e-16, 'notional_contract_error_max_abs': 8.881784197001252e-16, 'long_trades': 14, 'short_trades': 112, 'reason_counts': {'roll7_bracket_sl': 4, 'roll7_bracket_tp': 19, 'roll7_final': 11, 'roll7_time_exit': 92}}}}

## Replay Checks

- Validation replay: `True`
- OOS replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701/roll6_two_stage_exposure_hold_compressed_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701/roll6_two_stage_exposure_hold_compressed_top20.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701/validation_lf0p400_sf1p200_cap5p00_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701/oos_lf0p400_sf1p200_cap5p00_ledger.csv`
