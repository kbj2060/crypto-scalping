# Omega 4.6.2 v5 Roll2 OOS-Max Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701`
- Reference: `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `REDTEAM_FAIL`
- Research pass: `False`
- Full live pass: `False`

## Selected Candidate

- Exposure spec: `lf0.800_sf1.500_cap5.00`
- Validation near-max band: `10.0pp`
- OOS ordering used: `True`
- OOS MDD buffer to -20%: `-15.0439pp`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 99.4419% | 99.4419% | 1.9069h | 1.9069h | 2.0000h | 2.0000h |
| oos | 64.7032% | 64.7032% | 1.9279h | 1.9279h | 2.0000h | 2.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- `selected_gate_pass`: {'oos_max': False, 'base_gate': False}
- `oos_max_contract`: {'reference': {'exposure_spec': 'lf0.800_sf1.500_cap5.00', 'exposure_long_factor': 0.8, 'exposure_short_factor': 1.5, 'exposure_cap_notional': 5.0, 'roll2_max_hours': 2.0, 'roll2_long_tp_move': 0.02, 'roll2_long_sl_move': 0.03, 'roll2_short_tp_move': 0.025, 'roll2_short_sl_move': 0.0385, 'validation_pnl': 99.44188506846217, 'validation_mdd': -48.813766852704475, 'validation_trades': 590, 'validation_wr': 0.48135593220338985, 'validation_avg_hold_hours': 1.9069209039548025, 'validation_max_hold_hours': 2.0, 'validation_hold_over_24h_count': 0, 'validation_max_leverage': 5.0, 'validation_avg_notional': 3.4762398246135398, 'validation_max_notional': 5.0, 'validation_max_margin_fraction': 1.0, 'validation_skipped': 69, 'validation_overlap_count': 0, 'validation_accounting_error_max_abs': 0.0, 'validation_notional_contract_error_max_abs': 4.440892098500626e-16, 'validation_long_trades': 125, 'validation_short_trades': 465, 'validation_reason_counts': '{"roll7_bracket_sl": 6, "roll7_bracket_tp": 38, "roll7_final": 19, "roll7_time_exit": 527}', 'oos_pnl': 64.70316845529938, 'oos_mdd': -35.043870048512396, 'oos_trades': 320, 'oos_wr': 0.509375, 'oos_avg_hold_hours': 1.9278645833333337, 'oos_max_hold_hours': 2.0, 'oos_hold_over_24h_count': 0, 'oos_max_leverage': 5.0, 'oos_avg_notional': 3.479585954134653, 'oos_max_notional': 5.0, 'oos_max_margin_fraction': 1.0, 'oos_skipped': 71, 'oos_overlap_count': 0, 'oos_accounting_error_max_abs': 0.0, 'oos_notional_contract_error_max_abs': 2.220446049250313e-16, 'oos_long_trades': 38, 'oos_short_trades': 282, 'oos_reason_counts': '{"roll7_bracket_sl": 3, "roll7_bracket_tp": 16, "roll7_final": 10, "roll7_time_exit": 291}', 'validation_roll2_hold_compressed_gate_pass': False, 'oos_safety_gate_pass': False, 'research_roll2_hold_compressed_gate_pass': False}, 'candidate': {'validation': {'pnl': 99.4418850684636, 'mdd': -48.81376685270442, 'trades': 590, 'wr': 0.48135593220338985, 'avg_hold_hours': 1.9069209039548025, 'max_hold_hours': 2.0, 'hold_over_24h_count': 0, 'max_leverage': 5.0, 'avg_notional': 3.4762398246135398, 'max_notional': 5.0, 'max_margin_fraction': 1.0, 'skipped': 69, 'overlap_count': 0, 'accounting_error_max_abs': 4.0245584642661925e-16, 'notional_contract_error_max_abs': 4.440892098500626e-16, 'long_trades': 125, 'short_trades': 465, 'reason_counts': {'roll7_bracket_sl': 6, 'roll7_bracket_tp': 38, 'roll7_final': 19, 'roll7_time_exit': 527}}, 'oos': {'pnl': 64.70316845529922, 'mdd': -35.043870048512346, 'trades': 320, 'wr': 0.509375, 'avg_hold_hours': 1.9278645833333337, 'max_hold_hours': 2.0, 'hold_over_24h_count': 0, 'max_leverage': 5.0, 'avg_notional': 3.479585954134653, 'max_notional': 5.0, 'max_margin_fraction': 1.0, 'skipped': 71, 'overlap_count': 0, 'accounting_error_max_abs': 4.0245584642661925e-16, 'notional_contract_error_max_abs': 4.440892098500626e-16, 'long_trades': 38, 'short_trades': 282, 'reason_counts': {'roll7_bracket_sl': 3, 'roll7_bracket_tp': 16, 'roll7_final': 10, 'roll7_time_exit': 291}}}}

## Replay Checks

- Validation replay: `True`
- OOS replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701/roll2_oos_max_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701/roll2_oos_max_top20.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701/validation_lf0p800_sf1p500_cap5p00_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701/oos_lf0p800_sf1p500_cap5p00_ledger.csv`
