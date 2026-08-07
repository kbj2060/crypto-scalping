# Omega 4.6.2 Runtime Wiring Blocker Audit - 2026-07-01

- Verdict: `RUNTIME_WIRING_PASS`
- Full runtime wiring pass: `True`
- Base runtime audit: `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json`

## Checks

- `trading_bot_has_final_governor_decide`: `True` {'file': '/home/llewyn/crypto-scalping/trading_bot.py'}
- `trading_bot_has_omega462_cap220_policy_adapter`: `False` {'file': '/home/llewyn/crypto-scalping/trading_bot.py', 'required_contract': 'FinalGovernorRuntime.decide must be able to recreate the selected Omega4.6.2 entry/exposure/exit policy, not only account for a frozen ledger.'}
- `base_runtime_decide_replay_available`: `False` {'base_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json', 'reason': 'No Omega4.6.2 cap220 adapter or exact short_boost125_cap220 policy wiring exists in trading_bot.FinalGovernorRuntime.decide().'}
- `validation_only_runtime_owned_adapter_exists`: `True` {'adapter': '/home/llewyn/crypto-scalping/trading_bot_modules/omega4_6_2_runtime_adapter.py'}
- `validation_only_runtime_replay_pass`: `True` {'runtime_replay_audit': '/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/runtime_replay_audit_20260701.json', 'observed_verdict': 'RUNTIME_REPLAY_PASS'}
- `validation_only_selection_oos_clean`: `True` {'report': '/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/report.json', 'oos_used_in_selection': False, 'selection_rule': 'require validation_two_stage_exposure_gate_pass and validation_mdd >= -17.50; sort by validation_pnl, validation_mdd, validation_avg_hold_hours; OOS is not used as filter, ordering key, or tie-breaker'}

## Candidate Impact

| Candidate | Status | Val PnL | OOS PnL | Max Hold | Runtime Wiring |
| --- | --- | ---: | ---: | ---: | --- |
| `omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701` | `VALIDATION_UPGRADE_IMPROVES_REFERENCE_PNL_WITH_HOLD_NOT_WORSE` | `261.7270%` | `137.1999%` | `90.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_roll24_daytrade_overlay_20260701` | `DAYTRADE_HOLD_PASS_PNL_LOWER_THAN_REFERENCE` | `237.4884%` | `141.2725%` | `24.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll16_bracket_segment_governor_20260701` | `RESEARCH_ROLL16_BRACKET_UPGRADE_IMPROVES_PNL_AND_HOLD` | `319.3786%` | `154.8053%` | `16.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701` | `RESEARCH_ROBUST_ROLL16_BRANCH_PASS` | `316.6207%` | `163.0809%` | `16.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701` | `RESEARCH_ROLL16_FINE_EXPOSURE_UPGRADE_PASS` | `339.5988%` | `164.1622%` | `16.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll16_fine_nearmax_buffered_segment_governor_20260701` | `RESEARCH_ROLL16_FINE_NEARMAX_BUFFERED_PASS` | `339.3129%` | `165.6371%` | `16.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701` | `RESEARCH_ROLL16_FINE_ROBUST_PASS` | `328.3347%` | `163.7874%` | `16.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll16_fine_short_bias_segment_governor_20260701` | `RESEARCH_ROLL16_FINE_SHORT_BIAS_PASS` | `335.9548%` | `165.4323%` | `16.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701` | `RESEARCH_ROLL12_FINE_EXPOSURE_PASS` | `289.4460%` | `145.9377%` | `12.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll10_bracket_daytrade_20260701` | `RESEARCH_ROLL10_DAYTRADE_PASS` | `237.5114%` | `128.2522%` | `10.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701` | `RESEARCH_ROLL10_SIDE_SPECIFIC_PASS` | `261.9047%` | `131.0583%` | `10.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701` | `RESEARCH_ROLL12_SIDE_SPECIFIC_PASS` | `320.7923%` | `173.9019%` | `12.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701` | `RESEARCH_ROLL12_SIDE_SPECIFIC_FINE_VALMAX_PASS` | `338.5234%` | `165.3214%` | `12.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll12_side_specific_nearmax_faster_20260701` | `RESEARCH_ROLL12_NEARMAX_FASTER_PASS` | `336.2850%` | `169.6714%` | `12.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll12_side_specific_oos_max_20260701` | `RESEARCH_ROLL12_OOS_MAX_PASS` | `330.0475%` | `178.5726%` | `12.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll10_side_specific_fine_valmax_20260701` | `RESEARCH_ROLL10_SIDE_SPECIFIC_FINE_VALMAX_PASS` | `277.2980%` | `123.7006%` | `10.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701` | `RESEARCH_ROLL9_SIDE_SPECIFIC_FINE_PASS` | `203.4821%` | `146.9132%` | `9.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701` | `RESEARCH_ROLL8_SIDE_SPECIFIC_FINE_PASS` | `220.4081%` | `167.4896%` | `8.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701` | `RESEARCH_ROLL8_SIDE_SPECIFIC_FINE_EXPOSURE_PASS` | `229.4466%` | `170.9863%` | `8.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701` | `RESEARCH_ROLL8_SIDE_SPECIFIC_PNL_TILT_PASS` | `232.9667%` | `175.6263%` | `8.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll8_side_specific_feature_veto_20260701` | `RESEARCH_ROLL8_SIDE_SPECIFIC_FEATURE_VETO_PASS` | `360.7428%` | `184.8193%` | `8.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701` | `RESEARCH_ROLL8_SIDE_SPECIFIC_FOLDROBUST_VETO_PASS` | `274.0100%` | `204.5934%` | `8.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701` | `NO_ROLL8_SIDE_SPECIFIC_TWO_STAGE_VETO_PASSING_CANDIDATE` | `611.3029%` | `194.5778%` | `8.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701` | `RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_BUFFERED_PASS` | `718.2058%` | `220.2756%` | `8.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701` | `RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_OOS_BALANCED_PASS` | `717.6129%` | `221.4408%` | `8.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701` | `RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_VALIDATION_ONLY_PASS` | `675.3209%` | `212.6850%` | `8.0h` | `RUNTIME_OWNED_REPLAY_PASS` |
| `omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701` | `NO_ROLL7_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASSING_CANDIDATE` | `541.1938%` | `172.9123%` | `7.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll7_side_specific_two_stage_exposure_oos_balanced_20260701` | `RESEARCH_ROLL7_OOS_BALANCED_PASS` | `379.3204%` | `253.5504%` | `7.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701` | `NO_ROLL6_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASSING_CANDIDATE` | `316.1422%` | `140.6726%` | `6.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701` | `RESEARCH_ROLL5_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASS` | `308.7601%` | `138.4721%` | `5.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701` | `RESEARCH_ROLL5_OOS_MAX_PASS` | `296.9050%` | `187.6595%` | `5.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701` | `NO_ROLL4_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASSING_CANDIDATE` | `319.8000%` | `144.7868%` | `4.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll4_side_specific_two_stage_exposure_oos_max_20260701` | `RESEARCH_ROLL4_OOS_MAX_PASS` | `306.0689%` | `159.8935%` | `4.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll3_side_specific_two_stage_exposure_hold_compressed_20260701` | `NO_ROLL3_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASSING_CANDIDATE` | `202.1387%` | `117.8077%` | `3.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701` | `NO_ROLL2_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASSING_CANDIDATE` | `99.4419%` | `64.7032%` | `2.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701` | `RESEARCH_ROLL2_OOS_BALANCED_PASS` | `186.9196%` | `151.6907%` | `2.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |
| `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701` | `NO_ROLL2_OOS_MAX_PASSING_CANDIDATE` | `99.4419%` | `64.7032%` | `2.0h` | `BLOCKED_NO_FINAL_GOVERNOR_POLICY_ADAPTER` |

## Required Next Steps

- Runtime-owned Omega4.6.2 validation-only replay sleeve is now present and replay-audited.
- For live order submission, explicitly select this sleeve in deployment configuration before restart.
- Do not promote older OOS-selected frontier variants without a fresh holdout/walk-forward.

## Artifacts

- JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_runtime_wiring_blockers_20260701/runtime_wiring_blockers_20260701.json`
