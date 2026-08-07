# Omega 4.6.2 v5 Roll8 Two-Stage Exposure OOS Balanced Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701`
- Reference: `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_OOS_BALANCED_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Exposure

- Exposure spec: `lf0.950_sf1.080_cap4.60`
- Best buffered validation PnL: `718.2058%`
- Nearmax tolerance: `1.0pp`

| Split | Reference PnL | Candidate PnL | Reference MDD | Candidate MDD | Avg Hold | Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 611.3029% | 717.6129% | -19.1071% | -18.2147% | 5.8723h | 8.0000h |
| oos | 194.5778% | 221.4408% | -18.5253% | -19.9359% | 6.6409h | 8.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Replay Checks

- Validation exposure replay: `True`
- OOS exposure replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701/roll8_two_stage_exposure_oos_balanced_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701/validation_lf0p950_sf1p080_cap4p60_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701/oos_lf0p950_sf1p080_cap4p60_ledger.csv`
