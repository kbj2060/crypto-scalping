# Omega 4.6.2 v5 Roll2 OOS-Balanced Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701`
- Reference: `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL2_OOS_BALANCED_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Exposure spec: `lf0.300_sf1.100_cap4.80`
- Validation near-max band: `3.0pp`
- OOS ordering used: `True`
- OOS MDD buffer to -20%: `0.3297pp`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 189.1812% | 186.9196% | 1.9056h | 1.9056h | 2.0000h | 2.0000h |
| oos | 136.0997% | 151.6907% | 1.9153h | 1.9153h | 2.0000h | 2.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Replay Checks

- Validation replay: `True`
- OOS replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701/roll2_oos_balanced_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701/roll2_oos_balanced_top20.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701/validation_lf0p300_sf1p100_cap4p80_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701/oos_lf0p300_sf1p100_cap4p80_ledger.csv`
