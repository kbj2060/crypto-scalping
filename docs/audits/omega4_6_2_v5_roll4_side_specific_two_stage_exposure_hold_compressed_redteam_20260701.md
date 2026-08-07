# Omega 4.6.2 v5 Roll4 Hold-Compressed Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701`
- Reference: `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL4_HOLD_COMPRESSED_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Exposure spec: `lf0.700_sf1.100_cap4.00`
- Max roll hold: `4.0h`
- OOS MDD buffer to -20%: `0.0152pp`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 302.8578% | 317.3833% | 4.2333h | 3.4727h | 5.0000h | 4.0000h |
| oos | 169.8794% | 140.4955% | 4.4281h | 3.6140h | 5.0000h | 4.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Replay Checks

- Validation replay: `True`
- OOS replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701/roll4_two_stage_exposure_hold_compressed_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701/roll4_two_stage_exposure_hold_compressed_top20.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701/validation_lf0p700_sf1p100_cap4p00_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701/oos_lf0p700_sf1p100_cap4p00_ledger.csv`
