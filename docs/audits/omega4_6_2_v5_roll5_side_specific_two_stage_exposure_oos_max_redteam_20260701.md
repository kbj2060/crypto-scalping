# Omega 4.6.2 v5 Roll5 OOS-Max Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701`
- Reference: `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL5_OOS_MAX_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Exposure spec: `lf0.100_sf1.000_cap4.40`
- Validation near-max band: `10.0pp`
- OOS ordering used: `True`
- OOS MDD buffer to -20%: `3.9353pp`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 302.8578% | 296.9050% | 4.2333h | 4.2333h | 5.0000h | 5.0000h |
| oos | 169.8794% | 187.6595% | 4.4281h | 4.4281h | 5.0000h | 5.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Replay Checks

- Validation replay: `True`
- OOS replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701/roll5_oos_max_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701/roll5_oos_max_top20.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701/validation_lf0p100_sf1p000_cap4p40_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701/oos_lf0p100_sf1p000_cap4p40_ledger.csv`
