# Omega 4.6.2 v5 Roll24 Segment Governor Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll24_segment_governor_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Reference daytrade model: `omega4_6_2_v5_roll24_daytrade_overlay_20260701`
- Verdict: `DAYTRADE_RESEARCH_PASS_FULL_LIVE_BLOCKED`
- Daytrade research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Exposure spec: `long105_short107_cap405`
- Segment governor: `streak90_70_win12`

| Split | Reference PnL | Candidate PnL | Reference MDD | Candidate MDD | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 249.1403% | 276.9693% | -19.9363% | -19.4048% | 20.2917h | 20.2917h | 24.0000h | 24.0000h |
| oos | 142.1316% | 143.7794% | -18.6719% | -19.9164% | 20.1303h | 20.1303h | 24.0000h | 24.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Contract Checks

- Validation accounting error max abs: `3.5041414214731503e-16`
- OOS accounting error max abs: `3.3306690738754696e-16`
- Validation notional contract error max abs: `4.440892098500626e-16`
- OOS notional contract error max abs: `8.881784197001252e-16`
- Validation max notional: `4.011336402909415`
- OOS max notional: `4.05`
- Segment governor replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_segment_governor_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_segment_governor_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_segment_governor_20260701/v5_roll24_segment_governor_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_segment_governor_20260701/validation_long105_short107_cap405__streak90_70_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_segment_governor_20260701/oos_long105_short107_cap405__streak90_70_win12_ledger.csv`
