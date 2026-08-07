# Omega 4.6.2 v5 Roll12 OOS-Max Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll12_side_specific_oos_max_20260701`
- Reference: `omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL12_OOS_MAX_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Bracket spec: `fine_fast_val`
- Exposure spec: `lf0.75_sf1.02_cap4.20`
- Segment governor: `loss1_90_win12`
- Validation near-max band: `10.0pp`
- OOS ordering used: `True`
- OOS MDD buffer to -20%: `2.9858pp`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 338.5234% | 330.0475% | 9.5049h | 9.0355h | 12.0000h | 12.0000h |
| oos | 165.3214% | 178.5726% | 10.0224h | 9.8945h | 12.0000h | 12.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Replay Checks

- Segment governor replay: `True`
- Validation accounting error max abs: `3.1051550219984847e-16`
- OOS accounting error max abs: `3.7470027081099033e-16`
- Validation notional contract error max abs: `8.881784197001252e-16`
- OOS notional contract error max abs: `8.881784197001252e-16`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_oos_max_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_oos_max_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_oos_max_20260701/roll12_oos_max_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_oos_max_20260701/roll12_oos_max_top20.csv`
- Source ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701/roll12_side_specific_fine_valmax_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_oos_max_20260701/validation_fine_fast_val__lf0p75_sf1p02_cap4p20__loss1_90_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_oos_max_20260701/oos_fine_fast_val__lf0p75_sf1p02_cap4p20__loss1_90_win12_ledger.csv`
