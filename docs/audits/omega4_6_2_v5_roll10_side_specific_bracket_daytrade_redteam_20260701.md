# Omega 4.6.2 v5 Roll10 Side-Specific Bracket Daytrade Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701`
- Reference: `omega4_6_2_v5_roll10_bracket_daytrade_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL10_SIDE_SPECIFIC_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Bracket spec: `fast_short`
- Exposure spec: `lf0.90_sf1.02_cap4.20`
- Segment governor: `loss1_90_win10`
- Roll max hold: `10.0`
- Long TP/SL: `0.025` / `0.035`
- Short TP/SL: `0.025` / `0.04`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 237.5114% | 261.9047% | 8.1698h | 7.7241h | 10.0000h | 10.0000h |
| oos | 128.2522% | 131.0583% | 8.5778h | 8.0430h | 10.0000h | 10.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Contract Checks

- Validation accounting error max abs: `3.7470027081099033e-16`
- OOS accounting error max abs: `3.3306690738754696e-16`
- Validation notional contract error max abs: `8.881784197001252e-16`
- OOS notional contract error max abs: `8.881784197001252e-16`
- Segment governor replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701/roll10_side_specific_bracket_daytrade_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701/validation_fast_short__lf0p90_sf1p02_cap4p20__loss1_90_win10_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701/oos_fast_short__lf0p90_sf1p02_cap4p20__loss1_90_win10_ledger.csv`
