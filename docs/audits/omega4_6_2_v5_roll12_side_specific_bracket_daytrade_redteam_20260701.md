# Omega 4.6.2 v5 Roll12 Side-Specific Bracket Daytrade Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701`
- Reference: `omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL12_SIDE_SPECIFIC_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Bracket spec: `oos_top`
- Exposure spec: `lf0.90_sf1.02_cap4.20`
- Segment governor: `none`
- Roll max hold: `12.0`
- Long TP/SL: `0.025` / `0.04`
- Short TP/SL: `0.04` / `0.04`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 289.4460% | 320.7923% | 9.1649h | 9.4349h | 12.0000h | 12.0000h |
| oos | 145.9377% | 173.9019% | 9.7698h | 10.0224h | 12.0000h | 12.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Contract Checks

- Validation accounting error max abs: `3.1051550219984847e-16`
- OOS accounting error max abs: `3.7470027081099033e-16`
- Validation notional contract error max abs: `8.881784197001252e-16`
- OOS notional contract error max abs: `8.881784197001252e-16`
- Segment governor replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701/roll12_side_specific_bracket_daytrade_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701/validation_oos_top__lf0p90_sf1p02_cap4p20__none_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701/oos_oos_top__lf0p90_sf1p02_cap4p20__none_ledger.csv`
