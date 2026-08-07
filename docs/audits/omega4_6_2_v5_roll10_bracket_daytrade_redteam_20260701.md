# Omega 4.6.2 v5 Roll10 Bracket Daytrade Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll10_bracket_daytrade_20260701`
- Reference: `omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL10_DAYTRADE_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Exposure spec: `lf0.80_sf0.95_cap4.00`
- Segment governor: `none`
- Roll max hold: `10.0`
- TP/SL: `0.03` / `0.04`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 289.4460% | 237.5114% | 9.1649h | 8.1698h | 12.0000h | 10.0000h |
| oos | 145.9377% | 128.2522% | 9.7698h | 8.5778h | 12.0000h | 10.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Contract Checks

- Validation accounting error max abs: `3.5041414214731503e-16`
- OOS accounting error max abs: `3.608224830031759e-16`
- Validation notional contract error max abs: `4.440892098500626e-16`
- OOS notional contract error max abs: `4.440892098500626e-16`
- Segment governor replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_bracket_daytrade_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_bracket_daytrade_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_bracket_daytrade_20260701/roll10_bracket_daytrade_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_bracket_daytrade_20260701/validation_lf0p80_sf0p95_cap4p00__none__tp0p030_sl0p040_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_bracket_daytrade_20260701/oos_lf0p80_sf0p95_cap4p00__none__tp0p030_sl0p040_ledger.csv`
