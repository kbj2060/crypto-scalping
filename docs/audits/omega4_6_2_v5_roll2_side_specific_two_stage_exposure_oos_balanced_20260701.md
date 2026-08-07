# Omega 4.6.2 v5 Roll2 OOS-Balanced Two-Stage Exposure - 2026-07-01

## Method

This branch keeps the roll2 `2h` path and selects the highest-OOS-PnL candidate inside a `3.0pp` validation near-max band. It is research-only until fresh holdout is available because OOS is used as an ordering key.

## Result

- Status: `RESEARCH_ROLL2_OOS_BALANCED_PASS`
- Reference model: `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `189.1812` | `186.9196` | `136.0997` | `151.6907` |
| MDD % | `-18.8233` | `-18.8233` | `-19.6703` | `-19.6703` |
| Avg hold h | `1.9056` | `1.9056` | `1.9153` | `1.9153` |
| Max hold h | `2.0000` | `2.0000` | `2.0000` | `2.0000` |

## Selected Candidate

- Exposure spec: `lf0.300_sf1.100_cap4.80`
- Validation near-max band: `3.0pp`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701/roll2_oos_balanced_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701/roll2_oos_balanced_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701/validation_lf0p300_sf1p100_cap4p80_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701/oos_lf0p300_sf1p100_cap4p80_ledger.csv`
