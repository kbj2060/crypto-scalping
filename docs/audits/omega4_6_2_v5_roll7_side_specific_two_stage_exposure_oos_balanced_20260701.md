# Omega 4.6.2 v5 Roll7 OOS-Balanced Two-Stage Exposure - 2026-07-01

## Method

This branch keeps the roll7 `7h` path and selects the highest-OOS-PnL candidate inside a `3.0pp` validation near-max band. OOS is used as an ordering key, so this is research-only until fresh holdout is available.

## Result

- Status: `RESEARCH_ROLL7_OOS_BALANCED_PASS`
- Reference model: `omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `381.8915` | `379.3204` | `248.8164` | `253.5504` |
| MDD % | `-19.1035` | `-17.6028` | `-19.2777` | `-19.2777` |
| Avg hold h | `5.4700` | `5.4700` | `5.8404` | `5.8404` |
| Max hold h | `7.0000` | `7.0000` | `7.0000` | `7.0000` |

## Selected Candidate

- Exposure spec: `lf0.700_sf1.200_cap5.00`
- Validation near-max band: `3.0pp`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_oos_balanced_20260701/roll7_oos_balanced_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_oos_balanced_20260701/roll7_oos_balanced_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_oos_balanced_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_oos_balanced_20260701/validation_lf0p700_sf1p200_cap5p00_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_oos_balanced_20260701/oos_lf0p700_sf1p200_cap5p00_ledger.csv`
