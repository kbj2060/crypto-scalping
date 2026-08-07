# Omega 4.6.2 v5 Roll8 Two-Stage Exposure Validation-Only - 2026-07-01

## Method

This branch reuses the repaired buffered exposure grid, but removes OOS from selection. It selects only by validation gates and validation metrics, including a validation MDD floor of `-17.50%`. OOS is read out after selection and is not a filter, ordering key, or tie-breaker.

## Result

- Status: `RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_VALIDATION_ONLY_PASS`
- Reference model: `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `611.3029` | `675.3209` | `194.5778` | `212.6850` |
| MDD % | `-19.1071` | `-17.3157` | `-18.5253` | `-19.4083` |
| Avg hold h | `5.8723` | `5.8723` | `6.6409` | `6.6409` |
| Max hold h | `8.0000` | `8.0000` | `8.0000` | `8.0000` |

## Selected Candidate

- Exposure spec: `lf0.900_sf1.050_cap4.40`
- Long/short factor: `0.9` / `1.05`
- Cap notional: `4.4`
- Validation MDD floor: `-17.50%`
- OOS used in selection: `False`

## Artifacts

- Source buffered report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/roll8_two_stage_exposure_validation_only_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/roll8_two_stage_exposure_validation_only_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/validation_lf0p900_sf1p050_cap4p40_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/oos_lf0p900_sf1p050_cap4p40_ledger.csv`
