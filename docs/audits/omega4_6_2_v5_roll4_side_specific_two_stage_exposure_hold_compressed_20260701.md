# Omega 4.6.2 v5 Roll4 Side-Specific Two-Stage Exposure Hold Compressed - 2026-07-01

## Method

This branch reuses the roll5 hold-compressed construction, but sets max roll hold to `4h` and searches a lower exposure grid. Selection is validation-primary with OOS as a safety gate.

## Result

- Status: `NO_ROLL4_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASSING_CANDIDATE`
- Reference model: `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `308.7601` | `319.8000` | `138.4721` | `144.7868` |
| MDD % | `-18.3384` | `-30.3058` | `-19.4112` | `-29.2984` |
| Avg hold h | `4.2435` | `3.4829` | `4.4215` | `3.6285` |
| Max hold h | `5.0000` | `4.0000` | `5.0000` | `4.0000` |

## Selected Candidate

- Exposure spec: `lf0.700_sf1.200_cap5.00`
- Max roll hold: `4.0h`
- Research gate pass: `False`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701/roll4_two_stage_exposure_hold_compressed_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701/roll4_two_stage_exposure_hold_compressed_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701/validation_lf0p700_sf1p200_cap5p00_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701/oos_lf0p700_sf1p200_cap5p00_ledger.csv`
