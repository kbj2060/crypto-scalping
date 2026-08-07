# Omega 4.6.2 v5 Roll2 Side-Specific Two-Stage Exposure Hold Compressed - 2026-07-01

## Method

This branch reuses the roll5 hold-compressed construction, but sets max roll hold to `2h` and searches a lower exposure grid. Selection is validation-primary with OOS as a safety gate.

## Result

- Status: `NO_ROLL2_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASSING_CANDIDATE`
- Reference model: `omega4_6_2_v5_roll3_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `202.1387` | `99.4419` | `117.8077` | `64.7032` |
| MDD % | `-32.2986` | `-48.8138` | `-27.2904` | `-35.0439` |
| Avg hold h | `2.7819` | `1.9069` | `2.8168` | `1.9279` |
| Max hold h | `3.0000` | `2.0000` | `3.0000` | `2.0000` |

## Selected Candidate

- Exposure spec: `lf0.800_sf1.500_cap5.00`
- Max roll hold: `2.0h`
- Research gate pass: `False`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701/roll2_two_stage_exposure_hold_compressed_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701/roll2_two_stage_exposure_hold_compressed_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701/validation_lf0p800_sf1p500_cap5p00_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701/oos_lf0p800_sf1p500_cap5p00_ledger.csv`
