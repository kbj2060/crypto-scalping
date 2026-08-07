# Omega 4.6.2 v5 Roll6 Side-Specific Two-Stage Exposure Hold Compressed - 2026-07-01

## Method

This branch reuses the roll7 hold-compressed construction, but sets max roll hold to `6h` and searches a lower long-exposure grid. Selection is validation-primary with OOS as a safety gate.

## Result

- Status: `NO_ROLL6_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASSING_CANDIDATE`
- Reference model: `omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `541.1938` | `316.1422` | `172.9123` | `140.6726` |
| MDD % | `-22.1933` | `-24.6581` | `-23.7943` | `-30.3023` |
| Avg hold h | `5.6124` | `4.9759` | `6.0401` | `5.1296` |
| Max hold h | `7.0000` | `6.0000` | `7.0000` | `6.0000` |

## Selected Candidate

- Exposure spec: `lf0.400_sf1.200_cap5.00`
- Max roll hold: `6.0h`
- Research gate pass: `False`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701/roll6_two_stage_exposure_hold_compressed_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701/roll6_two_stage_exposure_hold_compressed_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701/validation_lf0p400_sf1p200_cap5p00_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701/oos_lf0p400_sf1p200_cap5p00_ledger.csv`
