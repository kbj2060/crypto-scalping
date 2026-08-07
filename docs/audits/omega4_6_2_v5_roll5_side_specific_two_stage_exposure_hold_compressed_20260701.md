# Omega 4.6.2 v5 Roll5 Side-Specific Two-Stage Exposure Hold Compressed - 2026-07-01

## Method

This branch reuses the roll6 hold-compressed construction, but sets max roll hold to `5h` and searches a lower exposure grid. Selection is validation-primary with OOS as a safety gate.

## Result

- Status: `RESEARCH_ROLL5_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASS`
- Reference model: `omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `316.1422` | `308.7601` | `140.6726` | `138.4721` |
| MDD % | `-24.6581` | `-18.3384` | `-30.3023` | `-19.4112` |
| Avg hold h | `4.9759` | `4.2435` | `5.1296` | `4.4215` |
| Max hold h | `6.0000` | `5.0000` | `6.0000` | `5.0000` |

## Selected Candidate

- Exposure spec: `lf0.700_sf1.000_cap4.40`
- Max roll hold: `5.0h`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701/roll5_two_stage_exposure_hold_compressed_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701/roll5_two_stage_exposure_hold_compressed_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701/validation_lf0p700_sf1p000_cap4p40_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701/oos_lf0p700_sf1p000_cap4p40_ledger.csv`
