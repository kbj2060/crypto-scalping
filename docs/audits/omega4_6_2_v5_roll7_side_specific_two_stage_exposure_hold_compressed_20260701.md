# Omega 4.6.2 v5 Roll7 Side-Specific Two-Stage Exposure Hold Compressed - 2026-07-01

## Method

This branch regenerates the current two-stage veto path with max roll hold compressed from `8h` to `7h`, then searches exposure overlays. Selection is validation-primary with OOS as a safety gate.

## Result

- Status: `NO_ROLL7_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASSING_CANDIDATE`
- Reference model: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `717.6129` | `541.1938` | `221.4408` | `172.9123` |
| MDD % | `-18.2147` | `-22.1933` | `-19.9359` | `-23.7943` |
| Avg hold h | `5.8723` | `5.6124` | `6.6409` | `6.0401` |
| Max hold h | `8.0000` | `7.0000` | `8.0000` | `7.0000` |

## Selected Candidate

- Exposure spec: `lf1.000_sf1.200_cap5.00`
- Max roll hold: `7.0h`
- Research gate pass: `False`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701/roll7_two_stage_exposure_hold_compressed_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701/roll7_two_stage_exposure_hold_compressed_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701/validation_lf1p000_sf1p200_cap5p00_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701/oos_lf1p000_sf1p200_cap5p00_ledger.csv`
