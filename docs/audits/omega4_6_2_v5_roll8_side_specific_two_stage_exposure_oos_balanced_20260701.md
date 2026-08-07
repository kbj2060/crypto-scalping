# Omega 4.6.2 v5 Roll8 Side-Specific Two-Stage Exposure OOS Balanced - 2026-07-01

## Method

This branch reuses the buffered exposure grid. It first requires research-gated candidates and validation PnL within `1.0pp` of the best buffered validation PnL, then selects the highest OOS PnL. This is explicitly OOS-balanced and therefore requires fresh holdout before any live claim.

## Result

- Status: `RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_OOS_BALANCED_PASS`
- Reference model: `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `611.3029` | `717.6129` | `194.5778` | `221.4408` |
| MDD % | `-19.1071` | `-18.2147` | `-18.5253` | `-19.9359` |
| Avg hold h | `5.8723` | `5.8723` | `6.6409` | `6.6409` |
| Max hold h | `8.0000` | `8.0000` | `8.0000` | `8.0000` |

## Selected Candidate

- Exposure spec: `lf0.950_sf1.080_cap4.60`
- Long/short factor: `0.95` / `1.08`
- Cap notional: `4.6`
- Best buffered validation PnL: `718.2058%`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701/roll8_two_stage_exposure_oos_balanced_ranking.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701/validation_lf0p950_sf1p080_cap4p60_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701/oos_lf0p950_sf1p080_cap4p60_ledger.csv`
