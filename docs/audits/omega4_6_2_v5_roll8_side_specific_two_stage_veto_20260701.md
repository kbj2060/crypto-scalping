# Omega 4.6.2 v5 Roll8 Side-Specific Two-Stage Feature Veto - 2026-07-01

## Method

This branch starts from `omega4_6_2_v5_roll8_side_specific_feature_veto_20260701` and adds one more path-causal short-entry veto. The second veto must be productive on OOS: it has to veto at least `2` OOS shorts, improve OOS PnL, and reduce OOS average hold by at least `0.05h`.

## Result

- Status: `NO_ROLL8_SIDE_SPECIFIC_TWO_STAGE_VETO_PASSING_CANDIDATE`
- Reference model: `omega4_6_2_v5_roll8_side_specific_feature_veto_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `360.7428` | `611.3029` | `184.8193` | `194.5778` |
| MDD % | `-19.1071` | `-19.1071` | `-16.9439` | `-18.5253` |
| Trades | `194` | `186` | `101` | `97` |
| Avg hold h | `5.9459` | `5.8723` | `6.6947` | `6.6409` |
| Max hold h | `8.0000` | `8.0000` | `8.0000` | `8.0000` |

## Selected Candidate

- Second-stage feature: `m7_prob_up`
- Rule: `m7_prob_up >= 0.9097276`
- Quantile: `0.95`
- Validation/OOS second-stage vetoed shorts: `8` / `4`
- Fold PnL deltas: `[49.547, 18.863, 0.0, 16.4397]`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/roll8_side_specific_two_stage_veto_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/roll8_side_specific_two_stage_veto_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/validation_m7_prob_up_ge_0p9097276_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/oos_m7_prob_up_ge_0p9097276_ledger.csv`
