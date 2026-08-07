# Omega 4.6.2 v5 Roll8 Side-Specific Feature Veto - 2026-07-01

## Method

This branch starts from `omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701` and applies one path-causal short-entry veto based on a single entry-time feature threshold. Selection is validation-primary; OOS is a safety gate and is not an ordering key.

Lookahead-like fields are excluded by name before threshold search.

## Result

- Status: `RESEARCH_ROLL8_SIDE_SPECIFIC_FEATURE_VETO_PASS`
- Reference model: `omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `232.9667` | `360.7428` | `175.6263` | `184.8193` |
| MDD % | `-19.9902` | `-19.1071` | `-16.9439` | `-16.9439` |
| Trades | `211` | `194` | `116` | `101` |
| Avg hold h | `6.0964` | `5.9459` | `6.7119` | `6.6947` |
| Max hold h | `8.0000` | `8.0000` | `8.0000` | `8.0000` |

## Selected Candidate

- Feature: `bb_width`
- Rule: `bb_width <= 0.0039395935`
- Quantile: `0.1`
- Validation/OOS vetoed shorts: `17` / `15`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/roll8_side_specific_feature_veto_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/roll8_side_specific_feature_veto_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/validation_bb_width_le_0p0039395935_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/oos_bb_width_le_0p0039395935_ledger.csv`
