# Omega 4.6.2 v5 Roll8 Side-Specific Fold-Robust Feature Veto - 2026-07-01

## Method

This branch starts from `omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701` and searches the same path-causal short-entry veto family as the feature-veto branch, but requires temporal validation-fold robustness before selection.

Fold gate:

- `4` chronological validation folds.
- No validation fold may have negative PnL delta versus the reference ledger.
- No validation fold may have positive average-hold delta versus the reference ledger.
- Each candidate still needs the same validation/OOS safety gates as the first feature-veto branch.

## Result

- Status: `RESEARCH_ROLL8_SIDE_SPECIFIC_FOLDROBUST_VETO_PASS`
- Reference model: `omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `232.9667` | `274.0100` | `175.6263` | `204.5934` |
| MDD % | `-19.9902` | `-19.1071` | `-16.9439` | `-16.9439` |
| Trades | `211` | `185` | `116` | `111` |
| Avg hold h | `6.0964` | `5.9689` | `6.7119` | `6.7042` |
| Max hold h | `8.0000` | `8.0000` | `8.0000` | `8.0000` |

## Selected Candidate

- Feature: `big_trade_ratio`
- Rule: `big_trade_ratio >= 0.63282428`
- Quantile: `0.85`
- Validation/OOS vetoed shorts: `26` / `5`
- Min fold PnL delta: `0.0000pp`
- Fold PnL deltas: `[0.0, 0.0, 3.2871, 10.2803]`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701/roll8_side_specific_foldrobust_veto_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701/roll8_side_specific_foldrobust_veto_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701/validation_big_trade_ratio_ge_0p63282428_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701/oos_big_trade_ratio_ge_0p63282428_ledger.csv`
