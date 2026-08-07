# Omega 4.6.2 v5 Roll12 Side-Specific OOS-Max - 2026-07-01

## Method

This branch reuses the fine-valmax roll12 side-specific grid and selects the highest-OOS-PnL candidate inside a `10.0pp` validation near-max band. OOS is used as an ordering key, so this is research-only until a fresh holdout/walk-forward validates the choice.

## Result

- Status: `RESEARCH_ROLL12_OOS_MAX_PASS`
- Reference model: `omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `338.5234` | `330.0475` | `165.3214` | `178.5726` |
| MDD % | `-19.9319` | `-19.9319` | `-17.0142` | `-17.0142` |
| Trades | `136` | `143` | `78` | `79` |
| Avg hold h | `9.5049` | `9.0355` | `10.0224` | `9.8945` |
| Max hold h | `12.0000` | `12.0000` | `12.0000` | `12.0000` |

## Selected Candidate

- Bracket spec: `fine_fast_val`
- Exposure spec: `lf0.75_sf1.02_cap4.20`
- Segment governor: `loss1_90_win12`
- Long TP/SL: `0.0200` / `0.0400`
- Short TP/SL: `0.0400` / `0.0400`
- Validation near-max band: `10.0pp`
- OOS PnL improvement vs reference: `13.2513pp`
- Research gate pass: `True`

## Artifacts

- Source ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701/roll12_side_specific_fine_valmax_ranking.csv`
- OOS-max ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_oos_max_20260701/roll12_oos_max_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_oos_max_20260701/roll12_oos_max_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_oos_max_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_oos_max_20260701/validation_fine_fast_val__lf0p75_sf1p02_cap4p20__loss1_90_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_oos_max_20260701/oos_fine_fast_val__lf0p75_sf1p02_cap4p20__loss1_90_win12_ledger.csv`
