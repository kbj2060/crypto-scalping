# Omega 4.6.2 v5 Roll10 Side-Specific Bracket Daytrade - 2026-07-01

## Method

This branch starts from the same v5 parent and 10h roll contract as `omega4_6_2_v5_roll10_bracket_daytrade_20260701`, but separates bracket labels by side. Long positions use one TP/SL pair and short positions use another TP/SL pair. Selection remains validation-primary with an OOS safety gate and a declared cap tie-breaker against the reference cap `4.2`.

## Result

- Status: `RESEARCH_ROLL10_SIDE_SPECIFIC_PASS`
- Reference model: `omega4_6_2_v5_roll10_bracket_daytrade_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `237.5114` | `261.9047` | `128.2522` | `131.0583` |
| MDD % | `-18.8794` | `-19.6570` | `-19.8280` | `-19.6438` |
| Trades | `158` | `167` | `91` | `97` |
| Avg hold h | `8.1698` | `7.7241` | `8.5778` | `8.0430` |
| Max hold h | `10.0000` | `10.0000` | `10.0000` | `10.0000` |

## Selected Candidate

- Bracket spec: `fast_short`
- Exposure spec: `lf0.90_sf1.02_cap4.20`
- Segment governor: `loss1_90_win10`
- Long TP/SL: `0.0250` / `0.0350`
- Short TP/SL: `0.0250` / `0.0400`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701/roll10_side_specific_bracket_daytrade_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701/roll10_side_specific_bracket_daytrade_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701/validation_fast_short__lf0p90_sf1p02_cap4p20__loss1_90_win10_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701/oos_fast_short__lf0p90_sf1p02_cap4p20__loss1_90_win10_ledger.csv`
