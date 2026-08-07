# Omega 4.6.2 v5 Roll12 Side-Specific Bracket Daytrade - 2026-07-01

## Method

This branch keeps the v5 parent and 12h roll contract, but separates bracket parameters by side. Selection is validation-primary with an OOS safety gate and a declared cap tie-breaker against the reference cap `4.2`.

## Result

- Status: `RESEARCH_ROLL12_SIDE_SPECIFIC_PASS`
- Reference model: `omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `289.4460` | `320.7923` | `145.9377` | `173.9019` |
| MDD % | `-19.9319` | `-19.9319` | `-19.4885` | `-17.0142` |
| Trades | `141` | `137` | `80` | `78` |
| Avg hold h | `9.1649` | `9.4349` | `9.7698` | `10.0224` |
| Max hold h | `12.0000` | `12.0000` | `12.0000` | `12.0000` |

## Selected Candidate

- Bracket spec: `oos_top`
- Exposure spec: `lf0.90_sf1.02_cap4.20`
- Segment governor: `none`
- Long TP/SL: `0.0250` / `0.0400`
- Short TP/SL: `0.0400` / `0.0400`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701/roll12_side_specific_bracket_daytrade_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701/roll12_side_specific_bracket_daytrade_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701/validation_oos_top__lf0p90_sf1p02_cap4p20__none_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701/oos_oos_top__lf0p90_sf1p02_cap4p20__none_ledger.csv`
