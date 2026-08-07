# Omega 4.6.2 v5 Roll10 Bracket Daytrade - 2026-07-01

## Method

This branch starts from the v5 parent and splits positions into `<=10h` path-causal segments. It is selected as a middle ground between the 12h daytrade branch and the non-promoted 8h probe.

## Result

- Status: `RESEARCH_ROLL10_DAYTRADE_PASS`
- Reference model: `omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `289.4460` | `237.5114` | `145.9377` | `128.2522` |
| MDD % | `-19.9319` | `-18.8794` | `-19.4885` | `-19.8280` |
| Trades | `141` | `158` | `80` | `91` |
| Avg hold h | `9.1649` | `8.1698` | `9.7698` | `8.5778` |
| Max hold h | `12.0000` | `10.0000` | `12.0000` | `10.0000` |

## Selected Candidate

- Exposure spec: `lf0.80_sf0.95_cap4.00`
- Segment governor: `none`
- TP/SL: `0.0300` / `0.0400`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_bracket_daytrade_20260701/roll10_bracket_daytrade_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_bracket_daytrade_20260701/roll10_bracket_daytrade_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_bracket_daytrade_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_bracket_daytrade_20260701/validation_lf0p80_sf0p95_cap4p00__none__tp0p030_sl0p040_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_bracket_daytrade_20260701/oos_lf0p80_sf0p95_cap4p00__none__tp0p030_sl0p040_ledger.csv`
