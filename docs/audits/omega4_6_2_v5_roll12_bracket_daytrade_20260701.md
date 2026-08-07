# Omega 4.6.2 v5 Roll12 Bracket Daytrade - 2026-07-01

## Method

This branch starts from the v5 parent and splits positions into `<=12h` path-causal segments. Each segment uses a fixed TP/SL bracket selected on validation with an OOS safety gate. It is intentionally an ultra-short day-trading branch, not a replacement for the higher-PnL 16h branch.

## Result

- Status: `RESEARCH_ROLL12_DAYTRADE_PASS`
- Reference model: `omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `316.6207` | `280.4343` | `163.0809` | `142.9816` |
| MDD % | `-17.4852` | `-19.5621` | `-19.1459` | `-19.1054` |
| Trades | `105` | `141` | `60` | `80` |
| Avg hold h | `12.3349` | `9.1649` | `13.0556` | `9.7698` |
| Max hold h | `16.0000` | `12.0000` | `16.0000` | `12.0000` |

## Selected Candidate

- Exposure spec: `long085_short100_cap430`
- Segment governor: `none`
- TP/SL: `0.0300` / `0.0400`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_bracket_daytrade_20260701/roll12_bracket_daytrade_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_bracket_daytrade_20260701/roll12_bracket_daytrade_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_bracket_daytrade_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_bracket_daytrade_20260701/validation_long085_short100_cap430__none__tp0p030_sl0p040_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_bracket_daytrade_20260701/oos_long085_short100_cap430__none__tp0p030_sl0p040_ledger.csv`
