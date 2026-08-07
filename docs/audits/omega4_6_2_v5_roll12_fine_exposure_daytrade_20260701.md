# Omega 4.6.2 v5 Roll12 Fine Exposure Daytrade - 2026-07-01

## Method

This branch keeps the roll12 TP/SL bracket fixed at `3.0%/4.0%` and fine-tunes long/short exposure around the passing roll12 candidate. Selection is validation-primary with an OOS safety gate and a lower-cap tie break.

## Result

- Status: `RESEARCH_ROLL12_FINE_EXPOSURE_PASS`
- Reference model: `omega4_6_2_v5_roll12_bracket_daytrade_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `280.4343` | `289.4460` | `142.9816` | `145.9377` |
| MDD % | `-19.5621` | `-19.9319` | `-19.1054` | `-19.4885` |
| Avg hold h | `9.1649` | `9.1649` | `9.7698` | `9.7698` |
| Max hold h | `12.0000` | `12.0000` | `12.0000` | `12.0000` |

## Selected Candidate

- Exposure spec: `lf0.90_sf1.02_cap4.20`
- Segment governor: `none`
- TP/SL: `0.0300` / `0.0400`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701/roll12_fine_exposure_daytrade_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701/roll12_fine_exposure_daytrade_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701/validation_lf0p90_sf1p02_cap4p20__none__tp0p030_sl0p040_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701/oos_lf0p90_sf1p02_cap4p20__none__tp0p030_sl0p040_ledger.csv`
