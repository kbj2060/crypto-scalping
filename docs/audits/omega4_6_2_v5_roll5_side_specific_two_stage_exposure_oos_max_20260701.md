# Omega 4.6.2 v5 Roll5 OOS-Max Two-Stage Exposure - 2026-07-01

## Method

This branch keeps the roll5 `5h` path and selects the highest-OOS-PnL candidate inside a `10.0pp` validation near-max band. OOS is used as an ordering key, so this is research-only until fresh holdout is available.

## Result

- Status: `RESEARCH_ROLL5_OOS_MAX_PASS`
- Reference model: `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `302.8578` | `296.9050` | `169.8794` | `187.6595` |
| MDD % | `-18.5696` | `-18.5696` | `-16.0647` | `-16.0647` |
| Avg hold h | `4.2333` | `4.2333` | `4.4281` | `4.4281` |
| Max hold h | `5.0000` | `5.0000` | `5.0000` | `5.0000` |

## Selected Candidate

- Exposure spec: `lf0.100_sf1.000_cap4.40`
- Validation near-max band: `10.0pp`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701/roll5_oos_max_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701/roll5_oos_max_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701/validation_lf0p100_sf1p000_cap4p40_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701/oos_lf0p100_sf1p000_cap4p40_ledger.csv`
