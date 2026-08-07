# Omega 4.6.2 v5 Roll2 OOS-Max Two-Stage Exposure - 2026-07-01

## Method

This branch keeps the roll2 `2h` path and selects the highest-OOS-PnL candidate inside a `10.0pp` validation near-max band. OOS is used as an ordering key, so this is research-only until fresh holdout is available.

## Result

- Status: `NO_ROLL2_OOS_MAX_PASSING_CANDIDATE`
- Reference model: `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `99.4419` | `99.4419` | `64.7032` | `64.7032` |
| MDD % | `-48.8138` | `-48.8138` | `-35.0439` | `-35.0439` |
| Avg hold h | `1.9069` | `1.9069` | `1.9279` | `1.9279` |
| Max hold h | `2.0000` | `2.0000` | `2.0000` | `2.0000` |

## Selected Candidate

- Exposure spec: `lf0.800_sf1.500_cap5.00`
- Validation near-max band: `10.0pp`
- Research gate pass: `False`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701/roll2_oos_max_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701/roll2_oos_max_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701/validation_lf0p800_sf1p500_cap5p00_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701/oos_lf0p800_sf1p500_cap5p00_ledger.csv`
