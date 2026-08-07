# Omega 4.6.2 v5 Roll4 OOS-Max Two-Stage Exposure - 2026-07-01

## Method

This branch keeps the roll4 `4h` path and selects the highest-OOS-PnL candidate inside a `20.0pp` validation near-max band. OOS is used as an ordering key, so this is research-only until fresh holdout is available.

## Result

- Status: `RESEARCH_ROLL4_OOS_MAX_PASS`
- Reference model: `omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `317.3833` | `306.0689` | `140.4955` | `159.8935` |
| MDD % | `-16.8787` | `-16.8787` | `-19.9848` | `-19.9848` |
| Avg hold h | `3.4727` | `3.5346` | `3.6140` | `3.5878` |
| Max hold h | `4.0000` | `4.0000` | `4.0000` | `4.0000` |

## Selected Candidate

- Exposure spec: `lf0.000_sf1.100_cap4.00`
- Validation near-max band: `20.0pp`
- OOS MDD buffer to -20%: `0.0152pp`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_oos_max_20260701/roll4_oos_max_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_oos_max_20260701/roll4_oos_max_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_oos_max_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_oos_max_20260701/validation_lf0p000_sf1p100_cap4p00_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_oos_max_20260701/oos_lf0p000_sf1p100_cap4p00_ledger.csv`
