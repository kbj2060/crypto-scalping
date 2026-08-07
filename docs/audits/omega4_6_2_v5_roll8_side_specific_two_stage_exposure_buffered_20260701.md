# Omega 4.6.2 v5 Roll8 Side-Specific Two-Stage Exposure Buffered - 2026-07-01

## Method

This branch starts from `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701` and applies a ledger-level exposure overlay to already selected entries/exits. It does not change hold time. Selection is validation-primary with a validation MDD buffer floor of `-19.5%`; OOS is a safety gate only.

## Result

- Status: `RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_BUFFERED_PASS`
- Reference model: `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `611.3029` | `718.2058` | `194.5778` | `220.2756` |
| MDD % | `-19.1071` | `-19.1071` | `-18.5253` | `-19.9359` |
| Trades | `186` | `186` | `97` | `97` |
| Avg hold h | `5.8723` | `5.8723` | `6.6409` | `6.6409` |
| Max hold h | `8.0000` | `8.0000` | `8.0000` | `8.0000` |

## Selected Candidate

- Exposure spec: `lf1.000_sf1.080_cap4.60`
- Long/short factor: `1.0` / `1.08`
- Cap notional: `4.6`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/roll8_two_stage_exposure_buffered_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/roll8_two_stage_exposure_buffered_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/validation_lf1p000_sf1p080_cap4p60_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/oos_lf1p000_sf1p080_cap4p60_ledger.csv`
