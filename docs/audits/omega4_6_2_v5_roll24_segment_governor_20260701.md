# Omega 4.6.2 v5 Roll24 Segment Governor - 2026-07-01

## Method

This sweep starts from the v5 parent, splits trades into fixed 24h roll segments, and tunes only segment-level exposure plus a path-causal segment loss governor. Selection is validation-primary with an OOS safety gate; fresh holdout is required before any live claim.

## Result

- Status: `RESEARCH_DAYTRADE_UPGRADE_IMPROVES_REFERENCE_WITH_OOS_SAFETY_GATE`
- Reference model: `omega4_6_2_v5_roll24_daytrade_overlay_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `249.1403` | `276.9693` | `142.1316` | `143.7794` |
| MDD % | `-19.9363` | `-19.4048` | `-18.6719` | `-19.9164` |
| Trades | `64` | `64` | `39` | `39` |
| Avg hold h | `20.2917` | `20.2917` | `20.1303` | `20.1303` |
| Max hold h | `24.0000` | `24.0000` | `24.0000` | `24.0000` |

## Selected Candidate

- Exposure spec: `long105_short107_cap405`
- Segment governor: `streak90_70_win12`
- Validation upgrade gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_segment_governor_20260701/v5_roll24_segment_governor_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_segment_governor_20260701/v5_roll24_segment_governor_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_segment_governor_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_segment_governor_20260701/validation_long105_short107_cap405__streak90_70_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_segment_governor_20260701/oos_long105_short107_cap405__streak90_70_win12_ledger.csv`
