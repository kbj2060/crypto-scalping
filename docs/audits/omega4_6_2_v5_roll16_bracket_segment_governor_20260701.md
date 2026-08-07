# Omega 4.6.2 v5 Roll16 Bracket Segment Governor - 2026-07-01

## Method

This candidate starts from the v5 parent, splits positions into `<=16h` segments, and exits each segment early when a path-causal `4.5%` TP or `4.5%` SL is touched. Same-bar TP/SL ambiguity is handled conservatively by taking SL first. Segment exposure/governor selection is validation-primary with an OOS safety gate; fresh holdout is required before any live claim.

## Result

- Status: `RESEARCH_ROLL16_BRACKET_UPGRADE_IMPROVES_PNL_AND_HOLD`
- Reference model: `omega4_6_2_v5_roll24_segment_governor_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `276.9693` | `319.3786` | `143.7794` | `154.8053` |
| MDD % | `-19.4048` | `-19.9261` | `-19.9164` | `-19.1459` |
| Trades | `64` | `105` | `39` | `60` |
| Avg hold h | `20.2917` | `12.3349` | `20.1303` | `13.0556` |
| Max hold h | `24.0000` | `16.0000` | `24.0000` | `16.0000` |

## Selected Candidate

- Exposure spec: `long100_short100_cap430`
- Segment governor: `streak85_60_win12`
- TP/SL: `0.0450` / `0.0450`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_segment_governor_20260701/roll16_bracket_segment_governor_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_segment_governor_20260701/roll16_bracket_segment_governor_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_segment_governor_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_segment_governor_20260701/validation_long100_short100_cap430__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_segment_governor_20260701/oos_long100_short100_cap430__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
