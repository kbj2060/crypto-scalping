# Omega 4.6.2 v5 Roll16 Fine Exposure Segment Governor - 2026-07-01

## Method

This branch keeps the roll16 TP/SL bracket fixed at `4.5%/4.5%` and fine-tunes long/short exposure around the prior roll16 winner. Selection is validation-primary with an OOS safety gate.

## Result

- Status: `RESEARCH_ROLL16_FINE_EXPOSURE_UPGRADE_PASS`
- Reference model: `omega4_6_2_v5_roll16_bracket_segment_governor_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `319.3786` | `339.5988` | `154.8053` | `164.1622` |
| MDD % | `-19.9261` | `-19.9261` | `-19.1459` | `-19.8620` |
| Avg hold h | `12.3349` | `12.3349` | `13.0556` | `13.0556` |
| Max hold h | `16.0000` | `16.0000` | `16.0000` | `16.0000` |

## Selected Candidate

- Exposure spec: `lf1.00_sf1.04_cap4.30`
- Segment governor: `streak85_60_win12`
- TP/SL: `0.0450` / `0.0450`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701/roll16_fine_exposure_segment_governor_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701/roll16_fine_exposure_segment_governor_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701/validation_lf1p00_sf1p04_cap4p30__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701/oos_lf1p00_sf1p04_cap4p30__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
