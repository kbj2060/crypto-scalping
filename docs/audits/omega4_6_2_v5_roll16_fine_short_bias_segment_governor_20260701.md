# Omega 4.6.2 v5 Roll16 Fine Short-Bias Segment Governor - 2026-07-01

## Method

This branch selects a short-biased candidate from the roll16 fine exposure sweep using validation-only structural constraints plus an OOS safety gate:

- validation PnL within `6.0pp` of the fine max-PnL candidate,
- validation MDD at least `-18.5%`,
- long factor `<= 0.65`,
- short factor `>= 1.04`,
- cap `<= 4.30`.

## Result

- Status: `RESEARCH_ROLL16_FINE_SHORT_BIAS_PASS`
- Source best model: `omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Fine Best Val | Short-Bias Val | Fine Best OOS | Short-Bias OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `339.5988` | `335.9548` | `164.1622` | `165.4323` |
| MDD % | `-19.9261` | `-18.1606` | `-19.8620` | `-19.8620` |
| Avg hold h | `12.3349` | `12.3349` | `13.0556` | `13.0556` |

## Selected Candidate

- Exposure spec: `lf0.65_sf1.04_cap4.00`
- Segment governor: `streak85_60_win12`
- TP/SL: `0.0450` / `0.0450`
- Research gate pass: `True`

## Artifacts

- Short-bias ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_short_bias_segment_governor_20260701/roll16_fine_short_bias_segment_governor_ranking.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_short_bias_segment_governor_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_short_bias_segment_governor_20260701/validation_lf0p65_sf1p04_cap4p00__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_short_bias_segment_governor_20260701/oos_lf0p65_sf1p04_cap4p00__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
