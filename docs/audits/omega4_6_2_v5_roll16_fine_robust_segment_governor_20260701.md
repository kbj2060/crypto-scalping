# Omega 4.6.2 v5 Roll16 Fine Robust Segment Governor - 2026-07-01

## Method

This branch selects a robust candidate from the roll16 fine exposure sweep:

- candidate must pass the research gate and OOS safety gate,
- validation PnL must be within `15.0pp` of the fine max-PnL candidate,
- validation MDD must be at least `-18.0%`,
- exposure cap must be `<= 4.20`.

## Result

- Status: `RESEARCH_ROLL16_FINE_ROBUST_PASS`
- Source best model: `omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701`
- Reference robust model: `omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701`

| Metric | Old Robust Val | Fine Best Val | Fine Robust Val | Old Robust OOS | Fine Best OOS | Fine Robust OOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PnL % | `316.6207` | `339.5988` | `328.3347` | `163.0809` | `164.1622` | `163.7874` |
| MDD % | `-17.4852` | `-19.9261` | `-17.8231` | `-19.1459` | `-19.8620` | `-19.5044` |
| Avg hold h | `12.3349` | `12.3349` | `12.3349` | `13.0556` | `13.0556` | `13.0556` |

## Selected Candidate

- Exposure spec: `lf0.85_sf1.02_cap4.20`
- Segment governor: `streak85_60_win12`
- TP/SL: `0.0450` / `0.0450`
- Research gate pass: `True`

## Artifacts

- Robust ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701/roll16_fine_robust_segment_governor_ranking.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701/validation_lf0p85_sf1p02_cap4p20__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701/oos_lf0p85_sf1p02_cap4p20__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
