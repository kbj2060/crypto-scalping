# Omega 4.6.2 v5 Roll16 Bracket Robust Segment Governor - 2026-07-01

## Method

This branch uses the same roll16 bracket candidate grid as `omega4_6_2_v5_roll16_bracket_segment_governor_20260701`, but changes the selection rule to a validation robustness rule:

- candidate must pass the research gate and OOS safety gate,
- validation PnL must be within `3.0pp` of the best roll16 validation PnL,
- validation MDD must be at least `-18.0%`,
- exposure cap must be `<= 4.10`.

The selected branch is therefore not a fresh OOS optimization claim; it is a lower-risk validation branch with OOS safety still disclosed.

## Result

- Status: `RESEARCH_ROBUST_ROLL16_BRANCH_PASS`
- Source best model: `omega4_6_2_v5_roll16_bracket_segment_governor_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | 24h Reference Val | Roll16 Best Val | Robust Val | 24h Reference OOS | Roll16 Best OOS | Robust OOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PnL % | `276.9693` | `319.3786` | `316.6207` | `143.7794` | `154.8053` | `163.0809` |
| MDD % | `-19.4048` | `-19.9261` | `-17.4852` | `-19.9164` | `-19.1459` | `-19.1459` |
| Avg hold h | `20.2917` | `12.3349` | `12.3349` | `20.1303` | `13.0556` | `13.0556` |
| Max hold h | `24.0000` | `16.0000` | `16.0000` | `24.0000` | `16.0000` | `16.0000` |

## Selected Candidate

- Exposure spec: `long070_short100_cap410`
- Segment governor: `streak85_60_win12`
- TP/SL: `0.0450` / `0.0450`
- Research gate pass: `True`

## Artifacts

- Robust ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701/robust_roll16_bracket_segment_governor_ranking.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701/validation_long070_short100_cap410__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701/oos_long070_short100_cap410__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
