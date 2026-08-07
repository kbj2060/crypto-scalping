# Omega 4.6.2 Roll24 Segment Governor Sweep - 2026-07-01

## Method

This sweep rebuilds the v4 90h exit as 24h-or-less roll segments, then retunes exposure and applies a path-causal segment-level loss-window governor.

## Result

- Status: `NO_VALIDATION_DAYTRADE_UPGRADE_IMPROVED_ROLL24_REFERENCE`
- Reference model: `omega4_6_2_roll24_daytrade_overlay_20260701`
- Selection scope: `validation_only; OOS readout only`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `237.4884` | `338.1131` | `141.2725` | `196.4894` |
| MDD % | `-19.9815` | `-24.9060` | `-18.5806` | `-27.6614` |
| Trades | `64` | `64` | `39` | `39` |
| Avg hold h | `20.2917` | `20.2917` | `20.1303` | `20.1303` |
| Max hold h | `24.0000` | `24.0000` | `24.0000` | `24.0000` |

## Selected Candidate

- Exposure spec: `long120_short240_cap500`
- Segment governor: `streak75_55_win12`
- Validation upgrade gate pass: `False`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_segment_governor_sweep_20260701/roll24_segment_governor_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_segment_governor_sweep_20260701/roll24_segment_governor_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_segment_governor_sweep_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_segment_governor_sweep_20260701/validation_long120_short240_cap500__streak75_55_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_segment_governor_sweep_20260701/oos_long120_short240_cap500__streak75_55_win12_ledger.csv`
