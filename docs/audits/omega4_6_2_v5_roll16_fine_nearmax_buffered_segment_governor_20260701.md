# Omega 4.6.2 v5 Roll16 Fine Near-Max Buffered Segment Governor - 2026-07-01

## Method

This branch selects a near-max validation candidate from the roll16 fine exposure sweep with a larger validation MDD buffer:

- validation PnL within `0.5pp` of the fine max-PnL candidate,
- validation MDD at least `-19.05%`,
- same short factor, cap, and segment governor as the fine max-PnL candidate,
- OOS is used only as a safety gate.

## Result

- Status: `RESEARCH_ROLL16_FINE_NEARMAX_BUFFERED_PASS`
- Source best model: `omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Fine Best Val | Buffered Val | Fine Best OOS | Buffered OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `339.5988` | `339.3129` | `164.1622` | `165.6371` |
| MDD % | `-19.9261` | `-19.0000` | `-19.8620` | `-19.8620` |
| Avg hold h | `12.3349` | `12.3349` | `13.0556` | `13.0556` |
| Max hold h | `16.0000` | `16.0000` | `16.0000` | `16.0000` |

## Selected Candidate

- Exposure spec: `lf0.95_sf1.04_cap4.30`
- Segment governor: `streak85_60_win12`
- TP/SL: `0.0450` / `0.0450`
- Research gate pass: `True`

## Artifacts

- Near-max ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_nearmax_buffered_segment_governor_20260701/roll16_fine_nearmax_buffered_segment_governor_ranking.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_nearmax_buffered_segment_governor_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_nearmax_buffered_segment_governor_20260701/validation_lf0p95_sf1p04_cap4p30__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_nearmax_buffered_segment_governor_20260701/oos_lf0p95_sf1p04_cap4p30__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
