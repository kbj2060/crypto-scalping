# Omega 4.6.2 v5 Roll12 Side-Specific Fine Valmax - 2026-07-01

## Method

This branch keeps the 12h side-specific contract and runs a narrower fine grid around the prior side-specific winner. Selection is validation-primary with an OOS safety gate; OOS metrics are not ordering keys.

## Result

- Status: `RESEARCH_ROLL12_SIDE_SPECIFIC_FINE_VALMAX_PASS`
- Reference model: `omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `320.7923` | `338.5234` | `173.9019` | `165.3214` |
| MDD % | `-19.9319` | `-19.9319` | `-17.0142` | `-17.0142` |
| Trades | `137` | `136` | `78` | `78` |
| Avg hold h | `9.4349` | `9.5049` | `10.0224` | `10.0224` |
| Max hold h | `12.0000` | `12.0000` | `12.0000` | `12.0000` |

## Selected Candidate

- Bracket spec: `fine_val_max`
- Exposure spec: `lf0.90_sf1.02_cap4.20`
- Segment governor: `none`
- Long TP/SL: `0.0225` / `0.0500`
- Short TP/SL: `0.0400` / `0.0400`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701/roll12_side_specific_fine_valmax_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701/roll12_side_specific_fine_valmax_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701/validation_fine_val_max__lf0p90_sf1p02_cap4p20__none_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701/oos_fine_val_max__lf0p90_sf1p02_cap4p20__none_ledger.csv`
