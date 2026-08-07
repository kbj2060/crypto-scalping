# Omega 4.6.2 v5 Roll10 Side-Specific Fine Valmax - 2026-07-01

## Method

This branch keeps the 10h side-specific contract and runs a narrower fine grid around the prior 10h side-specific winner. Selection is validation-primary with an OOS safety gate; OOS metrics are not ordering keys.

## Result

- Status: `RESEARCH_ROLL10_SIDE_SPECIFIC_FINE_VALMAX_PASS`
- Reference model: `omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `261.9047` | `277.2980` | `131.0583` | `123.7006` |
| MDD % | `-19.6570` | `-19.6102` | `-19.6438` | `-19.6438` |
| Trades | `167` | `172` | `97` | `97` |
| Avg hold h | `7.7241` | `7.4981` | `8.0430` | `8.0430` |
| Max hold h | `10.0000` | `10.0000` | `10.0000` | `10.0000` |

## Selected Candidate

- Bracket spec: `fine10_valmax`
- Exposure spec: `lf0.90_sf1.02_cap4.20`
- Segment governor: `loss1_90_win10`
- Long TP/SL: `0.0200` / `0.0450`
- Short TP/SL: `0.0250` / `0.0400`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_fine_valmax_20260701/roll10_side_specific_fine_valmax_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_fine_valmax_20260701/roll10_side_specific_fine_valmax_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_fine_valmax_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_fine_valmax_20260701/validation_fine10_valmax__lf0p90_sf1p02_cap4p20__loss1_90_win10_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll10_side_specific_fine_valmax_20260701/oos_fine10_valmax__lf0p90_sf1p02_cap4p20__loss1_90_win10_ledger.csv`
