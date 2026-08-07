# Omega 4.6.2 v5 Roll9 Side-Specific Fine Valmax - 2026-07-01

## Method

This branch keeps the v5 parent and side-specific bracket family, but compresses the roll contract from the 10h reference to 9h. Selection is validation-primary with OOS as a safety gate only.

## Result

- Status: `RESEARCH_ROLL9_SIDE_SPECIFIC_FINE_PASS`
- Reference model: `omega4_6_2_v5_roll10_side_specific_fine_valmax_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `277.2980` | `203.4821` | `123.7006` | `146.9132` |
| MDD % | `-19.6102` | `-19.8890` | `-19.6438` | `-19.4446` |
| Trades | `172` | `185` | `97` | `105` |
| Avg hold h | `7.4981` | `6.9653` | `8.0430` | `7.4238` |
| Max hold h | `10.0000` | `9.0000` | `10.0000` | `9.0000` |

## Selected Candidate

- Bracket spec: `fine9_fast`
- Exposure spec: `lf0.70_sf1.00_cap3.80`
- Segment governor: `none`
- Long TP/SL: `0.0200` / `0.0300`
- Short TP/SL: `0.0250` / `0.0400`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701/roll9_side_specific_fine_valmax_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701/roll9_side_specific_fine_valmax_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701/validation_fine9_fast__lf0p70_sf1p00_cap3p80__none_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701/oos_fine9_fast__lf0p70_sf1p00_cap3p80__none_ledger.csv`
