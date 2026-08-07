# Omega 4.6.2 v5 Roll8 Side-Specific Fine Valmax - 2026-07-01

## Method

This branch keeps the v5 parent and side-specific fine bracket family, then compresses the roll contract from the 9h reference to 8h. Selection is validation-primary with OOS as a safety gate only.

## Result

- Status: `RESEARCH_ROLL8_SIDE_SPECIFIC_FINE_PASS`
- Reference model: `omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `203.4821` | `220.4081` | `146.9132` | `167.4896` |
| MDD % | `-19.8890` | `-19.4679` | `-19.4446` | `-16.1774` |
| Trades | `185` | `212` | `105` | `114` |
| Avg hold h | `6.9653` | `6.0672` | `7.4238` | `6.8311` |
| Max hold h | `9.0000` | `8.0000` | `9.0000` | `8.0000` |

## Selected Candidate

- Bracket spec: `fine8_fast`
- Exposure spec: `lf0.75_sf0.95_cap4.20`
- Segment governor: `none`
- Long TP/SL: `0.0200` / `0.0300`
- Short TP/SL: `0.0250` / `0.0400`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701/roll8_side_specific_fine_valmax_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701/roll8_side_specific_fine_valmax_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701/validation_fine8_fast__lf0p75_sf0p95_cap4p20__none_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701/oos_fine8_fast__lf0p75_sf0p95_cap4p20__none_ledger.csv`
