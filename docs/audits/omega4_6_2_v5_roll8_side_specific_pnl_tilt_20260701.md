# Omega 4.6.2 v5 Roll8 Side-Specific PnL Tilt - 2026-07-01

## Method

This branch keeps the 8h roll contract and tilts the short bracket by tightening short SL from 4.0% to 3.85%, then searches a narrow exposure grid. Selection is validation-primary; OOS is a safety gate and not an ordering key.

## Result

- Status: `RESEARCH_ROLL8_SIDE_SPECIFIC_PNL_TILT_PASS`
- Reference model: `omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `229.4466` | `232.9667` | `170.9863` | `175.6263` |
| MDD % | `-19.9714` | `-19.9902` | `-16.5912` | `-16.9439` |
| Trades | `212` | `211` | `114` | `116` |
| Avg hold h | `6.0672` | `6.0964` | `6.8311` | `6.7119` |
| Max hold h | `8.0000` | `8.0000` | `8.0000` | `8.0000` |

## Selected Candidate

- Bracket spec: `short_sl385`
- Exposure spec: `lf0.900_sf1.005_cap4.20`
- Segment governor: `none`
- Long TP/SL: `0.0200` / `0.0300`
- Short TP/SL: `0.0250` / `0.0385`
- Research gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/roll8_side_specific_pnl_tilt_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/roll8_side_specific_pnl_tilt_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/validation_short_sl385__lf0p900_sf1p005_cap4p20__none_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/oos_short_sl385__lf0p900_sf1p005_cap4p20__none_ledger.csv`
