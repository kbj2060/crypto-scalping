# Omega 4.6.2 Hold Fine Exit + Sizing Overlay - 2026-07-01

## Method

This sweep keeps the Omega 4.6.2 cap220 source ledgers and the paper exit+sizing framework, then searches finer hard-stop horizons between 72h and 96h plus validation-only exposure rescaling.

## Result

- Status: `NO_VALIDATION_UPGRADE_IMPROVED_REFERENCE_PNL_AND_HOLD`
- Selection scope: `validation_only; OOS readout only`
- Reference model: `omega4_6_2_paper_optstop_exit_sizing_overlay_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `231.0344` | `231.0344` | `105.9861` | `105.9861` |
| MDD % | `-19.9436` | `-19.9436` | `-14.8066` | `-14.8066` |
| Avg hold h | `58.0870` | `58.0870` | `62.2500` | `62.2500` |
| Max hold h | `96.0000` | `96.0000` | `96.0000` | `96.0000` |

## Selected Candidate

- Stop spec: `hard96__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0`
- Exposure spec: `balanced148_cap340`
- Validation upgrade gate pass: `False`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_hold_fine_exit_sizing_overlay_20260701/hold_fine_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_hold_fine_exit_sizing_overlay_20260701/hold_fine_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_hold_fine_exit_sizing_overlay_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_hold_fine_exit_sizing_overlay_20260701/validation_hard96__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0__balanced148_cap340_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_hold_fine_exit_sizing_overlay_20260701/oos_hard96__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0__balanced148_cap340_ledger.csv`
