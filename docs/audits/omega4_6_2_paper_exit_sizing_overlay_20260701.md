# Omega 4.6.2 Paper Exit + Sizing Overlay - 2026-07-01

## Method

HF paper scan pointed to exit-time stochastic control plus cost/risk-sensitive rewards. The first exit-only sweep proved that early stopping alone cuts TP-runner convexity, so this second sweep jointly tests:

- optimal-stopping style lifecycle compression,
- margin_fraction/notional rescaling under leverage cap 5,
- validation-only selection with OOS readout.

## Selected Candidate

- Stop spec: `hard96__loss48_4p5__trail72_7p0_gap2p5__stall96_lb24_min6p5`
- Exposure spec: `balanced148_cap340`
- Status: `VALIDATION_SELECTED_CANDIDATE_IMPROVES_PNL_AND_HOLD_TIME`

| Metric | Baseline Val | Selected Val | Baseline OOS | Selected OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `211.1426` | `231.0344` | `79.3178` | `105.9861` |
| MDD % | `-13.7182` | `-19.9436` | `-10.1341` | `-14.8066` |
| Avg hold h | `63.2681` | `58.0870` | `67.7885` | `62.2500` |
| Max hold h | `120.0000` | `96.0000` | `120.0000` | `96.0000` |
| Max notional | `1.9176` | `2.8381` | `2.1967` | `3.2511` |

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_paper_optstop_exit_sizing_overlay_20260701/paper_exit_sizing_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_paper_optstop_exit_sizing_overlay_20260701/validation_hard96__loss48_4p5__trail72_7p0_gap2p5__stall96_lb24_min6p5__balanced148_cap340_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_paper_optstop_exit_sizing_overlay_20260701/oos_hard96__loss48_4p5__trail72_7p0_gap2p5__stall96_lb24_min6p5__balanced148_cap340_ledger.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_paper_optstop_exit_sizing_overlay_20260701/report.json`

This remains research-only until runtime-native replay and fresh holdout are done.
