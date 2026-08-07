# Omega 4.6.2 Paper-Inspired Optimal-Stopping Exit Overlay - 2026-07-01

## Paper Rationale

- HF paper `2302.07320`: stochastic control with exit time; model-free policy/value learning can incorporate transaction costs.
- HF paper `2003.03051`: cost-sensitive reward uses log-growth minus risk and transaction-cost penalties.
- HF paper `2505.04553`: risk-sensitive RL frames variance/expected-shortfall style objectives through augmented state and actor-critic optimization.

Applied interpretation: keep Omega4.6.2 entry/exposure fixed, and replace blunt max-hold-only lifecycle with a validation-selected stopping overlay using loss cut, trailing giveback, profit-stall, and hard time stop.

## Selected Variant

- Spec: `hard120__loss48_4p5__trail72_7p0_gap2p5__stall96_lb24_min6p5`
- Selection rule: validation-only; OOS is readout only.

| Metric | Baseline Val | Selected Val | Baseline OOS | Selected OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `211.1426` | `195.8547` | `79.3178` | `79.3178` |
| MDD % | `-13.7182` | `-13.7182` | `-10.1341` | `-10.1341` |
| Avg hold h | `63.2681` | `61.6196` | `67.7885` | `67.7885` |
| Max hold h | `120.0000` | `120.0000` | `120.0000` | `120.0000` |

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_cap220_paper_optstop_exit_overlay_20260701/paper_optstop_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_cap220_paper_optstop_exit_overlay_20260701/validation_hard120__loss48_4p5__trail72_7p0_gap2p5__stall96_lb24_min6p5_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_cap220_paper_optstop_exit_overlay_20260701/oos_hard120__loss48_4p5__trail72_7p0_gap2p5__stall96_lb24_min6p5_ledger.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_cap220_paper_optstop_exit_overlay_20260701/report.json`

## Status

`NO_VALIDATION_CANDIDATE_IMPROVED_BOTH_PNL_AND_HOLD_TIME`
