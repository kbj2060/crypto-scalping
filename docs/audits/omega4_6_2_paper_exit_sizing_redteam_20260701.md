# Omega 4.6.2 Paper Exit + Sizing Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_paper_optstop_exit_sizing_overlay_20260701`
- Base: `omega4_6_2_cap220_short_boost125_time_stop120h_20260630`
- Verdict: `RESEARCH_UPGRADE_PASS_FULL_LIVE_BLOCKED`
- Research upgrade pass: `True`
- Full live pass: `False`

## Selected Candidate

- Stop spec: `hard96__loss48_4p5__trail72_7p0_gap2p5__stall96_lb24_min6p5`
- Exposure spec: `balanced148_cap340`
- Selection scope: `validation_only; OOS readout only`

| Split | Baseline PnL | Candidate PnL | Baseline MDD | Candidate MDD | Baseline Avg Hold | Candidate Avg Hold | Baseline Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 211.1426% | 231.0344% | -13.7182% | -19.9436% | 63.2681h | 58.0870h | 120.0000h | 96.0000h |
| oos | 79.3178% | 105.9861% | -10.1341% | -14.8066% | 67.7885h | 62.2500h | 120.0000h | 96.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `max_hold_24h_daytrading_requirement`: {'validation': 96.0, 'oos': 96.0}

## Research Failures

- None.

## Warnings

- `validation_mdd_buffer_over_1pp`: {'validation_mdd': -19.943561382303866, 'buffer_to_20pct': 0.05643861769613423}

## Contract Checks

- Validation accounting error max abs: `1.457167719820518e-16`
- OOS accounting error max abs: `9.71445146547012e-17`
- Validation notional contract error max abs: `8.881784197001252e-16`
- OOS notional contract error max abs: `6.661338147750939e-16`
- Validation max leverage: `5.0`
- OOS max leverage: `5.0`
- Validation max notional: `2.838051426395742`
- OOS max notional: `3.251114229694377`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_paper_optstop_exit_sizing_overlay_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_paper_optstop_exit_sizing_overlay_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_paper_optstop_exit_sizing_overlay_20260701/paper_exit_sizing_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_paper_optstop_exit_sizing_overlay_20260701/validation_hard96__loss48_4p5__trail72_7p0_gap2p5__stall96_lb24_min6p5__balanced148_cap340_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_paper_optstop_exit_sizing_overlay_20260701/oos_hard96__loss48_4p5__trail72_7p0_gap2p5__stall96_lb24_min6p5__balanced148_cap340_ledger.csv`
