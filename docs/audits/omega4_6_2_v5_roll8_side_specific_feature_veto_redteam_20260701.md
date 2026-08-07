# Omega 4.6.2 v5 Roll8 Side-Specific Feature Veto Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll8_side_specific_feature_veto_20260701`
- Reference: `omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL8_FEATURE_VETO_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Veto

- Feature: `volume`
- Rule: `volume <= 5173.597`
- Quantile: `0.15`
- Validation/OOS vetoed shorts: `32` / `19`
- Search scope: `validation_primary_single_entry_feature_short_veto_with_oos_reference_safety_gate; fresh_holdout_required`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 232.9667% | 323.6915% | 6.0964h | 5.9423h | 8.0000h | 8.0000h |
| oos | 175.6263% | 207.0208% | 6.7119h | 6.6821h | 8.0000h | 8.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Replay Checks

- Validation feature-veto replay: `True`
- OOS feature-veto replay: `True`
- Lookahead regex: `(timestamp|exit_|raw_exit|mfe|mae|trade_return|net_per_notional|log_return|win$|reason|hold|entry_i|exit_i|segment|roundtrip|tp_move|sl_move|source|ledger|report|notional|leverage|margin|return|pnl|profit|loss|stop|take_profit|stop_loss|paper_|borrow_|roll8_|roll24_|cum|peak|dd)`

## Contract Checks

- Validation accounting error max abs: `3.191891195797325e-16`
- OOS accounting error max abs: `3.469446951953614e-16`
- Validation notional contract error max abs: `4.440892098500626e-16`
- OOS notional contract error max abs: `4.440892098500626e-16`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/roll8_side_specific_feature_veto_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/roll8_side_specific_feature_veto_top20.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/validation_volume_le_5173p597_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701/oos_volume_le_5173p597_ledger.csv`
