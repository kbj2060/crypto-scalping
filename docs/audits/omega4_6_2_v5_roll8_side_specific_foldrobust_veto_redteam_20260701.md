# Omega 4.6.2 v5 Roll8 Side-Specific Fold-Robust Veto Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701`
- Reference: `omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL8_FOLDROBUST_VETO_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Veto

- Feature: `big_trade_ratio`
- Rule: `big_trade_ratio >= 0.63282428`
- Quantile: `0.85`
- Validation/OOS vetoed shorts: `26` / `5`
- Validation fold PnL deltas: `[0.0, 0.0, 3.2871, 10.2803]`
- Validation fold max avg-hold delta: `0.0`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 232.9667% | 274.0100% | 6.0964h | 5.9689h | 8.0000h | 8.0000h |
| oos | 175.6263% | 204.5934% | 6.7119h | 6.7042h | 8.0000h | 8.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Replay Checks

- Validation feature-veto replay: `True`
- OOS feature-veto replay: `True`
- Fold summary parity: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701/roll8_side_specific_foldrobust_veto_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701/roll8_side_specific_foldrobust_veto_top20.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701/validation_big_trade_ratio_ge_0p63282428_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701/oos_big_trade_ratio_ge_0p63282428_ledger.csv`
