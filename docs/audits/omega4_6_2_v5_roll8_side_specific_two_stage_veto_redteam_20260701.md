# Omega 4.6.2 v5 Roll8 Side-Specific Two-Stage Veto Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701`
- Reference: `omega4_6_2_v5_roll8_side_specific_feature_veto_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL8_TWO_STAGE_VETO_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Second-Stage Veto

- Feature: `cvp_vah_val_width`
- Rule: `cvp_vah_val_width <= 0.14`
- Quantile: `0.05`
- Validation/OOS second-stage vetoed shorts: `12` / `22`
- Validation fold PnL deltas: `[0.0, 0.0, 3.6938, 0.0]`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 323.6915% | 338.2678% | 5.9423h | 5.8358h | 8.0000h | 8.0000h |
| oos | 207.0208% | 218.8726% | 6.6821h | 6.4733h | 8.0000h | 8.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Replay Checks

- Validation second-stage replay: `True`
- OOS second-stage replay: `True`
- Fold summary parity: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/roll8_side_specific_two_stage_veto_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/roll8_side_specific_two_stage_veto_top20.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/validation_cvp_vah_val_width_le_0p14_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/oos_cvp_vah_val_width_le_0p14_ledger.csv`
