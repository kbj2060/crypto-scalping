# Omega 4.6.2 v5 Roll8 Two-Stage Exposure Buffered Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701`
- Reference: `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_BUFFERED_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Exposure

- Exposure spec: `lf1.000_sf1.200_cap5.00`
- Long/short factor: `1.0` / `1.2`
- Cap notional: `5.0`
- Validation MDD buffer floor: `-19.5%`

| Split | Reference PnL | Candidate PnL | Reference MDD | Candidate MDD | Avg Hold | Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 338.2678% | 463.0793% | -19.1071% | -19.4697% | 5.8358h | 8.0000h |
| oos | 218.8726% | 299.3083% | -15.9112% | -16.6077% | 6.4733h | 8.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Replay Checks

- Validation exposure replay: `True`
- OOS exposure replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/roll8_two_stage_exposure_buffered_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/roll8_two_stage_exposure_buffered_top20.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/validation_lf1p000_sf1p200_cap5p00_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701/oos_lf1p000_sf1p200_cap5p00_ledger.csv`
