# Omega 4.6.2 v5 Roll8 Side-Specific Fine Exposure Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701`
- Reference: `omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL8_FINE_EXPOSURE_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Bracket spec: `fine8_fast`
- Exposure spec: `lf0.900_sf0.975_cap4.20`
- Segment governor: `none`
- Roll max hold: `8.0`
- Long TP/SL: `0.02` / `0.03`
- Short TP/SL: `0.025` / `0.04`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 220.4081% | 229.4466% | 6.0672h | 6.0672h | 8.0000h | 8.0000h |
| oos | 167.4896% | 170.9863% | 6.8311h | 6.8311h | 8.0000h | 8.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Contract Checks

- Validation accounting error max abs: `2.983724378680108e-16`
- OOS accounting error max abs: `3.469446951953614e-16`
- Validation notional contract error max abs: `4.440892098500626e-16`
- OOS notional contract error max abs: `4.440892098500626e-16`
- Segment governor replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701/roll8_side_specific_fine_exposure_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701/validation_fine8_fast__lf0p900_sf0p975_cap4p20__none_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701/oos_fine8_fast__lf0p900_sf0p975_cap4p20__none_ledger.csv`
