# Omega 4.6.2 v5 Roll8 Side-Specific PnL Tilt Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701`
- Reference: `omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL8_PNL_TILT_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Bracket spec: `short_sl385`
- Exposure spec: `lf0.900_sf1.005_cap4.20`
- Segment governor: `none`
- Roll max hold: `8.0`
- Long TP/SL: `0.02` / `0.03`
- Short TP/SL: `0.025` / `0.0385`
- Validation avg-hold tolerance: `0.05h`

| Split | Reference PnL | Candidate PnL | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 229.4466% | 232.9667% | 6.0672h | 6.0964h | 8.0000h | 8.0000h |
| oos | 170.9863% | 175.6263% | 6.8311h | 6.7119h | 8.0000h | 8.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Contract Checks

- Validation accounting error max abs: `3.191891195797325e-16`
- OOS accounting error max abs: `3.469446951953614e-16`
- Validation notional contract error max abs: `8.881784197001252e-16`
- OOS notional contract error max abs: `4.440892098500626e-16`
- Segment governor replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/roll8_side_specific_pnl_tilt_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/validation_short_sl385__lf0p900_sf1p005_cap4p20__none_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701/oos_short_sl385__lf0p900_sf1p005_cap4p20__none_ledger.csv`
