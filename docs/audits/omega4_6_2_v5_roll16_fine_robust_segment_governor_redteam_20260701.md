# Omega 4.6.2 v5 Roll16 Fine Robust Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701`
- Source best: `omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701`
- Reference: `omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL16_FINE_ROBUST_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Exposure spec: `lf0.85_sf1.02_cap4.20`
- Segment governor: `streak85_60_win12`
- Roll max hold: `16.0`
- TP/SL: `0.045` / `0.045`

| Split | Old Robust PnL | Fine Best PnL | Fine Robust PnL | Old Robust MDD | Fine Best MDD | Fine Robust MDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 316.6207% | 339.5988% | 328.3347% | -17.4852% | -19.9261% | -17.8231% |
| oos | 163.0809% | 164.1622% | 163.7874% | -19.1459% | -19.8620% | -19.5044% |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Contract Checks

- Validation accounting error max abs: `3.8163916471489756e-16`
- OOS accounting error max abs: `3.3306690738754696e-16`
- Validation notional contract error max abs: `8.881784197001252e-16`
- OOS notional contract error max abs: `8.881784197001252e-16`
- Segment governor replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701/roll16_fine_robust_segment_governor_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701/validation_lf0p85_sf1p02_cap4p20__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701/oos_lf0p85_sf1p02_cap4p20__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
