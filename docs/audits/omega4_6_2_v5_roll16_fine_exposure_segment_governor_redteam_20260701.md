# Omega 4.6.2 v5 Roll16 Fine Exposure Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701`
- Reference: `omega4_6_2_v5_roll16_bracket_segment_governor_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROLL16_FINE_EXPOSURE_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Exposure spec: `lf1.00_sf1.04_cap4.30`
- Segment governor: `streak85_60_win12`
- Roll max hold: `16.0`
- TP/SL: `0.045` / `0.045`

| Split | Reference PnL | Candidate PnL | Reference MDD | Candidate MDD | Reference Avg Hold | Candidate Avg Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 319.3786% | 339.5988% | -19.9261% | -19.9261% | 12.3349h | 12.3349h |
| oos | 154.8053% | 164.1622% | -19.1459% | -19.8620% | 13.0556h | 13.0556h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Contract Checks

- Validation accounting error max abs: `3.8163916471489756e-16`
- OOS accounting error max abs: `3.608224830031759e-16`
- Validation notional contract error max abs: `4.440892098500626e-16`
- OOS notional contract error max abs: `4.440892098500626e-16`
- Segment governor replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701/roll16_fine_exposure_segment_governor_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701/validation_lf1p00_sf1p04_cap4p30__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701/oos_lf1p00_sf1p04_cap4p30__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
