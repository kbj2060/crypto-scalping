# Omega 4.6.2 v5 Robust Roll16 Bracket Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701`
- Source best: `omega4_6_2_v5_roll16_bracket_segment_governor_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Verdict: `RESEARCH_ROBUST_PASS_FULL_LIVE_BLOCKED`
- Research pass: `True`
- Full live pass: `False`

## Selected Candidate

- Exposure spec: `long070_short100_cap410`
- Segment governor: `streak85_60_win12`
- Roll max hold: `16.0`
- TP/SL: `0.045` / `0.045`

| Split | 24h Reference PnL | Roll16 Best PnL | Robust PnL | 24h Reference MDD | Roll16 Best MDD | Robust MDD | Robust Avg Hold | Robust Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 276.9693% | 319.3786% | 316.6207% | -19.4048% | -19.9261% | -17.4852% | 12.3349h | 16.0000h |
| oos | 143.7794% | 154.8053% | 163.0809% | -19.9164% | -19.1459% | -19.1459% | 13.0556h | 16.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Contract Checks

- Validation accounting error max abs: `3.608224830031759e-16`
- OOS accounting error max abs: `3.0531133177191805e-16`
- Validation notional contract error max abs: `4.440892098500626e-16`
- OOS notional contract error max abs: `4.440892098500626e-16`
- Segment governor replay: `True`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701/robust_roll16_bracket_segment_governor_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701/validation_long070_short100_cap410__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701/oos_long070_short100_cap410__streak85_60_win12__tp0p045_sl0p045_ledger.csv`
