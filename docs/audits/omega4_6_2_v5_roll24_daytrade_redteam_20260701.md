# Omega 4.6.2 v5 Roll24 Daytrade Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_v5_roll24_daytrade_overlay_20260701`
- Parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Reference daytrade model: `omega4_6_2_roll24_daytrade_overlay_20260701`
- Verdict: `DAYTRADE_RESEARCH_PASS_FULL_LIVE_BLOCKED`
- Daytrade research pass: `True`
- PnL upgrade vs reference: `True`
- Full live pass: `False`

| Split | Reference PnL | v5 Roll24 PnL | Reference MDD | v5 Roll24 MDD | Reference Avg Hold | v5 Roll24 Avg Hold | Reference Max Hold | v5 Roll24 Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 237.4884% | 249.1403% | -19.9815% | -19.9363% | 20.2917h | 20.2917h | 24.0000h | 24.0000h |
| oos | 141.2725% | 142.1316% | -18.5806% | -18.6719% | 20.1303h | 20.1303h | 24.0000h | 24.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Contract Checks

- Validation accounting error max abs: `2.740863092043355e-16`
- OOS accounting error max abs: `3.0531133177191805e-16`
- Validation notional contract error max abs: `8.881784197001252e-16`
- OOS notional contract error max abs: `4.440892098500626e-16`
- Validation trades: `64`
- OOS trades: `39`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_daytrade_overlay_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_daytrade_overlay_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_daytrade_overlay_20260701/validation_v5_roll24_daytrade_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_daytrade_overlay_20260701/oos_v5_roll24_daytrade_ledger.csv`
