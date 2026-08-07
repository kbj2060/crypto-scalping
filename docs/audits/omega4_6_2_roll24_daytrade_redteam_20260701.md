# Omega 4.6.2 Roll24 Daytrade Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_roll24_daytrade_overlay_20260701`
- Reference: `omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701`
- Verdict: `DAYTRADE_RESEARCH_PASS_FULL_LIVE_BLOCKED`
- Daytrade research pass: `True`
- PnL upgrade vs reference: `False`
- Full live pass: `False`

| Split | Reference PnL | Roll24 PnL | Reference MDD | Roll24 MDD | Reference Avg Hold | Roll24 Avg Hold | Reference Max Hold | Roll24 Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 261.7270% | 237.4884% | -19.9829% | -19.9815% | 56.6123h | 20.2917h | 90.0000h | 24.0000h |
| oos | 137.1999% | 141.2725% | -14.5938% | -18.5806% | 60.5577h | 20.1303h | 90.0000h | 24.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}

## Research Failures

- None.

## Contract Checks

- Validation accounting error max abs: `3.157196726277789e-16`
- OOS accounting error max abs: `3.3306690738754696e-16`
- Validation notional contract error max abs: `8.881784197001252e-16`
- OOS notional contract error max abs: `2.220446049250313e-16`
- Validation trades: `64`
- OOS trades: `39`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_daytrade_overlay_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_daytrade_overlay_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_daytrade_overlay_20260701/validation_roll24_daytrade_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_daytrade_overlay_20260701/oos_roll24_daytrade_ledger.csv`
