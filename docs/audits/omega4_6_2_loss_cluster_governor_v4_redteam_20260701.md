# Omega 4.6.2 Loss-Cluster Governor v4 Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701`
- Reference: `omega4_6_2_loss_cluster_governor_v3_20260701`
- Verdict: `RESEARCH_UPGRADE_PASS_FULL_LIVE_BLOCKED`
- Research upgrade pass: `True`
- Full live pass: `False`

## Selected Candidate

- Stop spec: `hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5`
- Exposure spec: `long120_short190_cap408`
- Governor spec: `loss1_55_win12`

| Split | Reference PnL | Candidate PnL | Reference MDD | Candidate MDD | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 254.0296% | 261.7270% | -19.5829% | -19.9829% | 56.6123h | 56.6123h | 90.0000h | 90.0000h |
| oos | 133.3448% | 137.1999% | -14.2976% | -14.5938% | 60.5577h | 60.5577h | 90.0000h | 90.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `max_hold_24h_daytrading_requirement`: {'validation': 90.0, 'oos': 90.0}

## Research Failures

- None.

## Warnings

- `validation_mdd_buffer_over_1pp`: {'validation_mdd': -19.982921929838483, 'buffer_to_20pct': 0.01707807016151719}
- `validation_mdd_buffer_over_0p10pp`: {'validation_mdd': -19.982921929838483, 'buffer_to_20pct': 0.01707807016151719}

## Contract Checks

- Validation accounting error max abs: `2.914335439641036e-16`
- OOS accounting error max abs: `1.1102230246251565e-16`
- Validation notional contract error max abs: `8.881784197001252e-16`
- OOS notional contract error max abs: `2.220446049250313e-16`
- Validation max leverage: `5.0`
- OOS max leverage: `5.0`
- Validation max notional: `3.6434443987512894`
- OOS max notional: `4.08`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701/loss_cluster_governor_v4_fine_exposure_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701/validation_hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5__long120_short190_cap408__loss1_55_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701/oos_hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5__long120_short190_cap408__loss1_55_win12_ledger.csv`
