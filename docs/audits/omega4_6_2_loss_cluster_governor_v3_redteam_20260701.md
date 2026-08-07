# Omega 4.6.2 Loss-Cluster Governor v3 Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_loss_cluster_governor_v3_20260701`
- Reference: `omega4_6_2_loss_cluster_governor_20260701`
- Verdict: `RESEARCH_UPGRADE_PASS_FULL_LIVE_BLOCKED`
- Research upgrade pass: `True`
- Full live pass: `False`

## Selected Candidate

- Stop spec: `hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5`
- Exposure spec: `long120_short186_cap400`
- Governor spec: `loss1_55_win12`

| Split | Reference PnL | Candidate PnL | Reference MDD | Candidate MDD | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 247.2782% | 254.0296% | -19.8771% | -19.5829% | 56.8152h | 56.6123h | 90.0000h | 90.0000h |
| oos | 128.6403% | 133.3448% | -14.5838% | -14.2976% | 60.5577h | 60.5577h | 90.0000h | 90.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `max_hold_24h_daytrading_requirement`: {'validation': 90.0, 'oos': 90.0}

## Research Failures

- None.

## Warnings

- `validation_mdd_buffer_over_1pp`: {'validation_mdd': -19.582925176992084, 'buffer_to_20pct': 0.41707482300791554}

## Contract Checks

- Validation accounting error max abs: `2.498001805406602e-16`
- OOS accounting error max abs: `1.6653345369377348e-16`
- Validation notional contract error max abs: `8.881784197001252e-16`
- OOS notional contract error max abs: `8.881784197001252e-16`
- Validation max leverage: `5.0`
- OOS max leverage: `5.0`
- Validation max notional: `3.5667403061460003`
- OOS max notional: `4.0`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v3_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v3_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v3_20260701/loss_cluster_governor_v3_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v3_20260701/validation_hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5__long120_short186_cap400__loss1_55_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v3_20260701/oos_hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5__long120_short186_cap400__loss1_55_win12_ledger.csv`
