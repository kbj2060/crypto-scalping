# Omega 4.6.2 Loss-Cluster Governor Red-Team Audit - 2026-07-01

- Model: `omega4_6_2_loss_cluster_governor_20260701`
- Reference: `omega4_6_2_paper_optstop_exit_sizing_overlay_20260701`
- Verdict: `RESEARCH_UPGRADE_PASS_FULL_LIVE_BLOCKED`
- Research upgrade pass: `True`
- Full live pass: `False`

## Selected Candidate

- Stop spec: `hard90__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0`
- Exposure spec: `short178_long100_cap383`
- Governor spec: `loss1_65_win12`

| Split | Reference PnL | Candidate PnL | Reference MDD | Candidate MDD | Reference Avg Hold | Candidate Avg Hold | Reference Max Hold | Candidate Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 231.0344% | 247.2782% | -19.9436% | -19.8771% | 58.0870h | 56.8152h | 96.0000h | 90.0000h |
| oos | 105.9861% | 128.6403% | -14.8066% | -14.5838% | 62.2500h | 60.5577h | 96.0000h | 90.0000h |

## Blocking Items

- `runtime_native_replay_complete`: {'inherited_status': 'FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `fresh_holdout_walkforward_complete`: {'fresh_holdout_available': False, 'reason': 'Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.', 'prior_audit': '/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_cap220_runtime_native_walkforward_20260701.json'}
- `max_hold_24h_daytrading_requirement`: {'validation': 90.0, 'oos': 90.0}

## Research Failures

- None.

## Warnings

- `validation_mdd_buffer_over_1pp`: {'validation_mdd': -19.877112240764696, 'buffer_to_20pct': 0.12288775923530437}

## Contract Checks

- Validation accounting error max abs: `2.220446049250313e-16`
- OOS accounting error max abs: `1.3877787807814457e-16`
- Validation notional contract error max abs: `4.440892098500626e-16`
- OOS notional contract error max abs: `4.440892098500626e-16`
- Validation max leverage: `5.0`
- OOS max leverage: `5.0`
- Validation max notional: `3.4133321209354195`
- OOS max notional: `3.83`

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_20260701/report.json`
- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_20260701/loss_cluster_governor_ranking.csv`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_20260701/validation_hard90__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0__short178_long100_cap383__loss1_65_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_20260701/oos_hard90__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0__short178_long100_cap383__loss1_65_win12_ledger.csv`
