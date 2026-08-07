# Omega 4.6.2 Loss-Cluster Governor - 2026-07-01

## Method

This sweep tests a path-causal risk governor: only prior closed trade losses and current realized drawdown can reduce the next trade's notional. It then tries higher base exposure to recover PnL while compressing max hold to 90h.

## Result

- Status: `VALIDATION_UPGRADE_IMPROVES_REFERENCE_PNL_AND_HOLD`
- Selection scope: `validation_only; OOS readout only`
- Reference model: `omega4_6_2_paper_optstop_exit_sizing_overlay_20260701`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `231.0344` | `247.2782` | `105.9861` | `128.6403` |
| MDD % | `-19.9436` | `-19.8771` | `-14.8066` | `-14.5838` |
| Avg hold h | `58.0870` | `56.8152` | `62.2500` | `60.5577` |
| Max hold h | `96.0000` | `90.0000` | `96.0000` | `90.0000` |

## Selected Candidate

- Stop spec: `hard90__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0`
- Exposure spec: `short178_long100_cap383`
- Governor spec: `loss1_65_win12`
- Validation upgrade gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_20260701/loss_cluster_governor_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_20260701/loss_cluster_governor_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_20260701/validation_hard90__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0__short178_long100_cap383__loss1_65_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_20260701/oos_hard90__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0__short178_long100_cap383__loss1_65_win12_ledger.csv`
