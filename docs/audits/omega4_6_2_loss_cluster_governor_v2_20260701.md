# Omega 4.6.2 Loss-Cluster Governor v2 - 2026-07-01

## Method

This sweep uses the v1 loss-window governor idea but tries shorter hard stops and wider capped exposure under leverage cap 5.

## Result

- Status: `NO_VALIDATION_UPGRADE_IMPROVED_REFERENCE_PNL_AND_HOLD`
- Reference model: `omega4_6_2_loss_cluster_governor_20260701`
- Selection scope: `validation_only; OOS readout only`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `247.2782` | `284.8295` | `128.6403` | `142.3634` |
| MDD % | `-19.8771` | `-20.4817` | `-14.5838` | `-14.9634` |
| Avg hold h | `56.8152` | `56.8152` | `60.5577` | `60.5577` |
| Max hold h | `90.0000` | `90.0000` | `90.0000` | `90.0000` |

## Selected Candidate

- Stop spec: `hard90__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0`
- Exposure spec: `long120_short195_cap419`
- Governor spec: `loss1_55_win12`
- Validation upgrade gate pass: `False`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v2_20260701/loss_cluster_governor_v2_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v2_20260701/loss_cluster_governor_v2_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v2_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v2_20260701/validation_hard90__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0__long120_short195_cap419__loss1_55_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v2_20260701/oos_hard90__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0__long120_short195_cap419__loss1_55_win12_ledger.csv`
