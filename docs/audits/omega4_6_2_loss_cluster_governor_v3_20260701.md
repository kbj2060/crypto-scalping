# Omega 4.6.2 Loss-Cluster Governor v3 - 2026-07-01

## Method

This sweep keeps max hold at or below v1 but moves trail/stall exits earlier to reduce average hold. Selection remains validation-only; OOS is readout.

## Result

- Status: `VALIDATION_UPGRADE_IMPROVES_REFERENCE_PNL_AND_AVG_HOLD`
- Reference model: `omega4_6_2_loss_cluster_governor_20260701`
- Selection scope: `validation_only; OOS readout only`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `247.2782` | `254.0296` | `128.6403` | `133.3448` |
| MDD % | `-19.8771` | `-19.5829` | `-14.5838` | `-14.2976` |
| Avg hold h | `56.8152` | `56.6123` | `60.5577` | `60.5577` |
| Max hold h | `90.0000` | `90.0000` | `90.0000` | `90.0000` |

## Selected Candidate

- Stop spec: `hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5`
- Exposure spec: `long120_short186_cap400`
- Governor spec: `loss1_55_win12`
- Validation upgrade gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v3_20260701/loss_cluster_governor_v3_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v3_20260701/loss_cluster_governor_v3_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v3_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v3_20260701/validation_hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5__long120_short186_cap400__loss1_55_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v3_20260701/oos_hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5__long120_short186_cap400__loss1_55_win12_ledger.csv`
