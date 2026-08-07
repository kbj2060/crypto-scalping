# Omega 4.6.2 Loss-Cluster Governor v5 Fine Exposure - 2026-07-01

## Method

This sweep freezes the v4 stop design and performs a narrow validation-only search around the high-PnL exposure boundary. It keeps the loss-window governor path-causal and only changes long/short exposure factors, notional cap, and the first-loss scale.

## Result

- Status: `VALIDATION_UPGRADE_IMPROVES_REFERENCE_PNL_WITH_HOLD_NOT_WORSE`
- Reference model: `omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701`
- Selection scope: `validation_only; OOS readout only`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `261.7270` | `274.8817` | `137.1999` | `138.4476` |
| MDD % | `-19.9829` | `-19.9378` | `-14.5938` | `-14.5217` |
| Avg hold h | `56.6123` | `56.6123` | `60.5577` | `60.5577` |
| Max hold h | `90.0000` | `90.0000` | `90.0000` | `90.0000` |

## Selected Candidate

- Stop spec: `hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5`
- Exposure spec: `long1300_short1955_cap4106`
- Governor spec: `loss1_500_win12`
- Validation upgrade gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701/loss_cluster_governor_v5_fine_exposure_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701/loss_cluster_governor_v5_fine_exposure_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701/validation_hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5__long1300_short1955_cap4106__loss1_500_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701/oos_hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5__long1300_short1955_cap4106__loss1_500_win12_ledger.csv`
