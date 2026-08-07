# Omega 4.6.2 Loss-Cluster Governor v4 Fine Exposure - 2026-07-01

## Method

This sweep freezes the v3 exit/governor design and only fine-tunes the short exposure factor between 1.86 and 1.91. Selection is validation-only; OOS is readout.

## Result

- Status: `VALIDATION_UPGRADE_IMPROVES_REFERENCE_PNL_WITH_HOLD_NOT_WORSE`
- Reference model: `omega4_6_2_loss_cluster_governor_v3_20260701`
- Selection scope: `validation_only; OOS readout only`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `254.0296` | `261.7270` | `133.3448` | `137.1999` |
| MDD % | `-19.5829` | `-19.9829` | `-14.2976` | `-14.5938` |
| Avg hold h | `56.6123` | `56.6123` | `60.5577` | `60.5577` |
| Max hold h | `90.0000` | `90.0000` | `90.0000` | `90.0000` |

## Selected Candidate

- Stop spec: `hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5`
- Exposure spec: `long120_short190_cap408`
- Governor spec: `loss1_55_win12`
- Validation upgrade gate pass: `True`

## Artifacts

- Ranking: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701/loss_cluster_governor_v4_fine_exposure_ranking.csv`
- Top 20: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701/loss_cluster_governor_v4_fine_exposure_top20.csv`
- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701/validation_hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5__long120_short190_cap408__loss1_55_win12_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701/oos_hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5__long120_short190_cap408__loss1_55_win12_ledger.csv`
