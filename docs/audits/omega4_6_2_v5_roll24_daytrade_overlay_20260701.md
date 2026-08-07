# Omega 4.6.2 v5 Roll24 Daytrade Overlay - 2026-07-01

## Method

This overlay applies the same 24h roll segmentation used by the prior roll24 branch to the v5 loss-cluster parent ledger. The roll transformation is fixed; OOS is readout only.

## Result

- Status: `DAYTRADE_HOLD_AND_PNL_PASS`
- Parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Reference daytrade model: `omega4_6_2_roll24_daytrade_overlay_20260701`

| Metric | Reference Roll24 Val | v5 Roll24 Val | Reference Roll24 OOS | v5 Roll24 OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `237.4884` | `249.1403` | `141.2725` | `142.1316` |
| MDD % | `-19.9815` | `-19.9363` | `-18.5806` | `-18.6719` |
| Trades | `64` | `64` | `39` | `39` |
| Avg hold h | `20.2917` | `20.2917` | `20.1303` | `20.1303` |
| Max hold h | `24.0000` | `24.0000` | `24.0000` | `24.0000` |

## Parent Context

- Parent validation PnL: `274.8817%`
- Parent OOS PnL: `138.4476%`
- Parent max hold: `90.0000h` / `90.0000h`

## Artifacts

- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_daytrade_overlay_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_daytrade_overlay_20260701/validation_v5_roll24_daytrade_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll24_daytrade_overlay_20260701/oos_v5_roll24_daytrade_ledger.csv`
