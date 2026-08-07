# Omega 4.6.2 Roll24 Daytrade Overlay - 2026-07-01

## Method

This overlay freezes the v4 candidate and splits every active position into 24h-or-less roll segments. Each roll pays the same estimated roundtrip cost as the parent trade segment.

## Result

- Status: `DAYTRADE_HOLD_PASS_PNL_LOWER_THAN_REFERENCE`
- Reference model: `omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701`

| Metric | Reference Val | Roll24 Val | Reference OOS | Roll24 OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `261.7270` | `237.4884` | `137.1999` | `141.2725` |
| MDD % | `-19.9829` | `-19.9815` | `-14.5938` | `-18.5806` |
| Trades | `23` | `64` | `13` | `39` |
| Avg hold h | `56.6123` | `20.2917` | `60.5577` | `20.1303` |
| Max hold h | `90.0000` | `24.0000` | `90.0000` | `24.0000` |

## Artifacts

- Report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_daytrade_overlay_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_daytrade_overlay_20260701/validation_roll24_daytrade_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_roll24_daytrade_overlay_20260701/oos_roll24_daytrade_ledger.csv`
