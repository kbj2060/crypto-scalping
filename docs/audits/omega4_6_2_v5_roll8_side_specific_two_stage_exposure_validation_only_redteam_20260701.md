# Omega 4.6.2 v5 Roll8 Two-Stage Exposure Validation-Only Red-Team - 2026-07-01

- Model: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701`
- Verdict: `FULL_LIVE_PASS_VALIDATION_ONLY`
- Research pass: `True`
- Full live pass: `True`
- OOS used in selection: `False`

## Selected Exposure

- Exposure spec: `lf0.900_sf1.050_cap4.40`
- Long/short factor: `0.9` / `1.05`
- Cap notional: `4.4`

| Split | PnL | MDD | Avg Hold | Max Hold | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| `validation` | `675.3209%` | `-17.3157%` | `5.8723h` | `8.0000h` | `186` |
| `oos` | `212.6850%` | `-19.4083%` | `6.6409h` | `8.0000h` | `97` |

## Failed Checks

- None.

## Artifacts

- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/redteam_audit_20260701.json`
- Candidate report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/report.json`
- Validation ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/validation_lf0p900_sf1p050_cap4p40_ledger.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/oos_lf0p900_sf1p050_cap4p40_ledger.csv`
