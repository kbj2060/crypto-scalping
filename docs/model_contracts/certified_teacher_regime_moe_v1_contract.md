# Certified Teacher Regime MoE V1

- Model ID: `certified_teacher_regime_moe_v1`
- 2025 is the training/selection/holdout year. 2026 is fixed OOS.
- Inputs: certified AI outputs, M7 outputs, clean_regime_2024_unsup_v4_* features, causal market/microstructure features.
- Forbidden: legacy regime-v2/HDB/HMM, raw future/target/label/accounting columns, uncertified regime columns.
- Audit status: `pass`
- Blocking: `[]`

## Splits
- Fit: `2025-01-01 00:00:00` to `2025-08-31 23:55:00`
- Selection: `2025-09-01 00:00:00` to `2025-10-31 23:55:00`
- Holdout: `2025-11-01 00:00:00` to `2025-12-31 20:50:00`
- OOS: `2026-01-01 00:00:00` to `2026-02-28 16:00:00`

## Cost1 OOS
- PnL: `-11.5201964920859`
- MDD: `-13.486809273462551`
- Trades/day: `3.7329545454545454`

## Selected Feature Families
- Selected features: `112`
- Clean regime features: `23`
- AI features: `19`
- M7 features: `19`
