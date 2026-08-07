# AI Feature Provenance Audit 2026

Status: `conditional_pass_with_provenance_gap`

## Scope

- PatchTST, TiDE, DLinear specialist artifacts
- 2025/2026 AI feature checkpoints used by the clean-base family
- Batch inference chronology in `ensemble/ensemble_router.py`

## Findings

1. The 2025 and 2026 AI feature checkpoints are timestamp-separated:
   - `data/tmp/unified_build_ckpt/03_after_ai.csv`: 2025-01-01 00:00:00 through 2025-12-31 23:55:00
   - `data/tmp/unified_build_ckpt_2026/03_after_ai.csv`: 2026-01-01 00:00:00 through 2026-02-28 16:00:00

2. `ensemble/ensemble_router.py` avoids warmup `bfill` during batch inference and uses forward-fill only, then fills remaining warmup NaNs with 0.0. This passes the local look-ahead check for inference-time feature generation.

3. Existing specialist contracts for `data/nf_patchtst`, `data/nf_tide`, and `data/nf_dlinear` do not record the training data path or timestamp range. The source default points to `data/splits/year_oos/training_features_2024.csv` with `expected_year=2024`, but the artifact itself cannot prove that after the fact.

4. `ensemble/supervised/train_alternative_models.py` has been patched so future specialist contracts record:
   - source data path
   - expected year
   - actual timestamp range
   - actual years
   - row count after limit
   - `artifact_training_provenance_certified`

## Red-Team Conclusion

The strict no-leak walk-forward backtest can be read as accounting-valid, but the AI feature stack is not fully provenance-certified until the NeuralForecast specialists are force-retrained with the patched contract writer and the 2025/2026 AI feature checkpoints are rebuilt from those certified artifacts.

Required next step: force-retrain PatchTST/TiDE/DLinear/TimesNet specialists on the frozen 2024-only dataset, rebuild `03_after_ai.csv` for 2025 and 2026, then rerun `safe_cap_strict_noleak_walkforward`.
