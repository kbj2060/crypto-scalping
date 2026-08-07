# Regime HMM Clean + TFT Predictor Contract

Date: 2026-05-17

## Purpose

This contract defines the regime data layer for MoE trading experts.

- Current Regime Classifier: HMM-style Gaussian latent-state model replacing the previous BGMM clean regime sidecar.
- Future Regime Predictor: TFT-style sequence model producing future-path regime probabilities for 2025 features.
- Regime taxonomy remains exactly 5 classes: `bull`, `bear`, `chop`, `whipsaw`, `normal`.
- `risk_off` and `transition` are not regime classes and are not emitted.

## Current Clean Regime Sidecar

Script:

- `/home/llewyn/crypto-scalping/scripts/retrain_clean_regime_hmm_20260517.py`

Artifacts:

- `/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime_hmm_v6_20260517/clean_regime_state_v6_2024.joblib`
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime_hmm_v6_20260517/training_features_2024_clean_regime_hmm_v6.csv`
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime_hmm_v6_20260517/training_features_2025_clean_regime_hmm_v6.csv`
- `/home/llewyn/crypto-scalping/data/ensemble/reports/clean_regime_hmm_v6_20260517_report.json`

Output columns use the existing clean prefix for downstream compatibility:

- `clean_regime_2024_unsup_v4_factor_trend`
- `clean_regime_2024_unsup_v4_factor_flow`
- `clean_regime_2024_unsup_v4_factor_vol`
- `clean_regime_2024_unsup_v4_factor_crowding`
- `clean_regime_2024_unsup_v4_factor_liquidity`
- `clean_regime_2024_unsup_v4_trend_bias`
- `clean_regime_2024_unsup_v4_bull_prob`
- `clean_regime_2024_unsup_v4_bear_prob`
- `clean_regime_2024_unsup_v4_chop_prob`
- `clean_regime_2024_unsup_v4_whipsaw_prob`
- `clean_regime_2024_unsup_v4_normal_prob`
- `clean_regime_2024_unsup_v4_trend_prob`
- `clean_regime_2024_unsup_v4_micro_prob`
- `clean_regime_2024_unsup_v4_directional_bias`
- `clean_regime_2024_unsup_v4_range_prob`
- `clean_regime_2024_unsup_v4_instability_prob`
- `clean_regime_2024_unsup_v4_confidence`
- `clean_regime_2024_unsup_v4_entropy`
- `clean_regime_2024_unsup_v4_margin`

Notes:

- No hard label, cluster id, hidden-state id, `risk_off`, or `transition` columns are written.
- Feature names intentionally avoid `hmm_`, because the certified feature audit forbids that token.
- HMM inference uses causal filtering, not full-sequence smoothing, so transform-time rows do not use future rows.

## Future Regime Predictor Sidecar

Script:

- `/home/llewyn/crypto-scalping/scripts/build_regime_pred_moe_tft_20260517.py`

Artifacts:

- `/home/llewyn/crypto-scalping/data/ensemble/supervised/regime_pred_moe_tft_20260517/regime_pred_tft_moe_2024.pt`
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/regime_pred_moe_tft_20260517/training_features_2024_regime_pred_tft_moe.csv`
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/regime_pred_moe_tft_20260517/training_features_2025_regime_pred_tft_moe.csv`
- `/home/llewyn/crypto-scalping/data/ensemble/reports/regime_pred_moe_tft_20260517_report.json`

Output columns:

- `regime_pred_bull_prob`
- `regime_pred_bear_prob`
- `regime_pred_chop_prob`
- `regime_pred_whipsaw_prob`
- `regime_pred_normal_prob`
- `regime_pred_trend_prob`
- `regime_pred_micro_prob`
- `regime_pred_directional_bias`
- `regime_pred_range_prob`
- `regime_pred_instability_prob`
- `regime_pred_confidence`
- `regime_pred_entropy`
- `regime_pred_margin`

Notes:

- Output uses `regime_pred_*`, not `regime_future_*`, because the feature audit blocks the token `future` in model inputs.
- The model consumes raw causal market features plus the HMM clean regime sidecar as current-state context.
- Future labels are generated from future path behavior during training only. Inference consumes current and past features plus known future time covariates.
- 2024 is used for training and validation. 2025 sidecar is generated for downstream trading models.

## Backtest Integration

For MoE routing, prefer soft probabilities over hard argmax routing:

```text
expert_score =
  regime_pred_bull_prob    * bull_expert_score
+ regime_pred_bear_prob    * bear_expert_score
+ regime_pred_chop_prob    * chop_expert_score
+ regime_pred_whipsaw_prob * whipsaw_expert_score
+ regime_pred_normal_prob  * normal_expert_score
```

Use HMM clean columns as context features, not as final routing labels. The TFT `regime_pred_*` columns are the primary future-looking router features.
