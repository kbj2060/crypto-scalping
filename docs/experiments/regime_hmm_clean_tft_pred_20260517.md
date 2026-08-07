# Regime HMM Clean + TFT Predictor Experiment

Date: 2026-05-17

## Summary

Rebuilt the regime feature layer as requested:

- Previous clean regime sidecar was replaced with a 2024-fitted HMM-style current regime classifier.
- Previous HMM future regime work was replaced with a TFT-style sequence predictor.
- Both layers keep the 5-class taxonomy: `bull`, `bear`, `chop`, `whipsaw`, `normal`.
- `risk_off` and `transition` were removed from outputs.

## HMM Clean Regime

Command:

```bash
venv/bin/python scripts/retrain_clean_regime_hmm_20260517.py
```

Validation against current RegimeEngine labels:

```text
rows              26496
accuracy          0.4122
balanced_accuracy 0.3454
log_loss          1.4707
```

2025 output:

```text
path /home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime_hmm_v6_20260517/training_features_2025_clean_regime_hmm_v6.csv
rows 105064
columns 20 including timestamp
NaN count 0
probability sum min/max 1.0 / 1.0
```

The HMM probabilities are intentionally soft current-state context. They should not be treated as hard labels for expert selection.

## TFT Future Regime Predictor

Command:

```bash
venv/bin/python scripts/build_regime_pred_moe_tft_20260517.py
```

Architecture:

- Feature gate / variable selection style input gating
- Transformer encoder attention over a 72-bar sequence
- Multi-scale pooling over last 12, 36, and 72 bars
- Known future time covariates for target horizon
- 36-bar future-path regime label

Validation:

```text
rows              26459
accuracy          0.3078
balanced_accuracy 0.3249
log_loss          1.5309
selected_epochs   1
```

2025 output:

```text
path /home/llewyn/crypto-scalping/data/ensemble/supervised/regime_pred_moe_tft_20260517/training_features_2025_regime_pred_tft_moe.csv
rows 105064
columns 14 including timestamp
NaN count 0
probability sum min/max 1.0 / 1.0
```

2025 predicted probability means:

```text
bull    0.2003
bear    0.1865
chop    0.1456
whipsaw 0.2534
normal  0.2142
```

## Guardrails

- No hard argmax labels are written to sidecars.
- No `risk_off` or `transition` output columns are written.
- No model input feature column contains forbidden audit tokens outside the clean regime prefix.
- Validation thresholds are fit on pre-validation 2024 rows only.
- Validation split uses a 36-bar embargo before Q4 validation.
- Final model uses the validation-selected epoch count to avoid replaying overfit epoch depth on full 2024.

## Next Backtest Step

Join these sidecars by `timestamp` into the 2025 trading model dataset:

1. `/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime_hmm_v6_20260517/training_features_2025_clean_regime_hmm_v6.csv`
2. `/home/llewyn/crypto-scalping/data/ensemble/supervised/regime_pred_moe_tft_20260517/training_features_2025_regime_pred_tft_moe.csv`

Use `regime_pred_*` as the primary MoE router weights and `clean_regime_2024_unsup_v4_*` as current-state context features.
