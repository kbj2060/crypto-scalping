# Regime Pred MoE 20260517

Status: `generated_for_backtest_ablation`

## Summary

Built the first supervised 5-class trading-regime predictor for MoE feature routing.
The predictor is trained on 2024 only and writes 2025 `regime_pred_*` sidecar features.

Artifacts:

- Script: `scripts/build_regime_pred_moe_20260517.py`
- Model: `data/ensemble/supervised/regime_pred_moe_20260517/regime_pred_moe_2024.joblib`
- 2024 sidecar: `data/ensemble/supervised/regime_pred_moe_20260517/training_features_2024_regime_pred_moe.csv`
- 2025 sidecar: `data/ensemble/supervised/regime_pred_moe_20260517/training_features_2025_regime_pred_moe.csv`
- Report: `data/ensemble/reports/regime_pred_moe_20260517_report.json`
- Contract: `docs/model_contracts/regime_pred_moe_20260517_contract.md`

## Design

Classes are fixed to:

```text
bull
bear
chop
whipsaw
normal
```

`risk_off` and `transition` are intentionally not labels or outputs.

The model uses raw causal features plus selected clean BGMM state priors. It excludes
legacy regime one-hots, clean pseudo class probabilities, `risk_off`, `transition`,
integer `cluster`, future labels, and realized outcome columns.

## Validation Snapshot

2024 validation:

| Metric | Value |
|---|---:|
| Accuracy | `0.316527` |
| Balanced accuracy | `0.328258` |
| Log loss | `1.488740` |

2025 predicted labels:

| Label | Rows |
|---|---:|
| bull | `5898` |
| bear | `47782` |
| chop | `23698` |
| whipsaw | `14436` |
| normal | `13250` |

Mean 2025 confidence is `0.389132`; mean entropy is `0.873096`.

## Backtest Maintainer Note

Treat this as a candidate feature layer, not a promotion candidate. The first
backtest should be a strict feature ablation:

```text
baseline: existing 2025 training frame
candidate: same frame + numeric regime_pred_* columns
```

The sidecar intentionally excludes hard argmax label columns. Use probabilities
and derived numeric features only. Hard MoE routing is blocked until
soft-probability routing proves lift under the frozen baseline and runtime native
backtest contract.
