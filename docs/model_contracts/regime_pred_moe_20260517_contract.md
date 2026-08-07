# Regime Pred MoE 20260517 Contract

Status: `feature_layer_ready_for_ablation`

Last updated: 2026-05-17 KST

## Scope

- Model id: `regime_pred_moe_20260517`
- Architecture: LightGBM multiclass supervised predictor over future path-derived trading regimes.
- Purpose: produce soft 5-class state features for future MoE expert routing, not a standalone trading policy.
- Owner agents: Model Architect, Backtest Implementation Maintainer.
- Implementation script: `scripts/build_regime_pred_moe_20260517.py`
- Report artifact: `data/ensemble/reports/regime_pred_moe_20260517_report.json`
- Model artifact: `data/ensemble/supervised/regime_pred_moe_20260517/regime_pred_moe_2024.joblib`
- 2025 feature sidecar: `data/ensemble/supervised/regime_pred_moe_20260517/training_features_2025_regime_pred_moe.csv`

## Dataset Split

| Split | Source | Timestamp range | Rows | Use |
|---|---|---:|---:|---|
| Train | `data/splits/year_oos/training_features_2024.csv` + raw-only BGMM clean sidecar | 2024 before validation start | pre-validation rows only | Fit validation model and label thresholds |
| Validation | same 2024 source | 2024-10-01 onward | 26,459 | Architecture sanity check only |
| Full train | same 2024 source | 2024 | 105,343 labeled rows | Fit final predictor |
| Predict | `data/splits/year_oos/training_features_2025.csv` + raw-only BGMM clean sidecar | 2025 | 105,064 | Generate `regime_pred_*` features |

Audit:

- Timestamp overlap: train and predict years are separate.
- Warmup handling: final `horizon + 1` rows are excluded from label fitting where future path is unavailable.
- Leakage control: future OHLC is used only for label generation, never as inference input.
- Validation label thresholds: fitted only on pre-validation 2024 rows, then frozen for 2024 validation labels.
- Final model label thresholds: fitted on all 2024 rows only, then used to fit the final 2024-only model.
- Clean regime input: selected raw-only BGMM outputs only; no `risk_off`, `transition`, pseudo class probabilities, `state_code`, or integer `cluster`.

## Feature Contract

- Feature count: 77.
- Selected clean features:

```text
clean_regime_2024_unsup_v4_factor_trend
clean_regime_2024_unsup_v4_factor_flow
clean_regime_2024_unsup_v4_factor_vol
clean_regime_2024_unsup_v4_factor_crowding
clean_regime_2024_unsup_v4_factor_liquidity
clean_regime_2024_unsup_v4_trend_bias
clean_regime_2024_unsup_v4_cluster_confidence
clean_regime_2024_unsup_v4_cluster_entropy
clean_regime_2024_unsup_v4_cluster_prob_0
clean_regime_2024_unsup_v4_cluster_prob_1
clean_regime_2024_unsup_v4_cluster_prob_2
clean_regime_2024_unsup_v4_cluster_prob_3
clean_regime_2024_unsup_v4_cluster_prob_4
```

Forbidden as inputs:

```text
regime_bull/regime_bear/regime_chop/regime_whipsaw/regime_normal
clean_regime_2024_unsup_v4_risk_off_prob
clean_regime_2024_unsup_v4_transition_risk
clean_regime_2024_unsup_v4_bull_prob/bear_prob/chop_prob/whipsaw_prob/normal_prob
clean_regime_2024_unsup_v4_state_code
clean_regime_2024_unsup_v4_cluster
future/target/label/realized/trade_pnl/cash_after columns
```

## Label Contract

- Horizon: 36 bars.
- Classes:

```text
bull
bear
chop
whipsaw
normal
```

- `bull`: future long trend path quality dominates short quality.
- `bear`: future short trend path quality dominates long quality.
- `chop`: low range/low efficiency future path.
- `whipsaw`: high range, low net direction, adverse movement on both sides.
- `normal`: residual state.

Training label counts:

```text
bull: 22066
bear: 19506
chop: 38783
whipsaw: 4894
normal: 20094
```

## Output Contract

Required sidecar columns:

```text
regime_pred_bull_prob
regime_pred_bear_prob
regime_pred_chop_prob
regime_pred_whipsaw_prob
regime_pred_normal_prob
regime_pred_trend_prob
regime_pred_micro_prob
regime_pred_directional_bias
regime_pred_range_prob
regime_pred_instability_prob
regime_pred_confidence
regime_pred_entropy
regime_pred_margin
```

Hard argmax outputs are intentionally not written to the sidecar. Downstream models must use soft probabilities and derived numeric columns only.

## Current Quality

2024 validation:

```text
accuracy: 0.316527
balanced_accuracy: 0.328258
log_loss: 1.488740
```

2025 prediction:

```text
bull: 5898
bear: 47782
chop: 23698
whipsaw: 14436
normal: 13250
mean confidence: 0.389132
mean entropy: 0.873096
```

Interpretation: this is a soft state feature layer ready for controlled ablation, not a hard router. Validation labels use train-only thresholds; high entropy and moderate validation score mean MoE integration should start with soft weights and compare against the frozen baseline.

## Backtest Gates

- Merge the sidecar into the 2025 training frame by timestamp only.
- Run baseline versus `+ regime_pred_*` with identical downstream model/search budget.
- Do not use 2026 to select thresholds, architecture, or feature subsets.
- Do not create or use hard route labels from `regime_pred_*` until a backtest proves lift and calibration.
- Promotion requires runtime-native backtest parity per `docs/model_contracts/runtime_native_training_backtest_policy_20260515.md`.
