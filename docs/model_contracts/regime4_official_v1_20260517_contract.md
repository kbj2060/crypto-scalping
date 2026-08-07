# Regime4 Official V1 Contract

Date: 2026-05-17
Amended: 2026-05-21

## Decision

The official regime experiment line is now 4-class:

- `bull`
- `bear`
- `chop`
- `whipsaw`

`normal` is removed as an independent regime class. Empirically, the 5-class HMM almost never emitted `normal` as a hard state in 2025, and future labels built from the 5-class clean regime contained only 18 `normal` rows in 2024. It should be treated as a residual low-confidence condition, not a separate MoE expert.

## Current Regime Classifier

Current regime sidecar is the 4-class HMM raw-state12 model.

Script:

- `/home/llewyn/crypto-scalping/scripts/retrain_clean_regime4_hmm_raw_state12_20260517.py`

Artifacts:

- `/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/clean_regime4_raw_state12_v1_2024.joblib`
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2024_clean_regime4_raw_state12_v1.csv`
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2025_clean_regime4_raw_state12_v1.csv`
- `/home/llewyn/crypto-scalping/data/ensemble/reports/clean_regime4_raw_state12_v1_20260517_report.json`

Output prefix:

```text
clean_regime4_2024_unsup_v1_
```

Output columns:

```text
clean_regime4_2024_unsup_v1_bull_prob
clean_regime4_2024_unsup_v1_bear_prob
clean_regime4_2024_unsup_v1_chop_prob
clean_regime4_2024_unsup_v1_whipsaw_prob
clean_regime4_2024_unsup_v1_trend_prob
clean_regime4_2024_unsup_v1_micro_prob
clean_regime4_2024_unsup_v1_directional_bias
clean_regime4_2024_unsup_v1_range_prob
clean_regime4_2024_unsup_v1_instability_prob
clean_regime4_2024_unsup_v1_confidence
clean_regime4_2024_unsup_v1_entropy
clean_regime4_2024_unsup_v1_margin
clean_regime4_2024_unsup_v1_factor_trend
clean_regime4_2024_unsup_v1_factor_flow
clean_regime4_2024_unsup_v1_factor_vol
clean_regime4_2024_unsup_v1_factor_crowding
clean_regime4_2024_unsup_v1_factor_liquidity
clean_regime4_2024_unsup_v1_trend_bias
clean_regime4_2024_unsup_v1_risk_off_prob
clean_regime4_2024_unsup_v1_transition_risk
```

Input policy:

- Raw-only state12 engineered inputs.
- No `clean_regime_*` input features.
- No `normal`, cluster id, hidden-state id, or hard label output.
- `risk_off_prob` and `transition_risk` are causal auxiliary scores, not regime classes.

## Future Regime Predictor

Canonical future regime sidecar is the 4-class TFT VSN-selected h12 no-mdjd all74 model.

Script:

- `/home/llewyn/crypto-scalping/scripts/build_regime4_pred_tft_vsn_select_20260517.py`

Official reproduction command:

```bash
venv/bin/python scripts/build_regime4_pred_tft_vsn_select_20260517.py \
  --horizon 12 \
  --out-dir data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517 \
  --report data/ensemble/reports/regime4_pred_tft_h12_nomdjd_all74_20260517_report.json
```

Artifacts:

- `/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/regime4_pred_tft_vsn_selected_2024.pt`
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2024_regime4_pred_tft_vsn_selected.csv`
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv`
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2026_rebuilt_regime4_pred_tft_vsn_selected.csv`
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/regime4_pred_tft_vsn_importance.csv`
- `/home/llewyn/crypto-scalping/data/ensemble/reports/regime4_pred_tft_h12_nomdjd_all74_20260517_report.json`
- `/home/llewyn/crypto-scalping/data/ensemble/reports/regime4_transform_2026_h12_nomdjd_all74_20260517.json`

Output prefix:

```text
regime4_pred_
```

Output columns:

```text
regime4_pred_bull_prob
regime4_pred_bear_prob
regime4_pred_chop_prob
regime4_pred_whipsaw_prob
regime4_pred_trend_prob
regime4_pred_micro_prob
regime4_pred_directional_bias
regime4_pred_range_prob
regime4_pred_instability_prob
regime4_pred_confidence
regime4_pred_entropy
regime4_pred_margin
```

Training target:

```text
t + 12 bars clean_regime4 argmax
```

The h12 no-mdjd all74 line is official because it removes the stale `pred_mdjd`/`conf_mdjd` dependency while preserving the short-horizon MoE routing use case.

Deprecated future predictor:

```text
data/ensemble/supervised/regime4_pred_tft_vsn_h12_official_20260517/
```

The deprecated artifact depends on `pred_mdjd`/`conf_mdjd`. In 2026 rebuilt frames those fields were absent and runtime transform filled missing model inputs with artifact medians. That behavior is no longer acceptable for promoted/live sidecars.

Validation metrics for the h12 VSN-selected line are selection metrics because feature selection and model choice used the validation split. Treat them as candidate-selection evidence, not final OOS proof. Promotion requires frozen downstream backtest ablation.

Selection metrics for the canonical no-mdjd all74 line:

```text
selected: acc=0.6219226702914967, bal_acc=0.605649360143007, logloss=0.8988966259309439
all_features: acc=0.61939284096058, bal_acc=0.6072650350857556, logloss=0.9102113380899424
```

The multi-horizon 12/36/72 artifacts are auxiliary analysis outputs only. They are not the official future regime sidecar.

## Routing Guidance

Use soft routing, not hard argmax:

```text
trend_weight = regime4_pred_bull_prob + regime4_pred_bear_prob
micro_weight = regime4_pred_chop_prob + regime4_pred_whipsaw_prob

score =
  regime4_pred_bull_prob    * bull_expert_score
+ regime4_pred_bear_prob    * bear_expert_score
+ regime4_pred_chop_prob    * chop_expert_score
+ regime4_pred_whipsaw_prob * whipsaw_expert_score
```

Recommended fallback:

```text
if regime4_pred_confidence is low or regime4_pred_entropy is high:
    blend toward parent/global model rather than forcing one expert
```

## Compatibility

This is an official experiment line, not a destructive schema migration. Existing 5-class files remain available for comparison, but new MoE work should prefer:

```text
regime4_pred_*
```

The raw-state12 current sidecar above still exports `clean_regime4_2024_unsup_v1_*` for historical compatibility. Active DSAC/Router feature specs must not use that ambiguous export prefix. For downstream DSAC feature inventories, use the state24 sticky artifact and replace its original export prefix with the explicit downstream provenance prefix:

```text
clean_regime4_state24_sticky090_v2_*
```

Active DSAC specs must not contain `clean_regime4_2024_unsup_v1_*`. That prefix is allowed only for historical reproduction or for reading original artifact CSVs before the DSAC rename step. The fixed DSAC inventory/spec directory currently validates with zero legacy-prefix columns and state24-prefix columns present:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521/
```
