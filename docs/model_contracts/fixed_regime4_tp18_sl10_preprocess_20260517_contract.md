# Fixed Regime4 + TP 1.8 / SL 1.0 Preprocessing Contract

Date: 2026-05-17
Amended: 2026-05-21

## Status

Frozen preprocessing input.

This contract fixes two feature groups as standard preprocessing outputs:

```text
Regime4 current/future regime features
tp_sl_action_score with TP 1.8% / SL 1.0%
```

This does not promote a downstream model. It only fixes the data preprocessing surface.

## Canonical Outputs

Train:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_tp18_sl10_fixed.csv
```

Eval:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_tp18_sl10_fixed.csv
```

Manifest:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/fixed_regime4_tp18_sl10_preprocess_manifest.json
```

## Builders

Regime4 2026 sidecar transform:

```bash
venv/bin/python scripts/transform_regime4_official_sidecars_20260517.py
```

Fixed preprocessing build:

```bash
venv/bin/python scripts/build_fixed_regime4_tp_sl_preprocess_20260517.py --strict-eval-regime
```

The builder fails if timestamp joins are incomplete or required 2026 Regime4 sidecars are missing.

## DSAC Rename Policy

The original state24 Regime4 sidecar files emit the legacy prefix:

```text
clean_regime4_2024_unsup_v1_*
```

For DSAC feature inventory and future Router/DSAC experiments, this prefix is deprecated because multiple Regime4 artifacts reused it. Active DSAC inputs must drop `clean_regime4_2024_unsup_v1_*` and use the explicit state24 provenance prefix instead:

```text
clean_regime4_state24_sticky090_v2_*
```

Historical Alpha5/Alpha5.1/Alpha5.2 reports may still show `clean_regime4_2024_unsup_v1_*`; new DSAC fixed inventory/specs must not.

## Regime4 Contract

Classes:

```text
bull
bear
chop
whipsaw
```

Disabled as regime classes:

```text
normal
risk_off
transition
```

Enabled as current-Regime4 auxiliary features:

```text
clean_regime4_state24_sticky090_v2_factor_trend
clean_regime4_state24_sticky090_v2_factor_flow
clean_regime4_state24_sticky090_v2_factor_vol
clean_regime4_state24_sticky090_v2_factor_crowding
clean_regime4_state24_sticky090_v2_factor_liquidity
clean_regime4_state24_sticky090_v2_trend_bias
clean_regime4_state24_sticky090_v2_risk_off_prob
clean_regime4_state24_sticky090_v2_transition_risk
```

These are not HMM classes. The HMM still outputs only 4-class current regime
probabilities. The factor/risk/transition values are causal current-row scores
computed from raw market, flow, volatility, liquidity, funding, and AI/M7
features, using the Alpha4.3-compatible factor formulas under the Regime4
prefix.

Active DSAC current regime prefix:

```text
clean_regime4_state24_sticky090_v2_
```

The original state24 sidecar CSVs still emit `clean_regime4_2024_unsup_v1_*`. That legacy export prefix is renamed before active DSAC/Router feature spec generation.

Future regime prefix:

```text
regime4_pred_
```

Sidecars:

```text
/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2025_clean_regime4_raw_state12_v1.csv
/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime4_state24_sticky090_v2_20260517/training_features_2025_clean_regime4_state24_sticky090_v2.csv
/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv
/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2026_rebuilt_clean_regime4_raw_state12_v1.csv
/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime4_state24_sticky090_v2_20260517/training_features_2026_rebuilt_clean_regime4_state24_sticky090_v2.csv
/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2026_rebuilt_regime4_pred_tft_vsn_selected.csv
```

The raw-state12 sidecars are retained for historical Regime4 predictor reproduction. Active DSAC/Router specs use the state24 sticky090 v2 sidecars renamed to `clean_regime4_state24_sticky090_v2_*`.

Future Regime4 model:

```text
regime4_pred_tft_h12_nomdjd_all74_20260517
```

`pred_mdjd` and `conf_mdjd` are excluded from the future Regime4 predictor. The no-mdjd all74 model is used because it avoids 2026 median fallback and improved validation versus the prior selected official h12 artifact:

```text
prior h12 selected accuracy 0.6009, log_loss 0.9228, pred_mdjd included
no-mdjd all74 accuracy       0.6219, log_loss 0.8989, pred_mdjd/conf_mdjd excluded
```

2026 transform report:

```text
/home/llewyn/crypto-scalping/data/ensemble/reports/regime4_transform_2026_h12_nomdjd_all74_20260517.json
```

Active DSAC fixed specs:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521/
```

Current validation of those fixed specs: `clean_regime4_2024_unsup_v1_*` count is zero; `clean_regime4_state24_sticky090_v2_*` is the only active current Regime4 prefix.

## TP/SL Contract

Feature:

```text
tp_sl_action_score
```

Label generation:

```text
entry reference: next-bar open
horizon: 48 bars
TP: 1.8%
SL: 1.0%
same-bar TP/SL tie: SL wins
```

Source fixed TP/SL frames:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2025_patchtst__tide__dlinear.csv
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2026_patchtst__tide__dlinear.csv
```

## Audit Snapshot

2025:

```text
rows      105064
columns   185
range     2025-01-01 00:00:00 .. 2025-12-31 23:55:00
tp_sl NaN 0
clean4 probability sum 1.0, NaN 0
clean4 auxiliary NaN 0
clean4 columns 20
pred4 probability sum 1.0, NaN 0
pred4 columns 12
```

2026:

```text
rows      16897
columns   211
range     2026-01-01 00:00:00 .. 2026-02-28 16:00:00
tp_sl NaN 0
clean4 probability sum 1.0, NaN 0
clean4 auxiliary NaN 0
clean4 columns 20
pred4 probability sum 1.0, NaN 0
pred4 columns 12
```

## Usage Rule

New Alpha4/Regime4 experiments should use these fixed CSVs unless explicitly testing an alternative preprocessing layer.

If a future experiment changes Regime4 taxonomy, TP/SL values, horizon, or sidecar source, it is a new preprocessing contract and must not overwrite this one.

Downstream model artifacts trained before the auxiliary feature expansion must be
treated as stale with respect to this canonical preprocessing frame and should be
retrained before comparing Alpha5-style results.
