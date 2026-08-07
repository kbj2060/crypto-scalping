# Regime3 + Whipsaw Risk Policy - 2026-05-29

## Status

Design policy for next clean action-classifier and Alpha8+ candidates.

This document does not rewrite existing historical artifacts. It defines the preferred next feature contract.

## Decision

Do not use `whipsaw` as an independent direction/state class for new action classifiers.

Use:

- direction/structure regime: `bull`, `bear`, `chop`
- risk context: `whipsaw_risk`, `instability_prob`, `transition_risk`, `false_breakout_risk`

The model should answer two separate questions:

- Action classifier: "Is the medium-horizon market structure bull, bear, or chop?"
- Risk/sizing/exit layer: "Is this state too unstable or whipsaw-prone to enter or hold size?"

## Rationale

OOS visual inspection showed frequent `whipsaw` flips. As a class, it can make the action model unstable and can conflict with `chop`. Trading usage is clearer when `whipsaw` is treated as a risk score that tightens or vetoes trades rather than as a direction-like regime class.

The action model can still consume whipsaw information, but only as risk context or through downstream risk/veto layers. It must not treat `whipsaw` as a fourth target class for new primary action ownership experiments.

## Target Feature Contract

Current regime surface:

- `regime3_current_bull_prob`
- `regime3_current_bear_prob`
- `regime3_current_chop_prob`
- `regime3_current_confidence`
- `regime3_current_entropy`
- `regime3_current_margin`

Future medium-horizon regime surface:

- `regime3_pred_h12_bull_prob`, `regime3_pred_h12_bear_prob`, `regime3_pred_h12_chop_prob`
- `regime3_pred_h24_bull_prob`, `regime3_pred_h24_bear_prob`, `regime3_pred_h24_chop_prob`
- `regime3_pred_h48_bull_prob`, `regime3_pred_h48_bear_prob`, `regime3_pred_h48_chop_prob`
- corresponding confidence/entropy/margin fields if the predictor provides them

Whipsaw/risk surface:

- `whipsaw_risk`
- `instability_prob`
- `transition_risk`
- `false_breakout_risk`
- optional source-specific components only when clean provenance is documented

## Existing Regime4 Mapping For Research

Until a native Regime3 artifact is trained, research may derive diagnostic Regime3 features from audited Regime4 probabilities:

- bull = `clean_regime4_state24_sticky090_v2_bull_prob`
- bear = `clean_regime4_state24_sticky090_v2_bear_prob`
- chop = normalized combination of `chop_prob` and non-directional `whipsaw_prob`
- whipsaw risk = `clean_regime4_state24_sticky090_v2_whipsaw_prob`

This mapping is a research bridge only. Active/candidate promotion should prefer explicit Regime3 artifacts and must document the derivation if a bridge is used.

## Horizon Policy

Regime prediction should be medium-horizon, not ultra-short.

Default test horizons:

- `h12`: 60 minutes
- `h24`: 120 minutes
- `h48`: 240 minutes

The action classifier may run every 5 minutes, but its regime context should describe market structure over the next 1 to 4 hours.

## Layer Ownership

Action classifier may use:

- `regime3_current_*`
- `regime3_pred_h12/h24/h48_*`
- compact risk scores such as `whipsaw_risk` only as context

Risk/sizing/exit selector should use:

- `whipsaw_risk`
- `instability_prob`
- `transition_risk`
- `false_breakout_risk`
- confidence/entropy/margin

Risk/sizing/exit selector should apply these as:

- entry veto
- notional reduction
- shorter max hold
- tighter giveback
- smaller TP target in unstable chop

## Forbidden For New Active Candidates

- New action classifier target class `whipsaw`
- Silent alias from old Regime4 columns to new Regime3 names
- Compatibility fallback that fills missing Regime3 fields
- Legacy regime prefixes:
  - `clean_regime_2024_unsup_v4_*`
  - `clean_regime4_2024_unsup_v1_*`

If a required Regime3 field is missing, fail fast and regenerate the upstream feature artifact.

## Validation Requirements

Every Regime3 candidate must report:

- train/validation/OOS date split
- whether the Regime3 predictor was fit on 2024-only or another explicitly declared window
- whether 2026 was used for selection
- current and future regime confusion/transition statistics
- action-classifier ablation with and without risk surface
- Cost1/Cost2/Cost3 PnL, MDD, trades, win rate
- Red Team leakage/funding/feature-contract audit

