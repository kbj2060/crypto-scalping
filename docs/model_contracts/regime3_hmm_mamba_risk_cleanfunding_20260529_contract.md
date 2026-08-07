# Regime3 HMM + Mamba Risk Clean Funding Contract - 2026-05-29

## Status

Research artifact. Not wired into live trading.

## Purpose

Replace the unstable 4-class `bull/bear/chop/whipsaw` regime surface with:

- current structure: `bull`, `bear`, `chop`
- future medium-horizon structure: `h12`, `h24`, `h48`
- separate whipsaw/risk scores for veto/sizing/exit layers

## Training Data

- Source split: clean funding frames under `tmp/causal_regen_20260516/funding_clean_splits_20260528/`
- Fit source: `training_features_2024.csv`
- Validation: 2024 Q4 with horizon embargo
- Forward tests: 2025 and 2026 feature frames
- 2026 is not used for model selection.

## Artifacts

- Script: `scripts/train_regime3_hmm_mamba_20260529.py`
- Artifact directory: `data/ensemble/supervised/regime3_hmm_mamba_risk_cleanfunding_20260529/`
- Report: `data/ensemble/reports/regime3_hmm_mamba_risk_cleanfunding_20260529_report.json`
- Current HMM model: `regime3_current_hmm_2024.joblib`
- Future/risk Mamba model: `regime3_pred_mamba_shared_2024.pt`
- Sidecars:
  - `training_features_2024_regime3_hmm_mamba_risk.csv`
  - `training_features_2025_regime3_hmm_mamba_risk.csv`
  - `training_features_2026_rebuilt_regime3_hmm_mamba_risk.csv`

## Output Contract

Current regime columns:

- `regime3_current_bull_prob`
- `regime3_current_bear_prob`
- `regime3_current_chop_prob`
- `regime3_current_confidence`
- `regime3_current_entropy`
- `regime3_current_margin`
- `regime3_current_directional_bias`
- `regime3_current_trend_prob`

Future regime columns:

- `regime3_pred_h12_bull_prob`, `regime3_pred_h12_bear_prob`, `regime3_pred_h12_chop_prob`
- `regime3_pred_h24_bull_prob`, `regime3_pred_h24_bear_prob`, `regime3_pred_h24_chop_prob`
- `regime3_pred_h48_bull_prob`, `regime3_pred_h48_bear_prob`, `regime3_pred_h48_chop_prob`
- each horizon also emits `confidence`, `entropy`, `margin`, `directional_bias`, and `trend_prob`

Risk columns:

- `whipsaw_risk`
- `instability_prob`
- `transition_risk`
- `false_breakout_risk`

## Architecture

- Current state detector: Gaussian sticky HMM trained on causal raw/state12 features.
- Current labels for state-to-class mapping: causal ADX + EMA slope + BB width rule.
- Future predictor: shared CUDA Mamba encoder over 72 five-minute bars.
- Future labels: future return plus monotonic path score for h12/h24/h48.
- Risk labels: future path whipsaw, instability, transition, and false-breakout diagnostics.

## Contract Rules

- `whipsaw` is not a class in new action classifiers.
- Legacy prefixes are forbidden in Mamba inputs:
  - `clean_regime_2024_unsup_v4_*`
  - `clean_regime4_2024_unsup_v1_*`
- Any feature containing `regime` is excluded from the Regime3 Mamba input.
- Missing required Regime3 fields must fail fast in downstream active candidates; do not silently map Regime4 fields into Regime3 names.

## Initial Accuracy Snapshot

2024 Q4 validation:

- Current HMM: accuracy `0.7569`, balanced accuracy `0.7774`
- Mamba h12: accuracy `0.3785`, balanced accuracy `0.3666`
- Mamba h24: accuracy `0.4056`, balanced accuracy `0.3649`
- Mamba h48: accuracy `0.9459`, balanced accuracy `0.3356`

2026 forward test:

- Current HMM: accuracy `0.7305`, balanced accuracy `0.7655`
- Mamba h12: accuracy `0.2832`, balanced accuracy `0.3654`
- Mamba h24: accuracy `0.3221`, balanced accuracy `0.3836`
- Mamba h48: accuracy `0.9206`, balanced accuracy `0.3346`
- Risk AUCs: whipsaw `0.6078`, instability `0.8147`, transition `0.5218`, false breakout `0.7043`

## Interpretation

Current HMM is usable as a causal current-state context.

The Mamba future structure predictor is not yet strong enough to own action direction. It is acceptable as low-weight context/risk input only until the h12/h24 balanced accuracy improves materially.
