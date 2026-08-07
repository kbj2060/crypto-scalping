# Deep Learning Training Protocol

Date: 2026-05-14

This note records the default training discipline for future deep learning or RL-style alpha experiments. Model/Data Architect and Implementation Maintainer should apply these controls by default unless a specific experiment intentionally disables one and documents why.

## Required Defaults

- Chronological train/validation split. Do not random-split market rows across time.
- Keep 2026 fixed OOS fully out of model fitting, hyperparameter selection, and early stopping.
- Use validation-monitored early stopping and restore the best checkpoint, not the final epoch.
- Use a learning-rate scheduler such as `ReduceLROnPlateau` or cosine decay with a documented minimum LR.
- Use gradient clipping for sequence/transformer/RL networks.
- Log epoch-level train loss, validation loss, validation accuracy or task metric, LR changes, best epoch, and early-stop reason.
- Tune hyperparameters from the training log instead of only extending epochs.
- Include regularization appropriate to the model: weight decay, dropout, label smoothing or entropy regularization where applicable.
- Preserve cost-aware validation when the model affects live execution or trade selection.
- Emit Red Team audit fields for train window, validation/selection window, OOS window, label source, missing features, leakage checks, and whether the selected result beats the reference model.
- Emit funding-clean audit fields whenever the input state, labels, teacher scores, regime sidecars, or policy artifacts use `last_funding_rate`, `funding_*`, `mta_funding`, `ou_funding_z`, squeeze/crowding derivatives, or upstream model scores trained/scored from those inputs. Promotion requires clean funding provenance or direct clean split comparison with `max_abs_diff == 0.0`.
- For Omega/Omega4.x parent training, emit exact-threshold parent prediction artifacts for every promoted quality threshold: `train_predictions_qXXX.csv`, `validation_predictions_qXXX.csv`, and `oos_predictions_qXXX.csv`. For risk sidecars that consume parent outputs, record `risk_model.precomputed_prediction_dir` and `risk_model.precomputed_prediction_tag`.
- Omega/Omega4.x promotion requires `scripts/audit_omega_artifact_integrity_20260630.py` to return `promotion_pass=true`. Saved trade ledgers and candidate-event replays are diagnostic-only and must not replace per-bar parent prediction artifacts.

## Current Lesson

Longer training alone did not improve Alpha2 teacher overlays. The teacher imitation and outcome-pruning models over-filtered or mis-ranked parent trades despite using early stopping and LR reduction. Future Alpha2 deep work should target richer labels and state rather than simply deeper imitation:

- trade-level cost-adjusted outcome and drawdown contribution,
- real L2 shadow fill outcomes once enough live snapshots exist,
- per-signal counterfactual forward returns, not only executed trade records,
- explicit bad-trade/hazard labels for MDD reduction.
