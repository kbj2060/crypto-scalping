# Alpha3 FT-Transformer Parent Replacement Redesign Contract

Date: 2026-05-15

Architect: Bohr

Status: design plan, not promoted

## Purpose

This document records the redesigned plan for replacing or augmenting the Alpha3 HGB parent policy with an FT-Transformer style neural parent.

The previous FT-Transformer / TabNet / TFT parent replacement attempts may have been invalid or incomplete because they likely mixed two separate contracts:

- Training labels must be generated with the original `base h288` config.
- `margin110` is not a label-generation config. It is a runtime exposure overlay.

The redesign therefore treats this as a strict parent replacement contract, not a normal model tuning experiment.

## Non-Negotiable Parent Replacement Contract

Any future parent replacement experiment must satisfy all of the following:

1. Training labels are generated only with the `base h288` config from:
   - `data/ensemble/supervised/hf_v13_clean_regime_repro_20260511/v13_clean_regime_h288.pkl`
2. `margin110` is applied only at runtime as the exposure overlay from:
   - `data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl`
3. The execution contract is Alpha3 corrected:
   - `next_open_limit_touch0_fee20`
4. A changed parent candidate requires downstream retraining:
   - teacher/deep gate retraining
   - V21.2 jackpot runner retraining
   - runtime selection on validation only
5. 2026 OOS must not be used for model, epoch, threshold, architecture, or runtime selection.

If any of these are violated, the result is not a valid Alpha3 parent replacement comparison.

## Why The Previous 8-Head FT Replacement Is Not The First Target

The old design tried to predict all parent outputs from one neural backbone:

- `action`
- `quality`
- `notional`
- `leverage`
- `take_profit`
- `stop_loss`
- `max_hold`
- `cooldown`

This is too much for a first replacement attempt.

The major risks are:

- Multi-task loss interference between direction, sizing, leverage, and exit policy.
- Head output scale mismatch.
- Neural output distribution drift versus the original HGB parent.
- Downstream artifacts expecting HGB-like action, confidence, quality, and bucket distributions.
- Possible label contract violation if `margin110` was used during label generation.

Therefore, the new goal is not "FT beats HGB immediately." The first goal is:

> Build a neural-compatible parent family that preserves the HGB parent contract, output schema, and downstream retraining requirements.

## Recommended Experiment Order

### 1. HGB Baseline Contract Audit

Goal: lock the baseline before testing neural replacements.

Checks:

- Rebuild labels with `base h288`.
- Confirm label distribution parity with the original base parent.
- Confirm parent-only metrics for original HGB.
- Confirm full Alpha3 stack metrics under corrected execution.
- Save parent output distributions on train/validation/OOS.

Required outputs:

- Action distribution.
- Confidence distribution.
- Quality distribution.
- Bucket distributions for notional, leverage, TP, SL, max_hold, cooldown.
- Runtime Alpha3 metrics.

### 2. HGB-Compatible Neural Surrogate

This is the first neural candidate.

Goal: imitate the original HGB parent output distribution before trying to outperform it.

Inputs:

- Alpha3 parent feature matrix.
- Same feature columns as the HGB parent.
- No future or target columns.

Targets:

- Base h288 supervised labels.
- HGB soft labels / output distributions.
- HGB quality score.
- HGB bucket selections.

Loss:

- Distillation loss from HGB outputs.
- Supervised base h288 label loss.
- Calibration-aware action/quality loss.

Expected role:

- Direct parent replacement candidate only after calibration passes.
- Otherwise useful as a diagnostic teacher/student model.

### 3. HGB Residual Booster

This is the most realistic near-term improvement candidate.

Goal: keep HGB as the main parent and let the neural model make small, clipped residual corrections.

Flow:

```text
HGB Parent Output
      +
Neural Residual Booster
      ->
Calibrated Parent Output
```

Allowed residuals:

- Action confidence adjustment.
- Quality score adjustment.
- Notional scale adjustment.
- Optional TP/SL bucket nudge.

Guardrails:

- Residual must be clipped.
- No free-form action flip at first.
- Distribution drift must be audited.
- If residual increases PnL but worsens MDD/tail exposure, reject or constrain.

### 4. Grouped-Head FT-Transformer

This is the redesigned version of the original 8-head FT model.

Instead of one shared 8-head structure, heads are split by economic role:

```text
Parent Feature Matrix
    -> Shared Tabular Transformer Backbone
        -> Action/Quality Tower
            -> action logits
            -> quality score
        -> Exposure Tower
            -> notional bucket
            -> leverage bucket
        -> Exit Policy Tower
            -> TP bucket
            -> SL bucket
            -> max_hold bucket
            -> cooldown bucket
```

Design notes:

- Shared backbone should be shallow at first: 2-4 transformer blocks.
- Group-specific towers should be strong enough to reduce task interference.
- Action/quality tower is the primary tower.
- Exposure and exit towers must not be allowed to destroy action learning.
- Consider gradient scaling or PCGrad if multi-task conflict is observed.

### 5. Full Neural Replacement

This is intentionally last.

Only test this after:

- HGB-compatible surrogate is calibrated.
- Residual booster behavior is understood.
- Grouped-head FT passes output drift checks.
- Downstream retraining flow is stable.

## Layer I/O

### Input Layer

Input:

- Parent feature matrix.
- Same feature set as Alpha3 HGB parent unless explicitly testing a feature-reduction candidate.

Preprocessing:

- Fit scalers/encoders only on train split.
- No 2026 information.
- No target/future/label columns.
- Use base h288 label-generation contract.

Output:

- Clean tabular tensor or dataframe for parent candidate.

### Shared Backbone

Input:

- Clean parent features.

Candidate models:

- FT-Transformer.
- Tabular Transformer.
- Shallow MLP residual booster.

Output:

- Latent parent state vector.

### Action/Quality Tower

Input:

- Latent parent state.

Output:

- `action`: CASH / LONG / SHORT.
- `quality_score`.
- optional calibrated confidence.

Priority:

- Highest.
- This tower determines whether the model is even usable as a parent.

### Exposure Tower

Input:

- Latent parent state.
- Optional action/quality context.

Output:

- notional bucket.
- leverage bucket.

Guardrails:

- Must respect runtime `margin110` exposure overlay only after label training.
- Tail exposure drift must be checked.

### Exit Policy Tower

Input:

- Latent parent state.
- Optional action/quality context.

Output:

- take_profit bucket.
- stop_loss bucket.
- max_hold bucket.
- cooldown bucket.

Guardrails:

- CASH rows should not train TP/SL/hold as if they were valid trades.
- Exit outputs must be checked for invalid or extreme combinations.

## Training And Selection Procedure

1. Generate train labels with base h288 only.
2. Train candidate parent on train split.
3. Run validation using runtime `margin110` overlay.
4. Retrain teacher gate from the candidate parent decisions.
5. Retrain V21.2 runner from the candidate parent + teacher decisions.
6. Select teacher runtime and runner config on validation only.
7. Run 2026 OOS once after selection.

No candidate can reuse HGB downstream artifacts unless the test is explicitly marked as a non-promotable diagnostic.

## Required Audit Checks

Blocking failures:

- Labels generated with `margin110`.
- Any 2026 data used for selection.
- Execution contract not equal to Alpha3 corrected `next_open_limit_touch0_fee20`.
- Teacher gate not retrained for the candidate parent.
- V21.2 runner not retrained for the candidate parent.
- Forbidden feature columns or future/target leakage.
- Train/eval timestamp overlap.

Warnings:

- Action distribution drift versus HGB.
- Quality score distribution drift.
- Notional/leverage tail drift.
- TP/SL/max_hold/cooldown bucket drift.
- Trade count collapse.
- Cost3 fragility.
- MDD improvement caused only by over-filtering.

## Metrics To Compare

Parent-only:

- action distribution.
- parent-only PnL/MDD.
- parent-only trade count.
- long/short balance.
- quality and confidence calibration.

Full Alpha3 stack:

- cost1 PnL.
- cost2 PnL.
- cost3 PnL.
- MDD.
- trades/day.
- average notional.
- average leverage.
- exit reason distribution.
- maker/taker execution distribution where available.

Selection score should penalize:

- negative cost2/cost3.
- MDD expansion.
- extreme exposure concentration.
- reduced trade diversity.

## Promotion Rule

A parent replacement candidate can be promoted only if:

- It satisfies the parent replacement contract.
- It passes red-team leakage/accounting/execution audit.
- It beats or materially improves Alpha3 on validation without using 2026.
- Its final 2026 OOS result is competitive after all downstream layers are retrained.
- Its output distribution is explainable and not a one-off artifact.

## Current Recommendation

The next implementation should start with:

1. HGB baseline contract audit.
2. HGB-compatible neural surrogate.
3. HGB residual booster.

Do not begin with a full 8-head FT-Transformer replacement.

