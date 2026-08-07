# Alpha5.3 HMM Regime4 Dueling DQN Router Parent Contract

Date: 2026-05-17

## Purpose

Promote HMM Regime4 from ordinary parent input feature to routing state.

This architecture removes all legacy `clean_regime_2024_unsup_v4_*` inputs and
uses only the official 4-class HMM Regime4 state for parent MoE routing.

## Architecture

```text
fixed Regime4 TP/SL frame
  -> HMM Regime4 current probs
  -> 4 hard-split specialist parents
  -> Dueling DQN + PER action head per specialist
  -> hard/soft HMM router decision
  -> action-only evaluator
```

Specialists:

```text
bull_specialist
bear_specialist
chop_specialist
whipsaw_specialist
```

Each specialist parent uses:

```text
Dueling DQN action head
PER-like prioritized replay
```

Specialist parent output is restricted to:

```text
action_prob_long
action_prob_short
action_prob_cash
```

`action` and `side` are deterministic derivations from these probabilities.
They are not separate learned heads.

The following are not specialist parent outputs:

```text
notional_exposure
leverage
position_fraction
take_profit
stop_loss
max_hold_bars
cooldown_bars
quality_score
bucket heads
```

## Router Contract

Router columns:

```text
clean_regime4_2024_unsup_v1_bull_prob
clean_regime4_2024_unsup_v1_bear_prob
clean_regime4_2024_unsup_v1_chop_prob
clean_regime4_2024_unsup_v1_whipsaw_prob
```

Router modes:

```text
hard_current
soft_current_th0.00
soft_current_th0.05
soft_current_th0.10
```

Official first-line router source:

```text
HMM current Regime4 only
```

TFT future `regime4_pred_*` is intentionally excluded from this architecture.

## Feature Contract

Input frames:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_tp18_sl10_fixed.csv
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_tp18_sl10_fixed.csv
```

Forbidden:

```text
clean_regime_2024_unsup_v4_*
regime4_pred_*
normal class
cluster/state_code features
```

The four HMM probability columns are router state and are not included in
specialist parent input.

Allowed current HMM Regime4 auxiliary specialist inputs:

```text
clean_regime4_2024_unsup_v1_confidence
clean_regime4_2024_unsup_v1_entropy
clean_regime4_2024_unsup_v1_margin
clean_regime4_2024_unsup_v1_trend_prob
clean_regime4_2024_unsup_v1_micro_prob
clean_regime4_2024_unsup_v1_directional_bias
clean_regime4_2024_unsup_v1_range_prob
clean_regime4_2024_unsup_v1_instability_prob
clean_regime4_2024_unsup_v1_factor_trend
clean_regime4_2024_unsup_v1_factor_flow
clean_regime4_2024_unsup_v1_factor_vol
clean_regime4_2024_unsup_v1_factor_crowding
clean_regime4_2024_unsup_v1_factor_liquidity
clean_regime4_2024_unsup_v1_trend_bias
clean_regime4_2024_unsup_v1_risk_off_prob
clean_regime4_2024_unsup_v1_transition_risk
```

## Training

Script:

```text
/home/llewyn/crypto-scalping/scripts/train_eval_alpha5_3_hmm_dqn_router_parent_20260517.py
```

Specialist split:

```text
router_label = argmax(HMM current Regime4 probabilities)
```

Fallback:

```text
if specialist samples < min_samples or action labels have one class:
  fallback to global Dueling DQN parent
```

Dueling DQN:

```text
dueling value + advantage network
double-DQN target update
PER-like priority sampling from reward magnitude and TD error
behavior cloning auxiliary cross-entropy
```

## Evaluation Protocol

Selection:

```text
train      2025-01-01 .. 2025-09-30
selection  2025-10-01 .. 2025-12-31
OOS        2026-01-01 .. 2026-02-28
selection score: existing Alpha5 alpha2._score
```

Held fixed:

```text
teacher sequence gate disabled
V27/V31 deep scout disabled
```

Action-only evaluator:

```text
unit_exposure = 1.0 for normalized return measurement
enter long when routed action is long while flat
enter short when routed action is short while flat
exit when routed action becomes cash
flip when routed action becomes the opposite side
```

There are no fixed TP/SL, max-hold, cooldown, or quality-score constants in
this line. Position lifecycle is controlled only by the routed DQN action
stream.

## Required Audit

```text
legacy clean_regime_2024_unsup_v4_* count == 0
future regime4_pred_* count == 0
router probability columns present == 4
router probability columns in specialist input == 0
router probability sums ~= 1
normal class absent
cluster/state_code absent
selection_uses_2026 == false
specialist train row counts recorded
fallback specialist count recorded
specialist output contract == action_prob_long/action_prob_short/action_prob_cash
no fixed take_profit/stop_loss/max_hold_bars/cooldown_bars/quality_score
```

## Status

Architecture implemented and compile-checked.

Backtest status:

```text
not yet run
```
