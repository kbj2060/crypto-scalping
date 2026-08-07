# ETH MicroScalp Inventory-MoE Ensemble v2 Contract

- Model id: `eth_micro_scalp_inventory_moe_ensemble_v2_20260718`
- Date: 2026-07-18 KST
- Status: `research_no_viable_active_policy`
- Promotion: `false`
- Live/shadow activation: `blocked`

## Purpose

This line implements algorithmic position duration without a fixed or maximum holding period.
Every completed one-minute bar produces a new target position. Five-minute targets are auxiliary
forecasts and do not force a five-minute exit.

```text
current inventory: SHORT | CASH | LONG
next action:       SHORT | CASH | LONG
model output:      3 x 3 inventory-conditioned action values
```

The position remains open while the selected action continues to match the current inventory.
It closes or reverses only when the model's action value and expert consensus change.

## Architecture

Each seed contains:

1. A 60-minute causal price encoder with dilated residual blocks and causal attention pooling.
2. A separate causal microstructure encoder.
3. Three gated regime experts for momentum, mean-reversion, and liquidity-style latent states.
4. Nine action values: one SHORT/CASH/LONG vector for each current inventory state.
5. Seven auxiliary distribution targets for 1/2/3/5-minute returns, 5-minute MFE/MAE, and
   realized volatility.

Three independent seeds (`18`, `29`, `41`) are trained. Their mixed Q values are averaged, while
all nine regime-expert action heads remain separate. A position change requires a tune-selected
number of expert votes. This preserves disagreement as an execution veto instead of averaging
it away.

## Train-only teacher

A backward dynamic program is built only inside the fit interval. For every timestamp and every
possible current inventory, it calculates the cost-aware value of the three next actions.

```text
reward_t = action_t * next_price_return_t
           - 4.5bp * abs(action_t - previous_action)
           - causal_volatility_inventory_penalty_t * abs(action_t)
```

Future paths are used only to construct fit targets. They are never model inputs, and the teacher
is not constructed across tune, validation, development, or split boundaries.

## Feature and availability contract

- 43 ETH-local price, volume, flow, volatility, and regime features
- 24 raw microstructure, health, age, and spread features
- No BTC-derived inputs
- No `kelly_mult`, `signal_bias`, EAI, legacy prediction, trade ledger, or saved exit timestamp
- Raw order-book depth direction is not used
- Unavailable, stale, disconnected, unready, negative-age, or older-than-two-minute state fails
  closed to CASH

All scalers are fit on the fit split only. Five minutes are purged before each split boundary.

## Split contract

| Split | Interval | Use |
|---|---|---|
| Fit | 2026-05-03 to 2026-06-11 | Teacher, scalers, and neural model |
| Tune | 2026-06-11 to 2026-06-21 | Switch margin and expert consensus only |
| Locked validation | 2026-06-21 to 2026-07-01 | No selection |
| Development | 2026-07-01 to 2026-07-12 09:01 | Diagnostic only |

All these intervals are consumed development data for future work. None is promotion OOS.

## Selected research policy

Tune selected:

```text
switch_margin_bp = 0.0
min_expert_agreement = 4 of 9
```

The model can keep a position when fewer than four experts support a change, but it has no
time-based exit.

## Results

| Model | Tune | Locked validation | Development |
|---|---:|---:|---:|
| Dynamic HGB v0 | +1.60% | -13.64% | -19.00% |
| Single-seed Inventory-MoE v1 | +3.41% | -3.39% | +6.41% |
| **Three-seed / nine-expert v2** | **+7.59%** | **-0.71%** | **+11.24%** |

Detailed v2 behavior:

| Split | Gross additive | Cost | Net compounded | MDD | Changes | Median hold | Max hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tune | +10.67% | 3.06% | +7.59% | 4.71% | 34 | 116m | 1,048m |
| Validation | +1.47% | 1.80% | -0.71% | 8.95% | 20 | 264.5m | 3,772m |
| Development | +13.91% | 2.88% | +11.24% | 6.90% | 32 | 230.5m | 2,624m |

Validation cost stress:

| Cost per notional change | Validation return |
|---:|---:|
| 2.00bp | +0.28% |
| 3.25bp | -0.22% |
| 4.50bp | -0.71% |
| 5.50bp | -1.11% |
| 9.00bp | -2.49% |

The ensemble converted validation gross return from negative to positive and narrowed the base
cost failure to 0.71 percentage points. It still fails the 4.5bp contract and cannot be promoted.
The multi-hour median and 44-63 hour maximum holds also show that duration-free optimization did
not learn a consistently scalping-like lifecycle.

## Safety disposition

The checkpoint preserves the selected research policy, but the executable policy is:

```text
enabled = false
min_expert_agreement = 9
activation_allowed = false
```

No live runner or existing bot consumes this artifact. A lower fee or maker execution assumption
must not be used to activate it without actual order submission, queue, partial-fill, cancellation,
and adverse-selection evidence.

## Reproduction

```bash
venv/bin/python -m pytest -q -s \
  test/test_eth_micro_scalp_dynamic_20260718.py \
  test/test_eth_micro_scalp_inventory_moe_20260718.py \
  test/test_eth_micro_scalp_inventory_moe_ensemble_20260718.py

venv/bin/python scripts/train_eval_eth_micro_scalp_inventory_moe_ensemble_20260718.py
```

Artifacts:

- `data/ensemble/eth_micro_scalp_inventory_moe_ensemble_v2_20260718/ensemble.pt`
- `data/ensemble/eth_micro_scalp_inventory_moe_ensemble_v2_20260718/validation_diagnostic_ledger.csv`
- `data/ensemble/eth_micro_scalp_inventory_moe_ensemble_v2_20260718/development_diagnostic_ledger.csv`
- `data/ensemble/reports/eth_micro_scalp_inventory_moe_ensemble_v2_20260718.json`
