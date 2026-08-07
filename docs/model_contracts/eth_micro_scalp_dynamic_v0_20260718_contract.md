# ETH MicroScalp Dynamic v0 Contract

- Model id: `eth_micro_scalp_dynamic_v0_20260718`
- Date: 2026-07-18 KST
- Status: `research_no_viable_active_policy`
- Promotion: `false`
- Live/shadow activation: `blocked`

## Decision contract

The policy makes a new target-position decision after every completed one-minute bar.

```text
action_t = SHORT | CASH | LONG
position_t = action_t
```

There is no fixed holding period, maximum holding period, TP/SL, or cooldown. A position is
held while the model score continues to support it and is closed or reversed when the next
per-minute decision changes. The five-minute horizon is only the supervised return-forecast
target; it is not an execution holding period.

Missing, stale, disconnected, not-warmed, negative-age, or older-than-two-minute
microstructure state fails closed to `CASH`.

## Model and inputs

- One `HistGradientBoostingRegressor`
- Target: causal five-minute forward ETH log return in basis points
- Policy: stateful entry/exit hysteresis evaluated every minute
- Inputs: 21 ETH-local price/volume features, nine current raw microstructure features, and
  causal 1/5-minute differences plus 3/5/15-minute rolling means for OBI and trade-flow inputs
- Explicitly excluded: all BTC-derived features, `kelly_mult`, `signal_bias`, EAI, legacy model
  predictions, trade ledgers, saved exit timestamps, and raw order-book direction features

## Split contract

All timestamps are half-open intervals and each pre-boundary segment purges the five-minute
forecast target.

| Split | Interval | Purpose |
|---|---|---|
| Fit | 2026-05-03 to 2026-06-11 | Fit the HGB only |
| Tune | 2026-06-11 to 2026-06-21 | Select entry/exit hysteresis only |
| Locked validation | 2026-06-21 to 2026-07-01 | No model or policy selection |
| Development | 2026-07-01 to 2026-07-12 09:01 | Diagnostic only; already-consumed data |

Policy selection uses bar-by-bar account returns:

```text
turnover_t = abs(position_t - position_(t-1))
net_return_t = position_t * next_price_return_t
               - fee_per_notional_change * turnover_t
```

The base fee is 4.5bp per notional change. A SHORT-to-LONG reversal therefore incurs two
notional changes. Any open position is charged a final close at the end of an evaluation split.

## Initial result

Tune selected this research policy without looking at locked validation or development:

```text
entry_threshold_bp = 10.217974
exit_threshold_bp = -7.6634805
```

| Split | Net return | Gross additive | MDD | Entries/reversals | Median hold | Max hold |
|---|---:|---:|---:|---:|---:|---:|
| Tune | +1.60% | +6.42% | 5.54% | 52 | 16.5m | 852m |
| Locked validation | -13.64% | -8.51% | 14.06% | 66 | 12.0m | 1,388m |
| Development | -19.00% | -16.55% | 19.19% | 48 | 19.5m | 1,334m |

The failure is not explained by fees. Locked validation remains -10.74% at only 2bp per
notional change, and its gross return is already negative. The unconstrained holding contract
also allowed roughly 22-23 hour positions, so the selected policy is not a stable scalping
policy even though its median hold is short.

## Safety disposition

The model artifact preserves the tune-selected policy as `selected_research_policy`, but its
executable `policy` is explicitly disabled and `activation_allowed=false`. This prevents an
accidental research-to-live promotion while retaining the exact failed experiment for audit.

No threshold, feature, architecture, or holding rule may be changed using the locked validation
or development result and then reported against the same intervals as OOS. A subsequent model
family must declare these intervals consumed and freeze before collecting new fresh-forward
evidence.

## Reproduction

```bash
venv/bin/python -m pytest -q -s test/test_eth_micro_scalp_dynamic_20260718.py
venv/bin/python scripts/train_eval_eth_micro_scalp_dynamic_20260718.py
```

Artifacts:

- `data/ensemble/eth_micro_scalp_dynamic_v0_20260718/model.joblib`
- `data/ensemble/eth_micro_scalp_dynamic_v0_20260718/validation_diagnostic_ledger.csv`
- `data/ensemble/eth_micro_scalp_dynamic_v0_20260718/development_diagnostic_ledger.csv`
- `data/ensemble/reports/eth_micro_scalp_dynamic_v0_20260718.json`
