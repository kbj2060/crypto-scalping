# ETH Micro-Scalp v4 Dedicated Shadow Bot Contract

## Scope

The dedicated runner observes the frozen v4 research policy on completed
one-minute bars. It is separate from `trading_bot.py` and has no account,
position-router, leverage-sizing, or order-submission capability.

The v4 artifact must retain `activation_allowed=false` and a disabled artifact
execution policy. The exact 36 base and 24 microstructure feature contract,
artifact hash, public-source parity report, one-minute cadence, availability
health fields, and stream freshness are checked before each decision batch.

## Decisions and holding

The model evaluates SHORT/CASH/LONG on every completed minute using its prior
shadow position. There is no fixed or maximum holding period, time exit, TP/SL,
or cooldown. A position persists only while subsequent model decisions retain
it.

## Counterfactual accounting

A decision at minute `t` is settled only after the completed close at `t+1`
exists. Unit-notional gross return is:

`position[t] * (close[t+1] / close[t] - 1)`

Turnover cost is charged at `t` from the change between the previous and target
shadow positions. The online ledger does not add a synthetic terminal exit.
The latest decision remains unsettled until its next completed minute arrives.

The runner records 2.0, 4.5, 5.5, and 9.0 bp-per-notional-change scenarios.
These are counterfactual diagnostics, not actual exchange fills, account PnL,
or promotion evidence.

## Fail-closed behavior

Artifact, source parity, feature schema, cadence, freshness, position, price, or
ledger-integrity failures prevent new shadow decisions or settlements and write
a failed-closed state. They never route a fallback decision to another model or
to an execution system.
