# Clean Base Deep Gated Drawdown Budget V5 Contract

Status: `experimental_challenger`

## Architecture

- Parent alpha: Deep Gated Gross V2, preserving deep HIGH/MID/DEFENSIVE exposure buckets.
- Added risk layer: drawdown-budget governor with account drawdown caps, daily drawdown caps, loss-streak caps, hard loss stop, and profit-only trailing lock.
- Selector: validation chooses the highest-PnL configuration inside a 10%-range MDD band, while requiring cost2 survival and cost3 capital preservation.
- Cost stress behavior: 2x cost disables path stops and lowers notional to reduce turnover; 3x cost preserves capital.

## Runtime Invariants

- Entry side is never changed.
- Effective exit index can only be less than or equal to the audited Lifecycle exit.
- Gross and net notional must be <= 3.6.
- OOS data is used once for evaluation, not threshold selection.
- fee/slippage are charged on entry and exit notional.
- Runtime drawdown controls use only observed cash, observed mark-to-market path, and historical equity peaks.
