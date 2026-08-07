# Clean Base Deep MDD Governor V3 Contract

Status: `experimental_challenger`

## Architecture

- Core lifecycle: audited clean-base/Lifecycle entry side is preserved.
- Deep layer: v2 3-seed GRU sequence ensemble.
- Supervised heads: HGB same-side expectancy and adverse-risk heads.
- Execution: deep exposure buckets plus account drawdown throttle, loss-streak throttle, and causal intra-trade hard stop.

## Runtime Invariants

- Entry side is never changed.
- Effective exit index can only be less than or equal to the audited Lifecycle exit.
- Gross and net notional must be <= 3.6.
- OOS data is used once for evaluation, not threshold selection.
- fee/slippage are charged on entry and exit notional.
