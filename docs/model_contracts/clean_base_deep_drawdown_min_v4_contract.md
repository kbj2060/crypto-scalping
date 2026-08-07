# Clean Base Deep Drawdown Min V4 Contract

Status: `experimental_challenger`

## Architecture

- Core lifecycle: audited clean-base/Lifecycle entry side is preserved.
- Deep layer: V2 3-seed GRU sequence ensemble.
- Unsupervised state: KMeans over deep embeddings and target heads.
- Supervised heads: HGB same-side expectancy and adverse-risk heads.
- Execution: deep exposure buckets are capped by account drawdown, daily drawdown, loss streak, a causal equity MDD budget stop, a hard per-trade loss stop, and a profit-only trailing lock.

## Runtime Invariants

- Entry side is never changed.
- Effective exit index can only be less than or equal to the audited Lifecycle exit.
- Gross and net notional must be <= 3.6.
- OOS data is used once for evaluation, not threshold selection.
- fee/slippage are charged on entry and exit notional.
- MDD stop uses only observed mark-to-market equity and the historical equity peak.
