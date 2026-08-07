# Clean Base Deep Gated Gross V2 Contract

Status: `experimental_challenger`

## Architecture

- Core lifecycle: audited clean-base/Lifecycle entry side and exit are preserved.
- Deep layer: v2 3-seed GRU sequence ensemble.
- Unsupervised state: KMeans over deep embeddings.
- Supervised heads: HGB same-side expectancy and adverse-risk heads.
- Execution: deep conviction/adverse gates choose HIGH, MID, DEFENSIVE, or COST3_CAPITAL_PRESERVE exposure buckets.

## Runtime Invariants

- Entry side is never changed.
- Effective exit index equals the audited Lifecycle exit.
- Gross and net notional must be <= 3.6.
- OOS data is used once for evaluation, not threshold selection.
- fee/slippage are charged on entry and exit notional.
