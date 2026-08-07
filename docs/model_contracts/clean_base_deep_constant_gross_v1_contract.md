# Clean Base Deep Constant Gross V1 Contract

Status: `experimental_challenger`

## Architecture

- Core lifecycle: audited clean-base/Lifecycle entries, side, and exits are preserved.
- Deep layer: v2 3-seed GRU ensemble over market and AI feature sequences.
- Unsupervised state: KMeans over deep embeddings and target heads.
- Supervised heads: HGB same-side expectancy and adverse-risk heads.
- Execution: replace baseline variable notional with a validation-selected target gross exposure, unless deep risk gates force defensive exposure.

## Runtime Invariants

- Entry side is never changed.
- Effective exit index equals the audited Lifecycle exit.
- Gross and net notional must be <= 3.6.
- OOS data is used once for evaluation, not threshold selection.
- fee/slippage are charged on entry and exit notional.
