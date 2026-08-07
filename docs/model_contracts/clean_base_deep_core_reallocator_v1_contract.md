# Clean Base Deep Core Reallocator V1 Contract

Status: `experimental_challenger`

## Architecture

- Deep layer: v2 3-seed GRU ensemble.
- Unsupervised layer: KMeans state clustering.
- Supervised heads: HGB same/adverse heads plus hold/early-exit heads.
- Execution layer: core direction is preserved, but core notional can scale up/down within a 3.6 gross cap; exits can only move earlier.

## Runtime Invariants

- Entry index and side are preserved.
- Effective exit index can only be less than or equal to Lifecycle core exit.
- Gross and net notional must be <= 3.6.
- No OOS threshold selection.
