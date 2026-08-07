# Clean Base Deep State Risk Throttle V3 Contract

Status: `experimental_challenger`

## Architecture

- Deep layer: v2 3-seed GRU ensemble.
- Unsupervised layer: KMeans state clustering over ensemble embeddings.
- Supervised layer: HGB same-side utility and adverse-risk heads.
- Execution layer: risk throttle can shrink core notional; same-side sleeve can still add exposure when state is clean.

## Runtime Invariants

- Entry index, direction, and exit index are preserved.
- Effective core notional can only be less than or equal to original core notional.
- Sleeve can only be same-side and temporary.
- No OOS threshold selection.
