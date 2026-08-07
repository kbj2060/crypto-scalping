# Clean Base Feature Max Hazard Firewall V6 Contract

Status: `experimental_challenger`

## Architecture

- Parent alpha: Deep Gated Gross V2.
- Feature layer: all common causal numeric project features from train/eval CSV, excluding explicit label/future candidate columns.
- Deep layer: V2 3-seed GRU sequence ensemble and KMeans state.
- Supervised layer: original HGB heads plus wide all-feature HGB heads for same-side return, full return, and adverse path risk.
- Runtime layer: bucket-preserving hazard firewall. V2 HIGH/MID/DEFENSIVE is kept, but high-hazard entries are locally capped.
- Stop layer: causal hard-loss and profit-trailing lock, disabled in 2x cost mode to avoid turnover shock.

## Runtime Invariants

- Entry side is never changed.
- Effective exit index can only be less than or equal to audited Lifecycle exit.
- Gross and net notional must be <= 3.6.
- OOS data is used once for evaluation, not threshold selection.
- fee/slippage are charged on entry and exit notional.
- Runtime features use only current/past row values, deep predictions, and closed-trade account state.
