# Clean Base Deep State Hybrid V1 Contract

Status: `experimental_challenger`

## Architecture

- Deep layer: GRU sequence encoder over `48` bars of market and AI context.
- Unsupervised layer: KMeans state clustering over deep embeddings and deep heads.
- Supervised layer: HGB same-side utility and adverse-risk heads over dynamic trade state plus deep/state features.
- Execution layer: deterministic same-side sleeve only.

## Runtime Invariants

- Clean base/Lifecycle core entries, sides, exits, notionals, and leverage are preserved.
- The hybrid layer can only add a same-side sleeve or abstain.
- No OOS threshold selection.
- Forbidden runtime fields: `evt_candidate_side, evt_candidate_label, evt_side_margin, future close, future high/low, future realized return`.

## Feature Counts

- Sequence features: `75`
- Hybrid head features: `36`
