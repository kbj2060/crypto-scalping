# Clean Base Deep State Hybrid V2 Contract

Status: `experimental_challenger`

## Architecture

- Deep layer: 3-seed GRU ensemble over `72` bars.
- Per-seed hidden/embedding: `48` / `16`.
- Ensemble embedding width: `48`.
- Early stopping: train-internal chronological holdout, patience `7`.
- Unsupervised layer: KMeans with `6` clusters over ensemble embedding and deep heads.
- Supervised layer: HGB same-side utility and adverse-risk heads.
- Execution layer: deterministic same-side sleeve only.

## Runtime Invariants

- Clean base/Lifecycle core entries, sides, exits, notionals, and leverage are preserved.
- The hybrid layer can only add a same-side sleeve or abstain.
- No OOS threshold selection.
- Forbidden runtime fields: `evt_candidate_side, evt_candidate_label, evt_side_margin, future close, future high/low, future realized return`.
