# Clean Base Deep Gated Gross V2 Safe Cap Buckets

Status: `promote_candidate`

## Data Splits

- Parent train: `['2025-01-01 00:00:00', '2025-09-30 23:55:00']`
- Cap map train: `['2025-10-01 00:00:00', '2025-10-31 23:55:00']`
- Validation selection: `['2025-11-01 00:00:00', '2025-12-31 23:55:00']`
- OOS: `['2026-01-01 00:00:00', '2026-02-28 16:00:00']`

## Selected

- Parent config: `dgg_high3.6_mid3.0_def3.0_h-0.0100_m-0.0160_adv99.000_c30.00`
- Cap candidate: `learned_action_edge3_min10_buf0p0035_gatefinal`
- Scheme: `action_edge3`
- Fallback cap: `3.6`

## OOS Result

- PnL: `1597.463037%`
- MDD: `-33.074780%`
- 3x cost PnL: `241.211547%`

## Invariants

- Cap map is learned before validation.
- Validation selects the cap family.
- OOS is report-only.
- Unseen bucket fallback is capped by the configured safe fallback maximum.
