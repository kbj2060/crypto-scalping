# Portfolio Supervised Ranker Native - 2026-07-09

LightGBM supervised candidate ranker. Training labels are validation-only native counterfactual risk-adjusted trade outcomes.

Selected threshold: `-0.02`

| split | PnL | MDD | MTM MDD | trades | WR | decisions | cash |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 28.53% | -13.14% | -16.67% | 26 | 46.15% | 92 | 66 |
| oos_extended | 4.05% | -18.01% | -23.50% | 49 | 36.73% | 49 | 0 |
| oos_frozen_q1_2026 | 7.92% | -18.01% | -18.01% | 32 | 37.50% | 32 | 0 |
