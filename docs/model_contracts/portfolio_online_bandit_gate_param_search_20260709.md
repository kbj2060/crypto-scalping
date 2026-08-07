# Portfolio Online Bandit Gate Param Search - 2026-07-09

Validation-only search. OOS was evaluated once after config selection.

Selected config: `{'min_samples': 8, 'l2': 50.0, 'skip_margin': -0.2, 'tail_penalty_coef': 0.0, 'notional_penalty_coef': 0.0}`

| split | PnL | MDD | MTM MDD | trades | WR | decisions | skips |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 2.73% | -23.86% | -29.52% | 25 | 36.00% | 25 | 0 |
| oos_extended | 4.05% | -18.01% | -23.50% | 49 | 36.73% | 49 | 0 |
| oos_frozen_q1_2026 | 7.92% | -18.01% | -18.01% | 32 | 37.50% | 32 | 0 |
