# Portfolio RL Gate 2-Action Native Training - 2026-07-09

Validation-only native FQI training from native rule-take transitions. OOS is evaluated once after the policy is frozen.

| policy | split | PnL | MDD | MTM MDD | trades | WR |
|---|---|---:|---:|---:|---:|---:|
| rule_take_all | validation | 2.73% | -23.86% | -29.52% | 25 | 36.00% |
| rule_take_all | oos_extended | 4.05% | -18.01% | -23.50% | 49 | 36.73% |
| rule_take_all | oos_frozen_q1_2026 | 7.92% | -18.01% | -18.01% | 32 | 37.50% |
| native_rl_gate | validation | 54.45% | -13.51% | -16.78% | 20 | 55.00% |
| native_rl_gate | oos_extended | 4.05% | -18.01% | -23.50% | 49 | 36.73% |
| native_rl_gate | oos_frozen_q1_2026 | 7.92% | -18.01% | -18.01% | 32 | 37.50% |

Promotion note: evaluation is native bar-by-bar and does not consume saved trade ledgers, but training only covers rule-visited validation transitions.
