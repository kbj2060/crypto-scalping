# Portfolio RL Gate 2-Action Native Replay - 2026-07-08

Native evaluation of the frozen 2-action RL gate. Replay does not read saved trade ledgers or saved exit timestamps.

| policy | split | PnL | MDD | MTM MDD | trades | WR |
|---|---|---:|---:|---:|---:|---:|
| rule_take_all | validation | 2.73% | -23.86% | -29.52% | 25 | 36.00% |
| rule_take_all | oos_extended | 4.05% | -18.01% | -23.50% | 49 | 36.73% |
| rule_take_all | oos_frozen_q1_2026 | 7.92% | -18.01% | -18.01% | 32 | 37.50% |
| rl_gate | validation | 0.97% | -28.48% | -33.43% | 33 | 42.42% |
| rl_gate | oos_extended | -12.45% | -42.89% | -43.86% | 44 | 38.64% |
| rl_gate | oos_frozen_q1_2026 | -13.98% | -34.92% | -34.92% | 29 | 37.93% |

Caveat: the policy weights were trained by the earlier event-level prototype, so this is not a complete promotion-grade retrain.
