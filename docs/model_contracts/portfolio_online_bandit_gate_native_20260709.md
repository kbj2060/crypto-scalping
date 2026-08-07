# Portfolio Online Bandit Gate Native - 2026-07-09

Causal online-style gate. Each decision can learn only from trades closed before that timestamp.

| split | PnL | MDD | MTM MDD | trades | WR | decisions | skips |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 10.54% | -18.01% | -24.11% | 25 | 36.00% | 30 | 5 |
| oos_extended | -18.03% | -24.20% | -24.74% | 56 | 32.14% | 1668 | 1612 |
| oos_frozen_q1_2026 | -2.76% | -18.92% | -18.92% | 32 | 34.38% | 404 | 372 |

Paper-informed simplification: this uses a conservative contextual bandit rather than DT/CQL/IQL because the action space is binary and the available trade count is small.

HF papers referenced: Contextual Conservative Q-Learning (2301.01298), IQL (2110.06169), Decision Transformer comparison (2305.14550), PAC-Bayesian Offline Contextual Bandits (2210.13132).
