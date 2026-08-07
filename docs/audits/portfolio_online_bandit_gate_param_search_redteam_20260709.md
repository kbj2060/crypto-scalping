# Portfolio Online Bandit Gate Param Search Red-Team Audit - 2026-07-09

Promotion pass: `False`
P0 count: `0`

## Findings

- `P2` `oos_no_policy_effect`: selected config makes no OOS skips; equivalent to rule_take_all
- `P2` `validation_no_policy_effect`: selected config makes no validation skips
- `P1` `promotion_grade`: param search collapses to no-op rule baseline; not an RL improvement

## Results

| split | PnL | MDD | trades | decisions | skips |
|---|---:|---:|---:|---:|---:|
| validation | 2.73% | -23.86% | 25 | 25 | 0 |
| oos_extended | 4.05% | -18.01% | 49 | 49 | 0 |
| oos_frozen_q1_2026 | 7.92% | -18.01% | 32 | 32 | 0 |
