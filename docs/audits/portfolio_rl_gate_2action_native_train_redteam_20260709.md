# Portfolio RL Gate 2-Action Native-Train Red-Team Audit - 2026-07-09

Promotion pass: `False`
P0 count: `0`

## Findings

- `P1` `promotion_grade`: native training used only 25 rule-visited validation events; promotion_grade remains false
- `P2` `oos_no_policy_effect`: native RL gate exactly matches rule_take_all on OOS

## Ledger Metrics

| ledger | PnL | MDD | trades | WR | overlap |
|---|---:|---:|---:|---:|---:|
| validation_rule_take_all | 2.73% | -23.86% | 25 | 36.00% | 0 |
| validation_native_rl_gate | 54.45% | -13.51% | 20 | 55.00% | 0 |
| oos_rule_take_all | 4.05% | -18.01% | 49 | 36.73% | 0 |
| oos_native_rl_gate | 4.05% | -18.01% | 49 | 36.73% | 0 |
