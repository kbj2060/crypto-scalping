# Portfolio RL Gate 2-Action Native Red-Team Audit - 2026-07-08

Promotion pass: `False`
P0 count: `0`

## Findings

- `P1` `policy_training_event_ledger`: frozen policy was trained from event-level ledger prototype
- `P1` `promotion_grade`: got False
- `P1` `rl_validation_underperforms_rule`: RL validation PnL is below rule_take_all
- `P1` `rl_oos_underperforms_rule`: RL OOS PnL is below rule_take_all
- `P1` `rl_oos_mdd_underperforms_rule`: RL OOS MDD is worse than rule_take_all
- `P1` `rl_oos_mdd_budget`: RL OOS MDD=-42.89%

## Ledger Metrics

| ledger | PnL | MDD | trades | WR | overlap |
|---|---:|---:|---:|---:|---:|
| validation_rule_take_all | 2.73% | -23.86% | 25 | 36.00% | 0 |
| validation_rl_gate | 0.97% | -28.48% | 33 | 42.42% | 0 |
| oos_rule_take_all | 4.05% | -18.01% | 49 | 36.73% | 0 |
| oos_rl_gate | -12.45% | -42.89% | 44 | 38.64% | 0 |
