# Portfolio RL Gate 2-Action Red-Team Audit - 2026-07-08

Promotion pass: `False`
Blocker count: `4`

## Findings

- `P0` `fresh_forward_bar_by_bar`: expected true, got False
- `P0` `trade_ledgers_used_as_input`: expected false, got True
- `P0` `saved_parent_exit_timestamps_used`: expected false, got True
- `P0` `promotion_grade`: expected true, got False

## Ledger Metrics

| ledger | PnL | MDD | trades | WR | overlap |
|---|---:|---:|---:|---:|---:|
| validation_rl_gate | 72.67% | -10.56% | 23 | 47.83% | 0 |
| oos_rl_gate | 70.26% | -20.51% | 29 | 48.28% | 0 |
| validation_rule_take_all | 35.02% | -11.29% | 25 | 44.00% | 0 |
| oos_rule_take_all | 30.92% | -22.43% | 36 | 38.89% | 0 |
