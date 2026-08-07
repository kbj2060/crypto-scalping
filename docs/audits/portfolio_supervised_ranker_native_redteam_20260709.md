# Portfolio Supervised Ranker Native Red-Team Audit - 2026-07-09

Promotion pass: `False`
P0 count: `0`

## Findings

- `P1` `training_rows_thin`: training rows=25
- `P2` `oos_equivalent_to_rule_take_all`: OOS result matches rule_take_all baseline
- `P2` `oos_no_cash_filtering`: ranker made no OOS cash decisions
- `P1` `promotion_grade`: validation improvement did not transfer to OOS

## Results

| split | PnL | MDD | trades | decisions | cash |
|---|---:|---:|---:|---:|---:|
| validation | 28.53% | -13.14% | 26 | 92 | 66 |
| oos_extended | 4.05% | -18.01% | 49 | 49 | 0 |
| oos_frozen_q1_2026 | 7.92% | -18.01% | 32 | 32 | 0 |
