# Certified Teacher Utility Ranker V2

- Model ID: `certified_teacher_utility_ranker_v2`
- Architecture: teacher meta encoder + execution replay utility labels + candidate utility ranker + adaptive contract family.
- 2025 is model train/selection/holdout. 2026 is fixed OOS and is not used for selection.
- Audit status: `pass`
- Blocking: `[]`

## Splits
- Fit: `2025-01-01 00:00:00` to `2025-08-31 23:55:00`
- Selection: `2025-09-01 00:00:00` to `2025-10-31 23:55:00`
- Holdout: `2025-11-01 00:00:00` to `2025-12-31 23:55:00`
- OOS: `2026-01-01 00:00:00` to `2026-02-28 16:00:00`

## Cost1 OOS
- PnL: `-4.441029564940768`
- MDD: `-4.612642039781678`
- Trades/day: `2.8295454545454546`

## Output Contract
- `side`, `contract_family`, `expected_net_pct`, `q10_pct`, `notional`, `leverage`, `SL/TP/trailing/max_hold` from selected contract family.
