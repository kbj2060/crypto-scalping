# Portfolio Online Asset Switch Native Red-Team Audit - 2026-07-09

Promotion pass: `False`
P0 count: `0`

## Findings

- `P2` `equivalent_to_rule_take_all`: TAKE-only asset switch matched existing rule_take_all result
- `P1` `promotion_grade`: no demonstrated improvement over rule baseline

## Action Counts

```json
{
  "validation": {
    "TAKE_SOL": 10,
    "TAKE_BTC": 9,
    "TAKE_ETH": 6
  },
  "oos_extended": {
    "TAKE_BTC": 24,
    "TAKE_SOL": 22,
    "TAKE_ETH": 3
  }
}
```
