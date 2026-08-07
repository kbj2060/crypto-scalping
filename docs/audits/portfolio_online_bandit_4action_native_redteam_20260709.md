# Portfolio Online Bandit 4-Action Native Red-Team Audit - 2026-07-09

Promotion pass: `False`
P0 count: `0`

## Findings

- `P1` `oos_nonpositive_pnl`: OOS PnL=-0.34%
- `P1` `oos_skip_rate_high`: skip rate=95.11%

## Ledger Metrics

| split | PnL | MDD | trades | WR | overlap |
|---|---:|---:|---:|---:|---:|
| validation | 30.45% | -12.44% | 23 | 43.48% | 0 |
| oos_extended | -0.34% | -20.55% | 56 | 33.93% | 0 |
| oos_frozen_q1_2026 | 4.92% | -20.55% | 31 | 32.26% | 0 |

## Action Counts

```json
{
  "validation": {
    "SKIP": 12,
    "TAKE_BTC": 11,
    "TAKE_SOL": 10,
    "TAKE_ETH": 2
  },
  "oos_extended": {
    "SKIP": 1089,
    "TAKE_SOL": 44,
    "TAKE_ETH": 9,
    "TAKE_BTC": 3
  }
}
```
