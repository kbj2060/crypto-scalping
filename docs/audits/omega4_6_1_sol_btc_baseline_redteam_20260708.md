# Omega4.6.1 SOL/BTC Baseline Red-Team Audit - 2026-07-08

Overall pass: `False`

## SOL

Promotion pass: `False`

| split | PnL | MDD | trades | WR |
|---|---:|---:|---:|---:|
| validation | 56.75% | -15.87% | 28 | 42.86% |
| oos_extended | 13.92% | -29.38% | 39 | 38.46% |
| oos_frozen_q1_2026 | 41.98% | -21.03% | 20 | 50.00% |

Issues:

- `P1` `sol_oos_mdd_high`: OOS MDD=-29.38%
- `P2` `sol_report_missing_duration_gate_object`: SOL final report stores older grid format without explicit selected duration_gate object
- `P2` `sol_report_missing_replay_flags`: SOL final report predates explicit fresh-forward flags; audit recomputed ledgers instead

## BTC

Promotion pass: `True`

| split | PnL | MDD | trades | WR |
|---|---:|---:|---:|---:|
| validation | 12.39% | -6.49% | 10 | 40.00% |
| oos_extended | 29.23% | -10.65% | 24 | 41.67% |
| oos_frozen_q1_2026 | 10.17% | -10.65% | 16 | 37.50% |

Issues:

- `P2` `btc_thin_validation_trades`: validation gated trades=10
- `P2` `btc_thin_q1_trades`: Q1 gated trades=16

