# Portfolio Concurrent 3-Asset Native Replay - 2026-07-12

Status: `research_diagnostic_not_live_wired`.

First TRUE concurrent bar-by-bar replay across ETH/SOL/BTC: each asset has its own
independent position slot (unlike every prior portfolio_*.py script, which enforces a
single shared slot across all three assets and so structurally prevented cross-asset
overlap rather than measuring it to be zero). Native fresh-forward, no saved trade ledger
or saved exit timestamps used as replay input.

Concurrency model: `independent_open_positions_per_asset_shared_cash_pool`.
Entry equity convention: `new_position_sized_off_realized_cash_only_ignores_other_sleeves_unrealized_pnl`.
MTM equity formula: `cash + sum(move_i * notional_i * pos_i.entry_equity for open i)`.
Committed margin cap: `1.0`.
Total notional cap: `None`.
Same-direction notional cap: `None`.

## Portfolio results

| split | PnL | MDD (realized) | MTM MDD | trades | WR |
|---|---:|---:|---:|---:|---:|
| validation | 164.03% | -29.24% | -35.50% | 84 | 47.62% |
| oos_extended | 69.70% | -38.21% | -45.01% | 116 | 37.93% |
| oos_frozen_q1_2026 | 83.61% | -38.21% | -45.01% | 70 | 41.43% |

## Per-asset trade aggregates (from the same combined, shared-cash ledger)

Not a dedicated-capital PnL/MDD -- each trade's return is a fraction of the shared pool
at that trade's own entry time, which is itself affected by other assets' realized
gains/losses in between. For an isolated per-asset baseline, see the separate
`--assets <asset>` solo runs.

| split | asset | trades | WR | mean trade return | sum trade return |
|---|---|---:|---:|---:|---:|
| validation | eth | 27 | 44.44% | 0.0174 | 0.4711 |
| validation | sol | 39 | 48.72% | 0.0278 | 1.0827 |
| validation | btc | 18 | 50.00% | 0.0760 | 1.3675 |
| oos_extended | eth | 33 | 45.45% | 0.0154 | 0.5074 |
| oos_extended | sol | 51 | 33.33% | 0.0167 | 0.8512 |
| oos_extended | btc | 32 | 37.50% | 0.0247 | 0.7918 |
| oos_frozen_q1_2026 | eth | 20 | 50.00% | 0.0333 | 0.6660 |
| oos_frozen_q1_2026 | sol | 28 | 39.29% | 0.0191 | 0.5344 |
| oos_frozen_q1_2026 | btc | 22 | 36.36% | 0.0356 | 0.7833 |

## Concurrency diagnostics

| split | max concurrent | % bars 2+ open | % bars 3 open | eth&sol bars | eth&btc bars | sol&btc bars | combined MTM MDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 3 | 87.72% | 64.34% | 18199 | 18955 | 20169 | -35.50% |
| oos_extended | 3 | 88.17% | 69.18% | 36339 | 39698 | 41179 | -45.01% |
| oos_frozen_q1_2026 | 3 | 44.13% | 32.53% | 17212 | 18119 | 21174 | -45.01% |

## Cap-triggered skips

| split | cap | eth | sol | btc |
|---|---|---:|---:|---:|
| validation | margin<=1.00 | 0 | 0 | 0 |
| validation | total_notional | 0 | 0 | 0 |
| validation | same_direction_notional | 0 | 0 | 0 |
| oos_extended | margin<=1.00 | 0 | 0 | 0 |
| oos_extended | total_notional | 0 | 0 | 0 |
| oos_extended | same_direction_notional | 0 | 0 | 0 |
| oos_frozen_q1_2026 | margin<=1.00 | 0 | 0 | 0 |
| oos_frozen_q1_2026 | total_notional | 0 | 0 | 0 |
| oos_frozen_q1_2026 | same_direction_notional | 0 | 0 | 0 |

Replay flags:

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`

## Caveats

- Committed-margin cap is a trivial sanity check only; total-notional and
  same-direction caps (if set above) are the real portfolio-level risk controls.
  A per-asset allocation-percentage cap is still not implemented.
- New positions size off current *realized* cash only (ignore other sleeves' unrealized
  PnL) -- a conservative, explicit modeling choice, not the only valid one.
- Asset processing order is fixed `eth, sol, btc` each bar; margin-capped-skip counts
  above show whether this ordering meaningfully favors ETH when margin is scarce.
- Not a promotion artifact. No live wiring.
