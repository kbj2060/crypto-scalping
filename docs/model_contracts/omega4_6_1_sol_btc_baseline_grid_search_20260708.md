# Omega4.6.1 SOL/BTC Baseline Parameter Grid Search — 2026-07-08

Status: `research_parameter_search`.

Scope:

- SOL baseline family: single-component `zig075`, existing risk sidecars `q065/q070/q075`
- BTC baseline family: single-component `h48qual`, existing risk sidecars `q045/q055`
- Search axes: parent quality threshold tag, final `long_scale`, final `short_scale`,
  `ou_halflife` duration threshold

## SOL

Ledger-level grid best under `trades >= 20`, `VAL MDD >= -30%`:

- Component: `zig075`
- Quality threshold: `q070` / `0.70`
- `long_scale=0.5`
- `short_scale=2.5` or higher; `2.5/2.75/3.0` tie after leverage/notional caps
- Duration threshold: `ou_halflife > 0.0055208323`

Ledger-level VAL:

| split | PnL | MDD | trades | WR |
|---|---:|---:|---:|---:|
| VAL | +93.58% | -14.32% | 28 | 42.86% |
| OOS extended | +14.63% | -33.47% | 39 | 38.46% |
| OOS frozen Q1 2026 | +48.32% | -22.17% | 20 | 50.00% |

Exact replay note:

The existing exact replay baseline used `long_scale=1.0`, `short_scale=2.0`, with the same
`q070` and duration threshold:

| split | PnL | MDD | trades | WR |
|---|---:|---:|---:|---:|
| VAL | +56.75% | -15.87% | 28 | 42.86% |
| OOS extended | +13.92% | -29.38% | 39 | 38.46% |
| OOS frozen Q1 2026 | +41.98% | -21.03% | 20 | 50.00% |

The extended exact grid including `long_scale=0.5` was started but was too slow for the current
turn and was interrupted before writing a report. Until that exact replay completes, treat
`long_scale=0.5`, `short_scale=2.5` as the candidate optimum, and keep `long_scale=1.0`,
`short_scale=2.0` as the exact-replayed baseline.

## BTC

Exact replay best within the `h48qual` baseline family:

- Component: `h48qual`
- Quality threshold: `q055` / `0.55`
- `long_scale=0.5`
- `short_scale=2.5` or higher; `2.5/2.75/3.0` tie after caps
- Duration threshold: `ou_halflife > 0.00541154875`

Exact replay:

| split | PnL | MDD | trades | WR |
|---|---:|---:|---:|---:|
| VAL | +12.39% | -6.49% | 10 | 40.00% |
| OOS extended | +29.23% | -10.65% | 24 | 41.67% |
| OOS frozen Q1 2026 | +10.17% | -10.65% | 16 | 37.50% |

BTC `q045` underperformed the selected `q055`; the two-component router also underperformed the
single-component `h48qual q055` baseline.

Live wiring: none.
