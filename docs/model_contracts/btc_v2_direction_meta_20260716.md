# BTC v2 Direction + Meta research contract — 2026-07-16

Status: `research_only_not_promoted`

## Scope

This candidate is isolated from BTC v1 and the live path. It tests whether a
BTC-only direction model plus a separately trained take/skip Meta model can
produce a causal, cost-robust entry policy.

## Data and feature contract

- Hourly Direction/Meta inputs are stationary BTC-only features.
- F0 contains 28 parent features. F1 adds 16 explicitly named BTC
  microstructure features.
- ETH/cross-asset features, raw OHLC price levels, target/label/PnL fields,
  compatibility aliases, and silent missing-value repair are forbidden.
- Hourly and 5-minute inputs must be continuous. F1 is truncated explicitly to
  the common source end; missing joined microstructure rows fail immediately.

## Causal learning contract

- Direction labels: confirmed zigzag or trend-scan.
- Direction OOF: five chronological folds with a 72-hour purge.
- Meta training sees only purged OOF Direction events in the initial fit.
- Optional regime Meta uses four BTC-native states formed from the train-frozen
  `rvol_24` median and trailing `logret_24` direction. Each state is fitted
  independently; future rows cannot change the state boundaries.
- Optional monthly Meta refit may add only out-of-time events whose entire
  maximum execution horizon ended before the month being predicted.
- Q1 2026 is diagnostic only and is not used for candidate selection.
- The preregistered future window begins 2026-07-17 and requires at least 90
  elapsed days and 50 trades. It cannot be replaced with an earlier period.

## Execution contract

- Fresh causal 5-minute bar-by-bar replay.
- Hourly feature row becomes available one hour later; entry is on the next
  5-minute bar.
- One position at a time; same-bar TP/SL collision resolves stop first.
- `margin_fraction=0.15`, `leverage=2`, `notional=0.30`.
- PnL is `price_move * notional`; leverage is not multiplied twice.
- Default TP is `8 * ATR`, clamped to `0.8%..3.0%`.
- Default SL is `4 * ATR`, clamped to `0.5%..1.5%`.
- Default maximum hold is 72 five-minute bars.
- Base round-trip cost is 0.14%; stress cost is 0.42%.

## Historical gates

All gates must pass on 2025-09-01 through 2025-12-31:

1. positive PnL;
2. MDD no worse than -8%;
3. at least 40 trades;
4. at least three of four positive months;
5. non-negative PnL at 3x cost;
6. Meta score/realized-return Spearman greater than 0.10;
7. top-quintile lift at least 0.15%;
8. top-three positive-trade concentration no greater than 50%.

Promotion additionally requires the preregistered future gate and an artifact
audit with `promotion_pass=true`.

## Integrity declarations

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
