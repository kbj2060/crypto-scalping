# ETH Micro-Scalp v3 Fresh-Forward Observer Contract

## Purpose and boundary

This observer evaluates the frozen research policy after the v3 freeze without
connecting it to an order path. It does not activate the v3 artifact for live or
paper execution, submit orders, or invent fills.

- Model: `eth_micro_scalp_opportunity_moe_v3_20260718`
- First eligible feature timestamp: `2026-07-17 16:35:00 UTC`
- Decision frequency: every completed one-minute bar
- Position set: `SHORT`, `CASH`, `LONG`
- Fixed/max holding period: none
- Output database: separate observer DuckDB, never the live trading database

## Feature contract

The input CSV must contain strictly increasing, duplicate-free timestamps with an
exact one-minute difference between every pair of rows. It must contain `close`,
all 43 frozen base feature names, and all 24 frozen microstructure feature names
stored in the v3 checkpoint. Missing values, non-finite values, aliases, cadence
gaps, model hash drift, scaler drift, or trainer hash drift fail immediately.
The sole exception is a missing micro/order-book payload when its explicit
`micro_available` or `book_available` flag is false. This follows the frozen
training transform: unavailable payloads are neutralized only by the stored
scaler, while a missing payload marked available fails immediately.

Rows before the freeze may be supplied only as causal warm-up history for the
60-minute window. No decision before the first eligible timestamp is recorded.

The existing live decision frame is five-minute data and is therefore ineligible
as v3 input. Microstructure and order-book collection being current does not
repair the missing one-minute base feature stream.

The companion stream builder may use only the explicit public Binance USD-M
market-data GET allowlist recorded in its build report. It rebuilds features with
the frozen `FeatureEngineer`, joins local micro/order-book rows read-only, and
must pass a scaled overlap-parity gate against the frozen model cache before an
output CSV is atomically published. Account, user-data, and order endpoints are
outside the allowlist.

For this artifact, parity uses 360 rows ending `2026-06-30 16:00:00 UTC`, the
last interval where the archived metric and funding sources used to build the
frozen cache were both current. The later frozen-cache tail contains stale
metric/funding joins and is not a valid source-parity reference. This changes
only the diagnostic interval, not the thresholds or feature values.

## Persistence and restart

Decisions and observer metadata are committed atomically to a dedicated DuckDB.
The timestamp is the decision primary key. Restarting from the same stream reads
the last stored target inventory and inserts only later decisions.

Every position change receives a deterministic intent id, side, and unit-notional
change. These are research intents, not orders.

## Execution evidence

Execution observations have explicit provenance:

- `actual_exchange_fill`: requires an external order id and valid requested/fill
  quantities. Only a full fill is eligible for later performance accounting;
  submitted, partial, canceled, or rejected observations remain ineligible.
- `orderbook_counterfactual`: contains no order id and is never performance
  eligible, even when the book price would have touched a hypothetical limit.

The observer itself can only ingest observations. It cannot create actual fill
evidence. PnL must not be reported until every position-change intent has actual
exchange execution evidence.

## Promotion

Observer output alone does not promote the model. Any future promotion report
must preserve:

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
- `fixed_holding_period_used=false`
- actual and counterfactual execution evidence reported separately
