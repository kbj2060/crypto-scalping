# Source-Required Direction Feature Contract - 2026-05-28

## Purpose

This contract records direction features that should not be silently added to active model inputs until their historical and live sources are both present.

## Added Now

The active engineered feature block now includes BTC lead-lag features because the current offline frame already contains `close_btc`, `volume_btc`, and `quote_volume_btc`:

- `btc_ret_1`
- `btc_ret_3`
- `btc_ret_6`
- `btc_ret_12`
- `btc_ret_z_48`
- `eth_btc_ret_spread_12`
- `eth_btc_ret_spread_48`
- `eth_btc_beta_residual_z`
- `btc_lead_eth_follow_gap_3`
- `btc_breakout_eth_lag_dir`
- `btc_volume_impulse_z`
- `btc_eth_volume_rank_spread`
- `btc_impulse_x_eth_beta`

## Not Added To Active Inputs Yet

These feature families require additional persisted historical sources. Do not zero-fill them, alias them, or infer them from unrelated candle data in active/live paths.

| Feature family | Required source | Runtime examples already seen | Contract requirement |
|---|---|---|---|
| Orderbook imbalance | depth snapshots with bid/ask queue levels | `obi`, `bid_ask_spread`, `valid_depth` in microstructure runtime | Persist historical depth features before model training. Missing columns must fail fast. |
| Real tick CVD | aggregated trade stream with buyer/seller side | `cvd`, `delta`, `buy_volume`, `sell_volume` in collector code | Use trade-derived CVD, not candle taker proxy, when a historical table exists. |
| Liquidation map / cluster distance | force-order stream and liquidation cluster cache | `long_usd_1m`, `short_usd_1m`, `liq_cluster_direction`, `liq_cluster_strength`, `distance_to_cluster_pct` | Backfill/persist force-order history before training; stale or invalid stream must be explicit. |
| Cross-exchange premium / basis | spot, mark, index, and preferably multi-exchange prices | live code references mark/index-like state, but offline frame lacks full basis set | Add only after exact timestamp-aligned spot/mark/index history exists. |
| Perp-spot basis divergence | perp close and spot/index close | not present in current offline active frame | Contract must define exchange, symbol, and close-time alignment. |
| Side-specific OI / liquidation flow | long/short OI or account-side series plus liquidations | current frame has aggregate OI and ratios only | Do not derive side OI from aggregate OI alone. |
| Whale transfer / exchange inflow-outflow | on-chain or exchange-flow feed | not present in current offline active frame | External data provenance and lag policy required before model use. |

## Fail-Fast Rule

If any future active model declares one of the source-required columns, the dataset builder must require the exact column name and stop on absence. Legacy aliases, automatic fallback prefixes, or zero defaults are not allowed for active/live candidates.
