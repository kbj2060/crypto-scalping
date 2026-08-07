# SOL/BTC data freshness audit + fix (2026-07-21)

## Motivation

User downloaded ETH's raw data personally but had Claude fetch SOL's during an earlier session,
and asked for an integrity check given the difference in process.

## Audit findings

**Raw kline integrity**: no problems. SOLUSDT 5m klines (222,251 rows as of the audit): 0 timestamp
gaps, 0 duplicates, 0 NaN/negative OHLC. 5 zero-volume rows (2024-10-28, 2025-08-29) consistent with
low-liquidity windows, not corruption.

**Real problem found: staleness, not accuracy.**
- SOL's raw kline file (`binance_data/klines/SOLUSDT/SOLUSDT-5m-api.csv`) had been frozen at
  `mtime=2026-07-13 01:55`, last row `2026-07-12 16:50` -- ~9 days stale as of the audit (today
  2026-07-21), while ETH's equivalent pipeline updates daily.
- BTC's raw klines were actually current (updated as a side effect of ETH's own
  `update_features.py` pipeline, which handles ETH+BTC together), but **BTC's feature files**
  (`data/splits/year_oos/btc_features_2026.csv`) were still frozen at the same 07-13 mtime --
  klines were fresh but never reprocessed into features.
- Neither SOL nor BTC had a "current in-progress month" funding-rate REST supplement (the kind
  built for ETH this session, `scripts/fetch_current_month_funding_rest_20260720.py`) --
  `data.binance.vision`'s monthly fundingRate zip only publishes after month-end, so both assets'
  offline pipelines were missing the current month's funding entirely, independent of the kline
  staleness.
- Confirmed real historical asymmetry-bug precedent exists (`sol_adaptive_squeeze_v2_20260720.md`):
  the funding-divisor `/0.0002` constant was ETH-calibrated and genuinely broke SOL's signal
  (SOL funding std ~3.5x ETH's) -- already found and fixed, but establishes that "forgot to
  special-case SOL/BTC" is a real, recurring risk class in this pipeline, not just theoretical.

## Fix applied

1. `scripts/extend_klines_20260713.py --symbol SOLUSDT` / `--symbol BTCUSDT` -- both to
   2026-07-21 11:45.
2. `scripts/download_metrics_funding_generic_20260713.py --symbol {SOL,BTC}USDT --start 2026-07-12
   --end 2026-07-21` -- daily OI/metrics zips through 07-20 (07-21 not yet published, expected).
3. `scripts/fetch_current_month_funding_rest_20260720.py` generalized with a new `--out-dir` flag
   (SOL/BTC's raw-frame builders read `binance_data/funding_rate_other/`, different from ETH's
   `binance_data/funding_rate/`) and run for both assets' July funding.
4. Rebuilt `sol_raw_frame_2024_2026.csv` / `btc_raw_frame_2024_2026.csv`
   (`build_{sol,btc}_raw_frame_2026070{7,8}.py`) -- clean, 0 NaN after coverage trim.
5. Rebuilt `{sol,btc}_features_{2024,2025,2026}.csv` and SOL's live adaptive_squeeze variant
   (`data/splits/year_oos_adaptive_squeeze_sol_20260720/`).
6. Extended the live current-regime wide24 HMM sidecars for both assets (frozen 2024 joblib,
   causal `_transform`, no retraining) via new `scripts/extend_regime3_wide24_sol_btc_20260721.py`.
7. All outputs verified: 0 gaps, 0 duplicates on the extended `{sol,btc}_features_2026.csv`.

## A real (narrow) data artifact found during the fix -- not a bug, a boundary-day completion

Diffing old vs newly-rebuilt BTC features found 94 columns differing on historical rows -- but
**100% of the differing rows (202 of 55,499) fall on exactly 2026-07-12**, the last day covered by
the prior (2026-07-13) extension session. Same exact pattern confirmed independently for SOL's
adaptive_squeeze features (202 rows, same date). Interpretation: the metrics/OI archive for that
boundary day was likely only partially available when originally collected mid-day during the
07-13 session; re-collecting it now (with the full day's official archive available) fills in the
complete values. This is the same class of issue as the kline pipeline's own deliberate "re-fetch
the last bar to catch a still-forming candle" behavior, just at the daily-metrics-archive
granularity instead of the 5-minute-kline granularity. Not evidence of a wider historical-data
integrity problem (only 1 day out of ~2.5 years affected).

## What this does NOT cover yet

- SOL/BTC's newly-built CryptoMamba future-regime models
  (`regime3_cryptomamba_pred_{sol,btc}_h6_nocurrent_20260721`) and the current-regime docs42
  retrains from the same session still reference the **stale (pre-07-21) feature vintage** --
  not re-run as part of this fix. All SOL/BTC backtests from earlier in this session (docs42
  retrain, chop soft-sizing, CryptoMamba entry filter) compared baseline-vs-variant on the *same*
  stale vintage consistently, so those relative comparisons remain valid, but absolute numbers
  would shift slightly if re-run on the freshly extended data.
- BTC/SOL's raw-frame-building scripts (`build_{sol,btc}_raw_frame_2026070{7,8}.py`) do not have
  the same "preserve existing historical values on merge" guard that ETH's `update_features.py`
  got on 2026-07-13 -- they fully recompute the historical raw frame from local source files each
  run. Not observed to cause a problem here (the boundary-day finding above is the only historical
  drift found, and it's a legitimate completion), but it's a latent structural difference from
  ETH's pipeline worth closing if SOL/BTC data gets refreshed on an ongoing basis going forward.

## Status

Research/data-pipeline maintenance only. No `trading_bot.py`/`.env`/`runtime_config.py` changes --
this only refreshes offline research data, not the live trading path (which fetches funding/OI live
via `binance_live_fetcher.py`, unaffected by any of this).
