# BTC v3 Stage 2 - Feature Contract Verification - 2026-07-14

Status: `research_reference_stage2_of_btc_v3_plan`. **Finding: no feature rebuild was needed.**

## Why this stage turned out lighter than planned

The BTC v3 upgrade plan called for replacing non-stationary raw-level features (open/high/low/
close, raw OI, raw top-trader ratio) with returns/normalized-distance/rolling-rank equivalents,
per `docs/model_contracts/btc_v1_deep_analysis_20260714.md`'s finding of 2.9-4.6 sigma OOS drift
in the ORIGINAL BTC v1 TabM parent's 147-feature contract.

**That drift problem belongs to BTC v1's feature contract, not the v2/v3 lineage's.** The
Stage 1 sparse event dataset (and the v2 regime-trendscan candidate it's built on) already uses a
completely different, independently-built 28-feature set from
`scripts/build_1h_trendscan_dataset_20260705.py`'s `compute_features()` -- entirely log returns,
rolling realized vol, ATR%, RSI, MACD histogram, Bollinger width/position, volume z-score, taker
imbalance ratio, candle body/wick ratios, rolling skew/kurtosis, SMA-relative distance (already
normalized: `(close - sma50) / close`, not a raw level), a Hurst proxy, and cyclical hour/day-of-
week encodings. Zero raw price/OI/top-trader levels, zero ETH cross-features (BTC-only by
construction, inherited from the Sigma9 BTC leg).

## Verification performed

Rather than assume this is fine, measured it directly: loaded the full 2024-2026 hourly feature
parquet, split into a train-like reference window (`< 2025-07-01`) and a late/OOS-like window
(`>= 2026-01-01`), and computed each feature's median shift in reference-IQR units (same spirit as
the deep analysis's sigma-shift table, IQR-normalized instead of std-normalized since several of
these features are bounded/clipped).

**Result: every one of the 28 features shifted by less than 0.2 IQR** (worst: `rvol_48` at -0.181,
best: the cyclical calendar features at ~0.000). For comparison, the v1 raw-level features the deep
analysis flagged shifted 2.9-4.6 **standard deviations** -- a categorically different, much more
severe kind of break. This feature set does not show that failure mode.

## Conclusion

**Stage 2 is satisfied by the existing Stage 1 feature set -- no rebuild performed.** This is a
genuine, verified finding, not an assumption carried over from the plan. Stage 3 proceeds directly
using these same 28 features from the Stage 1 sparse event dataset.

## What was NOT done (explicitly out of scope, not silently skipped)

- No rolling-rank features were added (the plan's other suggested normalization style) -- not
  needed since the existing set is already stationary by the measurement above; would only be
  revisited if Stage 5's holdout evaluation later reveals a drift problem this train/OOS-window
  check didn't catch.
- Order-book/microstructure features are NOT added yet -- the BTC order-book/microstructure_1m
  recorders were only started 2026-07-14 (same day), nowhere near enough history to use. Tracked
  as a distinct future addition, not blocking Stage 3.
- ETH cross-features were not evaluated for re-inclusion -- the BTC-only design already matches
  what v2 attempt 3 (the best-performing of the three v2 attempts) used.

## Reference files

- `scripts/build_1h_trendscan_dataset_20260705.py` (`compute_features`, unmodified, source of the
  28-feature contract)
- `docs/model_contracts/btc_v3_stage1_sparse_events_20260714.md` (dataset this feature set is
  already part of)
- `docs/model_contracts/btc_v1_deep_analysis_20260714.md` (origin of the drift concern this stage
  checked against and did not find)
