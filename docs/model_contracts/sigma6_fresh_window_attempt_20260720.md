# Sigma6 fresh-window confirmation attempt -- inconclusive, real fragility found (2026-07-20)

## Goal

[`sigma6_regime_trend_20260705_contract.md`](sigma6_regime_trend_20260705_contract.md) flagged its
own OOS window (2026-03-02..06-30) as repeatedly re-examined by later work (Sigma8-11, the F4-B
dated-ledger rework), asking for confirmation on a genuinely fresh window before treating the
result (+45.9% lev4 / +16.6% lev3 cost1, MDD ~-15%) as validated. This session extended the full
5-artifact dependency chain through 2026-07-20 to attempt that confirmation.

## What was extended (all verified causal, no lookahead)

1. Raw ETH/BTC klines (`scripts/extend_klines_20260713.py`), funding/OI metrics
   (`scripts/download_metrics_funding_generic_20260713.py` + a new
   `scripts/fetch_current_month_funding_rest_20260720.py` for the in-progress month, since
   data.binance.vision's monthly fundingRate zip only publishes after a month completes).
2. `data/training_features_5m.csv` / `data/splits/year_oos/training_features_2026_rebuilt.csv`
   via `scripts/update_features.py --end 2026-07-20` -- verified 0 diff on all historical
   (pre-07-12) rows before overwriting.
3. Regime3-wide24 HMM sidecar via `scripts/apply_regime3_wide24_sidecar_extended_20260713.py`
   (re-run unchanged) -- reproducibility max abs diff 1.1e-16 on the old range.
4. CryptoMamba-h6 stability sidecar: no extend-only (frozen-checkpoint, inference-only) script
   existed before this session; wrote `scripts/apply_regime3_cmamba_h6_sidecar_extended_20260720.py`
   (loads the saved `.pt` state_dict/scaler/feature_cols, no retraining) plus re-ran
   `scripts/materialize_regime3_cmamba_h6_sidecar_contract_20260601.py`. **Known deviation**: the
   original model's `DEFAULT_TRANSFORMS` pointed at a frozen `funding_clean_splits_20260528`
   snapshot last extended only to 06-30; this session's extension used the live
   `data/splits/year_oos/training_features_2026_rebuilt.csv` instead (the only file with the fresh
   rows), which diverges from that snapshot on a handful of rows/columns (median abs diff in
   scaled feature space is 0.0 -- most rows match exactly -- but rare outlier rows differ by up to
   ~500 std devs, consistent with the "funding_clean" pass having scrubbed a handful of raw
   anomalies the live pipeline still has). Isolated impact on the final backtest: minor
   (+43.4%/+20.1% vs original +45.9%/+16.6% on the old OOS window when this was the ONLY component
   swapped -- see below).
5. Sigma3-1h ensemble tape: `build_1h_trendscan_dataset_20260705.py` needs `numba`, which requires
   numpy<2.3; this venv has numpy 2.3.5 (a known, previously-documented conflict). Found that
   `/home/llewyn/miniconda3/envs/quant_ai` already has a compatible numpy 2.2.5 + numba 0.64.0 and
   used that interpreter to re-run the *original, unmodified* `build_1h_trendscan_dataset_20260705.py`
   and `train_sigma3_1h_ensemble_20260705.py` directly -- no numba workaround needed. (A vectorized
   numba-free reimplementation, `scripts/build_1h_trendscan_dataset_extended_20260720.py`, was
   written first as a fallback and is left in the repo but was superseded by the real numba run.)

## Result: does not reproduce the frozen baseline, even on the unchanged original window

Re-running the *original* OOS window (2026-03-02..06-30 exactly, no new dates at all) against the
freshly-extended-and-retrained sigma3 tape:

| | lev4 | lev3 |
|---|---:|---:|
| Contract doc (frozen) | pnl +45.9%, mdd ~-15% | pnl +16.6%, mdd ~-15%, WR 50% |
| Reconstructed from `.orig` (pre-this-session) 1h parquets, no retrain | pnl +45.85%, mdd -15.08%, WR 44.4% | pnl +16.64%, mdd -16.04%, WR 50% |
| Same window, sigma3 ensemble retrained (numba, unmodified script) on re-derived 1h data | **pnl -22.0%, mdd -37.4%, WR 30.3%** | **pnl -9.7%, mdd -23.2%, WR 37.0%** |

The first row (`.orig` reconstruction) exactly reproduces the frozen numbers, confirming this
session's evaluation harness itself is correct. The regression appears only after retraining the
sigma3 HGB ensemble -- and it appears identically whether the retrain uses the numba-free
vectorized labeling or the true unmodified numba kernel, which rules out the numba-vs-vectorized
question as the cause.

**Isolated component tests** (swap one artifact at a time, holding the other two at their
pre-session `.orig`/backup state):
- regime3-wide24 HMM re-extended: no material change (near-identical to frozen numbers, as
  expected given the ~1e-16 reproducibility diff).
- CryptoMamba sidecar re-extended (via the live-file source-swap): minor change (+43.4%/+20.1% vs
  +45.9%/+16.6%) -- present but small.
- Sigma3-1h tape retrained (either method): **the regression appears here alone** (-14.1% lev4
  isolated, -22.0% lev4 combined with the other two extensions) -- swings the entire OOS sign.

## Root cause: training-sample-weight sensitivity, not a labeling bug

`train_sigma3_1h_ensemble_20260705.py` trains `HistGradientBoostingClassifier(early_stopping=False,
max_iter=250, ...)` with `sample_weight = clip(|ts_t_value|, 0.5, 12.0)` -- the *continuous*
trend-scan t-statistic, not just the discrete action label. Verified `sigma3_1h_2025.parquet`
(part of the training window, `TRAIN_END=2025-06-30`) is **byte-identical** between this session's
re-derivation and the pre-session `.orig` backup on every column including `ts_t_value` -- ruling
out 2025 as the source. **2024 has no `.orig` backup to compare against** (never explicitly saved
before this session's first edit), so the exact row(s)/column(s) responsible could not be pinned
down. Given HGB's greedy histogram-based splitting is sensitive to exact continuous sample weights
near bin boundaries, a handful of altered 2024 `ts_t_value` values is a plausible and sufficient
explanation for the ~4% (334/8874 rows) of flipped `primary_side` predictions that follow.

## Conclusion

**This is not a valid fresh-window confirmation or disconfirmation of Sigma6** -- the retrain step
itself doesn't reproduce the frozen baseline even on the *unchanged* original window, so nothing
can be concluded about the new July 2026 bars either way from this attempt. The more important,
independently-standing finding is: **Sigma6's frozen result is highly sensitive to retraining its
own upstream sigma3-1h ensemble** -- a training pipeline that is supposed to be reproducible (fixed
seeds, `early_stopping=False`) flips ~4% of predicted trade signals and the entire OOS PnL sign
when re-run against 2024 training data that should be identical but could not be verified as such.
This is independent evidence that Sigma6's "strongest generalizing result" status rests on a more
fragile foundation than previously documented -- the frozen tape/ledger artifacts should be treated
as exactly that (frozen, one specific realization), not as a reproducible recipe that can be safely
re-run and re-trusted without re-verifying the 2024 training inputs bit-for-bit first.

## Status / no live impact

`model_status=research_inconclusive`. No `trading_bot.py` or `.env` changes -- Sigma6 was never
live-wired. Extended raw/feature/regime data (items 1-4 above) are real, verified, reusable
extensions and were left in place; the sigma3-1h tape/ensemble artifacts (item 5) were also left in
their retrained-and-extended state for future use, but should not be treated as reproducing the
2026-07-05 contract's specific frozen numbers.

## Recommended next step (not done here)

Before trying this again: locate or reconstruct a `.orig`-equivalent backup of
`sigma3_1h_2024.parquet` as it existed when the frozen contract tape was built, diff it column-by-
column (especially `ts_t_value`) against a fresh 2024 rebuild, and pin down the exact source of
drift before trusting any retrain-based fresh-window result for Sigma6.
