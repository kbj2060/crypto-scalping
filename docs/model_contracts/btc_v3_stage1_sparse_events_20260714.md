# BTC v3 Stage 1 - Sparse Execution-Matched Event Dataset - 2026-07-14

Status: `research_reference_stage1_of_btc_v3_plan`. Not a promotion artifact -- this is the label
dataset Stage 3 will train a quality classifier on, not a trained model itself.

## What this fixes

Per `docs/model_contracts/btc_v1_deep_analysis_20260714.md`'s P1 findings:
1. **Dense/correlated training samples**: the existing `ts_action` label trains on every hourly
   bar within a trend segment, even though live trading only acts once per new signal
   (`is_new_parent_signal`). This script (`scripts/build_btc_v3_sparse_event_dataset_20260714.py`)
   keeps only transition bars -- one row per genuine event.
2. **Proxy label instead of execution outcome**: `ts_action` is a trend-significance statistic, not
   "would a real trade here have made money." Replaced with the REALIZED return of independently
   simulating each event under the exact same ATR stop/trailing/time-exit contract as the live v2
   candidate (constants and `_exit_fill` imported unmodified from
   `train_eval_btc_v2_regime_trendscan_20260714.py` -- no reimplementation drift risk).

Each event is simulated in isolation (not through the single-shared-position state machine the
live backtest uses), so the label reflects that specific signal's own standalone outcome, decoupled
from whatever portfolio-capacity/cooldown gating happens to apply downstream.

## Result

| metric | value |
|---|---|
| dense hourly bars (2024-01-01..2026-07-12) | 22,176 |
| sparse events (new-signal transitions) | 2,948 |
| valid simulated outcomes | 2,928 (99.3% of candidates -- 20 dropped for insufficient trailing bars / entry-not-touched) |
| win rate | 34.7% |
| mean trade return per event | +0.678% |
| median hold (5m bars) | 512 (~42.7 hours) |
| long / short split | 1,468 / 1,460 (near-balanced) |
| exit reasons | stop_loss 1,799 (61%), time_exit 932 (32%), trailing_exit 197 (7%) |

**Reduction: 86.7%** fewer training rows than the dense hourly approach, while producing a
dataset (2,928 samples) far larger than the old risk sidecar's 46 trades. Win rate below 50% with
positive mean return is the expected signature of a trend-following stop/trail exit contract (few
big wins, many small-to-moderate capped losses) -- not by itself a red flag.

Median hold (~43 hours) is far closer to actual BTC v1 holding behavior (33-573 hours observed)
than the original 4-hour quality label horizon was -- directly addresses the label/execution
mismatch finding, though still shorter than the longest observed live holds (this dataset caps at
`MAX_HOLD_BARS` from the v2 script, same as the live candidate's own contract).

## Output

`tmp/causal_regen_20260516/btc_v3_sparse_event_dataset_20260714/sparse_event_dataset.parquet` --
one row per event: the 28 BTC-only hourly features (unchanged from v2 for now; Stage 2 will replace
the non-stationary ones), `side`, `trade_return`, `win`, `hold_bars_5m`, `exit_reason`,
`entry_available_timestamp`.

## Holdout compliance

Built only from data through `2026-07-12 23:59:59`, strictly before `HOLDOUT_START=2026-07-14` per
`docs/model_contracts/btc_v3_holdout_policy_20260714.md` (enforced in code -- the script raises if
asked to build past the holdout).

## Not done in this pass

- Features are still the existing (potentially non-stationary raw-level) 28 BTC features -- Stage 2
  will address this before Stage 3 trains anything on this dataset.
- No classifier has been trained on this dataset yet -- that's Stage 3.
- No purged walk-forward split has been applied to this dataset yet -- Stage 3 will use
  `scripts/btc_v3_walkforward_harness_20260714.py`'s fold-generation logic (or a variant of it)
  against this per-event data.
