# BTC v3 research candidate: HMM entry-score + Mamba exit-hazard (2026-07-15)

status: research_candidate_not_live
supersedes: none (parallel to docs/model_contracts/btc_v3_stage1_sparse_events_20260714.md)

## Design

- Entry: same ts_action transition timing as Stage 1, gated by a causal Gaussian-HMM
  win-probability nowcast (`scripts/build_btc_v3_hmm_entry_score_20260715.py`, reusing
  `GaussianStateModel` from `scripts/retrain_clean_regime_hmm_20260517.py` unmodified) instead of
  Stage 1's raw ts_action-only gate.
- Exit: whichever of {existing ATR stop/trail/time-exit contract, Mamba exit-hazard >= 0.5}
  triggers first (`scripts/build_btc_v3_mamba_exit_hazard_20260715.py`, reusing the CBlock/CMBlock
  architecture from `CryptoMambaRegimePred` with a single-logit head).
- Entry-score threshold selected on train/validation only: max eval-window pnl on train/val period among thresholds with >=5 trades
  -> selected threshold = 0.45.

## Result (eval window 2025-10-01 00:00:00..2026-07-12 23:59:59)

| | Stage 1 baseline (ts_action + ATR-only) | This candidate (HMM entry + hazard exit) |
|---|---|---|
| n_events / trades | 871 | 73 |
| win_rate | 0.27210103329506313 | 0.3972602739726027 |
| mean_trade_return_pct | -0.0846464154369854 | -0.01902022906586281 |

Exit reason breakdown (candidate): {"hazard_exit": 68, "stop_loss": 5}

## Compliance

- fresh_forward_bar_by_bar: true
- trade_ledgers_used_as_input (live scoring path): false
- saved_parent_exit_timestamps_used: false
- future_rows_used_for_entry: false
- All training/threshold-selection decisions use only data before docs/model_contracts/btc_v3_holdout_policy_20260714.md's HOLDOUT_START (2026-07-14 00:00:00 UTC).

## Ablation: retrained exit-hazard without `stop_dist`/`trail_dist`

The table above is from the **second** run, after removing `stop_dist`/`trail_dist` from
`FEATURE_NAMES` in `scripts/build_btc_v3_mamba_exit_hazard_20260715.py` to test whether the first
run's 0.91 val AUC was an artifact of those two near-tautological features. Results:

| | Run 1 (with stop_dist/trail_dist) | Run 2 (without) |
|---|---|---|
| exit-hazard val AUC | 0.910 | 0.899 |
| candidate trades | 72 | 73 |
| candidate win_rate | 54.2% | 39.7% |
| candidate mean_trade_return_pct | +0.043% | -0.019% |
| exit reasons | hazard_exit 67 / stop_loss 5 | hazard_exit 68 / stop_loss 5 |

**The ablation did not actually test what it was meant to.** `stop_dist = move_atr + STOP_ATR_PRICE`
and `trail_dist` is likewise an affine function of `peak_move_atr`/`move_atr` -- since
`STOP_ATR_PRICE`/`TRAIL_ATR_PRICE`/`ARM_ATR_PRICE` are fixed known constants (not data-dependent),
a neural net can reconstruct the removed features from `move_atr`/`peak_move_atr` almost exactly
via its own bias terms. That is why AUC barely moved (0.91 -> 0.90): the tautological information
was never actually removed, only relabeled. `hazard_exit` still accounts for ~93% of exits in both
runs.

What DID change substantially is the downstream trading result (win rate 54%->40%, mean return
+0.043%->-0.019%) despite near-identical AUC and an unchanged architecture/procedure -- the only
difference between the two runs is a different random init interacting with a small, fixed
(0.5) hazard decision threshold. At n=72-73 trades, this is a strong signal that **the
comparison-script trading metrics are too noisy at this sample size to judge the candidate at
all**, regardless of which feature set is used.

## Judgment call

Neither run should be read as evidence this candidate beats Stage 1. Two independent problems
compound here: (1) the exit-hazard model's AUC is inflated by ATR-threshold-derived features that
can't be cleanly ablated without also removing `move_atr`/`peak_move_atr` themselves (which carry
genuine, non-tautological trade-management information), and (2) the ~72-trade eval sample is too
small for win-rate/PnL differences of this size to be meaningful -- the same architecture and
procedure produced meaningfully different trading outcomes across the two runs. A real test of
this architecture would need either (a) a much larger number of eval trades (e.g. evaluating over
a longer or multi-asset window), or (b) redesigning the in-trade features to avoid any ATR-relative
framing at all (e.g. raw normalized price/volume dynamics with no reference to the fixed contract's
constants) so the hazard model cannot trivially recover "how close is the existing stop." Both are
out of scope for this pass; as-is, this candidate is **not distinguishable from noise** and should
not be pursued toward Stage 3 without one of those two follow-ups.
