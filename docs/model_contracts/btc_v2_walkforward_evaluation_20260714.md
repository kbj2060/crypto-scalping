# BTC v2 Walk-Forward Evaluation - 2026-07-14

Status: `research_reference_not_promotion_artifact`. This is not a promotion candidate and was
not run through the Omega Artifact Integrity Promotion Gate. It re-evaluates the already-built
`btc_v2_regime_trendscan_hgb_20260714` candidate (see `docs/model_contracts/btc_v2_upgrade_research_20260714.md`)
across multiple independent forward windows instead of the single fixed train/validation/OOS split
that candidate was originally judged on, per `docs/model_contracts/btc_v1_deep_analysis_20260714.md`'s
P0 finding that BTC evaluation has repeatedly reused the same peeked window.

## Method

Reused `scripts/train_eval_btc_v2_regime_trendscan_20260714.py`'s existing pure functions
unmodified (`_read_hourly`, `_fit_parent`, `_read_five_minute`, `_merge_signal`,
`_candidate_side`, `_period`/`_fresh_forward_replay`). For each of 7 quarterly-rolling folds, the
module's global `TRAIN_END` was monkeypatched to the fold's train cutoff (not a code change to the
shared script), the 5-seed hourly HGB trend-scan parent was refit on data strictly up to that
cutoff, and the causal 5-minute fresh-forward replay was run strictly within the fold's own test
window only. The frozen 2024-fit BTC HMM regime gate was NOT refit per fold (loaded from its
existing CSV artifact, same as the original script) -- only the hourly parent classifier rolls
forward. The policy (`quality_threshold=0.55`, `regime_threshold=0.50`) was held fixed across all
folds -- it was NOT re-selected per fold, since the question being tested is whether the
already-chosen configuration generalizes across time, not a new hyperparameter search.

Script: scratchpad `walk_forward_btc_v2_regime_trendscan_20260714.py`. Raw results:
`/tmp/btc_v2_walkforward_results.json`.

## Regression check

Fold C's train boundary (`train_end=2025-06-30`) exactly matches the original single-split
script's `TRAIN_END`. Fold C's result (pnl=+4.80%, mdd=-6.04%, trades=23, wr=43.5%) is
**bit-for-bit identical** to the original script's reported `validation 2025-07..09` result,
confirming the `TRAIN_END` monkeypatch technique reproduces the original pipeline exactly.

## Results (7 quarterly-rolling folds)

| fold | train ≤ | test window | train rows | PnL | MDD | trades | WR |
|---|---|---|---:|---:|---:|---:|---:|
| A | 2024-12-31 | 2025-01-01..03-31 | 8,784 | -0.32% | -18.16% | 25 | 44.0% |
| B | 2025-03-31 | 2025-04-01..06-30 | 10,944 | +7.18% | -6.69% | 25 | 32.0% |
| C | 2025-06-30 | 2025-07-01..09-30 | 13,128 | +4.80% | -6.04% | 23 | 43.5% |
| D | 2025-09-30 | 2025-10-01..12-31 | 15,336 | -7.27% | -14.72% | 26 | 30.8% |
| E | 2025-12-31 | 2026-01-01..03-31 | 17,544 | -3.27% | -18.01% | 27 | 37.0% |
| F | 2026-03-31 | 2026-04-01..06-30 | 19,704 | **+14.59%** | -6.58% | 20 | 40.0% |
| G | 2026-06-30 | 2026-07-01..07-12 | 21,888 | -3.71% | -4.33% | 4 | 0.0% |

**Aggregate**: 3/7 folds positive (43%). Mean PnL +1.71%, std 7.58% (std nearly 4.5x the mean --
high fold-to-fold variance relative to the average). Mean MDD -10.65%, worst -18.16%. 150 trades
total across all folds.

## Interpretation

**This does not clear the bar for "generalizes."** A 43% positive-fold rate with mean PnL near
zero and standard deviation several times the mean is consistent with noise around a near-zero
edge, not a real signal. This corroborates `btc_v2_upgrade_research_20260714.md`'s decision not to
promote this candidate.

**However, it meaningfully changes the story vs. the original single-split OOS result.** The
original evaluation (one fixed fit through 2025-06-30, tested on all of 2026) found OOS PnL
-7.77% to -15.78% -- read as "BTC collapsed in 2026." Under walk-forward retraining, **2026 is not
uniformly bad**: fold E (Q1 2026) is weakly negative (-3.27%), fold F (Q2 2026) is the **best fold
of the entire 7** (+14.59%, MDD only -6.58%), and fold G (July, only 12 days / 4 trades) is too
thin to read either way. The original single fixed-fit result understated how much 2026's later
quarters differ from its Q1, because it never gave the model a chance to see anything past
2025-06-30. This is itself evidence FOR periodic refitting as a lever worth investigating, even
though the current candidate's overall walk-forward performance still isn't strong enough to
promote.

**Practical read**: the fold-to-fold sign flips (positive, positive, positive, negative, negative,
strongly positive, negative) don't show a stable regime-based pattern (e.g. "always bad in H2") --
they look more like each quarter independently rolling a biased coin close to 50/50. This is
consistent with the v1 deep analysis's finding that the underlying quality-head signal itself is
weak (OOS exact precision ~28%, no confidence-quintile lift) -- refitting cadence alone doesn't fix
a fundamentally weak per-event signal, it just changes which specific quarters happen to land on
the right side of a near-coinflip.

## What this does and doesn't tell us

- Does NOT promote `btc_v2_regime_trendscan_hgb_20260714` or any variant of it.
- Does NOT justify tuning thresholds against these fold results (that would just be overfitting to
  7 more windows instead of 1).
- DOES suggest that if this line of research continues, periodic/rolling refitting should be a
  first-class part of the design (not bolted on later), since the single-fit-forever approach
  demonstrably misses real drift between H1 and H2 2026.
- DOES reinforce, independently from the v1 deep analysis's classifier-diagnostic evidence, that
  the core direction/quality signal itself (not the fitting cadence) is the binding constraint --
  a properly-refreshed model still nets out near breakeven with high variance.

## Not done in this pass

- No re-tuning of `quality_threshold`/`regime_threshold` per fold (deliberately fixed, see Method).
- No monthly (vs quarterly) refit cadence tested -- quarterly was chosen for a first pass; monthly
  would give more folds (~30 vs 7) at higher compute cost and noisier per-fold trade counts.
- No genuinely-unseen future holdout reserved yet -- all folds above use data already present in
  the repo (through 2026-07-12), which has already been referenced in some capacity by every prior
  BTC v1/v2 evaluation this project has run. Per the original plan, data arriving from 2026-07-13
  onward should be set aside and not used for any further tuning, reserved for an eventual
  promotion check once/if a stronger candidate emerges.

## Reference files

- `scripts/train_eval_btc_v2_regime_trendscan_20260714.py` (unmodified, functions reused)
- `docs/model_contracts/btc_v2_upgrade_research_20260714.md` (original single-split result)
- `docs/model_contracts/btc_v1_deep_analysis_20260714.md` (evaluation-contract critique that
  motivated this walk-forward pass)
- `docs/model_contracts/live_model_v1_checkpoint_20260714.md` (current live BTC v1 baseline,
  unaffected by this research)
