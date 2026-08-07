# BTC v3 Frozen Holdout Policy - 2026-07-14

Status: `binding_policy_for_btc_v3_research`. This is Stage 0 of
`docs/model_contracts/btc_v1_deep_analysis_20260714.md`'s "BTC v3 upgrade plan" (full redesign,
user explicitly approved proceeding regardless of time cost, 2026-07-14).

## Why a new holdout is needed

Every prior BTC evaluation this project has run -- v1's own train/validation/OOS split, all three
v2 architecture attempts, the 7-fold walk-forward re-check, and the `TS_THRESHOLD` sweep -- has
looked at essentially the entire available history (2024-01-01 through 2026-07-13). There is no
slice of already-downloaded data left that qualifies as genuinely unseen. Continuing to tune
against any part of this range, no matter how careful the causal bar-by-bar mechanics are, risks
selection-level leakage (the researcher picks the config that happens to work on the window
everyone keeps re-checking).

## The rule

**BTC v3's promotion decision may use only data with `timestamp < HOLDOUT_START` for every
training, feature-engineering, label-design, hyperparameter, and walk-forward-fold decision.**

```
HOLDOUT_START = 2026-07-14 00:00:00 (UTC)
```

Data at or after this timestamp:
- May be downloaded and stored (klines, funding, OI, order-book/microstructure -- the newly-started
  BTC/SOL recorders from this same session should keep running regardless).
- May NEVER be read, plotted, summarized, or used to inform ANY decision about v3's design, labels,
  features, hyperparameters, or model selection while v3 is under development.
- May be evaluated exactly ONCE, at the end, after every other design decision for the specific v3
  candidate being promoted has already been finalized and frozen. If the candidate fails that
  single evaluation, it is not promoted -- going back to tune and re-checking the same holdout
  again voids the holdout's purpose and is not permitted under this policy.

## Practical consequence

Because `HOLDOUT_START` is "today," there is currently very little data past it (hours, not
months). **This means the holdout will not be usable for a real evaluation for a long time** --
weeks to months, depending on how much post-holdout history BTC's event rate needs to accumulate
enough trades for a meaningful read (BTC's own candidate-event rate has consistently been the
sparsest of the three assets, ~30 events per 180 days in every backtest so far). This is accepted
as a deliberate tradeoff per the user's explicit "take as long as necessary, optimize for the best
real result" instruction -- Stage 0-4 work (label/feature/model design, walk-forward validation)
proceeds entirely on pre-2026-07-14 data in the meantime, and the actual promotion checkpoint waits
for the holdout to mature.

## Enforcement

Any new BTC v3 script that reads a date range must assert its end date is strictly before
`HOLDOUT_START`. `scripts/btc_v3_walkforward_harness_20260714.py` enforces this in code (raises if
any fold's test window end is `>= HOLDOUT_START`).

## What this does NOT change

- BTC v1 (the currently live-wired model, `docs/model_contracts/live_model_v1_checkpoint_20260714.md`)
  is unaffected -- this policy governs v3 research only.
- The BTC/SOL order-book and `microstructure_1m` recorders enabled earlier the same session keep
  running unconditionally; their accumulating data is exactly what will eventually populate the
  post-holdout evaluation window (and, once mature, become a candidate new feature source per
  Stage 2 of the upgrade plan).
