# BTC v2 Trend-Scan Label Threshold Sweep - 2026-07-14

Status: `research_reference_not_promotion_artifact`. Cheap follow-up to
`docs/model_contracts/btc_v2_walkforward_evaluation_20260714.md`, per the user's request to try
cheap label-parameter changes before a full pipeline redesign. Not run through the Omega Artifact
Integrity Promotion Gate; not wired into `trading_bot.py`.

## What was swept

`scripts/build_1h_trendscan_dataset_btc_20260706.py`'s `TS_THRESHOLD` (the `|t-value| >= threshold`
statistical-significance cutoff for the hourly trend-scan label used by the
`btc_v2_regime_trendscan_hgb_20260714` candidate) at `{2.0, 2.5 (existing baseline, unmodified),
3.0, 3.5}`. `TS_WINDOWS=[3,6,12,24,36,48]` (hours) held fixed -- not swept, to keep this a genuinely
cheap single-parameter check.

For each threshold, a fresh 1h label parquet set was built (scratchpad
`build_btc_trendscan_label_variant.py`, reusing `compute_features`/`resample_1h`/`_trend_scan_fast`
unmodified) into its own `/tmp/btc_trendscan_thresh_{X}/` directory -- the existing baseline
artifact at `tmp/causal_regen_20260516/sigma9_1h_btc_20260706/` was never touched. The same
7-fold quarterly walk-forward protocol from the prior doc was re-run per threshold (script:
scratchpad `sweep_btc_v2_trendscan_threshold_20260714.py`), with `HOURLY_DIR` monkeypatched per
threshold and `TRAIN_END` monkeypatched per fold, policy (`quality_threshold=0.55`,
`regime_threshold=0.50`) held fixed throughout.

Note on label density: raising the threshold does not meaningfully sparsify the label (CASH ratio
only rises from ~0.4% at threshold=2.0 to ~4.5% at threshold=3.5 across 2024-2026) -- this sweep
changes label *quality/confidence*, not label *frequency*, unlike the sparse-event-label direction
suggested as a longer-term fix in the deep analysis doc.

## Results

| ts_threshold | positive folds | mean PnL | std PnL | mean MDD | worst MDD | total trades |
|---|---:|---:|---:|---:|---:|---:|
| 2.0 | 4/7 (57%) | +2.83% | 5.85% | -7.64% | -11.47% | 148 |
| 2.5 (baseline) | 3/7 (43%) | +1.71% | 7.58% | -10.65% | -18.16% | 150 |
| 3.0 | 3/7 (43%) | **-0.28%** | 8.66% | -9.96% | -18.14% | 139 |
| **3.5** | **5/7 (71%)** | **+2.67%** | **4.96%** | **-8.44%** | **-14.59%** | 131 |

Regression check: the `ts_threshold=2.5` row exactly reproduces the prior walk-forward doc's
baseline numbers (positive=3/7, mean_pnl=1.71%, std_pnl=7.58%), confirming the sweep harness
correctly falls back to the existing unmodified artifact for that threshold.

Full per-fold breakdown: `/tmp/btc_v2_threshold_sweep_results.json`.

## Interpretation

**Threshold=3.5 is the most consistent of the four tried** -- higher positive-fold rate (71% vs
43% baseline), lower variance (std nearly halved, 4.96% vs 7.58%), and a less severe worst-case
drawdown (-14.59% vs -18.16%). The relationship is not monotonic: 3.0 is actually the WORST of the
four (negative mean, still-high variance), so this isn't simply "stricter is always better."

**This is a real, non-trivial effect from a single cheap parameter** -- unlike the walk-forward
retraining result alone, changing label confidence measurably shifted the consistency profile. It
supports continuing to treat label design (not just retraining cadence or model architecture) as a
lever worth investigating for BTC.

**Caveat that must not be glossed over**: threshold=3.5 was selected by looking at all 7 folds and
picking the best result -- this is itself a form of selection-level peeking, the same pattern
flagged repeatedly elsewhere in this project's memory. A genuine confirmation would require
applying threshold=3.5 (chosen here) to a window that was not used in this selection at all. None
of the 7 folds above qualify for that, since all of them were part of the comparison used to pick
3.5 in the first place.

## Decision

Do not promote or wire threshold=3.5 (or any variant here) into the live bot. Treat it as the
current leading candidate value for a *properly held-out* confirmation once new data accumulates
past what's already been used in this sweep (2026-07-12), consistent with the reserved-holdout
principle from the prior walk-forward doc.

## Reference files

- `scripts/build_1h_trendscan_dataset_btc_20260706.py` (original, unmodified, `TS_THRESHOLD=2.5`)
- `scripts/train_eval_btc_v2_regime_trendscan_20260714.py` (candidate model, unmodified)
- `docs/model_contracts/btc_v2_walkforward_evaluation_20260714.md` (walk-forward protocol this
  sweep reuses)
- `docs/model_contracts/btc_v1_deep_analysis_20260714.md` (original diagnosis motivating this line
  of research)
