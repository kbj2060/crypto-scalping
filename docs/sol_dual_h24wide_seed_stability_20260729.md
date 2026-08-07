# SOL dual H24-wide seed-stability validation (2026-07-29)

## Decision

**FAIL — no live promotion and no baseline replacement.**

The originally reported run remains a valid historical run, but its performance
does not survive full retraining under different random seeds. The final router,
quality thresholds, H24 smoothing, risk mappings, regime margin scales, and test
windows were frozen before these runs. OOS was not used for selection.

## Frozen test setup

- Seeds: 17, 29, 43, 71, 101
- ZIG threshold: q0.60
- H24-wide threshold: q0.55
- Router: bull H24-wide x0.25; bear H24-wide x0.50; chop ZIG x1.00
- ZIG and H24 risk mappings copied exactly from the original candidate
- Parent and risk predictor retrained for every seed
- Validation: 2025-09-01 through 2025-12-31
- OOS: 2026-01-01 through 2026-03-31

| Seed | Validation PnL | Validation MDD | Trades | OOS PnL | OOS MDD | Trades |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | -8.85% | -13.15% | 48 | -1.29% | -14.92% | 35 |
| 29 | -1.20% | -9.59% | 49 | +11.48% | -16.93% | 40 |
| 43 | -6.85% | -17.51% | 61 | +23.52% | -7.53% | 40 |
| 71 | +6.66% | -10.14% | 55 | -17.24% | -30.88% | 40 |
| 101 | +4.66% | -8.01% | 50 | +0.26% | -9.10% | 36 |
| Mean | -1.12% | -11.68% | 52.6 | +3.35% | -15.87% | 38.2 |
| Median | -1.20% | -10.14% | 50 | +0.26% | -14.92% | 40 |

Population standard deviation was 6.11 percentage points for validation PnL
and 13.62 points for OOS PnL. None of the five seeds beat the published
baseline on both PnL and MDD in validation and OOS. Only one seed beat the OOS
baseline on both metrics, and no seed passed both windows.

## Diagnosis

The original run's ZIG contribution had a 60% trade win rate in both validation
and OOS. Across the new seeds, the complete router win rate was only 30.6% to
34.0% in validation and 22.5% to 45.0% in OOS. The original validation PnL
(+25.08%) is 4.29 population standard deviations above the new-seed mean.

Parent entry decisions also changed materially across seeds:

| Component | Window | Pairwise action agreement | Active-signal Jaccard |
| --- | --- | ---: | ---: |
| ZIG | Validation | 0.797 | 0.636 |
| ZIG | OOS | 0.790 | 0.662 |
| H24-wide | Validation | 0.793 | 0.492 |
| H24-wide | OOS | 0.811 | 0.500 |

This identifies parent prediction variance, especially disagreement over active
H24 signals and unstable ZIG trade quality, as the primary blocker. Changing
only risk scaling cannot make the candidate promotion-safe.

## Integrity and causal checks

All five runs passed the SOL Omega artifact-integrity audit for both exact-tag
component chains. Every evaluation report records:

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
- `oos_used_for_selection=false`

Diagnostic ledgers were written only after the frozen evaluations.

## Required next experiment

Do not select the best seed. The next candidate should reduce training variance
before performance selection, preferably by averaging direction, quality, and
exit probabilities across a fixed seed ensemble and then running a new,
untouched forward test. Promotion should require a predeclared multi-seed gate,
including positive median validation/OOS PnL, a bounded worst-seed MDD, and a
high pass rate against the baseline.
