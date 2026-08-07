# Clean Base Conditional Exposure Booster V1

Date: 2026-05-06 KST

Verdict: `implemented_but_reject_for_alpha_lift`

## Purpose

Build the first Model Architect recommendation:

```text
frozen clean base policy
frozen clean exit governor
conditional exposure booster
existing risk locks
same accounting and OOS gates
```

The booster is intentionally not allowed to change entry timing, action, side, cooldown, or exit configuration. It can only multiply admitted base notional on selected high-edge states.

## Artifacts

- Script: `scripts/train_eval_clean_base_conditional_exposure_booster_v1.py`
- Main report: `data/ensemble/reports/clean_base_conditional_exposure_booster_v1_2026.json`
- Main model: `data/ensemble/supervised/clean_base_conditional_exposure_booster_v1/booster.pkl`
- Main grid: `data/ensemble/reports/clean_base_conditional_exposure_booster_v1_grid.csv`
- H48 loose report: `data/ensemble/reports/clean_base_conditional_exposure_booster_v1_h48_loose_2026.json`
- H48 loose model: `data/ensemble/supervised/clean_base_conditional_exposure_booster_v1_h48_loose/booster.pkl`
- H48 loose grid: `data/ensemble/reports/clean_base_conditional_exposure_booster_v1_h48_loose_grid.csv`

## Training Design

Training uses only the clean train split:

```text
2025-01-01 00:00:00 through 2025-10-31 23:55:00
```

For every sampled active clean-base entry, the trainer scores counterfactual notional boosts:

```text
1.00x, 1.15x, 1.30x, 1.45x, 1.60x
```

Label score penalizes adverse path, peak-to-trough path risk, size, and cost. Tail, funding, and liquidity states can force the base `1.00x` label.

Runtime validation grid selects:

```text
max_boost
probability_floor
max_notional
account_drawdown_boost_cap
daily_drawdown_boost_cap
```

## Results

Clean base reference:

| Metric | Value |
|---|---:|
| OOS PnL | `177.329809%` |
| MDD | `-17.759665%` |
| Trades | `363` |
| Trades/day | `6.187500` |
| Avg notional | `0.600263` |

Main V1 labels:

| Label | Count |
|---|---:|
| `1.00x` | `39435` |
| `1.60x` | `4264` |

H48 loose labels:

| Label | Count |
|---|---:|
| `1.00x` | `69871` |
| `1.15x` | `65` |
| `1.30x` | `55` |
| `1.45x` | `58` |
| `1.60x` | `9951` |

Selected config for both runs:

```text
boost1.15_p0.30_maxn3.6_add0.015_ddd0.008
```

OOS for selected config:

| Run | PnL | MDD | Trades/day | Boosted entries | Avg boost |
|---|---:|---:|---:|---:|---:|
| Main V1 | `177.329809%` | `-17.759665%` | `6.187500` | `0` | `1.000000` |
| H48 loose | `177.329809%` | `-17.759665%` | `6.187500` | `0` | `1.000000` |

## Interpretation

The implementation is safe but does not lift alpha. Validation selection correctly falls back to no-op because boosted candidates underperform the clean base after MDD and coverage penalties.

This is useful information:

1. The clean base is hard to improve with a simple supervised exposure booster.
2. Naive notional scaling can improve headline OOS PnL, but it worsens MDD and trade coverage.
3. The booster did not repeat the MuZero/AZ failure because it cannot block entries, but it also failed to find a reliable high-edge subset.

## Red-Team Decision

Do not promote `clean_base_conditional_exposure_booster_v1`.

The selected no-op path passes the promotion gate only because it reproduces the clean base. It is not an alpha-lift model.

## Next Recommendation

Move to the next architecture candidate:

```text
Base-Preserved Counterfactual Trade Transformer
```

Use base trade sequences as units and predict lifecycle edits:

```text
no-op
hold extension
partial reduce
small boost
```

Hard constraints:

- no entry deletion
- no side flip
- no OOS-selected thresholds
- trades/day floor
- same clean split
- cost 1x/2x/3x stress
- realistic replay

This is more likely to improve alpha than a bar-level exposure classifier because the clean base edge appears to depend on trade lifecycle context, not just entry-time feature buckets.

