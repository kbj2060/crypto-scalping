# Omega1.2.1 True Leverage Baseline Red-Team Audit - 2026-06-13

## Verdict

`omega1_2_1_true_leverage_price_barrier_scale200_cap090` should not be cited as clean untouched OOS at the originally reported `+186.43%`.

It is not blocked for the same reason as the TP-runner-only baseline. There is no active TP-runner meta-selector OOS mining path here. The main issues are:

- the original reported replay used optimistic/non-runtime-equivalent assumptions;
- the preserve-price-barrier risk transform was a research choice documented with OOS diagnostics, so it is not an untouched post-selection test;
- source frames still contain forbidden legacy columns, although this audit path did not feed them into decision/state inputs.

## Metrics

Original reported preserve-price-barrier replay:

| Split | PnL | MDD | WR | Trades |
|---|---:|---:|---:|---:|
| Validation | `+276.67%` | `-20.34%` | `63.64%` | `33` |
| OOS | `+186.43%` | `-15.60%` | `72.22%` | `18` |

Clean intrabar/taker replay:

| Split | PnL | MDD | WR | Trades |
|---|---:|---:|---:|---:|
| Validation | `+49.16%` | `-33.16%` | `46.67%` | `45` |
| OOS | `+120.07%` | `-15.64%` | `65.00%` | `20` |

Failed same-equity TP/SL diagnostic:

| Split | PnL | MDD | WR | Trades |
|---|---:|---:|---:|---:|
| Validation | `-5.31%` | `-31.25%` | `36.19%` | `105` |
| OOS | `+52.66%` | `-17.91%` | `45.90%` | `61` |

## Audit Findings

- Direct forbidden-feature leak into `decision` or runner `state`: not found.
- Forbidden legacy columns in source `frame`: found, 40 columns per split.
- Original ledger intrabar timing sensitivity:
  validation `23/33` trades touched TP/SL earlier by high/low, with `2` reason differences.
  OOS `11/18` trades touched TP/SL earlier by high/low, with `0` reason differences.
- The original `+186.43%` therefore depends materially on close-threshold/maker-style replay assumptions.
- Clean replay still shows positive OOS (`+120.07%`), but validation collapses to `+49.16%` and MDD worsens to `-33.16%`.

## Recommendation

- Use the clean intrabar/taker replay numbers for conservative comparisons.
- Do not use the original `+186.43%` as clean OOS promotion evidence.
- Before any live promotion, run a fresh forward shadow period or a later untouched test period.
- Remove forbidden legacy columns from source frames in active research paths where possible, even if the current decision/state contract does not consume them.

## Artifact

- Report: `tmp/causal_regen_20260516/omega1_2_1_true_leverage_baseline_redteam_audit_20260613/report.json`
