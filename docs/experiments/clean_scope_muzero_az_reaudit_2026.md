# Clean-Scope MuZero/AZ Re-Audit

Date: 2026-05-06 KST

Verdict: `reject_overlay_keep_base_under_shadow_review`

## Purpose

Re-test the current MuZero/AZ architecture after removing the main data-range concern:

- train only on 2025-01-01 through 2025-10-31
- select controls and overlays only on 2025-11-01 through 2025-12-31 validation
- apply once to 2026-01-01 through 2026-02-28 OOS
- do not overwrite existing model artifacts

## Local Artifacts

- Script: `scripts/run_clean_scope_muzero_az_reaudit_2026.py`
- JSON report: `data/ensemble/reports/clean_scope_muzero_az_reaudit_2026.json`
- Realistic replay ledger: `data/ensemble/reports/clean_scope_muzero_az_reaudit_2026_ledger.csv`
- New model directory: `data/ensemble/supervised/clean_scope_muzero_az_2026/`

## Data Split Audit

| Split | Range | Rows |
|---|---|---:|
| Train | `2025-01-01 00:00:00` to `2025-10-31 23:55:00` | `87496` |
| Validation | `2025-11-01 00:00:00` to `2025-12-31 23:55:00` | `17568` |
| OOS eval | `2026-01-01 00:00:00` to `2026-02-28 16:00:00` | `16897` |

Strict timestamp overlaps:

| Pair | Overlap |
|---|---:|
| Train / validation | `0` |
| Train / eval | `0` |
| Validation / eval | `0` |

## Validation Selection

Clean base controls selected on validation:

```text
exit0.45_age6_max16_dd0.025_loss0.025_cd24
```

Validation result for selected base controls:

| Metric | Value |
|---|---:|
| PnL | `553.610081%` |
| MDD | `-12.656922%` |
| Trades | `695` |
| Trades/day | `11.394091` |

Clean Stage2 overlay selected on validation:

```text
clean_stage2_mz_g0.70_p0.00_d1_sf0.12
```

Validation result for selected Stage2 overlay:

| Metric | Value |
|---|---:|
| PnL | `51.014494%` |
| MDD | `-33.809189%` |
| Trades | `15` |
| Trades/day | `0.245916` |

## OOS Results

Canonical simple accounting, 2026 OOS:

| Model path | PnL | MDD | Trades | Trades/day |
|---|---:|---:|---:|---:|
| Clean base policy + clean exit governor | `177.329809%` | `-17.759665%` | `363` | `6.187500` |
| Clean MuZero/AZ Stage1 | `-68.005910%` | `-81.213664%` | `3` | `0.051136` |
| Clean selected Stage2 MuZero sleeve | `-68.390527%` | `-81.213664%` | `3` | `0.051136` |

Cost stress for clean selected Stage2:

| Cost multiplier | PnL | Survival |
|---|---:|---|
| 1x | `-68.390527%` | fail |
| 2x | `-62.461080%` | fail |
| 3x | `-62.774006%` | fail |

Clean base policy cost isolation:

| Cost multiplier | PnL | MDD | Trades/day |
|---|---:|---:|---:|
| 1x | `177.329809%` | `-17.759665%` | `6.187500` |
| 2x | `92.254878%` | `-18.222118%` | `5.420455` |
| 3x | `-7.969395%` | `-8.438401%` | `3.051136` |

## Realistic Replay Diagnostic

The replay added funding, simple impact, partial fills, maintenance/liquidation checks, and a trade ledger.

| Metric | Value |
|---|---:|
| PnL | `140.767269%` |
| MDD | `-32.331264%` |
| Trades | `101` |
| Trades/day | `1.721591` |
| Liquidations | `0` |
| Partial-fill events | `101` |

This replay is diagnostic only because it recomputes exit state under cost-adjusted equity/fill behavior and produces a different trade path from canonical simple accounting. It must not be used as promotion evidence over the failed canonical OOS result.

## Red-Team Conclusion

The previous rank-1 full OOS result remains reproducible, but it is not a clean live-promotion estimate because the original base policy was trained through 2025-12-31 and overlay selection had OOS selection risk.

After clean retraining:

1. The base policy still has OOS alpha at 1x and 2x costs.
2. The MuZero/AZ overlays collapse OOS trade frequency and PnL.
3. The selected Stage2 sleeve is validation-overfit and should not be promoted.
4. The next iteration should preserve the clean base policy and replace the overlay objective with a conservative gate that cannot reduce trades/day below a hard validation/OOS floor.

Promotion decision: `reject`.

