# Clean Base Lifecycle Editor V1

Date: 2026-05-06 KST

Verdict: `implemented_but_reject_for_promotion_gate`

## Purpose

Implement candidate 3:

```text
clean-base lifecycle editor v1
```

The experiment preserves the frozen clean base policy and frozen clean base exit governor. It does not train or select any 2026 OOS threshold. V1 is intentionally lightweight: it uses train-only exit-hazard buckets plus validation-selected deterministic lifecycle rules, not a full sequence transformer.

Hard constraints:

- no base entry deletion
- no side flip
- preserve base entry timing
- preserve base cooldown
- lifecycle edits only after the base admits a trade
- one-shot OOS on 2026-01-01 through 2026-02-28

## Artifacts

- Script: `scripts/train_eval_clean_base_lifecycle_editor_v1.py`
- Main report: `data/ensemble/reports/clean_base_lifecycle_editor_v1_2026.json`
- Grid: `data/ensemble/reports/clean_base_lifecycle_editor_v1_grid.csv`
- Model directory: `data/ensemble/supervised/clean_base_lifecycle_editor_v1/`
- Model: `data/ensemble/supervised/clean_base_lifecycle_editor_v1/lifecycle_editor.pkl`

## Data Split

| Split | Range | Rows | Use |
|---|---|---:|---|
| Train | `2025-01-01 00:00:00` to `2025-10-31 23:55:00` | `87496` | Fit lifecycle hazard buckets only |
| Validation | `2025-11-01 00:00:00` to `2025-12-31 23:55:00` | `17568` | Select deterministic lifecycle runtime config |
| OOS | `2026-01-01 00:00:00` to `2026-02-28 16:00:00` | `16897` | One-shot evaluation |

## Method

V1 first reconstructs the frozen clean-base admitted trade plan using the base policy, base risk controls, and frozen base exit governor. It then applies lifecycle edits inside those fixed trades:

- no-op
- entry-time notional shrink for high train-bucket exit hazard
- small entry-time notional boost for low train-bucket exit hazard
- validation-selected early-exit threshold and minimum-age adjustment

The edited replay never creates a new entry stream. Edited exits are allowed to happen earlier than the base exit but not later, so the original base entry index, side, and cooldown remain directly comparable.

Selected validation config:

```text
shift-0.03_scale1.00_maxd0.12_agep3_sh999.00x1.00_bo0.12x1.15
```

## Results

Clean base OOS reference:

| Metric | Value |
|---|---:|
| PnL | `177.329809%` |
| MDD | `-17.759665%` |
| Trades | `363` |
| Trades/day | `6.187500` |

Selected lifecycle editor validation:

| Metric | Value |
|---|---:|
| PnL | `908.422288%` |
| MDD | `-12.696352%` |
| Trades | `695` |
| Trades/day | `11.394091` |
| Avg notional | `0.788274` |

Selected one-shot 2026 OOS:

| Metric | Value |
|---|---:|
| PnL | `207.236888%` |
| MDD | `-18.016318%` |
| Trades | `363` |
| Trades/day | `6.187500` |
| Avg notional | `0.690302` |
| Avg leverage | `1.581454` |

Cost stress:

| Cost | PnL | MDD | Trades/day |
|---|---:|---:|---:|
| 1x | `207.236888%` | `-18.016318%` | `6.187500` |
| 2x | `127.776479%` | `-18.281324%` | `6.187500` |
| 3x | `68.834031%` | `-20.202676%` | `6.187500` |

Lifecycle edits on OOS:

| Edit | Count |
|---|---:|
| Boost | `363` |

Exits on OOS:

| Exit | Count |
|---|---:|
| Base exit | `269` |
| Lifecycle early exit | `94` |

Invariant audit:

```text
passed = true
entry_idx_changed = 0
side_changed = 0
cooldown_changed = 0
entry_deleted = 0
side_flip = 0
effective_exit_after_base_exit = 0
```

Realistic replay:

```text
run = false
```

V1 uses the canonical simple fixed-base-trade replay only. Funding, impact, partial-fill, and liquidation ledger replay were not run in this lightweight pass.

## Interpretation

The lifecycle editor improved OOS PnL versus the clean base and kept 1x, 2x, and 3x cost stress positive. It also preserved entry timing, side, and cooldown under an independent base-trade-plan versus lifecycle-plan audit.

It still fails the Red Team promotion gate because OOS MDD is `-18.016318%`, slightly worse than the clean-base gate of `-17.759665%`.

Decision: do not promote V1.
