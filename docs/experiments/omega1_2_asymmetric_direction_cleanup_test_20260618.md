# Omega1.2 Asymmetric Direction Cleanup Test

Date: 2026-06-18

## Purpose

This test evaluates the report proposal that the main bottleneck is a toxic raw direction candidate pool. The tested solution is to clean the direction head generation target directly:

- Simulate train-only directional `zigzag_action` candidates.
- If the candidate exits by `stop_loss`, relabel that directional target to `cash`.
- Retrain the same 3-head TabM expert structure.
- Evaluate through the operational parent path: `overlay._build_dec` plus `sleeve._apply_aggressive`.

This is a surgical test of the report's asymmetric direction-loss idea. It is not a live candidate.

## Baseline

Operational parent baseline:

| Split | PnL | MDD | WR | Trades | Exit reasons |
|---|---:|---:|---:|---:|---|
| Validation | 100.5427 | -10.6777 | 0.6364 | 33 | TP 21 / SL 12 |
| OOS | 72.7600 | -8.1082 | 0.7222 | 18 | TP 13 / SL 5 |

## Tested Variant

Variant:

```text
mode = stop_loss_to_cash
epochs = 20
max_exit_samples = 30000
quality_thresholds = 0.45,0.55,0.65,0.75,0.80,0.85,0.90
```

The expert models stopped early at 9 epochs.

Model artifacts:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega1_2_asymmetric_direction_cleanup_20260618/stop_loss_to_cash
```

## Train-Only Cleanup Diagnostic

The cleanup stage confirmed the toxic candidate pool diagnosis:

| Metric | Value |
|---|---:|
| Active directional train candidates | 69,280 |
| Simulated candidates | 69,277 |
| Relabeled to cash | 39,549 |
| Relabeled rate | 57.09% |
| Stop-loss exits | 39,549 |
| Take-profit exits | 28,905 |
| Forced-end exits | 823 |
| Mean net | 0.0028528 |
| Mean MAE | -0.0107292 |

Original class counts:

```text
cash=9,229
long=36,283
short=32,997
```

Purified class counts:

```text
cash=48,778
long=17,303
short=12,428
```

## Result

Best validation-selected result:

| Candidate | Val PnL | Val MDD | Val WR | Val Trades | OOS PnL | OOS MDD | OOS WR | OOS Trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stop_loss_to_cash_q0p45` | -8.0374 | -21.0832 | 0.3478 | 46 | -17.7608 | -25.7553 | 0.3125 | 32 |

Exit reasons:

| Split | TP | SL | Forced end |
|---|---:|---:|---:|
| Validation | 15 | 30 | 1 |
| OOS | 9 | 22 | 1 |

Baseline delta:

| Split | Delta PnL |
|---|---:|
| Validation | -108.5802 |
| OOS | -90.5209 |

All tested quality thresholds produced the same trade set and same result. That means the retrained quality head lost useful ranking resolution under this relabeling scheme.

## Interpretation

The report's diagnosis is directionally correct: the raw candidate pool is toxic. However, the tested direct relabeling fix is too destructive.

The main failure mode is that converting all stop-loss directional samples to `cash` collapses the original directional learning problem:

- The cash class becomes dominant.
- Many market states that may be valid under different TP/SL or hold dynamics are treated as unconditional no-trade.
- The quality head no longer separates candidates well; threshold changes from 0.45 to 0.90 did not change the selected trades.
- The resulting model still enters enough bad trades to lose money, but loses the original parent's high-quality sparse entry behavior.

## Decision

Reject this solution for live use.

The tested asymmetric cleanup should not replace the current parent. The current quality gate parent remains the best live candidate among this branch of tests.

## Next Testable Direction

Do not hard-relabel all SL samples to cash. The next version should preserve directional information and add risk information as an auxiliary target:

1. Keep original `zigzag_action` as the direction target.
2. Add a separate `barrier_outcome` head: `take_profit / stop_loss / timeout`.
3. Penalize only high-confidence direction predictions when the auxiliary barrier head predicts high SL probability.
4. Select with `P(direction)`, `P(TP first)`, `P(SL first)`, and uncertainty, instead of overwriting the core direction label.

That matches the report's triple-barrier recommendation without destroying the direction target.
