# Lifecycle V1 Drawdown Governor V1

Date: 2026-05-06 KST

Verdict: `implemented_but_reject_for_promotion_gate`

## Purpose

Implement an entry-only drawdown governor on top of the frozen `clean_base_lifecycle_editor_v1` lane.

The governor only reduces lifecycle V1 effective notional at trade entry with a fixed per-trade `risk_mult` in:

```text
1.00, 0.85, 0.70, 0.50
```

It does not create entries, delete entries, change side/action, increase leverage, increase notional above lifecycle V1, change exit timing, or resize mid-trade.

## Artifacts

- Script: `scripts/train_eval_lifecycle_v1_drawdown_governor_v1.py`
- Main report: `data/ensemble/reports/lifecycle_v1_drawdown_governor_v1_2026.json`
- Validation grid: `data/ensemble/reports/lifecycle_v1_drawdown_governor_v1_grid.csv`
- Fixed replay ledger: `data/ensemble/reports/lifecycle_v1_drawdown_governor_v1_ledger.csv`
- Model directory: `data/ensemble/supervised/lifecycle_v1_drawdown_governor_v1/`
- Model artifact: `data/ensemble/supervised/lifecycle_v1_drawdown_governor_v1/drawdown_governor.pkl`

## Base Lane

- Base layer id: `clean_base_lifecycle_editor_v1`
- Base report: `data/ensemble/reports/clean_base_lifecycle_editor_v1_2026.json`
- Base report SHA-256: `7f98351dc13d3a383493b4644567d405c5a8cbd20dd20b265fa3208eaedf47c1`

This experiment uses a custom fixed-base-trade replay: it reconstructs the selected lifecycle V1 trade plan and applies the drawdown governor over those fixed lifecycle trades.

## Split

| Split | Range | Rows | Use |
|---|---|---:|---|
| Train | `2025-01-01 00:00:00` to `2025-10-31 23:55:00` | `87496` | Train-only lifecycle hazard buckets and stress thresholds |
| Validation | `2025-11-01 00:00:00` to `2025-12-31 23:55:00` | `17568` | Select exactly one governor config |
| OOS | `2026-01-01 00:00:00` to `2026-02-28 16:00:00` | `16897` | One-shot accept/reject only |

OOS threshold/config selection: `false`

OOS selected-config run count: `1`

## Selected Config

```text
acct0.080_0.130_day0.020_0.030_gb0.020_tail0_soft0.85_hard0.50
```

Config:

```text
account_dd_soft = 0.08
account_dd_hard = 0.13
daily_dd_soft = 0.020
daily_dd_hard = 0.030
trade_giveback_cut = 0.020
tail_risk_cut_enabled = false
soft_mult = 0.85
hard_mult = 0.50
```

Validation grid rows: `972`

Selection score: `899.550366`

## Causality

Risk state is sourced from governed closed cash/equity state before sizing the next trade. Account drawdown and daily drawdown are computed before the current entry fee and before any current-trade outcome exists.

At entry there is no open position in this fixed sequential replay, so `current_unrealized = 0.0`; daily soft drawdown is therefore applied conservatively. Trade giveback uses only the prior closed trade. Current-trade giveback is not available until the next entry.

Tail/liquidity/funding stress uses current entry-row state plus prior closed trade PnL or train-derived adverse-risk thresholds. The selected config has `tail_risk_cut_enabled = false`.

## Results

Selected validation:

| Cost | PnL | MDD | Trades/day | Avg risk mult | 0.50 freq |
|---|---:|---:|---:|---:|---:|
| 1x | `846.294686%` | `-12.510810%` | `11.394091` | `0.995899` | `0.000000` |
| 2x | `392.901139%` | `-12.810509%` | `11.394091` | `0.995036` | `0.000000` |
| 3x | `152.159085%` | `-14.038950%` | `11.394091` | `0.962662` | `0.000000` |

One-shot OOS:

| Cost | PnL | MDD | Trades/day | Avg risk mult | 0.50 freq |
|---|---:|---:|---:|---:|---:|
| 1x | `189.538051%` | `-16.348826%` | `6.187500` | `0.995868` | `0.000000` |
| 2x | `115.280296%` | `-16.588854%` | `6.187500` | `0.994628` | `0.000000` |
| 3x | `54.164677%` | `-16.828500%` | `6.187500` | `0.806474` | `0.258953` |

OOS 1x risk multiplier counts:

```text
1.00 = 353
0.85 = 10
0.70 = 0
0.50 = 0
```

OOS 1x risk reasons:

```text
prior_trade_giveback = 10
daily_dd_soft = 1
```

## Telemetry Deltas

Versus `clean_base_lifecycle_editor_v1` OOS:

```text
pnl_delta = -17.698837
mdd_delta = +1.667493
avg_notional_delta = -0.007616
avg_risk_mult = 0.995868
```

The governor improves OOS MDD from `-18.016318%` to `-16.348826%`, but reduces PnL from `207.236888%` to `189.538051%`.

## Preservation Audit

```text
passed = true
trade_count_changed = 0
entry_idx_changed = 0
side_changed = 0
exit_timing_changed = 0
entry_deleted = 0
notional_increased_above_lifecycle_v1 = 0
leverage_changed = 0
invalid_risk_mult = 0
```

Decision frame audit also passed.

## Promotion Gate

| Gate | Required | Actual | Pass |
|---|---:|---:|---|
| OOS PnL | `>= 205.000000` | `189.538051` | no |
| OOS MDD | `>= -17.759665` | `-16.348826` | yes |
| Trades/day | `>= 6.000000` | `6.187500` | yes |
| Cost2 PnL | `>= 120.000000` | `115.280296` | no |
| Cost3 PnL | `>= 60.000000` | `54.164677` | no |
| Avg risk mult | `>= 0.850000` | `0.995868` | yes |
| 0.50 risk mult freq | `<= 0.100000` | `0.000000` | yes |
| Preservation audit | `pass` | `pass` | yes |

Decision: do not promote V1.

## Realistic Replay

Separate funding/impact/partial-fill realistic replay was not run. The ledger artifact is the custom fixed-base-trade replay ledger for the selected OOS config.
