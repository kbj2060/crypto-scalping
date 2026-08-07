# Portfolio Supervised Ranker Native Split Redteam - 2026-07-09

## Verdict

`promotion_pass=false`.

The requested split was executed:

- train: `2024-01-01..2024-12-31`
- calibration / auxiliary validation: `2025-01-01..2025-08-31`
- final validation: `2025-09-01..2025-12-31`
- OOS: `2026-01-01..2026-06-30`

## Results

| split | PnL | MDD | MTM MDD | trades | WR | decisions | cash |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_2024 | -36.07% | -44.08% | -45.83% | 73 | 35.62% | 246 | 173 |
| calibration_2025_01_08 | 49.94% | -25.78% | -28.90% | 49 | 46.94% | 225 | 176 |
| final_validation_2025_09_12 | -34.91% | -38.61% | -40.73% | 29 | 20.69% | 152 | 123 |
| oos_2026 | 22.63% | -16.56% | -20.47% | 39 | 41.03% | 179 | 140 |

Selected threshold: `-0.02`, selected only on `2025-01-01..2025-08-31`.

## Findings

- P0: none found for saved-ledger replay leakage. Replay is native bar-by-bar and does not use saved trade ledgers or saved parent exit timestamps as inputs.
- P1: final validation failure. The threshold that passed calibration produced `-34.91%` PnL and `-38.61%` MDD on the untouched final validation window.
- P1: 2024 parent prediction caveat. Existing `train_predictions_qXXX.csv` artifacts are actually 2025 Jan-Sep, not 2024. For 2024, the script scores the frozen parent bundles on 2024 features. This is acceptable as a research run for the requested split, but it is not a clean parent-model historical training reproduction.
- P2: 2024 SOL/BTC regime3 overlay coverage. The 2024 ETH wide24 sidecar was used for the six required `regime3_current_sensitive_wide24_*` columns; rows with missing overlay coverage were explicitly dropped: SOL 11 rows, BTC 28 rows.
- P2: training sample is still thin. The native flat-decision counterfactual set has 70 rows. That is better than the prior 25-row validation-only prototype, but still small for a robust meta-router.

## Contract Flags

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
- `promotion_grade=false`

## Artifacts

- Report: `tmp/causal_regen_20260516/portfolio_supervised_ranker_native_split_20260709/report.json`
- Model: `tmp/causal_regen_20260516/portfolio_supervised_ranker_native_split_20260709/ranker_lgbm.pkl`
- Candidate training set: `tmp/causal_regen_20260516/portfolio_supervised_ranker_native_split_20260709/train_2024_candidate_training_set.csv`
- Threshold grid: `tmp/causal_regen_20260516/portfolio_supervised_ranker_native_split_20260709/calibration_threshold_grid.jsonl`
