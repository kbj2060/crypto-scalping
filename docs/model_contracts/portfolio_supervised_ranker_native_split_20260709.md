# Portfolio Supervised Ranker Native Split - 2026-07-09

LightGBM candidate ranker trained on 2024 native counterfactual outcomes.

Selected threshold from 2025-01..2025-08 calibration: `-0.02`

| split | PnL | MDD | MTM MDD | trades | WR | decisions | cash |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_2024 | -36.07% | -44.08% | -45.83% | 73 | 35.62% | 246 | 173 |
| calibration_2025_01_08 | 49.94% | -25.78% | -28.90% | 49 | 46.94% | 225 | 176 |
| final_validation_2025_09_12 | -34.91% | -38.61% | -40.73% | 29 | 20.69% | 152 | 123 |
| oos_2026 | 22.63% | -16.56% | -20.47% | 39 | 41.03% | 179 | 140 |

Contract flags: `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`.

Promotion verdict: `promotion_pass=false` because the selected calibration threshold fails final validation (`-34.91%` PnL, `-38.61%` MDD).

2024 caveat: existing `train_predictions_qXXX.csv` artifacts are 2025 Jan-Sep, not 2024. This run scores the frozen parent bundles on 2024 features to satisfy the requested split. SOL/BTC 2024 rows with missing regime3_current overlay coverage were dropped explicitly (SOL 11, BTC 28).
