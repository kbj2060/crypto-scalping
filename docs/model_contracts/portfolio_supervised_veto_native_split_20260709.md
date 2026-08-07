# Portfolio Supervised Veto Native Split - 2026-07-09

2-action LightGBM veto gate. The rule top candidate is unchanged; the model only chooses TAKE_TOP or SKIP_TOP.

Selected threshold: `-0.08`

| split | PnL | MDD | MTM MDD | trades | WR | decisions | skips |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_2024 | -13.67% | -35.02% | -38.55% | 59 | 37.29% | 588 | 529 |
| calibration_2025_01_08 | 53.26% | -30.89% | -32.71% | 49 | 46.94% | 508 | 459 |
| final_validation_2025_09_12 | -37.26% | -42.29% | -42.73% | 29 | 20.69% | 242 | 213 |
| oos_2026 | -14.17% | -42.98% | -46.14% | 39 | 33.33% | 468 | 429 |

Promotion verdict: `promotion_pass=false`.
