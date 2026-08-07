# Alpha3 RL Exit Owner Fulltrain Contract - 2026-05-14/15 KST

## Scope

This experiment changes the RL layer role from exit-placement selection to an exit owner.

Alpha3 entry ownership remains frozen:

- HGB parent, teacher gate, V21.2 jackpot add-on, frozen V27 scout, and entry execution are unchanged.
- While a position is active, the RL exit owner may emit `hold` or close the full position with one reduce-only exit placement arm.
- Partial close, TWAP/VWAP slicing, and continuous limit offsets are not included in this first fulltrain run.

## Input / Output

Input state is built at every active-position bar from:

- current position side, owner, hold bars, unrealized PnL, MFE, MAE, notional, parent notional,
- TP, SL, max hold, effective TP/SL, entry edge, entry volatility anchor,
- Alpha3 decision frame,
- frozen V27 q outputs,
- current causal feature frame.

Output actions:

1. `hold`
2. `baseline_exit2_pen05`
3. `exit0_pen0`
4. `exit1_pen0`
5. `exit2_pen0`
6. `exit3_pen0`
7. `exit4_pen0`

Any non-hold action closes 100% of the active position through Alpha3's reduce-only post-only limit/fallback simulator.

## Model

Architecture: compact fitted-Q MLP.

- `LayerNorm(input_dim)`
- `Linear(input_dim, 160)` + `SiLU` + `Dropout(0.10)`
- `Linear(160, 160)` + `SiLU` + `Dropout(0.10)`
- `Linear(160, 7)`

Training:

- Train split: `2025-01-01..2025-09-30`
- Selection split: `2025-10-01..2025-12-31`
- 2026 is OOS only.
- Device: CUDA (`NVIDIA GeForce RTX 3070 Ti` in this run)
- Training states: `35,717`
- Episodes: `861`
- Fit/holdout states: `28,573 / 7,144`

## Result

Current replay input warning: the active `v31.DEFAULT_EVAL` file currently covers `2026-01-01..2026-02-28 16:00`, so these numbers are current-replay results, not the older full canonical `+747.76%` report horizon.

| Candidate | Cost1 PnL | Cost1 MDD | Cost2 PnL | Cost3 PnL | Score |
|---|---:|---:|---:|---:|---:|
| Alpha3 baseline exit2 pen05 | +358.84% | -27.59% | +283.14% | +215.42% | 541.22 |
| Fixed front-run exit4 pen0 | +369.62% | -27.14% | +285.00% | +218.20% | 553.83 |
| RL exit owner fulltrain selected | +358.84% | -27.59% | +283.14% | +215.42% | 541.22 |
| Validation best placement-only | +358.84% | -27.59% | +283.14% | +215.42% | 541.22 |

2025Q4 selection did not validate an early-exit owner policy. The best early policy had a negative selection score and the selected OOS path reduced to baseline-like behavior. Fixed `exit4_pen0` remains stronger on current OOS replay.

## Decision

Do not promote this first exit-owner RL model.

The expanded 2025Q1-Q3 training data fixed the dataset-size issue (`10,044` states to `35,717` states), but it did not produce a profitable timing owner. The reward/action design is still too crude: it only supports full close and OHLCV proxy execution, and does not include partial exits, market-vs-maker action separation, true L2 fill outcomes, or risk-budget-aware terminal rewards.

Next viable iteration:

- add explicit `market_close_100`, `maker_close_100_{0,2,4}`, `close_50`, and `hold_with_guard` actions,
- train on full Q1-Q3 position lifecycle episodes,
- add terminal giveback/MDD penalties instead of only immediate exit rewards,
- include live DuckDB L2 snapshot features as shadow-only inputs,
- keep `fixed_exit4_pen0` as the production comparison gate.

## Artifacts

- Script: `scripts/eval_alpha3_rl_exit_owner_fulltrain_20260514.py`
- Model: `data/ensemble/supervised/alpha3_rl_exit_owner_fulltrain_20260514/offline_rl_exit_q.pt`
- Summary: `data/ensemble/reports/alpha3_rl_exit_owner_fulltrain_20260514_summary.json`
- Grid: `data/ensemble/reports/alpha3_rl_exit_owner_fulltrain_20260514_grid.csv`
- Audit: `data/ensemble/reports/alpha3_rl_exit_owner_fulltrain_20260514_audit.json`
- Dataset: `data/ensemble/reports/alpha3_rl_exit_owner_fulltrain_20260514_dataset.json`
